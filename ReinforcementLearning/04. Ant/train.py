import pybullet_envs
from gym import make
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy


ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 1000000


def soft_update(target, source, tau = 0.002):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.model(state)
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim, actor_lr=2e-4, critic_lr=5e-4, a_low=-1, a_high=1):
        self.a_low = a_low
        self.a_high = a_high
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=critic_lr)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        
        self.replay_buffer = deque(maxlen=200000)

    def update(self, transition, sigma=2, c=1, updates_number=1, policy_delay=1, batch_size=128, gamma=0.99, tau=0.002):
        if policy_delay > updates_number:
            policy_delay = updates_number
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > batch_size * 16:
            for j in range(updates_number):
                # Sample batch
                transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer)-1)] for _ in range(batch_size)]
                state, action, next_state, reward, done = zip(*transitions)
                state = torch.tensor(np.array(state), device=self.device, dtype=torch.float)
                action = torch.tensor(np.array(action), device=self.device, dtype=torch.float)
                next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float)
                reward = torch.tensor(np.array(reward), device=self.device, dtype=torch.float)
                done = torch.tensor(np.array(done), device=self.device, dtype=torch.float)
                
                with torch.no_grad():
                    target_action = self.target_actor(next_state)
                    target_action = torch.clip(target_action + torch.clip(sigma * torch.randn_like(target_action), -c, c), self.a_low, self.a_high)
                    
                    Q_target = reward + gamma * (1 - done) \
                             * torch.minimum(self.target_critic_1(next_state, target_action), 
                                             self.target_critic_2(next_state, target_action))
                
                # Update critics
                Q1_loss = F.mse_loss(self.critic_1(state, action), Q_target)
                Q2_loss = F.mse_loss(self.critic_2(state, action), Q_target)
                
                self.critic_1_optim.zero_grad()
                Q1_loss.backward()
                self.critic_1_optim.step()
                
                self.critic_2_optim.zero_grad()
                Q2_loss.backward()
                self.critic_2_optim.zero_grad()
                            
                # Update actor
                if (j + 1) % policy_delay == 0:
                    actor_loss = -self.critic_1(state, self.actor(state)).mean()
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                
                # Update targets
                soft_update(self.target_critic_1, self.critic_1, tau=tau)
                soft_update(self.target_critic_2, self.critic_2, tau=tau)
                soft_update(self.target_actor, self.actor, tau=tau)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=self.device)
            return self.actor(state).cpu().numpy()[0]

    def save(self, name="agent.pkl"):
        torch.save(self.actor, name)


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns

if __name__ == "__main__":
    env = make(ENV_NAME)
    test_env = make(ENV_NAME)
    
    td3 = TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    eps = 0.2
    
    for i in range(TRANSITIONS):
        action = td3.act(state)
        action = np.clip(action + eps * np.random.randn(*action.shape), -1, +1)

        next_state, reward, done, _ = env.step(action)
        td3.update((state, action, next_state, reward, done))
        
        state = env.reset() if done else next_state
        
        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(test_env, td3, 5)
            rmean = np.mean(rewards)
            rstd = np.std(rewards)
            print(f"Step: {i + 1}, Reward mean: {rmean}, Reward std: {rstd}")
            td3.save(f"{i + 1}_{rmean}_{rstd}.pkl")
