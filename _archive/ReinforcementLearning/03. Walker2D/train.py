import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random
from tqdm import tqdm
import pandas as pd


ENV_NAME = "Walker2DBulletEnv-v0"

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4
ITERATIONS = 10000


def compute_lambda_returns_and_gae(trajectory, lam=0.97, gamma=0.99):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + gamma * (last_v * (1 - lam) + last_lr * lam)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim)
        )
        self.sigma = nn.Parameter(torch.zeros(action_dim))
        
    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions (use it to compute entropy loss)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        distr = Normal(mu, sigma)
        return torch.exp(distr.log_prob(action).sum(-1)), distr
        
    def act(self, state):
        # Returns an action, not-transformed action and distribution
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        distr = Normal(mu, sigma)
        pure_action = distr.sample()
        action = torch.tanh(pure_action)
        return action, pure_action, distr
        
        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        
    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim, actor_lr=3e-4, critic_lr=2e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), critic_lr)

    def update(self, trajectories, clip=0.2, entropy_coef=1e-2, batches_per_update=64, batch_size=64):
        transitions = [t for traj in trajectories for t in traj]  # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        
        for _ in range(batches_per_update):
            idx = np.random.randint(0, len(transitions), batch_size)  # Choose random batch
            s = torch.tensor(state[idx]).float().to(self.device)
            a = torch.tensor(action[idx]).float().to(self.device)
            op = torch.tensor(old_prob[idx]).float().to(self.device)  # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float().to(self.device)  # Estimated by lambda-returns 
            adv = torch.tensor(advantage[idx]).float().to(self.device)  # Estimated by generalized advantage estimation 
            
            # TODO: Update actor here
            prob_new, distr = self.actor.compute_proba(s, a)
            entropy = distr.entropy().mean()
            ratio = prob_new / op
            actor_loss = -torch.mean(torch.minimum(ratio * adv, torch.clip(ratio, 1 - clip, 1 + clip) * adv))
            actor_loss -= entropy_coef * entropy
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            # TODO: Update critic here
            val = self.critic.get_value(s).flatten()
            critic_loss = F.mse_loss(val, v)
            # critic_loss = F.smooth_l1_loss(val, v)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(self.device)
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(self.device)
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self, name="agent.pkl"):
        torch.save(self.actor, name)


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent, lam=0.97, gamma=0.99):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory, lam=lam, gamma=gamma)

if __name__ == "__main__":
    env = make(ENV_NAME)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    
    t = tqdm(range(ITERATIONS))
    
    log = {
        "step": [],
        "rmean": [],
        "rstd": []
    }
    
    for i in t:
        trajectories = []
        steps_ctn = 0
        
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)
        
        if (i + 1) % (ITERATIONS // 100) == 0:
            rewards = evaluate_policy(env, ppo, 50)
            rmean = np.mean(rewards)
            rstd = np.std(rewards)
            t.set_description(f"Rmean: {rmean:0.4f}, Rstd: {rstd:0.4f}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            ppo.save(name=f"{i + 1}_{int(rmean)}_{int(rstd)}.pkl")
            
            log["step"].append(i + 1)
            log["rmean"].append(rmean)
            log["rstd"].append(rstd)
            
    pd.DataFrame(log).to_csv("log.csv")
            
