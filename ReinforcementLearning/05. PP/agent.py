import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, temperature=1):
        super().__init__()
        self.temperature = temperature
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        # self.model[-1].weight.data.uniform_(-1e-4, 1e-4)

    def forward(self, state):
        return torch.tanh(self.model(state) / self.temperature)
    
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1))


class Agent:
    def __init__(self, state_dim, actor_action_dim, critic_action_dim,
                 critic_lr=1e-2, actor_lr=1e-2, gamma=0.99, tau=0.01, 
                 hidden_size=64, device="cpu", temperature=30):
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor(state_dim, actor_action_dim, hidden_size, temperature).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, critic_action_dim, hidden_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        with torch.no_grad():
            self.actor_target = deepcopy(self.actor)
            self.critic_target = deepcopy(self.critic)

    # def act(self, state, sigma=-1):
    #     with torch.no_grad():
    #         state = torch.tensor(state, dtype=torch.float, device=self.device)
    #         action = self.actor(state).cpu().numpy()
    #         if sigma > 0:
    #             action = np.clip(action + sigma * np.random.randn(*action.shape), -1, 1)
    #         return action
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            action = self.actor(state).cpu().numpy()
            return action

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)
    
    def soft_update_targets(self):
        with torch.no_grad():
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

class MADDPG:
    def __init__(self, n_preds, n_preys, state_dim, action_dim, pred_cfg, 
                 prey_cfg, device="cpu", temperature=1, verbose=False):
        self.device = device
        self.n_preds = n_preds
        self.n_preys = n_preys
        #? baseline agents here
        self.agents = ([Agent(state_dim, action_dim, n_preds + n_preys,
                              **pred_cfg, 
                              device=self.device, 
                              temperature=temperature) for _ in range(self.n_preds)] + 
                       [Agent(state_dim, action_dim,  n_preds + n_preys,
                              **prey_cfg, 
                              device=self.device, 
                              temperature=temperature) for _ in range(self.n_preys)])
        
        if verbose:
            for idx, agent in enumerate(self.agents):
                print(f"=== AGENT {idx} ===")
                print("--- ACTOR ---")
                print(agent.actor)
                print("--- CRITIC ---")
                print(agent.critic)
        
    def update(self, batch):
        gstate, agent_states, actions, next_gstate, next_agent_states, rewards, done = batch
        
        # actions = actions.squeeze(-1)
        
        target_next_actions = torch.empty_like(actions, device=self.device)
        for idx, agent in enumerate(self.agents):
            target_next_actions[idx] = agent.actor_target(next_agent_states[idx]).squeeze(-1)
            
        losses = []
        for idx, agent in enumerate(self.agents):
            q = agent.critic(gstate, actions.T).squeeze(-1)
            q_target = (rewards[idx] 
                        + agent.gamma * (1 - done) 
                        * agent.critic_target(next_gstate, target_next_actions.T).squeeze(-1))
            
            assert q.shape == q_target.shape
            
            critic_loss = F.mse_loss(q, q_target.detach())
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            with torch.no_grad():
                current_actions = deepcopy(actions)
            current_actions[idx] = agent.actor(agent_states[idx]).squeeze(-1)
            
            actor_loss = -agent.critic(gstate, current_actions.T).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            agent.soft_update_targets()
            
            losses.append((actor_loss.item(), critic_loss.item()))
        return losses
            
    def save(self, path):
        torch.save(self, path)
        