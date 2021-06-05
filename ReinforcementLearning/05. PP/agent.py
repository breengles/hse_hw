from utils import ReplayBuffer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import grad_clamp
from copy import deepcopy
from tqdm import tqdm


class Actor(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_size=64, temperature=1):
        super().__init__()
        self.temperature = temperature
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_agents),
        )
        self.model[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        return torch.tanh(self.model(state) / self.temperature)
    
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_agents)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1))


class DDPGAgent:
    def __init__(self, team, state_dim, actor_action_dim, critic_action_dim,
                 critic_lr=1e-2, actor_lr=1e-2, gamma=0.99, tau=0.01, 
                 hidden_size=64, device="cpu", **kwargs):
        self.team = team
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor(state_dim, actor_action_dim, hidden_size)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor.to(device)
        self.actor_target.to(device)

        self.critic = Critic(state_dim, critic_action_dim, 1, hidden_size)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def act(self, agent_states):
        agent_states = torch.tensor(agent_states, dtype=torch.float, device=self.device)
        with torch.no_grad():
            return self.actor(agent_states).cpu().numpy()

    def soft_update(self):
        with torch.no_grad():
            for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                param_target.data.mul_(1 - self.tau)
                param_target.data.add_(self.tau * param.data)
                
            for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                param_target.data.mul_(1 - self.tau)
                param_target.data.add_(self.tau * param.data)
        
    def save(self, path, step):
        torch.save(self.critic.state_dict(), f"{path}_critic_{step}.pt")
        for idx, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{path}_actor{idx}_{step}.pt")


class MADDPG:
    def __init__(self, pred_cfg, prey_cfg, buffer, device="cpu"):
        self.device = device
        self.buffer = buffer
        self.n_preds = pred_cfg["n_agents"]
        self.n_preys = prey_cfg["n_agents"]
        self.agents = ([DDPGAgent(**pred_cfg, device=self.device) for _ in range(self.n_preds)] + 
                       [DDPGAgent(**prey_cfg, device=self.device) for _ in range(self.n_preys)])
        
    def update(self, batch_size):
        gstate, agent_states, actions, next_gstate, next_agent_states, rewards, done = self.buffer.sample(batch_size)
        
        actions = actions.squeeze(-1)
        
        next_actions = torch.empty_like(actions, device=self.device)
        for idx, agent in enumerate(self.agents):
            next_actions[idx] = agent.actor_target(next_agent_states[idx]).to(self.device).squeeze(-1)
            
        losses = []
        
        for idx, agent in enumerate(self.agents):
            q = agent.critic(gstate, actions.T)
            with torch.no_grad():
                q_target = rewards[idx] + agent.gamma * (1 - done) * agent.critic_target(next_gstate, next_actions.T)
            assert q.shape == q_target.shape
            
            critic_loss = F.mse_loss(q, q_target)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            with torch.no_grad():
                current_actions = deepcopy(actions)
            current_actions[idx] = agent.actor(agent_states[idx]).squeeze(-1)
            
            actor_loss = -torch.mean(agent.critic(gstate, current_actions.T))
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            agent.soft_update()
            
            losses.append((actor_loss.item(), critic_loss.item()))
        return losses
            
    def save(self, path):
        torch.save(self, path)
        