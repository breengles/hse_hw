from models.actor import Actor
from models.critic import Critic
from utils import ReplayBuffer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import state2tensor, grad_clamp

from copy import deepcopy
from tqdm import tqdm

SEED = 0





class Agent:
    def __init__(self, state_dim, action_dim, n_agents, buffer, device="cuda", critic_lr=1e-5, actor_lr=1e-5, gamma=0.99, hidden_size=32, tau=1e-3):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.buffer = buffer
        self.device = device

        self.actor = Actor(state_dim, n_agents, hidden_size)
        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, 1, hidden_size)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic.to(self.device)
        self.critic_target.to(self.device)
        self.actor.to(self.device)
        self.actor_target.to(self.device)

    def act(self, state_dict):
        state_tensor = state2tensor(state_dict, self.device)
        with torch.no_grad():
            return self.actor(state_tensor)

    def soft_update(self, model, target):
        with torch.no_grad():
            for param, param_target in zip(model.parameters(), target.parameters()):
                param_target.data.mul_(1 - self.tau)
                param_target.data.add_(self.tau * param.data)
    
    def update_critic(self, batch):
        state, action, next_state, reward, done = batch
        q = torch.hstack([self.critic(state, action)] * self.n_agents)
        # pred Q value for each action
        with torch.no_grad():
            q_target = reward + self.gamma * (1 - done) * self.critic_target(next_state, self.actor_target(next_state))
        
        loss = F.mse_loss(q, q_target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_clamp(self.critic)
        self.critic_optimizer.step()
        self.soft_update(self.critic, self.critic_target)
        
    def update_actor(self, state):
        loss = -torch.mean(self.critic(state, self.actor(state)))
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        grad_clamp(self.actor)
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.actor_target)
        
    
    def update(self, batch_size):
        batch = tuple(map(lambda x: x.to(self.device), self.buffer.sample(batch_size)))
        self.update_critic(batch)
        self.update_actor(batch[0])
        
    def save(self, path, step):
        torch.save(self.critic.state_dict(), f"{path}_critic_{step}.pt")
        torch.save(self.actor.state_dict(), f"{path}_actor_{step}.pt")
        