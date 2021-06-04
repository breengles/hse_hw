from models.actor import Actor
from models.critic import Critic
from models.dqn import DQN
from utils import ReplayBuffer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import state2tensor, grad_clamp, action2discrete, discrete2action

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
            # return torch.sigmoid(self.actor(state_tensor) / 5)
            # return 2 * torch.sigmoid(self.actor(state_tensor) / 5) - 1
            return torch.tanh(self.actor(state_tensor) / 1e6)
            # return self.actor(state_tensor) / 1e6

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
        batch = self.buffer.sample(batch_size)
        # batch = tuple(map(lambda x: x.to(self.device), self.buffer.sample(batch_size)))
        self.update_critic(batch)
        self.update_actor(batch[0])
        
    def save(self, path, step):
        torch.save(self.critic.state_dict(), f"{path}_critic_{step}.pt")
        torch.save(self.actor.state_dict(), f"{path}_actor_{step}.pt")


# какое-то говно
# class Discrete:
#     def __init__(self, state_dim, action_dim, n_agents, buffer, 
#                  device="cuda", gamma=0.99, hidden_size=64, tau=1e-3,
#                  num_bins=4, lr=1e-3):
#         self.n_agents = n_agents
#         self.gamma = gamma
#         self.tau = tau
#         self.buffer = buffer
#         self.device = device
#         self.num_bins = num_bins

#         self.dqn = DQN(state_dim, action_dim, self.num_bins, hidden_size=64)
#         self.dqn_target = deepcopy(self.dqn)

#         self.optim = torch.optim.Adam(self.dqn.parameters(), lr=lr)

#         self.dqn.to(self.device)
#         self.dqn_target.to(self.device)
    
#     def act(self, state_dict, action_others):
#         state_tensor = state2tensor(state_dict, self.device)
#         discrete_action = torch.argmax(self.dqn(state_tensor, action_others))
            
            
        
#     def update(self, batch_size):
#         state, action, next_state, reward, done = self.buffer.sample(batch_size)
#         discrete_action = action2discrete(np.array(action))

#         with torch.no_grad():
#             next_action = ???
#             q_target = reward + self.gamma * (1 - done) * self.dqn_target(next_state, next_action).max(dim=1)[0].view(-1)  # pred Q value for each action

#         qle = self.dqn(state, action).gather(1, action)  # take Q value for action

#         loss = F.mse_loss(qle, q_target.unsqueeze(dim=1))
#         self.optimizer.zero_grad()
#         loss.backward()

#         # gradient clamping
#         for param in self.dqn.parameters():
#             param.grad.data.clamp_(-1, 1)

#         self.optimizer.step()
