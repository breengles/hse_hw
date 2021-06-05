from models.actor import Actor
from models.critic import Critic
from models.dqn import DQN
from utils import ReplayBuffer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import grad_clamp

from copy import deepcopy
from tqdm import tqdm


class Agent:
    def __init__(self, state_dim, action_dim, n_agents, buffer, device="cuda", critic_lr=1e-5, actor_lr=1e-5, gamma=0.99, hidden_size=32, tau=1e-3):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.buffer = buffer
        self.device = device

        self.actors = [Actor(state_dim, 1, hidden_size) for _ in range(n_agents)]
        self.actors_target = [deepcopy(a) for a in self.actors]
        self.actors_optims = [torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        for actor, target in zip(self.actors, self.actors_target):
            actor.to(self.device)
            target.to(self.device)

        # self.actor = Actor(state_dim, n_agents, hidden_size)
        # self.actor_target = deepcopy(self.actor)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # self.actor.to(self.device)
        # self.actor_target.to(self.device)

        self.critic = Critic(state_dim, action_dim, 1, hidden_size)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    # def act(self, state_dict):
    #     state_tensor = state2tensor(state_dict, self.device)
    #     with torch.no_grad():
    #         return torch.tanh(self.actor(state_tensor))
        
    def act(self, rel_state):
        actions = []
        with torch.no_grad():
            for actor in self.actors:
                actions.append(torch.tanh(actor(rel_state) / 30).cpu().numpy())
        return np.array(actions)

    def soft_update(self, model, target):
        with torch.no_grad():
            for param, param_target in zip(model.parameters(), target.parameters()):
                param_target.data.mul_(1 - self.tau)
                param_target.data.add_(self.tau * param.data)
    
    def update_critic(self, global_state, action, next_global_state, next_rel_state, reward, done):
    #     state, action, next_state, reward, done = batch
    #     q = torch.hstack([self.critic(state, action)] * self.n_agents)
    #     # pred Q value for each action
    #     with torch.no_grad():
    #         q_target = reward + self.gamma * (1 - done) * self.critic_target(next_state, self.actor_target(next_state))
        
    #     loss = F.mse_loss(q, q_target)
    #     self.critic_optimizer.zero_grad()
    #     loss.backward()
    #     grad_clamp(self.critic)
    #     self.critic_optimizer.step()
    #     self.soft_update(self.critic, self.critic_target)
    
        q = torch.hstack([self.critic(global_state, action)] * self.n_agents)
        # pred Q value for each action
        with torch.no_grad():
            next_target_actions = torch.empty_like(action, device=self.device)
            for idx, target in enumerate(self.actors_target):
                next_target_actions[idx] = target(next_rel_state[idx])
            q_target = reward + self.gamma * (1 - done) * self.critic_target(next_global_state, next_target_actions)
        
        loss = F.mse_loss(q, q_target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_clamp(self.critic)
        self.critic_optimizer.step()
        self.soft_update(self.critic, self.critic_target)        
        
    def update_actors(self, global_state, rel_state, action_others):
        # loss = -torch.mean(self.critic(state, self.actor(state)))
        # self.actor_optimizer.zero_grad()
        # loss.backward()
        # grad_clamp(self.actor)
        # self.actor_optimizer.step()
        # self.soft_update(self.actor, self.actor_target)
        
        actions = action_others.clone()
        for idx, (actor, target, optim) in enumerate(zip(self.actors, self.actors_target, self.actors_optims)):
            tmp = actions[idx].clone()
            actions[idx] = actor(rel_state[idx])
            loss = -torch.mean(self.critic(global_state, actions))
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.soft_update(actor, target)
            actions[idx] = tmp.clone()
        
    def update(self, batch_size):
        gstate, rel_state, action, next_gstate, next_rel_state, reward, done = self.buffer.sample(batch_size)
        self.update_critic(gstate, action, next_gstate, next_rel_state, reward, done)
        self.update_actors(gstate, rel_state, action)
        
    def save(self, path, step):
        torch.save(self.critic.state_dict(), f"{path}_critic_{step}.pt")
        for idx, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{path}_actor{idx}_{step}.pt")
