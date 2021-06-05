import torch
import torch.nn.functional as F
from copy import deepcopy
from agent import Actor, Critic
import numpy as np
from utils import grad_clamp


class Agent:
    def __init__(self, state_dim, action_dim, n_agents, buffer, device="cuda", critic_lr=1e-5, actor_lr=1e-5, gamma=0.99, hidden_size=32, tau=1e-3, temperature=1):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.buffer = buffer
        self.device = device

        self.actors = [Actor(state_dim, 1, hidden_size, temperature=temperature) for _ in range(n_agents)]
        self.actors_target = [deepcopy(a) for a in self.actors]
        self.actors_optims = [torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        for actor, target in zip(self.actors, self.actors_target):
            actor.to(self.device)
            target.to(self.device)

        self.critics = [Critic(state_dim, action_dim, 1, hidden_size) for _ in range(n_agents)]
        with torch.no_grad():
            self.critics_target = [deepcopy(c) for c in self.critics]
        self.critics_optims = [torch.optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]
        for critic, target in zip(self.critics, self.critics_target):
            critic.to(self.device)
            target.to(self.device)

        self.critic = Critic(state_dim, action_dim, 1, hidden_size)
        with torch.no_grad():
            self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def act(self, rel_state):
        actions = []
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                actions.append(actor(rel_state[i]).cpu().numpy())
        return np.array(actions).flatten()

    def soft_update(self, model, target):
        with torch.no_grad():
            for param, param_target in zip(model.parameters(), target.parameters()):
                param_target.data.mul_(1 - self.tau)
                param_target.data.add_(self.tau * param.data)
    
    def update_critic(self, global_state, action, next_global_state, next_rel_state, reward, done):
        q = torch.hstack([self.critic(global_state, action)] * self.n_agents)
        # pred Q value for each action
        with torch.no_grad():
            next_target_actions = torch.empty_like(action, device=self.device)
            for idx, target in enumerate(self.actors_target):
                next_target_actions[idx] = target(next_rel_state[idx]).flatten()
            q_target = reward + self.gamma * (1 - done) * self.critic_target(next_global_state, next_target_actions)
        
        loss = F.mse_loss(q, q_target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_clamp(self.critic)
        self.critic_optimizer.step()
        self.soft_update(self.critic, self.critic_target)        
    
    def update_actors(self, global_state, rel_state, action_others):
        for idx, (actor, target, optim) in enumerate(zip(self.actors, self.actors_target, self.actors_optims)):
            with torch.no_grad():
                actions = deepcopy(action_others)
            actions[idx] = actor(rel_state[idx]).flatten()
            loss = -torch.mean(self.critic(global_state, actions))
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.soft_update(actor, target)
        
    def update(self, batch_size):
        gstate, rel_state, action, next_gstate, next_rel_state, reward, done = self.buffer.sample(batch_size)
        self.update_critic(gstate, action, next_gstate, next_rel_state, reward, done)
        self.update_actors(gstate, rel_state, action)
        
    def save(self, path, step):
        torch.save(self.critic.state_dict(), f"{path}_critic_{step}.pt")
        for idx, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{path}_actor{idx}_{step}.pt")
