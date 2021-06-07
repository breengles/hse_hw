from utils import grad_clamp
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from examples.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent

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
        self.model[-1].weight.data.uniform_(-3e-3, 3e-3)

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
    def __init__(self, team, state_dim, actor_action_dim, critic_action_dim,
                 critic_lr=1e-2, actor_lr=1e-2, gamma=0.99, tau=0.01, 
                 hidden_size=64, device="cpu", temperature=30):
        self.team = team
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.kind = "trainable"

        self.actor = Actor(state_dim, actor_action_dim, hidden_size, temperature).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, critic_action_dim, hidden_size).to(self.device)
        self.critic2 = Critic(state_dim, critic_action_dim, hidden_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        with torch.no_grad():
            self.actor_target = deepcopy(self.actor)
            self.critic_target = deepcopy(self.critic)
            self.critic_target2 = deepcopy(self.critic)

    def act(self, state):
        if self.team == "prey":
            if state[4] > 1:
                return np.array([-2])
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
            self._soft_update(self.critic_target2, self.critic2)

class MADDPG:
    def __init__(self, n_preds, n_preys, state_dim, action_dim, pred_cfg, 
                 prey_cfg, device="cpu", temperature=1, verbose=False, 
                 pred_baseline=False, prey_baseline=False):
        self.device = device
        self.n_preds = n_preds
        self.n_preys = n_preys
        self.pred_baseline = pred_baseline
        self.prey_baseline = prey_baseline

        self.trainable_agents = []
        
        if pred_baseline:
            self.pred_agents = [ChasingPredatorAgent()]
        else:
            self.pred_agents = [Agent("pred", state_dim, action_dim, n_preds + n_preys,
                                      **pred_cfg, device=self.device, 
                                      temperature=temperature) for _ in range(self.n_preds)]
            self.trainable_agents.extend(self.pred_agents)
            
        if prey_baseline:
            self.prey_agents = [FleeingPreyAgent()]
        else:
            self.prey_agents = [Agent("prey", state_dim, action_dim,  n_preds + n_preys,
                                      **prey_cfg, device=self.device, 
                                      temperature=temperature) for _ in range(self.n_preys)]
            self.trainable_agents.extend(self.prey_agents)

        self.agents = self.pred_agents + self.prey_agents
        
        if verbose:
            for idx, agent in enumerate(self.trainable_agents):
                print(f"=== AGENT {idx} ===")
                print("--- ACTOR ---")
                print(agent.actor)
                print("--- CRITIC ---")
                print(agent.critic)
        
    def update(self, batch):
        (_, next_state_dict, 
         gstate, agent_states, actions, next_gstate, next_agent_states, rewards, done) = batch
        target_next_actions = torch.empty_like(actions, device=self.device)
        
        if self.pred_baseline:
            for idx, s in enumerate(next_state_dict):
                target_next_actions[:self.n_preds, idx] = \
                    torch.tensor(self.pred_agents[0].act(s), device=self.device)
        else:
            for idx, agent in enumerate(self.pred_agents):
                target_next_actions[idx] = agent.actor_target(next_agent_states[idx]).squeeze(-1)
        
        if self.prey_baseline:
            for idx, s in enumerate(next_state_dict):
                target_next_actions[-self.n_preys:, idx] = \
                    torch.tensor(self.prey_agents[0].act(next_state_dict, device=self.device))
        else:
            for idx, agent in enumerate(self.prey_agents):
                target_next_actions[-self.n_preys + idx:] = agent.actor_target(next_agent_states[idx]).squeeze(-1)
            
        losses = []
        
        for idx, agent in enumerate(self.trainable_agents):
            q1 = agent.critic(gstate, actions.T).squeeze(-1)
            q2 = agent.critic2(gstate, actions.T).squeeze(-1)
            q_target = rewards[idx] + agent.gamma * (1 - done) \
                        * torch.minimum(agent.critic_target(next_gstate, target_next_actions.T).squeeze(-1),
                                        agent.critic_target2(next_gstate, target_next_actions.T).squeeze(-1))
            
            # q1 = agent.critic(agent_states[idx], actions.T).squeeze(-1)
            # q2 = agent.critic2(agent_states[idx], actions.T).squeeze(-1)
            # q_target = rewards[idx] + agent.gamma * (1 - done) \
            #             * torch.minimum(agent.critic_target(next_agent_states[idx], 
            #                                                 target_next_actions.T).squeeze(-1),
            #                             agent.critic_target2(next_agent_states[idx], 
            #                                                  target_next_actions.T).squeeze(-1))
            
            assert q1.shape == q_target.shape
            assert q2.shape == q_target.shape
            
            critic_loss = F.mse_loss(q1, q_target.detach())
            critic2_loss = F.mse_loss(q2, q_target.detach())
            
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            grad_clamp(agent.critic)
            agent.critic_optimizer.step()
            
            agent.critic_optimizer2.zero_grad()
            critic2_loss.backward()
            grad_clamp(agent.critic2)
            agent.critic_optimizer2.step()
            
            with torch.no_grad():
                current_actions = deepcopy(actions)
            current_actions[idx] = agent.actor(agent_states[idx]).squeeze(-1)
            
            actor_loss = -agent.critic(gstate, current_actions.T).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            grad_clamp(agent.actor)
            agent.actor_optimizer.step()
            
            agent.soft_update_targets()
            
            losses.append((actor_loss.item(), critic_loss.item()))
        return losses
            
    def save(self, path):
        torch.save(self, path)
        