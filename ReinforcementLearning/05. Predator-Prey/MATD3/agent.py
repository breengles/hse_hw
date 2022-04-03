from utils import grad_clamp
import torch
from tqdm import tqdm
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
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, action_dim),
        )
        self.model[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, return_raw=False):
        out = self.model(state) / self.temperature
        if return_raw:
            return out, torch.tanh(out)
        else:
            return torch.tanh(out)
    
    
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
                 hidden_size=64, device="cpu", temperature=30, actor=None,
                 actor_target=None, critics=None, critics_target=None):
        self.team = team
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.kind = "trainable"

        if actor is None:
            self.actor = Actor(state_dim, actor_action_dim, hidden_size, temperature).to(self.device)
        else:
            self.actor = actor.to(self.device)
        
        if critics is None:
            self.critic = Critic(state_dim, critic_action_dim, hidden_size).to(self.device)
            self.critic2 = Critic(state_dim, critic_action_dim, hidden_size).to(self.device)
        else:
            self.critic = critics[0].to(self.device)
            self.critic2 = critics[1].to(self.device)
            
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        with torch.no_grad():
            if actor_target is None:
                self.actor_target = deepcopy(self.actor)
            else:
                self.actor_target = actor_target.to(self.device)
            
            if critics_target is None:
                self.critic_target = deepcopy(self.critic)
                self.critic2_target = deepcopy(self.critic)
            else:
                self.critic_target = critics_target[0].to(self.device)
                self.critic2_target = critics_target[1].to(self.device)

    def act(self, state, sigma=-1):
        with torch.no_grad():
            if self.team == "prey":
                if state[4] > 1:
                    return np.array([-2])
            
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            
            action = self.actor(state).cpu().numpy()
            if sigma > 0:
                action = np.clip(action + np.random.normal(scale=sigma, size=action.shape), -1, 1)
            return action

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)
    
    def soft_update_targets(self):
        with torch.no_grad():
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
            self._soft_update(self.critic2_target, self.critic2)

class MADDPG:
    def __init__(self, n_preds, n_preys, state_dim, action_dim, pred_cfg, 
                 prey_cfg, saverate=1000, device="cpu", temperature=1, verbose=False, 
                 pred_baseline=False, prey_baseline=False, actor_update_delay=1,
                 shared_actor=False, shared_critic=False, actor_action_reg_coef=1e-5):
        self.actor_action_reg_coef = actor_action_reg_coef
        self.n_preds = n_preds
        self.n_preys = n_preys
        self.saverate = saverate
        self.device = device
        self.verbose = verbose
        self.pred_baseline = pred_baseline
        self.prey_baseline = prey_baseline
        self.actor_update_delay = actor_update_delay
        
        if shared_actor:
            actor_pred = Actor(state_dim, action_dim, pred_cfg["hidden_size"], 
                               temperature).to(self.device)
            actor_prey = Actor(state_dim, action_dim, pred_cfg["hidden_size"], 
                               temperature).to(self.device)
            with torch.no_grad():
                actor_pred_target = deepcopy(actor_pred)
                actor_prey_target = deepcopy(actor_prey)
        else:
            actor_pred = None
            actor_prey = None
            actor_pred_target = None
            actor_prey_target = None
            
        if shared_critic:
            critics_pred = [Critic(state_dim, n_preds + n_preys, 
                                   pred_cfg["hidden_size"]).to(self.device),
                            Critic(state_dim, n_preds + n_preys, 
                                   pred_cfg["hidden_size"]).to(self.device)]
            critics_prey = [Critic(state_dim, n_preds + n_preys, 
                                   pred_cfg["hidden_size"]).to(self.device),
                            Critic(state_dim, n_preds + n_preys, 
                                   pred_cfg["hidden_size"]).to(self.device)]
            with torch.no_grad():
                critics_target_pred = deepcopy(critics_pred)
                critics_target_prey = deepcopy(critics_prey)
                
        else:
            critics_pred = None
            critics_prey = None
            critics_target_pred = None
            critics_target_prey = None
            
        self.trainable_agents = []
        if pred_baseline:
            self.pred_agents = [ChasingPredatorAgent()]
        else:
            self.pred_agents = [Agent("pred", state_dim, action_dim, n_preds + n_preys,
                                      **pred_cfg, device=self.device, 
                                      critics=critics_pred, 
                                      critics_target=critics_target_pred) for _ in range(self.n_preds)]
            self.trainable_agents.extend(self.pred_agents)
            
        if prey_baseline:
            self.prey_agents = [FleeingPreyAgent()]
        else:
            self.prey_agents = [Agent("prey", state_dim, action_dim, n_preds + n_preys,
                                      **prey_cfg, device=self.device, 
                                      temperature=temperature, 
                                      actor=actor_prey,
                                      actor_target=actor_prey_target,
                                      critics=critics_prey,
                                      critics_target=critics_target_prey) for _ in range(self.n_preys)]
            self.trainable_agents.extend(self.prey_agents)
        
        self.agents = self.pred_agents + self.prey_agents
        
        if self.verbose:
            print(f"Pred baseline: {self.pred_baseline}")
            print(f"Prey baseline: {self.prey_baseline}")
            print("=== ALL AGENTS ===")
            for idx, agent in enumerate(self.agents):
                print(f"AGENT {idx} ({agent.team}):", agent)
            print("=== TRAINABLE AGENTS ===")
            for idx, agent in enumerate(self.trainable_agents):
                print(f"AGENT {idx} ({agent.team}):", agent)
                print(id(agent.actor))
                print(agent.actor)
                print(id(agent.critic))
                print(agent.critic)
        
    def update(self, buffer, batch_size, step, beta):
        batch, (weights, idxes) = buffer.sample(batch_size, beta=beta)
        (_, next_state_dict, gstate, agent_states, actions, next_gstate, 
         next_agent_states, rewards, done) = batch
        
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        
        target_next_actions = torch.empty_like(actions, device=self.device)
        
        # if self.pred_baseline:
        #     for idx, s in enumerate(next_state_dict):
        #         target_next_actions[idx, :self.n_preds] = \
        #             torch.tensor(self.pred_agents[0].act(s), device=self.device)
        # else:
        #     for idx, agent in enumerate(self.pred_agents):
        #         target_next_actions[:, idx] = agent.actor_target(next_agent_states[:, idx]).squeeze(-1)
    
        # if self.prey_baseline:
        #     for idx, s in enumerate(next_state_dict):
        #         target_next_actions[idx, -self.n_preys:] = \
        #             torch.tensor(self.prey_agents[0].act(s), device=self.device)
        # else:
        #     for idx, agent in enumerate(self.prey_agents):
        #         target_next_actions[:, -self.n_preys + idx] = \
        #             agent.actor_target(next_agent_states[:, -self.n_preys + idx]).squeeze(-1)
                    
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
                    torch.tensor(self.prey_agents[0].act(s), device=self.device)
        else:
            for idx, agent in enumerate(self.prey_agents):
                target_next_actions[-self.n_preys + idx] = \
                    agent.actor_target(next_agent_states[-self.n_preys + idx]).squeeze(-1)
                    
        if (step + 1) % self.saverate == 0:
            tqdm.write((f"=== Step {step + 1} ==="))
        
        for idx, agent in enumerate(self.trainable_agents):
            q1 = agent.critic(gstate, actions.T).squeeze(-1)
            q2 = agent.critic2(gstate, actions.T).squeeze(-1)
            with torch.no_grad():
                crit1 = agent.critic_target(next_gstate, target_next_actions.T).squeeze(-1)
                crit2 = agent.critic2_target(next_gstate, target_next_actions.T).squeeze(-1)
                q_target = rewards[idx] + agent.gamma * (1 - done) * torch.minimum(crit1, crit2)
            
            assert q1.shape == q_target.shape
            assert q2.shape == q_target.shape
            
            # critic_loss = F.mse_loss(q1, q_target)
            # critic2_loss = F.mse_loss(q2, q_target)
            
            if weights is not None:
                critic_loss = (q1 - q_target) ** 2 * weights
                critic2_loss = (q2 - q_target) ** 2 * weights
                prios = critic_loss + 1e-5
            else:
                critic_loss = F.mse_loss(q1, q_target)
                critic2_loss = F.mse_loss(q2, q_target)
                prios = critic_loss  # it does not matter: using non-prioritized buffer
            
            critic_loss = critic_loss.mean()
            critic2_loss = critic2_loss.mean()
            
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            # grad_clamp(agent.critic)
            buffer.update_priorities(idxes, prios.data.cpu().numpy())
            agent.critic_optimizer.step()
            
            agent.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            # grad_clamp(agent.critic2)
            agent.critic2_optimizer.step()
            
            if (step + 1) % self.actor_update_delay == 0:
                """
                get new actions from all other agents?
                see https://github.com/Gouet/maddpg-pytorch-1/blob/40388d7c18e4662cf23c826d97e209df9003d86c/algorithms/maddpg.py#L149
                """
                with torch.no_grad():
                    current_actions = deepcopy(actions)
                action = agent.actor(agent_states[:, idx], return_raw=True)
                current_actions[:, idx] = action[1].squeeze(-1)
                
                actor_loss = -agent.critic(gstate, current_actions.T).mean()
                actor_loss += self.actor_action_reg_coef * (action[0] ** 2).mean()
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                # grad_clamp(agent.actor)
                agent.actor_optimizer.step()
                
                agent.soft_update_targets()
                
                if (step + 1) % self.saverate == 0:
                    tqdm.write(f"Agent{idx} ({agent.team}): Actor loss: {actor_loss:0.5f}, Critic loss: {critic_loss:0.5f}")
            
    def save(self, path):
        torch.save(self, path)
        