import torch, random, os, json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, n_agents, state_dim, action_dim, size: int = 10_000, device=torch.device("cpu")):
        self.n_agents = n_agents
        self.device = device
        self.size = size
        
        self.state_dicts = [None for _ in range(self.size)]
        self.next_state_dicts = [None for _ in range(self.size)]
        self.gstates = torch.empty((self.size, state_dim), dtype=torch.float, device=device)
        self.agent_states = torch.empty((n_agents, self.size, state_dim), dtype=torch.float, device=device)
        self.actions = torch.empty((n_agents, self.size), dtype=torch.float, device=device)
        self.next_gstates = torch.empty((self.size, state_dim), dtype=torch.float, device=device)
        self.next_agent_states = torch.empty((n_agents, self.size, state_dim), dtype=torch.float, device=device)
        self.rewards = torch.empty((n_agents, self.size), dtype=torch.float, device=device)
        self.dones = torch.empty((self.size), dtype=torch.float, device=device)
        
        self.pos = 0
        self.cur_size = 0

    def __len__(self):
        return self.cur_size

    # transition is (state, action, next_state, reward, done) for agent
    def add(self, transition):
        if self.cur_size < self.size:
            self.cur_size += 1
            
        (state_dict, next_state_dict, 
         gstate, agent_states, actions, next_gstates, next_agent_states, rewards, dones) = transition
        
        self.state_dicts[self.pos] = deepcopy(state_dict)
        self.next_state_dicts[self.pos] = deepcopy(next_state_dict)
        self.gstates[self.pos] = torch.tensor(gstate, device=self.device)
        self.next_gstates[self.pos] = torch.tensor(next_gstates, device=self.device)
        self.dones[self.pos] = torch.tensor(dones, device=self.device)
        
        for idx in range(self.n_agents):
            self.agent_states[idx, self.pos] = torch.tensor(agent_states[idx], device=self.device)
            self.next_agent_states[idx, self.pos] = torch.tensor(next_agent_states[idx], device=self.device)
            self.rewards[idx, self.pos] = torch.tensor(rewards[idx], device=self.device)
            self.actions[idx, self.pos] = torch.tensor(actions[idx], device=self.device)
        self.pos = (self.pos + 1) % self.size
        
    def sample(self, batch_size: int):
        assert self.__len__() >= batch_size
        ids = np.random.choice(self.__len__(), batch_size, replace=False)
        state_dicts = []
        next_state_dicts = []
        for idx in ids:
            state_dicts.append(self.state_dicts[idx])
            next_state_dicts.append(self.next_state_dicts[idx])
        return (state_dicts, next_state_dicts,
                self.gstates[ids], 
                self.agent_states[:, ids],
                self.actions[:, ids], 
                self.next_gstates[ids], 
                self.next_agent_states[:, ids], 
                self.rewards[:, ids], 
                self.dones[ids])


def grad_clamp(model):
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)


def set_seed(env, seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    # torch.use_deterministic_algorithms(True)
    
    
class Logger:
    def __init__(self, params):
        self.params = params
        self.history = {}
    
    def log(self, key, value):
        try:
            self.history[key].append(value)
        except KeyError:
            self.history[key] = [value]
    
    def save_params(self, file_path: str, mode: str = "a+"):
        with open(file_path, mode) as f:
            json.dump(self.params, f, indent=4)
    
    def save(self, file_path: str, mode: str = "w+"):
        pd.DataFrame(self.history).to_csv(file_path, mode=mode, index=False)
            
    def plot(self, x: str, y: str, 
             std: str = None, 
             size=(12, 8), 
             title: str = None, 
             label: str = None, 
             x_label: str = None, 
             y_label: str = None,
             alpha: float = 0.5):
        _, ax = plt.subplots(figsize=size)
        
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
        x_ = np.array(self.history[x])
        y_ = np.array(self.history[y])
        std_ = np.array(self.history[std]) if std is not None else 0
    
        plt.plot(x_, y_, label=label)
        plt.fill_between(x_, y_ - std_, y_ + std_, alpha=alpha)
        
        plt.legend()
        plt.show()


def rollout(env, agents):
    total_reward = []
    state_dict, _, states = env.reset()
    done = False
    while not done:
        actions = []
        states_ = []
        for agent in agents:
            if agent.__class__.__name__ == "ChasingPredatorAgent":
                states_.append(state_dict)
            else:
                states_.extend(states[env.n_preds:])
            if agent.__class__.__name__ == "FleeingPreyAgent":
                states_.append(state_dict)
            else:
                states_.extend(states[-env.n_preys:])
        actions = np.hstack([agent.act(state) for agent, state in zip(agents, states_)])
        state_dict, _, states, rewards, done = env.step(actions)
        total_reward.append(rewards)
    
    return np.vstack(total_reward).sum(axis=0)
