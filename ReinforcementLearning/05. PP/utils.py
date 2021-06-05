import torch, random, os, json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import deque
from random import sample


class ReplayBuffer:
    def __init__(self, state_dim, n_agents, buffer_size: int = 10_000, device=torch.device("cpu")):
        self.pos = 0
        self.device = device
        self.size = buffer_size
        self.gstates = torch.empty((buffer_size, state_dim), dtype=torch.float, device=device)
        self.rel_states = torch.empty((buffer_size, n_agents, state_dim), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, n_agents), dtype=torch.float, device=device)
        self.next_gstates = torch.empty((buffer_size, state_dim), dtype=torch.float, device=device)
        self.next_rel_states = torch.empty((buffer_size, n_agents, state_dim), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, n_agents), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)

    def __len__(self):
        return self.gstates.shape[0]

    # transition is (state, action, next_state, reward, done)
    def add(self, transition):
        (self.gstates[self.pos], 
         self.rel_states[self.pos],
         self.actions[self.pos], 
         self.next_gstates[self.pos], 
         self.next_rel_states[self.pos], 
         self.rewards[self.pos], 
         self.dones[self.pos]) = self._encode_transition(transition, self.device)
        self.pos = (self.pos + 1) % self.size
        
    def sample(self, batch_size: int):
        assert self.__len__() >= batch_size
        ids = np.random.choice(self.__len__(), batch_size, replace=False)
        return (self.gstates[ids], 
                self.rel_states[ids],
                self.actions[ids], 
                self.next_gstates[ids], 
                self.next_rel_states[ids], 
                self.rewards[ids], 
                self.dones[ids])

    @staticmethod
    def _encode_transition(transition, device="cpu"):
        gstate, rel_state, action, next_gstate, next_rel_state, reward, done = transition
        gstate = torch.tensor(gstate, device=device)
        rel_state = torch.tensor(rel_state, device=device)
        action = torch.tensor(action, device=device)
        next_gstate = torch.tensor(next_gstate, device=device)
        next_rel_state = torch.tensor(next_rel_state, device=device)
        reward = torch.tensor(reward, device=device)
        done = torch.tensor(done, device=device)
        return gstate, rel_state, action, next_gstate, next_rel_state, reward, done


def grad_clamp(model):
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)


def set_seed(env, seed):
    random.seed(seed)
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
