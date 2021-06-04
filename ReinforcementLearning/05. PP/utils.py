import torch, random, os, json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import deque
from random import sample


# class ReplayBuffer:
#     def __init__(self, size: int = 10000):
#         self.buffer = deque(maxlen=size)

#     def add(self, transition):
#         self.buffer.append(transition)

#     def sample(self, size):
#         assert len(self.buffer) >= size
#         tmp = sample(self.buffer, size)
#         return list(zip(*tmp))

#     def __len__(self):
#         return len(self.buffer)
    
    
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size: int = 10_000, device=torch.device("cpu")):
        self.pos = 0
        self.device = device
        self.size = buffer_size
        self.states = torch.empty((buffer_size, state_dim), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, action_dim), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, state_dim), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)

    def __len__(self):
        return self.states.shape[0]

    # transition is (state, action, next_state, reward, done)
    def add(self, transition):
        (self.states[self.pos], 
         self.actions[self.pos], 
         self.next_states[self.pos], 
         self.rewards[self.pos], 
         self.dones[self.pos]) = self._encode_transition(transition, self.device)
        self.pos = (self.pos + 1) % self.size
        
    def sample(self, batch_size: int):
        assert self.__len__() >= batch_size
        ids = np.random.randint(0, self.__len__(), batch_size)
        return (self.states[ids], 
                self.actions[ids], 
                self.next_states[ids], 
                self.rewards[ids], 
                self.dones[ids])

    def _encode_transition(self, transition, device="cpu"):
        curr_state, action, next_state, reward, done = transition
        curr_state = state2tensor(curr_state, device)
        next_state = state2tensor(next_state, device)
        action = torch.tensor(action, device=device, dtype=torch.float)
        reward = torch.tensor(reward, device=device, dtype=torch.float)
        done = torch.tensor(done, device=device, dtype=torch.int)
        return curr_state, action, next_state, reward, done


def grad_clamp(model):
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)


def state2tensor(state, device=torch.device("cpu")):
    res = []
    for _, team_arr in state.items():
        for agent_data in team_arr:
            res.extend(agent_data.values())

    return torch.tensor(res, device=device, dtype=torch.float32)


def calc_dist(agent1, agent2):
    p1 = np.array([agent1["x_pos"], agent1["y_pos"]])
    p2 = np.array([agent2["x_pos"], agent2["y_pos"]])

    return np.linalg.norm(p1 - p2)


def is_collision(agent1, agent2):
    dist = calc_dist(agent1, agent2)
    dist_min = agent1["radius"] + agent2["radius"]
    return dist < dist_min


def set_seed(seed):
    random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
