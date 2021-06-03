import torch, random, os
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
    def __init__(self, buffer_size: int = 10_000, device=torch.device("cpu")):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    # transition is (state, action, next_state, reward, done)
    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        ids = np.random.randint(0, self.__len__(), batch_size)

        curr_state, action, reward, next_state, done = [], [], [], [], []
        for idx in ids:
            cs, a, ns, r, d = self._encode_transition(self.buffer[idx])
            curr_state.append(cs)
            action.append(a)
            reward.append(r)
            next_state.append(ns)
            done.append(d)

        return list(map(lambda x: torch.vstack(x), (curr_state, action, reward, next_state, done)))

    def _encode_transition(self, transition):
        curr_state, action, next_state, reward, done = transition
        
        
        curr_state = state2tensor(curr_state, self.device)
        next_state = state2tensor(next_state, self.device)
        action = torch.tensor(action, device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        done = torch.tensor(done, device=self.device, dtype=torch.int32)

        return curr_state, action, reward, next_state, done


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
    torch.use_deterministic_algorithms(True)