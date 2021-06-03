import torch
import numpy as np
from collections import deque
from random import sample


class ReplayBuffer:
    def __init__(self, size: int = 10000):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, size):
        assert len(self.buffer) >= size
        tmp = sample(self.buffer, size)
        return list(zip(*tmp))

    def __len__(self):
        return len(self.buffer)


def grad_clamp(model):
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)


def state2tensor(state, device="cuda"):
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