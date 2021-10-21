import torch
from torch import Tensor
from typing import Tuple
import numpy as np


class Buffer(object):
    def __init__(self, state_dim, max_size, device="cpu"):
        self.max_size = max_size
        self.device = device

        self.state = torch.zeros(max_size, state_dim, device=device, dtype=torch.float)
        self.reward = torch.zeros(max_size, device=device, dtype=torch.float)
        self.action = torch.zeros(max_size, device=device, dtype=torch.int64)
        self.next_state = torch.zeros(max_size, state_dim, device=device, dtype=torch.float)
        self.done = torch.zeros(max_size, device=device, dtype=torch.bool)

        self.filled_i = 0
        self.curr_size = 0

    def push(self, state, action, next_state, reward, done):
        if self.curr_size < self.max_size:
            self.curr_size += 1

        self.state[self.filled_i] = torch.tensor(state, device=self.device)
        self.action[self.filled_i] = torch.tensor(action, device=self.device, dtype=torch.int64)
        self.reward[self.filled_i] = torch.tensor(reward, device=self.device)
        self.next_state[self.filled_i] = torch.tensor(next_state, device=self.device)
        self.done[self.filled_i] = torch.tensor(done, device=self.device, dtype=torch.bool)

        self.curr_size = min(self.max_size, self.curr_size + 1)
        self.filled_i = (self.filled_i + 1) % self.max_size

    def sample(self, batch_size: int, norm_rew: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indices = np.random.choice(self.curr_size, batch_size, replace=False)

        if norm_rew:
            mean = torch.mean(self.reward[: self.curr_size])
            std = torch.std(self.reward[: self.curr_size])
            rew = (self.reward[indices] - mean) / std
        else:
            rew = self.reward[indices]

        return (
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            rew,
            self.done[indices],
        )

    def __len__(self):
        return self.curr_size
