import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_bins, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_bins)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))
