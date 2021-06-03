import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_agents)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))
