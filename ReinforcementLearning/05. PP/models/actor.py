from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_agents),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)
