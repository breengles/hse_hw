import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, outdim=7):
        self.n_preds = 2
        self.n_preys = 5
        self.pred_state_dim = 4
        self.prey_state_dim = 5
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(self.n_preds * self.pred_state_dim + self.n_preys * self.prey_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, outdim),
        )
    
    def forward(self, state):
        return torch.tanh(self.model(state))
