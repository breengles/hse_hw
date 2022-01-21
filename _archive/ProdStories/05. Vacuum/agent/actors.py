import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_channels, action_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(obs_channels, 8, (3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.a = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )

    def forward(self, state):
        return self.a(self.features(state))


class DuelingActor(nn.Module):
    def __init__(self, obs_channels, action_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(obs_channels, 8, (3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.v = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.a = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )

    def forward(self, state):
        features = self.features(state)
        vals = self.v(features)
        advs = self.a(features)
        return vals + advs - advs.mean()
