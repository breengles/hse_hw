import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam


def soft_update(source, target, tau=0.002):
    with torch.no_grad():
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * sp.data)


class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim=64, actor_lr=2e-4, device="cpu"):
        self.device = device

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.target = copy.deepcopy(self.actor)

        self.optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        self.actor.to(device)
        self.target.to(device)

    def update(self, batch, gamma=0.99, tau=0.002):
        states, actions, next_states, rewards, dones = batch

        with torch.no_grad():
            q_target = self.target(next_states).max(dim=1)[0].view(-1)
            q_target[dones] = 0

        q_target = rewards + gamma * q_target

        qle = self.actor(states).gather(1, actions.unsqueeze(dim=1))

        loss = F.mse_loss(qle, q_target.unsqueeze(dim=1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.actor, self.target, tau)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float, device=self.device)
            return torch.argmax(self.actor(state)).cpu().numpy().item()

    def save(self, name="agent.pkl"):
        torch.save(self.actor.model, name)
