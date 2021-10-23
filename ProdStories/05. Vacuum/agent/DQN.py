import copy
import os
from datetime import datetime
from enum import Enum, auto

import numpy as np
import torch
import wandb
from torch.nn import functional as F
from torch.optim import Adam
from tqdm.auto import trange
from utils.buffer import Buffer
from utils.evaluation import evaluate_policy, generate_gif
from utils.random_utils import set_seed
from wrapper import Wrapper

from agent.actors import Actor, DuelingActor


class Algo(Enum):
    VANILLA = auto()
    DOUBLE = auto()
    DUELING = auto()
    DD = auto()


KINDS = {
    "vanilla": Algo.VANILLA,
    "double": Algo.DOUBLE,
    "dueling": Algo.DUELING,
    "duel": Algo.DUELING,
    "doubledueling": Algo.DD,
    "dd": Algo.DD,
}


def soft_update(source, target, tau=0.002):
    with torch.no_grad():
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * sp.data)


class DQN:
    def __init__(self, env_config, lr=2e-4, gamma=0.999, tau=0.002, eps_max=0, eps_min=0, device="cpu", kind=None):
        self.env = Wrapper(**env_config)
        self.env_config = env_config

        self.gamma = gamma
        self.tau = tau
        self.eps_max = eps_max
        self.eps_min = eps_min

        self.kind = KINDS.get(kind, Algo.VANILLA)

        if self.kind == Algo.VANILLA:
            self.actor = Actor(self.env.observation_space.shape[-1] - 1, self.env.action_space.n)
        elif self.kind in (Algo.DUELING, Algo.DD):
            self.actor = DuelingActor(self.env.observation_space.shape[-1] - 1, self.env.action_space.n)

        self.target = copy.deepcopy(self.actor)

        self.optimizer = Adam(self.actor.parameters(), lr=lr)

        self.actor.to(device)
        self.target.to(device)
        self.device = device

    def update(self, batch):
        states, actions, next_states, rewards, dones = batch

        with torch.no_grad():
            if self.kind in (Algo.DOUBLE, Algo.DD):
                actions_next = torch.argmax(self.actor(next_states), dim=-1)
                q_next = self.target(next_states).gather(1, actions_next.unsqueeze(1))
            else:
                q_next = self.target(next_states).max(dim=1)[0].view(-1)

            q_next[dones] = 0

        q_target = rewards + self.gamma * q_next.flatten()

        q_current = self.actor(states).gather(1, actions.unsqueeze(1)).flatten()

        loss = F.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.actor, self.target, self.tau)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float, device=self.device)
            return torch.argmax(self.actor(state)).cpu().numpy().item()

    def save(self, name="agent.pkl"):
        torch.save(self.actor.model, name)

    def train(
        self,
        transitions=1_000_000,
        buffer_size=100_000,
        batch_size=512,
        seed=42,
        saverate=None,
    ):
        if seed is not None:
            set_seed(self.env, seed)

        if saverate is None:
            saverate = transitions // 100

        dir_name = "experiments/" + str(datetime.now().strftime("%d_%m_%Y/%H:%M:%S.%f")) + "/"
        os.makedirs(dir_name, exist_ok=True)

        state_dim = np.prod(self.env.observation_space.shape)

        buffer = Buffer(state_dim, max_size=buffer_size, device=self.device)

        state, done = self.env.reset(), False

        for step in trange(transitions):
            eps = self.eps_max - (self.eps_max - self.eps_min) * step / transitions

            if done:
                state = self.env.reset()
                done = False

            action = self.env.action_space.sample() if np.random.uniform() < eps else self.act(state)

            next_state, reward, done, *_ = self.env.step(action)

            transition = (state, action, next_state, reward, done)
            buffer.push(*transition)

            state = next_state

            if len(buffer) >= batch_size * 16:
                self.update(buffer.sample(batch_size))

                if (step + 1) % saverate == 0:
                    reward_mean, reward_std, metric_mean, metric_std = evaluate_policy(self.env_config, self, 5)

                    torch.save(
                        {
                            "epoch": step + 1,
                            "model_state_dict": self.actor.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "metric": metric_mean,
                        },
                        dir_name + "model.pth",
                    )
                    artifact = wandb.Artifact("model", type="model")
                    artifact.add_file(dir_name + "model.pth")
                    wandb.log_artifact(artifact)

                    wandb.log(
                        {
                            "reward/mean": reward_mean,
                            "reward/std": reward_std,
                            "metric/mean": metric_mean,
                            "metric/std": metric_std,
                        },
                    )

                    generate_gif(self.env_config, self)

        return self
