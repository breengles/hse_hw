#!/usr/bin/env python

import os
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
import yaml
from tqdm.auto import trange

from DQN import DQN
from datetime import datetime

from buffer import Buffer
from mapgen import Dungeon, VideoRecorder
from utils import set_seed, wandb_init
from wrapper import Wrapper


def generate_gif(agent, env_config, seed=42):
    env = VideoRecorder(
        Dungeon(**env_config),
        video_path="videos",
        size=512,
        fps=60,
        extension="gif",
    )

    set_seed(env, seed)

    rollout(env, agent)

    wandb.log({"video": wandb.Video(env.filename, fps=30, format="gif")})


def rollout(env, agent):
    state, done = env.reset().flatten(), False

    total_reward = 0

    with torch.no_grad():
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            state = state.flatten()
            total_reward += reward

    return total_reward


def evaluate_policy(env_config, agent, episodes=5, seed=42):
    env = Dungeon(**env_config)

    set_seed(env=env, seed=seed)
    rewards = []

    for _ in range(episodes):
        rewards.append(rollout(env, agent))

    return np.mean(rewards), np.std(rewards)


def train(
    transitions=1_000_000,
    hidden_dim=64,
    buffer_size=100_000,
    batch_size=512,
    actor_lr=1e-3,
    gamma=0.998,
    tau=0.005,
    sigma_max=0.2,
    sigma_min=0,
    seed=42,
    saverate=None,
    env_config=None,
    device="cpu",
):
    env = Wrapper(**env_config)

    if seed is not None:
        set_seed(env, seed)

    if saverate is None:
        saverate = transitions // 100

    dir_name = "experiments/" + str(datetime.now().strftime("%d_%m_%Y/%H:%M:%S.%f")) + "/"
    os.makedirs(dir_name, exist_ok=True)

    state_dim = np.prod(env.observation_space.shape)

    agent = DQN(state_dim, env.action_space.n, hidden_dim, actor_lr, device=device)
    buffer = Buffer(state_dim, max_size=buffer_size, device=device)

    state, done = env.reset().flatten(), False

    for step in trange(transitions):
        sigma = sigma_max - (sigma_max - sigma_min) * step / transitions

        # if done:
        #     state = env.reset()
        #     done = False

        action = env.action_space.sample() if np.random.uniform() < sigma else agent.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = next_state.flatten()

        transition = (state, action, next_state, reward, done)
        buffer.push(*transition)

        state = env.reset().flatten() if done else next_state

        if len(buffer) >= batch_size * 16:
            agent.update(buffer.sample(batch_size), gamma=gamma, tau=tau)

            if (step + 1) % saverate == 0:
                reward_mean, reward_std = evaluate_policy(env_config, agent)

                wandb.log(
                    {
                        "reward_mean": reward_mean,
                        "reward_std": reward_std,
                    },
                )

    return agent


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--local", "-l", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["wandb"]["local"] = args.local
    run, cfg = wandb_init(cfg)

    agent = train(**cfg["train"], env_config=cfg["env_config"])

    generate_gif(agent, cfg["env_config"])
