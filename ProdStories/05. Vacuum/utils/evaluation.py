import numpy as np
import torch

from wrapper import Wrapper
from utils.random_utils import set_seed
import wandb
from mapgen import Dungeon, VideoRecorder
from tqdm.auto import trange


def rollout(env, agent):
    state, done = env.reset().transpose(2, 0, 1), False

    total_reward = 0
    total_metric = 0

    with torch.no_grad():
        while not done:
            state, reward, done, metric, _ = env.step(agent.act(state))
            state = state.transpose(2, 0, 1)
            total_reward += reward
            total_metric += metric

    return total_reward, total_metric


def evaluate_policy(env_config, agent, episodes=5, seed=42):
    env = Wrapper(**env_config)

    set_seed(env=env, seed=seed)
    rewards = []
    metrics = []

    for _ in trange(episodes, desc="Evaluation", leave=False):
        rew, met = rollout(env, agent)
        rewards.append(rew)
        metrics.append(met)

    return np.mean(rewards), np.std(rewards), np.mean(metrics), np.std(metrics)


def generate_gif(env_config, agent, seed=42, extension="mp4"):
    env = VideoRecorder(
        Wrapper(**env_config),
        video_path="videos",
        size=512,
        fps=60,
        extension=extension,
    )

    set_seed(env, seed)

    rollout(env, agent)

    wandb.log({"video": wandb.Video(env.filename, fps=15, format=extension)})
