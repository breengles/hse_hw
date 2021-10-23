import numpy as np
import torch
import wandb
from tqdm.auto import trange
from wrapper import Wrapper

from utils.random_utils import set_seed


def rollout(env, agent, render=False):
    state, done = env.reset(), False

    frames = [env.render(mode="rgb_array", size=512).transpose(2, 0, 1)] if render else None

    total_reward = 0
    total_metric = 0
    with torch.no_grad():
        while not done:
            state, reward, done, metric, _ = env.step(agent.act(state))

            if render:
                frames.append(env.render(mode="rgb_array", size=512).transpose(2, 0, 1))

            total_reward += reward
            total_metric += metric

    return total_reward, total_metric, frames


def evaluate_policy(env_config, agent, episodes=5, seed=42):
    env = Wrapper(**env_config)
    set_seed(env=env, seed=seed)

    rewards = []
    metrics = []
    for _ in trange(episodes, desc="Evaluation", leave=False):
        rew, met, _ = rollout(env, agent)
        rewards.append(rew)
        metrics.append(met)

    return np.mean(rewards), np.std(rewards), np.mean(metrics), np.std(metrics)


def generate_gif(env_config, agent, seed=42, extension="mp4"):
    env = Wrapper(**env_config)
    set_seed(env, seed)

    _, _, frames = rollout(env, agent, render=True)

    wandb.log({"video": wandb.Video(np.array(frames), fps=30, format=extension)})
