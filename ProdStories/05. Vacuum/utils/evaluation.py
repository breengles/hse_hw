import numpy as np
import torch
import wandb
from tqdm.auto import trange
from wrapper import Wrapper
from utils.random_utils import set_seed
from wrapper import VideoRecorder


def rollout(env, agent):
    state, done = env.reset(), False

    total_reward = 0
    with torch.no_grad():
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward

    return total_reward


def evaluate_policy(env_config, agent, episodes=5, seed=42):
    env = Wrapper(**env_config)
    set_seed(env=env, seed=seed)

    rewards = []
    for _ in trange(episodes, desc="Evaluation", leave=False):
        rew = rollout(env, agent)
        rewards.append(rew)

    return np.mean(rewards), np.std(rewards)


def generate_gif(env_config, agent, seed=42, video_path="videos", extension="mp4"):
    env = VideoRecorder(Wrapper(**env_config), video_path, extension=extension)
    set_seed(env.env, seed)

    obs, done = env.reset(), False
    while not done:
        action = agent.act(obs)
        obs, _, done, _ = env.step(action)

    wandb.log({"video": wandb.Video(env.filename, fps=15, format=extension)})
