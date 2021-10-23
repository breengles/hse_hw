#!/usr/bin/env python

from argparse import ArgumentParser

import yaml
from agent.DQN import DQN
from utils.evaluation import generate_gif
from utils.wandb_init import wandb_init

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--local", "-l", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["wandb"]["local"] = args.local
    run, cfg = wandb_init(cfg)

    agent = DQN(env_config=cfg["env"], **cfg["agent"])
    agent = agent.train(**cfg["train"])

    generate_gif(agent, cfg["env"])
