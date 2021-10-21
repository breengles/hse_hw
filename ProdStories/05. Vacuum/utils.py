import os
from datetime import datetime
import random

import numpy as np
import torch
import wandb


def set_seed(env, seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def wandb_init(global_config):
    cfg = global_config["wandb"]

    if cfg["local"]:
        os.environ["WANDB_MODE"] = "offline"
    else:
        wandb.login()

    now = datetime.now()

    run = wandb.init(
        project=cfg["project"],
        name=f'{cfg["name"]}:{now.hour}:{now.minute}:{now.second}-{now.day}.{now.month}.{now.year}',
        group=cfg.get("group", None),
        notes=cfg["notes"],
        entity=cfg["entity"],
        config=global_config,
    )

    return run, wandb.config
