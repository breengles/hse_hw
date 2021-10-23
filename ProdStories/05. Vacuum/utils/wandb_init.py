import os
import wandb
from datetime import datetime


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
