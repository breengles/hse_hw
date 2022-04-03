import os
import random

import numpy as np
import torch


def set_seed(env, seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
