#!/usr/bin/env python3

import os, torch, json, sys
import numpy as np
from train import rollout
from utils import set_seed, rollout
from wrapper import VectorizeWrapper
from predators_and_preys_env.env import PredatorsAndPreysEnv


def render(path, num_evals=25, device="cpu"):
    model = path
    with open(os.path.dirname(path) + "/params.json") as j:
        params = json.load(j)
    config = params["env_config"]
    is_baseline = params["pred_baseline"] or params["prey_baseline"]
    
    maddpg = torch.load(model, map_location=device)
    
    for agent in maddpg.agents:
        agent.device = device
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=config, render=render), 
                           return_state_dict=is_baseline)
    for _ in range(num_evals):
        set_seed(env, np.random.randint(1, 10000))
        rollout(env, maddpg.agents)
        
    
if __name__ == "__main__":
    render(sys.argv[-1])
    