#!/usr/bin/env python3

import os, torch, json, sys
import numpy as np
from train import rollout
from utils import set_seed, rollout
from wrapper import VectorizeWrapper
# from oleg_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from predators_and_preys_env.env import PredatorsAndPreysEnv
from argparse import ArgumentParser
from examples.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent


def render(env, agents, num_evals=25):
    for _ in range(num_evals):
        set_seed(env, np.random.randint(1, 10000))
        rollout(env, agents)
        
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--pred", action="store_true")
    parser.add_argument("--prey", action="store_true")
    parser.add_argument("--num-evals", type=int, default=25)
    
    opts = parser.parse_args()
    
    maddpg = torch.load(opts.model, map_location="cpu")

    with open(os.path.dirname(opts.model) + "/params.json") as j:
        params = json.load(j)

    pred_baseline = False
    if not (opts.pred or opts.prey):
        agents = maddpg.agents
        pred_baseline = params["pred_baseline"]
    else:
        agents = []
        if opts.pred:
            pred_baseline = True
            agents.append(ChasingPredatorAgent())
        else:
            agents.extend(maddpg.pred_agents)
        
        if opts.prey:
            agents.append(FleeingPreyAgent())
        else:
            agents.extend(maddpg.prey_agents)
        
    for agent in agents:
        agent.device = "cpu"
        
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=params["env_config"], 
                                                render=True), 
                           pred_baseline=pred_baseline,
                           return_state_dict=True)   
    render(env, agents, num_evals=opts.num_evals)
    