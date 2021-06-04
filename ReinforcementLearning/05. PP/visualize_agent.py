#!/usr/bin/env python3

import torch
import sys
from agent import Agent
from train import evaluate_policy
from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from game_configs import *


if __name__ == "__main__":
    path_to_exp_dir = "experiments/0b6db652-effd-41f2-9f99-6d1fdb37f602/"
    step = int(sys.argv[-1])
    pred_actor_dict = torch.load(f"{path_to_exp_dir}predator_actor_{step}.pt")
    prey_actor_dict = torch.load(f"{path_to_exp_dir}prey_actor_{step}.pt")
    
    config = simple
    device = "cpu"
    hidden_size = 64
    
    env = PredatorsAndPreysEnv(config=config, render=True)
    n_predators, n_preys, n_obstacles = (
        env.predator_action_size,
        env.prey_action_size,
        config["game"]["num_obsts"],
    )
    
    predator_agent = Agent(
        state_dim=n_predators * 4 + n_preys * 5 + n_obstacles * 3,
        action_dim=n_predators,
        n_agents=n_predators,
        hidden_size=hidden_size,
        device=device,
        buffer=None,  # it does not matter at eval
    )
    prey_agent = Agent(
        state_dim=n_predators * 4 + n_preys * 5 + n_obstacles * 3,
        action_dim=n_preys,
        n_agents=n_preys,
        hidden_size=hidden_size,
        device=device,
        buffer=None,  # it does not matter at eval
    )
    
    predator_agent.actor.load_state_dict(pred_actor_dict)
    prey_agent.actor.load_state_dict(prey_actor_dict)
    predator_agent.actor.eval()
    prey_agent.actor.eval()

    evaluate_policy(config, predator_agent, predator_agent, render=True)

