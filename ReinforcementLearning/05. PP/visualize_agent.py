#!/usr/bin/env python3

import torch
from agent import Agent
from train import evaluate_policy
from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG

game_1v1 = {
    "game": {
        "num_obsts": 16,
        "num_preds": 1,
        "num_preys": 1,
        "x_limit": 12,
        "y_limit": 12,
        "obstacle_radius_bounds": [0.8, 2.0],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/100,
        "frameskip": 5
    },
    "environment": {
        "frameskip": 5,
        "time_limit": 1000
    }
}

if __name__ == "__main__":
    config = game_1v1
    device = "cpu"
    
    env = PredatorsAndPreysEnv(config=config, render=True)
    n_predators, n_preys, n_obstacles = (
        env.predator_action_size,
        env.prey_action_size,
        config["game"]["num_obsts"],
    )
    
    predator_agent = Agent(
        n_agents=n_predators,
        buffer=None,
        state_dim=n_predators * 4 + n_preys * 5 + n_obstacles * 3,
        action_dim=n_predators,
        hidden_size=64,
        actor_lr=1,
        critic_lr=1,
        tau=1,
        gamma=1,
        device=device
    )
    prey_agent = Agent(
        n_agents=n_preys,
        buffer=None,
        state_dim=n_predators * 4 + n_preys * 5 + n_obstacles * 3,
        action_dim=n_preys,
        hidden_size=64,
        actor_lr=1,
        critic_lr=1,
        tau=1,
        gamma=1,
        device=device
    )
    
    pred_actor_dict = torch.load("experiments/fe1c14a0-fb29-42c8-b02c-16c6d572fc70/predator_actor_100.pt")
    prey_actor_dict = torch.load("experiments/fe1c14a0-fb29-42c8-b02c-16c6d572fc70/prey_actor_100.pt")
    
    predator_agent.actor.load_state_dict(pred_actor_dict)
    prey_agent.actor.load_state_dict(prey_actor_dict)

    predator_agent.actor.eval()
    prey_agent.actor.eval()

    evaluate_policy(game_1v1, predator_agent, predator_agent, render=True)

