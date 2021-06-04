#!/usr/bin/env python3

import os, torch, uuid
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from agent import Agent
from utils import ReplayBuffer, set_seed, Logger


def add_noise(action, sigma, lower=-1, upper=1):
    return torch.clip(action + sigma * torch.randn_like(action), lower, upper)


def evaluate_policy(config, predator_agent, prey_agent, n_evals=10, render=False):
    env = PredatorsAndPreysEnv(config, render=render)
    preys_reward, predators_reward = [], []
    for _ in range(n_evals):
        r_prey, r_predator = 0, 0
        done = False
        state = env.reset()
        while not done:
            a_prey = prey_agent.act(state).cpu()
            a_predator = predator_agent.act(state).cpu()

            next_state, reward, done = env.step(a_predator, a_prey)

            r_prey_step = np.mean(reward["preys"])
            r_predator_step = np.mean(reward["predators"])

            r_prey += r_prey_step
            r_predator += r_predator_step

            state = deepcopy(next_state)

        # preys_alive = sum(map(lambda x: x["is_alive"], state["preys"]))
        preys_reward.append(r_prey)
        predators_reward.append(r_predator)

    return {"preys": preys_reward, "predators": predators_reward}


def train(device, 
          transitions=200_000, 
          hidden_size=64, 
          buffer_size=10000, batch_size=512, 
          actor_lr=1e-3, critic_lr=1e-3, 
          gamma=0.998, tau=0.005, 
          sigma_max=0.2, sigma_min=0, 
          render=False, seed=None, info=False, saverate=-1,
          return_agents=True, config=DEFAULT_CONFIG):
    if saverate == -1:
        saverate = transitions // 100
    logger = Logger(locals())
    saved_agent_dir = "experiments/" + str(uuid.uuid4()) + "/"
    print("Experiment is saved:", saved_agent_dir)
    os.makedirs(saved_agent_dir, exist_ok=True)
    logger.save_params(saved_agent_dir + "params.json")
    
    if seed is not None:
        set_seed(seed)
        
    env = PredatorsAndPreysEnv(config=config, render=render)
    n_predators, n_preys, n_obstacles = (
        env.predator_action_size,
        env.prey_action_size,
        config["game"]["num_obsts"],
    )
    
    state_dim = n_predators * 4 + n_preys * 5 + n_obstacles * 3
    
    predator_buffer = ReplayBuffer(state_dim, n_predators, buffer_size=buffer_size, device=device)
    prey_buffer = ReplayBuffer(state_dim, n_preys, buffer_size=buffer_size, device=device)

    predator_agent = Agent(
        n_agents=n_predators,
        buffer=predator_buffer,
        state_dim=state_dim,
        action_dim=n_predators,
        hidden_size=hidden_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        gamma=gamma,
        device=device
    )
    prey_agent = Agent(
        n_agents=n_preys,
        buffer=prey_buffer,
        state_dim=state_dim,
        action_dim=n_preys,
        hidden_size=hidden_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        gamma=gamma,
        device=device
    )
    
    with open(saved_agent_dir + "models.txt", "w+") as f:
        f.write(str(predator_agent.actor))
        f.write("\n")
        f.write(str(prey_agent.actor))

    print("Filling up buffer...")
    state = env.reset()
    for _ in range(buffer_size):
        a_prey = np.random.uniform(-1, 1, n_preys)
        a_predator = np.random.uniform(-1, 1, n_predators)

        next_state, reward, done = env.step(a_predator, a_prey)
        r_prey = reward["preys"]
        r_predator = reward["predators"]

        prey_buffer.add((state, a_prey, next_state, r_prey, done))
        predator_buffer.add((state, a_predator, next_state, r_predator, done))

        state = env.reset() if done else next_state
        
    print("Finished. Now training...")
    state = env.reset()
    for i in tqdm(range(transitions)):
        sigma = sigma_max - (sigma_max - sigma_min) * i / transitions
        
        a_prey = prey_agent.act(state)
        a_prey = add_noise(a_prey, sigma, lower=0).cpu()
        
        a_predator = predator_agent.act(state)
        a_predator = add_noise(a_predator, sigma, lower=0).cpu()
        
        # print(a_prey, a_predator)

        next_state, reward, done = env.step(a_predator, a_prey)
        r_prey = reward["preys"]
        r_predator = reward["predators"]

        prey_agent.buffer.add((state, a_prey, next_state, r_prey, done))
        predator_agent.buffer.add((state, a_predator, next_state, r_predator, done))

        predator_agent.update(batch_size)
        prey_agent.update(batch_size)

        state = next_state if not done else env.reset()

        if (i + 1) % saverate == 0:
            rewards = evaluate_policy(config, predator_agent, prey_agent)

            predator_mean = np.mean(rewards["predators"])
            predator_std = np.std(rewards["predators"])

            prey_mean = np.mean(rewards["preys"])
            prey_std = np.std(rewards["preys"])

            if info:
                print(f"Step: {i + 1}")
                print(f"Predator: mean = {predator_mean} | std = {predator_std}")
                print(f"Prey:     mean = {prey_mean} | std = {prey_std}")

            predator_agent.save(saved_agent_dir + "predator", i + 1)
            prey_agent.save(saved_agent_dir + "prey", i + 1)
            
            # predator_agent.save("saved_models/predator_", i + 1)
            # prey_agent.save("saved_models/prey_", i + 1)
            
            logger.log("step", i + 1)
            logger.log("predator_mean", predator_mean)
            logger.log("predator_std", predator_std)
            logger.log("prey_mean", prey_mean)
            logger.log("prey_std", prey_std)
            logger.save(saved_agent_dir + "log.csv")
            
    if return_agents:
        return predator_agent, prey_agent, logger
    else:
        return logger
            

if __name__ == "__main__":
    train("cpu", transitions=1000, info=True, saverate=200)
