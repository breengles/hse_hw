#!/usr/bin/env python3

import os, torch, uuid, pprint, json
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from agent import Agent
from utils import ReplayBuffer, set_seed, Logger
from wrapper import VectorizeWrapper
from argparse import ArgumentParser


def render(config, path_to_experiment_dir, step, device="cpu", hidden_size=64):
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
    
    for i, actor in enumerate(predator_agent.actors):
        actor.load_state_dict(torch.load(f"{path_to_experiment_dir}/predator_actor{i}_{step}.pt"))
        actor.eval()

    for i, actor in enumerate(prey_agent.actors):
        actor.load_state_dict(torch.load(f"{path_to_experiment_dir}/prey_actor{i}_{step}.pt"))
        actor.eval()

    evaluate_policy(config, predator_agent, predator_agent, render=True)



def add_noise(action, sigma, lower=-1, upper=1):
    return torch.clip(action + sigma * torch.randn_like(action), lower, upper)


def evaluate_policy(config, predator_agent, prey_agent, device="cpu", n_evals=10, render=False):
    env = VectorizeWrapper(PredatorsAndPreysEnv(config, render=render))
    preys_reward, predators_reward = [], []
    for _ in range(n_evals):
        r_prey, r_pred = 0, 0
        done = False
        _, rel_pred_states, rel_prey_states = env.reset()
        while not done:
            a_pred = predator_agent.act(torch.tensor(rel_pred_states, device=device))
            a_prey = prey_agent.act(torch.tensor(rel_prey_states, device=device))

            _, next_rel_pred_state, next_rel_prey_state, r_pred_, r_prey_, done = env.step(a_pred, a_prey)

            r_pred_step = np.mean(r_pred_)
            r_prey_step = np.mean(r_prey_)

            r_prey += r_prey_step
            r_pred += r_pred_step

            rel_pred_states = deepcopy(next_rel_pred_state)
            rel_prey_states = deepcopy(next_rel_prey_state)

        # preys_alive = sum(map(lambda x: x["is_alive"], state["preys"]))
        preys_reward.append(r_prey)
        predators_reward.append(r_pred)

    return {"preys": preys_reward, "predators": predators_reward}


def train(device, 
          transitions=200_000, 
          hidden_size=64, 
          buffer_size=10000, batch_size=512, 
          actor_lr=1e-3, critic_lr=1e-3, 
          gamma=0.998, tau=0.005, 
          sigma_max=0.2, sigma_min=0, 
          render=False, seed=None, info=False, saverate=-1,
          return_agents=True, config=DEFAULT_CONFIG, verbose=False):
    if saverate == -1:
        saverate = transitions // 100
        
    if info:
        pprint.pprint(config)
        print(pprint.pformat(config))
        
    logger = Logger(locals())
    saved_agent_dir = "experiments/" + str(uuid.uuid4()) + "/"
    print("Experiment is saved:", saved_agent_dir)
    os.makedirs(saved_agent_dir, exist_ok=True)
    logger.save_params(saved_agent_dir + "params.json")
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=config, render=render))
    if seed is not None:
        set_seed(env, seed)
    
    # env = PredatorsAndPreysEnv(config=config, render=render)
    
    n_preds, n_preys, n_obstacles = (
        env.predator_action_size,
        env.prey_action_size,
        config["game"]["num_obsts"],
    )
    
    state_dim = n_preds * 4 + n_preys * 5 + n_obstacles * 3
    
    predator_agent = Agent(
        n_agents=n_preds,
        buffer=ReplayBuffer(state_dim, n_preds, buffer_size=buffer_size, device=device),
        state_dim=state_dim,
        action_dim=n_preds,
        hidden_size=hidden_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        gamma=gamma,
        device=device
    )
    prey_agent = Agent(
        n_agents=n_preys,
        buffer=ReplayBuffer(state_dim, n_preys, buffer_size=buffer_size, device=device),
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
        f.write("=== Predator agent ===")
        for actor in predator_agent.actors:
            f.write(str(actor))
        f.write("\n")
        f.write("=== Prey agent ===")
        for actor in prey_agent.actors:
            f.write(str(actor))

    print("Filling up buffer...")
    gstate, rel_pred_state, rel_prey_state = env.reset()
    for _ in range(16 * batch_size):
        a_pred = np.random.uniform(-1, 1, n_preds)
        a_prey = np.random.uniform(-1, 1, n_preys)

        next_gstate, next_rel_pred_state, next_rel_prey_state, r_pred, r_prey, done = env.step(a_pred, a_prey)
        
        # transition:
        # (global_state, rel_state, action, next_global_state, next_rel_state, reward, done)
        predator_agent.buffer.add((gstate, 
                                   rel_pred_state, 
                                   a_pred, 
                                   next_gstate, 
                                   next_rel_pred_state,
                                   r_pred, 
                                   done))
        prey_agent.buffer.add((gstate, 
                               rel_prey_state, 
                               a_prey, 
                               next_gstate, 
                               next_rel_prey_state,
                               r_prey, 
                               done))

        if done:
            gstate, rel_pred_state, rel_prey_state = env.reset()
        else:
            gstate = next_gstate
            rel_pred_state = next_rel_pred_state
            rel_prey_state = next_rel_prey_state
    
    debug = open(saved_agent_dir + "debug.txt", "a+") if verbose else None
    
    print("Finished. Now training...")
    gstate, rel_pred_state, rel_prey_state = env.reset()
    for i in tqdm(range(transitions)):
        sigma = sigma_max - (sigma_max - sigma_min) * i / transitions
        
        a_pred = predator_agent.act(torch.tensor(rel_pred_state, device=device))
        a_prey = prey_agent.act(torch.tensor(rel_prey_state, device=device))
        
        a_prey = add_noise(a_prey, sigma).cpu()
        a_predator = add_noise(a_predator, sigma).cpu()

        next_gstate, next_rel_pred_state, next_rel_prey_state, r_pred, r_prey, done = env.step(a_pred, a_prey)
        
        if debug is not None:
            debug.write(f"{i + 1}: {a_pred} | {a_prey} | {r_pred} | {r_prey}\n")
            debug.flush()

        predator_agent.buffer.add((gstate, 
                                   rel_pred_state, 
                                   a_pred, 
                                   next_gstate, 
                                   next_rel_pred_state,
                                   r_pred, 
                                   done))
        prey_agent.buffer.add((gstate, 
                               rel_prey_state, 
                               a_prey, 
                               next_gstate, 
                               next_rel_prey_state,
                               r_prey, 
                               done))

        predator_agent.update(batch_size)
        prey_agent.update(batch_size)

        if done:
            gstate, rel_pred_state, rel_prey_state = env.reset()
        else:
            gstate = next_gstate
            rel_pred_state = next_rel_pred_state
            rel_prey_state = next_rel_prey_state

        if (i + 1) % saverate == 0:
            rewards = evaluate_policy(config, predator_agent, prey_agent, device=device)

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
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render", type=str, default="")
    parser.add_argument("--render-step", type=int, default=0)
    parser.add_argument("--transitions", type=int, default=100000)
    parser.add_argument("--saverate", type=int, default=-1)
    parser.add_argument("--buffer", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.998)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=0.2)
    parser.add_argument("--sigma-min", type=float, default=0.0)

    opts = parser.parse_args()
    
    if opts.config:
        with open(opts.config, "r") as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
    
    if opts.train:
        log = train(device=opts.device,
                    transitions=opts.transitions,
                    saverate=opts.saverate,
                    buffer_size=opts.buffer,
                    batch_size=opts.batch,
                    config=config,
                    actor_lr=opts.actor_lr,
                    critic_lr=opts.critic_lr,
                    gamma=opts.gamma,
                    tau=opts.tau,
                    sigma_max=opts.sigma_max,
                    sigma_min=opts.sigma_min,
                    hidden_size=opts.hidden_size,
                    info=opts.info,
                    verbose=opts.verbose,
                    render=False,
                    )
        log.plot("step", "predator_mean", "predator_std", title="Predator")
        
    if opts.render:
        assert opts.render_step > 0
        render(config, opts.render, opts.render_step, opts.device, opts.hidden_size)