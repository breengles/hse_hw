#!/usr/bin/env python3

import os, torch, uuid, pprint, json
from tqdm import tqdm, trange
import numpy as np
from copy import deepcopy
from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from agent import Agent, MADDPG
from utils import ReplayBuffer, set_seed, Logger
from wrapper import VectorizeWrapper
from argparse import ArgumentParser


def render(config, path, step, device="cpu"):
    maddpg = torch.load(path + f"{step}.pt")
    maddpg.to(device).eval()
    eval_maddpg(config, maddpg, device=device, render=True)


def add_noise(action, sigma, lower=-1, upper=1):
    return np.clip(action + np.random.normal(scale=sigma, size=action.shape), lower, upper)


def rollout(env, agents):    
    _, states = env.reset()
    total_reward = []
    
    done = False
    while not done:
        actions = np.hstack([agent.act(agent_state) for agent, agent_state in zip(agents, states)])
        _, states, rewards, done = env.step(actions)
        total_reward.append(rewards)
    
    return np.vstack(total_reward).sum(axis=0)


def eval_maddpg(config, maddpg, device="cpu", n_evals=10, render=False, seed=None):
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=config, render=render))
    if seed:
        set_seed(seed)
    
    rewards = [rollout(env, maddpg.agents, greedy=True) for _ in range(n_evals)]
    return np.vstack(rewards).mean(axis=0)


def train(transitions=200_000, 
          hidden_size=64, 
          buffer_size=10000, batch_size=512, 
          actor_lr=1e-3, critic_lr=1e-3, 
          gamma=0.998, tau=0.005, 
          sigma_max=0.2, sigma_min=0, 
          seed=None, info=False, saverate=-1,
          return_agents=True, config=DEFAULT_CONFIG, verbose=False, device="cpu"):
    if saverate == -1:
        saverate = transitions // 100
        
    if info:
        pprint.pprint(config)
        print(pprint.pformat(config))
        
    logger = Logger(locals())
    saved_dir = "experiments/" + str(uuid.uuid4()) + "/"
    print("Experiment is saved:", saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    logger.save_params(saved_dir + "params.json")
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=config, render=False))
    if seed is not None:
        set_seed(env, seed)
    
    n_preds, n_preys, n_obstacles = (
        env.n_preds, env.n_preys,
        config["game"]["num_obsts"],
    )
    
    state_dim = n_preds * 4 + n_preys * 5 + n_obstacles * 3
    
    pred_config = {
        "team": "predator",
        "n_agents": n_preds,
        "state_dim": state_dim,
        "actor_action_dim": 1,
        "critic_action_dim": n_preds + n_preys,
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "gamma": gamma,
        "tau": tau,
        "hiddent_size": hidden_size,
    }
    
    prey_config = {
        "team": "predator",
        "n_agents": n_preys,
        "state_dim": state_dim,
        "actor_action_dim": 1,
        "critic_action_dim": n_preds + n_preys,
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "gamma": gamma,
        "tau": tau,
        "hiddent_size": hidden_size,
    }
    
    buffer = ReplayBuffer(n_agents=n_preds + n_preys, state_dim=state_dim, action_dim=1, size=buffer_size, device=device)
    maddpg = MADDPG(pred_config, prey_config, buffer, device)
    
    print("Filling buffer...")
    gstate, agent_states = env.reset()
    done = False
    for _ in trange(16 * batch_size):
        if done:
            gstate, agent_states = env.reset()
        actions = np.random.uniform(-1, 1, n_preds + n_preys)
        next_gstate, next_agent_states, rewards, done = env.step(actions)
        maddpg.buffer.add((
            gstate, agent_states, 
            actions, 
            next_gstate, next_agent_states, 
            rewards, done
        ))

    print("Finished. Now training...")
    gstate, agent_states = env.reset()
    done = False
    for step in trange(transitions):
        actions = np.hstack([agent.act(state) for agent, state in zip(maddpg.agents, agent_states)])
        
        next_gstate, next_agent_states, rewards, done = env.step(actions)
        
        maddpg.buffer.add((
            gstate, agent_states,
            actions,
            next_gstate, next_agent_states,
            rewards, done
        ))
        
        if done:
            gstate, agent_states = env.reset()
        else:
            gstate = next_gstate
            agent_states = next_agent_states
    
        losses = maddpg.update(batch_size)
        
        if (step + 1) % saverate == 0:
            rewards_ = eval_maddpg(config, maddpg, device=device, seed=seed)
            maddpg.save(saved_dir + f"{step + 1}.pt")
            
            print(f"=== Step {step + 1} ===")
            for i in range(len(maddpg.agents)):
                actor_loss, critic_loss = losses[i]
                print(f"Agent{i + 1} -- Reward: {rewards[i]}, Critic loss: {actor_loss:0.5f}, Actor loss: {critic_loss:0.5f}")
    
    return maddpg, logger
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render", type=str, default="")
    parser.add_argument("-rs","--render-step", type=int, default=0)
    parser.add_argument("-t", "--transitions", type=int, default=100000)
    parser.add_argument("--saverate", type=int, default=-1)
    parser.add_argument("-b", "--buffer", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.998)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=0.2)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("-temp", "--temperature", type=float, default=30)
    parser.add_argument("--seed", type=int, default=None)

    opts = parser.parse_args()
    
    if opts.config:
        with open(opts.config, "r") as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
    
    if opts.train:
        _, _ = train(device=opts.device,
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
                    return_agents=False,
                    seed=opts.seed
                    )
        
    if opts.render:
        assert opts.render_step > 0
        render(config=config, path=opts.render, step=opts.render_step, device=opts.device)