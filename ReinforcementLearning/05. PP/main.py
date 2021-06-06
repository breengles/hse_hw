#!/usr/bin/env python3

import os, torch, uuid, pprint, json
from tqdm import trange
import numpy as np
from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from agent import MADDPG
from utils import ReplayBuffer, set_seed, Logger
from wrapper import VectorizeWrapper
from argparse import ArgumentParser


def render(config, path, step, num_evals=200, device="cpu", seed=None):
    maddpg = torch.load(path + f"/{step}.pt", map_location=device)
    
    for agent in maddpg.agents:
        agent.device = device
        # agent.actor.to(device)
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=config, render=render))
    if seed:
        set_seed(env, seed)
    for _ in range(num_evals):
        rollout(env, maddpg.agents)


def rollout(env, agents):
    total_reward = []
    _, states = env.reset()
    done = False
    while not done:
        actions = np.hstack([agent.act(state) for agent, state in zip(agents, states)])
        _, states, rewards, done = env.step(actions)
        total_reward.append(rewards)
    
    return np.vstack(total_reward).sum(axis=0)


def eval_maddpg(config, maddpg, n_evals=25, render=False, seed=None):
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=config, render=render))
    if seed:
        set_seed(env, seed)
    
    rewards = [rollout(env, maddpg.agents) for _ in range(n_evals)]
    return np.vstack(rewards).mean(axis=0)


def train(transitions=200_000, hidden_size=64,  buffer_size=10000, 
          batch_size=512, actor_lr=1e-3, critic_lr=1e-3, gamma=0.998, tau=0.005, 
          sigma_max=0.2, sigma_min=0, seed=None, info=False, saverate=-1,
          env_config=DEFAULT_CONFIG, update_rate=1, num_updates=1,
          temperature=1, device="cpu", buffer_init=False, verbose=False):
    if saverate == -1:
        saverate = transitions // 100
        
    if info:
        pprint.pprint(env_config)
        print(pprint.pformat(env_config))
        
    logger = Logger(locals())
    saved_dir = "experiments/" + str(uuid.uuid4()) + "/"
    print("Experiment is saved:", saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    logger.save_params(saved_dir + "params.json")
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=env_config, render=False))
    if seed is not None:
        set_seed(env, seed)
    
    n_preds, n_preys, n_obstacles = (
        env.n_preds, env.n_preys,
        env_config["game"]["num_obsts"],
    )
    
    state_dim = n_preds * 4 + n_preys * 5 + n_obstacles * 3
    
    pred_config = {
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "gamma": gamma,
        "tau": tau,
        "hidden_size": hidden_size,
    }
    
    prey_config = {
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "gamma": gamma,
        "tau": tau,
        "hidden_size": hidden_size,
    }
    
    buffer = ReplayBuffer(n_agents=n_preds + n_preys, 
                          state_dim=state_dim, 
                          action_dim=1, 
                          size=buffer_size, 
                          device=device)
    maddpg = MADDPG(n_preds, n_preys, state_dim, 1, pred_config, prey_config, 
                    device=device, temperature=temperature, verbose=verbose)
    
    if buffer_init:
        print("Filling buffer...")
        gstate, agent_states = env.reset()
        done = False
        for _ in trange(16 * batch_size):
            if done:
                gstate, agent_states = env.reset()
            actions = np.random.uniform(-1, 1, n_preds + n_preys)
            next_gstate, next_agent_states, rewards, done = env.step(actions)
            buffer.add((
                gstate, agent_states, 
                actions, 
                next_gstate, next_agent_states, 
                rewards, done
            ))
        print("Finished. Now training...")

    gstate, agent_states = env.reset()
    done = False
    for step in trange(transitions):
        if done:
            gstate, agent_states = env.reset()
            done = False
        sigma = sigma_max - (sigma_max - sigma_min) * step / transitions
        actions = np.hstack([agent.act(state) for agent, state in zip(maddpg.agents, agent_states)])
        actions = np.clip(actions + np.random.normal(scale=sigma, size=actions.shape), -1, 1)
        
        next_gstate, next_agent_states, reward, done = env.step(actions)
        
        buffer.add((
            gstate, agent_states,
            actions,
            next_gstate, next_agent_states,
            reward, done
        ))
        
        agent_states = next_agent_states
        gstate = next_gstate
        
        if step % update_rate == 0 and (step > 16 * batch_size or buffer_init):
            for _ in range(num_updates):
                batch = buffer.sample(batch_size)
                losses = maddpg.update(batch)
        
            if (step + 1) % saverate == 0:
                rewards = eval_maddpg(env_config, maddpg, seed=seed)
                maddpg.save(saved_dir + f"{step + 1}.pt")
                
                print(f"=== Step {step + 1} ===")
                for i in range(len(maddpg.agents)):
                    actor_loss, critic_loss = losses[i]
                    print(f"Agent{i + 1} -- Reward: {rewards[i]}, Actor loss: {actor_loss:0.5f}, Critic loss: {critic_loss:0.5f}")
                
                
                preds_total_reward = np.sum(rewards[:n_preds])
                preys_total_reward = np.sum(rewards[-n_preys:])
                logger.log("step", step + 1)
                logger.log("preds_total_reward", preds_total_reward)
                logger.log("preys_total_reward", preys_total_reward)
                logger.save(saved_dir + "log.csv")
                
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
    parser.add_argument("--buffer", type=int, default=200000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--env-config", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--sigma-max", type=float, default=0.3)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--update-rate", type=int, default=1)
    parser.add_argument("--num-updates", type=int, default=1)
    parser.add_argument("--buffer-init", action="store_true")

    opts = parser.parse_args()
    
    if opts.env_config:
        with open(opts.env_config, "r") as f:
            env_config = json.load(f)
    else:
        env_config = DEFAULT_CONFIG
    
    if opts.train:
        train(transitions=opts.transitions, 
              hidden_size=opts.hidden_size, 
              buffer_size=opts.buffer, 
              batch_size=opts.batch, 
              actor_lr=opts.actor_lr, 
              critic_lr=opts.critic_lr, 
              gamma=opts.gamma, 
              tau=opts.tau, 
              sigma_max=opts.sigma_max, 
              sigma_min=opts.sigma_min, 
              seed=opts.seed, 
              info=opts.info, 
              saverate=opts.saverate, 
              env_config=env_config, 
              update_rate=opts.update_rate, 
              num_updates=opts.num_updates, 
              temperature=opts.temperature, 
              device=opts.device,
              verbose=opts.verbose,
              buffer_init=opts.buffer_init)
        
    if opts.render:
        assert opts.render_step > 0
        render(config=env_config, path=opts.render, step=opts.render_step, device=opts.device, seed=opts.seed)
        
# Sanyok
# ./main.py --train -t 1000000 --buffer 1000000 --batch 256 --env-config configs/2v1.json --seed 10 --saverate 10000 --actor-lr 0.001 --critic-lr 0.001 --gamma 0.99 --tau 0.001 --sigma-max 0.3

# ./main.py --train -t 2000000 --buffer 1000000 --batch 1024 --env-config configs/1v1.json --seed 42 --actor-lr 0.001 --critic-lr 0.001 --gamma 0.99 --tau 0.001