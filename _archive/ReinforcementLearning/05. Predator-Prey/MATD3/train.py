#!/usr/bin/env python3

import os, pprint, json
from tqdm import trange, tqdm
import numpy as np
from agent import MADDPG
from utils import ReplayBuffer, set_seed, Logger, rollout, Buffer, mse
from wrapper import VectorizeWrapper
from argparse import ArgumentParser
from datetime import datetime
import torch
from PER import PrioritizedReplayBuffer
from sklearn.metrics import mean_squared_error
from better_baseline import PredatorAgent, PreyAgent
import pprint


def eval_maddpg(config, maddpg, n_evals=25, render=False, seed=None, 
                distance_reward=True, time_penalty=False):
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=config, render=render, 
                                                distance_reward=distance_reward,
                                                time_penalty=time_penalty))
    if seed:
        set_seed(env, seed)
    rewards = [rollout(env, maddpg.agents) for _ in range(n_evals)]
    return np.vstack(rewards).mean(axis=0)


def train(title="", transitions=200_000, hidden_size=64,  buffer_size=10000, 
          batch_size=512, actor_lr=1e-3, critic_lr=1e-3, gamma=0.998, tau=0.005, 
          sigma_max=0.2, sigma_min=0, seed=None, info=False, saverate=-1,
          env_config=None, update_rate=1, num_updates=1, temperature=1, 
          device="cpu", buffer_init=False, verbose=False, 
          pred_baseline=False, prey_baseline=False, time_penalty=False,
          actor_update_delay=1, shared_actor=False, shared_critic=False,
          actor_reg=1e-5, distance_reward=True):
    
    if saverate == -1:
        saverate = transitions // 100
        
    if info:
        pprint.pprint(env_config)
    
    logger = Logger(locals())
    uniq_dir_name = datetime.now().strftime("%d_%m_%Y/%H:%M:%S.%f")
    saved_dir = "experiments/" + str(uniq_dir_name) + "/"
    # print("Experiment is saved:", saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    logger.save_params(saved_dir + "params.json")
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=env_config, 
                                                render=False,
                                                time_penalty=time_penalty,
                                                distance_reward=distance_reward))
    
    baseline_pred = PredatorAgent()
    baseline_prey = PreyAgent()
    
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
    
    # buffer = PrioritizedReplayBuffer(n_agents=n_preds + n_preys, 
    #                                  state_dim=state_dim, size=buffer_size, 
    #                                  alpha=0.25, device=device)

    # preds_feature, preys_feature, _ = env.reset()
    # buffer = Buffer(prey_feature_dim=preds_feature.shape[0], 
    #                 preys_feature_dim=preys_feature.shape[0], 
    #                 action_dim=n_preds + n_preys, max_size=buffer_size)
    
    maddpg = MADDPG(n_preds, n_preys, state_dim, 1, pred_config, prey_config, 
                    device=device, temperature=temperature, verbose=verbose,
                    pred_baseline=pred_baseline, prey_baseline=prey_baseline,
                    actor_update_delay=actor_update_delay, saverate=saverate,
                    shared_actor=shared_actor, shared_critic=shared_critic,
                    actor_action_reg_coef=actor_reg)
    
    if buffer_init:
        print("Filling buffer...")
        state_dict, gstate, agent_states = env.reset()
        
        done = False
        
        for _ in trange(batch_size, leave=False):
            if done:
                state_dict, gstate, agent_states = env.reset()
            actions = np.random.uniform(-1, 1, n_preds + n_preys)
            next_state_dict, next_gstate, next_agent_states, rewards, done = env.step(actions)
            buffer.push((
                state_dict, next_state_dict,
                gstate, agent_states, 
                actions, 
                next_gstate, next_agent_states, 
                rewards, done
            ))
            state_dict, gstate, agent_states = next_state_dict, next_gstate, next_agent_states
        print("Finished. Now training...")

    state_dict, gstate, agent_states = env.reset()
    
    pprint.pprint(state_dict)
    quit()
    
    done = False
    t = trange(transitions, desc=uniq_dir_name + "/ | " + title)
    for step in t:
        if done:
            state_dict, gstate, agent_states = env.reset()
            done = False

        states = []
        if pred_baseline:
            states.append(state_dict)
        else:
            states.extend(agent_states[:n_preds])
        
        if prey_baseline:
            states.append(state_dict)
        else:
            states.extend(agent_states[-n_preys:])

        beta = min(1.0, 0.4 + step * (1.0 - 0.4) / transitions)
        sigma = sigma_max - (sigma_max - sigma_min) * step / transitions
        actions = np.hstack([agent.act(state, sigma=sigma) for agent, state in zip(maddpg.agents, states)])
        
        
        next_state_dict, next_gstate, next_agent_states, reward, done = env.step(actions)
        
        baseline_pred_actions = baseline_pred.act(state_dict)
        baseline_prey_actions = baseline_prey.act(state_dict)
        baseline_actions = baseline_pred_actions + baseline_prey_actions
        reward = -mse(baseline_actions, actions)
        
        buffer.push((
            state_dict, next_state_dict,
            gstate, agent_states,
            actions,
            next_gstate, next_agent_states,
            reward, done
        ))
        
        state_dict = next_state_dict
        gstate = next_gstate
        agent_states = next_agent_states
        
        if step % update_rate == 0 and (step > batch_size or buffer_init):
            for _ in range(num_updates):
                # batch, (weights, idxes) = buffer.sample(batch_size)
                maddpg.update(buffer, batch_size=batch_size, step=step, beta=beta)
        
            if (step + 1) % saverate == 0:
                rewards = eval_maddpg(env_config, maddpg, seed=seed,
                                      distance_reward=distance_reward, 
                                      time_penalty=time_penalty)
                
                maddpg.save(saved_dir + f"{step + 1}.pt")
                
                tqdm.write(f"--- rewards ---")
                for idx, agent in enumerate(maddpg.agents):
                    tqdm.write(f"Agent{idx} ({agent.team}): {rewards[idx]}")
                
                preds_total_reward = np.sum(rewards[:n_preds])
                preys_total_reward = np.sum(rewards[-n_preys:])
                logger.log("step", step + 1)
                logger.log("preds_total_reward", preds_total_reward)
                logger.log("preys_total_reward", preys_total_reward)
                logger.save(saved_dir + "log.csv")
                
    print("Experiment is done:", saved_dir)
    print("Config:")
    pprint.pprint(logger.params)
    return maddpg, logger
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-t", "--transitions", type=int, default=100000, 
                        help="number of transitions on train")
    parser.add_argument("--buffer", type=int, default=200000, help="buffer size")
    parser.add_argument("--buffer-init", action="store_true", 
                        help="fill up buffer with uniformly random action")
    parser.add_argument("--batch", type=int, default=512, help="batch size")
    parser.add_argument("--env", type=str, default="oleg", help="which env to take")
    parser.add_argument("--env-config", type=str, default="", help="specify env config to")
    parser.add_argument("--saverate", type=int, default=-1, help="how often to evaluate and save model")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=0.3)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("--update-rate", type=int, default=1, help="how often to update model")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("-aud", "--actor-update-delay", type=int, default=50)
    parser.add_argument("--actor-reg", type=float, default=0)
    parser.add_argument("--shared-actor", action="store_true")
    parser.add_argument("--shared-critic", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--num-updates", type=int, default=1, help="how many updates of model at every step")
    parser.add_argument("-tp", "--time-penalty", action="store_true", help="enable time penalty for agents")
    parser.add_argument("-ndr", "--no-distance-reward", action="store_true")
    parser.add_argument("-pred", "--pred-baseline", action="store_true", help="enable predator baseline")
    parser.add_argument("-prey", "--prey-baseline", action="store_true", help="enable prey baseline")
    parser.add_argument("--temperature", type=float, default=30,
                        help="temperature in tanh input of actor network")
    parser.add_argument("--hidden-size", type=int, default=64, help="hidden size in agent network")
    parser.add_argument("--cuda", action="store_true", help="enable cuda")
    parser.add_argument("--info", action="store_true", help="print out env config at the start")
    parser.add_argument("--verbose", action="store_true", help="print out some debug info")
    parser.add_argument("--omp", type=int, default=-1)

    opts = parser.parse_args()

    args = vars(opts)
    print("\n--- loaded options ---")
    for name, value in args.items():
        print(f"{name}: {value}")
    print()
    
    if opts.omp > 0:
        torch.set_num_threads(opts.omp)

    shared_actor = False
    shared_critic = False
    if opts.shared:
        shared_actor = True
        shared_critic = True

    if opts.shared_actor:
        shared_actor = True
    if opts.shared_critic:
        shared_critic = True

    if opts.cuda:
        device = "cuda"
    else:
        device = "cpu"

    if opts.env.lower() == "oleg":
        from oleg_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
    elif opts.env.lower() == "kirill":
        from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
    
    if opts.env_config:
        with open(opts.env_config, "r") as f:
            env_config = json.load(f)
    else:
        env_config = DEFAULT_CONFIG
    
    title = f"#preds: {env_config['game']['num_preds']} (baseline: {opts.pred_baseline})" \
            f", #preys: {env_config['game']['num_preys']} (baseline: {opts.prey_baseline})" \
            f", #obsts: {env_config['game']['num_obsts']}"
    
    train(title=title,
            transitions=opts.transitions, 
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
            device=device,
            verbose=opts.verbose,
            buffer_init=opts.buffer_init,
            time_penalty=opts.time_penalty,
            pred_baseline=opts.pred_baseline,
            prey_baseline=opts.prey_baseline,
            actor_update_delay=opts.actor_update_delay,
            shared_actor=shared_actor,
            shared_critic=shared_critic,
            actor_reg=opts.actor_reg,
            distance_reward=not opts.no_distance_reward)

# OMP_NUM_THREADS=1 ./train.py  -t 10000000 --buffer 2500000 --batch 2048 --env-config configs/1v1_1.json --seed 42 --sigma-max 0.3 --sigma-min 0.1 --saverate 20000
