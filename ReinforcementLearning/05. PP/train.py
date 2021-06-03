import os

from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from agent import Agent
from utils import ReplayBuffer

DEVICE = torch.device("cuda")
HIDDEN_SIZE = 64
N_HIDDEN_LAYERS = 3
BUFFER_SIZE = 10000
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
TAU = 0.005
GAMMA = 0.998
TOTAL_TRANSITIONS = int(2e5)
SIGMA_MAX = 0.2
SIGMA_MIN = 0.01
BATCH_SIZE = 512
IS_RENDERING = False

SEED = 65537
rs = RandomState(MT19937(SeedSequence(SEED)))
torch.manual_seed(SEED)


def add_noise(action, sigma, lower=-1, upper=1):
    return np.clip(action + np.random.normal(scale=sigma, size=action.shape), lower, upper)


def evaluate_policy(env, predator_agent, prey_agent, n_evals=10):
    preys_reward, predators_reward = [], []
    for _ in range(n_evals):
        r_prey, r_predator = 0, 0
        done = False
        state = env.reset()
        while not done:
            a_prey = prey_agent.act(state)
            a_predator = predator_agent.act(state)

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


def train():
    env = PredatorsAndPreysEnv(render=IS_RENDERING)
    n_predators, n_preys, n_obstacles = (
        env.predator_action_size,
        env.prey_action_size,
        DEFAULT_CONFIG["game"]["num_obsts"],
    )
    predator_buffer = ReplayBuffer(BUFFER_SIZE)
    prey_buffer = ReplayBuffer(BUFFER_SIZE)

    predator_agent = Agent(
        n_agents=n_predators,
        buffer=predator_buffer,
        state_dim=n_predators * 4 + n_preys * 5 + n_obstacles * 3,
        action_dim=n_predators,
        hidden_size=HIDDEN_SIZE,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        tau=TAU,
        gamma=GAMMA,
        device=DEVICE
    )
    prey_agent = Agent(
        n_agents=n_preys,
        buffer=prey_buffer,
        state_dim=n_predators * 4 + n_preys * 5 + n_obstacles * 3,
        action_dim=n_preys,
        hidden_size=HIDDEN_SIZE,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        tau=TAU,
        gamma=GAMMA,
        device=DEVICE
    )

    print("Filling up buffer...")
    state = env.reset()
    for _ in tqdm(range(BUFFER_SIZE)):
        a_prey = np.random.uniform(-1, 1, n_preys)
        a_predator = np.random.uniform(-1, 1, n_predators)

        next_state, reward, done = env.step(a_predator, a_prey)
        r_prey = reward["preys"]
        r_predator = reward["predator"]

        prey_buffer.add((state, a_prey, next_state, r_prey, done))
        predator_buffer.add((state, a_predator, next_state, r_predator, done))

        state = env.reset() if done else next_state
        
    print("Finished. Now training...")
    # done = True
    state = env.reset()
    for i in tqdm(range(TOTAL_TRANSITIONS)):
        sigma = SIGMA_MAX - (SIGMA_MAX - SIGMA_MIN) * i / TOTAL_TRANSITIONS
        
        a_prey = prey_agent.act(state)
        a_prey = add_noise(a_prey, sigma)
        
        a_predator = predator_agent.act(state)
        a_predator = add_noise(a_predator, sigma)

        next_state, reward, done = env.step(a_predator, a_prey)
        r_prey = reward["prey"]
        r_predator = reward["predator"]

        prey_agent.buffer.add((state, a_prey, next_state, r_prey, done))
        predator_agent.buffer.add((state, a_predator, next_state, r_predator, done))

        predator_agent.update(BATCH_SIZE)
        prey_agent.update(BATCH_SIZE)

        state = next_state if not done else env.reset()

        if (i + 1) % (TOTAL_TRANSITIONS // 100) == 0:  # evaluate every 1%
            rewards = evaluate_policy(env, predator_agent, prey_agent)

            predator_mean = np.mean(rewards["predators"])
            predator_std = np.std(rewards["predators"])

            prey_mean = np.mean(rewards["preys"])
            prey_std = np.std(rewards["preys"])

            print(f"Step: {i + 1}")
            print(f"Predator: mean = {predator_mean} | std = {predator_std}")
            print(f"Prey:     mean = {prey_mean} | std = {prey_std}")

            predator_agent.save_models("predator")
            prey_agent.save_models("prey")
            

if __name__ == "__main__":
    train()
