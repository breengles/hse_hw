from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from wrapper import VectorizeWrapper
from examples.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
import torch
from tqdm import trange
from better_baseline import PreyAgent, PredatorAgent
from joblib import delayed, Parallel


def vectorize_state(state_dicts):
    def _state_to_array(state_dicts_):
        states = []
        for state_dict in state_dicts_:
            states.extend(list(state_dict.values()))
        return states
    
    return [*_state_to_array(state_dicts["predators"]), *_state_to_array(state_dicts["preys"]), *_state_to_array(state_dicts["obstacles"])]


# agent_pred = ChasingPredatorAgent()
# agent_prey = FleeingPreyAgent()

agent_pred = PredatorAgent()
agent_prey = PreyAgent()

env = PredatorsAndPreysEnv()

ds_size = 100_000_000
saverate = 10_000_000
skip = 5

states = np.zeros(shape=(ds_size // skip, 33))
actions = np.zeros(shape=(ds_size // skip, 7))

def get_dataset():
    done = False
    for idx in trange(ds_size):
        state_dict = env.reset()
        
        if done:
            state_dict = env.reset()
            done = False
        
        action = np.hstack([agent.act(state_dict) for agent in (agent_pred, agent_prey)])
        state = vectorize_state(state_dict)
        if idx % skip == 0:
            states[idx // skip] = state[:-10 * 3]
            actions[idx // skip] = action
        state_dict = env.step(action[:2], action[2:])
        
        if (idx + 1) % saverate == 0:
            tensors = [torch.tensor(states[:idx // skip], dtype=torch.float), 
                    torch.tensor(actions[:idx // skip], dtype=torch.float)]
            ds = TensorDataset(*tensors)
            torch.save(ds, f"dataset/{(idx + 1) // skip}.pkl")


if __name__ == "__main__":
    get_dataset()
    