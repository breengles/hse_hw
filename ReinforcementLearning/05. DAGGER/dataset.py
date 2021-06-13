from predators_and_preys_env.env import PredatorsAndPreysEnv
import numpy as np
from torch.utils.data import TensorDataset
import torch
from tqdm import trange
from better_baseline import PreyAgent, PredatorAgent


def vectorize_state(state_dicts):
    def _state_to_array(state_dicts_):
        states = []
        for state_dict in state_dicts_:
            states.extend(list(state_dict.values()))
        return states
    
    return [*_state_to_array(state_dicts["predators"]), *_state_to_array(state_dicts["preys"]), *_state_to_array(state_dicts["obstacles"])]


def get_dataset(transitions=100_000_000, saverate=-1, skip=5):
    agent_pred = PredatorAgent()
    agent_prey = PreyAgent()
    env = PredatorsAndPreysEnv()
    
    saverate = saverate if saverate > 0 else transitions // 10
    
    states = np.zeros(shape=(transitions, 33))
    actions = np.zeros(shape=(transitions, 7))
    
    cur_size = 0
    
    done = False
    state_dict = env.reset()
    for idx in trange(transitions):
        if done:
            state_dict = env.reset()
            done = False
            
        action = np.hstack([agent.act(state_dict) for agent in (agent_pred, agent_prey)])
        next_state_dict, _, done = env.step(action[:2], action[2:])
        
        if idx % skip == 0 or done:
            state = vectorize_state(state_dict)
            states[cur_size] = state[:-10 * 3]
            actions[cur_size] = action
            cur_size += 1
        state_dict = next_state_dict
        
        if (idx + 1) % saverate == 0:
            tensors = [torch.FloatTensor(states[:cur_size]), 
                       torch.FloatTensor(actions[:cur_size])]
            ds = TensorDataset(*tensors)
            torch.save(ds, f"dataset/{cur_size}.pkl")


if __name__ == "__main__":
    get_dataset()
    