from predators_and_preys_env.env import PredatorsAndPreysEnv
import numpy as np
from torch.utils.data import TensorDataset
import torch
from tqdm import trange
from examples.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent
from better_baseline import PreyAgent, PredatorAgent
from utils import vectorize_state


def get_dataset(config=None, transitions=60_000_000, saverate=-1, skip=5, 
                delete_obsts=True):
    agent_pred = PredatorAgent()
    agent_prey = PreyAgent()
    # agent_pred = ChasingPredatorAgent()
    # agent_prey = FleeingPreyAgent()
    if config is None:
        env = PredatorsAndPreysEnv()
    else:
        env = PredatorsAndPreysEnv(config)
    
    saverate = saverate if saverate > 0 else transitions // 10
    
    states = np.zeros(shape=(transitions, 33)) if delete_obsts else np.zeros(shape=(transitions, 63))
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
            states[cur_size] = state[:-10 * 3] if delete_obsts else state
            actions[cur_size] = action
            cur_size += 1
        state_dict = next_state_dict
        
        if (idx + 1) % saverate == 0:
            tensors = [torch.FloatTensor(states[:cur_size]), 
                       torch.FloatTensor(actions[:cur_size])]
            ds = TensorDataset(*tensors)
            torch.save(ds, f"dataset/kirill_{cur_size}.pkl")


if __name__ == "__main__":
    for_kirill = {
        "game": {
            "num_obsts": 10,
            "num_preds": 2,
            "num_preys": 5,
            "x_limit": 9,
            "y_limit": 9,
            "obstacle_radius_bounds": [0.8, 1.5],
            "prey_radius": 0.8,
            "predator_radius": 1.0,
            "predator_speed": 6.0,
            "prey_speed": 9.0,
            "world_timestep": 1/40,
            "frameskip": 2
        },
        "environment": {
            "frameskip": 2,
            "time_limit": 300
        }
    }
    get_dataset(config=for_kirill, skip=3, delete_obsts=False)
    # get_dataset(skip=3, delete_obsts=False)
    