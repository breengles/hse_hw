from copy import deepcopy
from itertools import chain


def vectorize_state(state_dicts):
    def _state_to_array(state_dicts_):
        states = []
        for state_dict in state_dicts_:
            states.extend(list(state_dict.values()))
        return states
    
    return [*_state_to_array(state_dicts["predators"]), *_state_to_array(state_dicts["preys"]), *_state_to_array(state_dicts["obstacles"])]


def death_masking(state_dicts):
    state_dicts = deepcopy(state_dicts)
    for i, prey in enumerate(state_dicts["preys"]):
        if not prey["is_alive"]:
            prey["x_pos"] = 0
            prey["y_pos"] = 0
            prey["radius"] = 0
            prey["speed"] = 0
            prey["is_alive"] = i + 2
    # state_dicts["preys"] = sorted(state_dicts["preys"], key=lambda d: d["is_alive"])
    return state_dicts


def relative_agents_states(state_dicts):
    new_agents_states = []
    for i, agent in enumerate(chain(state_dicts["predators"], state_dicts["preys"])):
        new_agent_state = list(agent.values())
            
        for j, other_agent in enumerate(chain(state_dicts["predators"], state_dicts["preys"], state_dicts["obstacles"])):
            if i == j:
                continue
            # passing dead prey
            if "is_alive" in other_agent and other_agent["is_alive"] != 1:
                new_agent_state.extend(list(other_agent.values()))
                continue
            
            new_other_agent_state = list(other_agent.values())
            
            # x/y pos relative to the agent
            new_other_agent_state[0] = new_other_agent_state[0] - new_agent_state[0]
            new_other_agent_state[1] = new_other_agent_state[1] - new_agent_state[1]
            
            new_agent_state.extend(new_other_agent_state)
        
        new_agents_states.append(new_agent_state)
    
    return new_agents_states


def grad_clamp(model):
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)


