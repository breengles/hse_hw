from copy import deepcopy
from itertools import chain


class VectorizeWrapper:
    def __init__(self, env, return_state_dict=False):
        self.env = env
        self.return_state_dict = return_state_dict # TODO: for baseline agents
        
        self.predator_action_size = env.predator_action_size
        self.prey_action_size = env.prey_action_size
        
    @staticmethod
    def _vectorize_state(state_dicts):
        def _state_to_array(state_dicts_):
            states = []
            for state_dict in state_dicts_:
                states.extend(list(state_dict.values()))
            return states
        
        return [*_state_to_array(state_dicts["predators"]), *_state_to_array(state_dicts["preys"]), *_state_to_array(state_dicts["obstacles"])]
    
    @staticmethod
    def _vectorize_reward(reward_dicts):
        return list(reward_dicts["predators"]), list(reward_dicts["preys"])
            
    def _relative_agents_states(self, state_dicts):
        new_agents_states = []
        for i, agent in enumerate(chain(state_dicts["predators"], state_dicts["preys"])):
            new_agent_state = list(agent.values())
                
            for j, other_agent in enumerate(chain(state_dicts["predators"], state_dicts["preys"], state_dicts["obstacles"])):
                if i == j:
                    continue
                
                new_other_agent_state = list(other_agent.values())
                
                # x/y pos relative to the agent
                new_other_agent_state[0] = new_other_agent_state[0] - new_agent_state[0]
                new_other_agent_state[1] = new_other_agent_state[1] - new_agent_state[1]
                
                new_agent_state.extend(new_other_agent_state)
            
            new_agents_states.append(new_agent_state)
        
        #        preds                                            preys
        return new_agents_states[:self.predator_action_size], new_agents_states[-self.prey_action_size:]
    
    def step(self, pred_actions, prey_actions):
        next_state_dict, reward, done = self.env.step(pred_actions, prey_actions)
        global_state = self._vectorize_state(next_state_dict)
        rel_preds_states, rel_preys_states = self._relative_agents_states(next_state_dict)
        r_pred, r_prey = self._vectorize_reward(reward)
        return global_state, rel_preds_states, rel_preys_states, r_pred, r_prey, done
        
    def reset(self):
        state_dict = self.env.reset()
        global_state = self._vectorize_state(state_dict)
        rel_preds_states, rel_preys_states = self._relative_agents_states(state_dict)
        return global_state, rel_preds_states, rel_preys_states
        
    def seed(self, seed):
        self.env.seed(seed)
        