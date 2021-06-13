from copy import deepcopy
from itertools import chain
import numpy as np


class VectorizeWrapper:
    def __init__(self, env, pred_baseline=False):
        self.env = env
        self.pred_baseline = pred_baseline
        
        self.n_preds = env.predator_action_size
        self.n_preys = env.prey_action_size
        
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
        return list(reward_dicts["predators"]) + list(reward_dicts["preys"])
            
    def _relative_agents_states(self, state_dicts):
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
    
    def step(self, actions):
        pred_actions = actions[:self.n_preds]
        prey_actions = actions[-self.n_preys:]
        state_dict, reward, done = self.env.step(pred_actions, prey_actions)
        
        if not self.pred_baseline:
            state_dict = self._death_masking(state_dict)
        global_state = self._vectorize_state(state_dict)
        agent_states = self._relative_agents_states(state_dict)
        rewards = self._vectorize_reward(reward)
        
        state_dict_ = deepcopy(state_dict)
        return state_dict_, global_state, agent_states, rewards, done
        
        # preds_feature = self.generate_features_predator(state_dict)
        # preys_feature = self.generate_features_prey(state_dict)
        # return preds_feature, preds_feature, state_dict, reward, done
        
    def reset(self):
        state_dict = self.env.reset()
        state_dict_ = deepcopy(state_dict)
        global_state = self._vectorize_state(state_dict)
        agent_states = self._relative_agents_states(state_dict)
        
        return state_dict_, global_state, agent_states
        
        # preds_feature = self.generate_features_predator(state_dict)
        # preys_feature = self.generate_features_prey(state_dict)
        # return preds_feature, preys_feature, state_dict
        
    def seed(self, seed):
        self.env.seed(seed)
     
    @staticmethod
    def _death_masking(state_dicts):
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

    @staticmethod        
    def generate_features_predator(state_dict):
        features = []

        for predator in state_dict['predators']:
            x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], predator['y_pos'], predator['radius'], predator['speed']

            features += [x_pred, y_pred]

            prey_list = []

            for prey in state_dict['preys']:
                x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], \
                                                            prey['radius'], prey['speed'], prey['is_alive']
                angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
                distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

                prey_list += [[angle, distance, int(alive), r_prey]]

            prey_list = sorted(prey_list, key=lambda x: x[1])
            prey_list = [item for sublist in prey_list for item in sublist]
            features += prey_list

            obs_list = []

            for obs in state_dict['obstacles']:
                x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
                angle = np.arctan2(y_obs - y_pred, x_obs - x_pred) / np.pi
                distance = np.sqrt((y_obs - y_pred) ** 2 + (x_obs - x_pred) ** 2)

                obs_list += [[angle, distance, r_obs]]

            obs_list = sorted(obs_list, key=lambda x: x[1])
            obs_list = [item for sublist in obs_list for item in sublist]
            features += obs_list

        return np.array(features, dtype=np.float32)

    @staticmethod
    def generate_features_prey(state_dict):
        features = []

        for prey in state_dict['preys']:
            x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], \
                prey['y_pos'], prey['radius'], prey['speed'], prey['is_alive']

            features += [x_prey, y_prey, alive, r_prey]

            pred_list = []

            for predator in state_dict['predators']:
                x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], \
                    predator['y_pos'], predator['radius'], predator['speed']

                angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
                distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

                pred_list += [[angle, distance, int(alive), r_prey]]

            pred_list = sorted(pred_list, key=lambda x: x[1])
            pred_list = [item for sublist in pred_list for item in sublist]
            features += pred_list

            obs_list = []

            for obs in state_dict['obstacles']:
                x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
                angle = np.arctan2(y_obs - y_prey, x_obs - x_prey) / np.pi
                distance = np.sqrt((y_obs - y_prey) ** 2 + (x_obs - x_prey) ** 2)

                obs_list += [[angle, distance, r_obs]]

            obs_list = sorted(obs_list, key=lambda x: x[1])
            obs_list = [item for sublist in obs_list for item in sublist]
            features += obs_list

        return np.array(features, dtype=np.float32)
