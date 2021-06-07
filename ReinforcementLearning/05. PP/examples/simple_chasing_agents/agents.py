import numpy as np
from predators_and_preys_env.agent import PredatorAgent, PreyAgent


def distance(first, second):
    return ((first["x_pos"] - second["x_pos"])**2 + (first["y_pos"] - second["y_pos"])**2)**0.5


class ChasingPredatorAgent(PredatorAgent):
    def __init__(self, *args, **kwargs):
        self.kind = "baseline"
        
    def act(self, state_dict):
        action = []
        for predator in state_dict["predators"]:
            closest_prey = None
            for prey in state_dict["preys"]:
                if not prey["is_alive"]:
                    continue
                if closest_prey is None:
                    closest_prey = prey
                else:
                    if distance(closest_prey, predator) > distance(prey, predator):
                        closest_prey = prey
            if closest_prey is None:
                action.append(0.)
            else:
                action.append(np.arctan2(closest_prey["y_pos"] - predator["y_pos"],
                                         closest_prey["x_pos"] - predator["x_pos"]) / np.pi)
        return np.array(action)
    

class FleeingPreyAgent(PreyAgent):
    def __init__(self, *args, **kwargs):
        self.kind = "baseline"
        
    def act(self, state_dict):
        action = []
        for prey in state_dict["preys"]:
            closest_predator = None
            for predator in state_dict["predators"]:
                if closest_predator is None:
                    closest_predator = predator
                else:
                    if distance(closest_predator, prey) > distance(prey, predator):
                        closest_predator = predator
            if closest_predator is None:
                action.append(0.)
            else:
                action.append(1 + np.arctan2(closest_predator["y_pos"] - prey["y_pos"],
                                             closest_predator["x_pos"] - prey["x_pos"]) / np.pi)
        return np.array(action)
    
    
# class ChasingPredatorAgent(PredatorAgent):
#     def __init__(self, n_preds, n_preys, pred_state_dim=4, prey_state_dim=5):
#         self.n_preds = n_preds
#         self.n_preys = n_preys
        
#         self.kind = "baseline"
#         self.pred_state_dim = pred_state_dim
#         self.prey_state_dim = prey_state_dim
    
#     def get_pred_coord(self, global_state, idx):
#         return np.array((global_state[idx * self.pred_state_dim], 
#                          global_state[idx * self.pred_state_dim + 1]))
    
#     def get_prey_coord(self, global_state, idx):
#         return np.array((global_state[self.n_preds * self.pred_state_dim 
#                                       + idx * self.prey_state_dim], 
#                          global_state[self.n_preds * self.pred_state_dim 
#                                       + idx * self.prey_state_dim + 1]))
        
#     def get_prey_status(self, global_state, idx):
#         return global_state[self.n_preds * self.pred_state_dim 
#                             + idx * self.prey_state_dim + self.prey_state_dim] != 1

#     def dist(self, global_state, pred_idx, prey_idx):
#         pred_coord = self.get_pred_coord(global_state, pred_idx)
#         prey_coord = self.get_prey_status(global_state, prey_idx)
#         return np.linalg.norm(pred_coord - prey_coord)
        
#     def act(self, global_state):
#         action = []
#         for i in range(self.n_preds):
#             closest_prey = None
#             for j in range(self.n_preys):
#                 if self.get_prey_status(global_state, j):
#                     continue
#                 if closest_prey is None:
#                     closest_prey = j
#                 else:
#                     if self.dist(global_state, i, j) < self.dist(global_state, i, closest_prey):
#                         closest_prey = j
#             if closest_prey is None:
#                 action.append(0.0)
#             else:
#                 pred_coord = self.get_pred_coord(global_state, i)
#                 prey_coord = self.get_pred_coord(global_state, closest_prey)
#                 a = np.arctan2(prey_coord[1] - pred_coord[1], prey_coord[0] - pred_coord[0]) / np.pi
#                 action.append(a)

#         return np.array(action)


# class FleeingPreyAgent(PreyAgent):
#     def __init__(self, n_preds, n_preys, pred_state_dim=4, prey_state_dim=5):
#         self.n_preds = n_preds
#         self.n_preys = n_preys
        
#         self.kind = "baseline"
#         self.pred_state_dim = pred_state_dim
#         self.prey_state_dim = prey_state_dim
        
#     def get_pred_coord(self, global_state, idx):
#         return np.array((global_state[idx * self.pred_state_dim], 
#                          global_state[idx * self.pred_state_dim + 1]))
    
#     def get_prey_coord(self, global_state, idx):
#         return np.array((global_state[self.n_preds * self.pred_state_dim 
#                                       + idx * self.prey_state_dim], 
#                          global_state[self.n_preds * self.pred_state_dim 
#                                       + idx * self.prey_state_dim + 1]))
        
#     def get_prey_status(self, global_state, idx):
#         return global_state[self.n_preds * self.pred_state_dim 
#                             + idx * self.prey_state_dim + self.prey_state_dim] != 1

#     def dist(self, global_state, pred_idx, prey_idx):
#         pred_coord = self.get_pred_coord(global_state, pred_idx)
#         prey_coord = self.get_prey_status(global_state, prey_idx)
#         return np.linalg.norm(pred_coord - prey_coord)
        
#     def act(self, global_state):
#         action = []
#         for i in range(self.n_preys):
#             closest_pred = None
#             for j in range(self.n_preds):
#                 if self.get_prey_status(global_state, j):
#                     continue
#                 if closest_prey is None:
#                     closest_prey = j
#                 else:
#                     if self.dist(global_state, i, j) < self.dist(global_state, i, closest_prey):
#                         closest_prey = j
#             if closest_prey is None:
#                 action.append(0.0)
#             else:
#                 pred_coord = self.get_pred_coord(global_state, i)
#                 prey_coord = self.get_pred_coord(global_state, closest_prey)
#                 a = np.arctan2(prey_coord[1] - pred_coord[1], prey_coord[0] - pred_coord[0]) / np.pi
#                 action.append(a)

#         return np.array(action)
