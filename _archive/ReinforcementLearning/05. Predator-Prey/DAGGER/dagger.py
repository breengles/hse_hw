import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from model import Net, SinglePred, SinglePrey
from better_baseline import PreyAgent, PredatorAgent
from utils import vectorize_state
from torch.utils.data import TensorDataset
from predators_and_preys_env.env import PredatorsAndPreysEnv
import numpy as np
from copy import deepcopy


def sample_trajectories(agent_pred, agent_prey, size, length, device="cpu"):
    ds_state = []
    ds_dict = []
    for _ in range(size):
        traj_state, traj_dict = get_trajectory(agent_pred, agent_prey, length, device=device)
        ds_state.extend(traj_state.cpu().numpy())
        ds_dict.extend(traj_dict.cpu().numpy())
    return ds_state, ds_dict


def get_trajectory(agent_pred, agent_prey, length, device="cpu"):
    env = PredatorsAndPreysEnv()
    traj_dict = []
    traj_state = []
    
    preds = list(range(0, 2 * 4))
    
    state_dict = env.reset()
    done = False
    for i in range(length):
        if done:
            break
        state = np.array(vectorize_state(state_dict)[:-10 * 3])
        
        with torch.no_grad():
            actions = []
            for i in range(2):
                idxes = list(range(i * 4, (i + 1) * 4)) \
                    + list(range(2 * 4, 2 * 4 + 5 * 5))
                s = torch.FloatTensor([state[idxes]])
                actions.append(agent_pred(s).squeeze(-1))
                
            for i in range(5):
                prey = list(range(2 * 4 + i * 5, 2 * 4 + (i + 1) * 5))
                idxes = preds + prey
                s = torch.FloatTensor([state[idxes]])
                actions.append(agent_prey(s).squeeze(-1))
        
        next_state_dict, _, done = env.step(actions[:2], actions[2:])
        
        traj_state.append(deepcopy(state))
        traj_dict.append(deepcopy(state_dict))
        
        state_dict = next_state_dict
    return traj_state, traj_dict


def get_from_experts(pred, prey, state_dict):
    return np.hstack([agent.act(state_dict) for agent in (pred, prey)])


def train(num_epoch=1000, batch_size=256, saverate=10, device="cpu", path="",
          num_trajectories=16, length_trajectory=500):
    env = PredatorsAndPreysEnv()
    pred_expert = PredatorAgent()
    prey_expert = PreyAgent()
    
    # net = Net()
    # if path:
    #     net.load_state_dict(torch.load(path, map_location=device))
    #     net.train()
    # net.to(device)
    # optim = torch.optim.Adam(net.parameters(), lr=1e-2)
    
    net_pred = SinglePred()
    net_prey = SinglePrey()    
    
    net_pred.to(device)
    net_prey.to(device)
    optim_pred = torch.optim.Adam(net_pred.parameters(), lr=3e-4)
    optim_prey = torch.optim.Adam(net_prey.parameters(), lr=3e-4)
    
    # init dataset
    states = []
    actions = []
    state_dict = env.reset()
    done = False
    for _ in range(batch_size):
        if done:
            state_dict = env.reset()
            done = False
            
        action = np.hstack([agent.act(state_dict) for agent in (pred_expert, prey_expert)])
        next_state_dict, _, done = env.step(action[:2], action[2:])
        
        state = vectorize_state(state_dict)
        states.append(state[:-10 * 3])  # get rid off obstacles' features
        actions.append(action)
        
        state_dict = next_state_dict

    states = torch.FloatTensor(states, device="cpu")
    actions = torch.FloatTensor(actions, device="cpu")

    for t in trange(num_epoch):
        ds = TensorDataset(states, actions)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1)
        for X, y in dl:
            X_ = X.to(device)
            y_ = y.to(device)
            
            for i in range(2):
                idxes = list(range(i * 4, (i + 1) * 4)) \
                    + list(range(2 * 4, 2 * 4 + 5 * 5))
                X_agent = X_[:, idxes]
                y_pred = net_pred(X_agent).squeeze(-1)
                
                assert y_pred.shape == y_[:, i].shape
                
                loss_pred = F.mse_loss(y_pred, y_[:, i])
                optim_pred.zero_grad()
                loss_pred.backward()
                optim_pred.step()
                
            for i in range(5):
                preds = list(range(0, 2 * 4))
                prey = list(range(2 * 4 + i * 5, 2 * 4 + (i + 1) * 5))
                idxes = preds + prey
                X_agent = X_[:, idxes]
                y_prey = net_prey(X_agent).squeeze(-1)
                
                assert y_prey.shape == y_[:, 2 + i].shape
                
                loss_prey = F.mse_loss(y_prey, y_[:, 2 + i])
                optim_prey.zero_grad()
                loss_prey.backward()
                optim_prey.step()
            
            # y_hat = net(X_)
            # loss = F.mse_loss(y_hat, y_)
            # optim.zero_grad()
            # loss.backward()
            # optim.step()
            
        new_ds_state, new_ds_dict = sample_trajectories(net_pred, net_prey, 
                                                        num_trajectories, 
                                                        length_trajectory)
        new_actions = []
        for new_state_dict in new_ds_dict:
            new_actions.append(get_from_experts(pred_expert, prey_expert, new_state_dict))
            
        states = torch.cat((states, torch.FloatTensor(new_ds_state)))
        actions = torch.cat((actions, torch.FloatTensor(new_actions)))
        # states.extend(new_ds_state)
        # actions.extend(new_actions)
            
        if (t + 1) % saverate == 0:
            with torch.no_grad():
                tqdm.write(f"Epoch {t + 1:>5d} | Loss pred {loss_pred:0.5f} | Loss prey {loss_pred:0.5f}")
                torch.save(net_pred.state_dict(), f"dagger/pred_{t + 1}.pt")
                torch.save(net_prey.state_dict(), f"dagger/prey_{t + 1}.pt")
                

if __name__ == "__main__":
    # ds = torch.load("dataset/20195482.pkl")
    # model_path = "model.pt"
    train(device="cpu", batch_size=16, saverate=1)
