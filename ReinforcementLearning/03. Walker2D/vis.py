import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random
from tqdm import tqdm
import pandas as pd
import uuid
from train import Actor
from gym import wrappers

ENV_NAME = "Walker2DBulletEnv-v0"

class Agent:
    def __init__(self, name="agent.pkl"):
        # self.model = torch.load(__file__[:-8] + f"{name}", map_location="cpu")
        
        self.model = Actor(22, 6).to("cpu")
        self.model.load_state_dict(torch.load(__file__[:-8] + name))
        self.model.eval()
        
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).unsqueeze(0).float()
            return self.model.act(state)[0].flatten().numpy()

    def reset(self):
        pass



def evaluate_policy(agent, episodes=5):
    env = make(ENV_NAME)
    
    # env = wrappers.Monitor(env, 'vidos_' + str(uuid.uuid4()), force=True)
    
    for _ in range(episodes):
        done = False
        env.render()
        state = env.reset()
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            env.camera_adjust()
    env.close()


if __name__ == "__main__":
    agent = Agent(name="agent1.pkl")
    evaluate_policy(agent)
    