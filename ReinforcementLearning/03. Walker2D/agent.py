import random
import numpy as np
import os
import torch
from torch import nn


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl", map_location="cpu")
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).unsqueeze(0).float()
            return self.model.act(state)[0].flatten().numpy()

    def reset(self):
        pass
