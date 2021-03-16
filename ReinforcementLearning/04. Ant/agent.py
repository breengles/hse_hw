import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.device = "cpu"
        self.model = torch.load(__file__[:-8] + "/agent.model.pkl", map_location=self.device)
        self.model.eval()
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float, device=self.device)
            return self.model(state).numpy()

    def reset(self):
        pass
