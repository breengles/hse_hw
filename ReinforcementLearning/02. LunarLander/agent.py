import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(__file__[:-8] + "/agent.pkl").to(self.device)
        
    def act(self, state):
        with torch.no_grad():
            state_ = torch.tensor(state).to(self.device).float()
            return torch.argmax(self.model(state_)).cpu().numpy().item()

    def reset(self):
        pass

