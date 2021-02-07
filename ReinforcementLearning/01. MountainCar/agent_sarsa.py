import random
import numpy as np
import os
from .train import transform_state


class Agent:
    def __init__(self):
        self.qlearning_estimate = np.load(__file__[:-8] + "/sarsa.npy")
        
    def act(self, state):
        state = transform_state(state)
        if random.random() < 0.1:
            return random.choice([0, 1, 2])
        else:
            return np.argmax(self.qlearning_estimate[state])
            
    def reset(self):
        pass

