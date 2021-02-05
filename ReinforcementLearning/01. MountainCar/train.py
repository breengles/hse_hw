from gym import make
import numpy as np
import torch
import copy
from collections import deque
import random

GAMMA = 0.98
GRID_SIZE_X = 30
GRID_SIZE_Y = 30


# Simple discretization 
def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X-1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y-1)
    return x + GRID_SIZE_X*y


class QLearning:
    def __init__(self, state_dim, action_dim):
        self.Q = np.zeros((state_dim, action_dim)) + 2.

    def update(self, transition):
        state, action, next_state, reward, done = transition
        pass

    def act(self, state):
        return 0

    def save(self):
        np.save("agent.npz", self.Q)


def evaluate_policy(agent, episodes=5):
    env = make("MountainCar-v0")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(transform_state(state)))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("MountainCar-v0")
    ql = QLearning(state_dim=GRID_SIZE_X*GRID_SIZE_Y, action_dim=3)
    eps = 0.1
    transitions = 4000000
    trajectory = []
    state = transform_state(env.reset())
    for i in range(transitions):
        total_reward = 0
        steps = 0
        
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = ql.act(state)

        next_state, reward, done, _ = env.step(action)
        reward += abs(next_state[1]) / 0.07
        next_state = transform_state(next_state)

        trajectory.append((state, action, next_state, reward, done))
        
        if done:
            for transition in reversed(trajectory):
                ql.update(transition)
            trajectory = []
        
        state = next_state if not done else transform_state(env.reset())
        
        if (i + 1) % (transitions//100) == 0:
            rewards = evaluate_policy(ql, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            ql.save()
