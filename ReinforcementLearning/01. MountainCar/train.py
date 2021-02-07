from gym import make
import numpy as np
import torch
import copy
from collections import deque
import random


GAMMA = 0.98
GRID_SIZE_X = 30
GRID_SIZE_Y = 30
SEED = 42


# Simple discretization
def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X - 1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y - 1)
    return x + GRID_SIZE_X * y


class Sarsa:
    def __init__(self, env, state_dim, action_dim, alpha=0.1):
        self.Q = np.zeros((state_dim, action_dim)) + 2.
        self.alpha = alpha
        self.env = env

    def update(self, transition, eps=0.1):
        state, action, next_state, reward, done = transition
        if done:
            self.Q[next_state] = 0
        a_ = self._act(next_state, eps)
        self.Q[state, action] += self.alpha * (reward + GAMMA * self.Q[next_state, a_] - self.Q[state, action])

    def _act(self, state, eps=0.1):
        if random.random() < eps:
            return self.env.action_space.sample()
        else:
            return self.act(state)
        
    def act(self, state):
        return np.argmax(self.Q[state])
    
    def save(self, name="agent.npy"):
        np.save(name, self.Q)

class QLearning:
    def __init__(self, state_dim, action_dim, alpha=0.1):
        self.Q = np.zeros((state_dim, action_dim)) + 2.
        self.alpha = alpha

    def update(self, transition):
        state, action, next_state, reward, done = transition
        if done:
            self.Q[next_state] = 0
        self.Q[state, action] += self.alpha * (reward + GAMMA * np.max(self.Q[next_state]) - self.Q[state, action])

    def act(self, state, eps=0.1):
        return np.argmax(self.Q[state])

    def save(self, name="agent.npy"):
        np.save(name, self.Q)


def rollout(env, agent, to_render=False):
    done = False
    state = env.reset()
    total_reward = 0.

    while not done:
        state, reward, done, _ = env.step(agent.act(transform_state(state)))
        
        if to_render:
            env.render()
        total_reward += reward
    
    # if to_render:
    env.close()
    return total_reward


def evaluate_policy(agent, episodes=5, to_render=False):
    env = make("MountainCar-v0")
    
    env.seed(SEED)
    env.action_space.seed(SEED)

    returns = []
    for _ in range(episodes):
        total_reward = rollout(env, agent, to_render=to_render)
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("MountainCar-v0")
    ql = QLearning(state_dim=GRID_SIZE_X * GRID_SIZE_Y, action_dim=3, alpha=0.1)
    eps = 0.1
    transitions = 4000000
    trajectory = []
    
    env.seed(SEED)
    env.action_space.seed(SEED)
        
    state = transform_state(env.reset())
    for i in range(transitions):
        total_reward = 0
        steps = 0

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = ql.act(state, eps=eps)

        next_state, reward, done, _ = env.step(action)
        reward += abs(next_state[1]) / 0.07  # not ~
        done_ = next_state[0] > 0.5
        next_state = transform_state(next_state)

        trajectory.append((state, action, next_state, reward, done_))

        if done:
            for transition in reversed(trajectory):
                ql.update(transition)
            trajectory = []

        state = next_state if not done else transform_state(env.reset())

        if (i + 1) % (transitions // 100) == 0:
            rewards = evaluate_policy(ql, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            ql.save()
