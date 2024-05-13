import numpy as np
import math

import gym

from src.Q_learing import Q_learning

# discreteSpace = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
# state = [-1.1, -1, 0.1, 2]

# print(makeDiscreteState(discreteSpace, state))#(0, 1, 2, 3)
def rewardFunc(observation):
    angle = abs(observation[2])
    position = abs(observation[0])
    reward = ((0.1045 - angle) / 0.1045 + (1.2 - position) / 1.2)
    return reward

def train(env, agent, episodes=1000):
    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25)))
    get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25)))
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        epsilon = get_epsilon(episode)
        lr = get_lr(episode)

        while not done:
    
            action = agent.step(state, Lambda=epsilon)
            next_state, reward, done, _, __ = env.step(action)
            reward = rewardFunc(next_state)
            
            agent.update(state, next_state, action, reward, lr=lr)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {episode}, Total Reward: {total_reward}")
                break
            
        

# Example usage
env = gym.make('CartPole-v0')
action_space = [0, 1]  # Example for CartPole: actions are 0 (left) and 1 (right)
observationSpace = (1, 1, 6, 3)  # Simplified and discretized space for the example
observationBound = [[0], [0], [-0.5, 0.5], [-math.radians(50), math.radians(50)]]

agent = Q_learning(observationSpace, observationBound, action_space)
train(env, agent)
print(agent.Q_form)