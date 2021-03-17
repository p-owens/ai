
import gym
import random
import atari_py
import keras.backend as K
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Lambda, Input, Layer, Dense
from keras.optimizers import Adam
from tensorflow.keras.models import Model
import Q_Network as network


NUMBER_OF_EPISODES = 500
MAX_ITERATIONS = 500  ## Max Time Steps Per Game (Limited to 500 by Environment)

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
BATCH_SIZE = 32







if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    DQN = network.DQNSolver()
    env.reset()
    for episode in range(NUMBER_OF_EPISODES):
        observation = env.reset()

        for iteration in range(MAX_ITERATIONS):

            action = DQN.act(observation)

            next_observation, reward, done, info = env.step(action)  # take a random action

            DQN.remember(observation,action,reward, next_observation)

            DQN.experience_replay()

            observation = next_observation

    env.close()