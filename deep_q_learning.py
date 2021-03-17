import gym
import atari_py
import keras.backend as K
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Lambda, Input, Layer, Dense
from keras.optimizers import Adam
from tensorflow.keras.models import Model


NUMBER_OF_EPISODES = 500
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self):
        self.exploration_rate = EXPLORATION_MAX
        self.gamma = 0.9
        self.q_model = self.model()
        self.target_model = self.model()

        self.update_target_network()


    def model(self):
        input_shape =(210, 160, 3)
        model = Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=[4, 4], input_shape=input_shape, activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid"))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=[2, 2], activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid"))
        model.add(tf.keras.layers.Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(4, activation="linear"))





if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    DQN = DQNSolver.model();
    env.reset()
    for _ in range(1000):
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())  # take a random action

    env.close()