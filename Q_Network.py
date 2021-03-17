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


NUMBER_OF_EPISODES = 500
MAX_ITERATIONS = 500  ## Max Time Steps Per Game (Limited to 500 by Environment)

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000



class DQNSolver:

    def __init__(self):
        self.exploration_rate = EXPLORATION_MAX
        self.gamma = 0.9
        self.q_model = self.model()
        self.target_model = self.model()
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.update_target_network()

        self.update_counter = 0


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

        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(4)
        state = state.reshape(-1, 210, 160, 3)
        q_values = self.q_model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):


        if len(self.memory) < BATCH_SIZE:
            return


        batch = random.sample(self.memory, BATCH_SIZE)



        state_batch, q_values_batch = [], []

        for state, action, reward, next_state in batch:
            largest_loss = 0
            index_value_of_largest_lost = 0
            state = state.reshape(-1, 210, 160, 3)
            next_state = next_state.reshape(-1, 210, 160, 3)
            q_values = self.q_model.predict(state)

            target_q_values = self.target_model.predict(next_state)
            q_values[0][action] = reward + self.gamma * np.amax(target_q_values)
            error = q_values - target_q_values
            loss = tf.reduce_min(tf.square(error))

            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

            '''if self.memory_counter < self.SIZE_OF_PRIORITY_MEMORY:
                self.least_error_memory.append((state, action, reward, next_state, loss))
                self.priority_memory.append((state, action, reward, next_state))
                self.memory_counter += 1
            else:
                for i in range(self.memory_counter):
                    if self.least_error_memory[i][4] > largest_loss:
                        largest_loss = self.least_error_memory[i][4]
                        index_value_of_largest_lost = i
            if largest_loss > loss:
                self.least_error_memory.pop(index_value_of_largest_lost)
                self.least_error_memory.append((state, action, reward, next_state, loss))
                self.priority_memory.pop(index_value_of_largest_lost)
                self.priority_memory.append((state, action, reward, next_state))'''

        self.q_model.fit(np.array(state_batch), np.array(q_values_batch),
                         verbose=0,
                         )
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        if self.update_counter % 100 == 0:
            self.update_target_network()

        self.update_counter += 1

    def update_target_network(self):

        self.target_model.set_weights(self.q_model.get_weights())

