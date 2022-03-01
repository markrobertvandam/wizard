import numpy as np

from collections import deque
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import adam_v2

import random

REPLAY_MEMORY_SIZE = 200  # How many last steps to keep for model training, 200 means remember last 10 games
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training, 100 means at least 5 games played
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training


# Agent class
class GuessingAgent:
    def __init__(self, input_size, guess_max):

        self.input_size = input_size
        self.guess_max = guess_max
        self.avg_reward = 0  # stores avg reward every game

        # Main model
        self.model = self.create_model()

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def create_model(self):

        model = Sequential()
        model.add(Input(self.input_size))  # input size is 68
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(50))
        model.add(
            Dense(self.guess_max, activation="linear")
        )  # guess_max = how many rounds (output_size) (20)
        model.compile(
            loss="mse",
            optimizer=adam_v2.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        print(model.summary())
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward) in enumerate(minibatch):

            new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            np.array(X),
            np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
        )

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, *state.shape))[0]
