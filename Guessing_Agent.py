import numpy as np

from collections import deque
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import adam_v2

import random

REPLAY_MEMORY_SIZE = 2000  # How many last steps to keep for model training, 2000 means remember last 100 games
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in memory to start training, 1000 means at least 50 games
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training


# Agent class
class GuessingAgent:
    def __init__(self, input_size, guess_max):

        self.input_size = input_size
        self.guess_max = guess_max
        self.accuracy = 0.02

        # Main model
        self.model = self.create_model()

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    @staticmethod
    def dense_layer(num_units):
        return Dense(num_units, activation="relu")

    def create_model(self):

        input1 = Input(self.input_size)  # input size for observation state is 68
        # input2 = Input(1)  # scalar input for action taken
        #
        # combined = concatenate([input1, input2])
        x = self.dense_layer(128)(input1)
        x = self.dense_layer(64)(x)
        x = self.dense_layer(32)(x)
        x = Dense(self.guess_max, activation="softmax")(x)
        # guess_max = how many rounds + 1 (output_size) (21)

        model = Model(inputs=input1, outputs=x)

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=adam_v2.Adam(learning_rate=0.0001),
            metrics=["accuracy"],
        )
        # print(model.summary())
        return model

    # Adds data to a memory replay array
    # (current state, tricks won)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        x = []
        y = []

        # Now we need to enumerate our batches
        for (current_state, trick_wins) in minibatch:
            # append to our training data
            x.append(current_state)
            y.append(trick_wins)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            np.array(x),
            np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
        )

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, *state.shape))[0]