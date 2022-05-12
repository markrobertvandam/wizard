import numpy as np

from collections import deque
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import adam_v2

import Playing_Agent
import random

REPLAY_MEMORY_SIZE = 42000  # How many of last   to keep for model training, 42000 means remember last ~200 games
MIN_REPLAY_MEMORY_SIZE = 4200  # Minimum number of tricks in memory to start training, 10500 means at least ~20 games
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training


# Agent class
class PlayingNetwork:
    def __init__(self, input_size):

        self.input_size = input_size

        # Main model
        self.model = self.create_model()

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    @staticmethod
    def dense_layer(num_units):
        return Dense(num_units, activation="relu")

    def create_model(self):

        input1 = Input(self.input_size)  # input size for observation state is 3732
        # input2 = Input(1)  # scalar input for action taken
        #
        # combined = concatenate([input1, input2])
        x = self.dense_layer(2048)(input1)
        # x = self.dense_layer(1024)(input1)
        x = self.dense_layer(512)(x)
        x = self.dense_layer(256)(x)
        x = self.dense_layer(128)(x)
        x = self.dense_layer(64)(x)
        x = self.dense_layer(32)(x)
        x = Dense(1, activation="linear")(x)  # evaluation

        model = Model(inputs=input1, outputs=x)

        model.compile(
            loss="mse",
            optimizer=adam_v2.Adam(learning_rate=0.0001),
            metrics=["accuracy"],
        )
        #print(model.summary())
        return model

    # Adds data to a memory replay array
    # (state, reward)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self):
        # print("Replay size: ", len(self.replay_memory))
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get states (x) and rewards (y) from minibatch
        states = np.array(
            [
                Playing_Agent.PlayingAgent.key_to_state(self.input_size, transition[0])
                for transition in minibatch
            ]
        )
        rewards = np.array([transition[1] for transition in minibatch])

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            states,
            rewards,
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
        )

    def predict(self, state):
        return self.model.predict(state.reshape(-1, *state.shape))[0]
