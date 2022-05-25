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
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
DISCOUNT = 0.9


# Agent class
class PlayingNetwork:
    def __init__(self, input_size, name):

        self.input_size = input_size
        self.name = name

        # Main model
        self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.target_update_counter = 0

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
        if self.input_size > 3000:
            if self.name == "small":
                x = self.dense_layer(1024)(input1)
            else:
                x = self.dense_layer(2048)(input1)
            if self.name == "large":
                x = self.dense_layer(1024)(x)
            x = self.dense_layer(512)(x)
            x = self.dense_layer(256)(x)
            x = self.dense_layer(128)(x)
        else:
            x = self.dense_layer(128)(input1)
        x = self.dense_layer(64)(x)
        x = Dense(60, activation="linear")(x)  # evaluation

        model = Model(inputs=input1, outputs=x)

        model.compile(
            loss="mse",
            optimizer=adam_v2.Adam(learning_rate=0.0001),
            metrics=["accuracy"],
        )
        # print(model.summary())
        return model

    # Adds data to a memory replay array
    # (state, reward)
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
        current_states = np.array([Playing_Agent.PlayingAgent.key_to_state(self.input_size, transition[0])
                                   for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([Playing_Agent.PlayingAgent.key_to_state(self.input_size, transition[3])
                                       for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        sparse_state = Playing_Agent.PlayingAgent.key_to_state(self.input_size, state)
        return self.model.predict(np.array(sparse_state).reshape(-1, *sparse_state.shape))[0]
