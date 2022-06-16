import math
import numpy as np

from collections import deque

from keras.metrics import mean_squared_error as mse
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import adam_v2

import matplotlib.pyplot as plt
import Playing_Agent
import random

REPLAY_MEMORY_SIZE = 42000  # How many of last   to keep for model training, 42000 means remember last ~200 games
MIN_REPLAY_MEMORY_SIZE = 4200  # Minimum number of tricks in memory to start training, 10500 means at least ~20 games
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 20  # Terminal states (end of episodes)
DISCOUNT = 0.7

# Agent class
class PlayingNetwork:
    def __init__(self, input_size, name, masking=False):

        self.input_size = input_size
        self.name = name
        self.masking = masking

        # Main model
        self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.q_memory_counter = 0

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Array for plotting how avg q changes
        self.avg_q_memory = []
        self.ptp_q_memory = []
        self.x_q_mem = []

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
            optimizer=adam_v2.Adam(learning_rate=0.0001, clipvalue=0.5),
            metrics=[mse]
        )
        # print(model.summary())
        return model

    # Adds data to a memory replay array
    # (state, reward)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self):
        self.q_memory_counter += 1
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
        future_qs_list = self.model.predict(new_current_states)
        future_q_vals = self.target_model.predict(new_current_states)

        X = []
        y = []

        ptp_q_memory = []
        avg_q_memory = []
        # Now we need to enumerate our batches
        for index, (_, action, reward, _, done) in enumerate(minibatch):
            current_state = current_states[index]

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                # best action predicted by online model
                best_future_action = np.argmax(future_qs_list[index])

                # evaluation of best action done by target model
                max_future_q = future_q_vals[index][best_future_action]
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            if self.masking:
                # for each illegal move:
                    # set current_qs[illegal_move] = -1000
                pass

            # Every 10 games add to memory, start after 50 games
            if self.q_memory_counter % 2100 == 0 and self.q_memory_counter >= 10500:
                avg_q_memory.append(np.average(current_qs))
                ptp_q_memory.append(np.ptp(current_qs))

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        if len(avg_q_memory) > 0 and len(ptp_q_memory) > 0:
            # average over the 32 batches
            avg = round(np.average(avg_q_memory), 2)
            ptp = round(np.average(ptp_q_memory), 2)

            self.avg_q_memory.append(avg)
            self.ptp_q_memory.append(ptp)
            self.x_q_mem.append(self.q_memory_counter//210)

        # Fit on all samples as one batch, log only on terminal state

        self.model.fit(np.array(X), np.array(y),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0,
                       shuffle=False)

        if self.q_memory_counter % 21000 == 0:
            print("Saving q_mem...")
            f = open("plots/q_plots/values.txt", "w")
            f.write(f"{self.avg_q_memory}\n")
            f.write(f"{self.ptp_q_memory}\n")
            f.close()

            plt.plot(self.x_q_mem, self.ptp_q_memory)
            plt.xlabel("Games (n)", fontsize=10)
            plt.ylabel("PTP of q-values", fontsize=10)
            plt.savefig(f"plots/q_plots/ptp_plot")
            plt.close()

            plt.plot(self.x_q_mem, self.avg_q_memory)
            plt.xlabel("Games (n)", fontsize=10)
            plt.ylabel("AVG of q-values", fontsize=10)
            plt.savefig(f"plots/q_plots/avg_plot")
            plt.close()

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        sparse_state = Playing_Agent.PlayingAgent.key_to_state(self.input_size, state)
        return self.model.predict(np.array(sparse_state).reshape(-1, *sparse_state.shape))[0]
