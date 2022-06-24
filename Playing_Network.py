import numpy as np

from collections import deque

from keras.metrics import mean_squared_error as mse
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Softmax
from keras.optimizers import adam_v2

from tensorflow import math

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
    def __init__(self, input_size, name, masking: bool, dueling: bool):

        self.input_size = input_size
        self.name = name
        self.masking = masking
        self.dueling = dueling

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

        # Initialize Atoms
        self.num_actions = 60
        self.num_atoms = 51  # 51 for C51
        self.v_max = 10  # Max reward is 1, max avg q-score is ~3 in DDQN
        self.v_min = 0  # min score is 0
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

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

        distribution_list = self.dense_layer(60*51)(x)
        reshaped = Reshape((60, 51))(distribution_list)
        probs = Softmax(axis=1)(reshaped)
        model = Model(inputs=input1, outputs=probs)

        model.compile(
            loss='categorical_crossentropy',
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
        m_prob = [np.zeros((self.num_actions, self.num_atoms)) for _ in range(MINIBATCH_SIZE)]
        current_states, actions, rewards, new_current_states, illegal_moves, done = [], [], [], [], [], []
        for index in range(MINIBATCH_SIZE):
            transition = minibatch[index]
            # Get current states from minibatch, then query NN model for Q values
            sparse_state = Playing_Agent.PlayingAgent.key_to_state(self.input_size, transition[0])
            reshaped = sparse_state.reshape(-1, *sparse_state.shape)
            current_states.append(reshaped)
            actions.append(transition[1])
            rewards.append(transition[2])
            sparse_new_state = Playing_Agent.PlayingAgent.key_to_state(self.input_size, transition[3])
            reshaped = sparse_new_state.reshape(-1, *sparse_new_state.shape)
            new_current_states.append(reshaped)
            illegal_moves.append(transition[4])
            done.append(transition[5])

        # Get future states from minibatch, then query NN model for Q values
        # TODO: batch predict
        # z = self.model.predict(new_current_states)
        # z_ = self.target_model.predict(new_current_states)

        z = [self.model.predict(i)[0] for i in new_current_states]
        z_ = [self.target_model.predict(i)[0] for i in new_current_states]
        # #
        # # Get Optimal Actions for the next states (from distribution z)
        q = np.sum(np.multiply(z, np.array(self.z)), axis=-1)  # shaped (32, 60)
        optimal_action_idxs = np.argmax(q, axis=1)  # optimal action per batch (32,)

        # # Project Next State Value Distribution (of optimal action) to Current State
        # for i in range(MINIBATCH_SIZE):
        #     if done[i]:  # Terminal State
        #         # # unnecessary because ml and mu always equal bj
        #         # # Distribution collapses to a single point
        #         # Tz = rewards[i]  # reward is only 0 or 1, always between v_max and v_min
        #         # bj = (Tz - self.v_min) / self.delta_z
        #         # m_l, m_u = bj, bj  # bj is always 0 or 5, no need for floor/ceil
        #         # m_prob[i][actions[i]][m_l] += (m_u - bj)
        #         # m_prob[i][actions[i]][m_u] += (bj - m_l)
        #         pass
        #     else:
        #         for j in range(self.num_atoms):
        #             # reward {0, 1} + discount * atom {0, 0.2... 10}
        #             Tz = min(self.v_max, max(self.v_min, rewards[i] + DISCOUNT * self.z[j]))
        #             bj = (Tz - self.v_min) / self.delta_z
        #             m_l, m_u = math.floor(bj), math.ceil(bj)
        #             m_prob[i][actions[i]][int(m_l)] += z_[i][optimal_action_idxs[i]][j] * (m_u - bj)
        #             m_prob[i][actions[i]][int(m_u)] += z_[i][optimal_action_idxs[i]][j] * (bj - m_l)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(current_states).reshape(MINIBATCH_SIZE, self.input_size), np.array(m_prob),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0,
                       shuffle=False)

        if self.q_memory_counter % 21000 == 0:
            save_name = "DDDQN_small"
            print("Saving q_mem...")
            f = open(f"plots/q_plots/values_{save_name}", "w")
            f.write(f"{self.avg_q_memory}\n")
            f.write(f"{self.ptp_q_memory}\n")
            f.close()

            plt.plot(self.x_q_mem, self.ptp_q_memory)
            plt.xlabel("Games (n)", fontsize=10)
            plt.ylabel("PTP of q-values", fontsize=10)
            plt.savefig(f"plots/q_plots/ptp_plot_{save_name}")
            plt.close()

            plt.plot(self.x_q_mem, self.avg_q_memory)
            plt.xlabel("Games (n)", fontsize=10)
            plt.ylabel("AVG of q-values", fontsize=10)
            plt.savefig(f"plots/q_plots/avg_plot_{save_name}")
            plt.close()

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        sparse_state = Playing_Agent.PlayingAgent.key_to_state(self.input_size, state)
        reshaped = sparse_state.reshape(-1, *sparse_state.shape)
        return np.array(self.model.predict(reshaped))[0]
