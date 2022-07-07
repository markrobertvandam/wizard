import numpy as np

from keras.metrics import mean_squared_error as mse
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import adam_v2

from tensorflow import math

import replay_buffer
from utility_functions import key_to_state

import matplotlib.pyplot as plt

REPLAY_MEMORY_SIZE = 42000  # How many of last   to keep for model training, 42000 means remember last ~200 games
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 20  # Terminal states (end of episodes)
DISCOUNT = 0.7

# Agent class
class PlayingNetwork:
    def __init__(self, input_size, save_bool: bool, name: str,
                 masking: bool, dueling: bool, double: bool, priority: bool):

        self.input_size = input_size
        self.save_bool = save_bool
        self.name = name
        self.masking = masking
        self.dueling = dueling
        self.double = double
        self.priority = priority

        # Main model
        self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.q_memory_counter = 0

        # An array with last n steps for training
        self.replay_memory = replay_buffer.Memory(size=REPLAY_MEMORY_SIZE)

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

        q = self.dense_layer(64)(x)
        q = Dense(60, activation="linear")(q)  # q-values for actions

        if self.dueling:
            v = self.dense_layer(64)(x)
            v = Dense(1, activation="linear")(v)  # evaluation of state

            mean = math.reduce_mean(q, axis=1, keepdims=True)
            new_q = v + (q - mean)
            model = Model(inputs=input1, outputs=new_q)
        else:
            model = Model(inputs=input1, outputs=q)

        model.compile(
            loss="mse",
            optimizer=adam_v2.Adam(learning_rate=0.0001, clipvalue=0.5)
        )
        # print(model.summary())
        return model

    # Adds data to a memory replay array
    # (state, reward)
    def update_replay_memory(self, transition: list, priority=None):
        """
        Add new experience to replay memory, with max priority default (No TD known yet)
        """
        if priority is None:
            priority = self.replay_memory.max_priority
        # (current_state, action, reward, new_state, illegal_moves, done) = transition
        # print(f"Curr state: {key_to_state(self.input_size, current_state)}\n\n"
        #       f"Action: {action}\n\n"
        #       f"Reward: {reward}\n"
        #       f"New state: {key_to_state(self.input_size, new_state)}\n\n"
        #       f"Illegal_moves: {illegal_moves}\n"
        #       f"Done: {done}\n"
        #       f"Priority: {priority}\n"
        #       f"Max-prior: {self.replay_memory.max_priority}")
        # exit()
        self.replay_memory.append(transition, priority)

    # Trains main network every step during episode
    def train(self) -> float:
        # Get a minibatch of random samples from memory replay table
        minibatch, sampled_idxs, is_weights = self.replay_memory.sample(MINIBATCH_SIZE)
        if len(minibatch) != 32:
            print(f"MINIBATCH IS NOT 32 length")
            exit()

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([key_to_state(self.input_size, transition[0])
                                   for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([key_to_state(self.input_size, transition[3])
                                       for transition in minibatch])

        future_qs_list = self.model.predict(new_current_states)
        future_q_vals = self.target_model.predict(new_current_states)

        X = []
        y = []
        errors = []

        ptp_q_memory = []
        avg_q_memory = []
        # Now we need to enumerate our batches
        for index, (_, action, reward, _, illegal_moves, done) in enumerate(minibatch):
            current_state = current_states[index]

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:

                if self.double:
                    # best action predicted by online model
                    best_future_action = np.argmax(future_qs_list[index])

                    # evaluation of best action done by target model
                    max_future_q = future_q_vals[index][best_future_action]
                else:
                    # evaluation of best action chosen by target model
                    max_future_q = np.max(future_q_vals[index])

                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]

            if self.priority:
                # save error for priority calculation
                # error = mse([new_q], [current_qs[action]])
                # errors.append(error.numpy())
                errors.append(0.2)

            # Update Q value for given state
            current_qs[action] = new_q

            if self.masking:
                for move in illegal_moves:
                    current_qs[move] = -1000

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
        if self.priority:
            # update priorities of sampled experiences
            for i in range(len(sampled_idxs)):
                self.replay_memory.update(sampled_idxs[i], errors[i])
        loss = self.model.train_on_batch(np.array(X), np.array(y), is_weights)

        if self.save_bool and self.q_memory_counter % 21000 == 0:
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

        return loss

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]