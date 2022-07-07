import numpy as np
import Sum_Tree

from collections import deque


class Memory(object):
    def __init__(self, size: int):
        self.size = size
        self.curr_write_idx = 0
        self.available_samples = 0
        self.buffer = deque(maxlen=size)
        self.base_node, self.leaf_nodes = Sum_Tree.create_tree([0 for _ in range(self.size)])
        self.frame_idx = 0
        self.action_idx = 1
        self.reward_idx = 2
        self.terminal_idx = 3
        self.beta = 0.4
        self.alpha = 0.6
        self.min_priority = 0.01
        self.max_priority = 1

    def append(self, experience: list, priority: float):
        if self.curr_write_idx >= len(self.buffer):
            self.buffer.append(experience)
        else:
            self.buffer[self.curr_write_idx] = experience
        self.update(self.curr_write_idx, priority)
        self.curr_write_idx += 1
        # reset the current writer position index if creater than the allowed size
        if self.curr_write_idx >= self.size:
            self.curr_write_idx = 0
        # max out available samples at the memory buffer size
        if self.available_samples + 1 < self.size:
            self.available_samples += 1
        else:
            self.available_samples = self.size - 1

    def update(self, idx: int, priority: float):
        adjusted_p = self.adjust_priority(priority)
        if adjusted_p > self.max_priority:
            self.max_priority = adjusted_p
        Sum_Tree.update(self.leaf_nodes[idx], adjusted_p)

    def adjust_priority(self, priority: float):
        return np.power(priority + self.min_priority, self.alpha)

    def sample(self, minibatch_size: int) -> tuple:
        sampled_idxs = []
        is_weights = []
        for _ in range(minibatch_size):
            # get random float value used for getting a sample
            sample_val = np.random.uniform(0, self.base_node.value)

            # retrieve sample node (experience) from sum tree, given the float
            samp_node = Sum_Tree.retrieve(sample_val, self.base_node)
            sampled_idxs.append(samp_node.idx)

            # (P(i) = (pi ^ a / sum of all (priorities ^ a))
            probability = samp_node.value / self.base_node.value

            # (N * P(i))
            is_weights.append((self.available_samples + 1) * probability)

        # apply the beta factor and normalise so that the maximum is_weight < 1
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, -self.beta)
        is_weights = is_weights / np.max(is_weights)

        # now load up the state and next state variables according to sampled idxs
        # transition: (current_state, action, reward, new_state, illegal_moves, done)
        transitions = [self.buffer[idx] for idx in sampled_idxs]

        if len(transitions) != 32 or len(sampled_idxs) != 32 or len(is_weights) != 32:
            print("FUCK, BATCH IS WRONG!")
            exit()

        print(f"len is correct, there are {len(transitions)} transitions returned from sample.")
        return transitions, sampled_idxs, is_weights
