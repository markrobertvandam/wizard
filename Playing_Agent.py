import numpy as np
import random

from Playing_Network import PlayingNetwork
from scipy.sparse import coo_matrix


class Node:
    """
    class for nodes of the node-tree when doing
    Monte-Carlo Treesearch
    """
    def __init__(
        self, state, root=0, card=None, parent=None
    ):
        self.state = state
        self.parent = parent
        self.children = []
        self.root = root

        # card that was played to get to the state
        self.card = card


def write_state(play_state: np.ndarray, output_path: str, input_size: int, actual=False) -> None:
    """
    Write playing state to text file for debugging
    :param play_state: actual playing state
    :param output_path: path to textfile
    :param input_size: input size of the playing model
    :param actual: simulated node or actual node reached in play
    :return:
    """
    f = open(f"{output_path}.txt", "a")
    f.write("\n\n\n")
    np.set_printoptions(threshold=np.inf)
    if actual:
        f.write("Actual node\n")
    else:
        f.write("Simulated node\n")
    f.write("Hand: " + str(np.nonzero(play_state[:60])[0].tolist()) + "\n")
    current_pos = 60
    # if cheater
    if input_size % 100 == 15 or input_size % 100 == 13:
        f.write("Hand2: " + str(np.nonzero(play_state[60:120])[0].tolist()) + "\n")
        f.write("Hand3: " + str(np.nonzero(play_state[120:180])[0].tolist()) + "\n")
        current_pos = 180
    f.write("Trump: " + str(play_state[current_pos: current_pos + 5]) + "\n")
    current_pos += 5

    # if old
    if input_size % 100 == 31:
        f.write("Guesses: " + str(play_state[current_pos: current_pos + 2]) + "\n")
        current_pos += 2
    else:
        f.write("Guesses: " + str(play_state[current_pos: current_pos + 3]) + "\n")
        current_pos += 3
    f.write("Round: " + str(play_state[current_pos]) + "\n")
    f.write("Tricks needed: " + str(play_state[current_pos + 1]) + "\n")
    current_pos += 2

    f.write(
        "Tricks needed others: " + str(play_state[current_pos: current_pos + 2]) + "\n"
    )
    current_pos += 2

    # if not old
    if input_size % 100 == 93:
        f.write("Order: " + str(play_state[current_pos]) + "\n")
        f.write(
            "played trick: "
            + str(
                np.nonzero(play_state[current_pos + 1: current_pos + 121])[0].tolist()
            )
            + "\n"
        )
        current_pos += 121
    elif input_size % 100 == 95:
        f.write("Order: " + str(play_state[current_pos: current_pos + 3]) + "\n")
        f.write(
            "played trick: "
            + str(
                np.nonzero(play_state[current_pos + 3: current_pos + 123])[0].tolist()
            )
            + "\n"
        )
        current_pos += 123
    elif input_size % 100 == 31:
        f.write(
            "played trick: "
            + str(np.nonzero(play_state[current_pos: current_pos + 60])[0].tolist())
            + "\n"
        )
        current_pos += 60

    # if not small
    if input_size > 3600:
        f.write(
            "played round: "
            + str(np.nonzero(play_state[current_pos:])[0].tolist())
            + "\n"
        )
    f.close()


# Agent class
class PlayingAgent:
    def __init__(self, input_size: int, name=None, verbose=0):

        self.game = None
        self.input_size = input_size
        self.nodes = dict()
        self.network_policy = PlayingNetwork(input_size, name)
        self.verbose = verbose
        self.counter = 0
        self.parent_node = None
        self.last_terminal_node = None
        self.pred_counter = 0

    def get_node(self, state_space: np.ndarray) -> Node:
        """"
        get node from node dictionary using key_state
        """
        key_state = self.state_to_key(state_space)
        return self.nodes[key_state]

    @staticmethod
    def state_to_key(state_space: np.ndarray) -> tuple:
        compressed_state = coo_matrix(state_space)
        key_state = tuple(
            np.concatenate(
                (compressed_state.data, compressed_state.row, compressed_state.col)
            )
        )
        return key_state

    @staticmethod
    def key_to_state(input_size: int, node_state: tuple) -> np.ndarray:
        split = int(len(node_state) / 3)
        sparse_state = (
            coo_matrix(
                (
                    node_state[:split],
                    (node_state[split: split * 2], node_state[split * 2:]),
                )
            )
            .toarray()[0]
            .astype("float32")
        )
        sparse_state = np.pad(
            sparse_state, (0, input_size - len(sparse_state)), "constant"
        )
        return sparse_state

    # function for randomly selecting a child node
    def rollout_policy(self,
                       legal_moves: list,
                       player_hand: list,) -> tuple:
        if self.verbose >= 2:
            print("Rollout policy used...")
        if len(player_hand) == 1:
            move = legal_moves[0]
        elif len(legal_moves) > 0:
            move = random.choice(legal_moves)

        else:
            print("Parent has no legal moves!")
            return tuple((0, 0))
        if self.verbose >= 2:
            print("Move obtained randomly is: ", move)
        return move

    # function for backpropagation
    def backpropagate(self, node: Node, deck_dict: dict, result: int, done=True) -> None:
        self.counter += 1
        if self.counter % 2000 == 0:
            print(self.counter)
        if node.root:
            # done propagating entire game, nodes can be reset
            self.nodes = dict()
            return
        if self.verbose >= 3:
            print("Node card: ", node.card)
        action = deck_dict[node.card]
        self.network_policy.update_replay_memory([node.parent.state, action, result, node.state, done])
        self.network_policy.train()
        self.backpropagate(node.parent, deck_dict, result, False)

    def best_child(self,
                   node: Node,
                   deck_dict: dict,
                   legal_moves: list,
                   player_hand: list,) -> tuple:

        if len(player_hand) == 1:
            move = legal_moves[0]
            return move

        elif len(legal_moves) > 0:
            q_vals = self.network_policy.get_qs(node.state)
            card_indexes = [deck_dict[card] for card in legal_moves]
            legal_q_vals = [q_vals[index] for index in card_indexes]
            best_child = np.argmax(legal_q_vals)
            move = legal_moves[best_child]

            return move

        else:
            print("No cards in hand / no legal moves!")
            exit()

    def predict(self,
                deck_dict: dict,
                legal_moves: list,
                player_order: list,) -> tuple:
        """
        Use network to get best move
        :return:
        """
        self.pred_counter += 1
        node = self.parent_node
        return self.best_child(node,
                               deck_dict,
                               legal_moves,
                               player_order,)

    def unseen_state(self, play_state: np.ndarray) -> None:
        """
        Create root node for unseen state
        :param play_state: playing state to create a node for
        :return:
        """
        if self.verbose >= 2:
            print("Adding unseen root node..")
        key_state = self.state_to_key(play_state)
        root_node = Node(key_state, root=1)
        if self.verbose:
            write_state(play_state, "state_err1", self.input_size, True)
        self.nodes[key_state] = root_node
        self.parent_node = root_node

    def create_child(
        self,
        move: tuple,
        output_path,
        play_state,
        played_cards: list,
        terminal_node=False,
    ) -> None:
        """
        saves a child node where the given move
        is played in the given playing state
        :param move: the move to play and simulate
        :param played_cards: cards played in trick so far
        :param terminal_node: boolean whether the child node is a terminal node
        :return:
        """
        if self.verbose >= 2:
            print("Creating a child node...", move, played_cards)
        parent = self.parent_node

        if self.verbose:
            write_state(play_state, output_path, self.input_size)

        key_state = self.state_to_key(play_state)

        node = Node(key_state, card=move, parent=parent)
        parent.children.append(node)
        self.nodes[key_state] = node

        if terminal_node:
            if self.verbose:
                print("Setting node as terminal_node")
            self.last_terminal_node = self.nodes[key_state]

        self.parent_node = node
        # print("End of create child, created child for move: ", move)
