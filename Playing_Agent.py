import numpy as np
import random

from Playing_Network import PlayingNetwork
from utility_functions import state_to_key, key_to_state


class Node:
    """
    class for nodes of the node-tree when doing
    Monte-Carlo Treesearch
    """
    def __init__(
        self, state, legal_cards, before_play: bool, root=False, card=None, parent=None
    ):
        self.state = state
        self.parent = parent
        self.child = None
        self.legal_moves = legal_cards
        self.root = root

        # card that was played to get to the state
        self.card = card

        # is it S or S' (before or after agents action)
        self.before_play = before_play


def write_state(play_state: np.ndarray, output_path: str, input_size: int, type_node="After play") -> None:
    """
    Write playing state to text file for debugging
    :param play_state: actual playing state
    :param output_path: path to textfile
    :param input_size: input size of the playing model
    :param type_node: type of node reached in play
    :return:
    """
    f = open(f"{output_path}.txt", "a")
    f.write("\n\n\n")
    np.set_printoptions(threshold=np.inf)
    f.write(f"{type_node} node\n")
    f.write("Hand: " + str(np.nonzero(play_state[:60])[0].tolist()) + "\n")
    current_pos = 60
    # # if cheater
    # if input_size % 100 == 15 or input_size % 100 == 13:
    #     f.write("Hand2: " + str(np.nonzero(play_state[60:120])[0].tolist()) + "\n")
    #     f.write("Hand3: " + str(np.nonzero(play_state[120:180])[0].tolist()) + "\n")
    #     current_pos = 180
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

    # if not olds
    if input_size % 100 == 93 or input_size % 100 == 13:
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

    if input_size == 313:
        f.write(
            "possible cards player2: "
            + str(np.nonzero(~(play_state[current_pos:current_pos+60] != 0))[0].tolist())
            + "\n"
        )
        f.write(
            "possible cards player3: "
            + str(np.nonzero(~(play_state[current_pos+60:] != 0))[0].tolist())
            + "\n"
        )
    f.close()


# Agent class
class PlayingAgent:
    def __init__(self, input_size: int, save_bool=False, name=None, verbose=0, mask=False, dueling=False, double=False):

        self.game = None
        self.input_size = input_size
        self.nodes = dict()
        self.network_policy = PlayingNetwork(input_size, save_bool, name, masking=mask, dueling=dueling, double=double)
        self.verbose = verbose
        self.counter = 0
        self.parent_node = None
        self.last_terminal_node = None
        self.pred_counter = 0

    def get_node(self, state_space: np.ndarray) -> Node:
        """"
        get node from node dictionary using key_state
        """
        key_state = state_to_key(state_space)
        return self.nodes[key_state]

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

        if node.before_play:
            if self.verbose >= 3:
                print("Node card: ", node.child.card)
            action = deck_dict[node.child.card]
            legal_moves = [deck_dict[move] for move in node.legal_moves]
            illegal_moves = [move for move in range(60) if move not in legal_moves]

            if self.verbose >= 3:
                # checking if the (S, a, S') pairs are correct
                write_state(key_to_state(self.input_size, node.state), "sas", self.input_size, type_node="before play")
                f = open("sas.txt", "a")
                f.write(f"\nAction taken: {action}, result: {result}\n")
                f.close()
                write_state(key_to_state(self.input_size, node.child.state), "sas", self.input_size, type_node="after play")
                f = open("sas.txt", "a")
                f.write("\n\n")
                f.close()

            self.network_policy.update_replay_memory([node.state, action, result, node.child.state, illegal_moves, done])
            self.network_policy.train()

            # Only add (S, a, S'), go next if current node is S'
            if node.root:
                # done propagating entire game, nodes can be reset
                self.nodes = dict()
                return

        self.backpropagate(node.parent, deck_dict, result, False)

    def best_child(self,
                   state,
                   deck_dict: dict,
                   legal_moves: list,
                   player_hand: list,) -> tuple:

        if len(player_hand) == 1:
            move = legal_moves[0]
            return move

        elif len(legal_moves) > 0:
            if self.verbose >= 3:
                sparse_state = key_to_state(self.input_size, state)
                write_state(sparse_state, "predict_nodes", self.input_size, "predict")
            q_vals = self.network_policy.get_qs(state)
            card_indexes = [deck_dict[card] for card in legal_moves]
            legal_q_vals = [q_vals[index] for index in card_indexes]
            best_child = np.argmax(legal_q_vals)
            move = legal_moves[best_child]

            return move

        else:
            print("No cards in hand / no legal moves!")
            exit()

    def predict(self,
                state,
                deck_dict: dict,
                legal_moves: list,
                player_order: list,) -> tuple:
        """
        Use network to get best move
        :return:
        """
        # predict based on state right before playing, what actions are best
        # state before action + action -> state after action
        # parent = state right before play
        # action = move taken
        # child = state after play
        self.pred_counter += 1
        return self.best_child(state,
                               deck_dict,
                               legal_moves,
                               player_order,)

    def unseen_state(self, play_state: np.ndarray, legal_cards: list) -> None:
        """
        Create root node for unseen state
        :param play_state: playing state to create a node for
        :return:
        """
        if self.verbose >= 2:
            print("Adding unseen root node..")
        key_state = state_to_key(play_state)
        root_node = Node(key_state, legal_cards, before_play=True, root=True)
        if self.verbose:
            write_state(play_state, "state_err1", self.input_size, "root")
        self.nodes[key_state] = root_node
        self.parent_node = root_node

    def new_child_state(self, play_state: np.ndarray, legal_cards: list) -> None:
        """
        Create node for child state before play
        :param play_state: playing state to create a node for
        :return:
        """
        if self.verbose >= 2:
            print("Adding unseen new node for before play..")
        key_state = state_to_key(play_state)
        new_node = Node(key_state, legal_cards, before_play=True, parent=self.parent_node)
        if self.verbose:
            write_state(play_state, "state_err1", self.input_size, "before play")
        self.nodes[key_state] = new_node
        self.parent_node.child = new_node
        self.parent_node = new_node

    def create_child(
        self,
        move: tuple,
        output_path,
        play_state,
        played_cards: list,
        legal_cards: list,
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

        key_state = state_to_key(play_state)

        node = Node(key_state, legal_cards, before_play=False, card=move, parent=parent)
        parent.child = node
        self.nodes[key_state] = node

        if terminal_node:
            if self.verbose:
                print("Setting node as terminal_node")
            self.last_terminal_node = self.nodes[key_state]

        self.parent_node = node
        # print("End of create child, created child for move: ", move)
