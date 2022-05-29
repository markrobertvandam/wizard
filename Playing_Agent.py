import copy
import game
import numpy as np
import random
import time

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
    if input_size == 3915 or input_size == 315:
        f.write("Hand2: " + str(np.nonzero(play_state[60:120])[0].tolist()) + "\n")
        f.write("Hand3: " + str(np.nonzero(play_state[120:180])[0].tolist()) + "\n")
        current_pos = 180
    f.write("Trump: " + str(play_state[current_pos: current_pos + 5]) + "\n")
    current_pos += 5

    # if old
    if input_size == 3731:
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
    if input_size != 3731:
        f.write("Order: " + str(play_state[current_pos: current_pos + 3]) + "\n")
        f.write(
            "played trick: "
            + str(
                np.nonzero(play_state[current_pos + 3: current_pos + 123])[0].tolist()
            )
            + "\n"
        )
        current_pos += 123
    else:
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
        self.trained = False

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
                       player_order: list,
                       game_instance,
                       requested_color: int,
                       played_cards: list,
                       player_hand: list,
                       run_type="learning",) -> tuple:
        if self.verbose >= 2:
            print("Rollout policy used...")
        if len(player_hand) == 1:
            move = legal_moves[0]
            child = self.create_child(move,
                                      player_order,
                                      game_instance,
                                      requested_color,
                                      played_cards,
                                      run_type,
                                      terminal_node=True
                                      )
            self.parent_node = child
        elif len(legal_moves) > 0:
            move = random.choice(legal_moves)
            child = self.create_child(move,
                                      player_order,
                                      game_instance,
                                      requested_color,
                                      played_cards,
                                      run_type,
                                      )
            self.parent_node = child
        else:
            print("Parent has no legal moves!")
            return tuple((0, 0))
        if move != self.parent_node.card:
            print("Move is not correct, move and parent move respectively are: ", move, self.parent_node.card)
            exit()
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
        t1 = time.time()
        self.network_policy.train()
        self.backpropagate(node.parent, deck_dict, result, False)

    def best_child(self,
                   node: Node,
                   deck_dict: dict,
                   legal_moves: list,
                   player_order: list,
                   game_instance,
                   requested_color: int,
                   played_cards: list,
                   player_hand: list,
                   run_type: str,) -> tuple:

        if len(player_hand) == 1:
            move = legal_moves[0]
            child = self.create_child(move,
                                      player_order,
                                      game_instance,
                                      requested_color,
                                      played_cards,
                                      run_type,
                                      terminal_node=True
                                      )
            self.parent_node = child

            if move != self.parent_node.card:
                print("ERROOR ERROORR ERRORR")
                exit()
            return move

        elif len(legal_moves) > 0:
            q_vals = self.network_policy.get_qs(node.state)
            card_indexes = [deck_dict[card] for card in legal_moves]
            legal_q_vals = [q_vals[index] for index in card_indexes]
            best_child = np.argmax(legal_q_vals)
            move = legal_moves[best_child]

            child = self.create_child(move,
                                      player_order,
                                      game_instance,
                                      requested_color,
                                      played_cards,
                                      run_type,
                                     )
            self.parent_node = child

            if move != self.parent_node.card:
                print("ERROOR ERROORR ERRORR")
                exit()
            return move

        else:
            print("No cards in hand / no legal moves!")
            exit()

    def predict(self,
                deck_dict: dict,
                legal_moves: list,
                player_order: list,
                game_instance,
                requested_color: int,
                played_cards: list,
                player_hand: list,
                run_type = "learning",) -> tuple:
        """
        Use network to get best move
        :return:
        """
        node = self.parent_node
        return self.best_child(node,
                               deck_dict,
                               legal_moves,
                               player_order,
                               game_instance,
                               requested_color,
                               played_cards,
                               player_hand,
                               run_type
                               )

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
        player_order: list,
        game_instance,
        requested_color: int,
        played_cards: list,
        run_type: str,
        terminal_node=False,
    ) -> Node:
        """
        simulates and saves a child node where the given move
        is played in the given playing state
        :param move: the move to play and simulate
        :param player_order: list of players in turn order
        :param game_instance: instance of Game in current play state before move
        :param requested_color: requested color
        :param played_cards: cards played in trick so far
        :param run_type: type of agent {"learning", "learned", "heuristic", "random"}
        :param terminal_node: boolean whether the child node is a terminal node
        :return:
        """
        if self.verbose >= 2:
            print("Creating a child node...", move, played_cards)
        parent = self.parent_node

        # make temporary copy of the game
        temp_game = self.temp_game(game_instance, played_cards)

        player = len(played_cards)
        player_order_names = [p.player_name for p in player_order]
        new_player_dict = {
            "player1": temp_game.player1,
            "player2": temp_game.player2,
            "player3": temp_game.player3,
        }
        new_player_order = [new_player_dict[p] for p in player_order_names]
        game_class_players_order = [p.player_name for p in game_instance.players]
        shuffle_seed = []

        # game players are in some random order, temp_game players is still ordered
        for i in game_class_players_order:
            for p in temp_game.players:
                if p.player_name == i:
                    shuffle_seed.append(p)
        temp_game.players = shuffle_seed

        if self.verbose >= 3:
            print("Temporary player order: ", [p.player_name for p in new_player_order])
            print("Player that is learning: ", player, played_cards)

        # if theres more tricks to follow, children state is after move
        if not terminal_node:
            # simulate the play with selected move
            temp_game.play_trick(
                new_player_order,
                requested_color,
                player,
                card=move,
                player_limit=player + 1,
            )
            if player == 2:
                temp_game.wrap_up_trick(new_player_order)

        # else, terminal node, wrap up the round
        else:
            temp_game.play_trick(new_player_order, requested_color, player, card=move)
            temp_game.wrap_up_trick(new_player_order)

        play_state = temp_game.playing_state_space(
            new_player_order,
            new_player_order[player],
            temp_game.played_cards,
            temp=True,
        )

        if self.verbose:
            write_state(play_state, game_instance.output_path, self.input_size)

        key_state = self.state_to_key(play_state)

        node = Node(key_state, card=move, parent=parent)
        parent.children.append(node)
        if run_type == "learning":
            self.nodes[key_state] = node

        if terminal_node and run_type == "learning":
            self.last_terminal_node = self.nodes[key_state]

        return node
        # print("End of create child, created child for move: ", move)

    @staticmethod
    def temp_game(game_instance, played_cards):
        """
        returns a temporary copy of game_instance to simulate different plays
        :param game_instance: the game to copy
        :param played_cards: cards played in trick so far
        :return:
        """
        old_guess_agent = None
        old_epsilon = None

        old_play_agent = None
        old_play_epsilon = None

        if game_instance.player1.guess_type.startswith("learn"):
            old_guess_agent = game_instance.player1.guess_agent
            old_epsilon = game_instance.player1.epsilon
        if game_instance.player1.player_type.startswith("learn"):
            old_play_agent = game_instance.player1.play_agent
            old_play_epsilon = game_instance.player1.player_epsilon

        temp_game = game.Game(
            full_deck=copy.deepcopy(game_instance.full_deck),
            deck_dict=copy.deepcopy(game_instance.deck_dict),
            guess_type=copy.deepcopy(game_instance.player1.guess_type),
            player_type=copy.deepcopy(game_instance.player1.player_type),
            guess_agent=copy.copy(old_guess_agent),
            playing_agent=copy.copy(old_play_agent),
            epsilon=copy.deepcopy(old_epsilon),
            player_epsilon=copy.deepcopy(old_play_epsilon),
            verbose=copy.deepcopy(game_instance.player1.verbose),
            use_agent=copy.deepcopy(game_instance.use_agent),
        )
        temp_game.deck = copy.deepcopy(game_instance.deck)
        temp_game.trump = copy.deepcopy(game_instance.trump)
        temp_game.game_round = copy.deepcopy(game_instance.game_round)
        temp_game.played_cards = copy.deepcopy(played_cards)
        temp_game.played_round = copy.deepcopy(game_instance.played_round)
        temp_game.guesses = copy.deepcopy(game_instance.guesses)
        temp_game.output_path = copy.deepcopy(game_instance.output_path)

        temp_game.player1.hand = copy.deepcopy(game_instance.player1.hand)
        temp_game.player2.hand = copy.deepcopy(game_instance.player2.hand)
        temp_game.player3.hand = copy.deepcopy(game_instance.player3.hand)

        temp_game.player1.player_guesses = copy.deepcopy(
            game_instance.player1.player_guesses
        )
        temp_game.player2.player_guesses = copy.deepcopy(
            game_instance.player2.player_guesses
        )
        temp_game.player3.player_guesses = copy.deepcopy(
            game_instance.player3.player_guesses
        )

        temp_game.player1.trick_wins = copy.deepcopy(game_instance.player1.trick_wins)
        temp_game.player2.trick_wins = copy.deepcopy(game_instance.player2.trick_wins)
        temp_game.player3.trick_wins = copy.deepcopy(game_instance.player3.trick_wins)

        return temp_game
