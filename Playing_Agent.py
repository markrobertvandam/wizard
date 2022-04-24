import copy
import game
import numpy as np
import random

from Playing_Network import PlayingNetwork
from scipy.sparse import coo_matrix


class Node:
    def __init__(
        self, state, root=0, card=None, parent=None, expanded=False, terminal=False
    ):
        self.state = state
        self.parent = parent
        self.wins = 0
        self.children = []
        self.root = root
        self.expanded = expanded
        self.terminal = terminal
        self.card = card


def write_state(play_state, output_path, actual=False):
    f = open(f"{output_path}.txt", "a")
    f.write("\n\n\n")
    np.set_printoptions(threshold=np.inf)
    if actual:
        f.write("Actual node\n")
    else:
        f.write("Simulated node\n")
    f.write("Hand: " + str(np.nonzero(play_state[:60])[0].tolist()) + "\n")
    f.write("Trump: " + str(play_state[60:65]) + "\n")
    f.write("Guesses: " + str(play_state[65:67]) + "\n")
    f.write("Round: " + str(play_state[67]) + "\n")
    f.write("Tricks needed: " + str(play_state[68]) + "\n")
    f.write("Tricks needed others: " + str(play_state[69:71]) + "\n")
    f.write(
        "played trick: " + str(np.nonzero(play_state[71:131])[0].tolist()) + "\n"
    )
    f.write("played round: " + str(np.nonzero(play_state[131:])[0].tolist()) + "\n")
    f.close()


# Agent class
class PlayingAgent:
    def __init__(self, verbose=0):

        self.game = None
        self.nodes = dict()
        self.network_policy = PlayingNetwork(3731)
        self.verbose = verbose
        self.counter = 0
        self.parent_node = None
        self.last_terminal_node = None

    def get_node(self, state_space):
        key_state = self.state_to_key(state_space)
        return self.nodes[key_state]

    @staticmethod
    def state_to_key(state_space):
        compressed_state = coo_matrix(state_space)
        key_state = tuple(np.concatenate((compressed_state.data, compressed_state.row, compressed_state.col)))
        return key_state

    @staticmethod
    def key_to_state(node_state):
        split = int(len(node_state)/3)
        sparse_state = coo_matrix((node_state[:split],
                                   (node_state[split:split*2], node_state[split*2:]))).toarray()[0].astype('float32')
        sparse_state = np.pad(sparse_state, (0, 3731 - len(sparse_state)), 'constant')
        return sparse_state

    # function for randomly selecting a child node
    def rollout_policy(self):
        node = self.parent_node
        if self.verbose == 2:
            print("Rollout policy used...")
        self.parent_node = random.choice(node.children)
        return self.parent_node.card

    # function for backpropagation
    def backpropagate(self, node: Node, result):
        self.counter += 1
        if self.counter % 2000 == 0:
            print(self.counter)
        if node.root:
            return
        node.wins += result / 100
        if self.verbose == 3:
            print("Node card: ", node.card)
        self.network_policy.update_replay_memory([node.state, result])
        self.network_policy.train()
        self.backpropagate(node.parent, result)

    def best_child(self, node):
        best_child = node.children[0]
        max_value = self.evaluate_state(best_child)
        for child in node.children[1:]:
            value = self.evaluate_state(child)
            if value > max_value:
                best_child = child
                max_value = value
            elif value == max_value and random.getrandbits(1):
                best_child = child
                max_value = value

        self.parent_node = best_child
        return best_child.card

    def evaluate_state(self, node):
        sparse_state = self.key_to_state(node.state)
        return self.network_policy.predict(sparse_state)

    def predict(self):
        """
        Use network to get best move
        :param play_state: feature vector of current state
        :return:
        """
        node = self.parent_node
        return self.best_child(node)

    def unseen_state(self, play_state):
        """
        Create root node for unseen state
        :param play_state:
        :return:
        """
        if self.verbose == 2:
            print("Adding unseen root node..")
        key_state = self.state_to_key(play_state)
        root_node = Node(key_state, root=1, expanded=True)
        if self.verbose:
            write_state(play_state, "state_err1", True)
        self.nodes[key_state] = root_node
        self.parent_node = root_node

    def expand(
        self,
        legal_moves,
        player_order,
        game_instance,
        requested_color,
        played_cards,
        player_hand,
    ):
        if self.verbose == 2:
            print("Expanding the following moves: ", legal_moves)
        if len(player_hand) > 1:
            for move in legal_moves:
                self.create_child(
                    move,
                    player_order,
                    game_instance,
                    requested_color,
                    played_cards,
                )
        else:
            # terminal node
            self.create_child(
                legal_moves[0],
                player_order,
                game_instance,
                requested_color,
                played_cards,
                terminal_node=True,
            )

    def create_child(
        self,
        move,
        player_order,
        game_instance,
        requested_color,
        played_cards,
        terminal_node=False,
    ):
        if self.verbose == 2:
            print("Creating a child node...", move, played_cards)
        parent = self.parent_node

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

        if self.verbose == 3:
            print("Temporary player order: ", [p.player_name for p in new_player_order])
            print("Player that is learning: ", player, played_cards)

        # if theres more tricks to follow, children state is after move
        if not terminal_node:
            # simulate the play with selected move
            temp_game.play_trick(new_player_order, requested_color, player,
                                 card=move, player_limit=player + 1)
            if player == 2:
                temp_game.wrap_up_trick(new_player_order)

        # else, terminal node, wrap up the round
        else:
            temp_game.play_trick(new_player_order, requested_color, player,
                                 card=move)
            temp_game.wrap_up_trick(new_player_order)
        play_state = temp_game.playing_state_space(
            new_player_order[player], temp_game.played_cards, temp=True
        )

        if self.verbose:
            write_state(play_state, game_instance.output_path)

        key_state = self.state_to_key(play_state)

        if key_state not in self.nodes.keys():
            node = Node(key_state, card=move, parent=parent, terminal=terminal_node)
            parent.children.append(node)
            self.nodes[key_state] = node

        if terminal_node:
            self.last_terminal_node = self.nodes[key_state]

    @staticmethod
    def temp_game(game_instance, played_cards):
        temp_game = game.Game(
            full_deck=copy.deepcopy(game_instance.full_deck),
            deck_dict=copy.deepcopy(game_instance.deck_dict),
            run_type=copy.deepcopy(game_instance.player1.player_type),
            guess_agent=copy.copy(game_instance.player1.guess_agent),
            playing_agent=copy.copy(game_instance.player1.play_agent),
            epsilon=copy.deepcopy(game_instance.player1.epsilon),
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
