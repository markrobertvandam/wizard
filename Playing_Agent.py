import copy
import game
import numpy as np
import random

from Playing_Network import PlayingNetwork


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


# Agent class
class PlayingAgent:
    def __init__(self, verbose=0):

        self.game = None
        self.nodes = dict()
        self.network_policy = PlayingNetwork(3732)
        self.last_terminal_nodes = None
        self.verbose = verbose
        self.counter = 0

    # function for randomly selecting a child node
    def rollout_policy(self, node_space):
        node = self.nodes[tuple(node_space)]
        if self.verbose:
            print("Rollout policy used...")
        return random.choice(node.children).card

    # function for backpropagation
    def backpropagate(self, node: Node, result):
        self.counter += 1
        if self.counter % 100 == 0:
            print(self.counter)
        if node.root:
            return
        node.wins += result / 100
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

        return best_child.card

    def evaluate_state(self, node):
        return self.network_policy.predict(node.state)

    def predict(self, play_state):
        """
        Use network to get best move
        :param play_state: feature vector of current state
        :return:
        """
        node = self.nodes[tuple(play_state)]
        node.expanded = True
        return self.best_child(node)

    def unseen_state(self, play_state):
        """
        Create node with children for unseen state
        :param play_state:
        :return:
        """
        if self.verbose:
            print("Adding unseen node..")
        root_node = Node(play_state, root=1, expanded=True)
        self.nodes[tuple(play_state)] = root_node

    def expand(
        self,
        legal_moves,
        parent_space,
        player_order,
        game_instance,
        requested_color,
        played_cards,
        player_hand,
    ):
        if self.verbose:
            print("Expanding the following moves: ", legal_moves)
        if len(player_hand) > 1:
            for move in legal_moves:
                self.create_child(
                    parent_space,
                    move,
                    player_order,
                    game_instance,
                    requested_color,
                    played_cards,
                )
        else:
            # terminal node
            self.create_child(
                parent_space,
                legal_moves[0],
                player_order,
                game_instance,
                requested_color,
                played_cards,
                terminal_node=True,
            )

    def create_child(
        self,
        parent_space,
        move,
        player_order,
        game_instance,
        requested_color,
        played_cards,
        terminal_node=False,
    ):
        if self.verbose:
            print("Creating a child node...", move, played_cards)
        parent = self.nodes[tuple(parent_space)]

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
        temp_game.guesses = copy.deepcopy(game_instance.guesses)

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

        player = len(played_cards)
        player_order_names = [p.player_name for p in player_order]
        new_player_dict = {
            "player1": temp_game.player1,
            "player2": temp_game.player2,
            "player3": temp_game.player3,
        }
        new_player_order = [new_player_dict[p] for p in player_order_names]
        temp_game.play_trick(new_player_order, requested_color, player, card=move)

        if not terminal_node:
            print("PLAYING TILL PLAYER BCS TERMINAL")
            temp_game.play_till_player(new_player_order, player_limit=player)
        play_state = temp_game.playing_state_space(
            new_player_order[player], temp_game.played_cards
        )
        print("Simulated cards: ", temp_game.played_cards)
        f = open("state_diff.txt", "a")
        f.write("\n\n\n")
        np.set_printoptions(threshold=np.inf)
        f.write("Hand: " + str(np.nonzero(play_state[:60])[0].tolist()) + "\n")
        f.write("Trump: " + str(play_state[60:65]) + "\n")
        f.write("Guesses: " + str(play_state[65:67]) + "\n")
        f.write("Round: " + str(play_state[67]) + "\n")
        f.write("Tricks needed: " + str(play_state[68]) + "\n")
        f.write("Tricks needed others: " + str(play_state[69:71]) + "\n")
        f.write(
            "played trick: " + str(np.nonzero(play_state[71:131])[0].tolist()) + "\n"
        )
        f.close()
        node = Node(play_state, card=move, parent=parent, terminal=terminal_node)
        parent.children.append(node)
        self.nodes[tuple(play_state)] = node

        if terminal_node:
            self.last_terminal_node = node
