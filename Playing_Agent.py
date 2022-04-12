import copy
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
    def __init__(self):

        self.game = None
        self.nodes = dict()
        self.network_policy = PlayingNetwork(3732)
        self.last_terminal_node = None

    # function for randomly selecting a child node
    def rollout_policy(self, node: Node):
        return random.choice(node.children).card

    # function for backpropagation
    def backpropagate(self, node: Node, result):
        if node.root:
            return
        node.wins += result / 100
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
        node = self.nodes[play_state]
        node.expanded = True
        return self.best_child(node)

    def unseen_state(self, play_state):
        """
        Create node with children for unseen state
        :param play_state:
        :return:
        """
        root_node = Node(play_state, root=1, expanded=True)
        self.nodes[play_state] = root_node

    def expand(
        self,
        legal_moves,
        parent,
        player_order,
        game_instance,
        requested_color,
        played_cards,
    ):
        if len(legal_moves) > 1:
            for move in legal_moves:
                self.create_child(
                    parent,
                    move,
                    player_order,
                    game_instance,
                    requested_color,
                    played_cards,
                )
        else:
            # terminal node
            self.create_child(
                parent,
                legal_moves[0],
                player_order,
                game_instance,
                requested_color,
                played_cards,
                terminal_node=True,
            )

    def create_child(
        self,
        parent,
        move,
        player_order,
        game_instance,
        requested_color,
        played_cards,
        terminal_node=False,
    ):
        temp_game = copy.deepcopy(game_instance)
        player = len(played_cards)
        temp_game.play_trick(player_order, requested_color, player, card=move)
        temp_game.play_till_player(player_order, player_limit=player)
        play_state = temp_game.playing_state_space(
            player_order[player], temp_game.played_cards
        )
        node = Node(play_state, card=move, parent=parent, terminal=terminal_node)
        parent.children.append(node)
        self.nodes[play_state] = node

        if terminal_node:
            self.last_terminal_node = node

        return node
