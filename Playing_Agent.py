import copy
import random

from math import sqrt, log
from Playing_Network import PlayingNetwork


class Node:
    def __init__(self, state, legal_moves, leaf=0, root=0):
        self.state = state
        self.parent = None
        self.wins = 0
        self.children = []
        self.legal_moves = legal_moves
        self.leaf = leaf
        self.root = root
        self.visited = 0
        self.terminal = 0
        self.winner = None

    def fully_expanded(self):
        return len(self.legal_moves) == len(self.children)


# Agent class
class PlayingAgent:
    def __init__(self):

        self.game = None
        self.nodes = dict()
        self.network_policy = PlayingNetwork(3732)

    # main function for the Monte Carlo Tree Search
    def monte_carlo_tree_search(self, root):
        while True:
            leaf = self.traverse(root)
            simulation_result = self.rollout(leaf)
            self.backpropagate(leaf, simulation_result)
            if True:
                break

        return self.best_child(root)

    # function for node traversal
    def traverse(self, node: Node):
        if not node.leaf:
            node = self.best_child(node)
            return self.traverse(node)

        # Expand the node
        elif node.visited == 0:
            for move in node.legal_moves:
                new_node = self.create_node(node, move)
            node = self.pick_unvisited(node)

        return self.rollout(node)

    # function for the result of the simulation
    def rollout(self, node: Node):
        while node.terminal is False:
            node = self.rollout_policy(node)
        return node.winner

    # function for randomly selecting a child node
    def rollout_policy(self, node: Node):
        return self.create_child(node, random.choice(node.legal_moves))

    # function for backpropagation
    def backpropagate(self, node: Node, result):
        if node.root:
            return
        node.value = self.update_stats(node, result)
        self.backpropagate(node.parent)

    def pick_unvisited(self, node: Node):
        return random.choice(node.children)

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

        return best_child

    # TODO: Replace with network
    def evaluate_state(self, node):
        return self.network_policy.predict(node.state)

    def predict(self, game,  play_state, legal_moves):
        """
        Use network to get best move
        :param play_state: feature vector of current state
        :param legal_moves: legal cards to play
        :return:
        """
        start_node = Node(play_state, legal_moves)
        for move in legal_moves:
            self.create_child(game, start_node, move)
        return self.best_child(start_node)

    def create_child(self, game, parent, move):
        #TODO: Make the move and get new state
        temp_game = copy.deepcopy(game)
        node = Node()
        return node