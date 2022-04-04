import random

from math import sqrt, log


class Node:
    def __init__(self, leaf, root):
        self.state = None
        self.parent = None
        self.wins = 0
        self.children = []
        self.legal_moves = []
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

        self.avg_reward = 0  # stores avg reward every game
        self.accuracy = 0.02
        self.nodes = dict()

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
        return self.create_node(node, random.choice(node.legal_moves))

    # function for backpropagation
    def backpropagate(self, node: Node, result):
        if node.root:
            return
        node.value = self.update_stats(node, result)
        self.backpropagate(node.parent)

    def pick_unvisited(self, node: Node):
        return random.choice(node.children)

    def best_child(self, node):
        max_value = -10000
        for child in node.children:
            value = self.ucb1(child)
            if value > max_value:
                best_child = child
                max_value = value
            elif value == max_value and random.getrandbits(1):
                best_child = child
                max_value = value

        return best_child

    def ucb1(self, node):
        wins = node.wins
        sims = node.visited
        avg_value = wins / sims
        ucb1 = avg_value + sqrt(2 * log(node.parent.visited) / sims)
        return ucb1

    def predict(self, play_state):
        root_node = self.nodes[play_state]
        return self.best_child(root_node)

    def create_node(self, parent, move):
        node = Node()
        return node