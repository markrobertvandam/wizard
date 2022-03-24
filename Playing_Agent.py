import random

# Agent class
class PlayingAgent:
    def __init__(self, input_size):

        self.input_size = input_size
        self.avg_reward = 0  # stores avg reward every game
        self.accuracy = 0.02

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
    def traverse(self, node):
        if node.expanded:
            node = self.best_uct(node)
            self.traverse(node)

        elif len(node.children) > 0:
            return self.pick_unvisited(node.children)

        # in case no children are present / node is terminal
        else:
            node.leaf = True
            return node

    # function for the result of the simulation
    def rollout(self, node):
        while node.leaf is False:
            node = self.rollout_policy(node)
        return node.winner

    # function for randomly selecting a child node
    def rollout_policy(self, node):
        return self.pick_random(node.children)

    # function for backpropagation
    def backpropagate(self, node, result):
        if node.root:
            return
        node.stats = self.update_stats(node, result)
        self.backpropagate(node.parent)

    # function for selecting the best child
    # node with highest number of visits
    def best_child(self, node):
        #TODO: pick child with highest number of visits
        pass

    def predict(self, legal_cards):
        return random.choice(legal_cards)