import numpy as np
import random

import Playing_Network
from utility_functions import key_to_state, state_to_key, temp_game, write_state


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
        self.wins = 0
        self.actual_encounters = 0
        self.children = []
        self.root = root

        # card that was played to get to the state
        self.card = card


# Agent class
class PlayingAgent:
    def __init__(self, input_size: int, name=None, verbose=0, interactive=False, diff=False, punish=False, score=False):

        self.game = None
        self.input_size = input_size
        self.interactive = interactive
        self.nodes = dict()
        self.network_policy = Playing_Network.PlayingNetwork(input_size, name)
        self.verbose = verbose
        self.diff = diff
        self.punish = punish
        self.score = score
        self.counter = 0
        self.cntr = [0] * 20
        self.full_cntr = [0] * 20
        self.parent_node = None
        self.last_terminal_node = None

    def get_node(self, state_space: np.ndarray) -> Node:
        """"
        get node from node dictionary using key_state
        """
        key_state = state_to_key(state_space)
        return self.nodes[key_state]

    # function for randomly selecting a child node
    def rollout_policy(self) -> tuple:
        parent = self.parent_node
        if self.verbose >= 2:
            print("Rollout policy used...")
        if len(parent.children) > 0:
            child = random.choice(parent.children)
        else:
            print("Parent has no children!!! oh no!")
            exit()

        # Child becomes new selected parent node
        self.parent_node = child

        # Remove simulated children, no longer needed
        del parent.children[:]

        return self.parent_node.card

    # function for backpropagation
    def backpropagate(self, node: Node, result: int, score: int, diff: int, loss=0.0) -> float:
        self.counter += 1
        if self.counter % 2000 == 0:
            print(self.counter)

        if node.root:
            # root node is not added to memory
            return loss

        node.wins += result == 1
        if self.verbose >= 3:
            print("Node card: ", node.card)
        if self.diff:
            # use difference between own score and highest opponent score as result
            self.network_policy.update_replay_memory([node.state, diff])
        if self.score:
            # use own score as result
            self.network_policy.update_replay_memory([node.state, score])
        elif self.punish:
            # use (1 - difference between tricks won and tricks guessed) as result
            self.network_policy.update_replay_memory([node.state, result])
        else:
            # use 1 for correct guess, 0 for wrong guess
            if result != 1:
                result = 0
            self.network_policy.update_replay_memory([node.state, result])

        # Training after every memory update
        loss += self.network_policy.train()

        return self.backpropagate(node.parent, result, diff=diff, score=score, loss=loss)

    def best_child(self, node: Node) -> tuple:
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

        # Selected child becomes new parent
        self.parent_node = best_child

        # Remove simulated children
        del self.parent_node.parent.children[:]

        return best_child.card

    def evaluate_state(self, node: Node) -> float:
        sparse_state = key_to_state(self.input_size, node.state)
        return self.network_policy.predict(sparse_state)

    def predict(self) -> tuple:
        """
        Use network to get best move
        :return:
        """
        node = self.parent_node
        return self.best_child(node)

    def unseen_state(self, play_state: np.ndarray) -> None:
        """
        Create root node for unseen state
        :param play_state: playing state to create a node for
        :return:
        """
        if self.verbose >= 2:
            print("Adding unseen root node..")
        key_state = state_to_key(play_state)
        root_node = Node(key_state, root=1)
        if self.verbose:
            write_state(play_state, "state_err1", self.input_size, True)
        self.nodes[key_state] = root_node
        self.parent_node = root_node

    def expand(
        self,
        legal_moves: list,
        player_order: list,
        game_instance,
        requested_color: int,
        played_cards: list,
        player_hand: list,
        run_type="learning",
    ) -> None:
        """
        Creates child nodes for all legal moves
        :param legal_moves: the legal moves for player in current state
        :param player_order: list of players in turn order
        :param game_instance: instance of Game in current state
        :param requested_color: requested color
        :param played_cards: cards played in trick so far
        :param player_hand: cards in the players' hand
        :param run_type: type of agent {"learning", "learned", "heuristic", "random"}
        :return:
        """
        # Write node that is getting expanded
        if self.verbose >= 3:
            write_state(key_to_state(self.input_size, self.parent_node.state), "expanded-nodes", self.input_size)

        if self.verbose >= 2:
            print("Expanding the following moves: ", legal_moves)
        if len(player_hand) > 1:
            for move in legal_moves:
                self.create_child(
                    move,
                    player_order,
                    game_instance,
                    requested_color,
                    played_cards,
                    run_type,
                )
        else:
            # terminal node
            self.create_child(
                legal_moves[0],
                player_order,
                game_instance,
                requested_color,
                played_cards,
                run_type,
                terminal_node=True,
            )

    def create_child(
        self,
        move: tuple,
        player_order: list,
        game_instance,
        requested_color: int,
        played_cards: list,
        run_type: str,
        terminal_node=False,
    ) -> None:
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
        temp_instance = temp_game(game_instance, played_cards, interactive=self.interactive)

        player_index = len(played_cards)
        player_order_names = [p.player_name for p in player_order]
        new_player_dict = {
            "player1": temp_instance.player1,
            "player2": temp_instance.player2,
            "player3": temp_instance.player3,
        }
        new_player_order = [new_player_dict[p] for p in player_order_names]
        game_class_players_order = [p.player_name for p in game_instance.players]
        shuffle_seed = []

        # game players are in some random order, temp_game players is still ordered
        for i in game_class_players_order:
            for p in temp_instance.players:
                if p.player_name == i:
                    shuffle_seed.append(p)
        temp_instance.players = shuffle_seed

        if self.verbose >= 3:
            print("Temporary player order: ", [p.player_name for p in new_player_order])
            print("Player that is learning: ", player_index, played_cards)

        player = new_player_order[player_index]
        # if theres more tricks to follow, children state is after move
        if not terminal_node or self.interactive:
            if player_index == 2:
                # simulate the play with selected move (final move of trick)
                _, new_player_order = temp_instance.play_trick(
                    new_player_order,
                    requested_color,
                    player_index,
                    card=move,
                    player_limit=player_index + 1,
                    temp=True
                )
            else:
                # simulate play with selected non-final move
                temp_instance.play_trick(
                    new_player_order,
                    requested_color,
                    player_index,
                    card=move,
                    player_limit=player_index + 1,
                    temp=True
                )

        # else, terminal node, wrap up the round (no more choices by the agent left)
        else:
            _, new_player_order = temp_instance.play_trick(new_player_order, requested_color,
                                                           player_index, card=move, temp=True)

        play_state = temp_instance.playing_state_space(
            new_player_order,
            player,
            temp_instance.played_cards,
            temp=True,
        )

        if self.verbose:
            write_state(play_state, game_instance.output_path, self.input_size)

        key_state = state_to_key(play_state)

        node = Node(key_state, card=move, parent=parent)
        parent.children.append(node)
        if run_type == "learning":
            if key_state not in self.nodes.keys():
                self.nodes[key_state] = node

            if terminal_node:
                self.last_terminal_node = node
