import numpy as np
import os
import pickle
import random

import Playing_Agent
import utility_functions as util

#
# TODO: MAKE HAND SORTED ONCE AT START


class Player:
    def __init__(
        self,
        player_name: str,
        guess_type="random",
        player_type="random",
        guess_agent=None,
        play_agent=None,
        epsilon=None,
        player_epsilon=None,
        verbose=False,
        soft_guess=False,
        reoccur_path="",
        reoccur_bool=False,
    ) -> None:

        if player_name == "player1" and reoccur_path != "reoccur" and reoccur_bool:
            if not os.path.exists(reoccur_path):
                os.makedirs(reoccur_path)
            self.reoccur_path = reoccur_path

        self.reoccur_bool = reoccur_bool
        self.verbose = verbose
        self.player_name = player_name
        self.hand = []
        self.idx_dict = dict()
        self.win_cards = []
        self.player_guesses = 0
        self.trick_wins = 0
        self.guess_type = guess_type
        self.player_type = player_type
        if self.guess_type.startswith("learn"):
            self.guess_agent = guess_agent
            self.current_state = None
            self.epsilon = epsilon

        if self.player_type.startswith("learn"):
            self.play_agent = play_agent
            self.player_epsilon = player_epsilon
            self.soft_guess = soft_guess

        # for playing state
        self.possible_cards_one = [1] * 60
        self.possible_cards_two = [1] * 60

    def draw_cards(self, amount: int, deck: list) -> list:
        """
        :param amount: How many cards should be drawn
        :param deck: current state of deck
        :return: return the deck after drawing
        """
        for card in range(amount):
            self.hand.append(deck.pop())
        self.hand.sort(key=lambda x: (x[0], x[1]))
        return deck

    def play_card(
        self,
        trump: int,
        requested_color: int,
        played_cards: list,
        player_order,
        game_instance,
        state_space=None,
    ) -> tuple:
        """
        Plays a card from players hand
        :param trump: The trump card of the current round
        :param requested_color: The requested color that has to be played if possible
                                (blue, yellow, red, green, None yet, None this round)
        :param played_cards: Cards played so far
        :param player_order: order in which players play the current trick
        :param game_instance: instance of game to use for simulating moves with playing agent
        :param state_space: Observation state used for playing agent
        :return: the played card
        """
        requested_cards = []
        white_cards = []

        for idx in range(len(self.hand)):
            card_in_hand = self.hand[idx]
            self.idx_dict[card_in_hand] = idx
            if requested_color < 4:
                if card_in_hand[1] == 0 or card_in_hand[1] == 14:
                    white_cards.append(card_in_hand)
                elif card_in_hand[0] == requested_color:
                    requested_cards.append(card_in_hand)

        if not requested_cards:
            legal_cards = self.hand[:]
        else:
            legal_cards = requested_cards + white_cards

        #  print(f"\ncolor: {requested_color}\n hand: {self.hand}\n legal: {legal_cards}\n")
        card = None

        if self.player_type == "learning":
            if self.verbose >= 3:
                print("Round: ", game_instance.game_round)
                print("Amount of nodes: ", len(self.play_agent.nodes.keys()))

            # NODE IS HERE BEFORE PLAY
            key_state = util.state_to_key(state_space)
            self.play_agent.full_cntr[game_instance.game_round - 1] += 1

            # ROOT NODE (cards in hand == round) -> add root and children
            if len(self.hand) == game_instance.game_round:
                # Unseen root node
                if key_state not in self.play_agent.nodes.keys():
                    self.play_agent.unseen_state(state_space)
                # Previously seen root node
                else:
                    self.play_agent.parent_node = self.play_agent.get_node(state_space)

                # expand either way in case of unseen children
                self.play_agent.expand(
                    legal_cards,
                    player_order,
                    game_instance,
                    requested_color,
                    played_cards,
                    self.hand,
                )

                if np.random.random() > self.player_epsilon:
                    # evaluate the best state
                    card = self.play_agent.predict()
                else:
                    # rollout till end of game
                    card = self.play_agent.rollout_policy()

            # its a child, either terminal or not
            else:
                # expand in case of unseen children, then predict best move
                self.play_agent.expand(
                    legal_cards,
                    player_order,
                    game_instance,
                    requested_color,
                    played_cards,
                    self.hand,
                )
                if np.random.random() > self.player_epsilon:
                    # evaluate the best resulting state and return the corresponding move
                    card = self.play_agent.predict()
                else:
                    # rollout play
                    card = self.play_agent.rollout_policy()

            if self.player_name == "player1":
                new_parent = self.play_agent.parent_node
                new_state = new_parent.state
                sparse_state = util.key_to_state(self.play_agent.input_size, new_state)
                nodes = self.play_agent.nodes

                nodes[new_state].actual_encounters += 1

                if self.verbose >= 2:
                    util.write_state(sparse_state, "all-states", self.play_agent.input_size)

                # if node was encountered before
                if nodes[new_state].actual_encounters > 1:
                    self.play_agent.cntr[game_instance.game_round - 1] += 1

                    if self.reoccur_bool:
                        if len(self.hand) > 1:
                            util.write_state(sparse_state, os.path.join(self.reoccur_path, "reoccured-states"),
                                             self.play_agent.input_size)
                            file_path = os.path.join(self.reoccur_path, "reoccured-states.pkl")
                        else:
                            file_path = os.path.join(self.reoccur_path, "reoccured-terminal.pkl")

                        if os.path.exists(file_path):
                            file_reader = open(file_path, 'rb')
                            try:
                                file_reader.seek(0)
                                data = pickle.load(file_reader)
                            except EOFError:
                                print("Can't find pre-existing data, making new data")
                                data = []
                            except FileNotFoundError:
                                print("File does not exist anymore")
                                print(os.path.exists(file_path))
                                file_reader.close()
                                file_reader = open(file_path, 'rb+')
                                data = []

                            data.append(new_state)
                            file_reader.close()
                        else:
                            data = [new_state]

                        file_writer = open(file_path, 'wb')
                        pickle.dump(data, file_writer)
                        file_writer.close()

        elif self.player_type == "learned":
            key_state = util.state_to_key(state_space)
            # ROOT NODE (cards in hand == round) -> add root and children
            if len(self.hand) == game_instance.game_round:
                self.play_agent.parent_node = Playing_Agent.Node(key_state, root=1)

            self.play_agent.expand(
                legal_cards,
                player_order,
                game_instance,
                requested_color,
                played_cards,
                self.hand,
                run_type="learned",
            )
            # get action from network
            card = self.play_agent.predict()

        # Play the only legal card if theres only one
        elif len(legal_cards) == 1:
            card = legal_cards[0]

        elif self.player_type == "random":
            card = random.choice(legal_cards)


        elif self.player_type == "heuristic":
            # dodge win as high as possible if I am already at my goal
            if self.player_guesses <= self.trick_wins:
                sorted_legal = sorted(legal_cards, key=lambda x: x[1], reverse=True)
                for card_option in sorted_legal:
                    if util.trick_winner(
                        played_cards
                        + [card_option]
                        + [(0, 0)] * (2 - len(played_cards)),
                        trump,
                    ) != len(played_cards):
                        card = card_option
                        break

            # Still need wins
            else:
                reversed_sort_legal = sorted(legal_cards, key=lambda x: x[1])
                # see if I can for sure win this round with lowest card possible
                if len(played_cards) == 2:
                    for card_option in reversed_sort_legal:
                        if (
                            util.trick_winner(played_cards + [card_option], trump)
                            == 2
                        ):
                            card = card_option
                            break

                    if card is None:
                        card = reversed_sort_legal[0]
                        # Throw away lowest non-trump (if there is one), unless card is much higher than trump
                        if card[0] == trump:
                            for card_option in reversed_sort_legal[1:]:
                                if (
                                    card_option[0] != trump
                                    and card_option[1] - card[1] < 10
                                ):
                                    card = card_option
                                    break

        if card is None:
            card = legal_cards[0]
            # card = random.choice(legal_cards)

        if self.verbose >= 3:
            print("Card at end of play_card: ", self.hand, card)

        if card not in self.idx_dict.keys():
            print("Card not in hand? error..")
            print(legal_cards, card, self.idx_dict, self.hand)
            print(f"Player: {self.player_name}, Type: {self.player_type}")

        elif card != self.hand[self.idx_dict[card]]:
            print("Oh no! it went wrong!, card is not at that index?")
            self.hand.remove(card)

        else:
            del self.hand[self.idx_dict[card]]

        assert type(card) == tuple, f"tuple card expected, got: {card}, which is type {type(card)}"
        return card

    def guess_wins(self, max_guesses: int, trump: int, state_space=None) -> int:
        """
        Guess how many tricks the player will win this round
        :param max_guesses: max guesses (amount of tricks in this round)
        :param trump: trump in this round
        :param state_space: observation state used for guessing agent
        :return: None
        """
        if self.guess_type == "random":
            self.player_guesses = random.randrange(max_guesses + 1)
        else:
            # print("im the smart one")
            if self.guess_type == "learning":
                self.current_state = state_space
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    self.player_guesses = self.guess_agent.get_guess(state_space)
                else:
                    # Get random action
                    self.player_guesses = random.randrange(max_guesses + 1)
            elif self.guess_type == "learned":
                self.current_state = state_space
                self.player_guesses = np.argmax(self.guess_agent.get_qs(state_space))
            else:
                guesses = 0
                for card in self.hand:

                    # count wizards as win
                    if card[1] == 14:
                        guesses += 1
                        self.win_cards.append(card)

                    # count high trumps as win
                    elif card[0] == trump and card[1] > 7:
                        guesses += 1
                        self.win_cards.append(card)

                    # count low trumps when there are few cards
                    elif max_guesses < 5 and card[0] == trump and card[1] > 0:
                        guesses += 1
                        self.win_cards.append(card)

                    # count non-trump high cards as win except for early rounds
                    elif max_guesses > 4 and card[1] > 10:
                        guesses += 1
                        self.win_cards.append(card)

                self.player_guesses = guesses
            if self.verbose >= 2:
                print("\nGuessed: ", self.player_guesses)
                print("Hand: ", self.get_hand(), "Trump: ", trump)

        return self.player_guesses

    def update_agent(self):
        # safety catch
        if self.guess_type == "learning":
            self.guess_agent.update_replay_memory(
                (self.current_state, self.trick_wins)
            )
            self.guess_agent.train()

    def get_hand(self):
        return self.hand[:]

    def get_guesses(self):
        return self.player_guesses

    def get_trick_wins(self):
        return self.trick_wins
