import numpy as np
import random

from Playing_Agent import Node
import game

#
# TODO: MAKE HAND SORTED ONCE AT START


class Player:
    def __init__(
        self,
        player_name: str,
        player_type="random",
        guess_agent=None,
        play_agent=None,
        epsilon=None,
        player_epsilon=None,
        verbose=False,
    ) -> None:
        self.verbose = verbose
        self.player_name = player_name
        self.hand = []
        self.idx_dict = dict()
        self.win_cards = []
        self.player_guesses = 0
        self.trick_wins = 0
        self.player_type = player_type
        if self.player_type.startswith("learn"):
            self.guess_agent = guess_agent
            self.guess_agent.avg_reward = 0
            self.play_agent = play_agent
            self.current_state = None
            self.epsilon = epsilon
            self.player_epsilon = player_epsilon

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

            # ROOT NODE (cards in hand == round) -> add root and children
            if len(self.hand) == game_instance.game_round:
                if (
                    self.play_agent.state_to_key(state_space)
                    not in self.play_agent.nodes.keys()
                ):
                    self.play_agent.unseen_state(state_space)
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
                self.play_agent.parent_node.expanded = True
                if np.random.random() > self.player_epsilon:
                    # evaluate the best state
                    card = self.play_agent.predict()
                else:
                    # rollout till end of game
                    card = self.play_agent.rollout_policy()

        elif self.player_type == "learned" and self.play_agent.trained:
            key_state = self.play_agent.state_to_key(state_space)
            # ROOT NODE (cards in hand == round) -> add root and children
            if len(self.hand) == game_instance.game_round:
                self.play_agent.parent_node = Node(key_state, root=1, expanded=True)

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

        elif self.player_type == "random" or (
            self.player_type == "learned" and not self.play_agent.trained
        ):
            card = random.choice(legal_cards)

        elif self.player_type == "heuristic":
            # dodge win as high as possible if I am already at my goal
            if self.player_guesses == self.trick_wins:
                sorted_legal = sorted(legal_cards, key=lambda x: x[1], reverse=True)
                for card_option in sorted_legal:
                    if game.Game.trick_winner(
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
                            game.Game.trick_winner(played_cards + [card_option], trump)
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
            print(legal_cards, card, self.idx_dict)

        elif card != self.hand[self.idx_dict[card]]:
            print("Oh no! it went wrong!, card is not at that index?")
            self.hand.remove(card)

        else:
            del self.hand[self.idx_dict[card]]
        return card

    def guess_wins(self, max_guesses: int, trump: int, state_space=None) -> int:
        """
        Guess how many tricks the player will win this round
        :param max_guesses: max guesses (amount of tricks in this round)
        :param trump: trump in this round
        :param state_space: observation state used for guessing agent
        :return: None
        """
        if self.player_type == "random" or (
            self.player_type == "learned" and not self.guess_agent.trained
        ):
            self.player_guesses = random.randrange(max_guesses + 1)
        else:
            # print("im the smart one")
            if self.player_type == "learning":
                self.current_state = state_space
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    self.player_guesses = np.argmax(
                        self.guess_agent.get_qs(state_space)
                    )
                else:
                    # Get random action
                    self.player_guesses = random.randrange(max_guesses + 1)
            elif self.player_type == "learned":
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

    def update_agent(self, reward):
        # safety catch
        if self.player_type == "learning":
            self.guess_agent.update_replay_memory(
                (self.current_state, self.player_guesses, reward)
            )
            self.guess_agent.avg_reward += reward / self.guess_agent.guess_max
            self.guess_agent.train()

    def get_hand(self):
        return self.hand[:]

    def get_guesses(self):
        return self.player_guesses

    def get_trick_wins(self):
        return self.trick_wins
