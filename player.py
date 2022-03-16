import numpy as np
import random
import game

#
# TODO: MAKE HAND SORTED ONCE AT START


class Player:
    def __init__(
        self,
        player_name: str,
        player_type="random",
        guess_agent=None,
        epsilon=None,
        verbose=False,
    ) -> None:
        self.verbose = verbose
        self.player_name = player_name
        self.hand = []
        self.win_cards = []
        self.player_guesses = 0
        self.trick_wins = 0
        self.player_type = player_type
        if self.player_type.startswith("learn"):
            self.guess_agent = guess_agent
            self.guess_agent.avg_reward = 0
            self.current_state = None
            self.epsilon = epsilon

    def get_hand(self):
        return self.hand[:]

    def get_guesses(self):
        return self.player_guesses

    def get_trick_wins(self):
        return self.trick_wins

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

    def play_card(self, trump: int, requested_color: int, played_cards: list) -> tuple:
        """
        Plays a card from players hand
        :param trump: The trump card of the current round
        :param requested_color: The requested color that has to be played if possible
                                (blue, yellow, red, green, None yet, None this round)
        :param played_cards: Cards played so far
        :return: the played card
        """
        requested_cards = []
        white_cards = []
        if requested_color < 4:
            for card in self.hand:
                if card[1] == 0 or card[1] == 14:
                    white_cards.append(card)
                elif card[0] == requested_color:
                    requested_cards.append(card)

        if not requested_cards:
            legal_cards = self.hand[:]
        else:
            legal_cards = requested_cards + white_cards

        #  print(f"\ncolor: {requested_color}\n hand: {self.hand}\n legal: {legal_cards}\n")
        card = None
        # Play the only legal card if theres only one
        if len(legal_cards) == 1:
            card = legal_cards[0]

        elif self.player_type == "random":
            card = random.choice(legal_cards)

        elif self.player_type == "heuristic" or self.player_type.startswith("learn"):
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
                                if card_option[0] != trump and card_option[1] - card[1] < 10:
                                    card = card_option
                                    break
        if card is None:
            card = random.choice(legal_cards)

        self.hand.remove(card)
        return card

    def guess_wins(self, max_guesses: int, trump: int, state_space=None) -> int:
        """
        Guess how many tricks the player will win this round
        :param max_guesses: max guesses (amount of tricks in this round)
        :param trump: trump in this round
        :return: None
        """
        if self.player_type == "random":
            self.player_guesses = random.randrange(max_guesses)
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
                    self.player_guesses = np.random.randint(
                        0, self.guess_agent.guess_max
                    )
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
            if self.verbose:
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
