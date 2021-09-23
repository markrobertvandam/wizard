import random
import game


class Player:
    def __init__(self, player_name: str, player_type="random") -> None:
        self.player_name = player_name
        self.hand = []
        self.win_cards = []
        self.guesses = 0
        self.trick_wins = 0
        self.player_type = player_type

    def draw_cards(self, amount: int, deck: list) -> list:
        for card in range(amount):
            self.hand.append(deck.pop())
        return deck

    def play_card(self, trump: int, requested_color: int, played_cards: list) -> tuple:
        requested_cards = []
        white_cards = []
        if requested_color != 4:
            for card in self.hand:
                if card[1] == 0 or card[1] == 14:
                    white_cards.append(card)
                elif card[0] == requested_color:
                    requested_cards.append(card)

        if not requested_cards:
            legal_cards = self.hand[:]
        else:
            legal_cards = requested_cards + white_cards

        # print(f"color: {requested_color}\n hand: {self.hand}\n legal: {legal_cards}")
        card = random.choice(legal_cards)
        if self.player_type == "heuristic":

            # dodge win as high as possible if I am already at my goal
            if self.guesses == self.trick_wins:
                for card_option in sorted(legal_cards, key=lambda x: x[1], reverse=True):
                    if game.Game.trick_winner(
                        played_cards + [card_option] + [(0, 0)] * (2 - len(played_cards)), trump
                    ) != len(played_cards):
                        card = card_option
                        break

            # see if I can win this round with lowest card possible
            elif len(played_cards) == 2:
                for card_option in sorted(legal_cards, key=lambda x: x[1]):
                    if game.Game.trick_winner(
                        played_cards + [card_option], trump
                    ) == 2:
                        card = card_option
                        break

        self.hand.remove(card)
        return card

    def guess_wins(self, max_guesses: int, trump: int) -> None:
        if self.player_type == "random":
            self.guesses = random.randrange(max_guesses)
        else:
            # print("im the smart one")
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
                elif max_guesses < 4 and card[0] == trump and card[1] > 0:
                    guesses += 1
                    self.win_cards.append(card)

                # count non-trump high cards as win except for early rounds
                elif max_guesses > 4 and card[1] > 10:
                    guesses += 1
                    self.win_cards.append(card)

            self.guesses = guesses
