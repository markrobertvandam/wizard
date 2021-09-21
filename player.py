import random


class Player:
    def __init__(self, player_type="random") -> None:
        self.hand = []
        self.score = 0
        self.guesses = 0
        self.player_type = player_type

    def draw_cards(self, amount: int, deck: list) -> list:
        for card in range(amount):
            self.hand.append(deck.pop())
        return deck

    def play_card(self) -> None:
        pass

    def guess_wins(self, max_guesses: int) -> None:
        if self.player_type == "random":
            self.guesses = random.randrange(max_guesses)
