import random


class Player:
    def __init__(self, player_type="random") -> None:
        self.hand = []
        self.guesses = 0
        self.trick_wins = 0
        self.player_type = player_type

    def draw_cards(self, amount: int, deck: list) -> list:
        for card in range(amount):
            self.hand.append(deck.pop())
        return deck

    def play_card(self, trump: int, requested_color: int) -> tuple:
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

        print(f"color: {requested_color}\n hand: {self.hand}\n legal: {legal_cards}")
        if self.player_type == "random":
            card = random.choice(legal_cards)
        else:
            card = random.choice(legal_cards)
        self.hand.remove(card)
        return card

    def guess_wins(self, max_guesses: int, trump: int) -> None:
        if self.player_type == "random":
            self.guesses = random.randrange(max_guesses)
        else:
            print("im the smart one")
            guesses = 0
            for card in self.hand:
                if (card[0] == trump and (card[1] > 6 or max_guesses < 3)) or card[
                    1
                ] > 10:
                    guesses += 1
            self.guesses = guesses
