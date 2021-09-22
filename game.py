from player import Player
import random


class Game:
    def __init__(self) -> None:
        self.full_deck = []
        self.deck = []
        self.trump = 4  # placeholder trump, only 0-3 exist
        self.game_round = 1
        self.player1 = Player("heuristic")
        self.player2 = Player()
        self.player3 = Player()
        self.players = [self.player1, self.player2, self.player3]
        self.played_cards = []

        for card_value in range(15):  # (joker, 1-13, wizard)
            for suit in range(4):  # (blue, yellow, red, green)
                self.full_deck.append((suit, card_value))

    def play_round(self) -> None:

        print("Deck length: ", len(self.deck), len(self.full_deck))
        for player in self.players:
            self.deck = player.draw_cards(self.game_round, self.deck)

        if self.game_round < 20:  # No trump card in final round
            self.trump = self.deck.pop()[0]
            print("Trump: ", self.trump)
        else:
            self.trump = 4

        # Guessing phase
        for player in self.players:
            player.guess_wins(self.game_round, self.trump)
            print("Hand: ", player.hand)
            print("Player guess: ", player.guesses, "\n")

        # Playing phase
        for trick in range(self.game_round):
            self.played_cards = []
            self.played_cards.append(self.players[0].play_card(self.trump, 4))
            requested_color = self.played_cards[0][0]
            for player in self.players[1:]:
                self.played_cards.append(player.play_card(self.trump, requested_color))
            self.players[self.trick_winner()].trick_wins += 1

        print(
            f"Round {self.game_round}: ",
            self.player1.trick_wins - self.player1.guesses,
            self.player2.trick_wins - self.player2.guesses,
            self.player3.trick_wins - self.player3.guesses,
        )

    def play_game(self) -> None:
        for game_round in range(10):
            self.deck = self.full_deck[:]
            random.shuffle(self.deck)
            self.play_round()
            self.game_round += 1
            self.players = self.players[1:] + self.players[:1]  # Rotate player order
            for player in self.players:  # reset trick wins
                player.trick_wins = 0

    def trick_winner(self) -> int:
        strongest_card = 0
        if self.played_cards[0][1] == 14:  # If first player played a wizard
            return 0

        for i in range(1, 3):

            if self.played_cards[i][1] == 14:  # If i-th player played a wizard
                return i

            # if i-th card is trump and strongest card is not
            if (
                self.played_cards[i][0] == self.trump
                and self.played_cards[strongest_card][0] != self.trump
            ):
                if self.played_cards[i][1] > 0:  # joker does not count as trump card
                    strongest_card = i

            # if cards are the same suit
            if self.played_cards[i][0] == self.played_cards[strongest_card][0]:
                if self.played_cards[i][1] > self.played_cards[strongest_card][1]:
                    strongest_card = i

        return strongest_card
