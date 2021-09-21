from player import Player


class Game:
    def __init__(self) -> None:
        self.deck = []
        self.trump = ()
        self.game_round = 1
        for card_value in range(15):
            for suit in range(4):
                self.deck.append((suit, card_value))
        self.player1 = Player("heuristic")
        self.player2 = Player()
        self.player3 = Player()
        self.players = [self.player1, self.player2, self.player3]

    def play_round(self) -> None:
        for player in self.players:
            self.deck = player.draw_cards(self.game_round, self.deck)
        self.trump = self.deck.pop()
        for player in self.players:
            player.guess_wins(self.game_round)
        print(self.deck)

    def play_game(self) -> None:
        for game_round in range(20):
            self.play_round()
            self.game_round += 1
            self.players = self.players[1:] + self.players[:1]  # Rotate player order
