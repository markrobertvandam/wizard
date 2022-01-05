from player import Player
import random


class Game:
    def __init__(self) -> None:
        self.full_deck = []
        self.deck = []
        self.trump = 4  # placeholder trump, only 0-3 exist
        self.game_round = 1
        self.player1 = Player("player1", "heuristic")
        self.player2 = Player("player2")
        self.player3 = Player("player3")
        self.players = [
            self.player1,
            self.player2,
            self.player3,
        ]
        # at the start of the game
        self.scores = {self.player1: 0, self.player2: 0, self.player3: 0}
        self.offs = [
            0,
            0,
        ]  # for keeping track of what goes wrong for the learning agent
        self.played_cards = []

        # Make the deck
        for card_value in range(15):  # (joker, 1-13, wizard)
            for suit in range(4):  # (blue, yellow, red, green)
                self.full_deck.append((suit, card_value))

    def play_round(self) -> None:

        # Players get dealt their hands
        for player in self.players:
            self.deck = player.draw_cards(self.game_round, self.deck)

        if self.game_round < 20:
            # Trump card becomes top card after hands are dealt
            self.trump = self.deck.pop()[0]
        else:
            # No trump card in final round
            self.trump = 4

        # Guessing phase
        for player in self.players:
            player.guess_wins(self.game_round, self.trump)
            # print("Hand: ", player.hand)
            # print("Player guess: ", player.guesses, "\n")

        # Playing phase
        player_order = self.players[:]  # order will change after every trick
        for trick in range(self.game_round):
            self.played_cards = []
            self.play_trick(player_order, 4, 0)
            # print(
            #     f"Order: {[player.player_name for player in player_order]}, Played: {self.played_cards}\n{self.trump}"
            # )
            winner = self.trick_winner(self.played_cards, self.trump)
            player_order[winner].trick_wins += 1
            player_order = player_order[winner:] + player_order[:winner]

        self.update_scores()

    def play_trick(self, player_order: list, requested_color: int, player: int) -> None:
        """
        plays one entire trick (each player plays 1 card)
        :param player_order: order in which players play
        :param requested_color: the requested color that has to be played if possible
                                (blue, yellow, red, green, None yet, None this round)
        :param player: how manieth player it is in this particular trick
        :return: None
        """
        while player != 3:
            self.played_cards.append(
                player_order[player].play_card(
                    self.trump, requested_color, self.played_cards
                )
            )
            if requested_color == 4:
                # Joker and Wizard do not change requested color
                if (
                    self.played_cards[player][1] != 0
                    and self.played_cards[player][1] != 14
                ):
                    requested_color = self.played_cards[player][0]

                # Wizard means no requested color this round
                elif self.played_cards[player][1] == 14:
                    requested_color = 5

            player += 1

    def play_game(self) -> tuple[list[int], list[int]]:
        for game_round in range(10):
            self.deck = self.full_deck[:]
            random.shuffle(self.deck)
            self.play_round()
            self.game_round += 1
            self.players = self.players[1:] + self.players[:1]  # Rotate player order
            for player in self.players:  # reset trick wins
                player.trick_wins = 0
                player.win_cards = []

        return (
            [
                self.scores[self.player1],
                self.scores[self.player2],
                self.scores[self.player3],
            ],
            self.offs,
        )

    @staticmethod
    def trick_winner(played_cards: list, trump: int) -> int:
        strongest_card = 0
        if played_cards[0][1] == 14:  # If first player played a wizard
            return 0

        for i in range(1, 3):

            if played_cards[i][1] == 14:  # If i-th player played a wizard
                return i

            # if i-th card is trump and strongest card is not
            if played_cards[i][0] == trump and played_cards[strongest_card][0] != trump:
                if played_cards[i][1] > 0:  # joker does not count as trump card
                    strongest_card = i

            # if cards are the same suit
            if played_cards[i][0] == played_cards[strongest_card][0]:
                if played_cards[i][1] > played_cards[strongest_card][1]:
                    strongest_card = i

            # if strongest card is a joker and i-th card is not
            if played_cards[strongest_card][1] == 0 and played_cards[i][1] != 0:
                strongest_card = i

        return strongest_card

    def update_scores(self):
        for player in self.players:
            #  print(player.player_name, player.trick_wins, player.guesses)
            off_mark = abs(player.trick_wins - player.guesses)
            if off_mark == 0:
                self.scores[player] += 20 + 10 * player.guesses
            else:
                self.scores[player] -= 10 * off_mark
                if player.player_name == "player1":
                    if player.guesses > player.trick_wins:
                        self.offs[0] += 1
                    else:
                        self.offs[1] += 1
        #  print("\n\n")
