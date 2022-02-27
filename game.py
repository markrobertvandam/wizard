from player import Player
import random
import numpy as np


class Game:
    def __init__(
        self,
        full_deck,
        guess_agent1=None,
        epsilon=None,
        playing_agent1=None,
        verbose=False,
    ) -> None:
        self.verbose = verbose
        self.full_deck = full_deck
        self.deck = []
        self.trump = 4  # placeholder trump, only 0-3 exist
        self.game_round = 1
        self.player1 = Player("player1", "learned", guess_agent1, epsilon, verbose)
        self.player2 = Player("player2", "heuristic")
        self.player3 = Player("player3", "heuristic")
        self.players = [
            self.player1,
            self.player2,
            self.player3,
        ]
        # at the start of the game
        self.scores = {self.player1: 0, self.player2: 0, self.player3: 0}

        # for keeping track of what goes wrong for the learning agent
        self.offs = [
            0,
            0,
        ]
        self.off_game = np.zeros(20)
        # for info per trick
        self.played_cards = []
        self.guesses = []

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
        self.guesses = []  # reset guesses every round
        for player in self.players:
            if player.player_type.startswith("learn"):
                self.guesses.append(
                    player.guess_wins(
                        self.game_round, self.trump, self.guessing_state_space(player)
                    )
                )
            else:
                self.guesses.append(player.guess_wins(self.game_round, self.trump))
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
        for game_round in range(20):
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
            off_mark = abs(player.get_trick_wins() - player.get_guesses())
            if player.player_name == "player1":
                self.off_game[off_mark] += 1
            if off_mark == 0:
                self.scores[player] += 20 + 10 * player.get_guesses()
                if player.player_type == "learning":
                    player.update_agent(1000)
            else:
                if player.player_type == "learning":
                    player.update_agent(-200 * off_mark**3)
                self.scores[player] -= 10 * off_mark
                if player.player_name == "player1":
                    if self.verbose:
                        print("Off-mark: ", off_mark)
                    if player.get_guesses() > player.get_trick_wins():
                        self.offs[0] += 1
                    else:
                        self.offs[1] += 1
        #  print("\n\n")

    def guessing_state_space(self, player: Player):
        cards_in_hand = np.hstack(player.get_hand())
        cards_in_hand = np.concatenate(
            (cards_in_hand, [4, 15] * int((20 - len(cards_in_hand) / 2)))
        )
        trump = [0, 0, 0, 0, 0]
        trump[self.trump] = 1
        previous_guesses = self.guesses[:]
        previous_guesses += [21] * (2 - len(previous_guesses))
        round_number = [self.game_round]
        state_space = np.concatenate(
            (cards_in_hand, trump, previous_guesses, round_number)
        )
        return state_space

    def get_game_performance(self):
        return self.off_game
