from math import floor
from player import Player

import copy
import random
import numpy as np


class Game:
    def __init__(
        self,
        full_deck: list,
        deck_dict: dict,
        run_type: str,
        shuffled_decks=None,
        shuffled_players=None,
        output_path=None,
        guess_agent=None,
        playing_agent=None,
        epsilon=None,
        player_epsilon=None,
        verbose=False,
        use_agent=False,
    ) -> None:
        self.verbose = verbose
        self.full_deck = full_deck
        self.deck_dict = deck_dict
        self.shuffled_decks = shuffled_decks
        self.deck = None
        self.output_path = output_path
        self.trump = 4  # placeholder trump, only 0-3 exist
        self.game_round = 1
        self.player1 = Player(
            "player1",
            run_type,
            guess_agent,
            playing_agent,
            epsilon,
            player_epsilon,
            verbose,
        )
        self.use_agent = use_agent
        if use_agent:
            guess_agent_fixed = copy.copy(guess_agent)
            playing_agent_fixed = copy.copy(playing_agent)
            self.player2 = Player(
                "player2", "learned", guess_agent_fixed, playing_agent_fixed
            )
            self.player3 = Player(
                "player3", "learned", guess_agent_fixed, playing_agent_fixed
            )
        else:
            self.player2 = Player("player2", "heuristic")
            self.player3 = Player("player3", "heuristic")

        self.players = [
            self.player1,
            self.player2,
            self.player3,
        ]
        new_players = []
        if shuffled_players:
            for player_name in shuffled_players:
                for player in self.players:
                    if player_name == player.player_name:
                        new_players.append(player)
            self.players = new_players
        else:
            random.shuffle(self.players)

        # at the start of the game
        self.scores = {self.player1: 0, self.player2: 0, self.player3: 0}

        # for keeping track of what goes wrong for the learning agent (too high guess, too low guess)
        self.offs = [
            0,
            0,
        ]
        self.off_game = np.zeros(21, dtype=int)
        self.guess_distribution = np.zeros(21, dtype=int)

        # for info per round/trick
        self.played_round = []
        self.played_cards = []
        self.guesses = []

    def play_game(self) -> tuple:
        for game_round in range(20):
            self.played_round = []
            if self.shuffled_decks is None:
                self.deck = self.full_deck[:]
                random.shuffle(self.deck)
            else:
                self.deck = self.shuffled_decks[game_round][:]
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
                if player.player_name == "player1":
                    self.guess_distribution[player.get_guesses()] += 1
            else:
                self.guesses.append(player.guess_wins(self.game_round, self.trump))
            # print("Hand: ", player.hand)
            # print("Player guess: ", player.guesses, "\n")

        # Playing phase
        player_order = self.players[:]  # order will change after every trick
        for trick in range(self.game_round):
            if self.verbose >= 3:
                print(
                    "Player order in regular round: ",
                    [p.player_name for p in player_order],
                )
            self.played_cards = []
            self.play_trick(player_order, 4, 0)
            # print(
            #     f"Order: {[player.player_name for player in player_order]}, Played: {self.played_cards}\n{self.trump}"
            # )
            if self.verbose >= 3:
                print("We made it HERE! Trick was played!")
            winner = self.trick_winner(self.played_cards, self.trump)
            if self.verbose >= 3:
                print(
                    "The winner is: ", winner, self.played_cards, "trump: ", self.trump
                )
            self.played_round.append(self.played_cards)

            if self.verbose >= 2:
                print("Played in actual trick: ", self.played_cards)
                print(
                    "Winner index: ", winner, "name: ", player_order[winner].player_name
                )
            player_order[winner].trick_wins += 1
            player_order = player_order[winner:] + player_order[:winner]

        self.update_scores()

    def play_trick(
        self,
        player_order: list,
        requested_color: int,
        player: int,
        card=None,
        player_limit=None,
    ) -> None:
        """
        plays one entire trick (each player plays 1 card)
        :param player_order: order in which players play
        :param requested_color: the requested color that has to be played if possible
                                (blue, yellow, red, green, None yet, None this round)
        :param player: how manieth player it is in this particular trick
        :return: None
        """
        if self.verbose >= 3:
            print("\nPlaytrick called with card: ", card)
        while player != 3:
            if self.verbose >= 3:
                print(
                    "Trick iteration with player",
                    player,
                    "Name: ",
                    player_order[player].player_name,
                )
                print(
                    "The player order at this moment is: ",
                    [p.player_name for p in player_order],
                )
            if player_limit and (3 > player_limit == player):
                break

            if card is None:
                playing_state = None
                if player_order[player].player_type.startswith("learn"):
                    playing_state = self.playing_state_space(
                        player_order, player_order[player], self.played_cards
                    )
                self.played_cards.append(
                    player_order[player].play_card(
                        self.trump,
                        requested_color,
                        self.played_cards,
                        player_order,
                        self,
                        playing_state,
                    )
                )
            else:
                if self.verbose >= 3:
                    print(
                        "Players and card: ",
                        [p.player_name for p in player_order],
                        card,
                    )
                    print(
                        "Players hand in playtrick with card: ",
                        player_order[player].hand,
                    )
                player_order[player].hand.remove(card)
                self.played_cards.append(card)

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
            card = None
            if self.verbose >= 2:
                print(
                    "finished one iteration of playtrick, chosen card: ",
                    self.played_cards[-1],
                )
        if self.verbose >= 3:
            print("Done with while loop ", player)

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

    def hand_state_space(self, player_order, player):
        if self.player1.guess_agent.input_size == 188 and self.player1.play_agent.input_size == 3915:
            one_hot_hand = 60 * [0]
            one_hot_hand2 = 60 * [0]
            one_hot_hand3 = 60 * [0]

            other_players = [p for p in player_order if p != player]

            cards_in_hand = player.get_hand()
            cards_in_hand2 = other_players[0].get_hand()
            cards_in_hand3 = other_players[1].get_hand()

            for card in cards_in_hand:
                one_hot_hand[self.deck_dict[card]] = 1

            for card in cards_in_hand2:
                one_hot_hand2[self.deck_dict[card]] = 1

            for card in cards_in_hand3:
                one_hot_hand3[self.deck_dict[card]] = 1

            return one_hot_hand + one_hot_hand2 + one_hot_hand3
        else:
            cards_in_hand = player.get_hand()
            one_hot_hand = 60 * [0]
            for card in cards_in_hand:
                one_hot_hand[self.deck_dict[card]] = 1

            return one_hot_hand

    def guessing_state_space(self, player: Player):
        # TODO: maybe add player order?
        state = []
        state += self.hand_state_space(self.players, player)
        trump = [0, 0, 0, 0, 0]
        trump[self.trump] = 1
        previous_guesses = self.guesses[:]
        if len(previous_guesses) >= 2:
            previous_guesses = previous_guesses[:2]
        avg_guess = floor(self.game_round / 3)
        previous_guesses += [avg_guess] * (2 - len(previous_guesses))
        round_number = [self.game_round]
        state += trump + previous_guesses + round_number

        state_space = np.array(state, dtype=int)
        return state_space

    def playing_state_space(
        self, player_order, player: Player, played_trick, temp=False
    ):
        state = []
        inp_size = player.play_agent.input_size
        if self.verbose >= 3:
            if temp:
                print("This is a temp game call!\n")
            else:
                print("This is a real game call!\n")

        state += self.hand_state_space(player_order, player)

        trump = [0, 0, 0, 0, 0]
        trump[self.trump] = 1
        state += trump

        # old system
        if inp_size == 3731:
            previous_guesses = []
        else:
            previous_guesses = [player.get_guesses()]

        round_number = [self.game_round]
        tricks_needed = [player.get_guesses() - player.get_trick_wins()]
        tricks_needed_others = []

        if self.verbose >= 2:
            print("Creating gamespace, players are in the following order: ")
            print([p.player_name for p in self.players], player.player_name)
        for other_player in self.players:
            # print(
            #     f"Player {other_player.player_name}, "
            #     f"guessed {other_player.get_guesses()} "
            #     f"and won {other_player.get_trick_wins()}"
            # )
            if player != other_player:
                tricks = other_player.get_guesses() - other_player.get_trick_wins()
                tricks_needed_others.append(tricks)
                previous_guesses.append(other_player.get_guesses())

        state += previous_guesses + round_number + tricks_needed + tricks_needed_others
        # old system
        if inp_size == 3731:
            played_this_trick = 60 * [0]
            for card in played_trick:
                played_this_trick[self.deck_dict[card]] = 1
            state += played_this_trick
        else:
            played_this_trick = 120 * [0]
            order_names = [int(p.player_name[-1]) for p in player_order]
            for card in played_trick:
                one_hot = self.deck_dict[card]
                offset = played_trick.index(card) * 60
                played_this_trick[one_hot + offset] = 1
            state += order_names + played_this_trick

        if inp_size > 3600:
            # 20 rounds of 3 cards that are one-hot encoded
            played_this_round = 3600 * [0]
            for trick in range(len(self.played_round)):
                trick_plays = self.played_round[trick]
                for turn in range(3):
                    card = trick_plays[turn]
                    one_hot = self.deck_dict[card]
                    played_this_round[one_hot + turn * 60 + trick * 180] = 1
            state += played_this_round

        state_space = np.array(state, dtype=int)
        return state_space

    def update_scores(self):
        for player in self.players:
            #  print(player.player_name, player.trick_wins, player.guesses)
            off_mark = abs(player.get_trick_wins() - player.get_guesses())
            if player.player_name == "player1":
                if off_mark > 19 or player.get_guesses() > 19:
                    print(
                        player.get_guesses(), player.get_trick_wins(), self.game_round
                    )
                self.off_game[off_mark] += 1
            if off_mark == 0:
                self.scores[player] += 20 + 10 * player.get_guesses()
                if player.player_type == "learning":
                    player.update_agent(100)
                    player.play_agent.backpropagate(
                        player.play_agent.last_terminal_node, 100
                    )
            else:
                if player.player_type == "learning":
                    player.update_agent(0)
                    player.play_agent.backpropagate(
                        player.play_agent.last_terminal_node, 0
                    )
                self.scores[player] -= 10 * off_mark
                if player.player_name == "player1":
                    if self.verbose >= 2:
                        print(
                            "player_won: ",
                            player.get_trick_wins(),
                            "player_guessed",
                            player.get_guesses(),
                        )
                    if player.get_guesses() > player.get_trick_wins():
                        self.offs[0] += 1
                    else:
                        self.offs[1] += 1
        #  print("\n\n")

    def get_game_performance(self):
        return self.off_game

    def get_distribution(self):
        return self.guess_distribution

    def get_output_path(self):
        return self.output_path

    def wrap_up_trick(self, player_order):
        winner = self.trick_winner(self.played_cards, self.trump)
        self.played_round.append(self.played_cards)
        player_order[winner].trick_wins += 1
        self.played_cards = []
        return winner

    def play_till_player(self, player_order: list, player_limit: int):
        winner = self.wrap_up_trick(player_order)
        player_order = player_order[winner:] + player_order[:winner]
        self.play_trick(player_order, 4, 0, player_limit=player_limit)
