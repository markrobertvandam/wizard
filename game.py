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
        output_path=None,
        guess_agent=None,
        playing_agent=None,
        epsilon=None,
        verbose=False,
        use_agent=False,
    ) -> None:
        self.verbose = verbose
        self.full_deck = full_deck
        self.deck_dict = deck_dict
        self.deck = []
        self.output_path = output_path
        self.trump = 4  # placeholder trump, only 0-3 exist
        self.game_round = 1
        self.player1 = Player(
            "player1", run_type, guess_agent, playing_agent, epsilon, verbose
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
        random.shuffle(self.players)

        # at the start of the game
        self.scores = {self.player1: 0, self.player2: 0, self.player3: 0}

        # for keeping track of what goes wrong for the learning agent
        self.offs = [
            0,
            0,
        ]
        self.off_game = np.zeros(21, dtype=int)

        # for info per round/trick
        self.played_round = []
        self.played_cards = []
        self.guesses = []

    def play_game(self) -> tuple:
        for game_round in range(20):
            self.played_round = []
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
            if self.verbose == 3:
                print(
                    "Player order in regular round: ",
                    [p.player_name for p in player_order],
                )
            self.played_cards = []
            self.play_trick(player_order, 4, 0)
            # print(
            #     f"Order: {[player.player_name for player in player_order]}, Played: {self.played_cards}\n{self.trump}"
            # )
            if self.verbose == 3:
                print("We made it HERE! Trick was played!")
            winner = self.trick_winner(self.played_cards, self.trump)
            if self.verbose == 3:
                print(
                    "The winner is: ", winner, self.played_cards, "trump: ", self.trump
                )
            self.played_round.append(self.played_cards)

            if self.verbose == 2:
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
        if self.verbose == 3:
            print("\nPlaytrick called with card: ", card)
        while player != 3:
            if self.verbose == 3:
                print("Trick iteration with player", player, "Name: ", player_order[player].player_name)
                print("The player order at this moment is: ", [p.player_name for p in player_order])
            if player_order[player] == player_limit:
                break
            playing_state = self.playing_state_space(
                player_order[player], self.played_cards
            )
            if card is None:
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
                if self.verbose == 3:
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
            if self.verbose == 2:
                print(
                    "finished one iteration of playtrick, chosen card: ",
                    self.played_cards[-1],
                )
        if self.verbose == 3:
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

    def guessing_state_space(self, player: Player):
        # TODO: maybe add player order?
        cards_in_hand = player.get_hand()
        one_hot_hand = np.zeros(60, dtype=int)
        for card in cards_in_hand:
            one_hot_hand[self.deck_dict[card]] = 1
        trump = [0, 0, 0, 0, 0]
        trump[self.trump] = 1
        previous_guesses = self.guesses[:]
        if len(previous_guesses) >= 2:
            previous_guesses = previous_guesses[:2]
        previous_guesses += [21] * (2 - len(previous_guesses))
        round_number = [self.game_round]
        state_space = np.concatenate(
            (one_hot_hand, trump, previous_guesses, round_number)
        )

        return state_space.astype(int)

    def playing_state_space(self, player: Player, played_trick, temp=False):
        # TODO: maybe remove guesses from state and add player order?
        if self.verbose == 3:
            if temp:
                print("This is a temp game call!\n")
            else:
                print("This is a real game call!\n")
        cards_in_hand = player.get_hand()
        one_hot_hand = np.zeros(60, dtype=int)
        for card in cards_in_hand:
            one_hot_hand[self.deck_dict[card]] = 1
        trump = [0, 0, 0, 0, 0]
        trump[self.trump] = 1
        previous_guesses = self.guesses[:]
        if len(previous_guesses) >= 2:
            previous_guesses = previous_guesses[:2]
        previous_guesses += [21] * (2 - len(previous_guesses))
        round_number = [self.game_round]

        tricks_needed = [player.get_guesses() - player.get_trick_wins()]
        tricks_needed_others = []

        if self.verbose > 2:
            print("Creating gamespace, players are in the following order: ")
            print([p.player_name for p in self.players], player.player_name)
        for other_player in self.players:
            if player != other_player:
                tricks = other_player.get_guesses() - other_player.get_trick_wins()
                tricks_needed_others.append(tricks)

        played_this_trick = np.zeros(60, dtype=int)
        for card in played_trick:
            played_this_trick[self.deck_dict[card]] = 1

        # 20 rounds of 3 cards that are one-hot encoded
        played_this_round = np.zeros(3600, dtype=int)
        for trick in range(len(self.played_round)):
            trick_plays = self.played_round[trick]
            for turn in range(3):
                card = trick_plays[turn]
                one_hot = self.deck_dict[card]
                played_this_round[one_hot + turn * 60 + trick * 180] = 1

        state_space = np.concatenate(
            (
                one_hot_hand,
                trump,
                previous_guesses,
                round_number,
                tricks_needed,
                tricks_needed_others,
                played_this_trick,
                played_this_round,
            )
        ).astype(int)

        return state_space

    def update_scores(self):
        for player in self.players:
            #  print(player.player_name, player.trick_wins, player.guesses)
            off_mark = abs(player.get_trick_wins() - player.get_guesses())
            if player.player_name == "player1":
                if off_mark > 19 or player.get_guesses() > 19:
                    print(player.get_guesses(), player.get_trick_wins(), self.game_round)
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
                    if self.verbose == 2:
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

    def get_output_path(self):
        return self.output_path

    def wrap_up_round(self, player_order):
        winner = self.trick_winner(self.played_cards, self.trump)
        self.played_round.append(self.played_cards)
        player_order[winner].trick_wins += 1
        self.played_cards = []
        return winner

    def play_till_player(self, player_order: list, player_limit):
        limit = player_order[player_limit]
        winner = self.wrap_up_round(player_order)
        player_order = player_order[winner:] + player_order[:winner]
        self.play_trick(player_order, 4, 0, player_limit=limit)
