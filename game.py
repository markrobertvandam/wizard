from math import floor

import player as player_class
import numpy as np
import os
import random
import utility_functions as util


class Game:
    def __init__(
        self,
        full_deck: list,
        deck_dict: dict,
        guess_type: str,
        player_type: str,
        shuffled_decks=None,
        shuffled_players=None,
        output_path=None,
        guess_agent=None,
        playing_agent=None,
        epsilon=None,
        player_epsilon=None,
        verbose=False,
        opp_guesstype="heuristic",
        opp_playertype="heuristic",
        guess_agent2=None,
        playing_agent2=None,
        guess_agent3=None,
        playing_agent3=None,
        save_folder="",
        reoccur_bool=False,
    ) -> None:
        self.verbose = verbose
        self.full_deck = full_deck
        self.deck_dict = deck_dict
        self.shuffled_decks = shuffled_decks
        self.deck = None
        self.output_path = output_path
        self.trump = 4  # placeholder trump, only 0-3 exist
        self.game_round = 1
        self.player1 = player_class.Player(
            "player1",
            guess_type,
            player_type,
            guess_agent,
            playing_agent,
            epsilon,
            player_epsilon,
            verbose,
            reoccur_path=os.path.join("reoccur", save_folder),
            reoccur_bool=reoccur_bool,
        )
        self.opp_guesstype = opp_guesstype
        self.opp_playertype = opp_playertype
        self.player2 = player_class.Player(
            "player2",
            opp_guesstype,
            opp_playertype,
            guess_agent2,
            playing_agent2,
            epsilon,
            player_epsilon,
            verbose,
        )
        self.player3 = player_class.Player(
            "player3",
            opp_guesstype,
            opp_playertype,
            guess_agent3,
            playing_agent3,
            epsilon,
            player_epsilon,
            verbose,
        )

        self.players = [
            self.player1,
            self.player2,
            self.player3,
        ]
        temp_players = []
        if shuffled_players:
            for player_name in shuffled_players:
                for player in self.players:
                    if player_name == player.player_name:
                        temp_players.append(player)
            self.players = temp_players
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
        self.round_offs = np.zeros(20, dtype=int)

        self.guess_distribution = np.zeros(21, dtype=int)
        self.actual_distribution = np.zeros(21, dtype=int)
        self.overshoot = np.zeros(20, dtype=int)
        self.total_loss = 0

        # for info per round/trick
        self.played_round = []
        self.played_cards = []
        self.guesses = []

    def play_game(self) -> tuple:
        """
        Plays a single game of wizard
        :return: tuple with all scores and player1 mistakes
        """
        for game_round in range(20):
            self.played_round = []
            if self.shuffled_decks is None:
                self.deck = self.full_deck[:]
                random.shuffle(self.deck)
            else:
                self.deck = self.shuffled_decks[game_round][:]
            if self.verbose >= 1:
                print(
                    f"\nInitial player order at start of round {self.game_round}: "
                    f"{[p.player_name for p in self.players]}")
            self.play_round()
            if self.verbose >= 2:
                print(f"Round {self.game_round} over.. \n\n")

            for player in self.players:
                player.possible_cards_one = [1] * 60
                player.possible_cards_two = [1] * 60

            self.game_round += 1
            self.players = self.players[1:] + self.players[:1]  # Rotate player order

            for player in self.players:  # reset trick wins
                player.trick_wins = 0
                player.win_cards = []

        return (
            self.total_loss,
            [
                self.scores[self.player1],
                self.scores[self.player2],
                self.scores[self.player3],
            ],
            self.offs,
            self.round_offs,
        )

    def play_round(self) -> None:
        """
        Play a single round of wizard
        :return: None
        """
        # Players get dealt their hands
        for player in self.players:
            self.deck = player.draw_cards(self.game_round, self.deck)

        if self.game_round < 20:
            # Trump card becomes top card after hands are dealt
            trump_card = self.deck.pop()
            if trump_card[1] == 0:
                # trump is a joker, no trump this round
                self.trump = 4
            elif trump_card[1] == 14:
                # trump is a wizard, starting player decides
                self.trump = trump_card[0]
            else:
                # trump is regular card
                self.trump = trump_card[0]
            if self.verbose >= 2:
                print(f"Trump card: {trump_card}")
            for player in self.players:
                player.possible_cards_one[self.deck_dict[trump_card]] = 0
                player.possible_cards_two[self.deck_dict[trump_card]] = 0
        else:
            # No trump card in final round
            self.trump = 4

        # Guessing phase
        self.guesses = []  # reset guesses every round
        for player in self.players:
            if player.guess_type.startswith("learn"):
                self.guesses.append(
                    player.guess_wins(
                        self.game_round, self.trump, self.guessing_state_space(player)
                    )
                )
            else:
                self.guesses.append(player.guess_wins(self.game_round, self.trump))
            if player.player_name == "player1":
                self.guess_distribution[player.get_guesses()] += 1

                # keep track of guesses higher than possible, 0 if not impossible
                overshot = max(0, player.get_guesses() - self.game_round)
                self.overshoot[overshot] += 1
                if self.verbose:
                    if player.get_guesses() > self.game_round:
                        print(f"Guess overshot, player guessed {player.get_guesses()} in round {self.game_round}")
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

            for player in self.players:
                if player.player_type.startswith("learn") and player.play_agent.input_size in [313, 315]:
                    # TODO: set possible cards to invert of one_hot_hand
                    # normal player only sees their own hand
                    cards_in_hand = player.get_hand()
                    for card in cards_in_hand:
                        move = self.deck_dict[card]
                        player.possible_cards_one[move] = 0
                        player.possible_cards_two[move] = 0

            if self.verbose >= 2:
                print(f"player order before changing: {[p.player_name for p in player_order]}")
            winner_index, player_order = self.play_trick(player_order, 4, 0)

            if self.verbose >= 2:
                print(f"player order after changing: {[p.player_name for p in player_order]}")
                print(f"We made it HERE! Trick was played!, winner is: {winner_index}")

        self.update_scores()

    def play_trick(
        self,
        player_order: list,
        requested_color: int,
        player_index: int,
        card=None,
        player_limit=None,
        temp=False,
    ) -> tuple:
        """
        plays one entire trick (each player plays 1 card)
        :param player_order: order in which players play
        :param requested_color: the requested color that has to be played if possible
                                (blue, yellow, red, green, None yet, None this round)
        :param player_index: how manieth player it is in this particular trick
        :param card: used to force the player to play that specific card, used in simulation
        :param player_limit: also used for simulation to play on until the players next turn
        :return: None
        """
        if self.verbose >= 3:
            print("\nPlaytrick called with card: ", card)
        while player_index != 3:
            player = player_order[player_index]
            if self.verbose >= 2:
                print(f"Temp in play_trick: {temp}")
                print(
                    "Trick iteration with player",
                    player_index,
                    "Name: ",
                    player.player_name,
                )
                print(
                    "The player order at this moment is: ",
                    [p.player_name for p in player_order],
                )
            if player_limit and (3 > player_limit == player_index):
                return None, None

            if card is None:
                playing_state = None
                if player.player_type.startswith("learn"):
                    playing_state = self.playing_state_space(
                        player_order, player, self.played_cards
                    )
                self.played_cards.append(
                    player.play_card(
                        self.trump,
                        requested_color,
                        self.played_cards,
                        player_order,
                        self,
                        playing_state,
                    )
                )
                self.update_possible_hands(self.played_cards[-1], requested_color, player_order, player_index)
                for player in self.players:
                    if player.player_type.startswith("learn") and player.play_agent.input_size in [313, 315]:
                        # TODO: set possible cards to invert of one_hot_hand
                        # normal player only sees their own hand
                        move = self.deck_dict[self.played_cards[-1]]
                        player.possible_cards_one[move] = 0
                        player.possible_cards_two[move] = 0
            else:
                # play the passed card to simulate that child-node
                if self.verbose >= 3:
                    print(
                        "Players and card: ",
                        [p.player_name for p in player_order],
                        card,
                    )
                    print(
                        "Players hand in playtrick with card: ",
                        player_order[player_index].hand,
                    )
                player_order[player_index].hand.remove(card)
                self.played_cards.append(card)

            if requested_color == 4:

                # Wizard means no requested color this round
                if self.played_cards[player_index][1] == 14:
                    requested_color = 5

                # Joker does not change requested color
                elif self.played_cards[player_index][1] != 0:
                    requested_color = self.played_cards[player_index][0]

            player_index += 1
            card = None
            if self.verbose >= 2:
                print(
                    "finished one iteration of playtrick, chosen card: ",
                    self.played_cards[-1],
                )
        if self.verbose >= 3:
            print("Done with while loop ", player_index)

        return self.wrap_up_trick(player_order, temp=temp)

    def update_possible_hands(self, card, requested_color, player_order, player):
        # card is not a white card, yet it is not requested color either
        if 0 < card[1] < 14 and requested_color < 4 and requested_color != card[0]:
            if self.verbose >= 2:
                print(f"{player} did not follow suit, they played {card} while requested color was {requested_color}")
                print(f"Player order is {[p.player_name for p in player_order]}")
            for i in range(1 + 15 * requested_color, 15 * (requested_color + 1) - 1):
                if player_order[player].player_name == "player1":
                    # update player2 and player3 knowledge on player1
                    self.player2.possible_cards_one[i] = 0
                    self.player3.possible_cards_one[i] = 0
                elif player_order[player].player_name == "player2":
                    # update player1 and player3 knowledge about player2
                    self.player1.possible_cards_one[i] = 0
                    self.player3.possible_cards_two[i] = 0
                elif player_order[player].player_name == "player3":
                    # update player1 and player2 knowledge about player3
                    self.player1.possible_cards_two[i] = 0
                    self.player2.possible_cards_two[i] = 0

    def hand_state_space(self, player_order: list, player, called: str) -> list:
        """
        returns state space representing the hand(s) of players
        :param player_order: list with the players in turn order
        :param player: the player for which state is retrieved
        :param called: whether it is a guessing or playing state space
        :return:
        """
        if (called == "guess" and player.guess_agent.input_size == 188) or \
           (called == "play" and False):
            # Cheating player sees all three hands
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
            # normal player only sees their own hand
            cards_in_hand = player.get_hand()
            one_hot_hand = 60 * [0]
            for card in cards_in_hand:
                one_hot_hand[self.deck_dict[card]] = 1

            return one_hot_hand

    def guessing_state_space(self, player) -> np.ndarray:
        # TODO: maybe add player order?
        """
        Obtain the state space used to predict during the guessing phase
        :param player: Player that needs the state space to make a guess
        :return: guessing state space
        """
        state = []
        state += self.hand_state_space(self.players, player, "guess")
        trump = [0, 0, 0, 0, 0]
        trump[self.trump] = 1
        previous_guesses = self.guesses[:]
        if len(previous_guesses) >= 2:
            previous_guesses = previous_guesses[:2]
        avg_guess = floor(self.game_round / 3)
        previous_guesses += [avg_guess] * (2 - len(previous_guesses))
        round_number = [self.game_round]
        state += trump + previous_guesses + round_number

        if player.guess_agent.input_size == 69:
            players_turn = self.players.index(player)
            state += [players_turn]

        if player.guess_agent.input_size == 71:
            order_names = [int(p.player_name[-1]) for p in self.players]
            state += order_names

        state_space = np.array(state, dtype=int)
        return state_space

    def playing_state_space(
        self, player_order: list, player, played_trick: list, temp=False
    ) -> np.ndarray:
        """
        Obtain the state space used by the playing agent to make a move
        :param player_order: list of players in turn order
        :param player: player that needs to make a move using the state space
        :param played_trick: Cards played in the current trick thus far
        :param temp: boolean whether this is for the actual game or simulated
        :return: playing state space
        """
        state = []
        inp_size = player.play_agent.input_size
        if self.verbose >= 2:
            if temp:
                print("This is a temp game call of playing_state!\n")
            else:
                print("This is a real game call of playing_state!\n")

        state += self.hand_state_space(player_order, player, "play")

        trump = [0, 0, 0, 0, 0]
        trump[self.trump] = 1
        state += trump

        if inp_size == 3731:
            # old system, only uses other players' guesses
            previous_guesses = []
        else:
            previous_guesses = [player.get_guesses()]

        round_number = [self.game_round]
        tricks_needed = [player.get_guesses() - player.get_trick_wins()]
        tricks_needed_others = []

        if self.verbose >= 2:
            print("Creating gamespace, players are in the following order: ")
            print([p.player_name for p in player_order], player.player_name)
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
        if inp_size % 100 == 31:
            # old system, played_trick is unordered
            played_this_trick = 60 * [0]
            for card in played_trick:
                played_this_trick[self.deck_dict[card]] = 1
            state += played_this_trick
        else:
            # played trick is ordered in order of play
            played_this_trick = 120 * [0]

            if inp_size % 100 == 93 or inp_size == 313:
                players_turn = player_order.index(player)
                state += [players_turn]
            elif inp_size % 100 == 95 or inp_size == 315:
                order_names = [int(p.player_name[-1]) for p in player_order]
                state += order_names

            for card in played_trick:
                one_hot = self.deck_dict[card]
                offset = played_trick.index(card) * 60
                played_this_trick[one_hot + offset] = 1
            state += played_this_trick

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

        if inp_size in [313, 315]:
            state += player.possible_cards_one + player.possible_cards_two

        state_space = np.array(state, dtype=int)
        if len(state) != inp_size:
            print(f"Input size is wrong, state is length {len(state)} while inp_size is {inp_size}")
            exit()
        return state_space

    def update_scores(self) -> None:
        """
        update scores and backpropagate for learning
        :return: None
        """
        scores = []
        offs = []
        for player in self.players:
            #  get the difference between tricks won and tricks guessed
            off_mark = abs(player.get_trick_wins() - player.get_guesses())
            offs.append(off_mark)
            if off_mark == 0:
                scores.append(20 + 10 * player.get_guesses())
            else:
                scores.append(-10 * off_mark)

        for player_index in range(3):
            # TODO: get diff between learning players' score and other highest score and maximize that
            player = self.players[player_index]
            off_mark = offs[player_index]
            score = scores[player_index]

            if player.player_name == "player1":
                self.actual_distribution[player.get_trick_wins()] += 1
                if off_mark > 0:
                    self.round_offs[self.game_round-1] += 1
                if off_mark > 19 or player.get_guesses() > 19:
                    print(
                        player.get_guesses(), player.get_trick_wins(), self.game_round
                    )
                self.off_game[off_mark] += 1

            # Update scores and memory of agents
            self.scores[player] += score
            if player.guess_type == "learning":
                player.update_agent()
            if player.player_type == "learning":
                other_scores = scores[:player_index] + scores[player_index + 1:]
                diff = score - max(other_scores)
                result = 1 - off_mark
                loss = player.play_agent.backpropagate(
                    player.play_agent.last_terminal_node, result=result, diff=diff, score=score
                )
                self.total_loss += loss

            # For keeping track of mistakes
            if player.player_name == "player1":
                if self.verbose >= 1:
                    print(
                        "player_won: ",
                        player.get_trick_wins(),
                        "player_guessed",
                        player.get_guesses(),
                    )
                if player.get_guesses() > player.get_trick_wins():
                    self.offs[0] += 1
                elif player.get_guesses() < player.get_trick_wins():
                    self.offs[1] += 1
        #  print("\n\n")

    def get_game_performance(self) -> np.ndarray:
        return self.off_game

    def get_distribution(self) -> tuple:
        return self.guess_distribution, self.actual_distribution

    def get_overshoot(self) -> np.ndarray:
        return self.overshoot

    def get_output_path(self) -> str:
        return self.output_path

    def wrap_up_trick(self, player_order: list, temp=False) -> tuple:
        """
        helper function for when a trick finished
        :param player_order: list of players in turn order
        :return: winner of wrapped up trick
        """
        winner_index = util.trick_winner(self.played_cards, self.trump)
        self.played_round.append(self.played_cards)
        player_order[winner_index].trick_wins += 1
        if self.verbose >= 1:
            if temp:
                print("This is a temp call: \n")
            print(
                f"\nOrder: {[player.player_name for player in player_order]}, "
                f"\nPlayed: {self.played_cards}"
                f"\n{self.trump}, "
                f"\nWinner: {winner_index}\n"
            )
        self.played_cards = []
        player_order = player_order[winner_index:] + player_order[:winner_index]
        return winner_index, player_order

    # def play_till_player(self, player_order: list, player_limit: int):
    #     winner = self.wrap_up_trick(player_order)
    #     player_order = player_order[winner:] + player_order[:winner]
    #     self.play_trick(player_order, 4, 0, player_limit=player_limit)
