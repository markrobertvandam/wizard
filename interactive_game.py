from math import floor

from player import Player
import numpy as np


class Game:
    def __init__(
        self,
        deck_dict: dict,
        guess_type: str,
        player_type: str,
        shuffled_players=None,
        guess_agent=None,
        playing_agent=None,
    ) -> None:

        self.deck_dict = deck_dict
        self.deck = None
        self.trump = 4  # placeholder trump, only 0-3 exist
        self.game_round = 1
        self.player1 = Player(
            "player1",
            deck_dict,
            guess_type,
            player_type,
            guess_agent,
            playing_agent,
        )

        self.player2 = Player("player2", deck_dict, "real", "real")
        self.player3 = Player("player3", deck_dict, "real", "real")

        self.players = [
            self.player1,
            self.player2,
            self.player3,
        ]
        temp_players = []
        for player_name in shuffled_players:
            for player in self.players:
                if player_name == player.player_name:
                    temp_players.append(player)
        self.players = temp_players

        # for info per round/trick
        self.played_round = []
        self.played_cards = []
        self.guesses = []

    def play_game(self) -> None:
        """
        Plays a single game of wizard
        :return: tuple with all scores and player1 mistakes
        """
        for game_round in range(20):
            self.played_round = []
            self.play_round()
            self.game_round += 1
            self.players = self.players[1:] + self.players[:1]  # Rotate player order
            for player in self.players:  # reset trick wins
                player.trick_wins = 0
                player.win_cards = []
        print(self.player1.play_agent.pred_counter)

    def play_round(self) -> None:
        """
        Play a single round of wizard
        :return: None
        """
        # Player hand
        hand = input("Cards in hand? (seperated by space): ")
        self.player1.hand = hand.split()

        if self.game_round < 20:
            # Trump card becomes top card after hands are dealt
            self.trump = input("What is the trump suit?: ")
        else:
            # No trump card in final round
            self.trump = 4

        # Guessing phase
        self.guesses = []  # reset guesses every round
        for player in self.players:
            if player.player_name == "player1":
                self.guesses.append(
                    player.guess_wins(
                        self.game_round, self.trump, self.guessing_state_space(player)
                    )
                )
            else:
                guess = int(input(f"Guess made by player {player.player_name}?: "))
                self.guesses.append(guess)
                player.player_guesses = guess

        # Playing phase
        player_order = self.players[:]  # order will change after every trick
        for trick in range(self.game_round):
            self.played_cards = []
            self.play_trick(player_order, 4, 0)

            winner = self.trick_winner(self.played_cards, self.trump)
            self.played_round.append(self.played_cards)

            player_order[winner].trick_wins += 1
            player_order = player_order[winner:] + player_order[:winner]

    def play_trick(
        self,
        player_order: list,
        requested_color: int,
        player: int,
    ) -> None:
        """
        plays one entire trick (each player plays 1 card)
        :param player_order: order in which players play
        :param requested_color: the requested color that has to be played if possible
                                (blue, yellow, red, green, None yet, None this round)
        :param player: how manieth player it is in this particular trick
        :return: None
        """
        while player != 3:
            if player_order[player] == "player1":
                playing_state = self.playing_state_space(
                    player_order, player_order[player], self.played_cards
                )
                self.played_cards.append(
                    self.player1.play_card(
                        self.trump,
                        requested_color,
                        self.played_cards,
                        player_order,
                        self,
                        playing_state,
                    )
                )
            else:
                card = input(f"What card is played by {player_order[player]}: ")
                self.played_cards.append(tuple(card))

            if requested_color == 4:

                # Wizard means no requested color this round
                if self.played_cards[player][1] == 14:
                    requested_color = 5

                # Joker does not change requested color
                elif self.played_cards[player][1] != 0:
                    requested_color = self.played_cards[player][0]

            player += 1

    @staticmethod
    def trick_winner(played_cards: list, trump: int) -> int:
        """
        Determine the winner of a trick
        :param played_cards: cards played in the trick
        :param trump: trump-suit, used to determine the winner
        :return: index of the winning card
        """
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

    def hand_state_space(self, player_order: list, player: Player, called: str) -> list:
        """
        returns state space representing the hand(s) of players
        :param player_order: list with the players in turn order
        :param player: the player for which state is retrieved
        :param called: whether it is a guessing or playing state space
        :return:
        """
        if (called == "guess" and player.guess_agent.input_size == 188) or \
           (called == "play" and player.play_agent.input_size % 100 == 15):
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

    def guessing_state_space(self, player: Player) -> np.ndarray:
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

        state_space = np.array(state, dtype=int)
        return state_space

    def playing_state_space(
        self, player_order: list, player: Player, played_trick: list, temp=False
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

            if inp_size % 100 == 93:
                players_turn = self.players.index(player)
                state += [players_turn]
            elif inp_size % 100 == 95:
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

        state_space = np.array(state, dtype=int)
        if len(state) != inp_size:
            print("Input size is wrong")
            exit()
        return state_space
