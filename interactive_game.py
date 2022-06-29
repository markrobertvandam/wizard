from math import floor

from player import Player
from utility_functions import trick_winner, str_to_card, card_to_str
import numpy as np


class Game:
    def __init__(
        self,
        deck_dict: dict,
        guess_type: str,
        player_type: str,
        game_round: int,
        shuffled_players=None,
        guess_agent=None,
        playing_agent=None,
    ) -> None:

        self.deck_dict = deck_dict
        self.deck = None
        self.trump = 4  # placeholder trump, only 0-3 exist
        self.game_round = game_round
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

        # for playing state
        self.possible_cards_one = [1] * 60
        self.possible_cards_two = [1] * 60

    def play_game(self) -> None:
        """
        Plays a single game of wizard
        :return: tuple with all scores and player1 mistakes
        """
        for game_round in range(20):
            self.played_round = []
            self.play_round()
            self.possible_cards_one = [1] * 60
            self.possible_cards_two = [1] * 60
            self.game_round += 1
            self.players = self.players[1:] + self.players[:1]  # Rotate player order
            for player in self.players:  # reset trick wins
                player.trick_wins = 0
                player.win_cards = []

    def play_round(self) -> None:
        """
        Play a single round of wizard
        :return: None
        """
        print(f"\nPlaying round {self.game_round}, order: {[p.player_name for p in self.players]}")
        # Player hand
        hand = set(input("Cards in hand? (seperated by space): ").split())
        if len(hand) != self.game_round:
            while len(hand) != self.game_round:
                print(f"Hand needs to have {self.game_round} cards")
                hand = input("Cards in hand? (seperated by space): ").split()
        converted_hand = []
        for card in hand:
            converted_hand.append(str_to_card(card, converted_hand))
        converted_hand.sort(key=lambda x: (x[0], x[1]))
        print(f"Player hand: {converted_hand}")
        self.player1.hand = converted_hand

        if self.game_round < 20:
            # Trump card becomes top card after hands are dealt
            suits = {"none": 4, "blue": 0, "yellow": 1, "red": 2, "green": 3, "b": 0, "y": 1, "r": 2, "g": 3, "n": 4}
            trump = input("What is the trump suit?: ")
            if trump not in suits:
                while trump not in suits:
                    print(f"Trump needs to be b(lue), y(ellow) r(ed), g(reen) or n(one)")
                    trump = input("What is the trump suit?: ")
            self.trump = suits[trump]
        else:
            # No trump card in final round
            self.trump = 4

        # Guessing phase
        self.guesses = []  # reset guesses every round
        for player in self.players:
            guess_state = None
            if player.player_name == "player1":
                if player.guess_type.startswith("learn"):
                    guess_state = self.guessing_state_space(player)
                self.guesses.append(
                    player.guess_wins(
                        self.game_round, self.trump, guess_state
                    )
                )
                print(f"Learning player guessed: {self.guesses[-1]}")
            else:
                while True:
                    try:
                        guess = int(input(f"Guess made by player {player.player_name}?: "))
                        break
                    except ValueError:
                        print("Please enter a valid number from 0-20")
                        continue
                self.guesses.append(int(guess))
                player.player_guesses = guess

        # Playing phase
        player_order = self.players[:]  # order will change after every trick
        for trick in range(self.game_round):
            self.played_cards = []
            winner_index, player_order = self.play_trick(player_order, 4, 0)

    def play_trick(
        self,
        player_order: list,
        requested_color: int,
        player_index: int,
    ) -> tuple:
        """
        plays one entire trick (each player plays 1 card)
        :param player_order: order in which players play
        :param requested_color: the requested color that has to be played if possible
                                (blue, yellow, red, green, None yet, None this round)
        :param player_index: how manieth player it is in this particular trick
        :return: None
        """
        while player_index != 3:
            if player_order[player_index].player_name == "player1":
                playing_state = None
                if self.player1.player_type.startswith("learn"):
                    playing_state = self.playing_state_space(
                        player_order, player_order[player_index], self.played_cards
                    )
                card, legal_cards = player_order[player_index].play_card(
                                        self.trump,
                                        requested_color,
                                        self.played_cards,
                                        player_order,
                                        self,
                                        playing_state,
                                    )
                self.played_cards.append(card)
                print(f"Learning player plays: {card_to_str(card)}")
            else:
                card = input(f"What card is played by {player_order[player_index].player_name}: ")
                card = str_to_card(card)
                self.played_cards.append(card)

            self.update_possible_hands(card, requested_color, player_order, player_index)

            if requested_color == 4:
                # Wizard means no requested color this round
                if self.played_cards[player_index][1] == 14:
                    requested_color = 5

                # Joker does not change requested color
                elif self.played_cards[player_index][1] != 0:
                    requested_color = self.played_cards[player_index][0]

            player_index += 1

            if self.player1.player_type.startswith("learn") and self.player1.play_agent.input_size == 313:
                move = self.deck_dict[card]
                self.possible_cards_one[move] = 0
                self.possible_cards_two[move] = 0

        winner_index, player_order = self.wrap_up_trick(player_order)
        return winner_index, player_order

    def update_possible_hands(self, card, requested_color, player_order, player):
        # card is not a white card, yet it is not requested color either
        if 0 < card[1] < 14 and requested_color < 4 and requested_color != card[0]:
            for i in range(1 + 15 * requested_color, 15 * (requested_color + 1) - 1):
                if player_order[player].player_name == "player2":
                    self.possible_cards_one[i] = 0
                elif player_order[player].player_name == "player3":
                    self.possible_cards_two[i] = 0

    def hand_state_space(self, player_order: list, player: Player, called: str) -> list:
        """
        returns state space representing the hand(s) of players
        :param player_order: list with the players in turn order
        :param player: the player for which state is retrieved
        :param called: whether it is a guessing or playing state space
        :return:
        """
        if (called == "guess" and player.guess_agent.input_size == 188) or \
           (called == "play" and (player.play_agent.input_size == 3915 or
                                  player.play_agent.input_size == 315)):
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

            if inp_size % 100 == 93 or inp_size % 100 == 13:
                players_turn = player_order.index(player)
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

        if inp_size == 313:
            # TODO: add 60 for each other player to determine which cards they might have
            state += self.possible_cards_one + self.possible_cards_two

        state_space = np.array(state, dtype=int)
        if len(state) != inp_size:
            print(f"Len state is {len(state)}, len inp is {inp_size}")
            print("Input size is wrong")
            exit()
        return state_space

    def wrap_up_trick(self, player_order: list) -> tuple:
        """
        helper function for when a trick finished
        :param player_order: list of players in turn order
        :return: winner of wrapped up trick
        """
        winner_index = trick_winner(self.played_cards, self.trump)
        self.played_round.append(self.played_cards)
        player_order[winner_index].trick_wins += 1

        print(f"Trick won by {winner_index}th player. Winner name: {player_order[winner_index].player_name}")
        print(f"Tricks still needed by me: {self.player1.get_guesses() - self.player1.get_trick_wins()}")

        self.played_cards = []
        player_order = player_order[winner_index:] + player_order[:winner_index]
        return winner_index, player_order
