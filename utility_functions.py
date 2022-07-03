import copy
import numpy as np

import game
import interactive_game
from scipy.sparse import coo_matrix


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


def key_to_state(input_size: int, node_state: tuple) -> np.ndarray:
    split = int(len(node_state) / 3)
    sparse_state = (
        coo_matrix(
            (
                node_state[:split],
                (node_state[split: split * 2], node_state[split * 2:]),
            )
        )
        .toarray()[0]
        .astype("float32")
    )
    sparse_state = np.pad(
        sparse_state, (0, input_size - len(sparse_state)), "constant"
    )
    return sparse_state


def state_to_key(state_space: np.ndarray) -> tuple:
    compressed_state = coo_matrix(state_space)
    key_state = tuple(
        np.concatenate(
            (compressed_state.data, compressed_state.row, compressed_state.col)
        )
    )
    return key_state


def str_to_card(card: str, hand=None) -> tuple:
    # cards are suit {0-3} (blue, yellow, red, green), value {0-14} (0 == joker, 14 == wizard)
    try:
        suit, value = card.split("_")
    except ValueError:
        suit, value = input("Please provide a suit and value, like 'y(ellow)_7': ").split("_")
    suits = {"blue": 0, "yellow": 1, "red": 2, "green": 3, "b": 0, "y": 1, "r": 2, "g": 3}
    if suit not in suits:
        while suit not in suits:
            if hand is not None:
                print(f"Hand so far: {hand}, wrong card: {card}")
            print(f"Suit needs to be b(lue), y(ellow) r(ed) or g(reen)")
            suit, value = input(f"What card: ").split("_")
    if value not in [str(i) for i in range(15)]:
        while value not in [str(i) for i in range(15)]:
            if hand is not None:
                print(f"Hand so far: {hand}, wrong card: {card}")
            print(f"Card value needs to be somewhere from 0-14")
            suit, value = input(f"What card: ").split("_")

    return suits[suit], int(value)


def card_to_str(card: tuple) -> str:
    # cards are suit {0-3} (blue, yellow, red, green), value {0-14} (0 == joker, 14 == wizard)
    try:
        suit, value = card
    except (TypeError, ValueError) as error:
        print(f"Obtained {card} in card_to_str, which is type {type(card)}. This caused {error}")
        exit()
    suits = {0: "blue", 1: "yellow", 2: "red", 3: "green"}
    return " ".join([suits[suit], str(value)])


def temp_game(game_instance, played_cards, interactive=False):
    """
    returns a temporary copy of game_instance to simulate different plays
    :param game_instance: the game to copy
    :param played_cards: cards played in trick so far
    :return:
    """
    old_guess_agent = None
    old_epsilon = None

    old_play_agent = None
    old_play_epsilon = None

    if game_instance.player1.guess_type.startswith("learn"):
        old_guess_agent = game_instance.player1.guess_agent
        old_epsilon = game_instance.player1.epsilon
    if game_instance.player1.player_type.startswith("learn"):
        old_play_agent = game_instance.player1.play_agent
        old_play_epsilon = game_instance.player1.player_epsilon

    old_guess_agent2 = None
    old_play_agent2 = None
    old_guess_agent3 = None
    old_play_agent3 = None
    if game_instance.use_agent:
        old_guess_agent2 = game_instance.player2.guess_agent
        old_play_agent2 = game_instance.player2.play_agent
        old_guess_agent3 = game_instance.player3.guess_agent
        old_play_agent3 = game_instance.player3.play_agent

    if not interactive:
        temp_instance = game.Game(
            full_deck=copy.deepcopy(game_instance.full_deck),
            deck_dict=copy.deepcopy(game_instance.deck_dict),
            guess_type=copy.deepcopy(game_instance.player1.guess_type),
            player_type=copy.deepcopy(game_instance.player1.player_type),
            guess_agent=copy.copy(old_guess_agent),
            playing_agent=copy.copy(old_play_agent),
            epsilon=copy.deepcopy(old_epsilon),
            player_epsilon=copy.deepcopy(old_play_epsilon),
            verbose=copy.deepcopy(game_instance.player1.verbose),
            use_agent=copy.deepcopy(game_instance.use_agent),
            guess_agent2=copy.copy(old_guess_agent2),
            playing_agent2=copy.copy(old_play_agent2),
            guess_agent3=copy.copy(old_guess_agent3),
            playing_agent3=copy.copy(old_play_agent3),
        )
        temp_instance.output_path = copy.deepcopy(game_instance.output_path)
        temp_instance.game_round = copy.deepcopy(game_instance.game_round)
    else:
        temp_instance = interactive_game.Game(
            deck_dict=copy.deepcopy(game_instance.deck_dict),
            guess_type=copy.deepcopy(game_instance.player1.guess_type),
            game_round=copy.deepcopy(game_instance.game_round),
            player_type=copy.deepcopy(game_instance.player1.player_type),
            guess_agent=copy.copy(old_guess_agent),
            playing_agent=copy.copy(old_play_agent),
        )

    temp_instance.deck = copy.deepcopy(game_instance.deck)
    temp_instance.trump = copy.deepcopy(game_instance.trump)
    temp_instance.played_cards = copy.deepcopy(played_cards)
    temp_instance.played_round = copy.deepcopy(game_instance.played_round)
    temp_instance.guesses = copy.deepcopy(game_instance.guesses)

    temp_instance.player1.hand = copy.deepcopy(game_instance.player1.hand)
    temp_instance.player2.hand = copy.deepcopy(game_instance.player2.hand)
    temp_instance.player3.hand = copy.deepcopy(game_instance.player3.hand)

    temp_instance.player1.player_guesses = copy.deepcopy(
        game_instance.player1.player_guesses
    )
    temp_instance.player2.player_guesses = copy.deepcopy(
        game_instance.player2.player_guesses
    )
    temp_instance.player3.player_guesses = copy.deepcopy(
        game_instance.player3.player_guesses
    )

    temp_instance.player1.trick_wins = copy.deepcopy(game_instance.player1.trick_wins)
    temp_instance.player2.trick_wins = copy.deepcopy(game_instance.player2.trick_wins)
    temp_instance.player3.trick_wins = copy.deepcopy(game_instance.player3.trick_wins)

    temp_instance.possible_cards_one = copy.deepcopy(game_instance.possible_cards_one)
    temp_instance.possible_cards_two = copy.deepcopy(game_instance.possible_cards_two)

    return temp_instance
