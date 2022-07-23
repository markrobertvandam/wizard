import numpy as np

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
    suit, value = card.split("_")
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
    suit, value = card
    suits = {0: "blue", 1: "yellow", 2: "red", 3: "green"}
    return " ".join([suits[suit], str(value)])


def write_state(play_state: np.ndarray, output_path: str, input_size: int, type_node="After play") -> None:
    """
    Write playing state to text file for debugging
    :param play_state: actual playing state
    :param output_path: path to textfile
    :param input_size: input size of the playing model
    :param type_node: type of node reached in play
    :return:
    """
    f = open(f"{output_path}.txt", "a")
    f.write("\n\n\n")
    np.set_printoptions(threshold=np.inf)
    f.write(f"{type_node} node\n")
    f.write("Hand: " + str(np.nonzero(play_state[:60])[0].tolist()) + "\n")
    current_pos = 60
    # # if cheater
    # if input_size % 100 == 15 or input_size % 100 == 13:
    #     f.write("Hand2: " + str(np.nonzero(play_state[60:120])[0].tolist()) + "\n")
    #     f.write("Hand3: " + str(np.nonzero(play_state[120:180])[0].tolist()) + "\n")
    #     current_pos = 180
    f.write("Trump: " + str(play_state[current_pos: current_pos + 5]) + "\n")
    current_pos += 5

    # if old
    if input_size % 100 == 31:
        f.write("Guesses: " + str(play_state[current_pos: current_pos + 2]) + "\n")
        current_pos += 2
    else:
        f.write("Guesses: " + str(play_state[current_pos: current_pos + 3]) + "\n")
        current_pos += 3
    f.write("Round: " + str(play_state[current_pos]) + "\n")
    f.write("Tricks needed: " + str(play_state[current_pos + 1]) + "\n")
    current_pos += 2

    f.write(
        "Tricks needed others: " + str(play_state[current_pos: current_pos + 2]) + "\n"
    )
    current_pos += 2

    # if not olds
    if input_size % 100 == 92:
        f.write(
            "played trick: "
            + str(
                np.nonzero(play_state[current_pos: current_pos + 120])[0].tolist()
            )
            + "\n"
        )
        current_pos += 120
    elif input_size % 100 == 93 or input_size == 313:
        f.write("Order: " + str(play_state[current_pos]) + "\n")
        f.write(
            "played trick: "
            + str(
                np.nonzero(play_state[current_pos + 1: current_pos + 121])[0].tolist()
            )
            + "\n"
        )
        current_pos += 121
    elif input_size % 100 == 95 or input_size == 315:
        f.write("Order: " + str(play_state[current_pos: current_pos + 3]) + "\n")
        f.write(
            "played trick: "
            + str(
                np.nonzero(play_state[current_pos + 3: current_pos + 123])[0].tolist()
            )
            + "\n"
        )
        current_pos += 123
    elif input_size % 100 == 31:
        f.write(
            "played trick: "
            + str(np.nonzero(play_state[current_pos: current_pos + 60])[0].tolist())
            + "\n"
        )
        current_pos += 60

    # if not small
    if input_size > 3600:
        f.write(
            "played round: "
            + str(np.nonzero(play_state[current_pos:])[0].tolist())
            + "\n"
        )

    if input_size in [313, 315]:
        f.write(
            "possible cards one: "
            + str(np.nonzero(play_state[current_pos:current_pos+60])[0].tolist())
            + "\n"
        )
        f.write(
            "possible cards two: "
            + str(np.nonzero(play_state[current_pos+60:])[0].tolist())
            + "\n"
        )
    f.close()
