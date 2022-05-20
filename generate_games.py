import argparse
import os
import pickle
import random


def parse_args() -> argparse.Namespace:
    """
    Function to parse arguments.
    Returns:
    parser: Argument parser containing arguments.
    """

    parser = argparse.ArgumentParser(description="Generate n-games")
    parser.add_argument("games", help="How many games to generate", type=int)
    parser.add_argument(
        "folder", help="Where to save decks and player orders", type=str
    )

    return parser.parse_args()


def write_decks(n) -> list:
    """
    generate 20 * n shuffled decks and save them in a pickle file
    :param n: n amount of wizard games
    :return: list of 20*n shuffled decks
    """
    full_deck = []
    for suit in range(4):  # (blue, yellow, red, green)
        for card_value in range(15):  # (joker, 1-13, wizard)
            full_deck.append((suit, card_value))

    shuffled_decks = []
    for i in range(20 * n):
        shuffled_decks.append(random.sample(full_deck, 60))

    return shuffled_decks


def write_orders(n):
    """
    generate n shuffled player orders and save them in a pickle file
    :param n: n amount of wizard games
    :return: list of n shuffled player orders
    """
    player_names = ["player1", "player2", "player3"]
    shuffled_players = []
    for i in range(n):
        shuffled_players.append(random.sample(player_names, 3))

    return shuffled_players


def main():
    args = parse_args()
    decks = write_decks(args.games)
    players = write_orders(args.games)
    if not os.path.exists(args.folder):
        os.mkdir(args.folder)
    with open(os.path.join(args.folder, "decks.pkl"), "wb+") as f:
        pickle.dump(decks, f)
    with open(os.path.join(args.folder, "players.pkl"), "wb+") as f:
        pickle.dump(players, f)
    # decks = pickle.load(open("test/decks.pkl", "rb"))
    # players = pickle.load(open("test/players.pkl", "rb"))
    # print(len(decks), len(players))
    # print(len(decks[0]), decks[10])
    # print(len(players[0]), players[10])


if __name__ == "__main__":
    main()
