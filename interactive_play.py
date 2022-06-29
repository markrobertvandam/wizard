import argparse
import os
import tensorflow as tf
import interactive_game

from Guessing_Agent import GuessingAgent
from Playing_Agent import PlayingAgent


def parse_args() -> argparse.Namespace:
    """
    Function to parse arguments.
    Returns:
    parser: Argument parser containing arguments.
    """

    parser = argparse.ArgumentParser(description="Run interactive game")
    parser.add_argument("guesser", help="Which guessing model", type=str)
    parser.add_argument("player", help="Which playing model", type=str)
    parser.add_argument(
        "guesser_input",
        help="argument to set guess inp_size",
        type=int,
    )
    parser.add_argument(
        "player_input",
        help="argument to set player inp_size",
        type=int,
    )
    parser.add_argument(
        "game_round",
        help="argument to set what around to start at",
        type=int,
    )
    parser.add_argument(
        "shuffled_players",
        nargs="+",
        help="argument to set player order",
        type=str,
    )
    parser.add_argument("--dueling", action="store_true",
                        help="use dueling DQN")

    return parser.parse_args()


def int_game(
    guessing_model: str,
    playing_model: str,
    guess_inp_size: int,
    player_inp_size: int,
    game_round: int,
    shuffled_players: list,
    dueling: bool,
) -> None:

    # Make the deck
    full_deck = []
    deck_dict = {}

    for suit in range(4):  # (blue, yellow, red, green)
        for card_value in range(15):  # (joker, 1-13, wizard)
            full_deck.append((suit, card_value))

            # to go from card to index
            deck_dict[(suit, card_value)] = card_value + suit * 15

    print("Pair: ", guessing_model, playing_model)
    guess_agent = GuessingAgent(input_size=guess_inp_size, guess_max=21)
    playing_agent = PlayingAgent(input_size=player_inp_size, dueling=dueling)

    if guessing_model == "heuristic" or guessing_model == "random":
        guess_type = guessing_model

    else:
        if guessing_model != "none":
            guess_agent.model = tf.keras.models.load_model(
                os.path.join("models", guessing_model)
            )
        guess_type = "learned"

    if playing_model == "heuristic" or playing_model == "random":
        player_type = playing_model

    else:
        if playing_model != "none":
            playing_agent.network_policy.model = tf.keras.models.load_model(
                os.path.join("models", playing_model)
            )
        player_type = "learned"

    print(guess_agent.model.summary())
    print(f"Guesser loss-function: ", guess_agent.model.loss)
    print(playing_agent.network_policy.model.summary())
    print(f"Player loss-function: ", playing_agent.network_policy.model.loss)

    wizard = interactive_game.Game(
        deck_dict,
        guess_type=guess_type,
        player_type=player_type,
        game_round=game_round,
        shuffled_players=shuffled_players,
        guess_agent=guess_agent,
        playing_agent=playing_agent,
    )
    wizard.play_game()


if __name__ == "__main__":
    args = parse_args()
    print(f"Dueling: {args.dueling}")
    print(f"Shuffled players: {args.shuffled_players}")
    int_game(
        args.guesser,
        args.player,
        args.guesser_input,
        args.player_input,
        args.game_round,
        args.shuffled_players,
        args.dueling,
    )
