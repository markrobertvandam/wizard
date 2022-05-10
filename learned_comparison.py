import argparse
import game
import matplotlib.pyplot as plt
import os
import random
import time
import tensorflow as tf
from Guessing_Agent import GuessingAgent
from Playing_Agent import PlayingAgent


def parse_args() -> argparse.Namespace:
    """
    Function to parse arguments.
    Returns:
    parser: Argument parser containing arguments.
    """

    parser = argparse.ArgumentParser(description="Run n-games")
    parser.add_argument("games", help="How many games to run", type=int)
    parser.add_argument(
        "--save_folder",
        help="folder name for plots",
        default="",
    )
    parser.add_argument(
        "guess_models",
        help="argument to load in the weights of saved guessing model(s)",
    )
    parser.add_argument(
        "play_models",
        help="argument to load in the weights of saved player model(s)",
    )
    parser.add_argument(
        "--verbose",
        help="optional argument to set how verbose the run is",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--use_agent",
        help="optional argument to set whether opponents should be fixed agents",
        default=0,
        type=bool,
    )

    return parser.parse_args()


def plot_accuracy(
    accuracy_history: list, game_instance: int, save_folder: str, iters_done: int
) -> None:
    plt.plot(list(range(iters_done + 10, game_instance + 1, 10)), accuracy_history)
    plt.xlabel("Games", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.savefig(f"plots/{save_folder}/accuracy_plot")
    plt.close()


def print_performance(agent_pair, score_counter, win_counter, total_offs):
    print("Agents: ", agent_pair)
    print("Scores: ", score_counter)
    print("Wins: ", win_counter)
    print("Mistakes: ", total_offs)


def learned_n_games(
    n: int,
    save_folder: str,
    guessing_models: list,
    player_models: list,
    verbose: int,
    use_agent: bool,
) -> None:

    input_sizes = {"cheater": (68, 3915), "porder": (68, 3795), "old": (68, 3731)}

    # Make the deck
    full_deck = []
    deck_dict = {}

    for suit in range(4):  # (blue, yellow, red, green)
        for card_value in range(15):  # (joker, 1-13, wizard)
            full_deck.append((suit, card_value))

            # to go from card to index
            deck_dict[(suit, card_value)] = card_value + suit * 15

    agent_pairs = []
    for guessing_model in guessing_models:
        for playing_model in player_models:
            input_size_guess, input_size_play = input_sizes[
                guessing_model.split("_")[-1]
            ]
            guess_agent = GuessingAgent(input_size=input_size_guess, guess_max=21)
            playing_agent = PlayingAgent(input_size=input_size_play, verbose=verbose)

            if guessing_model != "random":
                guess_agent.model = tf.keras.models.load_model(
                    os.path.join("models", guessing_model)
                )
                guess_agent.trained = True

            if playing_model != "random":
                playing_agent.network_policy.model = tf.keras.models.load_model(
                    os.path.join("models", playing_model)
                )
                playing_agent.trained = True

            agent_pairs.append((guess_agent, playing_agent))

    performance_dict = dict()
    for pair in agent_pairs:
        # win_counter, score_counter, total_offs(too high guess, too low guess), last_ten, accuracy_hist
        performance_dict[pair] = ([0, 0, 0], [0, 0, 0], [0, 0], 21 * [0], [])

    for game_instance in range(1, n + 1):
        shuffled_deck = full_deck[:]
        random.shuffle(shuffled_deck)
        for pair in agent_pairs:
            (guess_agent, playing_agent) = pair
            off_game, scores, offs = play_game(
                full_deck,
                deck_dict,
                shuffled_deck,
                guess_agent,
                playing_agent,
                verbose,
                use_agent,
            )

            # For command-line output
            (
                win_counter,
                score_counter,
                total_offs,
                last_ten_performance,
                accuracy_hist,
            ) = performance_dict[pair]
            last_ten_performance += off_game
            for player in range(3):
                score_counter[player] += scores[player] / n
                if scores[player] == max(scores):
                    win_counter[player] += 1
            total_offs[0] += offs[0]
            total_offs[1] += offs[1]

            if game_instance % 10 == 0:
                accuracy = last_ten_performance[0] / 200
                accuracy_hist.append(accuracy)
                print(
                    f"Agents: {(guess_agent, playing_agent)}, "
                    f"Game {game_instance}, accuracy: {accuracy}"
                    f"Last10: {last_ten_performance}"
                )
                last_ten_performance *= 0


def play_game(
    full_deck,
    deck_dict,
    shuffled_deck,
    guess_agent,
    play_agent,
    verbose,
    use_agent,
):

    wizard = game.Game(
        full_deck,
        deck_dict,
        "learned",
        shuffled_deck=shuffled_deck,
        output_path="state_err1",
        guess_agent=guess_agent,
        playing_agent=play_agent,
        verbose=verbose,
        use_agent=use_agent,
    )
    scores, offs = wizard.play_game()
    off_game = wizard.get_game_performance()
    return off_game, scores, offs


if __name__ == "__main__":
    args = parse_args()
    learned_n_games(
        600,
        args.save_folder,
        args.guess_models,
        args.play_models,
        args.verbose,
        args.use_agent,
    )
