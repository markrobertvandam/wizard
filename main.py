import argparse
import game
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from Guessing_Agent import GuessingAgent
from Playing_Agent import PlayingAgent
from tensorflow.python.client import device_lib


def parse_args() -> argparse.Namespace:
    """
    Function to parse arguments.
    Returns:
    parser: Argument parser containing arguments.
    """

    parser = argparse.ArgumentParser(description="Run n-games")
    parser.add_argument("games", help="How many games to run", type=int)
    parser.add_argument(
        "runtype",
        help="type of agent for player 1 (random, heuristic, learning, learned)",
    )
    parser.add_argument(
        "save", help="argument to determine whether model should be saved when learning"
    )
    parser.add_argument(
        "--save_folder",
        help="folder name for plots and models for this run",
        default="1",
    )
    parser.add_argument(
        "--verbose",
        help="optional argument to set how verbose the run is",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--model", help="optional argument to load in the weights of a saved model"
    )
    parser.add_argument(
        "--play_model",
        help="optional argument to load in the weights of a saved player model",
    )
    parser.add_argument(
        "--use_agent",
        help="optional argument to set whether opponents should be fixed agents",
        default=0,
        type=bool,
    )
    parser.add_argument(
        "--epsilon",
        help="optional argument to set starting epsilon",
        default=1,
        type=float,
    )
    parser.add_argument(
        "--iters_done",
        help="optional argument to set iters done in previous run",
        default=0,
        type=int,
    )

    return parser.parse_args()


def plot_accuracy(accuracy_history, game_instance, save_folder, iters_done):
    plt.plot(list(range(iters_done + 10, game_instance + 1, 10)), accuracy_history)
    plt.xlabel("Games", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.savefig(f"plots/{save_folder}/accuracy_plot")
    plt.close()


def avg_n_games(
    n,
    run_type,
    save_bool,
    save_folder,
    model_path,
    player_model,
    verbose,
    use_agent,
    epsilon,
    iters_done,
):
    input_size = 68
    guess_agent = GuessingAgent(input_size=input_size, guess_max=21)
    playing_agent = PlayingAgent(verbose=verbose)
    if model_path is not None:
        guess_agent.model = tf.keras.models.load_model(
            os.path.join("models", model_path)
        )
    if player_model is not None:
        playing_agent.model = tf.keras.models.load_model(
            os.path.join("models", player_model)
        )

    # Exploration settings
    epsilon_decay = 0.997
    min_epsilon = 0.02

    # For keeping track of performance
    win_counter = [0, 0, 0]
    score_counter = [0, 0, 0]
    total_offs = [0, 0]

    # Make the deck
    full_deck = []
    deck_dict = {}

    for suit in range(4):  # (blue, yellow, red, green)
        for card_value in range(15):  # (joker, 1-13, wizard)
            full_deck.append((suit, card_value))

            # to go from card to index
            deck_dict[(suit, card_value)] = card_value + suit * 15

    # Run n-amount of games
    last_ten_performance = np.zeros(21, dtype=int)
    accuracy_history = []
    last_max = iters_done
    max_acc = 0
    for game_instance in range(1 + iters_done, n + 1 + iters_done):
        if verbose:
            print("\nGame instance: ", game_instance)
        wizard = game.Game(
            full_deck,
            deck_dict,
            run_type,
            guess_agent,
            playing_agent,
            epsilon,
            verbose=verbose,
            use_agent=use_agent,
        )
        scores, offs = wizard.play_game()

        # For command-line output while training
        last_ten_performance += wizard.get_game_performance()
        for player in range(3):
            score_counter[player] += scores[player] / n
            if scores[player] == max(scores):
                win_counter[player] += 1
        total_offs[0] += offs[0]
        total_offs[1] += offs[1]

        if game_instance % 10 == 0:
            accuracy = last_ten_performance[0] / 200
            guess_agent.accuracy = accuracy
            accuracy_history.append(accuracy)
            print(
                f"Game {game_instance}, accuracy: {accuracy}, Epsilon: {round(epsilon,2)}, "
                f"Last10: {last_ten_performance}"
            )
            last_ten_performance *= 0

            # for early stopping
            if accuracy > max_acc and run_type == "learning":
                last_max = game_instance
                max_acc = accuracy

        if run_type == "learning":

            if game_instance % 1000 == 0:
                time_label = str(int(time.time() / 3600))[-3:]
                plot_accuracy(accuracy_history, game_instance, save_folder, iters_done)
                if save_bool.startswith("y"):
                    guess_agent.model.save(
                        f"models/{save_folder}/guessing{input_size}_"
                        f"{time_label}_"
                        f"{accuracy}_"
                        f"{game_instance}.model"
                    )

            if game_instance - last_max > 10000:
                if save_bool.startswith("y"):
                    time_label = str(int(time.time() / 3600))[-3:]
                    plot_accuracy(
                        accuracy_history, game_instance, save_folder, iters_done
                    )
                    guess_agent.model.save(
                        f"models/{save_folder}/guessing{input_size}_"
                        f"{time_label}_"
                        f"{accuracy}_"
                        f"{game_instance}.model"
                    )
                break

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            epsilon = max(min_epsilon, epsilon)

        if game_instance % 1500 == 0:
            print("Forgetting ", len(wizard.player1.play_agent.nodes.keys()), " nodes..")
            wizard.player1.play_agent.nodes = dict()

    print("Scores: ", score_counter)
    print("Wins: ", win_counter)
    print("Mistakes: ", total_offs)


if __name__ == "__main__":
    args = parse_args()
    avg_n_games(
        args.games,
        args.runtype,
        args.save,
        args.save_folder,
        args.model,
        args.play_model,
        args.verbose,
        args.use_agent,
        args.epsilon,
        args.iters_done,
    )
