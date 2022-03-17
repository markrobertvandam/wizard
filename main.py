import argparse
import game
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from Guessing_Agent import GuessingAgent
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
        "--verbose",
        help="optional argument to set how verbose the run is",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--model", help="optional argument to load in the weights of a saved model"
    )

    return parser.parse_args()


def plot_accuracy(accuracy_history, game_instance, time_label):
    plt.plot(list(range(10, game_instance + 1, 10)), accuracy_history)
    plt.xlabel("Games", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.savefig(f"plots/accuracy_plot{time_label}")
    plt.close()


def avg_n_games(n, run_type, save_bool, model_path, verbose):
    input_size = 68
    guess_agent = GuessingAgent(input_size=input_size, guess_max=20)
    if model_path is not None:
        guess_agent.model = tf.keras.models.load_model(
            os.path.join("models", model_path)
        )

    # Exploration settings
    epsilon = 1  # not a constant, going to be decayed
    epsilon_decay = 0.9985
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

            # to go from card to index for one-hot
            deck_dict[(suit, card_value)] = card_value + suit * 15

    # Run n-amount of games
    last_ten_performance = np.zeros(20)
    accuracy_history = []
    last_max = 0
    max_acc = 0
    for game_instance in range(1, n + 1):
        wizard = game.Game(
            full_deck, deck_dict, run_type, guess_agent, epsilon, verbose=verbose
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

            if game_instance % 2000 == 0:
                time_label = str(int(time.time() / 3600))[-3:]
                plot_accuracy(accuracy_history, game_instance, time_label)
                if save_bool.startswith("y"):
                    guess_agent.model.save(
                        f"models/guessing{input_size}_"
                        f"{time_label}_"
                        f"{round(guess_agent.avg_reward, 2)}_"
                        f"{game_instance}.model"
                    )

            if game_instance - last_max > 6000:
                if save_bool.startswith("y"):
                    guess_agent.model.save(
                        f"models/guessing{input_size}_"
                        f"{time_label}_"
                        f"{round(guess_agent.avg_reward, 2)}_"
                        f"{game_instance}.model"
                    )
                break

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            epsilon = max(min_epsilon, epsilon)

    print(score_counter, win_counter, total_offs)


if __name__ == "__main__":
    args = parse_args()
    avg_n_games(args.games, args.runtype, args.save, args.model, args.verbose)
