import argparse
import game
import numpy as np
import matplotlib.pyplot as plt
import os
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
        "guesstype",
        help="type of agent for player 1 (random, heuristic, learning, learned)",
    )
    parser.add_argument(
        "playertype",
        help="type of agent for player 1 (random, heuristic, learning, learned)",
    )
    parser.add_argument(
        "save", help="argument to determine whether model should be saved when learning"
    )
    parser.add_argument(
        "--save_folder",
        help="folder name for plots and models for this run",
        default="",
    )
    parser.add_argument(
        "--verbose",
        help="optional argument to set how verbose the run is",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--model",
        help="optional argument to load in the weights of a saved guessing model",
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
        "--player_epsilon",
        help="optional argument to set starting player_epsilon",
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


def plot_accuracy(
    accuracy_history: list, game_instance: int, save_folder: str, iters_done: int
) -> None:
    plt.plot(list(range(iters_done + 10, game_instance + 1, 10)), accuracy_history)
    plt.xlabel("Games", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.savefig(f"plots/{save_folder}/accuracy_plot")
    plt.close()


def save_models(
    guess_agent: GuessingAgent,
    playing_agent: PlayingAgent,
    save_folder: str,
    input_size_guess: int,
    input_size_play: int,
    accuracy: float,
    game_instance: int,
) -> None:
    time_label = str(int(time.time() / 3600))[-3:]
    guess_agent.model.save(
        f"models/{save_folder}/guessing{input_size_guess}_"
        f"{time_label}_"
        f"{accuracy}_"
        f"{game_instance}.model"
    )
    playing_agent.network_policy.model.save(
        f"models/{save_folder}/playing{input_size_play}_"
        f"{time_label}_"
        f"{accuracy}_"
        f"{game_instance}.model"
    )


def avg_n_games(
    n: int,
    guess_type: str,
    player_type: str,
    save_bool: str,
    save_folder: str,
    model_path: str,
    player_model: str,
    verbose: int,
    use_agent: bool,
    epsilon: float,
    player_epsilon: float,
    iters_done: int,
) -> None:
    input_size_guess = 68
    input_size_play = 3795

    name = None
    if save_folder != "":
        name = save_folder.split("_")[1]
    elif player_model is not None:
        name = player_model.split("/")[0].split("_")[1]

    guess_agent = GuessingAgent(input_size=input_size_guess, guess_max=21)
    playing_agent = PlayingAgent(input_size=input_size_play, name=name, verbose=verbose)
    if guess_type == "learned" or (guess_type == "learning" and model_path != ""):
        print(f"Loading saved model {model_path}")
        guess_agent.model = tf.keras.models.load_model(
            os.path.join("models", model_path)
        )

    if player_type == "learned" or (player_type == "learning" and player_model != ""):
        print(f"Loading saved model {player_model}")
        playing_agent.network_policy.model = tf.keras.models.load_model(
            os.path.join("models", player_model)
        )

    print(guess_agent.model.summary())
    print(f"Guesser loss-function: ", guess_agent.model.loss)
    print(playing_agent.network_policy.model.summary())
    print(f"Player loss-function: ", playing_agent.network_policy.model.loss)
    # Exploration settings
    epsilon_decay = 0.997
    player_decay = 0.999
    min_epsilon = 0.02

    print(f"Min epsilon and epsilon decay: {min_epsilon} and {epsilon_decay}")
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
    output_path = "state_err1"
    print("Guess type: ", guess_type, "Player type: ", player_type)
    for game_instance in range(1 + iters_done, n + 1 + iters_done):
        if verbose:
            print("\nGame instance: ", game_instance)
        wizard = game.Game(
            full_deck,
            deck_dict,
            guess_type=guess_type,
            player_type=player_type,
            output_path=output_path,
            guess_agent=guess_agent,
            playing_agent=playing_agent,
            epsilon=epsilon,
            player_epsilon=player_epsilon,
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
                f"Epsilon P: {round(player_epsilon,2)}, "
                f"Last10: {last_ten_performance}"
            )
            last_ten_performance *= 0

            # for early stopping
            if accuracy > max_acc and (player_type == "learning" or guess_type == "learning"):
                last_max = game_instance
                max_acc = accuracy

        if player_type == "learning" or guess_type == "learning":
            if game_instance % 1000 == 0:
                if save_bool.startswith("y"):
                    plot_accuracy(
                        accuracy_history, game_instance, save_folder, iters_done
                    )
                    save_models(
                        guess_agent,
                        playing_agent,
                        save_folder,
                        input_size_guess,
                        input_size_play,
                        accuracy,
                        game_instance,
                    )

            if game_instance - last_max > 10000:
                if save_bool.startswith("y"):
                    plot_accuracy(
                        accuracy_history, game_instance, save_folder, iters_done
                    )
                    save_models(
                        guess_agent,
                        playing_agent,
                        save_folder,
                        input_size_guess,
                        input_size_play,
                        accuracy,
                        game_instance,
                    )
                break

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            epsilon = max(min_epsilon, epsilon)

        if player_epsilon > min_epsilon:
            player_epsilon *= player_decay
            player_epsilon = max(0.25, player_epsilon)

        if game_instance % 900 == 0:
            print(
                "Forgetting ", len(wizard.player1.play_agent.nodes.keys()), " nodes.."
            )
            wizard.player1.play_agent.nodes = dict()

    print("Scores: ", score_counter)
    print("Wins: ", win_counter)
    print("Mistakes: ", total_offs)


if __name__ == "__main__":
    args = parse_args()
    avg_n_games(
        args.games,
        args.guesstype,
        args.playertype,
        args.save,
        args.save_folder,
        args.model,
        args.play_model,
        args.verbose,
        args.use_agent,
        args.epsilon,
        args.player_epsilon,
        args.iters_done,
    )
