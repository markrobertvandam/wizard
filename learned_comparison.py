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
        "save_folder",
        help="folder name for plots",
        default="",
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
    model_folders: list,
    verbose: int,
    use_agent: bool,
) -> None:

    input_sizes = {"cheater": (68, 3915), "porder": (68, 3795), "old": (68, 3731), "small": (68, 195),
                   "random": (68, 195), "random_player": (68, 195), "random_guesser": (68, 195)}

    # Make the deck
    full_deck = []
    deck_dict = {}

    for suit in range(4):  # (blue, yellow, red, green)
        for card_value in range(15):  # (joker, 1-13, wizard)
            full_deck.append((suit, card_value))

            # to go from card to index
            deck_dict[(suit, card_value)] = card_value + suit * 15

    agent_pairs = []

    guessing_models = []
    player_models = []
    inputs = []

    for model_folder in model_folders:
        if model_folder == "random_player":
            inputs.append(input_sizes["random_player"])
            guessing_models.append(guessing_models[-1])
            player_models.append("random")
        elif model_folder == "random_guesser":
            inputs.append(input_sizes["random_guesser"])
            guessing_models.append("random")
            player_models.append(player_models[-1])
        elif model_folder == "random":
            inputs.append(input_sizes["random"])
            guessing_models.append("random")
            player_models.append("random")
        else:
            path = os.path.join("models", model_folder)
            models = sorted(os.listdir(path))
            for model in models:
                inputs.append(input_sizes[model_folder.split("_")[-1]])
                if model.startswith("guessing"):
                    guessing_models.append(os.path.join(path, model))
                else:
                    player_models.append(os.path.join(path, model))
    for i in range(len(guessing_models)):
        input_size_guess, input_size_play = inputs[i]
        guessing_model = guessing_models[i]
        playing_model = player_models[i]
        guess_agent = GuessingAgent(input_size=input_size_guess, guess_max=21)
        print("Pair: ", guessing_model, playing_model)
        playing_agent = PlayingAgent(input_size=input_size_play, verbose=verbose)
        if not guessing_model != "random":
            guess_agent.model = tf.keras.models.load_model(guessing_model)
            guess_agent.trained = True

        if playing_model != "random":
            playing_agent.network_policy.model = tf.keras.models.load_model(playing_model)
            playing_agent.trained = True

        agent_pairs.append((guess_agent, playing_agent))

    performance_dict = dict()
    for pair in agent_pairs:
        # win_counter, score_counter, total_offs(too high guess, too low guess), last_ten, accuracy_hist
        performance_dict[pair] = [[0, 0, 0], [0, 0, 0], [0, 0], 21 * [0], []]

    for game_instance in range(1, n + 1):
        print("Game instance: ", game_instance)
        shuffled_decks = []
        shuffled_players = ["player1", "player2", "player3"]
        random.shuffle(shuffled_players)
        for i in range(20):
            shuffled_decks.append(random.sample(full_deck, 60))
        for pair in agent_pairs:
            (guess_agent, playing_agent) = pair
            off_game, scores, offs = play_game(
                full_deck,
                deck_dict,
                shuffled_decks,
                shuffled_players,
                guess_agent,
                playing_agent,
                verbose,
                use_agent,
            )
            # win_counter, score_counter, total_offs(too high guess, too low guess), last_ten, accuracy_hist
            # For command-line output
            performance = performance_dict[pair]
            performance[3] += off_game
            for player in range(3):
                performance[1][player] += scores[player] / n
                if scores[player] == max(scores):
                    performance[0][player] += 1
            performance[2][0] += offs[0]
            performance[2][1] += offs[1]

            if game_instance % 10 == 0:
                accuracy = performance[3][0] / 200
                performance[4].append(accuracy)
                print(
                    f"Agents: {(guess_agent, playing_agent)}, "
                    f"Game {game_instance}, accuracy: {accuracy}, "
                    f"Last10: {performance[3]}"
                )
                performance[3] *= 0

    for pair in agent_pairs:
        performance = performance_dict[pair]
        print("Scores: ", performance[1])
        print("Wins: ", performance[0])
        print("Mistakes (high guess, low guess): ", performance[2])


def play_game(
    full_deck,
    deck_dict,
    shuffled_decks,
    shuffled_players,
    guess_agent,
    play_agent,
    verbose,
    use_agent,
):

    wizard = game.Game(
        full_deck,
        deck_dict,
        "learned",
        shuffled_decks=shuffled_decks,
        shuffled_players=shuffled_players,
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
        args.games,
        args.save_folder,
        ["mcts_025_old", "mcts_small_porder", "mcts_medium_porder", "random_player"],
        args.verbose,
        args.use_agent,
    )
