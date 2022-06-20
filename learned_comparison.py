import argparse
import game
import numpy as np
import os
import pickle
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
    parser.add_argument("guesser", help="Which guessing model", type=str)
    parser.add_argument("player", help="Which playing model", type=str)
    parser.add_argument("games_folder", help="Where to find generated games", type=str)
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


def print_performance(agent_pair, score_counter, win_counter, total_offs):
    print("Agents: ", agent_pair)
    print("Scores: ", score_counter)
    print("Wins: ", win_counter)
    print("Mistakes: ", total_offs)


def learned_n_games(
    n: int,
    games_folder: str,
    guessing_model: str,
    playing_model: str,
    guess_inp_size: int,
    player_inp_size: int,
    verbose: int,
    use_agent: bool,
) -> None:

    # Make the deck
    full_deck = []
    deck_dict = {}

    for suit in range(4):  # (blue, yellow, red, green)
        for card_value in range(15):  # (joker, 1-13, wizard)
            full_deck.append((suit, card_value))

            # to go from card to index
            deck_dict[(suit, card_value)] = card_value + suit * 15

    all_decks = pickle.load(open(f"{games_folder}/decks.pkl", "rb"))
    all_players = pickle.load(open(f"{games_folder}/players.pkl", "rb"))
    print(len(all_decks), len(all_players), len(all_decks[0]), len(all_players[0]))

    print("Pair: ", guessing_model, playing_model)
    guess_agent = GuessingAgent(input_size=guess_inp_size, guess_max=21)
    playing_agent = PlayingAgent(input_size=player_inp_size, verbose=verbose)

    if guessing_model == "heuristic" or guessing_model == "random":
        guess_type = guessing_model

    else:
        guess_agent.model = tf.keras.models.load_model(
            os.path.join("models", guessing_model)
        )
        guess_type = "learned"

    if playing_model == "heuristic" or playing_model == "random":
        player_type = playing_model

    else:
        playing_agent.network_policy.model = tf.keras.models.load_model(
            os.path.join("models", playing_model)
        )
        player_type = "learned"

    print(guess_agent.model.summary())
    print(f"Guesser loss-function: ", guess_agent.model.loss)
    print(playing_agent.network_policy.model.summary())
    print(f"Player loss-function: ", playing_agent.network_policy.model.loss)

    pair_name = (guessing_model, playing_model)

    # win_counter, score_counter, total_offs(too high guess, too low guess), last_ten, accuracy_hist
    performance = [[0, 0, 0], [0, 0, 0], [0, 0], 21 * [0], []]
    total_distribution = np.zeros(21, dtype=int)
    total_actual = np.zeros(21, dtype=int)

    for game_instance in range(1, n + 1):
        print(f"Game instance: {game_instance}")
        shuffled_decks = all_decks[
            (game_instance - 1) * 20: (game_instance - 1) * 20 + 20
        ]
        shuffled_players = all_players[game_instance - 1]
        off_game, scores, offs, guess_distribution, actual_distribution = play_game(
            full_deck,
            deck_dict,
            guess_type,
            player_type,
            shuffled_decks,
            shuffled_players,
            guess_agent,
            playing_agent,
            verbose,
            use_agent,
        )
        total_distribution = np.add(total_distribution, guess_distribution)
        total_actual = np.add(total_actual, actual_distribution)
        # win_counter, score_counter, total_offs(too high guess, too low guess), last_ten, accuracy_hist
        # For command-line output
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

    print("Pair: ", pair_name)
    print("Scores: ", performance[1])
    print("Wins: ", performance[0])
    print("Mistakes (high guess, low guess): ", performance[2])
    print(f"Guesses: {list(total_distribution)}")
    print(f"Actual: {list(total_actual)}")


def play_game(
    full_deck,
    deck_dict,
    guess_type,
    player_type,
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
        guess_type=guess_type,
        player_type=player_type,
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
    distribution = wizard.get_distribution()
    return off_game, scores, offs, distribution[0], distribution[1]


if __name__ == "__main__":
    args = parse_args()
    learned_n_games(
        args.games,
        args.games_folder,
        args.guesser,
        args.player,
        args.guesser_input,
        args.player_input,
        args.verbose,
        args.use_agent,
    )
