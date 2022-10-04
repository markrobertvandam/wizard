import argparse
import game
import numpy as np
import os
import pickle
import statistics
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
        "--opp_guesstype",
        help="type of guessing agent for opponents (random, heuristic, learning, learned)",
        default="heuristic",
    )
    parser.add_argument(
        "--opp_playertype",
        help="type of playing agent for opponents (random, heuristic, learning, learned)",
        default="heuristic",
    )
    parser.add_argument(
        "--opp_model",
        help="Opponent guessing agents",
        default="",
        type=str,
    )
    parser.add_argument(
        "--opp_playmodel",
        help="Opponent playing agents",
        default="",
        type=str,
    )
    parser.add_argument(
        "--opp_size",
        help="argument to set guess inp_size for opponents",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--opp_playersize",
        help="argument to set player inp_size",
        type=int,
        default=0,
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
    opp_guesstype: str,
    opp_playertype: str,
    opp_model: str,
    opp_playmodel: str,
    opp_size: int,
    opp_play_size: int,
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

    guess_agent2 = None
    playing_agent2 = None
    guess_agent3 = None
    playing_agent3 = None

    if opp_guesstype == "learned":
        print("Creating guess agents for opponents..")
        if opp_size == 0:
            opp_size = guess_inp_size
        guess_agent2 = GuessingAgent(input_size=opp_size, guess_max=21)
        guess_agent3 = GuessingAgent(input_size=opp_size, guess_max=21)
        print("\n")
        guess_agent2.model.summary()

        if opp_model == "":
            print("No guessing agent passed to load to opponents")
        else:
            print(f"Loading saved guessing model {opp_model} to opponents")
            guess_agent2.model = tf.keras.models.load_model(
                os.path.join("models", opp_model)
            )
            guess_agent3.model = tf.keras.models.load_model(
                os.path.join("models", opp_model)
            )

    if opp_playertype == "learned":
        if opp_play_size == 0:
            opp_play_size = player_inp_size
        playing_agent2 = PlayingAgent(input_size=opp_play_size, verbose=verbose)
        playing_agent3 = PlayingAgent(input_size=opp_play_size, verbose=verbose)
        print("\n")
        playing_agent2.network_policy.model.summary()

        if opp_playmodel == "":
            print("No playing agent passed to load to opponents")
        else:
            print(f"Loading saved playing model {opp_playmodel} to opponents")
            playing_agent2.network_policy.model = tf.keras.models.load_model(
                os.path.join("models", opp_playmodel)
            )
            playing_agent3.network_policy.model = tf.keras.models.load_model(
                os.path.join("models", opp_playmodel)
            )

    print(guess_agent.model.summary())
    print(f"Guesser loss-function: ", guess_agent.model.loss)
    print(playing_agent.network_policy.model.summary())
    print(f"Player loss-function: ", playing_agent.network_policy.model.loss)

    pair_name = (guessing_model, playing_model)

    # win_counter, score_counter, total_offs(too high guess, too low guess), last_ten, accuracy_hist
    performance = [[0, 0, 0], [0, 0, 0], [0, 0], 21 * [0], []]

    scores_player1 = []
    scores_player2 = []
    scores_player3 = []

    total_round_offs = np.zeros(20, dtype=int)
    total_distribution = np.zeros(21, dtype=int)
    total_actual = np.zeros(21, dtype=int)
    total_overshoot = np.zeros(20, dtype=int)

    for game_instance in range(1, n + 1):
        print(f"Game instance: {game_instance}")
        shuffled_decks = all_decks[
            (game_instance - 1) * 20: (game_instance - 1) * 20 + 20
        ]
        shuffled_players = all_players[game_instance - 1]
        off_game, scores, offs, round_offs, guess_distribution, actual_distribution, overshoot = play_game(
            full_deck,
            deck_dict,
            guess_type,
            player_type,
            shuffled_decks,
            shuffled_players,
            guess_agent,
            playing_agent,
            verbose,
            opp_guesstype=opp_guesstype,
            opp_playertype=opp_playertype,
            guess_agent2=guess_agent2,
            playing_agent2=playing_agent2,
            guess_agent3=guess_agent3,
            playing_agent3=playing_agent3,
        )
        total_distribution = np.add(total_distribution, guess_distribution)
        total_actual = np.add(total_actual, actual_distribution)
        total_round_offs = np.add(total_round_offs, round_offs)
        total_overshoot = np.add(total_overshoot, overshoot)

        scores_player1.append(scores[0])
        scores_player2.append(scores[1])
        scores_player3.append(scores[2])

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
                f"Last10: {list(performance[3])}"
            )
            print(f"Total round offs: {list(total_round_offs)}")
            performance[3] *= 0

    print("Pair: ", pair_name)
    print("Avg Scores: ", performance[1])
    print("Max/Min/Median scores p1: ", {max(scores_player1)}, {min(scores_player1)},
          {statistics.median(scores_player1)})

    print("Max/Min/Median scores p2: ", {max(scores_player2)}, {min(scores_player2)},
          {statistics.median(scores_player2)})

    print("Max/Min/Median scores p3: ", {max(scores_player3)}, {min(scores_player3)},
          {statistics.median(scores_player3)})
    print("Wins: ", performance[0])
    print("Total draws: ", sum(performance[0]) - game_instance)
    print("Mistakes (high guess, low guess): ", performance[2])
    print(f"Overshot: {list(total_overshoot)}")
    print(f"Guesses: {list(total_distribution)}")
    print(f"Actual: {list(total_actual)}")
    print(f"Total round offs: {list(total_round_offs)}")


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
    opp_guesstype: str,
    opp_playertype: str,
    guess_agent2,
    playing_agent2,
    guess_agent3,
    playing_agent3,
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
        opp_guesstype=opp_guesstype,
        opp_playertype=opp_playertype,
        guess_agent2=guess_agent2,
        playing_agent2=playing_agent2,
        guess_agent3=guess_agent3,
        playing_agent3=playing_agent3,
    )
    loss, scores, offs, round_offs = wizard.play_game()
    off_game = wizard.get_game_performance()
    distribution = wizard.get_distribution()
    overshoot = wizard.get_overshoot()
    return off_game, scores, offs, round_offs, distribution[0], distribution[1], overshoot


if __name__ == "__main__":
    args = parse_args()
    print(f"Opponent guesstype: {args.opp_guesstype}")
    print(f"Opponent playertype: {args.opp_playertype}")
    print(f"Opponent model: {args.opp_model}")
    print(f"Opponent playermodel: {args.opp_playmodel}")
    learned_n_games(
        args.games,
        args.games_folder,
        args.guesser,
        args.player,
        args.guesser_input,
        args.player_input,
        args.verbose,
        args.opp_guesstype,
        args.opp_playertype,
        args.opp_model,
        args.opp_playmodel,
        args.opp_size,
        args.opp_playersize,
    )
