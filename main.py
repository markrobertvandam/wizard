import argparse
import game
import numpy as np
import matplotlib.pyplot as plt
import os
import statistics
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
        "--guesser_size",
        help="optional argument to set guesser input size to something other than 68",
        default=68,
        type=int,
    )
    parser.add_argument(
        "--player_size",
        help="optional argument to set player input size to something other than 192",
        default=192,
        type=int,
    )
    parser.add_argument(
        "--opp_guesser_size",
        help="optional argument to set opp guesser input size to something other than 68",
        default=68,
        type=int,
    )
    parser.add_argument(
        "--opp_player_size",
        help="optional argument to set opp player input size to something other than 192",
        default=192,
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
    parser.add_argument("--punish", action="store_true",
                        help="optional argument to punish play depending on difference with goal trick wins")
    parser.add_argument("--score", action="store_true",
                        help="optional argument to reward play based on actual point gains")
    parser.add_argument("--diff", action="store_true",
                        help="optional argument to reward play based on relative point gains compared to opp")
    parser.add_argument("--soft_guess", action="store_true",
                        help="optional argument to sample guessing softmax curve instead of taking argmax")
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
    parser.add_argument("--reoccur_bool", action="store_true",
                        help="optional argument to save reoccuring states")

    parser.add_argument("--train_every",
                        help="optional arg to make model train every game instead of every round",
                        default="round",
                        type=str,
                        )

    return parser.parse_args()


def plot_history(
    history: list, game_instance: int, save_folder: str, iters_done: int, y="Accuracy",
) -> None:

    if y == "Accuracy":
        x = list(range(iters_done + 10, game_instance + 1, 10))
    elif y == "Loss":
        x = list(range(iters_done + 30, game_instance + 1, 10))
    else:
        return
    plt.plot(x, history)
    plt.xlabel("Games", fontsize=15)
    plt.ylabel(y, fontsize=15)
    plt.savefig(f"plots/{save_folder}/{y}_plot")
    plt.close()


def save_models(
    guess_agent: GuessingAgent,
    playing_agent: PlayingAgent,
    save_folder: str,
    input_size_guess: int,
    input_size_play: int,
    accuracy: float,
    game_instance: int,
    name_nr="",
) -> None:
    time_label = str(int(time.time() / 3600))[-3:]
    guess_agent.model.save(
        f"models/{save_folder}/guessing{input_size_guess}_"
        f"{time_label}_"
        f"{accuracy}_"
        f"{game_instance}{name_nr}.model"
    )
    playing_agent.network_policy.model.save(
        f"models/{save_folder}/playing{input_size_play}_"
        f"{time_label}_"
        f"{accuracy}_"
        f"{game_instance}{name_nr}.model"
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
    guesser_size: int,
    player_size: int,
    opp_guesser_size: int,
    opp_player_size: int,
    opp_guesstype: str,
    opp_playertype: str,
    opp_model: str,
    opp_playmodel: str,
    punish: bool,
    score: bool,
    diff: bool,
    soft_guess: bool,
    epsilon: float,
    player_epsilon: float,
    iters_done: int,
    reoccur_bool: bool,
    train_every: str,
) -> None:
    input_size_guess = guesser_size
    input_size_play = player_size
    opp_size = opp_guesser_size
    opp_play_size = opp_player_size

    name = None
    if save_folder != "":
        models_path = os.path.join("models", save_folder)
        plot_path = os.path.join("plots", save_folder)
        if "_" in save_folder:
            name = save_folder.split("_")[1]
        if save_bool:
            if not os.path.exists(models_path):
                os.mkdir(models_path)
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)
    elif player_model is not None:
        name = player_model.split("/")[0].split("_")[1]

    guess_agent = GuessingAgent(input_size=input_size_guess, guess_max=21, soft_guess=soft_guess)
    playing_agent = PlayingAgent(input_size=input_size_play, name=name, verbose=verbose, punish=punish,
                                 score=score, diff=diff)
    guess_agent2 = None
    playing_agent2 = None
    guess_agent3 = None
    playing_agent3 = None

    if opp_guesstype.startswith("learn"):
        print("Creating guess agents for opponents..")
        guess_agent2 = GuessingAgent(input_size=opp_size, guess_max=21, soft_guess=soft_guess)
        guess_agent3 = GuessingAgent(input_size=opp_size, guess_max=21, soft_guess=soft_guess)
        print("Opposing guess model:\n")
        guess_agent2.model.summary()
        print(f"\nGuesser loss-function opponents: ", guess_agent2.model.loss)
        
        if opp_model == "":
            print("No guessing agent passed to load to opponents\n")
        else:
            print(f"Loading saved guessing model {opp_model} to opponents")
            guess_agent2.model = tf.keras.models.load_model(
                os.path.join("models", opp_model)
            )
            guess_agent3.model = tf.keras.models.load_model(
                os.path.join("models", opp_model)
            )

    if opp_playertype.startswith("learn"):
        print("Creating play agents for opponents..")
        playing_agent2 = PlayingAgent(input_size=opp_play_size, name=name, verbose=verbose, punish=punish,
                                      score=score, diff=diff)
        playing_agent3 = PlayingAgent(input_size=opp_play_size, name=name, verbose=verbose, punish=punish,
                                      score=score, diff=diff)
        print("Opposing play model:\n")
        playing_agent2.network_policy.model.summary()
        print(f"\nPlayer loss-function opponents: ", playing_agent2.network_policy.model.loss)

        if opp_playmodel == "":
            print("No playing agent passed to load to opponents\n")
        else:
            print(f"Loading saved playing model {opp_playmodel} to opponents")
            playing_agent2.network_policy.model = tf.keras.models.load_model(
                os.path.join("models", opp_playmodel)
            )
            playing_agent3.network_policy.model = tf.keras.models.load_model(
                os.path.join("models", opp_playmodel)
            )

    if guess_type == "learned" or (guess_type == "learning" and model_path is not None):
        print(f"Loading saved guessing model {model_path}")
        guess_agent.model = tf.keras.models.load_model(
            os.path.join("models", model_path)
        )

    if player_type == "learned" or (player_type == "learning" and player_model is not None):
        print(f"Loading saved playing model {player_model}")
        playing_agent.network_policy.model = tf.keras.models.load_model(
            os.path.join("models", player_model)
        )

    print(guess_agent.model.summary())
    print(f"Guesser loss-function: ", guess_agent.model.loss)
    print(playing_agent.network_policy.model.summary())
    print(f"Player loss-function: ", playing_agent.network_policy.model.loss)
    # Exploration settings
    epsilon_decay = 0.9985
    player_decay = 0.999
    min_epsilon = 0.03

    print(f"Min epsilon and epsilon decay: {min_epsilon} and {epsilon_decay}")
    # For keeping track of performance
    win_counter = [0, 0, 0]
    score_counter = [0, 0, 0]
    scores_player1 = []
    scores_player2 = []
    scores_player3 = []
    total_offs = [0, 0]
    total_round_offs = np.zeros(20, dtype=int)

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
    loss_history = []
    avg_loss = 0.0
    last_max = iters_done
    max_acc = 0
    output_path = "state_err1"
    print("Guess type: ", guess_type, "Player type: ", player_type)
    if opp_guesstype.startswith("learn") or opp_playertype.startswith("learn"):
        print(f"Opposing agents: {guess_agent2}, {playing_agent2}, {guess_agent3}, {playing_agent3}")

    train_every_games = None
    if train_every.isnumeric():
        train_every_games = int(train_every)

    for game_instance in range(1 + iters_done, n + 1 + iters_done):
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
            opp_guesstype=opp_guesstype,
            opp_playertype=opp_playertype,
            guess_agent2=guess_agent2,
            playing_agent2=playing_agent2,
            guess_agent3=guess_agent3,
            playing_agent3=playing_agent3,
            save_folder=save_folder,
            reoccur_bool=reoccur_bool,
            train_every=train_every,
        )
        game_loss, scores, offs, round_offs = wizard.play_game()
        total_round_offs += round_offs
        avg_loss += game_loss

        # For command-line output while training
        last_ten_performance += wizard.get_game_performance()
        scores_player1.append(scores[0])
        scores_player2.append(scores[1])
        scores_player3.append(scores[2])
        for player in range(3):
            score_counter[player] += scores[player] / n
            if scores[player] == max(scores):
                win_counter[player] += 1
        total_offs[0] += offs[0]
        total_offs[1] += offs[1]

        if train_every_games and (game_instance-iters_done) % train_every_games == 0:
            for player in wizard.players:
                avg_loss += wizard.train_network(player)

        if game_instance % 10 == 0:
            accuracy = last_ten_performance[0] / 200
            guess_agent.accuracy = accuracy
            accuracy_history.append(accuracy)
            if game_instance - iters_done >= 30:
                loss_history.append(avg_loss/10)
            print(
                f"Game {game_instance}, accuracy: {accuracy}, Avg loss: {avg_loss/10}, Epsilon: {round(epsilon,2)}, "
                f"Epsilon P: {round(player_epsilon,2)}, "
                f"Last10: {last_ten_performance}"
            )
            print(f"Total states: {playing_agent.full_cntr}")
            print(f"Re-occured states: {playing_agent.cntr}\n")
            print(f"Total mistakes made in each round: {list(total_round_offs)}")
            avg_loss = 0.0
            last_ten_performance *= 0

            # for early stopping
            if accuracy > max_acc and (player_type == "learning" or guess_type == "learning"):
                last_max = game_instance
                max_acc = accuracy

        if player_type == "learning" or guess_type == "learning":
            if game_instance % 1000 == 0:
                if save_bool.startswith("y"):
                    plot_history(
                        accuracy_history, game_instance, save_folder, iters_done
                    )
                    plot_history(
                        loss_history, game_instance, save_folder, iters_done, y="Loss"
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
                if save_bool == "yes-all":
                    save_models(
                        guess_agent2,
                        playing_agent2,
                        save_folder,
                        input_size_guess,
                        input_size_play,
                        accuracy,
                        game_instance,
                        name_nr="(2)"
                    )
                    save_models(
                        guess_agent3,
                        playing_agent3,
                        save_folder,
                        input_size_guess,
                        input_size_play,
                        accuracy,
                        game_instance,
                        name_nr="(3)",
                    )

            if game_instance - last_max > 10000:
                if save_bool.startswith("y"):
                    plot_history(
                        accuracy_history, game_instance, save_folder, iters_done
                    )
                    plot_history(
                        loss_history, game_instance, save_folder, iters_done, y="Loss"
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
                    if save_bool.startswith("yes-all"):
                        save_models(
                            guess_agent2,
                            playing_agent2,
                            save_folder,
                            input_size_guess,
                            input_size_play,
                            accuracy,
                            game_instance,
                            name_nr="(2)"
                        )
                        save_models(
                            guess_agent3,
                            playing_agent3,
                            save_folder,
                            input_size_guess,
                            input_size_play,
                            accuracy,
                            game_instance,
                            name_nr="(3)",
                        )
                break

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            epsilon = max(min_epsilon, epsilon)

        if player_epsilon > min_epsilon:
            player_epsilon *= player_decay
            player_epsilon = max(0.25, player_epsilon)

        if game_instance % 850 == 0:
            for player in wizard.players:
                if player.player_type.startswith("learn"):
                    print(
                        "Forgetting ", len(player.play_agent.nodes.keys()), " nodes.."
                    )
                    player.play_agent.nodes = dict()

    print("Avg Scores: ", score_counter)
    print("Max/Min/Median scores p1: ", {max(scores_player1)}, {min(scores_player1)},
          {statistics.median(scores_player1)})

    print("Max/Min/Median scores p2: ", {max(scores_player2)}, {min(scores_player2)},
          {statistics.median(scores_player2)})

    print("Max/Min/Median scores p3: ", {max(scores_player3)}, {min(scores_player3)},
          {statistics.median(scores_player3)})
    print("Wins: ", win_counter)
    print("Total draws: ", sum(win_counter) - game_instance)
    print("Mistakes: ", total_offs)
    print("Mistakes in each round: ", list(total_round_offs))


if __name__ == "__main__":
    args = parse_args()
    print(f"Save bool: '{args.save}'")
    print(f"Save folder: '{args.save_folder}'")
    if args.save.startswith("y") and args.save_folder == "":
        print("save bool is true but no save folder given.")
        exit()
    print(f"Opponent guesstype: {args.opp_guesstype}")
    print(f"Opponent playertype: {args.opp_playertype}")
    print(f"Opponent model: {args.opp_model}")
    print(f"Opponent playermodel: {args.opp_playmodel}")
    print(f"Punish based on distance from goal: {args.punish}")
    print(f"reward based on own score: {args.score}")
    print(f"reward based on differences in score: {args.diff}")
    print(f"Guess based on softmax curve instead of argmax: {args.soft_guess}")
    print(f"Save reoccuring states?: {args.reoccur_bool}")
    print(f"Train every: {args.train_every}")

    if not args.opp_guesstype.startswith("learn") and args.opp_model:
        print("Guessing agent given but not used")

    if not args.opp_playertype.startswith("learn") and args.opp_playmodel:
        print("Playing agent given but not used")

    avg_n_games(
        args.games,
        args.guesstype,
        args.playertype,
        args.save,
        args.save_folder,
        args.model,
        args.play_model,
        args.verbose,
        args.guesser_size,
        args.player_size,
        args.opp_guesser_size,
        args.opp_player_size,
        args.opp_guesstype,
        args.opp_playertype,
        args.opp_model,
        args.opp_playmodel,
        args.punish,
        args.score,
        args.diff,
        args.soft_guess,
        args.epsilon,
        args.player_epsilon,
        args.iters_done,
        args.reoccur_bool,
        args.train_every,
    )
