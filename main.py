import argparse
import game
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from Guessing_Agent import GuessingAgent
from Playing_Agent import PlayingAgent
from numpy.random import seed

seed(1)
tf.random.set_seed(1)


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
    parser.add_argument(
        "--n_step",
        help="optional argument to use n_step q-learning, default of 1 leads to normal DQN",
        default=1,
        type=int,
    )
    parser.add_argument("--double", action="store_true",
                        help="use double DQN")
    parser.add_argument("--mask", action="store_true",
                        help="mask illegal moves")
    parser.add_argument("--cardcount", action="store_true",
                        help="explicit card counting for play_agent")
    parser.add_argument("--dueling", action="store_true",
                        help="use dueling DQN")
    parser.add_argument("--priority", action="store_true",
                        help="use prioritized experience replay")
    parser.add_argument("--punish", action="store_true",
                        help="optional argument to punish play depending on difference with goal trick wins")

    return parser.parse_args()


def plot_history(
    history: list, game_instance: int, save_folder: str, iters_done: int, y="Accuracy",
) -> None:

    # first iterations not plotted in loss history because no learning yet
    base = 10
    if iters_done == 0:
        base = 30

    if y == "Accuracy":
        x = list(range(iters_done + 10, game_instance + 1, 10))
    elif y == "Loss":
        x = list(range(iters_done + base, game_instance + 1, 10))
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
    opp_guesstype: str,
    opp_playertype: str,
    opp_model: str,
    opp_playmodel: str,
    epsilon: float,
    player_epsilon: float,
    iters_done: int,
    double: bool,
    mask: bool,
    dueling: bool,
    priority: bool,
    punish: bool,
    n_step: int,
) -> None:
    input_size_guess = 68
    input_size_play = 192
    opp_size = 68
    opp_play_size = 192

    name = None
    if save_folder != "":
        models_path = os.path.join("models", save_folder)
        plot_path = os.path.join("plots", save_folder)
        if "_" in save_folder:
            name = save_folder.split("_")[1]
        if save_bool.startswith("y"):
            if not os.path.exists(models_path):
                os.mkdir(models_path)
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)
    elif player_model is not None:
        name = player_model.split("/")[0].split("_")[1]
    
    print(f"Inp_size guess: {input_size_guess} and Inp_size play: {input_size_play}")
    guess_agent = GuessingAgent(input_size=input_size_guess, guess_max=21)
    playing_agent = PlayingAgent(input_size=input_size_play, save_bool=(save_bool.startswith("y")),
                                 name=name, verbose=verbose, mask=mask, dueling=dueling, double=double,
                                 priority=priority, punish=punish, n_step=n_step)
    guess_agent2 = None
    playing_agent2 = None
    guess_agent3 = None
    playing_agent3 = None

    if opp_guesstype.startswith("learn"):
        print("Creating guess agents for opponents..")
        guess_agent2 = GuessingAgent(input_size=opp_size, guess_max=21)
        guess_agent3 = GuessingAgent(input_size=opp_size, guess_max=21)
        print("\n")
        guess_agent2.model.summary()

        if opp_model == "":
            print("No guessing agent passed to fixed opponents")
        else:
            print(f"Loading saved guessing model {opp_model} to opponents")
            guess_agent2.model = tf.keras.models.load_model(
                os.path.join("models", opp_model)
            )
            guess_agent3.model = tf.keras.models.load_model(
                os.path.join("models", opp_model)
            )

    if opp_playertype.startswith("learn"):
        playing_agent2 = PlayingAgent(input_size=opp_play_size, name=name, verbose=verbose,
                                      mask=mask, dueling=dueling, double=double,
                                      priority=priority, punish=punish, n_step=n_step)
        playing_agent3 = PlayingAgent(input_size=opp_play_size, name=name, verbose=verbose,
                                      mask=mask, dueling=dueling, double=double,
                                      priority=priority, punish=punish, n_step=n_step)
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

    if guess_type == "learned" or (guess_type == "learning" and model_path is not None):
        print(f"Loading saved model {model_path}")
        guess_agent.model = tf.keras.models.load_model(
            os.path.join("models", model_path)
        )

    if player_type == "learned" or (player_type == "learning" and player_model is not None):
        print(f"Loading saved model {player_model}")
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
    loss_history = []
    avg_loss = 0.0

    last_max = iters_done
    max_acc = 0
    output_path = "state_err1"
    print("Guess type: ", guess_type, "Player type: ", player_type)
    if opp_guesstype.startswith("learn") or opp_playertype.startswith("learn"):
        print(f"Opposing agents: {guess_agent2}, {playing_agent2}, {guess_agent3}, {playing_agent3}")
    for game_instance in range(1 + iters_done, n + 1 + iters_done):
        #print("\nGame instance: ", game_instance)
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
        )
        game_loss, scores, offs = wizard.play_game()
        avg_loss += game_loss

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

            if game_instance >= 30:
                loss_history.append(avg_loss/10)
            print(
                f"Game {game_instance}, Avg accuracy: {accuracy}, Avg loss: {avg_loss/10}, Epsilon: {round(epsilon,2)}, "
                f"Epsilon P: {round(player_epsilon,2)}, "
                f"Last10: {last_ten_performance}"
            )
            avg_loss = 0.0
            last_ten_performance *= 0

            # for early stopping
            if accuracy > max_acc and (player_type == "learning" or guess_type == "learning"):
                last_max = game_instance
                max_acc = accuracy

        if player_type == "learning" or guess_type == "learning":
            if player_type == "learning" and game_instance % 10 == 0:
                    print("Updating target network weights..")
                    target_model = playing_agent.network_policy.target_model
                    online_model = playing_agent.network_policy.model
                    target_model.set_weights(online_model.get_weights())

            if game_instance % 1000 == 0 and game_instance > iters_done:
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
                break

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            epsilon = max(min_epsilon, epsilon)

        if player_epsilon > min_epsilon:
            player_epsilon *= player_decay
            player_epsilon = max(0.03, player_epsilon)

    print("Scores: ", score_counter)
    print("Wins: ", win_counter)
    print("Mistakes: ", total_offs)


if __name__ == "__main__":
    args = parse_args()
    print(f"Opponent guesstype: {args.opp_guesstype}")
    print(f"Opponent playertype: {args.opp_playertype}")
    print(f"Opponent model: {args.opp_model}")
    print(f"Opponent playermodel: {args.opp_playmodel}")
    print(f"Double: {args.double}")
    print(f"Masking: {args.mask}")
    print(f"Dueling: {args.dueling}")
    print(f"Prioritized Experience Replay: {args.priority}")
    print(f"Punish based on distance from goal: {args.punish}")
    print(f"N-step: {args.n_step}")

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
        args.opp_guesstype,
        args.opp_playertype,
        args.opp_model,
        args.opp_playmodel,
        args.epsilon,
        args.player_epsilon,
        args.iters_done,
        args.double,
        args.mask,
        args.dueling,
        args.priority,
        args.punish,
        args.n_step
    )
