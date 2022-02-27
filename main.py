import game
import numpy as np
import os
import time
import tensorflow as tf
from Guessing_Agent import GuessingAgent


def avg_n_games(n, model_path="", save_bool="y"):
    input_size = 48
    guess_agent = GuessingAgent(input_size=input_size, guess_max=20)
    if model_path != "":
        guess_agent.model = tf.keras.models.load_model(
            os.path.join("models", model_path)
        )

    # Exploration settings
    epsilon = 1  # not a constant, going to be decayed
    epsilon_decay = 0.997
    min_epsilon = 0.02

    win_counter = [0, 0, 0]
    score_counter = [0, 0, 0]
    total_offs = [0, 0]

    full_deck = []

    # Make the deck
    for card_value in range(15):  # (joker, 1-13, wizard)
        for suit in range(4):  # (blue, yellow, red, green)
            full_deck.append((suit, card_value))

    last_ten_performance = np.zeros(20)
    for game_instance in range(n):
        if game_instance % 10 == 0:
            print(
                f"Game {game_instance}, avg_reward: {int(guess_agent.avg_reward)}, Epsilon: {round(epsilon,2)}, "
                f"Last10: {last_ten_performance}"
            )
            last_ten_performance *= 0
        wizard = game.Game(full_deck, guess_agent, epsilon, verbose=True)
        scores, offs = wizard.play_game()
        last_ten_performance += wizard.get_game_performance()
        for player in range(3):
            score_counter[player] += scores[player] / n
            if scores[player] == max(scores):
                win_counter[player] += 1
        total_offs[0] += offs[0]
        total_offs[1] += offs[1]

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            epsilon = max(min_epsilon, epsilon)

    print(score_counter, win_counter, total_offs)

    if save_bool.startswith("y"):
        guess_agent.model.save(
            f"models/guessing{input_size}_{round(guess_agent.avg_reward, 2)}_{str(int(time.time()/3600))[-3:]}_{n}.model"
        )


if __name__ == "__main__":
    n = int(input("How many games: "))
    model = input("Saved model-name: ")
    save = input("Save the model? (y/n): ")
    avg_n_games(n, model, save)
