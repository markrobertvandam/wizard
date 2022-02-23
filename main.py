import game
import time
from Guessing_Agent import GuessingAgent


def avg_n_games(n):
    # Exploration settings
    epsilon = 1  # not a constant, going to be decayed
    epsilon_decay = 0.996
    min_epsilon = 0.001

    win_counter = [0, 0, 0]
    score_counter = [0, 0, 0]
    total_offs = [0, 0]

    full_deck = []

    # Make the deck
    for card_value in range(15):  # (joker, 1-13, wizard)
        for suit in range(4):  # (blue, yellow, red, green)
            full_deck.append((suit, card_value))

    guess_agent = GuessingAgent(input_size=45, guess_max=20)
    for game_instance in range(n):
        if game_instance % 10 == 0:
            print(f"Game {game_instance}, avg_reward: {guess_agent.avg_reward}")
        wizard = game.Game(full_deck, guess_agent, epsilon)
        scores, offs = wizard.play_game()
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
    guess_agent.model.save(f'models/guessing_{round(guess_agent.avg_reward, 2)}_{round(time.time()/3600, 1)}_{n}.model')


if __name__ == "__main__":
    n = int(input("How many games: "))
    avg_n_games(n)
