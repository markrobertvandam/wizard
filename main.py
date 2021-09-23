import game


def avg_n_games(n):
    win_counter = [0, 0, 0]
    score_counter = [0, 0, 0]
    for game_instance in range(n):
        wizard = game.Game()
        scores = wizard.play_game()
        for player in range(3):
            score_counter[player] += scores[player] / n
            if scores[player] == max(scores):
                win_counter[player] += 1
                if player != 0:
                    print(scores)
    print(score_counter, win_counter)


if __name__ == "__main__":
    n = int(input("How many games: "))
    avg_n_games(n)
