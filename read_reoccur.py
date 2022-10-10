import argparse
import pickle
import Playing_Agent
import numpy as np
import utility_functions as util

from collections import Counter

def parse_args() -> argparse.Namespace:
    """
    Function to parse arguments.
    Returns:
    parser: Argument parser containing arguments.
    """

    parser = argparse.ArgumentParser(description="Load in reoccur pickle files")
    parser.add_argument("--destination", help="Where are the pickle files saved", default="reoccur/")
    parser.add_argument(
        "--player_size",
        help="optional argument to set player input size to something other than 192",
        default=192,
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    file_nonterminal = open(f"{args.destination}/reoccured-states.pkl", 'rb')
    reoccur_nonterminal = pickle.load(file_nonterminal)
    file_nonterminal.close()

    file_terminal = open(f"{args.destination}/reoccured-terminal.pkl", 'rb')
    reoccur_terminal = pickle.load(file_terminal)
    file_terminal.close()

    print(util.key_to_state(args.player_size, reoccur_nonterminal[0]), len(reoccur_nonterminal), len(reoccur_terminal))
    print(len(set(reoccur_nonterminal)), len(set(reoccur_terminal)))
    cntr = Counter(reoccur_terminal)
    for value in [20, 100, 215, 1000, 2000, 5000, 9011]:
        top_half = cntr.most_common(value)
        top_half_states = [pair[0] for pair in top_half]
        top_half_counts = [pair[1] for pair in top_half]
        print("\n")
        print(f"Total states: {sum(top_half_counts)}")
        print(f"Unique states: {len(top_half)}")
        rounds = []
        for state in top_half_states:
            input_size = 192
            play_state = util.key_to_state(input_size, state)
            rounds.append(play_state[68])
        cntr_rounds = Counter(rounds)
        print(f"Unique states round 1-5: {sum([cntr_rounds[i] for i in range(1,6)])}")
        print(f"Unique states round 6-10: {sum([cntr_rounds[i] for i in range(6, 11)])}")
        print(f"Unique states round 11+: {sum([cntr_rounds[i] for i in range(11, 21)])}")



