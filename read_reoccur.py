import argparse
import pickle
import Playing_Agent
import utility_functions as util

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
