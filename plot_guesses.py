import argparse
import os
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    """
    Function to parse arguments.
    Returns:
    parser: Argument parser containing arguments.
    """

    parser = argparse.ArgumentParser(description="Make plots")
    parser.add_argument("file", help="log file with guesses", type=str)

    return parser.parse_args()


def plot_distribution(
    x_values: list,
    y_values: list,
    save_folder: str,
    name: str,
) -> None:
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(x_values, y_values)
    fig.savefig(f"wizard/plots/{save_folder}/{name}_plot")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    filepath = os.path.join("logs", "comp")
    filepath = os.path.join(filepath, args.file)

    f = open(filepath, "r")
    lines = f.readlines()[-50:]
    guesses = None
    actual = None
    for line in lines:
        if line.startswith("Guesses:"):
            guesses = line.split('[')[1].rstrip(']').split()
        if line.startswith("Actual:"):
            actual = line.split('[')[1].rstrip(']').split()

    if guesses is not None:
        plot_distribution([x for x in range(0, 21)], guesses, "bar-test", "guesses")
    if actual is not None:
        plot_distribution([x for x in range(0, 21)], actual, "bar-test", "actual")
