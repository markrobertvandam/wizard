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
    parser.add_argument("dir", help="maindir with subdirs", type=str)

    return parser.parse_args()


def plot_accuracy(
    x_values: list, y_values:list, names: list, save_folder:str, name: str,
) -> None:
    print(x_values, y_values, names)
    fig = plt.figure()
    for i in range(len(x_values)):
        plt.plot(x_values[i], y_values[i], label=names[i], linestyle="-")
    plt.xlabel("Iterations (n)", fontsize=13)
    if name == "accuracy":
        plt.ylabel("Accuracy in %", fontsize=13)
    else:
        plt.ylabel("Wins of 1000 games", fontsize=13)
    fig.legend()
    fig.savefig(f"plots/{save_folder}/{name}_plot")
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    x_values = []
    y_values = []
    win_totals = []
    names = []
    for directory in os.listdir(args.dir):
        x = []
        y = []
        win_dir = []
        names.append(directory)
        folder_path = os.path.join(args.dir, directory)
        for filename in sorted(os.listdir(folder_path)):
            total_corr = 0
            if filename.endswith(".log"):
                f = open(os.path.join(folder_path, filename), "r")
                lines = f.readlines()

                for line in lines:
                    if line.startswith("Agents"):
                        corr = line.split("[")[1].split()[0]
                        total_corr += int(corr)
                    if line.startswith("Wins: "):
                        wins = int(line[8:].split(",")[0])
                iters = lines[2][7:]
                accuracy = total_corr/20000

                x.append(iters)
                y.append(accuracy)
                win_dir.append(wins)

        x_values.append(x)
        y_values.append(y)
        win_totals.append(win_dir)

    plot_accuracy(x_values, y_values, names, "test", "accuracy")
    plot_accuracy(x_values, win_totals, names, "test", "wins")
