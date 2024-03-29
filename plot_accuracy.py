import argparse
import matplotlib
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
    parser.add_argument("arg", help="plotting arg to grey out specific plots", type=str)
    return parser.parse_args()


def plot_accuracy(
    x_values: list,
    y_values: list,
    names: list,
    save_folder: str,
    name: str,
    substr: str,
) -> None:
    fig, ax = plt.subplots()
    for i in range(len(x_values)):
        if len(x_values[i]) > 1:
            if substr == "none":
                plt.plot(x_values[i], y_values[i], label=names[i], linestyle="-")
            elif substr.startswith("not_"):
                if any([sub in names[i] for sub in substr.split("_")]):
                    pass
                    # plt.plot(x_values[i], y_values[i], label=names[i], linestyle="-", color="grey")
                else:
                    plt.plot(x_values[i], y_values[i], label=names[i], linestyle="-")
            elif substr in names[i]:
                plt.plot(x_values[i], y_values[i], label=names[i], linestyle="-")
            else:
                pass
                # plt.plot(x_values[i], y_values[i], label=names[i], linestyle="-", color="grey")
        else:
            # plot horizontal line for fixed agents
            plt.axhline(y=y_values[i][0], label=names[i], linestyle="--")
    plt.xlabel("Iterations (n)", fontsize=13)
    plt.xticks(fontsize=8)
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    if name == "accuracy":
        plt.ylabel("Accuracy in %", fontsize=13)
    elif name == "wins":
        plt.ylabel("Wins of 1000 games", fontsize=13)
    elif name == "relative_scores":
        plt.ylabel("Avg. score diff to winning heuristic", fontsize=13)
    fig.legend(loc="center right", bbox_to_anchor=(0.87,0.92), ncol=3, fontsize=7)
    fig.savefig(f"wizard/plots/{save_folder}/{name}_plot_{substr}")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    x_values = []
    y_values = []
    win_totals = []
    score_avg = []
    relative_scores = []
    my_scores = []
    names = []
    for directory in sorted(next(os.walk(args.dir))[1]):
        x = []
        y = []
        win_dir = []
        score = []
        rel_score = []
        my_score = []
        if not directory.startswith("."):
            print(f"\nDIR: {directory}")
            folder_path = os.path.join(args.dir, directory)
            log_files = [file for file in os.listdir(folder_path) if file.endswith(".log")]
            if len(log_files) > 0:
                names.append(directory)
                for filename in sorted(os.listdir(folder_path)):
                    avg_distribution = [0] * 21
                    total_corr = 0
                    iters = ""
                    if filename.endswith(".log"):
                        f = open(os.path.join(folder_path, filename), "r")
                        lines = f.readlines()

                        for line in lines:
                            if line.startswith("Agents"):
                                distribution = line.split("[")[1].split("]")[0].split(", ")
                                corr = distribution[0]
                                total_corr += int(corr)

                                for i in range(21):
                                    if len(distribution) < 21:
                                        print(distribution, filename)
                                        exit()
                                    avg_distribution[i] += float(distribution[i]) / 100

                            if line.startswith("Wins: "):
                                win_dir.append(int(line[8:].split(",")[0]))
                            if line.startswith("Scores: "):
                                score.append(
                                    [round(float(x), 2) for x in line[10:-2].split(",")]
                                )
                                rel_score.append(round(score[-1][0] - max(score[-1][1:]), 2))
                                my_score.append(score[-1][0])
                            if line.startswith("Iters: "):
                                iters = line.split()[1]
                        accuracy = total_corr / 200
                        x.append(iters)
                        y.append(accuracy)

                    avg_distribution = [int(i) for i in avg_distribution]
                    print(f"Average distribution for file {filename}: {avg_distribution}")

                x_values.append(x)
                y_values.append(y)
                win_totals.append(win_dir)
                score_avg.append(score)
                relative_scores.append(rel_score)
                my_scores.append(my_score)

    print(score_avg)
    print(relative_scores)
    print(my_scores)
    folder = args.dir.split("/")[-2]
    plot_accuracy(x_values, y_values, names, folder, "accuracy", args.arg)
    plot_accuracy(x_values, win_totals, names, folder, "wins", args.arg)
    plot_accuracy(x_values, relative_scores, names, folder, "relative_scores", args.arg)
