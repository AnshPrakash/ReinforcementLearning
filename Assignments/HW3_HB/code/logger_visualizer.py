import os, sys, glob
import re
import yaml
import argparse
import matplotlib.pyplot as plt


def main(args):
    # check if logger file in recursive subdirectories
    logger_root = args.path
    logger_path = glob.glob(logger_root + "/**/logger-*.log", recursive=True)

    config_file = os.path.join(logger_root, "config.yaml")

    if not os.path.exists(config_file):
        print(
            f"ERROR: No config file found in {logger_root}, please check the path. path must end with 8 character hash."
        )
        sys.exit(1)
    else:
        with open(config_file, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

    if len(logger_path) != 1:
        print(
            f"ERROR: No logger file found in {logger_root}, please check the path. path must end with 8 character hash."
        )
        sys.exit(1)

    else:
        logger_path = logger_path[0]

    # get algorithm from logger_path
    splits = logger_path.split("/")

    # hash
    algo = splits[3]
    logHash = splits[5]

    # check logHash is a hash
    if not len(logHash) == 8:
        print(f"ERROR: Path input is not a desired logger path")

    # read the logger file
    with open(logger_path, "r") as f:
        lines = f.readlines()

    # Lists to hold the extracted data, only visualize the latest logger file
    epochs = []
    Js = []
    Rs = []
    Vs = []
    Hs = []

    # Regular expression to find lines with epoch data
    pattern = r"Epoch (\d+) \| J: ([\d.]+) R: ([\d.]+) V: ([\d.]+) H: (-?[\d.]+)"

    # Extract data using regex
    for line in lines:
        if "###" in line:
            # clean lists
            epochs = []
            Js = []
            Rs = []
            Vs = []
            Hs = []
            continue

        match = re.search(pattern, line)
        if match:
            epoch, J, R, V, H = match.groups()
            epochs.append(int(epoch))
            Js.append(float(J))
            Rs.append(float(R))
            Vs.append(float(V))
            Hs.append(float(H))

    assert len(epochs) == len(Js) == len(Rs) == len(Vs) == len(Hs), "Data length mismatch"

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    fig.suptitle(f"Training metrics of {algo} {logHash}")
    axs[0, 0].plot(epochs, Rs, linewidth=3)
    axs[0, 0].set_title("R")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 1].plot(epochs, Js, linewidth=3)
    axs[0, 1].set_title("J")
    axs[0, 1].set_xlabel("Epoch")
    axs[1, 0].plot(epochs, Vs, linewidth=3)
    axs[1, 0].set_title("V")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 1].plot(epochs, Hs, linewidth=3)
    axs[1, 1].set_title("H")
    axs[1, 1].set_xlabel("Epoch")

    # Display the dictionary in the top right corner of the figure
    info_text = "\n".join([f"{key}: {value}" for key, value in cfg.items()])
    fig.text(
        0,
        0.7,
        info_text,
        ha="right",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.tight_layout()

    plt.savefig(f"training_metrics_{algo}_{logHash}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Visualize logger tool.")

    # Add the 'logger_path' argument
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="Path to the logger directory, e.g., out/logs/hopper_fixed_hurdles/{algo}/{seed}/{hash}",
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()

    main(args)
