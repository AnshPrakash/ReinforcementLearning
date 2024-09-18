import os
import sys
import numpy as np
import argparse
from torch import manual_seed
from mushroom_rl_extensions.environments.mujoco_envs.hopper_hurdles import HopperHurdles

from mushroom_rl.core import Core
from mushroom_rl.core.serialization import Serializable
from mushroom_rl.utils.dataset import compute_J
import numpy as np

# Script to render your trained agent in the environment


def main(args):
    agent_path = args.ckpt
    # check if the agent_path exists
    if not agent_path.endswith(".msh"):
        print(
            f"ERROR: {agent_path} is not a valid checkpoint file. Please provide a .msh file."
        )
        sys.exit(1)

    if not os.path.exists(agent_path):
        print(f"ERROR: {agent_path} does not exist")
        sys.exit(1)

    # Seed
    seed = args.seed
    np.random.seed(seed)
    manual_seed(seed)

    # env
    mdp = HopperHurdles("hopper_fixed_hurdles", horizon=10000, gamma=0.99)

    print("Loading agent from: ", agent_path)

    # Agent of your choice
    agent = Serializable.load(agent_path)

    core = Core(agent, mdp)
    dataset = core.evaluate(n_steps=2000, render=True)
    J = compute_J(dataset, mdp.info.gamma)
    R = compute_J(dataset)
    print(J)
    print(R)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Example script to demonstrate command line argument parsing."
    )

    # Add the '--config' argument
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the best agent checkpoint (.msh) file.",
    )

    # Add the '--seed' argument
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )

    # Parse the arguments
    args = parser.parse_args()

    main(args)
