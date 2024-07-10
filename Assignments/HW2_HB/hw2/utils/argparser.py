import argparse


def argparser():
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_mdp = parser.add_argument_group("mdp")
    arg_mdp.add_argument(
        "--env_name", default="space_invaders", help="Name of the environment."
    )
    arg_mdp.add_argument("--gamma", type=float, default=0.99)

    arg_mem = parser.add_argument_group("Replay Memory")
    arg_mem.add_argument(
        "--initial_replay_size",
        type=int,
        default=5000,
        help="Initial size of the replay memory.",
    )
    arg_mem.add_argument(
        "--max_replay_size",
        type=int,
        default=100000,
        help="Max size of the replay memory.",
    )

    arg_alg = parser.add_argument_group("Algorithm")
    arg_alg.add_argument(
        "--agent_class",
        choices=[
            "NaiveDQN",
            "DQNwoReplayBuffer",
            "DQN",
            "DoubleDQN",
            "MaxminDQN",
            "DoubleMaxminDQN",
            "AveragedDQN",
        ],
        default="DQN",
        help="Name of the agent algorithm to use to learn.",
    )
    arg_alg.add_argument(
        "--network",
        default="QConvNetwork",
        help="Name of the network architecture used to approximate the Q function.",
    )
    arg_alg.add_argument("--eps", type=float, default=1.0)
    arg_alg.add_argument("--final_eps", type=float, default=0.1)
    arg_alg.add_argument("--exploration_steps", type=int, default=1000)
    arg_alg.add_argument("--target_update_frequency", type=int, default=1000)
    arg_alg.add_argument("--train_frequency", type=int, default=1)
    arg_alg.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for each fit of the network.",
    )
    arg_alg.add_argument("--n_epochs", type=int, default=100, help="Number of epochs.")
    arg_alg.add_argument(
        "--n_steps", type=int, default=25000, help="Number of learning steps per epoch."
    )
    arg_alg.add_argument(
        "--n_episodes_test",
        type=int,
        default=5,
        help="Number of episodes (rollouts) for evaluation per epoch.",
    )
    arg_alg.add_argument("--n_features", type=int, nargs="+", default=[128])

    arg_net = parser.add_argument_group("Optimizer")
    arg_net.add_argument(
        "--lr",
        type=float,
        default=0.00025,
        help="Learning rate value of the optimizer.",
    )

    arg_utils = parser.add_argument_group("Utils")
    arg_utils.add_argument(
        "--use_cuda",
        action="store_true",
        help="Flag specifying whether to use the GPU.",
    )  #
    arg_utils.add_argument(
        "--render_train",
        action="store_true",
        help="Flag specifying whether to render the training.",
    )  #
    arg_utils.add_argument(
        "--render_eval",
        action="store_true",
        help="Flag specifying whether to render the evaluation.",
    )  #
    arg_utils.add_argument(
        "--debug",
        action="store_true",
        help="Flag specifying whether the script has to be" "run in debug mode.",
    )  #
    arg_utils.add_argument(
        "--use_timestamp",
        action="store_true",
        help="Add timestamp to the results folder.",
    )  #
    arg_utils.add_argument(
        "--results_dir", type=str, default="logs/", help="Results directory name."
    )  #
    arg_utils.add_argument(
        "--exp_name", type=str, default=None, help="Name of the experiment."
    )  #
    arg_utils.add_argument("--n_exp", type=int, default=1)

    args = parser.parse_args()

    return args
