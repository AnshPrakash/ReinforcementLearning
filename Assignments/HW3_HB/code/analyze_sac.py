import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import numpy as np
from torch import manual_seed
from mushroom_rl_extensions.environments.mujoco_envs.hopper_hurdles import HopperHurdles
from mushroom_rl.core import Core
from mushroom_rl.core.serialization import Serializable
from mushroom_rl_extensions.utils.dataset import compute_action_R


def main(model_path):
    # --- Parameters --- #
    # Seed
    np.random.seed(0)
    manual_seed(0)

    # MDP
    horizon = 10000
    gamma = 0.99
    mdp = HopperHurdles("hopper_fixed_hurdles", horizon=horizon, gamma=gamma)

    # Agent of your choice
    agent = Serializable.load(os.path.join(model_path, "logger/agent-0-best.msh"))

    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=1)

    states, value_estimates, actions = compute_action_R(agent._critic_approximator, dataset)

    # --- Plot physical quantities of the robot --- #
    fig, axes = plt.subplots(3, 1, figsize=(7, 5))
    axes = axes.flatten()

    def prepare_axis(ax, ylabel):
        ax.grid()
        ax.set_xticklabels([])
        ax.set_xlim((0, 4.5))
        ax.set_ylabel(ylabel)

        ax.axvline(0.8, c="black", linewidth=3)
        ax.axvline(2.3, c="black", linewidth=3)

    axes[0].plot(np.array(states[0])[:, 11], value_estimates[0])
    prepare_axis(axes[0], "Q(states, actions)")

    axes[1].plot(np.array(states[0])[:, 11], np.array(actions[0])[:, 2])
    prepare_axis(axes[1], "actions[:, 2]")

    axes[2].plot(np.array(states[0])[:, 11], np.array(states[0])[:, 0])
    axes[2].grid()
    axes[2].set_xlabel("states[:, 11]")
    axes[2].set_xlim((0, 4.5))
    axes[2].set_ylabel("states[:, 0]")
    axes[2].vlines(0.8, colors="black", linewidth=3, ymin=0, ymax=0.85)
    axes[2].vlines(2.3, colors="black", linewidth=3, ymin=0, ymax=0.85)

    plt.tight_layout()

    plt.savefig(f"Analyze_SAC_physical_quantities_{model_path.split('/')[-1]}_plot.pdf", bbox_inches="tight")

    # --- Plot the action-value function and the policy before the agent jumps over the first hurdle --- #
    state_before_jump = np.array(
        [
            [
                1.06524998,
                -0.02760322,
                -0.66356733,
                -0.01443749,
                0.32449337,
                1.72157482,
                0.13616555,
                -1.70326602,
                -3.68260198,
                0.45998769,
                -6.16320599,
                0.54716365,
            ]
        ]
    )
    action_0 = 0.93314886
    n_points = 100

    actions_1 = np.linspace(-1, 1, n_points)
    actions_2 = np.linspace(-1, 1, n_points)

    action_values = np.zeros((n_points, n_points))

    for idx_action_1, action_1 in enumerate(actions_1):
        for idx_action_2, action_2 in enumerate(actions_2):
            action = np.array([[action_0, action_1, action_2]])
            action_values[idx_action_1, idx_action_2] = agent._critic_approximator(state_before_jump, action)

    distribution_pi = agent.policy.distribution(state_before_jump)
    sampled_actions = np.array(
        [np.tanh(distribution_pi.rsample().detach().numpy()) for _ in range(n_points * n_points)]
    )
    action_value_sampled_actions = np.array(
        [agent._critic_approximator(state_before_jump, action) for action in sampled_actions]
    )
    average_value_on_policy = int(action_value_sampled_actions.mean())

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 3.4),
    )
    axes = axes.flatten()

    color_bar = axes[0].pcolormesh(actions_1, actions_2, action_values.T, cmap=cm.RdYlGn)
    fig.colorbar(color_bar)
    axes[0].axis("equal")
    axes[0].set_xlim(-1, 1)
    axes[0].set_xlabel("action[1]")
    axes[0].set_ylim(-1, 1)
    axes[0].set_ylabel("action[2]")
    axes[0].set_title(r"$Q(s, a)$")

    histogram, _, _, _ = axes[1].hist2d(sampled_actions[:, 0, 1], sampled_actions[:, 0, 2], bins=50, cmap=cm.Greys)
    fig.colorbar(
        cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=histogram.min(), vmax=histogram.max()), cmap=cm.Greys),
        ax=axes[1],
    )
    axes[1].axis("equal")
    axes[1].set_xlim(-1.01, 1.01)
    axes[1].set_xlabel("action[1]")
    axes[1].set_ylim(-1.01, 1.01)
    axes[1].set_ylabel("action[2]")
    axes[1].set_title(r"$\pi(s)$")

    axes[0].text(0.5, 1.7, "SAC agent's caracteristics", fontsize=25)

    axes[0].text(
        -0.6,
        1.4,
        r"$E[Q(s, \pi(s)] \approx $"
        + f"{average_value_on_policy}, where s is a state right before the first hurdle "
        + r"(x $\approx$ 0.5)",
        fontsize=15,
    )
    fig.subplots_adjust(wspace=0.4)

    plt.savefig(f"Analyze_SAC_Q_and_pi_{model_path.split('/')[-1]}_plot.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Analyze SAC trained model.")

    # Add the 'logger_path' argument
    parser.add_argument(
        "--model_path",
        "-p",
        type=str,
        help="Path to the logger directory containing the model you want to analyze, e.g., out/logs/hopper_fixed_hurdles/SAC/{seed}/{hash}",
        required=True,
    )
    # Parse the arguments
    args = parser.parse_args()

    main(args.model_path)
