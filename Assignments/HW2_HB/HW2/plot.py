# visualization
import matplotlib.pyplot as plt

# data handling
import numpy as np

# mushroom_rl
from mushroom_rl.utils.plot import plot_mean_conf

# utils
import os

data_dir = {
    "DQN": "logs/space_invaders/DQN/dqn",
    "DQNwoReplayBuffer": "logs/space_invaders/DQNwoReplayBuffer/dqn_wo_replaybuffer",
    "DoubleDQN": "logs/space_invaders/DoubleDQN/double_dqn",
    "MaxminDQN": "logs/space_invaders/MaxminDQN/maxmin_dqn",
    "DoubleMaxminDQN": "logs/space_invaders/DoubleMaxminDQN/double_maxmin_dqn",
    "AveragedDQN": "logs/space_invaders/AveragedDQN/averaged_dqn",
    "NaiveDQN": "logs/space_invaders/NaiveDQN/naive_dqn",
}

"""
Discounted Average Return
"""

grid = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "H"]]

# Plot discounted Average Return
fig, axs = plt.subplot_mosaic(grid, figsize=(15, 15))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, (ax, (key, value)) in enumerate(zip((axs.keys()), data_dir.items())):
    data = np.load(os.path.join(value, "Js.npy"))

    plot_mean_conf(data, axs[ax], color=colors[i], label=key)
    axs[ax].set_xlabel("Epochs")
    axs[ax].set_ylabel("Discounted Average Return")

    plot_mean_conf(data, axs["H"], color=colors[i], label=key)

axs["H"].set_xlabel("Epochs")
axs["H"].set_ylabel("Discounted Average Return")

plt.legend()
plt.savefig("J_minatar.png")

# Plot Value at initial states
# [YOUR_CODE!]

# Plot Episode Length
# [YOUR_CODE!]
