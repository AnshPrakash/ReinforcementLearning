from Ex1_env_HB import HikerAndBear

from mushroom_rl.solvers.dynamic_programming import value_iteration, policy_iteration

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("tkagg")

env = HikerAndBear()

figures_path = "fig/"


def visualize_value_matrices(V, label):
    """
    This function visualizes the value matrices.
    Args:
        V: value matrix when
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ### YOUR CODE HERE ###
    # reshape the value matrices
    Vc = V.reshape((2, 10, 10))

    V1 = Vc[0]
    V2 = Vc[1]

    ######################

    ax[0].set_title("Value matrix - bear in position 0")
    ax[1].set_title("Value matrix - bear in position 1")
    ax[0].imshow(V1, cmap="jet")
    ax[1].imshow(V2, cmap="jet")

    # write the values in the cells
    for i in range(V1.shape[0]):
        for j in range(V1.shape[1]):
            ax[0].text(j, i, round(V1[i, j], 2), ha="center", va="center", color="w")
            ax[1].text(j, i, round(V2[i, j], 2), ha="center", va="center", color="w")

    # set title and show
    fig.suptitle(label)

    # save the figure
    fig.savefig(figures_path + label + ".png")


# action to arrow mapping
# 0: right, 1: up, 2: left, 3: down
actions = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}


def visualize_policy(pi, label):
    """
    This function visualizes the policy matrices.
    Args:
        pi: policy matrix
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ## YOUR CODE HERE ###
    # reshape the policy matrices
    pi_c = pi.reshape((2, 10, 10))

    pi1 = pi_c[0]
    pi2 = pi_c[1]

    ######################

    # overlay the bear
    pi1[5, 0] = 5
    pi2[5, 5] = 5

    ax[0].set_title("Policy matrix - bear in position 0")
    ax[1].set_title("Policy matrix - bear in position 1")
    ax[0].imshow(pi1, cmap="jet")
    ax[1].imshow(pi2, cmap="jet")

    # draw the policy arrows
    for i in range(pi1.shape[0]):
        for j in range(pi1.shape[1]):
            # skip if bear
            if pi1[i, j] == 5:
                ax[1].arrow(
                    j,
                    i,
                    0.25 * actions[pi2[i, j]][0],
                    0.25 * actions[pi2[i, j]][1],
                    head_width=0.2,
                    head_length=0.1,
                    color="w",
                )
                continue
            elif pi2[i, j] == 5:
                ax[0].arrow(
                    j,
                    i,
                    0.25 * actions[pi1[i, j]][0],
                    0.25 * actions[pi1[i, j]][1],
                    head_width=0.2,
                    head_length=0.1,
                    color="w",
                )
                continue
            else:
                ax[0].arrow(
                    j,
                    i,
                    0.25 * actions[pi1[i, j]][0],
                    0.25 * actions[pi1[i, j]][1],
                    head_width=0.2,
                    head_length=0.1,
                    color="w",
                )
                ax[1].arrow(
                    j,
                    i,
                    0.25 * actions[pi2[i, j]][0],
                    0.25 * actions[pi2[i, j]][1],
                    head_width=0.2,
                    head_length=0.1,
                    color="w",
                )

    # set title and show
    fig.suptitle(label)

    # save the figure
    fig.savefig(figures_path + label + ".png")


p = env.p
r = env.r
V = value_iteration(p, r, gamma=0.9, eps=0.01)
V_p, pi = policy_iteration(p, r, gamma=0.9)

# visualize the value matrices
visualize_value_matrices(V, "Value_Iteration")

# visualize the policy matrices
visualize_value_matrices(V_p, "Policy_Iteration_Value")
visualize_policy(pi, "Policy_Iteration_Policy")


def policy_iteration_left(prob, reward, gamma, iterations):
    """
    Policy iteration algorithm to solve a dynamic programming problem.

    Args:
        prob (np.ndarray): transition probability matrix;
        reward (np.ndarray): reward matrix;
        gamma (float): discount factor.
        iterations (int): number of iterations to run the policy iteration.
    Returns:
        The optimal value of each state and the optimal policy.

    """

    ### YOUR CODE HERE ###

    # modify the existing policy iteration algorithm from MushroomRL
    # start with a left initialized policy

    n_states = prob.shape[0]
    n_actions = prob.shape[1]

    policy = np.ones(n_states, dtype=int)*2
    value = np.zeros(n_states)

    iter = 0
    while iter < iterations:
        p_pi = np.zeros((n_states, n_states))
        r_pi = np.zeros(n_states)
        i = np.eye(n_states)

        for state in range(n_states):
            action = policy[state]
            p_pi_s = prob[state, action, :]
            r_pi_s = reward[state, action, :]

            p_pi[state, :] = p_pi_s.T
            r_pi[state] = p_pi_s.T.dot(r_pi_s)

        value = np.linalg.solve(i - gamma * p_pi, r_pi)


        for state in range(n_states):
            vmax = value[state]
            for action in range(n_actions):
                if action != policy[state]:
                    p_sa = prob[state, action]
                    r_sa = reward[state, action]
                    va = p_sa.T.dot(r_sa + gamma * value)
                    if va > vmax and not np.isclose(va, vmax):
                        policy[state] = action
                        vmax = va
                        
        iter = iter + 1
    ######################

    return value, policy


# Left initialised policy, 3 iterations of PI
V_p_left, pi_left = policy_iteration_left(p, r, gamma=0.9, iterations=3)
visualize_value_matrices(V_p_left, "Policy_Left_Iteration_Value_3")
visualize_policy(pi_left, "Policy_Left_Iteration_Policy_3")

# Left initialised policy, 10 iterations of PI
V_p_left, pi_left = policy_iteration_left(p, r, gamma=0.9, iterations=10)
visualize_value_matrices(V_p_left, "Policy_Left_Iteration_Value_10")
visualize_policy(pi_left, "Policy_Left_Iteration_Policy_10")

np.save("pi_left_10.npy", pi_left)
