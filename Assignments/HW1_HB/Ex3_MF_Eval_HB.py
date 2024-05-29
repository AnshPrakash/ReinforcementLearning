import numpy as np
from Ex1_env_HB import HikerAndBear
from mushroom_rl.core import Core
from mushroom_rl.core.agent import Agent
from Ex2_DP_HB import visualize_value_matrices


figures_path = "fig/"


class NumpyPolicyAgent(Agent):
    """
    This class implements a simple agent that follows a fixed policy
    based on a numpy array.
    """

    def __init__(self, policy):
        self.policy = policy

    def draw_action(self, state):
        return self.policy[state]

    def episode_start(self):
        """
        Called by the agent when a new episode starts.

        """
        pass


def MC_policy_eval(agent, env, num_rollouts, gamma):
    """
    Monte Carlo Policy Evaluation
    Args:
        agent: np.array
            The policy to evaluate.
        mdp: MDP
            The MDP to evaluate the policy on.
        num_rollouts: int
            The number of rollouts to use.
        gamma: float
            The discount factor.
    Returns:
        np.array
            The value function of the policy.
    """

    ### YOUR CODE HERE ###

    ######################

    return value


def TD_policy_eval(agent, env, num_rollouts, alpha, gamma, n):
    """
    n-step TD Policy Evaluation
    Args:
        agent: np.array
            The policy to evaluate.
        mdp: MDP
            The MDP to evaluate the policy on.
        num_rollouts: int
            The number of rollouts to use.
        gamma: float
            The discount factor.
        alpha: float
            The learning rate.
        n: int
            The lookahead for the n-step TD.
    Returns:
        np.array
            The value function of the policy.
    """

    ### YOUR CODE HERE ###

    ######################

    return value


policy = np.load("/home/aryaman/Desktop/assignments-2024/1/HW1_HB_v4/pi_left_10.npy")
agent = NumpyPolicyAgent(policy)
env = HikerAndBear()
V_MC = MC_policy_eval(agent, env, 1000, 0.9)
visualize_value_matrices(V_MC, "MC_every_visit_policy_eval")

V_TD = TD_policy_eval(agent, env, 1000, 0.1, 0.9, 5)
visualize_value_matrices(V_TD, "TD_5_policy_eval")
