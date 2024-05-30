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
    V = np.zeros((env.p.shape[0] ,1))
    Ns = [0]*V.shape[0]
    for _ in num_rollouts:
        
        curr_state = env.reset()
        absorbing = not np.any(env.p[curr_state[0]])
        episode = []
        while not absorbing:
            action = agent.draw_action(curr_state)
            next_state, reward, absorbing, _ = env.step(action)
            episode.append( { "state" : curr_state,
                              "action" : action,
                              "next_state" : next_state,
                              "reward" : reward
                             })
            curr_state = next_state

        for (state, action, next_state, reward) in episode:
            V[state] += reward
            Ns[state] += 1
        V = np.divide(V, Ns , out = V, where = (Ns != 0))
    
    value = V
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
