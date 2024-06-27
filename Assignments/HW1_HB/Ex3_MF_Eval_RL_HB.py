from Ex1_env_HB import HikerAndBear
from Ex2_DP_HB import visualize_value_matrices
from mushroom_rl.core.serialization import Serializable
from Ex3_MF_Eval_HB import MC_policy_eval, TD_policy_eval


figures_path = "fig/"
agents_path = "agents/"

env = HikerAndBear()

agent_path = agents_path + "EpsGreedy_QLearning.zip"
agent = Serializable.load(agent_path)
V_MC = MC_policy_eval(agent, env, 1000, 0.9)
visualize_value_matrices(V_MC, "Q_LEARNING_MC_eval")
V_TD = TD_policy_eval(agent, env, 1000, 0.1, 0.9, 5)
visualize_value_matrices(V_TD, "Q_LEARNING_TD5_eval")

### YOUR CODE HERE ###


# SARSA

agent_path = agents_path + "EpsGreedy_SARSA.zip"
agent = Serializable.load(agent_path)
V_MC = MC_policy_eval(agent, env, 1000, 0.9)
visualize_value_matrices(V_MC, "SARSA_MC_eval")
V_TD = TD_policy_eval(agent, env, 1000, 0.1, 0.9, 5)
visualize_value_matrices(V_TD, "SARSA_TD5_eval")

######################

### YOUR CODE HERE ###

# SARSALambda
agent_path = agents_path + "EpsGreedy_SARSALambda.zip"
agent = Serializable.load(agent_path)
V_MC = MC_policy_eval(agent, env, 1000, 0.9)
visualize_value_matrices(V_MC, "SARSALambda_MC_eval")
V_TD = TD_policy_eval(agent, env, 1000, 0.1, 0.9, 5)
visualize_value_matrices(V_TD, "SARSALambda_TD5_eval")


######################


"""
Q&A

1.3.3
Observations:
Looking at the value maps between MC and TD(5) clearly shows better estimation by MC. This is because TD(5) can only propagate rewards upto 5 steps, whereas MC gets the goal state reward to the initial state for every episode.
Following the increasing values in the grid also shows the optimal path followed by each of the algorithms.
SARSA lambda:  It seems to have found the simplest and safest path to goal states. Additionally, it also assigns better values to equivalent states.
SARSA: It found better policy than SARSA lambda as it is not wasting too many steps going too far from the bear when it has no more advantage. 
Q-Learning: It has also found optimal policy, but it is not too stringent on following limit paths from different initial states(Notice values close to initial  states). It also gives value to equivalent states, where this information is missing in vanila SARSA.

"""