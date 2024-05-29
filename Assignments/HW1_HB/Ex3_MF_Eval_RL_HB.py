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

######################

### YOUR CODE HERE ###

# SARSALambda

######################
