from Ex1_env_HB import HikerAndBear

from mushroom_rl.algorithms.value import SARSA, QLearning, SARSALambda
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter

from pathlib import Path
from tqdm import trange
import numpy as np
import matplotlib
import scipy.stats as st
from matplotlib import pyplot as plt

matplotlib.use("tkagg")

figures_path = "fig/"
agents_path = "agents/"

# action to arrow mapping
# 0: right, 1: up, 2: left, 3: down
actions = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}


def run(policy, agent, params, num_epochs=100, n_steps=1000, n_episodes_test=5, seed=0):
    # create the policy
    pi = policy(params["eps"])
    # create the agent

    ### YOUR CODE HERE ###

    if agent == SARSALambda:
        ###

    ######################

    else:
        agent = agent(mdp.info, pi, learning_rate=params["learning_rate"])

    # set up the core
    core = Core(agent, mdp)
    Js_behaviour = np.zeros(num_epochs)  # performance of behaviour policy
    Js_greedy = np.zeros(num_epochs)  # performance of policy with epsilon = 0


    for i in range(num_epochs):
        agent.policy.set_epsilon(params["eps"])
        
        ### YOUR CODE HERE ###

        # train the agent

        ######################

        
        ### YOUR CODE HERE ###

        # evaluate the policy used in training
        dataset = 

        # evaluate the greedy policy
        # set epsilon to 0 for testing
        dataset = 

        ######################

    return Js_behaviour, Js_greedy, core


def run_experiment(
    policies,
    agents,
    params,
    num_runs=10,
    num_epochs=100,
    n_steps=1000,
    n_episodes_test=5,
):
    seeds = np.random.randint(0, 10000, num_runs)  #####
    data_behaviour = {}
    data_greedy = {}

    for pi in policies:
        for agent in agents:
            print("#" * 30)
            print("Policy: ", pi.__name__)
            print("Agent : ", agent.__name__)
            print("#" * 30)
            Js_seeds = []
            for i in trange(num_runs):
                print("Run: ", i)
                ### YOUR CODE HERE ###

                # train the agent

                Js_behaviour, Js_greedy, core = 
                Js_behaviour_seeds.append(Js_behaviour)
                Js_greedy_seeds.append(Js_greedy)

                ######################
            # visualize the final seed's policy
            # core.evaluate(n_episodes=1, render=True)

            # save final agent
            agent_name = pi.__name__ + "_" + agent.__name__ + ".zip"
            core.agent.save(Path(agents_path) / agent_name, full_save=True)
            data_behaviour[pi.__name__ + "_" + agent.__name__] = np.array(
                Js_behaviour_seeds
            )
            data_greedy[pi.__name__ + "_" + agent.__name__] = np.array(Js_greedy_seeds)
    return data


# initialize the mdp
mdp = HikerAndBear()

# Logger
logger = Logger("RL_sose2024 HW1 : Hiker and Bear", results_dir=None)
logger.strong_line()
logger.info("Environment: Hiker and Bear")
logger.info("Experiment Algorithm: QLearning, SARSA, SARSALambda")

params = {
    "learning_rate": Parameter(0.05),
    "eps": Parameter(0.25),
    "lambda": Parameter(0.8),
}

policies = [EpsGreedy]
agents = [QLearning, SARSA, SARSALambda]

### YOUR CODE HERE ###

# aquire the data
data_behaviour, data_greedy = 

######################


def get_mean_and_confidence(data):
    """
    Compute the mean and 95% confidence interval
    Args:
        data (np.ndarray): Array of experiment data of shape (n_runs, n_epochs).
    Returns:
        The mean of the dataset at each epoch along with the confidence interval.
    """
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)
    interval, _ = st.t.interval(0.95, n - 1, scale=se)
    return mean, interval


def plot_mean_conf(
    data, ax, color="blue", line="-", facecolor=None, alpha=0.4, label=None
):
    """
    Method to plot mean and confidence interval for data on pyplot axes.
    """
    facecolor = color if facecolor is None else facecolor

    mean, conf = get_mean_and_confidence(np.array(data))
    upper_bound = mean + conf
    lower_bound = mean - conf

    ax.plot(mean, color=color, linestyle=line, label=label)
    ax.fill_between(
        np.arange(np.size(mean)),
        upper_bound,
        lower_bound,
        facecolor=facecolor,
        alpha=alpha,
    )


# plotting behaviour policy performance
fig = plt.figure()
ax = fig.gca()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, v in enumerate(data_behaviour.items()):
    plot_mean_conf(v[1], ax, color=colors[i], label=v[0])
plt.legend(loc=4)
plt.xlabel("Epochs")
plt.ylabel("J")
fig.suptitle("Results - Behaviour Policy")
fig.savefig(f"{figures_path}/J_behaviour_policy" + ".png")

# plotting greedy policy performance
fig = plt.figure()
ax = fig.gca()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, v in enumerate(data_greedy.items()):
    plot_mean_conf(v[1], ax, color=colors[i], label=v[0])
plt.legend(loc=4)
plt.xlabel("Epochs")
plt.ylabel("J")
fig.suptitle("Results - Greedy Policy")
fig.savefig(f"{figures_path}/J_greedy_policy" + ".png")
