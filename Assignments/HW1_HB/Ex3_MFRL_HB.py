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
    np.random.seed(seed)
    # create the policy
    pi = policy(params["eps"])
    # create the agent

    ### YOUR CODE HERE ###

    if agent == SARSALambda:
        agent = SARSALambda(mdp.info, pi, learning_rate = params["learning_rate"], lambda_coeff=params["lambda"])

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
        core.learn(n_episodes= None, n_steps = n_steps, n_steps_per_fit = 1)
        ######################

        
        ### YOUR CODE HERE ###

        # evaluate the policy used in training
        dataset = core.evaluate(n_episodes = n_episodes_test, quiet=True)
        J = np.mean(compute_J(dataset,mdp.info.gamma))
        Js_behaviour[i] = J

        # evaluate the greedy policy
        # set epsilon to 0 for testing
        agent.policy.set_epsilon(Parameter(0.0))
        dataset = core.evaluate(n_episodes = n_episodes_test, quiet=True)
        J = np.mean(compute_J(dataset,mdp.info.gamma))
        Js_greedy[i] = J

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
            # Js_seeds = []
            Js_behaviour_seeds = []
            Js_greedy_seeds = []
            for i in trange(num_runs):
                print("Run: ", i)
                ### YOUR CODE HERE ###

                # train the agent

                Js_behaviour, Js_greedy, core = run(policy = pi, agent = agent,
                                                     params = params,num_epochs=num_epochs,
                                                     n_steps=n_steps, n_episodes_test=n_episodes_test, seed=seeds[i])
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
    return (data_behaviour, data_greedy)


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
data_behaviour, data_greedy = run_experiment (
                                    policies,
                                    agents,
                                    params,
                                    num_runs=10,
                                    num_epochs=100,
                                    n_steps=1000,
                                    n_episodes_test=5,
                                )

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



"""
Q&A

1.3.2
Q. Which algorithm learns the fastest? Can you think of a reason why?
Answer:
SARSALambda learns the fastest. It could be because of following reasons:
Q-Learning tries to use behavioral and greedy policy together and using greedy policy in the initial stage leads to a lot of mistakes which causes slower learning.
SARSA lambda uses eligibility trace which helps to assign rewards proportionally, whereas SARSA doesn’t try to solve this credit assignment and takes longer to converge.

Q. Compare the final performance of the greedy policies of Q Learning and SARSA. Which one performs better, and why?
Ans:
Final performance of the greedy policies of Q learning and SARSA are equivalent. This is because both the algorithms converged as we can also see the J-curve plateau.

Q. Compare the behaviour policies of Q Learning and SARSA. How does the final performance compare? Why is this different to the final performance of the greedy policies?

Again, both these policies converged to similar J-values as the J-curve started to plateau.  
However, behavioral policy is different from greedy policy as the converged values are lower than in greedy policy. This is because the greedy policy moved closer to the optimal policy and behavioural policy still takes random actions for exploration and these could be suboptimal.


"""