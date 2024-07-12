# data handling
import numpy as np

# mushroom
from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.core import Core, Logger
from cart_pole import CartPole
from mushroom_rl.features import Features
from mushroom_rl.features.basis import FourierBasis, GaussianRBF
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.utils.plot import plot_mean_conf
from mushroom_rl.utils.parameters import Parameter

# utils
from joblib import Parallel, delayed
from tqdm import trange
import matplotlib.pyplot as plt


def experiment(params, seed, exp_id=None):

    n_epochs = params["n_epochs"]
    n_steps = params["n_steps"]
    n_episodes_test = params["n_episodes_test"]
    render = params["render"]
    alpha = params["alpha"]
    n_harmonics = params["n_harmonics"]
    n_centers = params["n_centers"]
    n_tilings = params["n_tilings"]
    n_tiles = params["n_tiles"]

    learning_rate = learning_rate = Parameter(alpha)

    func_approx = params["func_approx"]

    np.random.seed(seed)

    # Logger
    logger = Logger("CartPole_TrueOnlineSARSALambda", results_dir=None)
    logger.strong_line()
    logger.info("Environment: CartPole")
    logger.info("Experiment Algorithm: TrueOnlineSARSALambda")
    logger.info(f"Function Approximator: {func_approx}")

    # MDP
    mdp = CartPole(horizon=2000)
    low = mdp.info.observation_space.low
    high = mdp.info.observation_space.high

    # Policy
    epsilon = Parameter(value=0.005)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    if func_approx == "tiles":
        # [YOUR CODE!]
        tilings = Tiles.generate(
                                 n_tilings, [n_tiles, n_tiles],
                                 low,
                                 high
                                )

        features = Features(tilings=tilings)

    elif func_approx == "gaussian":
        # [YOUR CODE!]
        n_features = len(low)
        grbf = GaussianRBF.generate(
                                        [n_centers]*n_features,
                                         np.array(low),
                                         np.array(high)
                                        )
        features = Features(basis_list=grbf)

    elif func_approx == "fourier":
        # [YOUR CODE!]
        fourier_basis = FourierBasis.generate(np.array(low), np.array(high), n=n_harmonics)
        features = Features(basis_list=fourier_basis)

    elif func_approx == "raw":
        # [YOUR CODE!]
        id = lambda x : x
        features = Features( n_outputs= 2, function= id )

    approximator_params = dict(
        input_shape=(features.size,),
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
    )
    algorithm_params = {
        "learning_rate": learning_rate,
        "lambda_coeff": params["lambda_coeff"],
    }

    # [YOUR CODE!] 
    agent = TrueOnlineSARSALambda(mdp.info, pi,
                        approximator_params=approximator_params,
                        features=features, **algorithm_params)


    # Algorithm
    core = Core(agent, mdp)

    # Train
    Js = []
    dJs = []
    ELs = []
    # [YOUR CODE!]
    for i in range(n_epochs):
        agent.policy.set_epsilon(epsilon)
        
        
        # train the agent
        core.learn(n_episodes = None, n_steps = n_steps, n_steps_per_fit = 1)
        ######################


        # evaluate the greedy policy
        # set epsilon to 0 for testing
        agent.policy.set_epsilon(Parameter(0.0))
        dataset = core.evaluate(n_episodes = n_episodes_test, quiet=True)
        dJ = np.mean(compute_J(dataset,mdp.info.gamma))
        J = np.mean(compute_J(dataset))
        EL = np.mean(compute_episodes_length(dataset))
        
        dJs.append(dJ)
        Js.append(J)
        ELs.append(EL)
        
        ######################
    

    if exp_id == 0:
        core.evaluate(n_episodes=1, render=False, quiet=True)

    return Js, dJs, ELs


if __name__ == "__main__":
    n_experiment = 30

    seeds = np.random.randint(0, 1e5, size=(n_experiment,))

    func_approx = [ "raw", "fourier", "gaussian", "tiles"]
    # func_approx = ["raw", "fourier", "gaussian", "tiles"]
    # func_approx = ["gaussian"]

    params = {
        "n_epochs": 30,
        "n_steps": 2000,
        "n_episodes_test": 5,
        "render": False,
        "alpha": 0.01,  # [TUNE PARAMETER!]: Modify the learning rate for each algorithm (if needed)!
        "n_harmonics": 5, # [TUNE PARAMETER!]
        "n_centers":  5,  # [TUNE PARAMETER!]
        "n_tilings":  5,  # [TUNE PARAMETER!]
        "n_tiles":  5,    # [TUNE PARAMETER!]
        "lambda_coeff": 0.9,
    }

    J = {}
    dJ = {}
    EL = {}
    for f in func_approx:

        params["func_approx"] = f

        data = Parallel(n_jobs=-1)(
            delayed(experiment)(params, seeds[i], exp_id=i) for i in range(n_experiment)
        )
        # data = experiment(params, seeds[0], exp_id = 0)

        J[f] = np.array([j[0] for j in data])
        dJ[f] = np.array([j[1] for j in data])
        EL[f] = np.array([j[2] for j in data])

    # Plot Undiscounted Average Return
    fig = plt.figure()
    ax = fig.gca()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (key, value) in enumerate(J.items()):
        plot_mean_conf(value, ax, color=colors[i], label=key)

    plt.legend(loc=4)
    plt.xlabel("Epochs")
    plt.ylabel("Average Return")
    plt.savefig("J_func_approx.png")

    # Plot discounted Average Return
    fig = plt.figure()
    ax = fig.gca()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (key, value) in enumerate(dJ.items()):
        plot_mean_conf(value, ax, color=colors[i], label=key)

    plt.legend(loc=4)
    plt.xlabel("Epochs")
    plt.ylabel("Discounted Average Return")
    plt.savefig("dJ_func_approx.png")

    # Plot Episode Length
    fig = plt.figure()
    ax = fig.gca()

    for i, (key, value) in enumerate(EL.items()):
        plot_mean_conf(value, ax, color=colors[i], label=key)

    plt.legend(loc=4)
    plt.xlabel("Epochs")
    plt.ylabel("Episode Length")
    plt.savefig("EL_func_approx.png")
