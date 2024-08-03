import numpy as np
from tqdm import trange

from mushroom_rl.core import Core
from mushroom_rl.core.logger import Logger
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl_extensions.utils.dataset import compute_Q, compute_V, compute_H


def run_single_off_policy_experiment(
    log_dir,
    seed,
    agent,
    mdp,
    n_epochs,
    n_steps_initial,
    n_steps_training_per_epoch,
    n_steps_per_fit,
    n_episodes_test_per_epoch,
):
    # Logger
    logger = Logger(
        log_name="logger",
        results_dir=log_dir,
        log_console=True,
        seed=seed,
        console_log_level=30,
    )
    logger.strong_line()
    logger.info("Experiment Algorithm: " + type(agent).__name__)

    # Algorithm
    core = Core(agent, mdp)

    # Metrics
    Js = []
    Rs = []
    Vs = []
    Hs = []

    # Evaluation before training
    dataset = core.evaluate(n_episodes=n_episodes_test_per_epoch, quiet=True, render=False)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    Js.append(J)
    R = np.mean(compute_J(dataset))
    Rs.append(R)
    V = np.mean(compute_Q(dataset, agent))
    Vs.append(V)
    H = np.mean(compute_H(dataset, agent))
    Hs.append(H)
    logger.epoch_info(0, J=J, R=R, V=V, H=H)

    core.learn(n_steps=n_steps_initial, n_steps_per_fit=n_steps_initial)  # Off-policy warmup transitions

    for n in trange(n_epochs, leave=False):
        # TODO: use the learn function in the MushroomRL core object to train the agent
        # [START YOUR CODE HERE]
        core.learn(n_steps=n_steps_training_per_epoch, n_steps_per_fit=n_steps_per_fit, quiet=True)
        # [END YOUR CODE HERE]

        dataset = core.evaluate(n_episodes=n_episodes_test_per_epoch, quiet=True, render=False)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        Js.append(J)
        R = np.mean(compute_J(dataset))
        Rs.append(R)
        V = np.mean(compute_Q(dataset, agent))
        Vs.append(V)
        H = np.mean(compute_H(dataset, agent))
        Hs.append(H)

        # Log info & save best agent
        logger.epoch_info(n + 1, J=J, R=R, V=V, H=H)
        logger.log_best_agent(agent, J)
    return Js, Rs, Vs, Hs


def run_single_on_policy_experiment(
    log_dir,
    seed,
    agent,
    mdp,
    n_epochs,
    n_steps_training_per_epoch,
    n_steps_per_fit,
    n_episodes_test_per_epoch,
):
    # Logger
    logger = Logger(
        log_name="logger",
        results_dir=log_dir,
        log_console=True,
        seed=seed,
        console_log_level=30,
    )
    logger.strong_line()
    logger.info("Experiment Algorithm: " + type(agent).__name__)

    # Algorithm
    core = Core(agent, mdp)

    # Metrics
    Js = []
    Rs = []
    Vs = []
    Hs = []

    # Evaluation before training
    dataset = core.evaluate(n_episodes=n_episodes_test_per_epoch, quiet=True, render=False)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    Js.append(J)
    R = np.mean(compute_J(dataset))
    Rs.append(R)
    V = np.mean(compute_V(dataset, agent._V))
    Vs.append(V)
    H = np.mean(compute_H(dataset, agent))
    Hs.append(H)
    logger.epoch_info(0, J=J, R=R, V=V, H=H)

    for n in trange(n_epochs, leave=False):
        core.learn(
            n_steps=n_steps_training_per_epoch,
            n_steps_per_fit=n_steps_per_fit,
            quiet=True,
            render=False,
        )

        dataset = core.evaluate(n_episodes=n_episodes_test_per_epoch, quiet=True, render=False)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        Js.append(J)
        R = np.mean(compute_J(dataset))
        Rs.append(R)
        V = np.mean(compute_V(dataset, agent._V))
        Vs.append(V)
        H = np.mean(compute_H(dataset, agent))
        Hs.append(H)

        # Log info & save best agent
        logger.epoch_info(n + 1, J=J, R=R, V=V, H=H)
        logger.log_best_agent(agent, J)
    return Js, Rs, Vs, Hs
