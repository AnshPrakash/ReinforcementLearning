import numpy as np

from mushroom_rl.utils.dataset import get_init_states


def compute_Q(dataset, agent):
    """
    Compute the estimated Q-value of an agent's policy for the initial state of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider
        agent (mushroom_rl agent): the agent to evaluate

    Returns:
        The approximated q-value of the initial state of each episode in the dataset.

    """
    qs = list()

    initial_states = get_init_states(dataset)

    for s0 in initial_states:
        action = agent.draw_action(np.expand_dims(s0, axis=0))
        if len(agent._critic_approximator) == 2:
            q = agent._critic_approximator.predict(
                np.expand_dims(s0, axis=0),
                np.atleast_2d(action),
                prediction="min",
            )
        else:
            q = agent._critic_approximator(np.expand_dims(s0, axis=0), action)
        qs.append(q)
    return qs


def compute_V(dataset, value_estimator):
    """
    Compute the estimated value of an agent's policy for the initial state of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider
        value_estimator (torch.nn.module): the agent's value approximator

    Returns:
        The approximated value of the initial state of each episode in the dataset.

    """
    vs = list()

    initial_states = get_init_states(dataset)

    for s0 in initial_states:
        v = value_estimator(np.expand_dims(s0, axis=0))
        vs.append(v)

    if len(vs) == 0:
        return [0.0]
    return vs


def compute_H(dataset, agent):
    """
    Compute the average policy entropy of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider;
        agent (mushroom_rl agent): the agent to evaluate

    Returns:
        The average policy entropy of each episode in the dataset.

    """
    es = list()
    es_episode = []
    for transition in dataset:
        if hasattr(agent.policy, "entropy"):
            e = agent.policy.entropy(np.atleast_2d(transition[0]))
        else:
            e = 0
        es_episode.append(e)
        if transition[-1]:  # absorbing state
            es.append(np.mean(es_episode))
            es_episode = []
    return es


def compute_action_R(value_estimator, dataset):
    states = []
    value_estimates = []
    actions = []

    states_episode = []
    value_estimates_episode = []
    actions_episode = []

    episode_steps = 0

    for i in range(len(dataset)):
        states_episode.append(dataset[i][0])
        value_estimates_episode.append(
            value_estimator(np.expand_dims(dataset[i][0], axis=0), np.expand_dims(dataset[i][1], axis=0))
        )
        actions_episode.append(dataset[i][1])

        episode_steps += 1
        if dataset[i][-1] or i == len(dataset) - 1:
            states.append(states_episode)
            value_estimates.append(value_estimates_episode)
            actions.append(actions_episode)

            states_episode = []
            value_estimates_episode = []
            actions_episode = []

            episode_steps = 0

    return states, value_estimates, actions
