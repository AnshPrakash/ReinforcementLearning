# utils
from joblib import Parallel, delayed
import os

# deeplearning framework
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# mushroom
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter, LinearParameter
from mushroom_rl.utils.dataset import (
    compute_J,
    compute_episodes_length,
    get_init_states,
)
from mushroom_rl.algorithms.value.dqn import DQN, DoubleDQN, AveragedDQN, MaxminDQN
from mushroom_rl.approximators.parametric.torch_approximator import *

# hw2
from hw2.algorithms.value import DQNwoReplayBuffer, DoubleMaxminDQN, NaiveDQN
from hw2.utils.argparser import argparser
from hw2.environments import MinAtar


class QConvNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d( in_channels= n_input,
                               out_channels = 16,
                               kernel_size = 3,
                               stride=1
                            ) # [YOUR CODE!] [ DONE!]

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden  =  nn.Linear(num_linear_units, 128) # [YOUR CODE!] [ DONE!]

        # Output layer:
        self.output = nn.Linear(128, n_output) # [YOUR CODE!] [ DONE!]

    def forward(self, state, action=None):
        # Apply relu to the output of self.conv, self.fc_hidden layers.
        # [YOUR CODE!] [ DONE!]
        # print("input", state.shape)
        # print("Action", action)
        x = self.conv(state)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc_hidden(x)
        x = torch.relu(x)
        q = self.output(x)
        # print("output", q.shape)
        
        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1,action))
            return q_acted
        


# Implement the compute_V function to compute the estimated value function at the initial states.
# Follow the Deep Q-Learning practical session to implement this function.
def compute_V(dataset, q):
    """
    Args:
        dataset (list): the dataset to consider;
        q (torch.nn.module): the q approximator.

    Returns:
        The approximated value of initial state of each episode in the dataset.

    """
    # [YOUR CODE!] [ DONE!]
    vs = []
    
    initial_states =  get_init_states(dataset)
    
    for s0 in initial_states:
        v = np.max(q.predict(np.expand_dims(s0, axis=0)))
        vs.append(v)
        
    if len(vs) == 0:
        return [0.]
    
    return vs


def run(
    pi,
    agent_class,
    mdp,
    params,
    approximator_params,
    num_epochs=100,
    n_steps=1000,
    n_episodes_test=5,
    seed=1,
    exp_name=None,
    log_dir=None,
    exp_id=0,
):
    np.random.seed(seed)

    single_logger = Logger(
        f"seed_{exp_id if seed is None else seed}",
        results_dir=log_dir,
        log_console=True,
    )
    log_dir = single_logger.path

    epsilon = LinearParameter(
        params["eps"],
        threshold_value=params["final_eps"],
        n=params["exploration_steps"],
    )
    epsilon_test = Parameter(value=params["eps_test"])
    policy = pi(epsilon=epsilon)

    algorithm_params = dict(
        approximator_params=approximator_params,
        initial_replay_size=params["initial_replay_size"],
        max_replay_size=params["max_replay_size"],
        batch_size=params["batch_size"],
        target_update_frequency=params["target_update_frequency"]
        // params["train_frequency"],
    )

    # remove the unneccessary arguments for NaiveDQN
    if agent_class.__name__ == "NaiveDQN":
        del algorithm_params["initial_replay_size"]
        del algorithm_params["max_replay_size"]
        del algorithm_params["target_update_frequency"]

    if params["n_approximators"] > 1:
        agent = agent_class(
            mdp.info,
            policy,
            TorchApproximator,
            params["n_approximators"],
            **algorithm_params,
        )
    else:
        agent = agent_class(mdp.info, policy, TorchApproximator, **algorithm_params)

    core = Core(agent, mdp)

    Js = []
    ELs = []
    Vs = []

    # collect samples to initialize the replay_memory
    # [YOUR CODE!] [DONE!]
    
    # fit_standard add dataset to replay buffer
    # https://mushroomrl.readthedocs.io/en/1.5.3/_modules/mushroom_rl/algorithms/value/dqn/dqn.html
    # https://github.com/MushroomRL/mushroom-rl/blob/1a4f54ed23101fbcf48bfe2022a5ff74c37c5b8f/mushroom_rl/core/core.py#L167
    initial_replay_size = params["initial_replay_size"]
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)
    
    # evaluate the initial policy given the randomly initialized Q function
    # [YOUR CODE!] [DONE!]
    agent.policy.set_epsilon(epsilon_test)
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
    J = compute_J(dataset, mdp.info.gamma)
    V = compute_V(dataset, agent.policy.get_q())
    episode_lengths = compute_episodes_length(dataset)
    Js.append(np.mean(J))
    Vs.append(np.mean(V))
    ELs.append(np.mean(episode_lengths))
    
    train_frequency = params["train_frequency"]
    n_episodes = 20
    # implement the training and evaluation loop
    # [YOUR CODE!] [DONE!]
    for epoch in range(num_epochs):
        agent.policy.set_epsilon(epsilon)
        core.learn(n_episodes=n_episodes, n_steps_per_fit=train_frequency)
        agent.policy.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
        J = compute_J(dataset, mdp.info.gamma)
        V = compute_V(dataset, agent.policy.get_q())
        episode_lengths = compute_episodes_length(dataset)
        Js.append(np.mean(J))
        Vs.append(np.mean(V))
        ELs.append(np.mean(episode_lengths))
        
    return Js, ELs, Vs


def seeding_experiment(
    pi,
    agent,
    mdp,
    params,
    approximator_params,
    num_run=25,
    num_epochs=100,
    n_steps=1000,
    n_episodes_test=5,
    exp_name=None,
    log_dir=None,
):
    seeds = np.random.randint(0, 1e5, size=(num_run,))  # list of seeds

    # pipeline
    out = Parallel(n_jobs=-1)(
        delayed(run)(
            pi,
            agent,
            mdp,
            params,
            approximator_params,
            num_epochs=num_epochs,
            n_steps=n_steps,
            n_episodes_test=n_episodes_test,
            seed=seeds[i],
            exp_name=exp_name,
            log_dir=log_dir,
            exp_id=i,
        )
        for i in range(num_run)
    )

    Js = np.array([o[0] for o in out])
    ELs = np.array([o[1] for o in out])
    Vs = np.array([o[2] for o in out])

    np.save(os.path.join(log_dir, "Js.npy"), Js)
    np.save(os.path.join(log_dir, "ELs.npy"), ELs)
    np.save(os.path.join(log_dir, "Vs.npy"), Vs)

    return Js, ELs, Vs


if __name__ == "__main__":

    args = argparser()

    num_run = args.n_exp
    num_epochs = args.n_epochs
    n_steps = args.n_steps
    n_episodes_test = args.n_episodes_test
    gamma = args.gamma
    learning_rate = args.lr

    policy_class = EpsGreedy
    n_approximators = 1
    optimizer = {
        "class": optim.RMSprop,
        "params": {"lr": learning_rate, "alpha": 0.95, "centered": True, "eps": 0.01},
    }
    loss = F.smooth_l1_loss

    # Based on the args.agent_class value, please assign the right class to agent_class
    # For algorithms which use more than one model for the online and target approximators, please update the n_approximators to 2.
    # [YOUR CODE!] [DONE!]
    def get_class(class_name):
        if "NaiveDQN": 
            return NaiveDQN
        if "DQNwoReplayBuffer": 
            return DQNwoReplayBuffer
        if "DQN": 
            return DQN
        if "DoubleDQN": 
            return DoubleDQN
        if "MaxminDQN": 
            return  MaxminDQN
        if "DoubleMaxminDQN": 
            return DoubleMaxminDQN
        if "AveragedDQN": 
            return AveragedDQN
        return None
        
    agent_class = get_class(args.agent_class)
        
    params = {
        "eps": args.eps,
        "final_eps": args.final_eps,
        "eps_test": args.final_eps,
        "initial_replay_size": args.initial_replay_size,
        "max_replay_size": args.max_replay_size,
        "batch_size": args.batch_size,
        "target_update_frequency": args.target_update_frequency,
        "exploration_steps": args.exploration_steps,
        "train_frequency": args.train_frequency,
        "n_approximators": n_approximators,
    }

    env_name = args.env_name
    exp_name = (
        "%s_%s" % (env_name, agent_class.__name__)
        if args.exp_name is None
        else args.exp_name
    )

    log_dir = os.path.join("logs", env_name, agent_class.__name__)
    os.makedirs(log_dir, exist_ok=True)

    logger = Logger(
        exp_name,
        results_dir=log_dir,
        log_console=True,
        use_timestamp=args.use_timestamp,
    )
    logger.strong_line()
    logger.info("Experiment Algorithm: " + agent_class.__name__)
    log_dir = logger.path

    # mdp
    mdp = MinAtar(env_name, gamma=gamma)

    # network
    if args.network == "QConvNetwork":
        network = QConvNetwork
    else:
        raise NotImplementedError

    # Approximator
    approximator_params = dict(
        network=network,
        loss=loss,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
        n_features=args.n_features,
        optimizer=optimizer,
        use_cuda=args.use_cuda,
    )

    Js, ELs, Vs = seeding_experiment(
        policy_class,
        agent_class,
        mdp,
        params,
        approximator_params,
        num_run=num_run,
        num_epochs=num_epochs,
        n_steps=n_steps,
        n_episodes_test=n_episodes_test,
        exp_name=exp_name,
        log_dir=log_dir,
    )
