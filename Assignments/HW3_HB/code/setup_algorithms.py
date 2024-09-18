import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.algorithms.actor_critic import TD3, SAC, PPO

from mushroom_rl.policy import (
    ClippedGaussianPolicy,
    GaussianTorchPolicy,
)
from networks import (
    BoundedActorNetwork,
    ActorNetwork,
    ActionValueCriticNetwork,
    ValueCriticNetwork,
)


def setup_ppo_agent(mdp, n_features, lr, eps, batch_size, n_epochs_policy, lam, ent_coeff):
    # Actor
    policy_params = dict(
        std_0=1.0,
        n_features=n_features,
        use_cuda=False,
    )
    policy = GaussianTorchPolicy(
        ValueCriticNetwork, mdp.info.observation_space.shape, mdp.info.action_space.shape, **policy_params
    )
    # Critic
    # TODO: define the critic's input shape
    # [START YOUR CODE HERE
    critic_input_shape = mdp.info.observation_space.shape
    # [END YOUR CODE HERE]
    critic_params = dict(
        network=ActorNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": lr}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Settings
    actor_optimizer = {"class": optim.Adam, "params": {"lr": lr}}

    agent = PPO(
        mdp.info,
        policy,
        actor_optimizer,
        critic_params,
        n_epochs_policy,
        batch_size,
        eps,
        lam,
        ent_coeff,
    )

    return agent


def setup_td3_agent(
    mdp,
    n_features,
    lr,
    policy_sigma,
    batch_size,
    initial_replay_size,
    max_replay_size,
    tau,
):
    # Actor
    policy_class = ClippedGaussianPolicy
    policy_params = {
        "sigma": np.eye(mdp.info.action_space.shape[0]) * policy_sigma,
        "low": mdp.info.action_space.low,
        "high": mdp.info.action_space.high,
    }

    # TODO: define the actor's input and output shape
    # [START YOUR CODE HERE

    actor_input_shape = mdp.info.observation_space.shape
    actor_output_shape = mdp.info.action_space.shape

    # [END YOUR CODE HERE]
    actor_params = dict(
        network=BoundedActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=actor_output_shape,
        action_scaling=(mdp.info.action_space.high - mdp.info.action_space.low) / 2,
    )
    actor_optimizer = {"class": optim.Adam, "params": {"lr": lr}}

    # Critic
    # TODO: define the critic's input shape
    # [START YOUR CODE HERE
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    # [END YOUR CODE HERE]
    critic_params = dict(
        network=ActionValueCriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": lr}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Settings
    initial_replay_size = initial_replay_size
    max_replay_size = max_replay_size
    batch_size = batch_size
    tau = tau

    # Agent
    agent = TD3(
        mdp.info,
        policy_class,
        policy_params,
        actor_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        tau,
    )
    return agent


def setup_sac_agent(
    mdp,
    n_features,
    lr,
    lr_alpha,
    batch_size,
    initial_replay_size,
    max_replay_size,
    tau,
    warmup_transitions,
):
    # Actor
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
    )
    actor_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
    )
    actor_optimizer = {"class": optim.Adam, "params": {"lr": lr}}

    # Critic
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=ActionValueCriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": lr}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Settings
    initial_replay_size = initial_replay_size
    max_replay_size = max_replay_size
    batch_size = batch_size
    tau = tau
    warmup_transitions = warmup_transitions

    # Agent
    agent = SAC(
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
    )
    return agent
