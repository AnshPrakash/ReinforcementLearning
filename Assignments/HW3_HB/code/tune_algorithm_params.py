import os
import argparse
import numpy as np
from pathlib import Path
import yaml
import json
import hashlib
import shutil
from mushroom_rl_extensions.environments.mujoco_envs.hopper_hurdles import HopperHurdles
import inspect
from run_single_experiment import (
    run_single_on_policy_experiment,
    run_single_off_policy_experiment,
)
from setup_algorithms import (
    setup_td3_agent,
    setup_sac_agent,
    setup_ppo_agent,
)
from torch import manual_seed

import inspect


def call_with_dict(func, args_dict):
    # get the function's parameters
    func_params = inspect.signature(func).parameters
    # filter out the arguments that are not in the function's parameters
    filtered_args = {k: v for k, v in args_dict.items() if k in func_params}
    return func(**filtered_args)


def tune_ppo(
    log_dir,
    seed,
    env_str,
    horizon,
    gamma,
    n_features,
    batch_size,
    learning_rate,
    eps,
    n_epochs_policy,
    lam,
    ent_coef,
    n_epochs,
    n_steps_training_per_epoch,
    n_steps_per_fit,
    n_episodes_test_per_epoch,
):
    # Seed
    np.random.seed(seed)
    manual_seed(seed)

    mdp = HopperHurdles(env_str, horizon, gamma)

    # Agent
    agent = setup_ppo_agent(
        mdp,
        n_features=n_features,
        lr=learning_rate,
        eps=eps,
        batch_size=batch_size,
        n_epochs_policy=n_epochs_policy,
        lam=lam,
        ent_coeff=ent_coef,
    )

    # Run single experiment
    Js, Rs, Vs, Hs = run_single_on_policy_experiment(
        log_dir,
        seed,
        agent,
        mdp,
        n_epochs=n_epochs,
        n_steps_training_per_epoch=n_steps_training_per_epoch,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test_per_epoch=n_episodes_test_per_epoch,
    )

    return Js, Rs, Vs, Hs


def tune_td3(
    log_dir,
    seed,
    env_str,
    horizon,
    gamma,
    n_features,
    batch_size,
    learning_rate,
    policy_sigma,
    initial_replay_size,
    max_replay_size,
    tau,
    n_epochs,
    n_steps_training_per_epoch,
    n_steps_per_fit,
    n_episodes_test_per_epoch,
):
    # Seed
    np.random.seed(seed)
    manual_seed(seed)

    # MDP
    horizon = horizon
    gamma = gamma

    mdp = HopperHurdles(env_str, horizon, gamma)

    # Agent
    agent = setup_td3_agent(
        mdp,
        n_features=n_features,
        lr=learning_rate,
        policy_sigma=policy_sigma,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        tau=tau,
    )
    # Run single experiment
    Js, Rs, Vs, Hs = run_single_off_policy_experiment(
        log_dir,
        seed,
        agent,
        mdp,
        n_epochs=n_epochs,
        n_steps_initial=initial_replay_size,
        n_steps_training_per_epoch=n_steps_training_per_epoch,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test_per_epoch=n_episodes_test_per_epoch,
    )

    return Js, Rs, Vs, Hs


def tune_sac(
    log_dir,
    seed,
    env_str,
    horizon,
    gamma,
    n_features,
    learning_rate,
    lr_alpha,
    batch_size,
    initial_replay_size,
    max_replay_size,
    tau,
    warmup_transitions,
    n_epochs,
    n_steps_training_per_epoch,
    n_steps_per_fit,
    n_episodes_test_per_epoch,
):
    # Seed
    np.random.seed(seed)
    manual_seed(seed)

    # MDP
    horizon = horizon
    gamma = gamma

    mdp = HopperHurdles(env_str, horizon, gamma)

    # Agent
    agent = setup_sac_agent(
        mdp,
        n_features=n_features,
        lr=learning_rate,
        lr_alpha=lr_alpha,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        tau=tau,
        warmup_transitions=warmup_transitions,
    )

    # Run single experiment
    Js, Rs, Vs, Hs = run_single_off_policy_experiment(
        log_dir,
        seed,
        agent,
        mdp,
        n_epochs=n_epochs,
        n_steps_initial=initial_replay_size,
        n_steps_training_per_epoch=n_steps_training_per_epoch,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test_per_epoch=n_episodes_test_per_epoch,
    )

    return Js, Rs, Vs, Hs


def create_logger_from_cfg(config_file):
    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    results_dir = cfg["results_dir"]
    seed = cfg["seed"]
    algorithm = cfg["algorithm"]

    env_str = cfg["env_str"]

    # generate unique hash based on the config
    cfg_str = json.dumps(cfg, sort_keys=True)
    hash_obj = hashlib.sha256(cfg_str.encode())
    cfg_hash = hash_obj.hexdigest()[:8]
    save_dir = os.path.join(results_dir, env_str, algorithm, str(seed), cfg_hash)
    print(f"INFO: Logging to {save_dir}")
    # Logging
    log_dir = Path(save_dir)

    # update the log_dir in the config
    cfg["cfg_hash"] = cfg_hash
    cfg["log_dir"] = log_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # copy config file into save_dir
    shutil.copy(config_file, save_dir + "/config.yaml")
    return cfg


def main():
    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError("Invalid config file provided!")

    cfg = create_logger_from_cfg(args.config)

    # load paramaters from config
    log_dir = cfg["log_dir"]
    seed = cfg["seed"]
    algorithm = cfg["algorithm"]

    if algorithm == "ppo":
        Js, Rs, Vs, Hs = call_with_dict(tune_ppo, cfg)
        np.save(Path(log_dir / f"Js-{seed}.npy"), Js)
        np.save(Path(log_dir / f"Rs-{seed}.npy"), Rs)
        np.save(Path(log_dir / f"Vs-{seed}.npy"), Vs)
        np.save(Path(log_dir / f"Hs-{seed}.npy"), Hs)
        print(f"Saved: {log_dir}")

    elif algorithm == "td3":
        Js, Rs, Vs, Hs = call_with_dict(tune_td3, cfg)
        np.save(Path(log_dir / f"Js-{seed}.npy"), Js)
        np.save(Path(log_dir / f"Rs-{seed}.npy"), Rs)
        np.save(Path(log_dir / f"Vs-{seed}.npy"), Vs)
        np.save(Path(log_dir / f"Hs-{seed}.npy"), Hs)
        print(f"Saved: {log_dir}")
    elif algorithm == "sac":
        Js, Rs, Vs, Hs = call_with_dict(tune_sac, cfg)
        np.save(Path(log_dir / f"Js-{seed}.npy"), Js)
        np.save(Path(log_dir / f"Rs-{seed}.npy"), Rs)
        np.save(Path(log_dir / f"Vs-{seed}.npy"), Vs)
        np.save(Path(log_dir / f"Hs-{seed}.npy"), Hs)
        print(f"Saved: {log_dir}")
    else:
        raise ValueError("Invalid algorithm provided!")


if __name__ == "__main__":
    main()
