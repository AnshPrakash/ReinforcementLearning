import os
from typing import Tuple
import types

import numpy as np
import mujoco
from pathlib import Path

from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.utils.spaces import Box


import numpy as np
from .hopper_utils import (
    Locomoter,
    is_convex_combination,
    task_mapping,
    normalize_reward,
)


class HopperHurdles(Locomoter):
    """
    The Hopper MuJoCo environment as presented in:
    "Infinite-Horizon Model Predictive Control for Periodic Tasks with Contacts". Tom Erez et. al.. 2012.
    """

    AVERAGE_VEL = 2.36
    MIN_VEL = -0.35
    MAX_VEL = 5.5

    def __init__(
        self,
        name,
        horizon=1000,
        gamma=0.99,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        n_substeps=4,
        normalize_reward: bool = False,
        **viewer_params,
    ):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent / "data" / f"{name}.xml").as_posix()

        assert os.path.exists(xml_path), f"File {xml_path} does not exist."

        exclude_current_positions_from_observation = False
        actuation_spec = ["thigh_joint", "leg_joint", "foot_joint"]

        observation_spec = [
            ("z_pos", "rootz", ObservationType.JOINT_POS),
            ("y_pos", "rooty", ObservationType.JOINT_POS),
            ("thigh_pos", "thigh_joint", ObservationType.JOINT_POS),
            ("leg_pos", "leg_joint", ObservationType.JOINT_POS),
            ("foot_pos", "foot_joint", ObservationType.JOINT_POS),
            ("x_vel", "rootx", ObservationType.JOINT_VEL),
            ("z_vel", "rootz", ObservationType.JOINT_VEL),
            ("y_vel", "rooty", ObservationType.JOINT_VEL),
            ("thigh_vel", "thigh_joint", ObservationType.JOINT_VEL),
            ("leg_vel", "leg_joint", ObservationType.JOINT_VEL),
            ("foot_vel", "foot_joint", ObservationType.JOINT_VEL),
        ]

        additional_data_spec = [
            ("x_pos", "rootx", ObservationType.JOINT_POS),
        ]

        if normalize_reward:
            assert is_convex_combination(forward_reward_weight, ctrl_cost_weight, healthy_reward)

        self.target_vel = self.MAX_VEL
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._normalize_reward = normalize_reward

        super().__init__(
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=actuation_spec,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        if not self._exclude_current_positions_from_observation:
            self.obs_helper.add_obs("x_pos", 1)
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)
        obs[5:] = np.clip(obs[5:], -10, 10)
        if not self._exclude_current_positions_from_observation:
            x_pos = self._read_data("x_pos")
            obs = np.concatenate([obs, x_pos])
        return obs

    def is_absorbing(self, obs):
        """Return True if the agent is unhealthy and terminate_when_unhealthy is True."""
        return self._terminate_when_unhealthy and not self._is_healthy(obs) or self._x_pos > 4.5  # early termination

    def _get_x_vel(self):
        x_pos = self._x_pos
        next_x_pos = self._next_x_pos
        return (next_x_pos - x_pos) / self.dt

    def _get_forward_reward(self, forward_reward_weight):
        forward_reward = self._get_x_vel()
        if self._normalize_reward:
            forward_reward = normalize_reward(forward_reward, self.MIN_VEL, self.MAX_VEL)
        return forward_reward_weight * forward_reward

    def _get_healthy_reward(self, obs):
        """Return the healthy reward if the agent is healthy, else 0."""
        return self._healthy_reward if self._is_healthy(obs) or self._terminate_when_unhealthy else 0

    def _get_ctrl_cost(self, action, ctrl_cost_weight):
        """Return the control cost."""
        ctrl_cost = -np.sum(np.square(action))
        if self._normalize_reward:
            ctrl_cost = normalize_reward(ctrl_cost, -len(action), 0)
        return ctrl_cost_weight * ctrl_cost

    def reward(self, obs, action, next_obs, absorbing):
        # TODO get healthy_reward, forward_reward, ctrl_cost using existing functions
        # Hint: the healthy reward is a function of next_obs
        # [START YOUR CODE HERE]

        healthy_reward = self._get_healthy_reward(next_obs)
        forward_reward = self._get_forward_reward(self._forward_reward_weight)
        ctrl_cost = self._get_ctrl_cost(action, self._ctrl_cost_weight)

        # [END YOUR CODE HERE]
        reward = (
            healthy_reward + forward_reward + ctrl_cost + 500 * float(self._x_pos > 4.5)
        )  # big reward if agent cross two hurdles
        return reward

    def _is_healthy(self, obs: np.ndarray) -> bool:
        """Check if the agent is healthy."""
        is_within_state_range = self._is_within_state_range(obs)
        is_within_z_range = self._is_within_z_range(obs)
        is_within_angle_range = self._is_within_angle_range(obs)
        return is_within_state_range and is_within_z_range and is_within_angle_range

    def _is_within_state_range(self, obs: np.ndarray) -> bool:
        """Check if state variables are within the healthy range."""
        idx = 1 if self._exclude_current_positions_from_observation else 2
        state_values = obs[idx:]
        min_state, max_state = self._healthy_state_range
        return all(min_state < value.item() < max_state for value in state_values)

    def _is_within_z_range(self, obs: np.ndarray) -> bool:
        """Check if Z position of torso is within the healthy range."""
        z_pos = self.obs_helper.get_from_obs(obs, "z_pos").item()
        min_z, max_z = self._healthy_z_range
        return min_z < z_pos < max_z

    def _is_within_angle_range(self, obs):
        """Check if Y angle of torso is within the healthy range."""
        y_angle = self.obs_helper.get_from_obs(obs, "y_pos").item()
        min_angle, max_angle = self._healthy_angle_range
        return min_angle < y_angle < max_angle

    def _get_error(
        self,
    ):
        x_vel = self._get_x_vel()
        return np.abs(x_vel - self.target_vel)

    def setup(self, obs):
        super().setup(obs)

        self._data.qpos[:] = (
            self._data.qpos + np.random.uniform(-self._reset_noise_scale, self._reset_noise_scale, self._model.nq)
        ).copy()
        self._data.qvel[:] = (
            self._data.qvel + np.random.uniform(-self._reset_noise_scale, self._reset_noise_scale, self._model.nv)
        ).copy()

        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs, action=None):
        info = {
            "x_vel": self._get_x_vel(),
            "healthy_reward": self._get_healthy_reward(obs),
            "forward_reward": self._get_forward_reward(forward_reward_weight=1),
            "error": self._get_error(),
        }
        if action is not None:
            info["ctrl_cost"] = self._get_ctrl_cost(action, ctrl_cost_weight=1)
        return info

    def _step_init(self, obs, action):
        super()._step_init(obs, action)
        self._x_pos = self._read_data("x_pos").item()

    def _step_finalize(self):
        super()._step_finalize()
        self._next_x_pos = self._read_data("x_pos").item()
