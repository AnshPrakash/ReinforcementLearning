from typing import Callable
import numpy as np

from mushroom_rl.environments.mujoco import MuJoCo


class Locomoter(MuJoCo):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("camera_params", {}).setdefault("follow", {})["distance"] = 15
        kwargs["camera_params"]["follow"]["elevation"] = -30
        kwargs["default_camera_mode"] = "follow"
        super().__init__(*args, **kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info = self._create_info_dictionary(obs, action)
        return obs, reward, done, info

    def _create_info_dictionary(self, obs, action=None):
        return super()._create_info_dictionary(obs)


def normalize_reward(
    reward: float,
    r_min: float,
    r_max: float,
    has_absorbing_state: bool = True,
    key: str = "None",
) -> float:
    assert (
        r_min < r_max
    ), f"r_min has to be less than r_max. r_min = {r_min}, r_max = {r_max}"
    if np.round(reward, 5) < r_min or np.round(reward, 5) > r_max:
        print(f"Warning: Clipping reward {reward} to [{r_min}, {r_max}] for key {key}.")
    reward = np.clip(reward, r_min, r_max)
    if has_absorbing_state:
        r_min = min(0, r_min)
        r_max = max(0, r_max)
    reward = (reward - r_min) / (r_max - r_min)
    assert (
        0 <= reward <= 1
    ), f"Reward not in [0, 1]. Reward = {reward}, r_min = {r_min}, r_max = {r_max}"
    return reward


def is_convex_combination(*args):
    for arg in args:
        if not (0 <= arg <= 1):
            return False
    argsum = sum(args)
    return np.isclose(argsum, 1)


def absolute_deviation(self, forward_reward_weight):
    r_min = (
        min(
            -np.abs(self.target_vel - self.MIN_VEL),
            -np.abs(self.target_vel - self.MAX_VEL),
        )
        + 2
    )
    # r_min = -self.MAX_VEL
    r_max = 0.0

    x_vel = self._get_x_vel()
    forward_reward = -np.abs(self.target_vel - x_vel)
    if self._normalize_reward:
        forward_reward = normalize_reward(forward_reward, r_min, r_max)
    return forward_reward_weight * forward_reward


def squared_deviation(self, forward_reward_weight):
    r_min = min(
        -np.square(self.target_vel - self.MIN_VEL),
        -np.square(self.target_vel - self.MAX_VEL),
    )
    r_max = 0.0

    x_vel = self._get_x_vel()
    forward_reward = -np.square(self.target_vel - x_vel)
    if self._normalize_reward:
        forward_reward = normalize_reward(forward_reward, r_min, r_max)
    return forward_reward_weight * forward_reward


def exponential_squared_deviation(self, forward_reward_weight):
    x_vel = self._get_x_vel()
    forward_reward = np.exp(-np.square(self.target_vel - x_vel))
    return forward_reward_weight * forward_reward


def exponential_absolute_deviation(self, forward_reward_weight):
    x_vel = self._get_x_vel()
    forward_reward = np.exp(-np.abs(self.target_vel - x_vel))
    return forward_reward_weight * forward_reward


def l1_norm(self, forward_reward_weight):
    velocity = self._get_vel()
    forward_reward = -np.sum(np.abs(velocity - self.TARGET_VELOCITY))
    if self._normalize_reward:
        forward_reward = normalize_reward(forward_reward, -2 * self.MAX_VEL, 0.0)
    return forward_reward_weight * forward_reward


def l2_norm(self, forward_reward_weight):
    xyz_velocity = self._get_vel()
    forward_reward = -np.linalg.norm(self.TARGET_VELOCITY - xyz_velocity).item()
    if self._normalize_reward:
        r_min = -12
        forward_reward = normalize_reward(forward_reward, r_min, 0.0)
    return forward_reward_weight * forward_reward


task_mapping: dict[str, Callable] = {
    "absolute_deviation": absolute_deviation,
    "squared_deviation": squared_deviation,
    "exponential_squared_deviation": exponential_squared_deviation,
    "exponential_absolute_deviation": exponential_absolute_deviation,
    "l1_norm": l1_norm,
    "l2_norm": l2_norm,
}
