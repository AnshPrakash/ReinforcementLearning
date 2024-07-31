from minatar import Environment as MinAtarEnvironment
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Discrete, Box
import numpy as np


class MinAtar(Environment):
    def __init__(
        self,
        name,
        gamma=0.99,
        sticky_action_prob=0.1,
        difficulty_ramping=True,
        random_seed=None,
    ):
        self.env = MinAtarEnvironment(
            name, sticky_action_prob, difficulty_ramping, random_seed
        )
        self.env_name = name

        action_space = Discrete(self.env.num_actions())
        shape = self.env.state_shape()
        observation_space = Box(low=0.0, high=1.0, shape=[shape[2], shape[0], shape[1]])
        mdp_info = MDPInfo(observation_space, action_space, gamma, np.inf)
        self.display_open = False
        super().__init__(mdp_info)

    def step(self, action):
        action = int(action[0])
        reward, absorbing = self.env.act(action)
        return self._get_state(), reward, absorbing, {}

    def reset(self, state=None):
        assert state is None
        self.env.reset()
        return self._get_state()

    def render(self, mode="human"):
        self.display_open = True
        return self.env.display_state()

    def stop(self):
        if self.display_open:
            self.env.close_display()
        self.display_open = False

    def _get_state(self):
        return np.moveaxis(self.env.state(), -1, 0).astype("float32")
