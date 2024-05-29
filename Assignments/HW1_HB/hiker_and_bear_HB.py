import numpy as np

from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.environments import FiniteMDP


class HikerAndBearBase(FiniteMDP):
    """
    The HikerAndBear environment.
    It is a mini grid environment with a bear that can spawn in 2 different positions on the middle row.
    The agent spawns in the bottom row and the goal (the hut) is in the centre of the top row.
    The agent can move up, down, left, and right.
    The closer the agent is to the bear, the higher the probability of being caught by the bear.
    The episode ends when the agent reaches the goal, when the agent is captured by the bear, or when time runs outs.
    """

    def __init__(
        self, size=(10, 10), n_bear_pos=2, gamma=0.99, horizon=100, action_prob=0.95
    ):
        # initialize the base class

        self._size = size
        self._n_bear_pos = n_bear_pos
        self.prob = action_prob

        # change in coordinates for each action
        self.directions = [[0, 1], [-1, 0], [0, -1], [1, 0]]

        # define a viewer for visualization
        self._viewer = Viewer(self._size[0] + 1, self._size[1] + 1, 500, 500)

        # build two finite MDPs for the 2 spawn position of the bear
        mu = self.initialize_state_distribution()
        p = self.compute_probabilities(self.prob)
        r = self.compute_rewards()

        # call the super class
        super().__init__(p, r, mu, gamma, horizon)

        # reset the environment
        self.reset()

    def initialize_state_distribution(self):
        """
        Raise not implemented error.
        """
        raise NotImplementedError("initialize_state_distribution not implemented")

    def state_(self, j_b, i_h, j_h):
        # Get state index from hiker and bear positions.

        return j_b * self._size[0] * self._size[1] + i_h * self._size[1] + j_h

    def compute_probabilities(self, prob):
        """
        Raise not implemented error.
        """
        raise NotImplementedError("compute_probabilities not implemented")

    def compute_rewards(self):
        """
        Raise not implemented error.
        """
        raise NotImplementedError("compute_rewards not implemented")

    def tf(self, i, j):
        # Transform the coordinates of the agent to the viewer coordinates

        x = j + 1
        y = self._size[0] - i

        return np.array([x, y])

    def idx2pos_bear(self, j_b):
        # Get the position of the bear from the index

        return j_b * 5

    def render(self):
        # Render the environment

        # Gridlines
        for row in range(0, self._size[0] + 1):
            for col in range(0, self._size[1] + 1):
                self._viewer.line(
                    np.array([col, 0]), np.array([col, self._size[1] + 1])
                )
                self._viewer.line(
                    np.array([0, row]), np.array([self._size[0] + 1, row])
                )
        for i in range(0, self._size[0]):
            for j in range(0, self._size[1]):
                self._viewer.square(
                    self.tf(i, j), 0, 1, color="forestgreen"
                )  # fill the cell with a green square

        self._viewer.square(
            self.tf(0, self._size[1] // 2), 0, 1, color="gold"
        )  # draw a gold square for the target at the center of the top row (goal position)

        # get the coordinates of the agent and the bear from the state
        state = self._state % self.state_(1, 0, 0)
        agent_pose = (state[0] // self._size[1], state[0] % self._size[1])
        bear_pose = (5, (self._state[0] - state[0]) // (self._size[0] * self._size[1]))

        # fill the cell with a different color according to the probability of the agent to be caught by the bear
        range_offset_x_A = 1
        range_offset_x_B = 4
        range_bearea_x_A = list(
            range(
                max(self.idx2pos_bear(bear_pose[1]) - range_offset_x_A, 0),
                min(
                    self.idx2pos_bear(bear_pose[1]) + range_offset_x_A + 1,
                    self._size[1],
                ),
            )
        )
        range_bearea_x_B = list(
            range(
                max(self.idx2pos_bear(bear_pose[1]) - range_offset_x_B, 0),
                min(
                    self.idx2pos_bear(bear_pose[1]) + range_offset_x_B + 1,
                    self._size[1],
                ),
            )
        )
        for j in range(self._size[1]):
            for i in [
                bear_pose[0] - 2,
                bear_pose[0] - 1,
                bear_pose[0],
                bear_pose[0] + 1,
            ]:
                self._viewer.square(self.tf(i, j), 0, 1, color="mediumseagreen")
        for j in range_bearea_x_B:
            for i in [
                bear_pose[0] - 2,
                bear_pose[0] - 1,
                bear_pose[0],
                bear_pose[0] + 1,
            ]:
                self._viewer.square(self.tf(i, j), 0, 1, color="khaki")
        for j in range_bearea_x_A:
            for i in [
                bear_pose[0] - 2,
                bear_pose[0] - 1,
                bear_pose[0],
                bear_pose[0] + 1,
            ]:
                self._viewer.square(self.tf(i, j), 0, 1, color="tomato")

        self._viewer.circle(
            self.tf(agent_pose[0], agent_pose[1]), 0.4, color="deepskyblue"
        )  # draw the agent as a blue arrow head
        self._viewer.circle(
            self.tf(bear_pose[0], self.idx2pos_bear(bear_pose[1])),
            0.4,
            color="saddlebrown",
        )  # draw the bear as a brown circle

        # render the viewer
        self._viewer.display(0.1)

    def stop(self):
        # Close the viewer
        if self._viewer is not None:
            self._viewer.close()
