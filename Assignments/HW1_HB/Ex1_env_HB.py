from hiker_and_bear_HB import HikerAndBearBase
import numpy as np


class HikerAndBear(HikerAndBearBase):
    def __init__(
        self, size=(10, 10), n_bear_pos=2, gamma=0.99, horizon=100, action_prob=0.95
    ):
        # initialize the base class

        super().__init__(size, n_bear_pos, gamma, horizon, action_prob)

    def initialize_state_distribution(self):
        # Initialize the state distribution.

        mu = np.zeros((self._n_bear_pos, self._size[0], self._size[1]))

        ### YOUR CODE HERE ###

        # fill the last row of the grid with probability 1
        mu[:,9,:] =  1
        # normalize the distribution
        mu = mu/mu.sum()
        # flatten the distribution
        mu = mu.flatten()

        ######################

        return mu

    def compute_probabilities(self, prob):
        # Compute the transition probabilities.

        # Initialize transition probability matrix with 0
        # The size = (observation space shape) x (action space shape) x (observation space shape)
        p = np.zeros(
            (
                self._n_bear_pos * self._size[0] * self._size[1],
                4,
                self._n_bear_pos * self._size[0] * self._size[1],
            )
        )

        # iterate over the states and fill the transition probability matrix
        for j_b in range(self._n_bear_pos):
            for i_h in range(self._size[0]):
                for j_h in range(self._size[1]):
                    # to set the goal as a terminal state
                    ### YOUR CODE HERE ###

                    ######################

                    # if the state is the the bear, terminate
                    ### YOUR CODE HERE ###

                    ######################

                    # iterate over the actions
                    for a in range(4):
                        # compute the new position of the hiker
                        new_i_h = i_h + self.directions[a][0]
                        new_j_h = j_h + self.directions[a][1]

                        # update the transition probability matrix - remember, the action may fail with probability 1-prob
                        if i_h <= 6 and i_h >= 3:  # bear area
                            if abs(j_h - self.idx2pos_bear(j_b)) < 2:
                                p[
                                    self.state_(j_b, i_h, j_h),
                                    a,
                                    self.state_(j_b, new_i_h, new_j_h),
                                ] += (
                                    prob - 0.9
                                )  # probability to perform the action
                                p[
                                    self.state_(j_b, i_h, j_h),
                                    a,
                                    self.state_(j_b, i_h, j_h),
                                ] += (
                                    1 - prob
                                )  # probability to stay in the same position (action may fail)
                                ### YOUR CODE HERE ###

                                # probability to be captured by the bear

                                ######################
                            elif abs(j_h - self.idx2pos_bear(j_b)) <= 4:
                                ### YOUR CODE HERE ###

                                ######################
                            else:
                                ### YOUR CODE HERE ###

                                ######################
                        else:
                            ### YOUR CODE HERE ###

                            # rest of the environment

                            ######################

        return p

    def compute_rewards(self):
        # Compute the rewards.

        # reward matrix - initialized with -0.01 reward for each transition to encourage the agent to reach the goal as fast as possible
        # the size is (observation space shape) x (action space shape) x (observation space shape)

        ### YOUR CODE HERE ###

        r = 

        ######################

        # iterate over the states and fill the reward matrix
        for j_b in range(self._n_bear_pos):
            for i_h in range(self._size[0]):
                for j_h in range(self._size[1]):
                    # iterate over the actions
                    for a in range(4):
                        ### YOUR CODE HERE ###

                        # compute the new position

                        ######################

                        # if the new position is the bear, the reward is -1.0 (and the episode is terminated)
                        if i_h <= 6 and i_h >= 3:
                            if abs(j_h - self.idx2pos_bear(j_b)) < 2:
                                ### YOUR CODE HERE ###

                                ######################
                            elif abs(j_h - self.idx2pos_bear(j_b)) <= 4:
                                ### YOUR CODE HERE ###

                                ######################
                            else:
                                ### YOUR CODE HERE ###

                                ######################

                        # if the new position is the goal, the reward is 1
                        ### YOUR CODE HERE ###

                        ######################

        return r
