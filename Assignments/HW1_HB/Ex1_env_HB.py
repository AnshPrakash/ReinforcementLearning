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
        mu[:,self._size[0]-1,:] =  1
        # normalize the distribution
        mu = mu/mu.sum()
        # flatten the distribution
        mu = mu.flatten()

        ######################

        return mu

    def set_prob(self, p, j_b, i_h, j_h,new_i_h, new_j_h, capture_prob, a, bear_pos_ih, prob) :
        p[
            self.state_(j_b, i_h, j_h),
            a,
            self.state_(j_b, new_i_h, new_j_h),
        ] += (
            (1 - capture_prob)*(prob)
        )  # probability to perform the action
        p[
            self.state_(j_b, i_h, j_h),
            a,
            self.state_(j_b, i_h, j_h),
        ] += (
            (1 - capture_prob)*(1 - prob)
        )  # probability to stay in the same position (action may fail)
        ### YOUR CODE HERE ###

        # probability to be captured by the bear
        p[
            self.state_(j_b, i_h, j_h),
            a,
            self.state_(j_b, bear_pos_ih, self.idx2pos_bear(j_b)),
        ] += capture_prob
        ######################
        return
    
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
        bear_pos_ih = 5
        goal_ih = 0
        goal_jh = 5
        # iterate over the states and fill the transition probability matrix
        for j_b in range(self._n_bear_pos):
            for i_h in range(self._size[0]):
                for j_h in range(self._size[1]):
                    # to set the goal as a terminal state
                    ### YOUR CODE HERE ###
                    if i_h == goal_ih and j_h == goal_jh :
                        """
                        A terminal state in mushroom_rl has 0 prob to any other states for any actions
                        (weird, I thought it should be 1.0 to itself for any action )
                        """
                        continue
                    ######################

                    # if the state is the the bear, terminate
                    ### YOUR CODE HERE ###
                    if i_h == bear_pos_ih and j_h == self.idx2pos_bear(j_b):
                        continue
                    ######################

                    # iterate over the actions
                    for a in range(4):
                        # compute the new position of the hiker
                        new_i_h = i_h + self.directions[a][0]
                        new_j_h = j_h + self.directions[a][1]

                        # update the transition probability matrix - remember, the action may fail with probability 1-prob
                        if i_h <= 6 and i_h >= 3:  # bear area
                            if abs(j_h - self.idx2pos_bear(j_b)) < 2:
                                capture_prob = 0.9
                                self.set_prob( p, j_b, i_h, j_h,new_i_h, new_j_h, capture_prob, a, bear_pos_ih, prob)
                                
                            elif abs(j_h - self.idx2pos_bear(j_b)) <= 4:
                                ### YOUR CODE HERE ###
                                capture_prob = 0.5
                                self.set_prob( p, j_b, i_h, j_h,new_i_h, new_j_h, capture_prob, a, bear_pos_ih, prob)


                                ######################
                            else:
                                ### YOUR CODE HERE ###
                                capture_prob = 0.001
                                self.set_prob( p, j_b, i_h, j_h,new_i_h, new_j_h, capture_prob, a, bear_pos_ih, prob)

                                ######################
                        else:
                            ### YOUR CODE HERE ###
                            
                            # rest of the environment
                            capture_prob = 0.0
                            if new_i_h >=0 and new_i_h < self._size[0] and new_j_h >=0 and new_j_h < self._size[1]:
                                # update only for valid actions
                                self.set_prob( p, j_b, i_h, j_h,new_i_h, new_j_h, capture_prob, a, bear_pos_ih, prob)
                            ######################

        print(np.sum(p, axis = 2))
        return p

    def compute_rewards(self):
        # Compute the rewards.

        # reward matrix - initialized with -0.01 reward for each transition to encourage the agent to reach the goal as fast as possible
        # the size is (observation space shape) x (action space shape) x (observation space shape)

        ### YOUR CODE HERE ###

        r = np.ones(
            (
                self._n_bear_pos * self._size[0] * self._size[1],
                4,
                self._n_bear_pos * self._size[0] * self._size[1],
            )
        )*(-0.01)
        bear_pos_ih = 5
        goal_ih = 0
        goal_jh = 5
        ######################

        # iterate over the states and fill the reward matrix
        for j_b in range(self._n_bear_pos):
            for i_h in range(self._size[0]):
                for j_h in range(self._size[1]):
                    # iterate over the actions
                    for a in range(4):
                        ### YOUR CODE HERE ###

                        # compute the new position
                        new_i_h = i_h + self.directions[a][0]
                        new_j_h = j_h + self.directions[a][1]
                        
                        if new_i_h >= 0 and new_i_h < self._size[0] and new_j_h >= 0 and new_j_h < self._size[1]:
                            #out of bound
                            continue

                        ######################

                        # if the new position is the bear, the reward is -1.0 (and the episode is terminated)
                        if i_h <= 6 and i_h >= 3:
                            r[
                                self.state_(j_b, i_h, j_h),
                                :,
                                self.state_(j_b, bear_pos_ih, self.idx2pos_bear(j_b))
                            ] = -1.0

                            # if abs(j_h - self.idx2pos_bear(j_b)) < 2:
                            #     ### YOUR CODE HERE ###

                            #     ######################
                            # elif abs(j_h - self.idx2pos_bear(j_b)) <= 4:
                            #     ### YOUR CODE HERE ###

                            #     ######################
                            # else:
                            #     ### YOUR CODE HERE ###

                            #     ######################

                        # if the new position is the goal, the reward is 1
                        ### YOUR CODE HERE ###
                        if (new_i_h, new_j_h) == (goal_ih, goal_jh):
                            r[
                                self.state_(j_b, i_h, j_h),
                                :,
                                self.state_(j_b, new_i_h, new_j_h)
                            ] = 1.0
                        ######################

        return r
