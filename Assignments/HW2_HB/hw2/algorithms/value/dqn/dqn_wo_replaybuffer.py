import numpy as np

from mushroom_rl.algorithms.value.dqn import DQN


class DQNwoReplayBuffer(DQN):

    def _fit_standard(self, dataset):
        state = np.array([d[0] for d in dataset])
        action = np.array([d[1] for d in dataset])
        reward = np.array([d[2] for d in dataset])
        next_state = np.array([d[3] for d in dataset])
        absorbing = np.array([d[4] for d in dataset])

        if self._clip_reward:
            reward = np.clip(reward, -1, 1)

        q_next = self._next_q(next_state, absorbing)
        q = reward + self.mdp_info.gamma * q_next

        self.approximator.fit(state, action, q, **self._fit_params)
