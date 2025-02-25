import numpy as np

from mushroom_rl.algorithms.value.dqn import DoubleDQN
from mushroom_rl.approximators import Regressor


class DoubleMaxminDQN(DoubleDQN):

    def __init__(self, mdp_info, policy, approximator, n_approximators, **params):
        """
        Constructor.

        Args:
            n_approximators (int): the number of approximators in the ensemble.

        """
        assert n_approximators > 1

        self._n_approximators = n_approximators

        super().__init__(mdp_info, policy, approximator, **params)

    def fit(self, dataset, **info):
        self._fit_params["idx"] = np.random.randint(self._n_approximators)

        super().fit(dataset, **info)

    def _initialize_regressors(
        self, approximator, apprx_params_train, apprx_params_target
    ):
        self.approximator = Regressor(
            approximator,
            n_models=self._n_approximators,
            prediction="min",
            **apprx_params_train
        )
        self.target_approximator = Regressor(
            approximator,
            n_models=self._n_approximators,
            prediction="min",
            **apprx_params_target
        )
        self._update_target()

    def _update_target(self):
        for i in range(len(self.target_approximator)):
            self.target_approximator[i].set_weights(self.approximator[i].get_weights())
