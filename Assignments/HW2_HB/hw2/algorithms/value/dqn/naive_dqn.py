from copy import deepcopy

import numpy as np

from mushroom_rl.core import Agent
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.utils.parameters import to_parameter


class NaiveDQN(Agent):
    def __init__(
        self,
        mdp_info,
        policy,
        approximator,
        approximator_params,
        batch_size,
        fit_params=None,
        predict_params=None,
    ):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the
               Q-function;
            approximator_params (dict): parameters of the approximator to
                build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;
            predict_params (dict, None): parameters for the prediction with the
                approximator;
        """
        self._fit_params = dict() if fit_params is None else fit_params
        self._predict_params = dict() if predict_params is None else predict_params

        self._batch_size = to_parameter(batch_size)

        self._fit = self._fit_standard

        apprx_params_train = deepcopy(approximator_params)

        self._initialize_regressors(approximator, apprx_params_train)

        policy.set_q(self.approximator)

        self._add_save_attr(
            _fit_params="pickle",
            _predict_params="pickle",
            _batch_size="mushroom",
            _n_approximators="primitive",
            approximator="mushroom",
        )

        super().__init__(mdp_info, policy)

    def fit(self, dataset, **info):
        self._fit(dataset)

    def _fit_standard(self, dataset):
        state = np.array([d[0] for d in dataset])
        action = np.array([d[1] for d in dataset])
        reward = np.array([d[2] for d in dataset])
        next_state = np.array([d[3] for d in dataset])
        absorbing = np.array([d[4] for d in dataset])

        q_next = self._next_q(next_state, absorbing)
        q = reward + self.mdp_info.gamma * q_next

        self.approximator.fit(state, action, q, **self._fit_params)

    def draw_action(self, state):
        action = super().draw_action(np.array(state))

        return action

    def _initialize_regressors(self, approximator, apprx_params_train):
        self.approximator = Regressor(approximator, **apprx_params_train)

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Maximum action-value for each state in ``next_state``.

        """
        q = self.approximator.predict(next_state, **self._predict_params)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        return np.max(q, axis=1)

    # NOTE: not important to include
    def set_logger(self, logger, loss_filename="loss_Q"):
        """
        Setter that can be used to pass a logger to the algorithm

        Args:
            logger (Logger): the logger to be used by the algorithm;
            loss_filename (str, 'loss_Q'): optional string to specify the loss filename.

        """
        self.approximator.set_logger(logger, loss_filename)
