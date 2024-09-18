import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionValueCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        """
        General action-value critic network architecture taking state and action as input.
        See https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_sac.py
        for reference.

        Args:
            input_shape (tuple).
            output_shape (tuple).
            n_features (int): number of features in a layer.
        """
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        # TODO:complete the following code by defining three fully connected layers
        # [START YOUR CODE HERE]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        # [END YOUR CODE HERE]

        # TODO initialize their weights using the Xavier uniform method.
        # For RELU layers, weights should be initialized using nn.init.calculate_gain('relu')
        # For Linear layers, weights should be initialized using nn.init.calculate_gain('linear')
        # [START YOUR CODE HERE]
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))
        # [END YOUR CODE HERE]

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)
        return torch.squeeze(q)


class ValueCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        """
        General value critic network architecture taking state as input.
        See https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_sac.py
        for reference.

        Args:
            input_shape (tuple).
            output_shape (tuple).
            n_features (int): number of features in a layer.
        """
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, **kwargs):
        # TODO: implement the forward pass of the first layer
        # use F.relu for activation functions, take care of the input shape
        # [START YOUR CODE HERE]
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        # [END YOUR CODE HERE]
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)
        return torch.squeeze(a)


class ActorNetwork(nn.Module):
    """
    General actor network architecture taking state as input.
    See https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_sac.py
    for reference.

    Args:
        input_shape (tuple).
        output_shape (tuple).
        n_features (int): number of features in a layer.
    """

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


class BoundedActorNetwork(nn.Module):
    """
    General actor network architecture taking state as input.
    See https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_sac.py
    for reference.

    Args:
        input_shape (tuple).
        output_shape (tuple).
        n_features (int): number of features in a layer.
    """

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._action_scaling = torch.tensor(kwargs["action_scaling"]).to(
            device=torch.device("cuda" if kwargs["use_cuda"] else "cpu")
        )
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        a = self._action_scaling * torch.tanh(a)
        return a
