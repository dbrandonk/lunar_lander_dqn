"""
This module defines a neural network model for the Lunar Lander environment in OpenAI Gym.

Classes:
- LunarLanderNet: A PyTorch neural network model that maps observations to actions.

Constants:
- LAYER_WIDTH: The number of units in each hidden layer of the neural network.
"""

from torch import Tensor, nn
from numpy import ndarray

LAYER_WIDTH = 64


class LunarLanderNet(nn.Module):
    """
    A PyTorch neural network model that maps observations to actions for the Lunar Lander
    environment.

    Args:
    - observation_space (gym.Space): The observation space of the Lunar Lander environment.
    - action_space (gym.Space): The action space of the Lunar Lander environment.

    Attributes:
    - fc1 (nn.Linear): The first fully connected layer of the neural network.
    - fc2 (nn.Linear): The second fully connected layer of the neural network.
    - fc3 (nn.Linear): The output layer of the neural network.

    Methods:
    - forward(data): Forward pass through the neural network.

    """
    def __init__(self, observation_space: ndarray, action_space: int) -> None:
        super(LunarLanderNet, self).__init__()

        self.fc1 = nn.Linear(observation_space.shape[0], LAYER_WIDTH)
        self.fc2 = nn.Linear(LAYER_WIDTH, LAYER_WIDTH)
        self.fc3 = nn.Linear(LAYER_WIDTH, action_space)

    def forward(self, data: Tensor) -> Tensor:
        """
        Forward pass through the neural network.

        Args:
        - data (torch.Tensor): A batch of observations from the Lunar Lander environment.

        Returns:
        - outs (torch.Tensor): The predicted Q-values for the possible actions.
        """
        outs = nn.functional.relu(self.fc1(data))
        outs = nn.functional.relu(self.fc2(outs))
        outs = self.fc3(outs)
        return outs
