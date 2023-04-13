"""
This module provides a couple of utility functions for working with PyTorch models and data.

Functions:
- train: Train a PyTorch model on the given data using the specified optimizer and criterion.
- predict: Use a trained PyTorch model to make predictions on new data.

"""
from typing import Dict
import torch
import numpy as np
from numpy import ndarray


def train(data: Dict[str, ndarray], model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: str) -> None:
    """
    Trains a PyTorch model on the given data using the specified optimizer and criterion.

    Args:
        data (dict): A dictionary containing the input features and targets for the training data.
                     The dictionary should have keys 'features' and 'target'.
        model (torch.nn.Module): The PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to use during training.
        criterion (torch.nn.modules.loss._Loss): The PyTorch loss function to use during training.
        device (torch.device): The device (CPU or GPU) to use for training.

    Returns:
        None
    """

    features = data['features']
    target = data['target']

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)

    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    features = features.to(torch.float32)
    features = features.to(device)

    target = target.to(device)

    model.train()
    out = model(features)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def predict(model: torch.nn.Module, data: ndarray, device: str) -> ndarray:
    """
    Uses a trained PyTorch model to make predictions on new data.

    Args:
        model (torch.nn.Module): The trained PyTorch model to use for prediction.
        data (numpy.ndarray): The input data to make predictions on.
        device (torch.device): The device (CPU or GPU) to use for prediction.

    Returns:
        numpy.ndarray: The predicted output for the input data.
    """

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    data = data.to(torch.float32)
    data = data.to(device)

    model.eval()

    with torch.no_grad():
        out = model(data)

    out = out.cpu().detach().numpy()

    return out
