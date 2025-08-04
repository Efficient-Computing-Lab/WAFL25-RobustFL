import torch

from src.settings import settings


def flip_labels(labels: torch.tensor, total_number_classes: int) -> torch.tensor:
    flipped_tensor = total_number_classes - 1 - labels
    return flipped_tensor


def add_gaussian_noise(parameters):
    """
    Adds Gaussian noise with user-defined mean and standard deviation to model parameters
    :param parameters: Model parameters
    """
    for param in parameters:
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * settings.attack.std + settings.attack.mean
            param.grad += noise


def flip_sign(parameters):
    """
    Flips sign of gradient for model parameters.
    :param parameters: Model parameters
    """
    for param in parameters:
        if param.grad is not None:
            param.grad *= -1  # Flip the sign of the gradients
