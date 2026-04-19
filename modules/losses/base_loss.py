import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all custom loss functions.
    Ensures a consistent interface and weight management across different loss components.
    """

    def __init__(self, weight: float = 1.0):
        """
        Initializes the base loss module.

        Args:
            weight (float): The multiplier/weight for this specific loss component (default: 1.0).
        """
        super().__init__()
        self.weight = weight

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between predictions and targets.

        Args:
            pred (torch.Tensor): The predicted tensor (usually the output of the model/warp).
            target (torch.Tensor): The ground truth or target tensor to compare against.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        pass