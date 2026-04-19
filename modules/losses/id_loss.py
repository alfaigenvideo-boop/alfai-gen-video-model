import logging
import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Any

from .base_loss import BaseLoss

logger = logging.getLogger(__name__)

class IDLoss(BaseLoss):
    """
    Identity consistency loss between two frames.
    Uses pretrained ResNet50 features to enforce temporal identity stability.
    """

    def __init__(self, device: str = "cuda", weight: float = 1.0):
        """
        Initializes the ID Loss module.

        Args:
            device (str): Computation device (default: "cuda").
            weight (float): Loss multiplier.
        """
        super().__init__(weight)
        self.device = device
        
        logger.info(f"Loading ResNet50 model for ID Loss on {self.device}...")
        
        # Using modern torchvision weights API instead of deprecated pretrained=True
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights).to(self.device)
        self.model.eval()
        
        # CRITICAL VRAM SAVER: Freeze ResNet parameters.
        # We only need to backpropagate to the generated frame, not train ResNet50.
        for param in self.model.parameters():
            param.requires_grad = False
            
        logger.info("ResNet50 model loaded and frozen successfully.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the identity consistency (cosine distance) between two frames.

        Args:
            pred (torch.Tensor): Current frame tensor (B, 3, H, W) -> corresponds to frame_t1.
            target (torch.Tensor): Reference frame tensor (B, 3, H, W) -> corresponds to frame_t.

        Returns:
            torch.Tensor: The computed scalar loss value, scaled by self.weight.
        """
        # Ensure tensors are on the correct device safely
        if pred.device.type != self.device:
            pred = pred.to(self.device)
        if target.device.type != self.device:
            target = target.to(self.device)

        # Extract features
        # Note: Original implementation directly feeds frames without ImageNet normalization
        f_pred = self.model(pred)
        f_target = self.model(target)

        # Normalize features for Cosine Similarity
        f_pred = F.normalize(f_pred, dim=-1)
        f_target = F.normalize(f_target, dim=-1)

        # Calculate Cosine Distance (1 - Cosine Similarity)
        loss = 1.0 - (f_pred * f_target).sum(dim=-1).mean()

        return self.weight * loss