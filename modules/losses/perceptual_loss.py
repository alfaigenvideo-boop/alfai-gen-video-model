import logging
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from typing import Any

from .base_loss import BaseLoss

logger = logging.getLogger(__name__)

class PerceptualLoss(BaseLoss):
    """
    VGG16-based perceptual loss between two frames.
    Uses early feature maps of pretrained VGG16 (up to layer 16) to ensure 
    high-frequency structural similarity.
    """

    def __init__(self, device: str = "cuda", weight: float = 1.0):
        """
        Initializes the Perceptual Loss module.

        Args:
            device (str): Computation device (default: "cuda").
            weight (float): Loss multiplier.
        """
        super().__init__(weight)
        self.device = device
        
        logger.info(f"Loading VGG16 model for Perceptual Loss on {self.device}...")
        
        # Using modern torchvision weights API instead of deprecated pretrained=True
        weights = VGG16_Weights.DEFAULT
        vgg = vgg16(weights=weights).features[:16]
        self.vgg = vgg.to(self.device).eval()

        # VRAM SAVER: Freeze VGG parameters (Correctly implemented in original code)
        # We only need to backpropagate to the generated frame, not train VGG16.
        for p in self.vgg.parameters():
            p.requires_grad = False
            
        logger.info("VGG16 model loaded and frozen successfully.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mean Squared Error (MSE) between the VGG16 feature maps of two frames.

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
        feat_pred = self.vgg(pred)
        feat_target = self.vgg(target)

        # Compute Mean Squared Error (MSE) on feature maps
        loss = torch.mean((feat_target - feat_pred) ** 2)

        return self.weight * loss