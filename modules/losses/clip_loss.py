import logging
import torch
import torch.nn.functional as F
import clip
from typing import List, Union

from .base_loss import BaseLoss

logger = logging.getLogger(__name__)

class CLIPLoss(BaseLoss):
    """
    CLIP image-text similarity loss.
    Uses OpenAI CLIP ViT-B/32 model to guide generation towards a text prompt.
    """

    def __init__(self, device: str = "cuda", weight: float = 1.0):
        """
        Initializes the CLIP loss module.

        Args:
            device (str): Computation device (default: "cuda").
            weight (float): Loss multiplier.
        """
        super().__init__(weight)
        self.device = device
        
        logger.info(f"Loading CLIP ViT-B/32 model on {self.device}...")
        
        # Note: Original implementation intentionally discards the CLIP preprocess function (_)
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        
        # CRITICAL VRAM SAVER: Freeze CLIP parameters.
        # We only need gradients to flow back to the input image, not to train CLIP itself.
        for param in self.model.parameters():
            param.requires_grad = False
            
        logger.info("CLIP model loaded and frozen successfully.")

    def forward(self, pred: torch.Tensor, target: List[str]) -> torch.Tensor:
        """
        Computes the cosine distance between image and text features.

        Args:
            pred (torch.Tensor): Predicted images (B, 3, H, W). Expected in [0, 1] range.
            target (List[str]): List of target text prompts.

        Returns:
            torch.Tensor: The computed scalar loss value, scaled by self.weight.
        """
        # Ensure predictions are on the correct device safely
        if pred.device.type != self.device:
            pred = pred.to(self.device)
            
        text_tokens = clip.tokenize(target).to(self.device)

        # Extract features
        img_feat = self.model.encode_image(pred)
        txt_feat = self.model.encode_text(text_tokens)

        # Normalize features for Cosine Similarity
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        # Calculate Cosine Distance (1 - Cosine Similarity)
        loss = 1.0 - (img_feat * txt_feat).sum(dim=-1).mean()

        return self.weight * loss