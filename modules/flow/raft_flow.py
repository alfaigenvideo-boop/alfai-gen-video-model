import logging
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from .base_flow import BaseFlowEstimator

logger = logging.getLogger(__name__)

class RAFTFlow(BaseFlowEstimator):
    """
    RAFT (Recurrent All-Pairs Field Transforms) Optical Flow implementation.
    Wraps the highly optimized torchvision pre-trained RAFT model for production inference.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initializes the RAFT model and its preprocessing pipeline.

        Args:
            device (str): Computation device (default: "cuda").
        """
        self.device = device
        logger.info(f"Loading RAFT-Large model on {self.device}...")
        
        # Load pre-trained weights and apply evaluation mode
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights).to(self.device).eval()
        self.preprocess = weights.transforms()
        
        logger.info("RAFT-Large model initialized successfully.")

    @torch.no_grad()
    def compute(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Computes the dense optical flow field from img1 to img2.

        Args:
            img1 (torch.Tensor): First frame tensor [B, C, H, W] (RGB, float, typically [-1, 1] or [0, 1]).
            img2 (torch.Tensor): Second frame tensor [B, C, H, W].

        Returns:
            torch.Tensor: High-resolution flow field [B, 2, H, W].
        """
        # Apply torchvision's specific normalization and scaling
        img1_p, img2_p = self.preprocess(img1, img2)
        img1_p, img2_p = img1_p.to(self.device), img2_p.to(self.device)

        # The model returns a list of flow predictions across GRU iterations.
        # We only need the final, most refined flow prediction [-1].
        flow = self.model(img1_p, img2_p)[-1]
        
        return flow

    def resize_to_latent(self, flow: torch.Tensor, latent_h: int = 128, latent_w: int = 128) -> torch.Tensor:
        """
        Resizes the optical flow spatially AND scales the flow vector magnitudes.
        
        Args:
            flow (torch.Tensor): Original flow tensor [B, 2, H_img, W_img].
            latent_h (int): Target latent height.
            latent_w (int): Target latent width.

        Returns:
            torch.Tensor: Resized and scaled flow [B, 2, latent_h, latent_w].
        """
        _, _, h_img, w_img = flow.shape
        scale_x = latent_w / w_img
        scale_y = latent_h / h_img

        # 1. Spatial resizing
        # Using 'area' interpolation because flow vectors represent physical movement,
        # and area averaging prevents harsh aliasing artifacts when downsampling drastically.
        flow_resized = F.interpolate(flow, size=(latent_h, latent_w), mode="area")
        
        # 2. Magnitude scaling (CRITICAL for correct warping)
        # flow[:, 0] represents movement in X axis, flow[:, 1] represents Y axis.
        # We must multiply by the scale ratios to avoid over-shooting in the small latent space.
        flow_resized_cloned = flow_resized.clone()
        flow_resized_cloned[:, 0] *= scale_x
        flow_resized_cloned[:, 1] *= scale_y

        return flow_resized_cloned