import logging
import torch
import torch.nn.functional as F
from typing import Any
from .base_flow import BaseFlowEstimator

logger = logging.getLogger(__name__)

class GMFlow(BaseFlowEstimator):
    """
    GMFlow (Global Matching Flow) Optical Flow implementation.
    Wraps an externally instantiated GMFlow model to match the pipeline interface.
    """

    def __init__(self, model: Any, device: str = "cuda"):
        """
        Initializes the GMFlow wrapper.

        Args:
            model (Any): A pre-instantiated GMFlow model object.
            device (str): Computation device.
        """
        self.model = model
        self.device = device
        logger.info("GMFlow wrapper initialized with externally provided model.")

    @torch.no_grad()
    def compute(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Computes the dense optical flow field from img1 to img2 using GMFlow.

        Args:
            img1 (torch.Tensor): First frame tensor [B, C, H, W].
            img2 (torch.Tensor): Second frame tensor [B, C, H, W].

        Returns:
            torch.Tensor: High-resolution flow field [B, 2, H, W].
        """
        return self.model(img1, img2)

    def resize_to_latent(self, flow: torch.Tensor, latent_h: int = 128, latent_w: int = 128) -> torch.Tensor:
        """
        Resizes the optical flow spatially AND scales the flow vector magnitudes.
        This fulfills the BaseFlowEstimator interface requirements.
        
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

        # Spatial resizing using 'area' to match RAFT logic
        flow_resized = F.interpolate(flow, size=(latent_h, latent_w), mode="area")
        
        # Magnitude scaling (CRITICAL for correct warping)
        flow_resized_cloned = flow_resized.clone()
        flow_resized_cloned[:, 0] *= scale_x
        flow_resized_cloned[:, 1] *= scale_y

        return flow_resized_cloned