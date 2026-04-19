from abc import ABC, abstractmethod
import torch

class BaseFlowEstimator(ABC):
    """
    Abstract base class for all optical flow estimation models.
    Ensures a consistent interface across different flow algorithms (e.g., RAFT, GMFlow).
    """

    @abstractmethod
    def compute(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Computes optical flow between two frames.

        Args:
            img1 (torch.Tensor): The first frame tensor, shape [B, C, H, W].
            img2 (torch.Tensor): The second frame tensor, shape [B, C, H, W].

        Returns:
            torch.Tensor: The estimated optical flow, shape [B, 2, H, W].
        """
        pass

    @abstractmethod
    def resize_to_latent(self, flow: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
        """
        Resizes the optical flow tensor to match the latent space dimensions.
        
        CRITICAL: Implementations of this method MUST also scale the flow vector 
        magnitudes (x and y values) proportionally to the spatial downscaling factor.

        Args:
            flow (torch.Tensor): Original high-resolution flow tensor [B, 2, H_img, W_img].
            latent_h (int): Target latent height (typically 128 for SDXL).
            latent_w (int): Target latent width (typically 128 for SDXL).

        Returns:
            torch.Tensor: Resized and magnitude-scaled flow tensor [B, 2, latent_h, latent_w].
        """
        pass