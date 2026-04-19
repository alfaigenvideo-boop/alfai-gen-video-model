import logging
import torch
import torch.nn as nn
from typing import Callable, Dict, Any, List

logger = logging.getLogger(__name__)

class LatentRefiner:
    """
    Latent Refinement Module (Test-Time Optimization).
    Performs gradient-based optimization directly on the latent space
    if the structural or semantic losses exceed defined thresholds.
    """

    def __init__(
        self,
        total_loss_fn: nn.Module,
        clip_th: float = 0.6,
        perc_th: float = 1.0,
        id_th: float = 0.3,
        steps: int = 3,
        lr: float = 1e-2
    ):
        """
        Initializes the Latent Refiner.

        Args:
            total_loss_fn (nn.Module): The instantiated TotalLoss orchestrator.
            clip_th (float): Threshold for CLIP loss (text alignment).
            perc_th (float): Threshold for Perceptual loss (structural consistency).
            id_th (float): Threshold for ID loss (identity consistency).
            steps (int): Number of Adam optimization steps to perform.
            lr (float): Learning rate for the latent optimization.
        """
        self.total_loss_fn = total_loss_fn
        self.clip_th = clip_th
        self.perc_th = perc_th
        self.id_th = id_th
        self.steps = steps
        self.lr = lr
        
        logger.info(f"LatentRefiner initialized -> Thresholds: CLIP:{clip_th}, PERC:{perc_th}, ID:{id_th} | LR:{lr}")

    def needs_refine(self, loss_dict: Dict[str, float]) -> bool:
        """
        Evaluates if the current frame requires latent refinement based on loss thresholds.
        """
        requires = (
            loss_dict.get("clip", 0.0) > self.clip_th or
            loss_dict.get("perceptual", 0.0) > self.perc_th or
            loss_dict.get("id", 0.0) > self.id_th
        )
        
        if requires:
            logger.debug(f"Refinement triggered based on loss dict: {loss_dict}")
            
        return requires

    def refine(
        self,
        latent: torch.Tensor,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        text_prompt: List[str],
        prev_frame: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimizes the latent representation using Adam to minimize the total loss.

        Args:
            latent (torch.Tensor): The generated latent tensor to be refined.
            decode_fn (Callable): Function to decode latents to RGB tensor (Must be differentiable).
            text_prompt (List[str]): The target text prompt for CLIP loss.
            prev_frame (torch.Tensor): The reference previous frame (RGB tensor) for ID/Perceptual loss.

        Returns:
            torch.Tensor: The refined latent tensor, detached from the computation graph.
        """
        # CRITICAL: Detach from the main diffusion graph and enable fresh gradients
        latent = latent.detach().clone()
        latent.requires_grad_(True)

        optimizer = torch.optim.Adam([latent], lr=self.lr)

        logger.info(f"Starting Latent Refinement ({self.steps} steps)...")

        for step in range(self.steps):
            optimizer.zero_grad()

            # Decode latent to differentiable pixel space
            image = decode_fn(latent)

            # Call TotalLoss with the canonical signature we defined earlier
            loss, loss_dict = self.total_loss_fn(
                pred_frame=image,
                text_prompt=text_prompt,
                target_frame=prev_frame,
                pred_frame_temporal=image
            )

            # Backpropagate and update the latent
            loss.backward()
            optimizer.step()
            
            logger.debug(f"Refinement Step {step+1}/{self.steps} - Total Loss: {loss.item():.4f}")

        # CRITICAL: Return detached latent to avoid memory leaks in the outer pipeline
        return latent.detach()