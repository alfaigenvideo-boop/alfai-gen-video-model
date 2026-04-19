import logging
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class TotalLoss(nn.Module):
    """
    Combines CLIP, Perceptual and ID losses with defined weights.
    Acts as the central orchestrator (nn.Module) for all generation guidance losses.
    """

    def __init__(
        self,
        clip_loss: nn.Module,
        perceptual_loss: nn.Module,
        id_loss: nn.Module,
        w_clip: float = 1.0,
        w_perceptual: float = 0.5,
        w_id: float = 0.2,
        **kwargs: Any
    ):
        """
        Initializes the TotalLoss orchestrator.

        Args:
            clip_loss (nn.Module): Instantiated CLIPLoss module.
            perceptual_loss (nn.Module): Instantiated PerceptualLoss module.
            id_loss (nn.Module): Instantiated IDLoss module.
            w_clip (float): Weight multiplier for CLIP loss.
            w_perceptual (float): Weight multiplier for Perceptual loss.
            w_id (float): Weight multiplier for ID loss.
            **kwargs: Catch-all for extra parameters passed by the builder.
        """
        super().__init__()
        
        # Registering components as PyTorch Sub-modules ensures they are 
        # properly tracked in the computation graph and device assignments.
        self.clip_loss = clip_loss
        self.perceptual_loss = perceptual_loss
        self.id_loss = id_loss

        self.w_clip = w_clip
        self.w_perceptual = w_perceptual
        self.w_id = w_id
        
        logger.info(f"TotalLoss initialized with weights -> CLIP: {self.w_clip}, Perceptual: {self.w_perceptual}, ID: {self.w_id}")

    def forward(
        self, 
        pred_frame: torch.Tensor, 
        text_prompt: List[str], 
        target_frame: torch.Tensor, 
        pred_frame_temporal: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the weighted aggregate loss.

        Args:
            pred_frame (torch.Tensor): The generated frame for CLIP evaluation.
            text_prompt (List[str]): The target text prompt for CLIP evaluation.
            target_frame (torch.Tensor): The reference previous frame (frame_t).
            pred_frame_temporal (torch.Tensor): The generated frame (frame_t1) for Temporal evaluation.
                (Often the same tensor as pred_frame, separated here for pipeline flexibility).

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - total: The scalar combined loss tensor attached to the computation graph.
                - details: A dictionary containing unweighted, detached individual losses for logging.
        """
        # Call sub-modules using the standardized BaseLoss interface (.forward(pred, target))
        l_clip = self.clip_loss(pred=pred_frame, target=text_prompt)
        l_perc = self.perceptual_loss(pred=pred_frame_temporal, target=target_frame)
        l_id = self.id_loss(pred=pred_frame_temporal, target=target_frame)

        # Aggregate weighted losses
        total = (
            self.w_clip * l_clip
            + self.w_perceptual * l_perc
            + self.w_id * l_id
        )

        # Extract detached float values for safe telemetry/logging (prevents memory leaks)
        details = {
            "clip": float(l_clip.detach().cpu()),
            "perceptual": float(l_perc.detach().cpu()),
            "id": float(l_id.detach().cpu()),
        }

        return total, details