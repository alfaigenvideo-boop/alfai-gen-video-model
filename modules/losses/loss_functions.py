import logging
from typing import Any
from .clip_loss import CLIPLoss
from .perceptual_loss import PerceptualLoss
from .id_loss import IDLoss
from .total_loss import TotalLoss

logger = logging.getLogger(__name__)

def build_loss(device: str = "cuda", **kwargs: Any) -> TotalLoss:
    """
    Builder function to instantiate and aggregate all loss components.
    
    Args:
        device (str): Computation device (default: "cuda").
        **kwargs: Flexible keyword arguments for loss weights (e.g., lambda_clip, lambda_id).
        
    Returns:
        TotalLoss: An aggregated loss module containing CLIP, Perceptual, and ID losses.
    """
    logger.info(f"Building total loss module on {device}...")
    
    try:
        # Instantiate individual loss components
        clip = CLIPLoss(device=device)
        id_loss = IDLoss(device=device)
        
        # PerceptualLoss uses .to(device) based on its current implementation
        perceptual = PerceptualLoss().to(device)
        
        # Aggregate into TotalLoss wrapper
        total_loss = TotalLoss(
            clip_loss=clip,
            perceptual_loss=perceptual,
            id_loss=id_loss,
            **kwargs
        )
        
        logger.info("Total loss module built successfully.")
        return total_loss
        
    except Exception as e:
        logger.error(f"Failed to build loss module: {e}")
        raise