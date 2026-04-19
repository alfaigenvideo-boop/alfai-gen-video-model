import logging
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class LatentWarper:
    """
    Manages warping and masking operations in SDXL Latent space (128x128).
    Includes advanced occlusion detection via forward-backward consistency checks.
    """
    
    def __init__(self, device: str = "cuda", consistency_threshold: float = 0.03):
        """
        Initializes the LatentWarper.
        
        Args:
            device (str): Computation device (default: "cuda").
            consistency_threshold (float): Threshold for forward-backward flow consistency.
        """
        self.device = device
        # Normalized coordinate space constants
        self._coord_min: float = -1.0
        self._coord_max: float = 1.0
        self.consistency_threshold = consistency_threshold
        
        logger.debug(f"LatentWarper initialized on {self.device} with consistency_threshold={self.consistency_threshold}")

    def _create_standard_grid(self, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        """Creates a normalized identity grid for grid_sampling."""
        x_coords = torch.linspace(self._coord_min, self._coord_max, width, device=self.device, dtype=dtype)
        y_coords = torch.linspace(self._coord_min, self._coord_max, height, device=self.device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    def _normalize_flow(self, flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Normalizes raw flow vectors to the [-1, 1] coordinate system."""
        # [B, C, H, W] -> [B, H, W, C]
        flow_permuted = flow.permute(0, 2, 3, 1)
        normalized_flow = torch.clone(flow_permuted)
        normalized_flow[..., 0] *= (2.0 / width)
        normalized_flow[..., 1] *= (2.0 / height)
        return normalized_flow

    def warp_and_create_mask(
        self, 
        latent_prev: torch.Tensor, 
        flow_fwd: torch.Tensor, 
        flow_bwd: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Warps previous latent to current frame and generates an inpaint mask.

        Args:
            latent_prev: Latent tensor from frame t-1 [B, 4, H, W]
            flow_fwd: Forward optical flow (T -> T+1) [B, 2, H, W]
            flow_bwd: Backward optical flow (T+1 -> T) [B, 2, H, W] (Optional)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - warped_latent: The warped latent representation.
                - inpaint_mask: Binary mask where 1.0 represents regions to inpaint.
        """
        # Ensure dtype consistency between flow and latents (Handles Half vs Float seamlessly)
        if flow_fwd.dtype != latent_prev.dtype:
            flow_fwd = flow_fwd.to(dtype=latent_prev.dtype)
            logger.debug("Converted flow_fwd dtype to match latent_prev.")
        
        if flow_bwd is not None and flow_bwd.dtype != latent_prev.dtype:
            flow_bwd = flow_bwd.to(dtype=latent_prev.dtype)
            logger.debug("Converted flow_bwd dtype to match latent_prev.")

        batch_size, channels, height, width = latent_prev.shape
        
        # 1. Coordinate Grid Generation
        base_grid = self._create_standard_grid(height, width, latent_prev.dtype)

        # 2. Flow Normalization
        flow_fwd_norm = self._normalize_flow(flow_fwd, height, width)
        
        # 3. Target Sampling Grid (Backward Warping Logic)
        sampling_grid = base_grid - flow_fwd_norm
        
        # 4. Perform Latent Warping
        warped_latent = F.grid_sample(
            latent_prev, 
            sampling_grid, 
            mode='bicubic', 
            padding_mode='zeros',
            align_corners=False
        )
        
        # 5. Mask Generation Logic
        
        # A) Boundary Check: Detect out-of-bounds sampling
        inside_bounds_mask = (sampling_grid[..., 0] >= self._coord_min) & \
                             (sampling_grid[..., 0] <= self._coord_max) & \
                             (sampling_grid[..., 1] >= self._coord_min) & \
                             (sampling_grid[..., 1] <= self._coord_max)
        
        # B) Occlusion Check: Forward-Backward Consistency
        if flow_bwd is not None:
            flow_bwd_norm = self._normalize_flow(flow_bwd, height, width)

            # Warp the backward flow to current grid for comparison
            warped_bwd_flow = F.grid_sample(
                flow_bwd_norm.permute(0, 3, 1, 2), 
                sampling_grid, 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
            
            # Calculate L2 norm of the flow residual
            flow_residual = (flow_fwd_norm + warped_bwd_flow).norm(dim=-1)
            consistency_mask = flow_residual < self.consistency_threshold
            
            valid_region_mask = inside_bounds_mask & consistency_mask
        else:
            valid_region_mask = inside_bounds_mask

        # Format Mask: [B, H, W] -> [B, 1, H, W]
        # 1.0 = Inpaint (Invalid), 0.0 = Keep (Valid)
        valid_pixels = valid_region_mask.unsqueeze(1).to(dtype=latent_prev.dtype)
        inpaint_mask = 1.0 - valid_pixels
        
        return warped_latent, inpaint_mask