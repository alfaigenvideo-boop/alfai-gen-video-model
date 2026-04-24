import torch
import logging
from typing import List, Tuple, Optional
from diffusers import ControlNetModel

logger = logging.getLogger(__name__)

class ControlNetWrapper:
    """
    ControlNet'leri Diffusers pipeline'ı olmadan, doğrudan UNet'e
    kalıntı (residual) tensörleri sağlamak için sarmalayan düşük seviyeli köprü.
    """
    def __init__(
        self, 
        pose_model_path: str = "thibaud/controlnet-openpose-sdxl-1.0",
        depth_model_path: str = "diffusers/controlnet-depth-sdxl-1.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.device = device
        self.dtype = dtype
        
        logger.info("ControlNet modelleri (Pose & Depth) yükleniyor...")
        self.controlnet_pose = ControlNetModel.from_pretrained(pose_model_path, torch_dtype=dtype).to(device)
        self.controlnet_depth = ControlNetModel.from_pretrained(depth_model_path, torch_dtype=dtype).to(device)
        
        # Test modunda çalıştır (Ağırlıklar güncellenmeyecek)
        self.controlnet_pose.eval()
        self.controlnet_depth.eval()

    @torch.no_grad()
    def get_residuals(
        self,
        noisy_latents: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        added_cond_kwargs: dict,
        pose_tensor: Optional[torch.Tensor] = None,   # GPU'daki RGB tensörü (PIL değil)
        depth_tensor: Optional[torch.Tensor] = None,  # GPU'daki RGB tensörü (PIL değil)
        pose_scale: float = 1.0,                      # PID'den gelecek
        depth_scale: float = 1.0                      # PID'den gelecek
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Gürültülü latent ve şartları (prompt) alıp, ControlNet'in UNet'e 
        enjekte edeceği ek özellikleri (residuals) hesaplar.
        """
        down_block_res_total = None
        mid_block_res_total = None

        # 1. Pose ControlNet İşlemi
        if pose_tensor is not None and pose_scale > 0:
            down_pose, mid_pose = self.controlnet_pose(
                noisy_latents, t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=pose_tensor,
                conditioning_scale=pose_scale,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )
            down_block_res_total = down_pose
            mid_block_res_total = mid_pose

        # 2. Depth ControlNet İşlemi
        if depth_tensor is not None and depth_scale > 0:
            down_depth, mid_depth = self.controlnet_depth(
                noisy_latents, t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=depth_tensor,
                conditioning_scale=depth_scale,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )
            
            # Eğer pose da varsa, tensörleri topluyoruz (Multi-ControlNet mantığı)
            if down_block_res_total is not None:
                down_block_res_total = tuple(p + d for p, d in zip(down_block_res_total, down_depth))
                mid_block_res_total += mid_depth
            else:
                down_block_res_total = down_depth
                mid_block_res_total = mid_depth

        return down_block_res_total, mid_block_res_total
