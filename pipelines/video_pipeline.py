import logging
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import SchedulerMixin

# ✅ CANONICAL IMPORT PATHS
from modules.flow.latent_warp import LatentWarper
from pipelines.frame_generator import generate_next_frame
from modules.flow.flow_factory import build_flow
from modules.losses.loss_functions import build_loss

# 🔥 TSM PATCH IMPORT
from models.unet.unet_tsm_patch import inject_tsm_into_unet_midblock

logger = logging.getLogger(__name__)

class SDXLVideoPipeline(DiffusionPipeline):
    """
    SDXL-based Video Generation Pipeline (Hybrid Architecture).
    Supports BOTH Batch Temporal (TSM) and Frame-by-Frame (Flow+Loss) generation.
    """

    model_cpu_offload_seq = "unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        device: Optional[str] = None,
        flow_model_name: str = "raft",
        loss_config: Optional[Dict[str, Any]] = None,
        # ✅ YENİ: Mimari Karar Parametreleri
        enable_tsm: bool = False,
        tsm_num_frames: int = 16,
    ):
        super().__init__()

        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.enable_tsm = enable_tsm
        self.tsm_num_frames = tsm_num_frames
        
        # ---------------------------------------------------
        # 🔥 TSM INJECTION (Sadece Batch modundaysa aktif)
        # ---------------------------------------------------
        if self.enable_tsm:
            logger.info(f"Injecting Temporal Shift Module (TSM) into UNet for {self.tsm_num_frames} frames...")
            inject_tsm_into_unet_midblock(self.unet, num_frames=self.tsm_num_frames)
        else:
            # Frame-by-Frame Alfai Mimarisi için modülleri yükle
            self.loss_config = loss_config or {}
            logger.info(f"Initializing Flow Model: {flow_model_name}")
            self.flow_model = build_flow(name=flow_model_name, device=self.device_name)
            
            logger.info("Initializing Loss Orchestrator (CLIP, Perceptual, ID)")
            self.loss_module = build_loss(device=self.device_name, **self.loss_config)
            
            self.warper = LatentWarper(device=self.device_name)

        logger.info(f"SDXLVideoPipeline initialized on {self.device_name}. TSM Enabled: {self.enable_tsm}")

    def decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        # SDXL VAE, fp16'da numerically unstable — decode sırasında NaN veya
        # büyük değerler üretir, bunlar clamp ile 0'a (siyah) çekilir.
        # Diffusers'ın resmi çözümü: decode sırasında geçici olarak fp32'ye geç.
        original_dtype = self.vae.dtype
        needs_upcast = original_dtype == torch.float16

        if needs_upcast:
            self.vae.to(dtype=torch.float32)

        latents = latents.to(dtype=torch.float32) / self.vae.config.scaling_factor

        frames = []
        with torch.no_grad():
            for i in range(latents.shape[0]):
                decoded = self.vae.decode(latents[i : i + 1]).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                img = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
                img_safe = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
                frames.append(Image.fromarray((img_safe * 255).astype("uint8")))

        # Decode bitti, VAE'yi eski dtype'a geri çek (VRAM'i serbest bırak)
        if needs_upcast:
            self.vae.to(dtype=original_dtype)

        return frames

    @torch.no_grad()
    def __call__(
        self,
        prompt_embeds: Union[torch.Tensor, List[torch.Tensor]],      
        added_cond_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]], 
        num_frames: int = 16,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        target_driving_frames: Optional[List[torch.Tensor]] = None,
        ip_adapter_weights: Optional[List[float]] = None,            
        **kwargs: Any,
    ):
        # Güvenlik kontrolü: TSM açıksa num_frames, tsm_num_frames ile aynı olmalı
        if self.enable_tsm and num_frames != self.tsm_num_frames:
            logger.warning(f"TSM is enabled for {self.tsm_num_frames} frames, but generation requested {num_frames}. Overriding request.")
            num_frames = self.tsm_num_frames

        if self.enable_tsm:
            return self._generate_batch_temporal(
                prompt_embeds, added_cond_kwargs, num_frames, width, height, 
                num_inference_steps, guidance_scale, generator, ip_adapter_weights
            )
        else:
            return self._generate_frame_by_frame(
                prompt_embeds, added_cond_kwargs, num_frames, width, height, 
                num_inference_steps, guidance_scale, generator, target_driving_frames, ip_adapter_weights
            )

    # =========================================================================
    # MOD 1: ISHTIAQ'S BATCH TEMPORAL (TSM) MİMARİSİ
    # =========================================================================
    def _generate_batch_temporal(self, prompt_embeds, added_cond_kwargs, num_frames, width, height, steps, cfg, generator, ip_weights):
        logger.info(f"Running in Batch Temporal (TSM) Mode for {num_frames} frames...")
        device = self._execution_device
        batch_size = 1
        do_cfg = cfg > 1.0

        # Dinamik (Liste) olan promptları tensor haline getir (B*T)
        curr_prompt_embeds = prompt_embeds[0] if isinstance(prompt_embeds, list) else prompt_embeds
        curr_added_cond = added_cond_kwargs[0] if isinstance(added_cond_kwargs, list) else added_cond_kwargs
        
        # TSM mimarisinde prompt tensorleri T kere çoğaltılır
        curr_prompt_embeds = curr_prompt_embeds.repeat(num_frames, 1, 1)
        curr_added_cond["text_embeds"] = curr_added_cond["text_embeds"].repeat(num_frames, 1)
        curr_added_cond["time_ids"] = curr_added_cond["time_ids"].repeat(num_frames, 1)

        # CFG için kopyalama
        if do_cfg:
            curr_prompt_embeds = torch.cat([curr_prompt_embeds, curr_prompt_embeds], dim=0)
            curr_added_cond["text_embeds"] = torch.cat([curr_added_cond["text_embeds"], curr_added_cond["text_embeds"]], dim=0)
            curr_added_cond["time_ids"] = torch.cat([curr_added_cond["time_ids"], curr_added_cond["time_ids"]], dim=0)

        # Tüm kareler için tek bir büyük latents tensörü (B*T, 4, H, W)
        latents = torch.randn(
            (batch_size * num_frames, 4, height // 8, width // 8),
            device=device,
            generator=generator,
            dtype=self.unet.dtype,
        )

        self.scheduler.set_timesteps(steps, device=device)
        
        # Ortalama bir IP-Adapter ağırlığı kullan (TSM tüm kareleri aynı anda işlediği için dinamik ağırlık zordur)
        avg_ip_weight = sum(ip_weights) / len(ip_weights) if ip_weights else None
        cross_attention_kwargs = {"scale": avg_ip_weight} if avg_ip_weight else None

        for t in self.scheduler.timesteps:
            latent_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_input = self.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.unet(
                latent_input,
                t,
                encoder_hidden_states=curr_prompt_embeds,
                added_cond_kwargs=curr_added_cond,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return type("Output", (), {"frames": self.decode_latents(latents)})

    # =========================================================================
    # MOD 2: ALFAI'S FRAME-BY-FRAME (FLOW + LOSS) MİMARİSİ
    # =========================================================================
    def _generate_frame_by_frame(self, prompt_embeds, added_cond_kwargs, num_frames, width, height, steps, cfg, generator, target_driving_frames, ip_weights):
        logger.info(f"Running in Frame-by-Frame (Flow & Loss) Mode for {num_frames} frames...")
        device = self._execution_device
        video_frames: List[Image.Image] = []
        is_dynamic_prompt = isinstance(prompt_embeds, list)
        is_dynamic_kwargs = isinstance(added_cond_kwargs, list)

        # --- FRAME 0 ---
        curr_prompt_embeds = prompt_embeds[0] if is_dynamic_prompt else prompt_embeds
        curr_added_cond = added_cond_kwargs[0] if is_dynamic_kwargs else added_cond_kwargs
        curr_ip_weight = ip_weights[0] if ip_weights else None
        cross_attention_kwargs = {"scale": curr_ip_weight} if curr_ip_weight else None

        latents = torch.randn((1, 4, height // 8, width // 8), device=device, generator=generator, dtype=self.unet.dtype)
        self.scheduler.set_timesteps(steps, device=device)
        do_cfg = cfg > 1.0

        for t in self.scheduler.timesteps:
            latent_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_input = self.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.unet(
                latent_input, t,
                encoder_hidden_states=curr_prompt_embeds,
                added_cond_kwargs=curr_added_cond,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        current_latents = latents
        video_frames.extend(self.decode_latents(current_latents))

        # --- NEXT FRAMES (P-Frames) ---
        for i in range(1, num_frames):
            curr_prompt_embeds = prompt_embeds[i] if is_dynamic_prompt else prompt_embeds
            curr_added_cond = added_cond_kwargs[i] if is_dynamic_kwargs else added_cond_kwargs
            curr_ip_weight = ip_weights[i] if ip_weights else None
            cross_attention_kwargs = {"scale": curr_ip_weight} if curr_ip_weight else None

            prev_image_pil = video_frames[-1]
            prev_image_tensor = torch.from_numpy(np.array(prev_image_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            prev_image_tensor = prev_image_tensor.to(device)

            target_image_tensor = target_driving_frames[i].to(device) if target_driving_frames and len(target_driving_frames) > i else prev_image_tensor

            current_flow = self.flow_model.compute(prev_image_tensor, target_image_tensor)
            latent_flow = self.flow_model.resize_to_latent(current_flow, latent_h=height // 8, latent_w=width // 8)

            warped_latents, mask = self.warper.warp_and_create_mask(current_latents, latent_flow)

            next_latents = generate_next_frame(
                unet=self.unet, scheduler=self.scheduler, previous_latents_warped=warped_latents, mask=mask,
                prompt_embeds=curr_prompt_embeds, added_cond_kwargs=curr_added_cond, cross_attention_kwargs=cross_attention_kwargs,
                height=height, width=width, num_inference_steps=steps, guidance_scale=cfg, device=device, generator=generator,
                loss_module=self.loss_module, text_prompt="", prev_image_tensor=prev_image_tensor,
                vae=self.vae,
            )

            video_frames.extend(self.decode_latents(next_latents))
            current_latents = next_latents

        return type("Output", (), {"frames": video_frames})