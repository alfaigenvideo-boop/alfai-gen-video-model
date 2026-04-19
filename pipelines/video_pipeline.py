import logging
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import SchedulerMixin

# ✅ CANONICAL IMPORT PATHS: Modüler mimariye uygun içe aktarmalar
from modules.flow.latent_warp import LatentWarper
from pipelines.frame_generator import generate_next_frame
from modules.flow.flow_factory import build_flow
from modules.losses.loss_functions import build_loss

logger = logging.getLogger(__name__)

class SDXLVideoPipeline(DiffusionPipeline):
    """
    SDXL-based Video Generation Pipeline.

    Responsibilities:
    - First frame generation (I-Frame)
    - True Optical Flow estimation and Latent warping orchestration
    - Loss-guided Next frame generation (P-Frames) for temporal consistency
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
    ):
        super().__init__()

        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)

        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_config = loss_config or {}
        
        # ✅ NEW: Flow Model Entegrasyonu (Factory üzerinden tak-çalıştır)
        logger.info(f"Initializing Flow Model: {flow_model_name}")
        self.flow_model = build_flow(name=flow_model_name, device=self.device_name)
        
        # ✅ NEW: Loss Model Entegrasyonu (Builder üzerinden)
        logger.info("Initializing Loss Orchestrator (CLIP, Perceptual, ID)")
        self.loss_module = build_loss(device=self.device_name, **self.loss_config)
        
        # ✅ UPDATED: LatentWarper
        self.warper = LatentWarper(device=self.device_name)
        
        logger.info(f"SDXLVideoPipeline initialized fully on {self.device_name}")

    # -------------------------------------------------
    # LATENT → IMAGE (DECODING)
    # -------------------------------------------------
    # -------------------------------------------------
    # LATENT → IMAGE
    # -------------------------------------------------
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        # MLOps FIX: UNet float16 üretiyor olabilir, ancak SDXL VAE siyah ekran
        # (NaN) hatası vermesin diye float32'de çalışır. Bu yüzden latent'i
        # decode etmeden hemen önce VAE'nin veri tipine cast ediyoruz.
        latents = latents.to(dtype=self.vae.dtype)
        
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).astype("uint8"))
    # -------------------------------------------------
    # MAIN CALL (INFERENCE GRAPH)
    # -------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt_embeds: torch.Tensor,
        added_cond_kwargs: Dict[str, Any],
        num_frames: int = 16,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        target_driving_frames: Optional[List[torch.Tensor]] = None, # Video-to-Video veya ControlNet hedefleri için
        **kwargs: Any,
    ):
        device = self._execution_device
        video_frames: List[Image.Image] = []

        logger.info(f"Starting video pipeline: {num_frames} frames to be generated.")

        # -------------------------------------------------
        # FRAME 0 – I FRAME
        # -------------------------------------------------
        logger.info("Frame 0 (I-Frame) üretiliyor...")

        latents = torch.randn(
            (1, 4, height // 8, width // 8),
            device=device,
            generator=generator,
            dtype=self.unet.dtype,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # CFG (Classifier-Free Guidance) Kontrolü
        do_classifier_free_guidance = guidance_scale > 1.0

        for t in self.scheduler.timesteps:
            # CFG aktifse latent tensörünü Batch Size 2 olacak şekilde çoğaltıyoruz
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # CFG Formülü Uygulaması: uncond + guidance_scale * (cond - uncond)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        current_latents = latents
        video_frames.append(self.decode_latents(current_latents))
        logger.info("Frame 0 tamam.")

        # -------------------------------------------------
        # NEXT FRAMES – P FRAMES (Predictive Frames)
        # -------------------------------------------------
        for i in range(1, num_frames):
            logger.info(f"Processing Frame {i}/{num_frames}...")

            # 1. Önceki kareyi piksel uzayında tensöre çevirme (Flow hesaplaması için)
            prev_image_pil = video_frames[-1]
            prev_image_tensor = torch.from_numpy(np.array(prev_image_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            prev_image_tensor = prev_image_tensor.to(device)

            # 2. Target (Hedef) Kare Belirleme
            # Eğer dışarıdan bir rehber video (driving video) gelmişse onu kullanır,
            # gelmemişse otonom bir akış için mevcut kare üzerinden iterasyon yapar (Auto-regressive proxy).
            if target_driving_frames is not None and len(target_driving_frames) > i:
                target_image_tensor = target_driving_frames[i].to(device)
            else:
                target_image_tensor = prev_image_tensor # Placeholder fallback

            # 3. Gerçek Optical Flow Hesaplaması ve Latent Uzaya Küçültme
            current_flow = self.flow_model.compute(prev_image_tensor, target_image_tensor)
            latent_flow = self.flow_model.resize_to_latent(current_flow, latent_h=height // 8, latent_w=width // 8)

            # 4. Latent Warping ve Oklüzyon Maskesi Çıkarımı
            warped_latents, mask = self.warper.warp_and_create_mask(
                latent_prev=current_latents,
                flow_fwd=latent_flow,
            )

            # 5. Loss Guidance ile Sonraki Kare Üretimi
            # TotalLoss ve prompt'u generate_next_frame fonksiyonuna enjekte ediyoruz.
            next_latents = generate_next_frame(
                unet=self.unet,
                scheduler=self.scheduler,
                previous_latents_warped=warped_latents,
                mask=mask,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                device=device,
                generator=generator,
                loss_module=self.loss_module,       # ✅ YENİ: Loss Modülü (Guidance)
                text_prompt=prompt_embeds,          # ✅ YENİ: CLIP Loss için Text Embedding
                prev_image_tensor=prev_image_tensor # ✅ YENİ: ID ve Perceptual Loss için kaynak kare
            )

            video_frames.append(self.decode_latents(next_latents))
            current_latents = next_latents

        return type("Output", (), {"frames": video_frames})