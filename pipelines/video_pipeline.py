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

from modules.controlnet_wrapper import ControlNetWrapper
from modules.ip_adapter_wrapper import IPAdapterWrapper
from modules.adaptive_control.adaptive_scheduler import AdaptiveScheduler
from modules.adaptive_control.temporal_metrics import FaceAnalyzer, CLIPScorer, TemporalMetrics
from pipelines.frame_generator import generate_next_frame

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
        adaptive_scheduler: AdaptiveScheduler = None,
        controlnet: ControlNetWrapper = None,
        ip_adapter: IPAdapterWrapper = None,
        face_analyzer: FaceAnalyzer = None,
        clip_scorer: CLIPScorer = None,
        loss_module: Optional[Any] = None,
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

        self.adaptive_scheduler = adaptive_scheduler
        self.controlnet = controlnet
        self.ip_adapter = ip_adapter
        self.face_analyzer = face_analyzer
        self.clip_scorer = clip_scorer
        self.loss_module = loss_module
        
        logger.info(f"SDXLVideoPipeline initialized fully on {self.device_name}")

    # -------------------------------------------------
    # TEXT ENCODING (PROMPT -> EMBEDS)
    # -------------------------------------------------
    def encode_prompt(self, prompt: str, device: str):
        """Metni SDXL'in beklediği embed tensörlerine çevirir."""
        # Eğer pipeline'a text_encoder'lar yüklenmemişse hata ver
        if not hasattr(self, "text_encoder") or not hasattr(self, "text_encoder_2"):
            raise ValueError("SDXL pipeline must be loaded with text_encoder and text_encoder_2")
            
        # 1. İlk Encoder (Base CLIP)
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]

        # 2. İkinci Encoder (OpenCLIP - SDXL Specific)
        text_inputs_2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=self.tokenizer_2.model_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids_2 = text_inputs_2.input_ids
        prompt_embeds_2 = self.text_encoder_2(text_input_ids_2.to(device), output_hidden_states=True)
        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

        # Embeddings'leri birleştir (Concat)
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        
        # CFG için Unconditional (Negatif) Embeddings
        uncond_tokens = [""] # Negatif prompt boş
        uncond_inputs = self.tokenizer(
            uncond_tokens, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        uncond_embeds = self.text_encoder(uncond_inputs.input_ids.to(device), output_hidden_states=True)
        uncond_pooled = uncond_embeds[0]
        uncond_embeds = uncond_embeds.hidden_states[-2]
        
        uncond_inputs_2 = self.tokenizer_2(
            uncond_tokens, padding="max_length", max_length=self.tokenizer_2.model_max_length, truncation=True, return_tensors="pt"
        )
        uncond_embeds_2 = self.text_encoder_2(uncond_inputs_2.input_ids.to(device), output_hidden_states=True)
        uncond_pooled_2 = uncond_embeds_2[0]
        uncond_embeds_2 = uncond_embeds_2.hidden_states[-2]
        
        uncond_embeds = torch.cat([uncond_embeds, uncond_embeds_2], dim=-1)
        
        # Batch=2 yap (Uncond + Cond)
        final_prompt_embeds = torch.cat([uncond_embeds, prompt_embeds])
        final_pooled_embeds = torch.cat([uncond_pooled_2, pooled_prompt_embeds_2]) # SDXL uses pooled from encoder 2
        
        return final_prompt_embeds, final_pooled_embeds
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
        latents = latents.to(dtype=torch.float32)
        self.vae.to(dtype=torch.float32)
        
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
        prompt: str,
        prompt_embeds: torch.Tensor,
        added_cond_kwargs: Dict[str, Any],
        identity_image: Image.Image,
        pose_tensors: List[torch.Tensor],
        depth_tensors: List[torch.Tensor],
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
        current_scales = {"ip_scale": 0.7, "pose_scale": 1.0, "depth_scale": 1.0, "mask_threshold": 0.03}
        prev_frame = None
        ip_embeds = self.ip_adapter.get_image_embeds(identity_image)

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
        prev_frame = video_frames[0]
        # -------------------------------------------------
        # NEXT FRAMES – P FRAMES (Predictive Frames)
        # -------------------------------------------------
        for i in range(1, num_frames):
            logger.info(f"Generating Frame {i} | Scales: {current_scales}")

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
                controlnet_wrapper=self.controlnet,
                pose_tensor=pose_tensors[i],
                depth_tensor=depth_tensors[i],
                pose_scale=current_scales["pose_scale"],
                depth_scale=current_scales["depth_scale"],
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

            current_frame = self.decode_latents(next_latents)
            video_frames.append(current_frame)

            # ✅ YENİ: Sensörleri Çalıştır (Geri besleme döngüsü)
            if self.adaptive_scheduler:
                id_score = self.face_analyzer.similarity(identity_image, current_frame)
                res = TemporalMetrics.compute_residual(prev_frame, current_frame)
                clip_s = self.clip_scorer.similarity(current_frame, prompt)
                
                # Bir sonraki kare için yeni katsayıları hesapla
                current_scales = self.adaptive_scheduler.step(id_score, res, clip_s)
                # IP-Adapter scale'i anlık güncelle
                self.ip_adapter.set_scale(current_scales["ip_scale"])

            prev_frame = current_frame

        return type("Output", (), {"frames": video_frames})
