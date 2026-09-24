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
from modules.prompt_to_prompt import PromptBlender
from modules.interpolation.latent_interpolation import noise_blend_schedule

# 🔥 TSM PATCH IMPORT
from models.unet.unet_tsm_patch import inject_tsm_into_unet_midblock

# 🎭 IP-ADAPTER FACEID IMPORT
from modules.ip_adapter_wrapper import IPAdapterFaceID, FaceIDExtractor

# 🧭 T2I-ADAPTER (lightweight geometry lock; ControlNet alternative)
from modules.t2i_adapter_wrapper import T2IAdapterWrapper

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

        # ---------------------------------------------------
        # 🎭 IP-ADAPTER FACEID (her iki modda da opsiyonel)
        # ---------------------------------------------------
        self.ip_adapter: Optional[IPAdapterFaceID] = None
        self.face_extractor: Optional[FaceIDExtractor] = None

        # ---------------------------------------------------
        # 🧭 T2I-ADAPTER (pose/depth/canny; her iki modda da opsiyonel)
        # ---------------------------------------------------
        self.t2i_adapter: Optional[T2IAdapterWrapper] = None

        logger.info(f"SDXLVideoPipeline initialized on {self.device_name}. TSM Enabled: {self.enable_tsm}")

    def load_ip_adapter(self, local_path: Optional[str] = None, scale: float = 0.8) -> bool:
        """
        Loads IP-Adapter FaceID weights and injects attention processors into UNet.
        Call after pipeline is created, before generation.

        Args:
            local_path: Local .bin file path. If None, downloads from HuggingFace.
            scale: Initial IP-Adapter scale (0.0 = disabled, 1.0 = max identity lock).

        Returns:
            True if successfully loaded.
        """
        logger.info("Loading IP-Adapter FaceID...")
        self.ip_adapter = IPAdapterFaceID(unet=self.unet, device=self.device_name, scale=scale)
        success = self.ip_adapter.load_weights(local_path=local_path)
        if not success:
            self.ip_adapter = None
        self.face_extractor = FaceIDExtractor(device="cpu")  # Face detection on CPU to save VRAM
        return success

    def load_t2i_adapter(
        self,
        adapter_type: str = "canny",
        local_path: Optional[str] = None,
        scale: float = 0.8,
        source: str = "hybrid",
        canny_low: int = 100,
        canny_high: int = 200,
        dtype: Optional[torch.dtype] = None,
    ) -> bool:
        """
        Loads T2I-Adapter weights. Call after pipeline is created, before generation.

        Args:
            adapter_type: canny | depth | pose | lineart | sketch
            local_path: Local adapter folder. If None, downloads from HuggingFace.
            scale: Residual strength (0.0 = off, 1.0 = full geometry lock).
            source: reference | previous_frame | hybrid
            dtype: Defaults to UNet dtype.
        """
        resolved_dtype = dtype or next(self.unet.parameters()).dtype
        logger.info(f"Loading T2I-Adapter ({adapter_type})...")
        self.t2i_adapter = T2IAdapterWrapper(
            device=self.device_name,
            dtype=resolved_dtype,
            scale=scale,
            source=source,
            condition_type=adapter_type,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        success = self.t2i_adapter.load(local_path=local_path)
        if not success:
            self.t2i_adapter = None
        return success

    def _resolve_t2i_residuals(
        self,
        frame_index: int,
        do_cfg: bool,
        ref_residuals: Optional[List[torch.Tensor]] = None,
        prev_image_pil: Optional[Image.Image] = None,
        batch_size: int = 1,
    ) -> Optional[List[torch.Tensor]]:
        """Picks reference vs previous-frame residuals and expands for CFG / TSM batch."""
        if self.t2i_adapter is None:
            return None

        source = self.t2i_adapter.source
        residuals = None

        if source == "reference":
            residuals = ref_residuals
        elif source == "previous_frame":
            if prev_image_pil is not None:
                residuals = self.t2i_adapter.encode_image(prev_image_pil)
            else:
                residuals = ref_residuals
        else:  # hybrid
            if frame_index == 0:
                residuals = ref_residuals
            elif prev_image_pil is not None:
                residuals = self.t2i_adapter.encode_image(prev_image_pil)
            else:
                residuals = ref_residuals

        if residuals is None:
            return None

        residuals = T2IAdapterWrapper.repeat_for_batch(residuals, batch_size)
        return T2IAdapterWrapper.expand_for_cfg(residuals, do_cfg)

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
        initial_latent: Optional[torch.Tensor] = None,   # I2V: DDIM inversion çıktısı
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,  # I2V: Face embeds (1, 4, 2048)
        t2i_residuals: Optional[List[torch.Tensor]] = None,  # I2V: T2I-Adapter residuals from reference
        **kwargs: Any,
    ):
        # Güvenlik kontrolü: TSM açıksa num_frames, tsm_num_frames ile aynı olmalı
        if self.enable_tsm and num_frames != self.tsm_num_frames:
            logger.warning(f"TSM is enabled for {self.tsm_num_frames} frames, but generation requested {num_frames}. Overriding request.")
            num_frames = self.tsm_num_frames

        if self.enable_tsm:
            return self._generate_batch_temporal(
                prompt_embeds, added_cond_kwargs, num_frames, width, height, 
                num_inference_steps, guidance_scale, generator, ip_adapter_weights,
                t2i_residuals,
            )
        else:
            return self._generate_frame_by_frame(
                prompt_embeds, added_cond_kwargs, num_frames, width, height, 
                num_inference_steps, guidance_scale, generator, target_driving_frames,
                ip_adapter_weights, initial_latent, ip_adapter_image_embeds,
                t2i_residuals,
            )

    # =========================================================================
    # MOD 1: ISHTIAQ'S BATCH TEMPORAL (TSM) MİMARİSİ
    # =========================================================================
    def _generate_batch_temporal(self, prompt_embeds, added_cond_kwargs, num_frames, width, height, steps, cfg, generator, ip_weights, t2i_residuals=None):
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

        t2i_cfg_residuals = self._resolve_t2i_residuals(
            frame_index=0,
            do_cfg=do_cfg,
            ref_residuals=t2i_residuals,
            prev_image_pil=None,
            batch_size=num_frames,
        )

        for t in self.scheduler.timesteps:
            latent_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_input = self.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.unet(
                latent_input,
                t,
                encoder_hidden_states=curr_prompt_embeds,
                added_cond_kwargs=curr_added_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                down_intrablock_additional_residuals=T2IAdapterWrapper.clone_residuals(t2i_cfg_residuals),
            ).sample

            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return type("Output", (), {"frames": self.decode_latents(latents)})

    # =========================================================================
    # MOD 2: ALFAI'S FRAME-BY-FRAME (FLOW + LOSS) MİMARİSİ
    # =========================================================================
    def _generate_frame_by_frame(self, prompt_embeds, added_cond_kwargs, num_frames, width, height, steps, cfg, generator, target_driving_frames, ip_weights, initial_latent=None, ip_adapter_image_embeds=None, t2i_residuals=None):
        logger.info(f"Running in Frame-by-Frame (Flow & Loss) Mode for {num_frames} frames...")
        device = self._execution_device
        video_frames: List[Image.Image] = []
        is_dynamic_prompt = isinstance(prompt_embeds, list)
        is_dynamic_kwargs = isinstance(added_cond_kwargs, list)

        # IP-Adapter ağırlık takvimi: [0.9, 0.9, ..., 0.6, ..., 0.3]
        # Mevcut PromptBlender'dan üret, yoksa sabit ağırlık kullan
        if ip_adapter_image_embeds is not None and self.ip_adapter is not None:
            faceid_weight_schedule = PromptBlender.generate_ip_weights(num_frames)
        else:
            faceid_weight_schedule = [0.0] * num_frames

        # --- FRAME 0 ---
        curr_prompt_embeds = prompt_embeds[0] if is_dynamic_prompt else prompt_embeds
        curr_added_cond = added_cond_kwargs[0] if is_dynamic_kwargs else added_cond_kwargs
        curr_ip_weight = ip_weights[0] if ip_weights else None
        cross_attention_kwargs = {"scale": curr_ip_weight} if curr_ip_weight else None

        # IP-Adapter: frame 0 için yüz embedding'ini prompt embed'e ekle
        frame0_prompt_embeds = curr_prompt_embeds
        if ip_adapter_image_embeds is not None and self.ip_adapter is not None:
            scale_0 = faceid_weight_schedule[0]
            self.ip_adapter.set_scale(scale_0)
            # image embeds: (1, 4, 2048) → CFG için duplicate et (uncond + cond)
            ip_embeds = ip_adapter_image_embeds.to(device=device, dtype=curr_prompt_embeds.dtype)
            ip_embeds_cfg = torch.cat([torch.zeros_like(ip_embeds), ip_embeds], dim=0)  # (2, 4, 2048)
            # prompt_embeds sonuna ekle: (2, 77+4, 2048)
            frame0_prompt_embeds = torch.cat([curr_prompt_embeds, ip_embeds_cfg], dim=1)
            logger.info(f"[I2V] Frame 0: IP-Adapter FaceID aktif (scale={scale_0:.2f}).")

        # I2V: initial_latent verilmişse DDIM inverted latent'ten başla
        # T2V: saf gürültüden başla
        if initial_latent is not None:
            logger.info("[I2V] Frame 0: DDIM inverted latent kullanılıyor (referans görsel ankrajı).")
            latents = initial_latent.to(device=device, dtype=self.unet.dtype)
        else:
            logger.info("[T2V] Frame 0: Saf gürültüden başlanıyor.")
            latents = torch.randn((1, 4, height // 8, width // 8), device=device, generator=generator, dtype=self.unet.dtype)

        self.scheduler.set_timesteps(steps, device=device)
        do_cfg = cfg > 1.0

        frame0_t2i = self._resolve_t2i_residuals(
            frame_index=0,
            do_cfg=do_cfg,
            ref_residuals=t2i_residuals,
            prev_image_pil=None,
        )
        if frame0_t2i is not None:
            logger.info(f"[T2I] Frame 0: T2I-Adapter aktif (type={self.t2i_adapter.condition_type}, scale={self.t2i_adapter.scale:.2f}).")

        for t in self.scheduler.timesteps:
            latent_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_input = self.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.unet(
                latent_input, t,
                encoder_hidden_states=frame0_prompt_embeds,
                added_cond_kwargs=curr_added_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                down_intrablock_additional_residuals=T2IAdapterWrapper.clone_residuals(frame0_t2i),
            ).sample

            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        current_latents = latents
        video_frames.extend(self.decode_latents(current_latents))

        # --- NEXT FRAMES (P-Frames) ---
        # SLERP: I2V modunda P-frame başlangıç latent'leri
        # DDIM latent'ten başlayarak giderek saf gürültüye blend yap
        # T2V modunda saf rastgele gürültü kullan
        if initial_latent is not None and num_frames > 1:
            p_frame_latents = noise_blend_schedule(
                ddim_latent=current_latents,
                num_frames=num_frames - 1,    # frame 0 zaten üretildi
                start_blend=0.15,             # frame 1: referansa çok yakın
                end_blend=0.70,               # son frame: daha serbest
                generator=generator,
            )
        else:
            p_frame_latents = None  # T2V: generate_next_frame içinde rastgele gürültü

        for i in range(1, num_frames):
            curr_prompt_embeds = prompt_embeds[i] if is_dynamic_prompt else prompt_embeds
            curr_added_cond = added_cond_kwargs[i] if is_dynamic_kwargs else added_cond_kwargs
            curr_ip_weight = ip_weights[i] if ip_weights else None
            cross_attention_kwargs = {"scale": curr_ip_weight} if curr_ip_weight else None

            # IP-Adapter: her frame için azalan ağırlık takvimi
            frame_prompt_embeds = curr_prompt_embeds
            if ip_adapter_image_embeds is not None and self.ip_adapter is not None:
                scale_i = faceid_weight_schedule[i]
                self.ip_adapter.set_scale(scale_i)
                ip_embeds = ip_adapter_image_embeds.to(device=device, dtype=curr_prompt_embeds.dtype)
                ip_embeds_cfg = torch.cat([torch.zeros_like(ip_embeds), ip_embeds], dim=0)
                frame_prompt_embeds = torch.cat([curr_prompt_embeds, ip_embeds_cfg], dim=1)

            prev_image_pil = video_frames[-1]
            prev_image_tensor = torch.from_numpy(np.array(prev_image_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            prev_image_tensor = prev_image_tensor.to(device)

            target_image_tensor = target_driving_frames[i].to(device) if target_driving_frames and len(target_driving_frames) > i else prev_image_tensor

            current_flow = self.flow_model.compute(prev_image_tensor, target_image_tensor)
            latent_flow = self.flow_model.resize_to_latent(current_flow, latent_h=height // 8, latent_w=width // 8)

            warped_latents, mask = self.warper.warp_and_create_mask(current_latents, latent_flow)

            next_latents = generate_next_frame(
                unet=self.unet, scheduler=self.scheduler, previous_latents_warped=warped_latents, mask=mask,
                prompt_embeds=frame_prompt_embeds, added_cond_kwargs=curr_added_cond, cross_attention_kwargs=cross_attention_kwargs,
                height=height, width=width, num_inference_steps=steps, guidance_scale=cfg, device=device, generator=generator,
                loss_module=self.loss_module, text_prompt="", prev_image_tensor=prev_image_tensor,
                vae=self.vae,
                initial_latent=p_frame_latents[i - 1] if p_frame_latents else None,  # SLERP latent
                down_intrablock_additional_residuals=self._resolve_t2i_residuals(
                    frame_index=i,
                    do_cfg=do_cfg,
                    ref_residuals=t2i_residuals,
                    prev_image_pil=prev_image_pil,
                ),
            )

            video_frames.extend(self.decode_latents(next_latents))
            current_latents = next_latents

        return type("Output", (), {"frames": video_frames})