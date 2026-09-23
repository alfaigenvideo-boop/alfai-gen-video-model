# inference.py
import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import yaml
import imageio
import numpy as np
from PIL import Image

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel as OfficialUNet,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from utils.model_utils import get_device, flush_vram
from pipelines.video_pipeline import SDXLVideoPipeline
from modules.interpolation.rife import RIFEInterpolator
from modules.inversion.ddim_inversion import DDIMInversion

# --- ALFAI CUSTOM UNET IMPORT ---
try:
    from models.unet.unet_base import UNet2DConditionModel as CustomUNet
except ImportError as e:
    print(f"HATA: models/unet/unet_base.py dosyası bulunamadı: {e}")
    print("Lütfen sys.path veya Python çalışma dizininizin doğru olduğundan emin olun.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("inference")

# SDXL base model ID (ağırlıklar yoksa HuggingFace'ten otomatik çekilir)
SDXL_HF_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# ============================================================
# INTERPOLATION CONFIG LOADER
# ============================================================
def load_interpolation_config():
    with open("configs/interpolation.yaml", "r") as f:
        return yaml.safe_load(f)["interpolation"]


# ============================================================
# VIDEO GENERATOR
# ============================================================
class VideoGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        self.dtype = (
            torch.float16
            if config.get("runtime", {}).get("precision", "fp16") == "fp16"
            else torch.float32
        )

        self.pipeline = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None
        self.text_encoder_2 = None

        self._load_models()

    # --------------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------------
    def _load_models(self):
        logger.info("🔄 Loading SDXL models...")

        unet_path = self.config.get("paths", {}).get("models", {}).get("sdxl_base", None)
        vae_path = self.config.get("paths", {}).get("models", {}).get("sdxl_vae", None)

        # --- Text Encoders & Tokenizers (SDXL dual-encoder) ---
        logger.info("Loading SDXL text encoders and tokenizers from HuggingFace...")
        self.tokenizer = CLIPTokenizer.from_pretrained(SDXL_HF_ID, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(SDXL_HF_ID, subfolder="tokenizer_2")

        self.text_encoder = CLIPTextModel.from_pretrained(
            SDXL_HF_ID, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device)
        self.text_encoder.eval()

        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            SDXL_HF_ID, subfolder="text_encoder_2", torch_dtype=self.dtype
        ).to(self.device)
        self.text_encoder_2.eval()

        # --- UNet ---
        # Yerel .safetensors dosyası varsa kullan, yoksa HuggingFace'ten yükle
        if unet_path and os.path.isfile(unet_path):
            logger.info(f"Loading UNet from local file: {unet_path}")
            temp_unet = OfficialUNet.from_single_file(unet_path, torch_dtype=self.dtype)
        else:
            logger.info(f"UNet local file not found, loading from HuggingFace: {SDXL_HF_ID}")
            temp_unet = OfficialUNet.from_pretrained(
                SDXL_HF_ID, subfolder="unet", torch_dtype=self.dtype
            )

        logger.info("Transferring weights to Custom UNet architecture...")
        custom_unet = CustomUNet(**temp_unet.config)
        custom_unet.load_state_dict(temp_unet.state_dict(), strict=False)
        custom_unet.to(device=self.device, dtype=self.dtype)
        del temp_unet
        flush_vram()

        # --- VAE ---
        if vae_path and os.path.isfile(vae_path):
            logger.info(f"Loading VAE from local file: {vae_path}")
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=self.dtype).to(self.device)
        else:
            logger.info(f"VAE local file not found, loading from HuggingFace: {SDXL_HF_ID}")
            vae = AutoencoderKL.from_pretrained(
                SDXL_HF_ID, subfolder="vae", torch_dtype=self.dtype
            ).to(self.device)

        # --- Scheduler ---
        logger.info("Loading scheduler...")
        scheduler = DDIMScheduler.from_pretrained(SDXL_HF_ID, subfolder="scheduler")

        # --- Video Pipeline ---
        logger.info("Setting up SDXLVideoPipeline...")
        self.pipeline = SDXLVideoPipeline(
            unet=custom_unet,
            vae=vae,
            scheduler=scheduler,
        )
        self.pipeline.to(self.device)

        if self.config.get("runtime", {}).get("enable_model_cpu_offload", False):
            self.pipeline.enable_model_cpu_offload()

        # DDIMInversion: referans görsel → latent dönüştürmek için
        # Null prompt embeds (sıfır tensör) ile başlatılıyor
        self.inverter = DDIMInversion(
            pipe=self.pipeline,
            device=self.device,
        )

        logger.info("✅ Models loaded successfully")

    # --------------------------------------------------------
    # PROMPT ENCODING (SDXL Dual Text Encoder)
    # --------------------------------------------------------
    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 576,
    ):
        """
        SDXL'in çift encoder'ını kullanarak prompt'u embeds'e çevirir.
        Döner: (prompt_embeds, negative_prompt_embeds, added_cond_kwargs, negative_added_cond_kwargs)
        CFG için cond + uncond birleştirilmiş tek tensör olarak döner.
        """
        def _encode_single(tokenizer, text_encoder, text):
            tokens = tokenizer(
                text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(self.device)
            with torch.no_grad():
                out = text_encoder(input_ids, output_hidden_states=True)
            # SDXL: penultimate hidden state
            hidden = out.hidden_states[-2]
            return hidden

        def _encode_single_2(tokenizer, text_encoder, text):
            tokens = tokenizer(
                text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(self.device)
            with torch.no_grad():
                out = text_encoder(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-2]
            pooled = out.text_embeds  # CLIPTextModelWithProjection'dan pooled embed
            return hidden, pooled

        # Cond
        h1_cond = _encode_single(self.tokenizer, self.text_encoder, prompt)
        h2_cond, pooled_cond = _encode_single_2(self.tokenizer_2, self.text_encoder_2, prompt)
        prompt_embeds_cond = torch.cat([h1_cond, h2_cond], dim=-1)  # (1, 77, 2048)

        # Uncond (negative)
        h1_uncond = _encode_single(self.tokenizer, self.text_encoder, negative_prompt)
        h2_uncond, pooled_uncond = _encode_single_2(self.tokenizer_2, self.text_encoder_2, negative_prompt)
        prompt_embeds_uncond = torch.cat([h1_uncond, h2_uncond], dim=-1)  # (1, 77, 2048)

        # CFG için cond + uncond birleştir (pipeline içinde chunk(2) ile ayrılacak)
        prompt_embeds = torch.cat([prompt_embeds_uncond, prompt_embeds_cond], dim=0)  # (2, 77, 2048)
        pooled_embeds = torch.cat([pooled_uncond, pooled_cond], dim=0)  # (2, 1280)

        # SDXL added_cond_kwargs: time_ids = [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]],
            dtype=self.dtype,
            device=self.device,
        ).repeat(2, 1)  # Uncond + Cond için 2 kopya

        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": time_ids,
        }

        return prompt_embeds, added_cond_kwargs

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 24,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        reference_image_path: Optional[str] = None,  # I2V: tek referans fotoğraf
        ip_adapter_scale: float = 0.0,
    ) -> str:
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        logger.info("🔤 Encoding prompt...")
        prompt_embeds, added_cond_kwargs = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
        )

        # --------------------------------------------------------
        # I2V: Referans fotoğraf varsa DDIM inversion ile latent ankrajı oluştur
        # --------------------------------------------------------
        initial_latent = None
        target_driving_frames = None
        if reference_image_path and Path(reference_image_path).exists():
            logger.info(f"📷 I2V Modu: Referans fotoğraf yükleniyor → {reference_image_path}")
            ref_img = Image.open(reference_image_path).convert("RGB").resize((width, height))
            ref_np = np.array(ref_img).astype(np.float32) / 255.0

            # [H,W,3] → [1,3,H,W], normalize to [-1,1] for VAE
            ref_tensor_vae = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0)
            ref_tensor_vae = (ref_tensor_vae * 2 - 1).to(device=self.device, dtype=torch.float32)

            # [0,1] aralığında pixel tensor (flow hesabı için)
            ref_tensor_flow = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # stop_at_t=600: görselin yapısını koru ama prompt'un etki etmesine yer bırak
            # (1000 = tamamen gürültü, 0 = orijinal görsel)
            logger.info("🔄 DDIM Inversion çalıştırılıyor (stop_at_t=600)...")
            initial_latent = self.inverter.invert(
                image=ref_tensor_vae,
                num_steps=num_inference_steps,
                stop_at_t=600,
            )
            logger.info("✅ Referans görsel latent uzayına dönüştürüldü.")

            # Tüm P-frame'lerde flow hedefi olarak referans kullanılır
            target_driving_frames = [ref_tensor_flow] * num_frames
        else:
            if reference_image_path:
                logger.warning(f"⚠️ Referans fotoğraf bulunamadı: {reference_image_path} — T2V modunda devam ediliyor.")
            logger.info("🎬 T2V Modu: Saf prompt'tan üretim yapılıyor.")

        # ----------------------------------------------------
        # RUN SDXL VIDEO PIPELINE
        # ----------------------------------------------------
        logger.info(f"🎬 Starting generation for: '{prompt}'")
        output = self.pipeline(
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.config.get("model", {}).get("generation", {}).get("guidance_scale", 7.5),
            generator=generator,
            target_driving_frames=target_driving_frames,
            initial_latent=initial_latent,  # I2V: DDIM inverted latent (T2V'de None)
        )

        frames = output.frames

        # ----------------------------------------------------
        # SAVE RAW VIDEO
        # ----------------------------------------------------
        os.makedirs("outputs", exist_ok=True)
        raw_video_path = os.path.join("outputs", "raw_output.mp4")

        imageio.mimsave(
            raw_video_path,
            [frame if not hasattr(frame, 'convert') else frame for frame in frames],
            fps=self.config.get("model", {}).get("generation", {}).get("fps", 8),
        )

        logger.info(f"🎬 Raw video saved at {raw_video_path}")
        flush_vram()

        # ----------------------------------------------------
        # POST-PROCESS WITH RIFE (opsiyonel)
        # ----------------------------------------------------
        try:
            interp_config = load_interpolation_config()
            if interp_config.get("enabled", False):
                logger.info("🚀 Running RIFE post-process...")
                interpolator = RIFEInterpolator(interp_config)
                final_video_path = interpolator.interpolate(raw_video_path)
                logger.info(f"✅ Smooth video saved at {final_video_path}")
                return final_video_path
        except Exception as e:
            logger.warning(f"RIFE post-process atlandı (hata veya config eksik): {e}")

        return raw_video_path