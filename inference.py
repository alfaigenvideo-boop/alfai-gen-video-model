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
from modules.controlnet_wrapper import load_driving_frames
from modules.prompt_to_prompt import PromptBlender

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
            logger.info(f"UNet local file not found ({unet_path}), loading from HuggingFace: {SDXL_HF_ID}")
            temp_unet = None
            if self.dtype == torch.float16:
                # fp16 varyantı ~5 GB; varsayılan fp32 dosya ~10 GB ve RAM'e sığmayabilir
                try:
                    temp_unet = OfficialUNet.from_pretrained(
                        SDXL_HF_ID, subfolder="unet", torch_dtype=self.dtype, variant="fp16"
                    )
                except Exception as e:
                    logger.warning(f"fp16 UNet varyantı yüklenemedi, fp32 dosya deneniyor: {e}")
            if temp_unet is None:
                temp_unet = OfficialUNet.from_pretrained(
                    SDXL_HF_ID, subfolder="unet", torch_dtype=self.dtype
                )

        logger.info("Transferring weights to Custom UNet architecture...")
        # Özel UNet'i doğrudan hedef dtype'ta kur: fp32 kurulum SDXL için ~10 GB ek RAM ister
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            custom_unet = CustomUNet(**temp_unet.config)
        finally:
            torch.set_default_dtype(prev_dtype)
        custom_unet.load_state_dict(temp_unet.state_dict(), strict=False)
        custom_unet.to(device=self.device, dtype=self.dtype)
        del temp_unet
        flush_vram()

        # --- VAE ---
        if vae_path and os.path.isfile(vae_path):
            logger.info(f"Loading VAE from local file: {vae_path}")
            # config verilmezse diffusers SD 1.5 VAE config'ini tahmin ediyor (scaling_factor 0.18215);
            # SDXL için doğru değer 0.13025 — yanlışı renk/kontrast bozulmasına yol açar
            vae = AutoencoderKL.from_single_file(
                vae_path, config=SDXL_HF_ID, subfolder="vae", torch_dtype=self.dtype
            ).to(self.device)
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

        gen_cfg = self.config.get("model", {}).get("generation", {})
        self.pipeline.flow_blend_end = float(gen_cfg.get("flow_blend_end", 1.0))
        logger.info(f"P-kare akış harmanlaması: adımların ilk %{self.pipeline.flow_blend_end * 100:.0f}'inde aktif.")

        if self.config.get("runtime", {}).get("enable_model_cpu_offload", False):
            self.pipeline.enable_model_cpu_offload()

        # DDIMInversion: referans görsel → latent dönüştürmek için
        self.inverter = DDIMInversion(
            pipe=self.pipeline,
            device=self.device,
        )

        # IP-Adapter FaceID: yüklenmeye çalış, başarısızsa sessizce devam et
        ip_cfg = self.config.get("ip_adapter", {})
        if ip_cfg.get("enabled", True):  # Default açık — insightface yüklendi mi kontrol eder
            local_path = ip_cfg.get("local_path", None)
            scale = ip_cfg.get("scale", 0.8)
            success = self.pipeline.load_ip_adapter(local_path=local_path, scale=scale)
            if success:
                logger.info("✅ IP-Adapter FaceID aktif. Yüz kimliği korumalı video üretimi hazır.")
            else:
                logger.warning("⚠️ IP-Adapter yüklenemedi. insightface yüklenmemiş olabilir.")
        else:
            logger.info("IP-Adapter config'de devre dışı bırakılmış.")

        # T2I-Adapter: geometri kilidi (ControlNet yerine hafif residual yolu)
        t2i_cfg = self.config.get("t2i_adapter", {})
        if t2i_cfg.get("enabled", True):
            success = self.pipeline.load_t2i_adapter(
                adapter_type=t2i_cfg.get("type", "canny"),
                local_path=t2i_cfg.get("local_path", None),
                scale=t2i_cfg.get("scale", 0.8),
                source=t2i_cfg.get("source", "hybrid"),
                canny_low=t2i_cfg.get("canny_low", 100),
                canny_high=t2i_cfg.get("canny_high", 200),
                dtype=self.dtype,
            )
            if success:
                logger.info("✅ T2I-Adapter aktif. Geometri kilidi (canny/depth/pose) hazır.")
            else:
                logger.warning("⚠️ T2I-Adapter yüklenemedi. Bu koşuda geometri kilidi kapalı.")
        else:
            logger.info("T2I-Adapter config'de devre dışı bırakılmış.")

        # ControlNet: her adımda çalışan geometri kilidi (depth/pose/canny, çoklu birim)
        cn_cfg = self.config.get("controlnet", {})
        if cn_cfg.get("enabled", False):
            success = self.pipeline.load_controlnet(config=cn_cfg, dtype=self.dtype)
            if success:
                logger.info("✅ ControlNet aktif. Adım-bazlı geometri kilidi hazır.")
            else:
                logger.warning("⚠️ ControlNet yüklenemedi. Bu koşuda ControlNet kapalı.")
        else:
            logger.info("ControlNet config'de devre dışı bırakılmış.")

        logger.info("✅ Models loaded successfully")

    def _text_encoders_to(self, device):
        """Metin kodlayıcılar (~1,6 GB) sadece prompt kodlanırken GPU'da dursun; sonra RAM'e."""
        self.text_encoder.to(device)
        self.text_encoder_2.to(device)
        if str(device) == "cpu":
            flush_vram()

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
        control_video_path: Optional[str] = None,  # ControlNet: hareket kaynağı (pose/depth dizisi)
        prompt_end: Optional[str] = None,  # Son karenin prompt'u: ilk → son arası SLERP ile geçiş
    ) -> str:
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        logger.info("🔤 Encoding prompt...")
        self._text_encoders_to(self.device)
        prompt_embeds, added_cond_kwargs = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
        )

        # Prompt geçişi: ilk kare `prompt`, son kare `prompt_end`, aradakiler SLERP (PromptBlender)
        if prompt_end and num_frames > 1:
            end_embeds, end_added = self._encode_prompt(
                prompt=prompt_end,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
            )
            prompt_embeds, added_cond_kwargs = PromptBlender.interpolate_embedding_schedule(
                keyframe_indices=[0, num_frames - 1],
                keyframe_prompt_embeds=[prompt_embeds, end_embeds],
                keyframe_added_cond=[added_cond_kwargs, end_added],
                num_frames=num_frames,
            )
            logger.info(f"🔀 Prompt geçişi aktif: {num_frames} kare boyunca → '{prompt_end}'")
            if self.pipeline.flow_blend_end >= 1.0:
                logger.warning(
                    "⚠️ flow_blend_end=1.0: P-kareler önceki kareyi neredeyse kopyalar, prompt geçişi "
                    "görünmeyebilir. configs/sdxl_video.yaml → flow_blend_end: 0.6 deneyin."
                )

        # Kodlama bitti: metin kodlayıcıları GPU'dan çıkar (UNet'e ~1,6 GB VRAM açılır)
        self._text_encoders_to("cpu")
        logger.info("Metin kodlayıcılar RAM'e taşındı (VRAM boşaltıldı).")

        # --------------------------------------------------------
        # --------------------------------------------------------
        # I2V: Referans fotoğraf varsa DDIM inversion + IP-Adapter face embeds
        # --------------------------------------------------------
        initial_latent = None
        start_timestep = None
        target_driving_frames = None
        ip_adapter_image_embeds = None
        t2i_residuals = None
        ref_img = None

        if reference_image_path and Path(reference_image_path).exists():
            import numpy as _np  # zaten import'lu ama explicit olsun
            logger.info(f"📷 I2V Modu: Referans fotoğraf yükleniyor → {reference_image_path}")
            ref_img = Image.open(reference_image_path).convert("RGB").resize((width, height))
            ref_np = np.array(ref_img).astype(np.float32) / 255.0

            # VAE encode için [-1, 1]
            ref_tensor_vae = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0)
            ref_tensor_vae = (ref_tensor_vae * 2 - 1).to(device=self.device, dtype=torch.float32)

            # Flow için [0, 1]
            ref_tensor_flow = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # DDIM Inversion
            logger.info("🔄 DDIM Inversion çalıştırılıyor (stop_at_t=420)...")
            initial_latent = self.inverter.invert(
                image=ref_tensor_vae,
                num_steps=num_inference_steps,
                stop_at_t=420,   # 600→420: ifade ve yapı daha iyi korunur
            )
            start_timestep = self.inverter.reached_t
            logger.info("✅ Referans görsel latent uzayına dönüştürüldü.")

            # IP-Adapter FaceID: Yüz embedding çıkar
            if (
                self.pipeline.ip_adapter is not None
                and self.pipeline.face_extractor is not None
                and self.pipeline.face_extractor.app is not None
            ):
                ref_img_uint8 = np.array(ref_img).astype("uint8")[:, :, ::-1].copy()   # RGB -> BGR
                face_embedding = self.pipeline.face_extractor.extract(ref_img_uint8)
                if face_embedding is not None:
                    ip_adapter_image_embeds = self.pipeline.ip_adapter.get_image_embeds(face_embedding)
                    logger.info("✅ IP-Adapter FaceID embedding hazır — kimlik ankrajı aktif.")
                else:
                    logger.warning("⚠️ Referans fotoğrafta yüz tespit edilemedi — IP-Adapter atlanıyor.")
            else:
                logger.info("IP-Adapter yüklenmemiş, face embedding atlanıyor.")

            # T2I-Adapter: referans görselden geometri koşulu (canny/depth/pose)
            if self.pipeline.t2i_adapter is not None:
                t2i_residuals = self.pipeline.t2i_adapter.encode_image(ref_img)
                if t2i_residuals is not None:
                    logger.info(
                        f"✅ T2I-Adapter residual hazır — geometri kilidi aktif "
                        f"({self.pipeline.t2i_adapter.condition_type})."
                    )
            else:
                logger.info("T2I-Adapter yüklenmemiş, geometri kilidi atlanıyor.")

            target_driving_frames = [ref_tensor_flow] * num_frames
        else:
            if reference_image_path:
                logger.warning(f"⚠️ Referans fotoğraf bulunamadı: {reference_image_path} — T2V modunda devam ediliyor.")
            logger.info("🎬 T2V Modu: Saf prompt'tan üretim yapılıyor.")

        # ControlNet: driving video (varsa) her kare için ayrı koşul haritası verir
        controlnet_driving_frames = None
        if control_video_path:
            if self.pipeline.controlnet is None:
                logger.warning("⚠️ --control-video verildi ama ControlNet yüklü değil (config: controlnet.enabled).")
            elif Path(control_video_path).exists():
                controlnet_driving_frames = load_driving_frames(control_video_path, num_frames, width, height)
            else:
                logger.warning(f"⚠️ Driving video bulunamadı: {control_video_path}")

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
            initial_latent=initial_latent,
            ip_adapter_image_embeds=ip_adapter_image_embeds,  # None ise devre dışı
            t2i_residuals=t2i_residuals,  # None ise T2I-Adapter frame 0'da kapalı; P-frame'de previous_frame/hybrid çalışır
            start_timestep=start_timestep,
            controlnet_reference_image=ref_img,
            controlnet_driving_frames=controlnet_driving_frames,
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