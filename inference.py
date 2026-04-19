import torch
import logging
import sys
import os
from typing import Optional, Dict, Any, List

# --- GEREKLİ IMPORTLAR ---
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
# Orijinal UNet (Geçici yükleyici olarak kullanacağız)
from diffusers import UNet2DConditionModel as OfficialUNet 
from diffusers.utils import logging as diffusers_logging

# Local modüller
from utils.model_utils import get_device, flush_vram
from pipelines.video_pipeline import SDXLVideoPipeline

# Senin Custom UNet'in (Hedef sınıf)
try:
    from models.unet.unet_base import UNet2DConditionModel as CustomUNet
except ImportError as e:
    print(f"HATA: models/unet/unet_base.py dosyası bulunamadı: {e}")
    sys.exit(1)

# Loglama Ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Gereksiz diffusers uyarılarını kapat
diffusers_logging.set_verbosity_error()

class VideoGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device(force_cpu=False)
        
        # Precision Ayarı (fp16 önerilir)
        precision = config['runtime'].get('precision', 'fp16')
        self.dtype = torch.float16 if precision == 'fp16' else torch.float32
        
        self.pipeline = None
        self._load_models()

    def _load_models(self):
        logger.info(f"Modeller yükleniyor... Cihaz: {self.device}, Tip: {self.dtype}")
        
        try:
            unet_path = self.config['paths']['models']['sdxl_base']
            vae_path = self.config['paths']['models']['sdxl_vae']

            # --- 1. ADIM: AĞIRLIKLARI ORİJİNAL SINIFLA AÇ ---
            if not os.path.exists(unet_path):
                raise FileNotFoundError(f"UNet modeli bulunamadı: {unet_path}")
                
            logger.info("1/3: Orijinal SDXL ağırlıkları okunuyor...")
            
            # Orijinal Diffusers sınıfını kullanarak yükle
            # (Bu işlem safetensors -> pytorch state_dict dönüşümünü yapar)
            temp_unet = OfficialUNet.from_single_file(
                unet_path, 
                torch_dtype=self.dtype
            )

            # --- 2. ADIM: CUSTOM MODELE TRANSFER ET ---
            logger.info("2/3: Ağırlıklar Custom UNet (unet_base.py) mimarisine aktarılıyor...")
            
            # Senin unet_base.py dosyanı, orijinal modelin konfigürasyonuyla başlat
            custom_unet = CustomUNet(**temp_unet.config)
            
            # Ağırlıkları (State Dict) aktar
            custom_unet.load_state_dict(temp_unet.state_dict())
            
            # Custom modeli doğru veri tipine çevir
            custom_unet.to(dtype=self.dtype)
            
            # RAM temizliği: Temp modeli sil
            del temp_unet
            flush_vram()

            # --- 3. ADIM: VAE, SCHEDULER VE PIPELINE ---
            logger.info(f"3/3: VAE yükleniyor: {vae_path}")
            if not os.path.exists(vae_path):
                 raise FileNotFoundError(f"VAE bulunamadı: {vae_path}")

            vae = AutoencoderKL.from_single_file(
                vae_path, 
                torch_dtype=self.dtype
            )

            # Scheduler'ı oluştur
            scheduler_config = self.config['model'].get('scheduler', {})
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)

            logger.info("Pipeline kuruluyor...")
            
            # --- DÜZELTME BURADA ---
            # Pipeline'a 'scheduler_config' değil, oluşturduğumuz 'scheduler' objesini veriyoruz.
            self.pipeline = SDXLVideoPipeline(
                vae=vae,
                unet=custom_unet,
                scheduler=scheduler 
            )
            
            self.pipeline.to(self.device)
            
            # CPU Offload (VRAM yetersizse runtime.yaml'dan açılabilir)
            if self.config['runtime'].get('enable_model_cpu_offload', False):
                self.pipeline.enable_model_cpu_offload()

            logger.info("✅ Tüm modeller başarıyla yüklendi.")

        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None

    def generate(self, prompt, negative_prompt="", num_frames=24, width=1024, height=576, seed=None):
        if self.pipeline is None:
            logger.error("Pipeline hazır değil. İşlem iptal edildi.")
            return []

        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f"🎬 Üretim Başlıyor: '{prompt}'")

        try:
            # pipelines/video_pipeline.py içindeki __call__ çalışır
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                width=width,
                height=height,
                num_inference_steps=self.config['model']['generation']['steps'],
                guidance_scale=self.config['model']['generation']['guidance_scale'],
                generator=generator
            )

            if self.config['runtime'].get('flush_vram_after_generation', True):
                flush_vram()
                
            return output.frames

        except Exception as e:
            logger.error(f"Generate hatası: {e}")
            import traceback
            traceback.print_exc()
            return []