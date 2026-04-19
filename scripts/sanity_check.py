import os
import sys

# 0. Python Path Düzeltmesi: Kök dizini (sdxl-video-generator) sisteme tanıtıyoruz.
# Bu işlemin Diffusers veya bizim modüllerimizin import edilmesinden ÖNCE yapılması zorunludur.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import logging
from PIL import Image
import numpy as np

# Diffusers kütüphanesi
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# Bizim modüler mimarimizden importlar (Artık sorunsuz bulunacak)
from pipelines.video_pipeline import SDXLVideoPipeline
from pipelines.postprocess import VideoPostProcessor
# ... geri kalan kodun tamamı ...
from modules.flow.flow_factory import build_flow
from modules.losses.loss_functions import build_loss
from modules.inversion.ddim_inversion import DDIMInversion
from modules.refinement.latent_refiner import LatentRefiner

# Logging Ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dummy_image(width=512, height=512):
    """Test için geçici bir referans görseli üretir."""
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

def test_end_to_end_pipeline():
    logger.info("SDXL Video Generator Uçtan Uca (E2E) Testi Başlıyor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Klasör Yapısının Kontrolü
    os.makedirs("outputs/videos", exist_ok=True)
    
    logger.info("1. Modüller Başlatılıyor...")
    try:
        # Gerçek üretim standartlarında modülleri ayağa kaldırıyoruz
        flow_model = build_flow(name="raft", device=device)
        loss_module = build_loss(device=device)
        refiner = LatentRefiner(total_loss_fn=loss_module, steps=2, lr=0.01)
        logger.info("Modüller başarıyla yüklendi (Flow, Loss, Refiner).")
    except Exception as e:
        logger.error(f"Modül başlatma hatası: {e}")
        return

    logger.info("2. SDXL Temel Ağırlıkları Yükleniyor (Düşük VRAM Modu)...")
    try:
        # Test için base modeli yüklüyoruz. (Production'da local path verilebilir)
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
        unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
        scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        
        # Kendi yazdığımız orkestratör pipeline'ı ayağa kaldırıyoruz [cite: 460, 538]
        pipeline = SDXLVideoPipeline(vae=vae, unet=unet, scheduler=scheduler)
        pipeline.to(device)
        
        inverter = DDIMInversion(pipe=pipeline, device=device)
        logger.info("Pipeline ve Inversion modülü ayağa kalktı.")
    except Exception as e:
        logger.error(f"Model yükleme hatası (HuggingFace token veya internet sorunu olabilir): {e}")
        return

    logger.info("3. Test Verileri Hazırlanıyor (Dummy Embeddings)...")
    test_prompt = "A futuristic cyberpunk city, neon lights, 4k resolution"
    dummy_ref_image = create_dummy_image(width=512, height=512)
    
    # SDXL Text Encoder çıktılarını taklit eden Dummy Tensörler (CFG için batch=2: Uncond + Cond)
    # SDXL prompt_embeds boyutu genellikle (batch, 77, 2048) şeklindedir.
    # pooled_prompt_embeds boyutu ise (batch, 1280) şeklindedir.
    prompt_embeds = torch.randn((2, 77, 2048), device=device, dtype=unet.dtype)
    pooled_prompt_embeds = torch.randn((2, 1280), device=device, dtype=unet.dtype)
    
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds, 
        "time_ids": torch.zeros((2, 6), device=device, dtype=unet.dtype)
    }

    logger.info("4. Video Üretim Döngüsü (Pipeline Inference) Başlıyor...")
    # Testi hızlı bitirmek için: sadece 3 frame, 10 diffusion step ve düşük çözünürlük (512x512)
    try:
        output = pipeline(
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            ref_image=dummy_ref_image, # Inversion için
            num_frames=3,
            width=512,
            height=512,
            num_inference_steps=10, 
            guidance_scale=7.5,
            flow_model=flow_model,
            loss_module=loss_module,
            refiner=refiner,
            inverter=inverter
        )
        frames = output.frames
        logger.info(f"Video üretimi tamamlandı. Üretilen kare sayısı: {len(frames)}")
    except Exception as e:
        logger.error(f"Üretim döngüsü sırasında çökme yaşandı: {e}")
        raise e # Stack trace'i görmek için hatayı fırlatıyoruz

    logger.info("5. Post-Process (Kodlama ve Export) İşlemi...")
    try:
        postprocessor = VideoPostProcessor(output_dir="outputs/videos", base_fps=8)
        
        # Senin harika mimarindeki __call__ metodunu doğrudan tetikliyoruz
        result = postprocessor(
            frames=frames, 
            filename="test_output.mp4", 
            apply_interpolation=False, # Test olduğu için RIFE'ı şimdilik atlıyoruz
            apply_upscale=False
        )
        
        if os.path.exists(result["video_path"]):
            logger.info(f"✅ UÇTAN UCA TEST BAŞARILI! Video şuraya kaydedildi: {result['video_path']}")
        else:
            logger.error("Test başarısız: MP4 dosyası diske yazılamadı.")
    except Exception as e:
        logger.error(f"Post-process sırasında hata: {e}")
        
if __name__ == "__main__":
    # Optimizasyonlar: PyTorch bellek yönetimini iyileştir
    torch.backends.cudnn.benchmark = True
    test_end_to_end_pipeline()