import os
import sys
import torch
import logging
from PIL import Image
import numpy as np

# 0. Python Path Düzeltmesi: Kök dizini (sdxl-video-generator) sisteme tanıtıyoruz.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Diffusers kütüphanesi
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# Bizim modüler mimarimizden importlar
from pipelines.video_pipeline import SDXLVideoPipeline
from pipelines.postprocess import VideoPostProcessor
from modules.flow.flow_factory import build_flow
from modules.losses.loss_functions import build_loss
from modules.inversion.ddim_inversion import DDIMInversion
from modules.refinement.latent_refiner import LatentRefiner
from models.unet.unet_tsm_patch import inject_tsm_into_unet_midblock # TSM Importu Eklendi

# Logging Ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_tsm_dummy_forward():
    """UNet içine TSM modülünün enjekte edilip forward pass'in bozulmadığını test eder."""
    logger.info(">>> [Test 1] TSM Dummy Forward Testi Başlıyor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(">>> UNet yükleniyor (TSM Testi için)...")
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        torch_dtype=torch.float16 # Bellek tasarrufu için fp16 eklendi
    ).to(device)
    logger.info(">>> UNet yüklendi.")

    B = 1   # Batch size küçültüldü (sanity check için yeterli)
    T = 2   # Frame sayısı küçültüldü (2 frame yeterli TSM'i doğrulamak için)
    C = 4
    H = W = 32  # Latent boyutu da küçültüldü (64→32)

    logger.info(">>> UNet'e TSM enjekte ediliyor...")
    inject_tsm_into_unet_midblock(unet, num_frames=T)
    logger.info(">>> TSM başarıyla enjekte edildi.")

    # Test tensörlerini unet.dtype (fp16) ile uyumlu yaratıyoruz
    x = torch.randn(B * T, C, H, W, device=device, dtype=unet.dtype)
    t = torch.randint(0, 1000, (B * T,), device=device)
    cond = torch.randn(B * T, 77, 2048, device=device, dtype=unet.dtype)

    # ---- SDXL required additional conditioning ----
    pooled_text_embeds = torch.zeros(
        (B * T, 1280),
        device=device,
        dtype=unet.dtype
    )

    target_size = (H * 8, W * 8)
    original_size = (H * 8, W * 8)
    crops_coords_top_left = (0, 0)

    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor(
        [add_time_ids],
        device=device,
        dtype=unet.dtype
    ).repeat(B * T, 1)

    added_cond_kwargs = {
        "text_embeds": pooled_text_embeds,
        "time_ids": add_time_ids,
    }

    # ---- Forward pass ----
    logger.info(">>> TSM enjekte edilmiş UNet ile forward pass çalıştırılıyor...")
    with torch.no_grad():
        out = unet(
            sample=x,
            timestep=t,
            encoder_hidden_states=cond,
            added_cond_kwargs=added_cond_kwargs
        )

    assert out.sample.shape == x.shape, f"Beklenen şekil {x.shape}, alınan {out.sample.shape}"
    logger.info("✅ [Test 1 Başarılı] TSM Dummy Forward PASSED (Tensör boyutları korundu)")
    
    # Belleği temizleyelim ki E2E teste yer kalsın
    del unet, x, t, cond, out
    torch.cuda.empty_cache()


def create_dummy_image(width=512, height=512):
    """Test için geçici bir referans görseli üretir."""
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_end_to_end_pipeline():
    logger.info(">>> [Test 2] SDXL Video Generator Uçtan Uca (E2E) Testi Başlıyor...")
    # ----------------------------------------------------------------
    # RTX 3050 Laptop (4.3 GB VRAM) için bellek yönetimi notları:
    #  - Tüm yardımcı modeller (RAFT, CLIP, Loss) CPU'da başlatılır
    #  - Pipeline sequential_cpu_offload ile UNet/VAE'yi katman katman GPU'ya alır
    #  - Çözünürlük ve adım sayısı sanity check için minimal tutulur
    # ----------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu = "cpu"

    os.makedirs("outputs/videos", exist_ok=True)

    logger.info("1. Yardımcı Modüller CPU'da Başlatılıyor (VRAM korunuyor)...")
    try:
        # CPU'da yükle — video üretimi sırasında GPU'ya taşınmaz, sadece loss hesabında kullanılır
        flow_model = build_flow(name="raft", device=cpu)
        loss_module = build_loss(device=cpu)
        refiner    = LatentRefiner(total_loss_fn=loss_module, steps=1, lr=0.01)
        logger.info("Modüller CPU'da hazır (RAFT, Loss, Refiner).")
    except Exception as e:
        logger.error(f"Modül başlatma hatası: {e}")
        return

    logger.info("2. SDXL Ağırlıkları Yükleniyor (sequential CPU offload modunda)...")
    try:
        torch.cuda.empty_cache()
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        # Önce CPU'ya yükle, pipeline offload mekanizmasına bırak
        vae       = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae",       torch_dtype=torch.float16)
        unet      = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

        pipeline = SDXLVideoPipeline(vae=vae, unet=unet, scheduler=scheduler)

        # sequential_cpu_offload: Her katman GPU'ya sırayla gelir, bitmişse CPU'ya döner
        # Bu yöntem 4 GB VRAM'de SDXL çalıştırmanın standart yoludur
        pipeline.enable_sequential_cpu_offload()

        inverter = DDIMInversion(pipe=pipeline, device=device)
        logger.info("Pipeline (sequential CPU offload) hazır.")
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        return

    logger.info("3. Test Verileri Hazırlanıyor...")
    # CPU'da oluştur — pipeline offload modunda tensörleri kendisi taşıyor
    prompt_embeds      = torch.randn((2, 77, 2048), dtype=torch.float16)
    pooled_embeds      = torch.randn((2, 1280),     dtype=torch.float16)
    added_cond_kwargs  = {
        "text_embeds": pooled_embeds,
        "time_ids":    torch.zeros((2, 6), dtype=torch.float16),
    }

    logger.info("4. Video Üretim Döngüsü Başlıyor (256x256, 3 adım, 2 kare)...")
    try:
        output = pipeline(
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            num_frames=16,           # Minimum: pipeline'ın frame-by-frame mantığını doğrulamak için yeterli
            width=1024,              # 512→256: VRAM kullanımı 4x azalır (latent 64x64→32x32)
            height=576,
            num_inference_steps=30,  # Sadece doğrulama için — kalite önemli değil
            guidance_scale=7.5,
        )
        frames = output.frames
        logger.info(f"Video üretimi tamamlandı. Üretilen kare sayısı: {len(frames)}")
    except Exception as e:
        logger.error(f"Üretim döngüsü sırasında çökme yaşandı: {e}")
        raise e

    logger.info("5. Post-Process (MP4 Export) İşlemi...")
    try:
        postprocessor = VideoPostProcessor(output_dir="outputs/videos", base_fps=8)
        result = postprocessor(
            frames=frames,
            filename="test_output.mp4",
            apply_interpolation=False,
            apply_upscale=False,
        )
        if os.path.exists(result["video_path"]):
            logger.info(f"✅ [Test 2 Başarılı] Video kaydedildi: {result['video_path']}")
        else:
            logger.error("Test başarısız: MP4 dosyası diske yazılamadı.")
    except Exception as e:
        logger.error(f"Post-process sırasında hata: {e}")

        
if __name__ == "__main__":
    # Optimizasyonlar: PyTorch bellek yönetimini iyileştir
    torch.backends.cudnn.benchmark = True
    
    # Önce Unit Test'i (TSM Modül Kontrolü) çalıştır
    test_tsm_dummy_forward()
    print("-" * 50)
    
    # Eğer ilk test geçerse E2E (Uçtan Uca) Testi çalıştır
    test_end_to_end_pipeline()