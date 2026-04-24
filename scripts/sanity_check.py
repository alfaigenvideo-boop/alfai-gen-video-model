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
from modules.controlnet_wrapper import ControlNetWrapper
from modules.ip_adapter_wrapper import IPAdapterWrapper
from modules.adaptive_control.adaptive_scheduler import AdaptiveScheduler
from modules.adaptive_control.temporal_metrics import FaceAnalyzer, CLIPScorer, TemporalMetrics

# Logging Ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_tensors_from_folder(folder_path, device, num_frames=50, size=(1024, 1024)):
    tensor_list = []
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:num_frames]
    
    for f in files:
        img = Image.open(os.path.join(folder_path, f)).convert("RGB").resize(size)
        arr = np.array(img).transpose(2, 0, 1) # H,W,C -> C,H,W
        tensor = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
        tensor_list.append(tensor.to(device).half())
        
    # Eğer klasörde 50'den az resim varsa, son resmi kopyalayarak listeyi tamamla
    while len(tensor_list) < num_frames:
        tensor_list.append(tensor_list[-1])
        
    return tensor_list

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
        adaptive_scheduler = AdaptiveScheduler()
        face_analyzer = FaceAnalyzer()
        clip_scorer = CLIPScorer(device=device)
        logger.info("Modüller başarıyla yüklendi (Flow, Loss, Refiner).")
    except Exception as e:
        logger.error(f"Modül başlatma hatası: {e}")
        return

    logger.info("2. SDXL Temel Ağırlıkları Yükleniyor...")
    try:
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # ÖNCE Modelleri yükle
        vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
        unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
        scheduler_diff = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        
        # MODELLER ARTIK YÜKLÜ, ŞİMDİ Wrapper'ları başlat
        # Repository linkini güncel ve çalışan bir versiyonla değiştirdim
        controlnet = ControlNetWrapper(
            device=device, 
            pose_model_path="xinsir/controlnet-openpose-sdxl-1.0",
            depth_model_path="diffusers/controlnet-depth-sdxl-1.0"
        )
        ip_adapter = IPAdapterWrapper(unet=unet, device=device)
        
        # EN SON Pipeline'ı kur
        pipeline = SDXLVideoPipeline(
            vae=vae, unet=unet, scheduler=scheduler_diff,
            adaptive_scheduler=adaptive_scheduler,
            controlnet=controlnet,
            ip_adapter=ip_adapter,
            face_analyzer=face_analyzer,
            clip_scorer=clip_scorer,
            loss_module=loss_module
        )
        pipeline.to(device)
        
        inverter = DDIMInversion(pipe=pipeline, device=device)
        logger.info("✅ Pipeline ve tüm wrapper'lar başarıyla bağlandı.")
        
    except Exception as e:
        logger.error(f"Kritik Model Yükleme Hatası: {e}")
        raise e # Hatayı burada görmek sorunu anında çözer

    logger.info("3. Test Verileri Hazırlanıyor (Dummy Embeddings)...")
    test_prompt = "Dynamic low-angle close-up, blonde woman dancing on stage, head tilted back, wild sweeping hair, glowing orange-golden stage lights from below, sharp highlights on face and neck, dark form-fitting top with straps, parted lips, high energy cyberpunk club background, cinematic lighting, 8k resolution, highly detailed."

    num_frames = 50
    base_data_dir = os.path.join(root_dir, "data", "inputs")
    
    # Referans Yüz (Klasördeki ilk kareyi kullanıyoruz)
    identity_image = Image.open(os.path.join(base_data_dir, "frames", "0001.jpg")).convert("RGB")
    
    # Hareket ve Derinlik tensör listeleri
    pose_list = load_tensors_from_folder(os.path.join(base_data_dir, "poses"), device, num_frames)
    depth_list = load_tensors_from_folder(os.path.join(base_data_dir, "depths"), device, num_frames)


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
            prompt=test_prompt,
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            identity_image=identity_image, 
            pose_tensors=pose_list,        
            depth_tensors=depth_list,      
            ref_image=identity_image,      
            num_frames=num_frames,
            width=1024,
            height=1024,
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
        postprocessor = VideoPostProcessor(output_dir="outputs/videos", base_fps=12)
        
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