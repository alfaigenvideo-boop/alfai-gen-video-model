import os
import sys
import torch
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Diffusers ve SDXL Text Encoder'ları
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

# Kendi modüllerimiz
from pipelines.video_pipeline import SDXLVideoPipeline
from pipelines.postprocess import VideoPostProcessor
from modules.dataset import VideoDatasetLoader  # Yeni Dataset sınıfımız
from modules.flow.flow_factory import build_flow
from modules.losses.loss_functions import build_loss
from modules.inversion.ddim_inversion import DDIMInversion
from modules.refinement.latent_refiner import LatentRefiner
from modules.controlnet_wrapper import ControlNetWrapper
from modules.ip_adapter_wrapper import IPAdapterWrapper
from modules.adaptive_control.adaptive_scheduler import AdaptiveScheduler
from modules.adaptive_control.temporal_metrics import FaceAnalyzer, CLIPScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. AYARLAR (CONFIG) ---
    NUM_FRAMES = 50
    RESOLUTION = (1024, 1024)
    # Yeni ve spesifik promptumuz:
    PROMPT = "Dynamic low-angle close-up, blonde woman dancing on stage, head tilted back, wild sweeping hair, glowing orange-golden stage lights from below, sharp highlights on face and neck, dark form-fitting top with straps, parted lips, high energy cyberpunk club background, cinematic lighting, 8k resolution, highly detailed."
    
    os.makedirs("outputs/videos", exist_ok=True)
    
    # --- 2. VERİ YÜKLEME ---
    data_dir = os.path.join(root_dir, "data", "inputs")
    dataset = VideoDatasetLoader(data_dir=data_dir, device=device, resolution=RESOLUTION)
    identity_image, pose_list, depth_list = dataset.load_data(num_frames=NUM_FRAMES)

    # --- 3. MODEL YÜKLEME (TEXT ENCODER'LAR EKLENDİ) ---
    logger.info("SDXL Ağırlıkları ve Text Encoder'lar Yükleniyor...")
    base_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    tokenizer = CLIPTokenizer.from_pretrained(base_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_id, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_id, subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)
    
    vae = AutoencoderKL.from_pretrained(base_id, subfolder="vae", torch_dtype=torch.float32).to(device)
    unet = UNet2DConditionModel.from_pretrained(base_id, subfolder="unet", torch_dtype=torch.float16).to(device)
    scheduler_diff = DDIMScheduler.from_pretrained(base_id, subfolder="scheduler")

    controlnet = ControlNetWrapper(device=device, pose_model_path="xinsir/controlnet-openpose-sdxl-1.0", depth_model_path="diffusers/controlnet-depth-sdxl-1.0")
    ip_adapter = IPAdapterWrapper(unet=unet, device=device)
    loss_module = build_loss(device=device)

    # --- 4. PİPELİNE KURULUMU ---
    
    pipeline = SDXLVideoPipeline(
        vae=vae, unet=unet, scheduler=scheduler_diff,
        adaptive_scheduler=AdaptiveScheduler(),
        controlnet=controlnet, ip_adapter=ip_adapter,
        face_analyzer=FaceAnalyzer(), clip_scorer=CLIPScorer(device=device),
        loss_module=loss_module
    )
    # Metin kodlayıcıları pipeline'a enjekte ediyoruz
    pipeline.tokenizer = tokenizer
    pipeline.text_encoder = text_encoder
    pipeline.tokenizer_2 = tokenizer_2
    pipeline.text_encoder_2 = text_encoder_2
    pipeline.to(device)

    refiner = LatentRefiner(total_loss_fn=loss_module, steps=2, lr=0.01)
    inverter = DDIMInversion(pipe=pipeline, device=device)

    # --- 5. METNİ TENSÖRE ÇEVİRME (Gerçek Prompt Kullanımı) ---
    logger.info("Prompt işleniyor...")
    prompt_embeds, pooled_prompt_embeds = pipeline.encode_prompt(PROMPT, device)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": torch.zeros((2, 6), device=device, dtype=torch.float16)}

    # --- 6. ÜRETİM ---
    logger.info("Video Üretimi Başlıyor...")
    output = pipeline(
        prompt=PROMPT,
        prompt_embeds=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        identity_image=identity_image,
        pose_tensors=pose_list,
        depth_tensors=depth_list,
        num_frames=NUM_FRAMES,
        width=RESOLUTION[0], height=RESOLUTION[1],
        num_inference_steps=25,
        guidance_scale=7.5,
        refiner=refiner,
        inverter=inverter
    )

    # --- 7. POST PROCESS ---
    logger.info("Video Kodlanıyor...")
    postprocessor = VideoPostProcessor(output_dir="outputs/videos", base_fps=12)
    result = postprocessor(frames=output.frames, filename="dancing_woman_01.mp4")
    logger.info(f"✅ İŞLEM TAMAM! Video: {result['video_path']}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    run_inference()
