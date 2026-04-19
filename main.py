# main.py
import os
import sys
import argparse
import logging
from omegaconf import OmegaConf

# Proje ana dizinini Python yoluna ekle (Aksi takdirde modüller bulunamayabilir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Yeni Mimari Importlarımız
from core.schemas import ModelPaths, RuntimeConfig, GenerationConfig
from utils.hardware_utils import CUDADeviceManager
from pipelines.pipeline_builder import SDXLVideoPipelineBuilder
from services.video_generator import VideoGenerator
from utils.video_utils import save_video_frames # Senin mevcut video kaydedicin

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

CONFIG_DIR_NAME = "configs"
PATHS_YAML = "paths.yaml"
RUNTIME_YAML = "runtime.yaml"
SDXL_YAML = "sdxl_video.yaml"

def load_and_parse_configs() -> tuple[ModelPaths, RuntimeConfig, GenerationConfig]:
    """
    YAML dosyalarını okur, OmegaConf ile birleştirir (Interpolation yapar)
    ve tip güvenli DTO'lara (Data Classes) dönüştürür.
    """
    config_dir = os.path.join(os.path.dirname(__file__), CONFIG_DIR_NAME)
    
    # OmegaConf ile dosyaları oku
    paths_conf = OmegaConf.load(os.path.join(config_dir, PATHS_YAML))
    runtime_conf = OmegaConf.load(os.path.join(config_dir, RUNTIME_YAML))
    sdxl_conf = OmegaConf.load(os.path.join(config_dir, SDXL_YAML))

    # Tüm konfigürasyonları birleştir. (Yer tutucular otomatik çözülür)
    cfg = OmegaConf.merge(paths_conf, runtime_conf, sdxl_conf)

    # 1. ModelPaths DTO'sunu oluştur (Arkadaşının input_video mantığını ekledik)
    paths = ModelPaths(
        unet_base=cfg.paths.models.sdxl_base,
        vae=cfg.paths.models.sdxl_vae,
        output_videos=cfg.paths.output.videos,
        # YAML'da tanımlanmışsa al, yoksa None dön (Opsiyonel girdi)
        input_video=cfg.paths.get('input', {}).get('video', None) 
    )

    # 2. RuntimeConfig DTO'sunu oluştur
    runtime = RuntimeConfig(
        precision=cfg.runtime.get('precision', 'fp16'),
        enable_model_cpu_offload=cfg.runtime.get('enable_model_cpu_offload', False),
        flush_vram_after_generation=cfg.runtime.get('flush_vram_after_generation', True),
        seed=cfg.runtime.get('seed', None)
    )

    # 3. GenerationConfig DTO'sunu oluştur
    gen_cfg_data = cfg.model.generation
    gen_config = GenerationConfig(
        steps=gen_cfg_data.get('steps', 50),
        guidance_scale=gen_cfg_data.get('guidance_scale', 7.5),
        width=gen_cfg_data.get('width', 1024),
        height=gen_cfg_data.get('height', 576),
        num_frames=gen_cfg_data.get('num_frames', 24),
        fps=gen_cfg_data.get('fps', 8),
        negative_prompt=gen_cfg_data.get('negative_prompt', ""),
        scheduler_config=OmegaConf.to_container(cfg.model.get('scheduler', {}))
    )

    return paths, runtime, gen_config

def main():
    # Argümanları al (Böylece terminalden prompt girilebilir)
    parser = argparse.ArgumentParser(description="SDXL Video Generation Pipeline")
    parser.add_argument("-p", "--prompt", type=str, 
                        default="A cinematic drone shot of a futuristic city at sunset, cyberpunk style, 8k",
                        help="Üretilecek video için metin (prompt)")
    args = parser.parse_args()

    logger.info("1. Konfigürasyon dosyaları okunuyor...")
    paths, runtime, gen_config = load_and_parse_configs()

    if args.input:
        paths.input_video = args.input

    logger.info("2. Sistem bileşenleri başlatılıyor...")
    device_manager = CUDADeviceManager()
    
    # Pipeline Builder'a konfigürasyonları verip Pipeline'ı üretiyoruz
    builder = SDXLVideoPipelineBuilder(paths, runtime, gen_config, device_manager)
    pipeline = builder.build()

    logger.info("3. Video Generator hazırlanıyor...")
    generator = VideoGenerator(pipeline, gen_config, device_manager)
    
    os.makedirs(paths.output_videos, exist_ok=True)

    if paths.input_video:
        logger.info(f"4. Girdi videosu ile üretim başlatılıyor (V2V): {paths.input_video}")
        frames = generator.generate_video_from_frames(
            prompt=args.prompt,
            input_path=paths.input_video,
            output_dir=paths.output_videos, 
            negative_prompt=gen_config.negative_prompt,
            seed=runtime.seed
        )
    else:
        logger.info("4. Sadece metin ile üretim başlatılıyor (T2V)...")
        frames = generator.generate(
            prompt=args.prompt,
            negative_prompt=gen_config.negative_prompt,
            num_frames=gen_config.num_frames,
            width=gen_config.width,
            height=gen_config.height,
            seed=runtime.seed
        )
    
    # Çıktı Kayıt İşlemleri (Eğer generate_video_from_frames doğrudan frame dönüyorsa)
    if frames:
        safe_name = "".join([c if c.isalnum() else "_" for c in args.prompt[:20]])
        output_path = os.path.join(paths.output_videos, f"{safe_name}.mp4")
        
        save_video_frames(frames, output_path, fps=gen_config.fps)
        logger.info(f"✅ Video başarıyla kaydedildi: {output_path}")
    else:
        # Arkadaşının metodu kendi içinde kaydediyor ve None dönüyorsa bu log işe yarar
        logger.info(f"✅ İşlem tamamlandı. Çıktılar '{paths.output_videos}' klasöründe.")

if __name__ == "__main__":
    main()
