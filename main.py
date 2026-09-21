# main.py
import os
import sys
import argparse
import logging
import yaml

# Proje ana dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# inference.py'deki VideoGenerator'ı doğrudan kullan
from inference import VideoGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

CONFIG_DIR_NAME = "configs"

def load_config() -> dict:
    """YAML konfigürasyon dosyalarını okur ve birleştirerek tek bir dict döner."""
    config_dir = os.path.join(os.path.dirname(__file__), CONFIG_DIR_NAME)

    config = {}
    for fname in ["paths.yaml", "runtime.yaml", "sdxl_video.yaml"]:
        with open(os.path.join(config_dir, fname), "r") as f:
            part = yaml.safe_load(f)
            if part:
                config.update(part)
    return config

def main():
    parser = argparse.ArgumentParser(description="SDXL Video Generation Pipeline")
    parser.add_argument("-p", "--prompt", type=str,
                        default="A cinematic drone shot of a futuristic city at sunset, cyberpunk style, 8k",
                        help="Üretilecek video için metin (prompt)")
    args = parser.parse_args()

    logger.info("1. Konfigürasyon dosyaları okunuyor...")
    config = load_config()

    logger.info("2. VideoGenerator başlatılıyor (modeller yükleniyor)...")
    generator = VideoGenerator(config)

    output_dir = "outputs/videos"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("3. Video üretimi başlatılıyor (T2V)...")
    gen_cfg = config["model"]["generation"]
    output_path = generator.generate(
        prompt=args.prompt,
        negative_prompt=gen_cfg.get("negative_prompt", ""),
        num_frames=gen_cfg.get("num_frames", 24),
        width=gen_cfg.get("width", 1024),
        height=gen_cfg.get("height", 576),
        seed=config.get("runtime", {}).get("seed", None),
    )

    logger.info(f"✅ Video başarıyla kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
