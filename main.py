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
    parser.add_argument("-r", "--ref", type=str, default=None,
                        help="[I2V Modu] Referans fotoğraf yolu. Verilirse flow anchor olarak kullanılır.")
    parser.add_argument("-c", "--control-video", type=str, default=None,
                        help="[ControlNet] Hareket kaynağı video (pose/depth haritaları her kare için buradan çıkarılır).")
    parser.add_argument("--fast", action="store_true",
                        help="Hızlı test modu: 4 kare, 5 adım, 512x512 — yaklaşık 2-3 dakika")
    parser.add_argument("--steps", type=int, default=None,
                        help="Denoising adım sayısı (config'i override eder)")
    args = parser.parse_args()

    logger.info("1. Konfigürasyon dosyaları okunuyor...")
    config = load_config()
    gen_cfg = config["model"]["generation"]

    # --fast modu parametrelerini override eder
    if args.fast:
        logger.info("⚡ HIZLI TEST MODU aktif: 4 kare, 5 adım, 512×512")
        num_frames = 4
        steps = 5
        width, height = 512, 512
    else:
        num_frames = gen_cfg.get("num_frames", 24)
        steps = args.steps or gen_cfg.get("steps", 30)
        width  = gen_cfg.get("width", 1024)
        height = gen_cfg.get("height", 576)

    logger.info("2. VideoGenerator başlatılıyor (modeller yükleniyor)...")
    generator = VideoGenerator(config)

    output_dir = "outputs/videos"
    os.makedirs(output_dir, exist_ok=True)

    mode = "I2V" if args.ref else "T2V"
    logger.info(f"3. Video üretimi başlatılıyor ({mode} modu, {num_frames} kare, {steps} adım)...")
    output_path = generator.generate(
        prompt=args.prompt,
        negative_prompt=gen_cfg.get("negative_prompt", ""),
        num_frames=num_frames,
        width=width,
        height=height,
        num_inference_steps=steps,
        seed=config.get("runtime", {}).get("seed", None),
        reference_image_path=args.ref,
        control_video_path=args.control_video,
    )

    logger.info(f"✅ Video başarıyla kaydedildi: {output_path}")


if __name__ == "__main__":
    main()

