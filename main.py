# main.py
import os
import sys
import argparse
import logging
import re
import yaml

# Proje ana dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# inference.py'deki VideoGenerator'ı doğrudan kullan
from inference import VideoGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

CONFIG_DIR_NAME = "configs"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_REF = re.compile(r"\$\{([^}]+)\}")


def _lookup(config: dict, dotted: str):
    node = config
    for key in dotted.split("."):
        node = node[key]
    return node


def _resolve_refs(node, config: dict):
    """YAML ${a.b} referanslarını çözer (yaml.safe_load bunu kendisi yapmaz)."""
    if isinstance(node, dict):
        return {k: _resolve_refs(v, config) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, config) for v in node]
    if isinstance(node, str):
        return _REF.sub(lambda m: str(_lookup(config, m.group(1))), node)
    return node


def _resolve_base_dir(config: dict):
    """base_dir göreliyse proje köküne göre çözülür; bu makinede yoksa proje kökü kullanılır."""
    paths = config.get("paths", {})
    base = paths.get("base_dir")
    if not base:
        paths["base_dir"] = PROJECT_ROOT
        return
    if not os.path.isabs(base):
        base = os.path.join(PROJECT_ROOT, base)
    if not os.path.isdir(base):
        logger.warning(f"paths.base_dir bu makinede yok ({base}) → proje kökü kullanılıyor: {PROJECT_ROOT}")
        base = PROJECT_ROOT
    paths["base_dir"] = os.path.normpath(base)


def load_config() -> dict:
    """YAML konfigürasyon dosyalarını okur ve birleştirerek tek bir dict döner."""
    config_dir = os.path.join(os.path.dirname(__file__), CONFIG_DIR_NAME)

    config = {}
    for fname in ["paths.yaml", "runtime.yaml", "sdxl_video.yaml"]:
        with open(os.path.join(config_dir, fname), "r") as f:
            part = yaml.safe_load(f)
            if part:
                config.update(part)
    _resolve_base_dir(config)
    return _resolve_refs(config, config)

def main():
    parser = argparse.ArgumentParser(description="SDXL Video Generation Pipeline")
    parser.add_argument("-p", "--prompt", type=str,
                        default="A cinematic drone shot of a futuristic city at sunset, cyberpunk style, 8k",
                        help="Üretilecek video için metin (prompt)")
    parser.add_argument("-r", "--ref", type=str, default=None,
                        help="[I2V Modu] Referans fotoğraf yolu. Verilirse flow anchor olarak kullanılır.")
    parser.add_argument("-c", "--control-video", type=str, default=None,
                        help="[ControlNet] Hareket kaynağı video (pose/depth haritaları her kare için buradan çıkarılır).")
    parser.add_argument("--prompt-end", type=str, default=None,
                        help="Son karenin prompt'u. Verilirse ilk kareden son kareye prompt yumuşakça geçer.")
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
        prompt_end=args.prompt_end,
    )

    logger.info(f"✅ Video başarıyla kaydedildi: {output_path}")


if __name__ == "__main__":
    main()