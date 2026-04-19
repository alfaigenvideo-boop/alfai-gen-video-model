# (Video save/load, gif export)
import os
import logging
import numpy as np
from typing import List, Union
from PIL import Image

# Loglama ayarı
logger = logging.getLogger(__name__)

def save_video_frames(frames: List[Image.Image], output_path: str, fps: int = 8):
    """
    PIL Image listesini alır ve belirtilen yola MP4 videosu olarak kaydeder.
    
    Args:
        frames (List[PIL.Image]): Üretilen resim kareleri listesi.
        output_path (str): Videonun kaydedileceği dosya yolu (örn: outputs/videos/test.mp4).
        fps (int): Saniyedeki kare sayısı.
    """
    
    # 1. Eğer liste boşsa işlem yapma (Dummy inference döndüğü için bu kontrol önemli)
    if not frames:
        logger.warning("Kaydedilecek frame bulunamadı (Liste boş). Video oluşturulmadı.")
        return

    # 2. Klasör yoksa oluştur
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        import imageio
        
        logger.info(f"Video kaydediliyor: {output_path} ({len(frames)} frames, {fps} fps)")
        
        # imageio ile yazıcıyı başlat
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='FFMPEG')
        
        for frame in frames:
            # PIL Image -> Numpy Array dönüşümü
            frame_np = np.array(frame)
            writer.append_data(frame_np)
            
        writer.close()
        logger.info("Video başarıyla kaydedildi.")

    except ImportError:
        logger.error("Video kaydetmek için 'imageio' ve 'imageio-ffmpeg' kütüphaneleri gerekli.")
        logger.error("Lütfen şu komutu çalıştırın: pip install imageio[ffmpeg]")
    except Exception as e:
        logger.error(f"Video kaydetme sırasında hata oluştu: {e}")

def export_to_gif(frames: List[Image.Image], output_path: str, duration: int = 100):
    """
    Alternatif olarak GIF kaydetmek istersek.
    """
    if not frames:
        return
        
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration,
        loop=0
    )
