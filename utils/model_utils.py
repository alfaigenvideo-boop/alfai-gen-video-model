# (Cihaz yönetimi, VRAM temizleme)

import torch
import gc
import logging

logger = logging.getLogger(__name__)

def get_device(force_cpu: bool = False) -> str:
    """
    En uygun cihazı seçer (CUDA > MPS (Mac) > CPU).
    Config dosyasındaki ayara göre CPU zorlanabilir.
    """
    if force_cpu:
        return "cpu"
        
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) desteği
        return "mps"
    else:
        return "cpu"

def flush_vram():
    """
    GPU belleğini (VRAM) temizler ve Garbage Collector'ı çalıştırır.
    Video üretiminde OOM (Out of Memory) hatası almamak için her işlemden sonra çağrılmalıdır.
    """
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        logger.debug("VRAM temizlendi.")
    except Exception as e:
        logger.warning(f"VRAM temizlenirken hata oluştu: {e}")

def print_gpu_memory():
    """
    Mevcut GPU bellek kullanımını loglar (Sadece CUDA için).
    """
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU Memory: {allocated:.2f}GB / {reserved:.2f}GB (Total: {total:.2f}GB)")