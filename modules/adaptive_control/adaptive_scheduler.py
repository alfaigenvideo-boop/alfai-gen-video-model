from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

# Aynı klasördeki PID modülünü çağırıyoruz
from .pid_controller import PIDConfig, PIDController

@dataclass
class SchedulerTargets:
    """Sensörlerin ulaşmaya çalışacağı ideal hedef değerler."""
    identity: float = 0.88   # Kimlik min %88 benzemeli
    residual: float = 0.08   # Titreme %8'in altında olmalı
    clip: float = 0.30       # Metinle uyum skoru

class AdaptiveScheduler:
    """
    Sensör verilerini okuyarak ControlNet, IP-Adapter ve Latent Inpaint Maskesi 
    için dinamik katsayılar üreten zamanlayıcı beyin.
    """
    def __init__(self, targets: SchedulerTargets = None, smoothing_factor: float = 0.6):
        self.targets = targets or SchedulerTargets()
        self.alpha = smoothing_factor
        
        # IP-Adapter (Yüz/Kimlik) için Kontrolcü
        self.pid_identity = PIDController(
            PIDConfig(kp=0.8, ki=0.05, kd=0.1, output_limit=(0.3, 1.0)),
            initial_value=0.7
        )
        # ControlNet Pose/Hareket için Kontrolcü
        self.pid_residual = PIDController(
            PIDConfig(kp=1.2, ki=0.10, kd=0.3, output_limit=(0.5, 2.0)),
            initial_value=1.0
        )
        # ControlNet Depth/Derinlik için Kontrolcü
        self.pid_clip = PIDController(
            PIDConfig(kp=0.8, ki=0.05, kd=0.2, output_limit=(0.5, 2.0)),
            initial_value=1.0
        )
        
        self.smoothed_identity = 0.0
        self.smoothed_residual = 0.0
        self.smoothed_clip = 0.0

    def _smooth_metrics(self, identity: float, residual: float, clip: float) -> Tuple[float, float, float]:
        """Verilerdeki anlık sıçramaları (spike) yumuşatır."""
        if self.smoothed_identity == 0.0:
            self.smoothed_identity, self.smoothed_residual, self.smoothed_clip = identity, residual, clip
        else:
            self.smoothed_identity = self.alpha * identity + (1 - self.alpha) * self.smoothed_identity
            self.smoothed_residual = self.alpha * residual + (1 - self.alpha) * self.smoothed_residual
            self.smoothed_clip = self.alpha * clip + (1 - self.alpha) * self.smoothed_clip
            
        return self.smoothed_identity, self.smoothed_residual, self.smoothed_clip

    def _calculate_dynamic_mask_threshold(self, current_residual: float) -> float:
        """
        TXT HEDEFİ: Hareketli sahnelerde maskeyi esnet (0.05), yavaşlarda daralt (0.02).
        Bunu Residual (Titreme/Hareket miktarı) verisine göre lineer olarak hesaplar.
        """
        # Residual genelde 0.02 (çok yavaş) ile 0.15 (çok hızlı) arasında değişir.
        # np.interp kullanarak bu aralığı [0.02, 0.05] maske eşiğine haritalıyoruz.
        min_res, max_res = 0.02, 0.15
        min_mask, max_mask = 0.02, 0.05
        
        dynamic_mask = np.interp(current_residual, [min_res, max_res], [min_mask, max_mask])
        return float(dynamic_mask)

    def step(self, identity_score: float, residual: float, clip_score: float) -> Dict[str, float]:
        """
        Ana döngü her kareden sonra bu fonksiyonu çağırır.
        Sonraki kare için gereken ayarları sözlük olarak döndürür.
        """
        id_smooth, res_smooth, clip_smooth = self._smooth_metrics(identity_score, residual, clip_score)
        
        # Hatayı hesapla (Hedef - Gerçekleşen)
        # Residual için tam tersi (Gerçekleşen - Hedef), çünkü residual'ın küçük olmasını istiyoruz.
        error_identity = self.targets.identity - id_smooth
        error_residual = res_smooth - self.targets.residual
        error_clip = self.targets.clip - clip_smooth
        
        # PID Delta güncellemelerini al
        delta_identity = self.pid_identity.update(error_identity)
        delta_residual = self.pid_residual.update(error_residual)
        delta_clip = self.pid_clip.update(error_clip)
        
        # Katsayıları güncelle (Matris/Etkileşim ağırlıkları)
        ip_delta = 0.9 * delta_identity - 0.4 * delta_residual - 0.2 * delta_clip
        ip_scale = self.pid_identity.adjust_value(ip_delta * 0.1)
        
        pose_scale = self.pid_residual.adjust_value(0.5 * delta_residual * 0.1)
        depth_scale = self.pid_clip.adjust_value(0.5 * delta_clip * 0.1)
        
        # YENİ: Dinamik Maske Eşiği Hesabı
        mask_threshold = self._calculate_dynamic_mask_threshold(res_smooth)
        
        return {
            "ip_scale": round(ip_scale, 3),
            "pose_scale": round(pose_scale, 3),
            "depth_scale": round(depth_scale, 3),
            "mask_threshold": round(mask_threshold, 3) # Latent Warper'a gidecek!
        }
