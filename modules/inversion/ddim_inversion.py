import logging
import torch
from diffusers import DDIMScheduler
from typing import Tuple, Union, Any

logger = logging.getLogger(__name__)

class DDIMInversion:
    """
    DDIM Inversion & Reconstruction (PARTIAL INVERSION DESTEKLİ)
    - stop_at_t: Inversion işleminin nerede duracağını belirler (Örn: 350).
    - start_at_t: Reconstruction işleminin nereden başlayacağını belirler.
    """

    def __init__(self, pipe, device="cuda", null_prompt_embeds=None, null_pooled_embeds=None):
        self.pipe = pipe
        self.device = device
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.scheduler.set_timesteps(50, device=device)
        self.pipe.vae.to(dtype=torch.float32)
        
        # PRODUCTION FIX: Pipeline'ın içinde encode_prompt aramak yerine 
        # dışarıdan alıyoruz veya VRAM dostu sıfır tensörleri kullanıyoruz.
        if null_prompt_embeds is not None and null_pooled_embeds is not None:
            self.prompt_embeds = null_prompt_embeds.to(device)
            self.pooled = null_pooled_embeds.to(device)
        else:
            # Fallback: Eğer dışarıdan verilmezse, SDXL'in beklediği boyutta boş tensörler
            self.prompt_embeds = torch.zeros((1, 77, 2048), device=device, dtype=self.pipe.unet.dtype)
            self.pooled = torch.zeros((1, 1280), device=device, dtype=self.pipe.unet.dtype)
    @torch.no_grad()
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device, dtype=torch.float32)
        latent = self.pipe.vae.encode(image).latent_dist.sample()
        latent = latent * self.pipe.vae.config.scaling_factor
        return latent.to(self.device, dtype=self.pipe.unet.dtype)

    @torch.no_grad()
    def invert(self, image: torch.Tensor, num_steps: int = 50, stop_at_t: int = 1000) -> torch.Tensor:
        """
        Resim -> Latent (0 -> stop_at_t)
        stop_at_t: 350 ise, işlem t=350 olunca durur.
        """
        latent = self.encode_image(image)
        self.scheduler.set_timesteps(num_steps, device=self.device)
        
        # Zamanı tersine çevir (0 -> 1000)
        # BUG FIX: reversed() bir iterator döndürür. İndeksleme yapabilmek için list'e çevirmeliyiz.
        timesteps_list = list(reversed(self.scheduler.timesteps))
        
        logger.info(f"Inversion çalışıyor... Hedef t={stop_at_t}")

        for i, t in enumerate(timesteps_list):
            # Eğer mevcut zaman adımı, hedefimizden büyükse dur.
            if t.item() > stop_at_t:
                logger.info(f"   -> t={t.item()} noktasına ulaşıldı, Inversion durduruluyor.")
                break

            # ==========================================
            # TENSÖR HESAPLAMALARI (Kural 7 Korumasında)
            # ==========================================
            
            # Alpha hesapları
            alpha_prod_t = self.scheduler.alphas_cumprod.to(self.device)[t.long()]
            
            if i < len(timesteps_list) - 1:
                next_t = timesteps_list[i + 1]
                alpha_prod_t_next = self.scheduler.alphas_cumprod.to(self.device)[next_t.long()]
            else:
                alpha_prod_t_next = torch.tensor(0.001, device=self.device)

            # UNet Tahmini
            added_cond = {"text_embeds": self.pooled, "time_ids": torch.zeros((1, 6), device=self.device)}
            noise_pred = self.pipe.unet(
                latent, 
                t, 
                encoder_hidden_states=self.prompt_embeds, 
                added_cond_kwargs=added_cond
            ).sample

            # DDIM Inversion Step Matematiği
            beta_prod_t = 1.0 - alpha_prod_t
            pred_original_sample = (latent - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            beta_prod_t_next = 1.0 - alpha_prod_t_next
            
            # Yeni latent tensörünün güncellenmesi
            latent = alpha_prod_t_next ** 0.5 * pred_original_sample + beta_prod_t_next ** 0.5 * noise_pred

        return latent

    @torch.no_grad()
    def reconstruct(self, noisy_latent: torch.Tensor, num_steps: int = 50, start_at_t: int = 1000) -> torch.Tensor:
        """
        Latent -> Resim (start_at_t -> 0)
        start_at_t: 350 ise, gürültü çözme işlemi 350'den başlar.
        """
        latent = noisy_latent.clone()
        self.scheduler.set_timesteps(num_steps, device=self.device)
        
        # Normal zaman akışı (1000 -> 0)
        timesteps = [t for t in self.scheduler.timesteps if t.item() <= start_at_t]

        logger.info(f"Reconstruction çalışıyor... Başlangıç t={start_at_t}")

        for i, t in enumerate(timesteps):
            added_cond = {"text_embeds": self.pooled, "time_ids": torch.zeros((1, 6), device=self.device)}
            
            noise_pred = self.pipe.unet(
                latent, 
                t, 
                encoder_hidden_states=self.prompt_embeds, 
                added_cond_kwargs=added_cond
            ).sample
            
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample

        return latent