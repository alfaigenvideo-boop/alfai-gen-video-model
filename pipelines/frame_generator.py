import logging
import torch
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

def generate_next_frame(
    unet: torch.nn.Module,
    scheduler: Any,
    previous_latents_warped: torch.Tensor,
    mask: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: Dict[str, Any],
    height: int,
    width: int,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    device: str = "cuda",
    generator: Optional[torch.Generator] = None,
    # ✅ YENİ: IP-Adapter Parametresi (video_pipeline'dan geliyor)
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    # Loss Guidance Parametreleri
    loss_module: Optional[torch.nn.Module] = None,
    text_prompt: Optional[Union[List[str], str]] = None, # Hem list hem string desteklesin
    prev_image_tensor: Optional[torch.Tensor] = None,
    vae: Optional[torch.nn.Module] = None,
    loss_scale: float = 0.5,
) -> torch.Tensor:
    """
    Generates a single predictive frame (P-Frame) using Flow-Guided Blending 
    and Loss-Guided Diffusion Optimization (Universal Guidance).
    Includes support for IP-Adapter via cross_attention_kwargs.
    """
    
    # 1. Başlangıç Gürültüsü (Latent t_T)
    latents = torch.randn(
        (1, 4, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=unet.dtype
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    do_classifier_free_guidance = guidance_scale > 1.0

    # 2. Denoising Loop (Adım Adım Gürültü Kaldırma ve Optimizasyon)
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Generating Frame", leave=False)):
        
        # --- A. FLOW-GUIDED BLENDING ---
        # Warped latent'a o anki timestep (t) kadar gürültü ekleyip, 
        # inpaint maskesi (1: Üret, 0: Koru) ile birleştiriyoruz.
        noise = torch.randn_like(previous_latents_warped)
        warped_latents_noisy = scheduler.add_noise(
            original_samples=previous_latents_warped, 
            noise=noise, 
            timesteps=t.reshape(1,)
        )
        latents = (latents * mask) + (warped_latents_noisy * (1.0 - mask))
        
        # --- B. LOSS GUIDANCE OPTIMIZATION (Geri Yayılım) ---
        # Eğer optimizasyon isteniyorsa latents üzerinde gradient akışını açıyoruz
        if loss_module is not None and vae is not None:
            latents = latents.detach().requires_grad_(True)

        # --- C. UNET TAHMİNİ (CFG ile) ---
        # Classifier-Free Guidance için latents'i kopyalıyoruz (Uncond + Cond)
        # --- C. UNET TAHMİNİ (CFG ile) ---
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # ❗ KRİTİK DÜZELTME: UNET TAHMİNİ HER ZAMAN no_grad İÇİNDE OLMALI
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=cross_attention_kwargs, 
                return_dict=False
            )[0]

            # Perform CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # --- D. LOSS HESAPLAMA VE GRADIENT UPDATE (Kritik Zamansal Tutarlılık Adımı) ---
        if loss_module is not None and prev_image_tensor is not None and vae is not None:
            # ❗ KRİTİK DÜZELTME: Geri yayılımı sadece Loss ve VAE için aktif ediyoruz
            with torch.enable_grad():
                # 1. x0'ı (temiz görüntüyü) yaklaşık olarak tahmin et (Epsilon prediction varsayımı ile)
                alpha_prod_t = scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                
                # ❗ KRİTİK DÜZELTME: noise_pred.detach() diyerek UNet ile olan türev bağını kopardık
                pred_x0_latent = (latents - beta_prod_t ** (0.5) * noise_pred.detach()) / alpha_prod_t ** (0.5)

                # 2. VAE BYPASS (SIFIR VRAM HARCAYAN RGB YAKLAŞIMI)
                # SDXL VAE'nin içinden türev almak 24GB+ VRAM gerektirir ve OOM'a yol açar.
                # Bunun yerine 4 kanallı latenti, standart bir matrisle doğrudan RGB'ye yansıtıyoruz!
                pred_x0_latent_scaled = pred_x0_latent / vae.config.scaling_factor
                
                rgb_factors = torch.tensor([
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=pred_x0_latent.device, dtype=pred_x0_latent.dtype)
                
                # Tensör çarpımı ile (B, 4, H, W) -> (B, 3, H, W) elde ediyoruz
                pred_x0_image = torch.einsum("b c h w, c d -> b d h w", pred_x0_latent_scaled, rgb_factors)
                pred_x0_image = (pred_x0_image / 2 + 0.5).clamp(0, 1)

                pred_x0_image = torch.nn.functional.interpolate(
                    pred_x0_image, 
                    size=(prev_image_tensor.shape[2], prev_image_tensor.shape[3]), 
                    mode="bilinear", 
                    align_corners=False
                )

                # 3. Tüm loss'ları hesapla (CLIP, Perceptual, ID)
                formatted_text_prompt = text_prompt if isinstance(text_prompt, list) else [text_prompt] if text_prompt is not None else [""]
                
                loss, _ = loss_module(
                    pred_frame=pred_x0_image,
                    text_prompt=formatted_text_prompt,
                    target_frame=prev_image_tensor,
                    pred_frame_temporal=pred_x0_image
                )

                # 4. Sadece 'latents' tensörüne doğru gradyan al ve güncelle
                grad = torch.autograd.grad(loss, latents)[0]
                latents = latents - loss_scale * grad
                
                # Gelecek adıma temiz geçmek için gradyanları kopar
                latents = latents.detach()

        # --- E. SCHEDULER STEP (Bir Sonraki Adıma Geçiş) ---
        with torch.no_grad():
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents