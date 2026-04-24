import logging
import torch
from tqdm import tqdm
from typing import Dict, Any, Optional, List

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
    # ✅ YENİ: Loss Guidance Parametreleri
    loss_module: Optional[torch.nn.Module] = None,
    text_prompt: Optional[List[str]] = None,
    prev_image_tensor: Optional[torch.Tensor] = None,
    vae: Optional[torch.nn.Module] = None,
    loss_scale: float = 0.5,
    controlnet_wrapper: Optional[Any] = None,
    pose_tensor: Optional[torch.Tensor] = None,
    depth_tensor: Optional[torch.Tensor] = None,
    pose_scale: float = 0.0,
    depth_scale: float = 0.0,
) -> torch.Tensor:
    """
    Generates a single predictive frame (P-Frame) using Flow-Guided Blending 
    and Loss-Guided Diffusion Optimization (Universal Guidance).
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
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        down_block_res, mid_block_res = None, None
        if controlnet_wrapper is not None and (pose_scale > 0 or depth_scale > 0):
            down_block_res, mid_block_res = controlnet_wrapper.get_residuals(
                noisy_latents=latent_model_input,
                t=t,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                pose_tensor=pose_tensor,
                depth_tensor=depth_tensor,
                pose_scale=pose_scale,
                depth_scale=depth_scale
            )

        # Loss hesaplayacaksak gradyanı korumalıyız, aksi halde no_grad
        context_manager = torch.enable_grad() if (loss_module is not None) else torch.no_grad()
        with context_manager:
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_res,
                mid_block_additional_residual=mid_block_res,
                return_dict=False
            )[0]

            # Perform CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # --- D. LOSS HESAPLAMA VE GRADIENT UPDATE (Kritik Zamansal Tutarlılık Adımı) ---
            if loss_module is not None and prev_image_tensor is not None and vae is not None:
                # 1. x0'ı (temiz görüntüyü) yaklaşık olarak tahmin et (Epsilon prediction varsayımı ile)
                alpha_prod_t = scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_x0_latent = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

                # 2. VAE ile piksel uzayına çık (Loss'lar RGB resim bekler)
                pred_x0_latent = pred_x0_latent / vae.config.scaling_factor
                pred_x0_image = vae.decode(pred_x0_latent).sample
                pred_x0_image = (pred_x0_image / 2 + 0.5).clamp(0, 1)

                # 3. Tüm loss'ları hesapla (CLIP, Perceptual, ID)
                loss, _ = loss_module(
                    pred_frame=pred_x0_image,
                    text_prompt=text_prompt if text_prompt else [""],
                    target_frame=prev_image_tensor,
                    pred_frame_temporal=pred_x0_image
                )

                # 4. Latent'a geri gradyan al ve güncelle
                grad = torch.autograd.grad(loss, latents)[0]
                latents = latents - loss_scale * grad
                
                # Gelecek adıma temiz geçmek için gradyanları kopar
                latents = latents.detach()

        # --- E. SCHEDULER STEP (Bir Sonraki Adıma Geçiş) ---
        with torch.no_grad():
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents
