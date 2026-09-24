# modules/interpolation/latent_interpolation.py
"""
Latent space interpolation utilities.

SLERP (Spherical Linear Interpolation) is used instead of LERP for latent spaces
because latent vectors lie on a hypersphere. LERP shrinks the magnitude at the
midpoint (0.7 * max instead of 1.0), causing blurry/low-energy generations.
SLERP preserves magnitude throughout the interpolation path.

Usage in video pipeline:
  - Keyframe transition: lerp/slerp between two latents across N frames
  - Frame initialization: blend DDIM-inverted latent with pure noise for P-frames
    (closer to reference early on, more free later)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    """
    Spherical Linear Interpolation between two latent tensors.

    Args:
        t:              Interpolation factor [0.0 → v0, 1.0 → v1]
        v0:             Start latent tensor (any shape)
        v1:             End latent tensor (same shape as v0)
        dot_threshold:  If dot product is above this, fall back to LERP
                        (vectors nearly parallel → SLERP numerically unstable)

    Returns:
        Interpolated latent tensor, same shape as inputs.
    """
    orig_shape = v0.shape
    # Flatten to (B, -1) for norm computation
    v0_flat = v0.reshape(v0.shape[0], -1).float()
    v1_flat = v1.reshape(v1.shape[0], -1).float()

    v0_norm = F.normalize(v0_flat, dim=-1)
    v1_norm = F.normalize(v1_flat, dim=-1)

    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (B, 1)

    # Paralel vektörler için LERP fallback
    is_parallel = (dot.abs() > dot_threshold)

    theta = torch.acos(dot.clamp(-1 + 1e-6, 1 - 1e-6))       # (B, 1)
    sin_theta = torch.sin(theta)                                # (B, 1)

    s0 = torch.sin((1.0 - t) * theta) / sin_theta              # (B, 1)
    s1 = torch.sin(t * theta) / sin_theta                      # (B, 1)

    slerp_flat = s0 * v0_flat + s1 * v1_flat

    # Paralel ise LERP kullan
    lerp_flat = (1.0 - t) * v0_flat + t * v1_flat
    result_flat = torch.where(is_parallel.expand_as(slerp_flat), lerp_flat, slerp_flat)

    return result_flat.reshape(orig_shape).to(dtype=v0.dtype)


def slerp_embeddings(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    """
    Token-wise SLERP for CLIP / SDXL prompt embeddings.

    Last dimension is the embedding axis (e.g. 2048 or 1280). Each token
    interpolates on its own sphere so scene semantics blend instead of
    collapsing the whole (seq, dim) vector into one direction.
    """
    orig_shape = v0.shape
    v0_tok = v0.reshape(-1, orig_shape[-1])
    v1_tok = v1.reshape(-1, orig_shape[-1])
    out = slerp(t, v0_tok, v1_tok, dot_threshold=dot_threshold)
    return out.reshape(orig_shape)


def lerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation between two latent tensors.
    Faster than SLERP but magnitude drops at midpoint.
    Use for quick tests or when |v0| ≈ |v1| ≈ 1.

    Args:
        t:  [0.0 → v0, 1.0 → v1]
    """
    return (1.0 - t) * v0 + t * v1


def slerp_sequence(
    v0: torch.Tensor,
    v1: torch.Tensor,
    num_frames: int,
    mode: str = "slerp",
) -> list:
    """
    Generates a sequence of interpolated latents between v0 and v1.

    Args:
        v0:         Keyframe A latent
        v1:         Keyframe B latent
        num_frames: Total number of frames (includes endpoints)
        mode:       'slerp' (default) or 'lerp'

    Returns:
        List of num_frames latent tensors [v0, ..., v1]
    """
    frames = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        if mode == "slerp":
            frames.append(slerp(t, v0, v1))
        else:
            frames.append(lerp(t, v0, v1))
    return frames


def noise_blend_schedule(
    ddim_latent: torch.Tensor,
    num_frames: int,
    start_blend: float = 0.0,
    end_blend: float = 0.85,
    generator: Optional[torch.Generator] = None,
) -> list:
    """
    P-frame başlangıç latent'lerini üretir:
    Frame 0 = DDIM latent (referansa yakın)
    Son frame ≈ saf gürültü (tam serbest üretim)

    SLERP kullanarak DDIM latent → random noise arsında yumuşak geçiş yapar.
    Bu sayede ilk P-frame'ler referansa daha yakın çıkar, son frame'ler
    daha dinamik/hareketli olur.

    Args:
        ddim_latent:  Frame 0'ın inverted latenti
        num_frames:   Toplam frame sayısı
        start_blend:  Frame 0 noise blend faktörü (0.0 = saf DDIM latent)
        end_blend:    Son frame noise blend faktörü (1.0 = saf gürültü)
        generator:    Tekrarlanabilir rastgele seed

    Returns:
        List of num_frames initial latent tensors
    """
    device = ddim_latent.device
    dtype = ddim_latent.dtype
    latents = []

    for i in range(num_frames):
        progress = i / max(num_frames - 1, 1)                  # 0.0 → 1.0
        blend = start_blend + (end_blend - start_blend) * progress  # start_blend → end_blend

        noise = torch.randn(ddim_latent.shape, device=ddim_latent.device, dtype=ddim_latent.dtype, generator=generator)
        blended = slerp(blend, ddim_latent, noise)
        latents.append(blended.to(dtype=dtype))

    return latents
