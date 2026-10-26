# modules/controlnet_wrapper.py
"""
SDXL ControlNet wrapper for the Alfai video pipeline.

ControlNet vs T2I-Adapter (both live in this repo):
  - T2I-Adapter: small encoder, residuals computed ONCE per condition image and
    added inside UNet down-blocks. Cheap, but it does not see the noisy latent,
    the timestep or the text, so its guidance is coarse.
  - ControlNet: a trainable COPY of the UNet encoder (down blocks + mid block).
    It runs EVERY denoising step on (noisy latent, t, text, condition image) and
    returns 9 down residuals + 1 mid residual through zero-convolutions. Stronger,
    timestep-aware geometry lock; costs one extra encoder pass per step.

Data flow per denoising step:
    noisy latent (CFG-batched) ─┐
    timestep t                  ├─► ControlNet_k ─► (down_res_k[0..8], mid_res_k)
    text-only prompt embeds     │        × conditioning_scale_k × keep_k(step)
    condition map_k (frame i) ──┘
    Σ_k residuals ─► UNet(down_block_additional_residuals, mid_block_additional_residual)

Condition sources (what map is used for frame i):
    reference      : map of the reference image for every frame (static geometry)
    driving_video  : map of driving-video frame i (this is what gives motion)
    previous_frame : map of the previously GENERATED frame (frame 0: reference)
    hybrid         : frame 0 → reference, i>0 → previous_frame
    auto           : driving_video if given, else reference
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

# Same preprocessors as T2I-Adapter (canny via OpenCV, depth/pose via controlnet_aux).
from modules.t2i_adapter_wrapper import ConditionPreprocessor

logger = logging.getLogger(__name__)

# Diffusers-format SDXL ControlNet checkpoints.
# "small" variants are several times lighter; pick them for 6–8 GB GPUs.
CONTROLNET_REPOS: Dict[str, Dict[str, str]] = {
    "canny": {
        "small": "diffusers/controlnet-canny-sdxl-1.0-small",
        "full": "diffusers/controlnet-canny-sdxl-1.0",
    },
    "depth": {
        "small": "diffusers/controlnet-depth-sdxl-1.0-small",
        "full": "diffusers/controlnet-depth-sdxl-1.0",
    },
    "pose": {
        "full": "xinsir/controlnet-openpose-sdxl-1.0",
    },
}

VALID_SOURCES = ("auto", "reference", "driving_video", "previous_frame", "hybrid")
VALID_OFFLOAD = ("none", "sequential")

# UNet must accept these kwargs for ControlNet residual injection.
_REQUIRED_UNET_KWARGS = ("down_block_additional_residuals", "mid_block_additional_residual")


# ============================================================
# CONFIG
# ============================================================
@dataclass
class ControlNetUnitConfig:
    type: str = "depth"
    variant: str = "small"
    local_path: Optional[str] = None
    scale: float = 0.5            # conditioning_scale (0 = off, 1 = paper default)
    guidance_start: float = 0.0   # fraction of denoising steps where unit turns on
    guidance_end: float = 0.8     # fraction where it turns off (late steps = texture)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ControlNetUnitConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ControlNetConfig:
    units: List[ControlNetUnitConfig] = field(default_factory=lambda: [ControlNetUnitConfig()])
    source: str = "auto"
    guess_mode: bool = False
    temporal_smoothing: float = 0.0   # EMA on condition maps across frames, 0 = off
    offload: str = "none"             # none | sequential (accelerate cpu_offload)
    canny_low: int = 100
    canny_high: int = 200

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ControlNetConfig":
        d = dict(d or {})
        units = [ControlNetUnitConfig.from_dict(u) for u in (d.pop("units", None) or [{}])]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(units=units, **known)
        if cfg.source not in VALID_SOURCES:
            logger.warning(f"Unknown controlnet source '{cfg.source}', falling back to 'auto'.")
            cfg.source = "auto"
        if cfg.offload not in VALID_OFFLOAD:
            logger.warning(f"Unknown controlnet offload '{cfg.offload}', falling back to 'none'.")
            cfg.offload = "none"
        cfg.temporal_smoothing = float(min(max(cfg.temporal_smoothing, 0.0), 0.95))
        return cfg


# ============================================================
# ONE CONTROLNET (model + preprocessor + schedule)
# ============================================================
class ControlNetUnit:
    def __init__(self, cfg: ControlNetUnitConfig, model: torch.nn.Module, preprocessor: ConditionPreprocessor):
        self.cfg = cfg
        self.model = model
        self.preprocessor = preprocessor

    @property
    def name(self) -> str:
        return f"{self.cfg.type}/{self.cfg.variant}"

    def is_active(self, step_index: int, num_steps: int) -> bool:
        """Same rule as diffusers' controlnet_keep: on inside [start, end] of the schedule."""
        if num_steps <= 0:
            return False
        start = step_index / num_steps
        end = (step_index + 1) / num_steps
        return not (start < self.cfg.guidance_start or end > self.cfg.guidance_end)


# ============================================================
# WRAPPER
# ============================================================
class ControlNetWrapper:
    """
    Usage:
        cn = ControlNetWrapper(ControlNetConfig.from_dict(cfg), device="cuda", dtype=torch.float16)
        cn.load()
        ref_maps = cn.build_condition_maps([ref_pil], width, height)          # per unit (1,3,H,W)
        drv_maps = cn.build_condition_maps(driving_pils, width, height)       # per unit (N,3,H,W)
        cond     = cn.condition_for_frame(i, ref_maps, drv_maps, prev_pil, width, height)
        down, mid = cn.get_residuals(latent_model_input, t, text_embeds, added_cond_kwargs,
                                     cond, step_index, num_steps, do_cfg)
        unet(..., down_block_additional_residuals=down, mid_block_additional_residual=mid)
    """

    def __init__(self, config: ControlNetConfig, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.units: List[ControlNetUnit] = []
        self._prev_maps: Optional[List[torch.Tensor]] = None  # online EMA state (previous_frame source)

    # --------------------------------------------------------
    # LOADING
    # --------------------------------------------------------
    @staticmethod
    def unet_supports_controlnet(unet: torch.nn.Module) -> bool:
        params = inspect.signature(unet.forward).parameters
        missing = [k for k in _REQUIRED_UNET_KWARGS if k not in params]
        if missing:
            logger.error(f"❌ UNet.forward does not accept {missing}; ControlNet cannot inject residuals.")
            return False
        return True

    def _resolve_repo(self, cfg: ControlNetUnitConfig) -> Optional[str]:
        if cfg.local_path:
            return cfg.local_path
        variants = CONTROLNET_REPOS.get(cfg.type)
        if not variants:
            logger.error(f"❌ Unknown ControlNet type '{cfg.type}'. Valid: {list(CONTROLNET_REPOS)}")
            return None
        if cfg.variant not in variants:
            fallback = next(iter(variants))
            logger.warning(f"ControlNet '{cfg.type}' has no '{cfg.variant}' variant; using '{fallback}'.")
            cfg.variant = fallback
        return variants[cfg.variant]

    def load(self) -> bool:
        """Loads every configured unit. Units that fail are dropped. True if ≥1 unit is ready."""
        try:
            from diffusers import ControlNetModel
        except ImportError:
            logger.error("❌ diffusers.ControlNetModel is unavailable. Upgrade diffusers (>=0.30).")
            return False

        self.units = []
        for ucfg in self.config.units:
            repo = self._resolve_repo(ucfg)
            if repo is None:
                continue
            try:
                logger.info(f"Loading ControlNet '{ucfg.type}' from {repo} ...")
                model = self._from_pretrained(ControlNetModel, repo)
                model.eval().requires_grad_(False)
                model = self._place(model)
                pre = ConditionPreprocessor(
                    condition_type=ucfg.type,
                    canny_low=self.config.canny_low,
                    canny_high=self.config.canny_high,
                    device="cpu",
                )
                self.units.append(ControlNetUnit(ucfg, model, pre))
                logger.info(
                    f"✅ ControlNet '{ucfg.type}' ready (scale={ucfg.scale:.2f}, "
                    f"window=[{ucfg.guidance_start:.2f}, {ucfg.guidance_end:.2f}])."
                )
            except Exception as e:
                logger.error(f"❌ Failed to load ControlNet '{ucfg.type}': {e}")

        if not self.units:
            logger.warning("ControlNet disabled for this run (no unit loaded).")
            return False
        return True

    def _from_pretrained(self, cls, repo: str):
        # Prefer the fp16 variant when running in fp16 (half the download); fall back to default files.
        if self.dtype == torch.float16:
            try:
                return cls.from_pretrained(repo, torch_dtype=self.dtype, variant="fp16")
            except Exception:
                pass
        return cls.from_pretrained(repo, torch_dtype=self.dtype)

    def _place(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.config.offload == "sequential" and str(self.device).startswith("cuda"):
            from accelerate import cpu_offload
            cpu_offload(model, execution_device=torch.device(self.device))
            return model
        return model.to(self.device)

    @property
    def enabled(self) -> bool:
        return len(self.units) > 0

    def reset(self):
        """Call at the start of each video so EMA state does not leak between runs."""
        self._prev_maps = None

    # --------------------------------------------------------
    # CONDITION MAPS
    # --------------------------------------------------------
    def resolve_source(self, has_reference: bool, has_driving: bool) -> str:
        src = self.config.source
        if src == "auto":
            src = "driving_video" if has_driving else "reference"
        if src == "driving_video" and not has_driving:
            logger.warning("ControlNet source 'driving_video' but no driving video given → 'reference'.")
            src = "reference"
        return src

    @staticmethod
    def _map_to_tensor(pil: Image.Image) -> torch.Tensor:
        arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0      # ControlNet expects [0, 1]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)            # (1, 3, H, W), CPU fp32

    def _process_one(self, unit: ControlNetUnit, image: Image.Image, width: int, height: int) -> torch.Tensor:
        cmap = unit.preprocessor.process(image)
        if cmap.size != (width, height):
            cmap = cmap.resize((width, height), Image.Resampling.BILINEAR)
        return self._map_to_tensor(cmap)

    def build_condition_maps(
        self, images: Sequence[Image.Image], width: int, height: int
    ) -> Optional[List[torch.Tensor]]:
        """Per unit: (N, 3, H, W) CPU tensor in [0, 1]. EMA-smoothed along N when enabled."""
        if not self.enabled or not images:
            return None
        alpha = self.config.temporal_smoothing
        out: List[torch.Tensor] = []
        for unit in self.units:
            maps = [self._process_one(unit, img, width, height) for img in images]
            if alpha > 0 and len(maps) > 1:
                for i in range(1, len(maps)):
                    maps[i] = alpha * maps[i - 1] + (1.0 - alpha) * maps[i]
            out.append(torch.cat(maps, dim=0))
        return out

    def condition_for_frame(
        self,
        frame_index: int,
        ref_maps: Optional[List[torch.Tensor]],
        driving_maps: Optional[List[torch.Tensor]],
        prev_image: Optional[Image.Image],
        width: int,
        height: int,
    ) -> Optional[List[torch.Tensor]]:
        """Picks this frame's condition map per unit, (1, 3, H, W) each, or None."""
        if not self.enabled:
            return None
        src = self.resolve_source(ref_maps is not None, driving_maps is not None)

        if src == "driving_video":
            maps = [m[min(frame_index, m.shape[0] - 1)].unsqueeze(0) for m in driving_maps]
        elif src == "reference" or frame_index == 0 or prev_image is None:
            if ref_maps is None:
                return None
            maps = [m[:1] for m in ref_maps]
        else:  # previous_frame / hybrid, i > 0
            new = self.build_condition_maps([prev_image], width, height)
            alpha = self.config.temporal_smoothing
            if alpha > 0 and self._prev_maps is not None:
                new = [alpha * p + (1.0 - alpha) * n for p, n in zip(self._prev_maps, new)]
            maps = new

        self._prev_maps = maps
        return maps

    def conditions_for_batch(
        self,
        num_frames: int,
        ref_maps: Optional[List[torch.Tensor]],
        driving_maps: Optional[List[torch.Tensor]],
    ) -> Optional[List[torch.Tensor]]:
        """TSM batch mode: per unit (num_frames, 3, H, W). previous_frame is impossible here."""
        if not self.enabled:
            return None
        src = self.resolve_source(ref_maps is not None, driving_maps is not None)
        if src in ("previous_frame", "hybrid"):
            logger.warning("TSM batch mode generates all frames jointly; ControlNet uses 'reference' instead.")
            src = "reference"
        if src == "driving_video":
            idx = torch.linspace(0, driving_maps[0].shape[0] - 1, num_frames).round().long()
            return [m[idx] for m in driving_maps]
        if ref_maps is None:
            return None
        return [m[:1].repeat(num_frames, 1, 1, 1) for m in ref_maps]

    # --------------------------------------------------------
    # RESIDUALS (called every denoising step)
    # --------------------------------------------------------
    @torch.no_grad()
    def get_residuals(
        self,
        latent_model_input: torch.Tensor,
        t: Union[torch.Tensor, int],
        text_embeds: torch.Tensor,
        added_cond_kwargs: Dict[str, torch.Tensor],
        cond_images: Optional[List[torch.Tensor]],
        step_index: int,
        num_steps: int,
        do_cfg: bool,
    ) -> Tuple[Optional[Tuple[torch.Tensor, ...]], Optional[torch.Tensor]]:
        """
        Args:
            latent_model_input: scheduler-scaled latent, already CFG-batched ([uncond, cond]).
            text_embeds: TEXT-ONLY prompt embeds (no IP-Adapter tokens), same batch as latent input.
            cond_images: per unit (N, 3, H, W) in [0, 1]; N = 1 or latent batch per CFG half.
        Returns:
            (down_block_additional_residuals, mid_block_additional_residual) or (None, None).
        """
        if not self.enabled or cond_images is None:
            return None, None

        guess = self.config.guess_mode and do_cfg
        if guess:
            # Guess mode: ControlNet sees only the conditional half; uncond half gets zero residuals.
            cn_latent = latent_model_input.chunk(2)[1]
            cn_text = text_embeds.chunk(2)[1]
            cn_added = {k: v.chunk(2)[1] for k, v in added_cond_kwargs.items()}
        else:
            cn_latent, cn_text, cn_added = latent_model_input, text_embeds, added_cond_kwargs

        half_batch = latent_model_input.shape[0] // (2 if do_cfg else 1)
        down_sum: Optional[List[torch.Tensor]] = None
        mid_sum: Optional[torch.Tensor] = None

        for unit, cond in zip(self.units, cond_images):
            if cond is None or unit.cfg.scale <= 0 or not unit.is_active(step_index, num_steps):
                continue

            cond = cond.to(device=latent_model_input.device, dtype=self.dtype)
            if cond.shape[0] == 1 and half_batch > 1:
                cond = cond.repeat(half_batch, 1, 1, 1)
            if do_cfg and not guess:
                cond = torch.cat([cond, cond], dim=0)

            down, mid = unit.model(
                cn_latent.to(self.dtype),
                t,
                encoder_hidden_states=cn_text.to(self.dtype),
                controlnet_cond=cond,
                conditioning_scale=float(unit.cfg.scale),
                added_cond_kwargs={k: v.to(self.dtype) for k, v in cn_added.items()},
                guess_mode=self.config.guess_mode,
                return_dict=False,
            )

            if guess:
                down = [torch.cat([torch.zeros_like(d), d], dim=0) for d in down]
                mid = torch.cat([torch.zeros_like(mid), mid], dim=0)

            if down_sum is None:
                down_sum, mid_sum = list(down), mid
            else:
                down_sum = [a + b for a, b in zip(down_sum, down)]
                mid_sum = mid_sum + mid

        if down_sum is None:
            return None, None
        unet_dtype = latent_model_input.dtype
        return tuple(d.to(unet_dtype) for d in down_sum), mid_sum.to(unet_dtype)


# ============================================================
# DRIVING VIDEO LOADER
# ============================================================
def load_driving_frames(path: str, num_frames: int, width: int, height: int) -> List[Image.Image]:
    """Reads a video and uniformly samples `num_frames` frames, resized to (width, height)."""
    import imageio.v3 as iio

    frames = [Image.fromarray(f).convert("RGB") for f in iio.imiter(path)]
    if not frames:
        raise ValueError(f"No frames could be read from driving video: {path}")
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    logger.info(f"Driving video: {len(frames)} frames read, {num_frames} sampled.")
    return [frames[i].resize((width, height), Image.Resampling.LANCZOS) for i in idx]
