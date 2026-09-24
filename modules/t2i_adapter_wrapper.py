# modules/t2i_adapter_wrapper.py
"""
T2I-Adapter wrapper for the SDXL video pipeline.

T2I-Adapter is the lightweight geometry lock (pose / depth / canny) used
instead of a full ControlNet while VRAM is limited (e.g. 6–8 GB).

It does not replace UNet. It produces encoder residuals that the existing
custom UNet already consumes via `down_intrablock_additional_residuals`.

Architecture:
  Condition image (reference or previous frame)
    → preprocessor (Canny / Depth / Pose)
    → T2IAdapter (~77M params)
    → list of residual feature maps
    → added inside UNet down-blocks each denoising step
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# SDXL T2I-Adapter checkpoints (TencentARC)
ADAPTER_REPOS = {
    "canny": "TencentARC/t2i-adapter-canny-sdxl-1.0",
    "depth": "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
    "pose": "TencentARC/t2i-adapter-openpose-sdxl-1.0",
    "lineart": "TencentARC/t2i-adapter-lineart-sdxl-1.0",
    "sketch": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
}

VALID_SOURCES = ("reference", "previous_frame", "hybrid")


def _to_pil(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported condition image type: {type(image)}")


class ConditionPreprocessor:
    """
    Builds a 3-channel condition map for T2I-Adapter.

    Canny uses OpenCV only (no extra neural net — preferred for VRAM tests).
    Depth / pose lazily load controlnet_aux detectors on CPU.
    """

    def __init__(
        self,
        condition_type: str = "canny",
        canny_low: int = 100,
        canny_high: int = 200,
        device: str = "cpu",
    ):
        self.condition_type = condition_type
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.device = device
        self._detector = None

    def _load_aux_detector(self):
        if self._detector is not None:
            return self._detector
        try:
            from controlnet_aux import MidasDetector, OpenposeDetector, LineartDetector, PidiNetDetector
        except ImportError as e:
            raise ImportError(
                "depth / pose / lineart / sketch require controlnet-aux. "
                "Install with: pip install controlnet-aux"
            ) from e

        logger.info(f"Loading controlnet_aux detector for '{self.condition_type}' on CPU...")
        if self.condition_type == "depth":
            self._detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        elif self.condition_type == "pose":
            self._detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        elif self.condition_type == "lineart":
            self._detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        elif self.condition_type == "sketch":
            self._detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        else:
            raise ValueError(f"No aux detector for type '{self.condition_type}'")
        logger.info(f"✅ Detector '{self.condition_type}' loaded.")
        return self._detector

    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Returns an RGB condition map (same size as input)."""
        pil = _to_pil(image)

        if self.condition_type == "canny":
            gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            edges_rgb = np.stack([edges] * 3, axis=-1)
            return Image.fromarray(edges_rgb)

        detector = self._load_aux_detector()
        result = detector(pil)
        if not isinstance(result, Image.Image):
            result = Image.fromarray(np.array(result))
        return result.convert("RGB").resize(pil.size, Image.Resampling.LANCZOS)


class T2IAdapterWrapper:
    """
    Loads an SDXL T2I-Adapter and encodes condition images into UNet residuals.

    Usage:
        wrapper = T2IAdapterWrapper(device="cuda", dtype=torch.float16, scale=0.8)
        wrapper.load(adapter_type="canny")
        residuals = wrapper.encode_image(pil_image)
        unet_residuals = wrapper.expand_for_cfg(residuals, do_cfg=True)
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        scale: float = 0.8,
        source: str = "hybrid",
        condition_type: str = "canny",
        canny_low: int = 100,
        canny_high: int = 200,
    ):
        self.device = device
        self.dtype = dtype
        self.scale = scale
        self.source = source if source in VALID_SOURCES else "hybrid"
        self.condition_type = condition_type
        self.adapter = None
        self.preprocessor = ConditionPreprocessor(
            condition_type=condition_type,
            canny_low=canny_low,
            canny_high=canny_high,
            device="cpu",
        )
        self._loaded = False

    def load(self, local_path: Optional[str] = None) -> bool:
        """
        Loads T2I-Adapter weights from a local folder or HuggingFace.

        Returns:
            True if successfully loaded.
        """
        try:
            from diffusers import T2IAdapter
        except ImportError:
            logger.error("❌ diffusers.T2IAdapter is unavailable. Upgrade diffusers (>=0.30).")
            return False

        if self.condition_type not in ADAPTER_REPOS:
            logger.error(
                f"❌ Unknown T2I-Adapter type '{self.condition_type}'. "
                f"Valid: {list(ADAPTER_REPOS)}"
            )
            return False

        try:
            if local_path:
                logger.info(f"Loading T2I-Adapter from local path: {local_path}")
                repo_or_path = local_path
            else:
                repo_or_path = ADAPTER_REPOS[self.condition_type]
                logger.info(f"Loading T2I-Adapter from HuggingFace: {repo_or_path}")

            self.adapter = T2IAdapter.from_pretrained(repo_or_path, torch_dtype=self.dtype)
            self.adapter.to(self.device)
            self.adapter.eval()
            self._loaded = True
            logger.info(
                f"✅ T2I-Adapter '{self.condition_type}' loaded "
                f"(scale={self.scale:.2f}, source={self.source})."
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load T2I-Adapter: {e}")
            logger.warning("T2I-Adapter will be disabled for this run.")
            self.adapter = None
            self._loaded = False
            return False

    def set_scale(self, scale: float):
        self.scale = float(scale)

    def _image_to_tensor(self, condition_map: Image.Image) -> torch.Tensor:
        arr = np.array(condition_map).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, np.ndarray],
        scale: Optional[float] = None,
    ) -> Optional[List[torch.Tensor]]:
        """
        Preprocess + encode a condition image into UNet residual maps.

        Returns:
            List of residual tensors, or None if adapter is not loaded.
        """
        if not self._loaded or self.adapter is None:
            return None

        condition_map = self.preprocessor.process(image)
        adapter_input = self._image_to_tensor(condition_map)
        residuals = self.adapter(adapter_input)
        if not isinstance(residuals, (list, tuple)):
            residuals = [residuals]

        applied_scale = self.scale if scale is None else scale
        encoded: List[torch.Tensor] = []
        for residual in residuals:
            encoded.append(residual.to(device=self.device, dtype=self.dtype) * applied_scale)
        return encoded

    @staticmethod
    def expand_for_cfg(
        residuals: Optional[List[torch.Tensor]],
        do_cfg: bool,
    ) -> Optional[List[torch.Tensor]]:
        """
        Duplicate residuals along batch for Classifier-Free Guidance.

        UNet pops from this list, so callers must pass a fresh copy every step
        (see `clone_residuals`).
        """
        if residuals is None:
            return None
        if not do_cfg:
            return [r.clone() for r in residuals]
        return [torch.cat([r, r], dim=0) for r in residuals]

    @staticmethod
    def clone_residuals(residuals: Optional[List[torch.Tensor]]) -> Optional[List[torch.Tensor]]:
        """Fresh list for a single UNet forward (UNet mutates via list.pop)."""
        if residuals is None:
            return None
        return [r.clone() for r in residuals]

    @staticmethod
    def repeat_for_batch(
        residuals: Optional[List[torch.Tensor]],
        batch_size: int,
    ) -> Optional[List[torch.Tensor]]:
        """Repeat residuals for TSM batch-temporal mode (B*T)."""
        if residuals is None:
            return None
        if batch_size <= 1:
            return [r.clone() for r in residuals]
        return [r.repeat(batch_size, 1, 1, 1) for r in residuals]
