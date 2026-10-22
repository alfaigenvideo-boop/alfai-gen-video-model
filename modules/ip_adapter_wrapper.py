# modules/ip_adapter_wrapper.py
"""
IP-Adapter FaceID Wrapper for SDXL Video Pipeline.

Architecture:
  Reference Photo
    → InsightFace ArcFace → 512-dim face embedding
    → MLPProjModel (projection) → cross-attention key/value pairs
    → IPAttnProcessor2_0 injected into UNet attention layers
    → Each frame "remembers" the face identity
"""

import logging
import os
import re
from typing import Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. FACE EMBEDDING EXTRACTOR (InsightFace ArcFace)
# ---------------------------------------------------------------------------

class FaceIDExtractor:
    """
    Extracts 512-dim ArcFace face embeddings from images using InsightFace.
    Falls back gracefully if insightface is not installed.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.app = None
        self._load_insightface()

    def _load_insightface(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            logger.info("Loading InsightFace (buffalo_l) for face embedding extraction...")
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
            logger.info("✅ InsightFace loaded successfully.")
        except ImportError:
            logger.warning(
                "⚠️ insightface not installed. Face embedding extraction disabled. "
                "Install with: pip install insightface onnxruntime-gpu"
            )
            self.app = None

    def extract(self, image_rgb_numpy) -> Optional[torch.Tensor]:
        """
        Extracts face embedding from a numpy image (H, W, 3) uint8 RGB.

        Returns:
            torch.Tensor of shape (1, 512) or None if no face detected.
        """
        if self.app is None:
            return None

        faces = self.app.get(image_rgb_numpy)
        if not faces:
            logger.warning("⚠️ No face detected in the reference image. IP-Adapter FaceID will be skipped.")
            return None

        # Use the largest face (most prominent)
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        embedding = torch.from_numpy(face.normed_embedding).unsqueeze(0)  # (1, 512)
        logger.info(f"✅ Face detected and embedded. Embedding shape: {embedding.shape}")
        return embedding.to(self.device)


# ---------------------------------------------------------------------------
# 2. MLP PROJECTION MODEL (ArcFace embedding → cross-attention compatible)
# ---------------------------------------------------------------------------

class MLPProjModel(nn.Module):
    """
    Projects 512-dim ArcFace embedding to (num_tokens, cross_attention_dim).
    IP-Adapter FaceID uses 4 tokens per image, cross_attention_dim=2048 for SDXL.
    """

    def __init__(self, cross_attention_dim: int = 2048, id_embeddings_dim: int = 512, num_tokens: int = 4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            id_embeds: (B, 512) ArcFace embeddings

        Returns:
            (B, num_tokens, cross_attention_dim)
        """
        x = self.proj(id_embeds)
        x = x.view(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 3. IP-ADAPTER ATTENTION PROCESSOR
# ---------------------------------------------------------------------------

class IPAttnProcessor2_0(nn.Module):
    """
    Replaces standard cross-attention with IP-Adapter aware cross-attention.
    Injects image embeddings as additional key-value pairs alongside text.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int = 4,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # IP tokenlerinin varlığını kontrol et:
        # Standart CLIP token uzunluğu 77. Eğer encoder_hidden_states 77'den uzunsa
        # IP-Adapter image tokenleri eklenmiş demektir.
        # DDIM Inversion ve IP token olmayan çağrılarda güvenli şekilde normal cross-attention yap.
        has_ip_tokens = (
            encoder_hidden_states is not None
            and encoder_hidden_states.shape[1] > 77   # 77 + num_tokens (4) = 81
            and self.scale > 0.0
        )

        if has_ip_tokens:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            text_hidden_states = encoder_hidden_states[:, :end_pos, :]
            ip_hidden_states = encoder_hidden_states[:, end_pos:, :]
        else:
            text_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            ip_hidden_states = None

        # Standard cross-attention with text
        query = attn.to_q(hidden_states)
        key = attn.to_k(text_hidden_states)
        value = attn.to_v(text_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # IP-Adapter cross-attention
        if ip_hidden_states is not None:
            # ✅ dtype fix: to_k_ip fp32 olabilir ama input fp16 — zorla eşleştir
            weight_dtype = self.to_k_ip.weight.dtype
            ip_hidden_casted = ip_hidden_states.to(dtype=weight_dtype)

            ip_key = self.to_k_ip(ip_hidden_casted)
            ip_value = self.to_v_ip(ip_hidden_casted)
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ip_attn = F.scaled_dot_product_attention(
                query.to(dtype=weight_dtype), ip_key, ip_value, dropout_p=0.0
            )
            ip_attn = ip_attn.to(dtype=hidden_states.dtype)
            ip_attn = ip_attn.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            hidden_states = hidden_states + self.scale * ip_attn

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# 4. MAIN IP-ADAPTER FACEID CLASS
# ---------------------------------------------------------------------------

class IPAdapterFaceID:
    """
    Loads IP-Adapter FaceID weights and injects attention processors into UNet.

    Usage:
        adapter = IPAdapterFaceID(unet, device="cuda")
        adapter.load_weights()
        face_embeds = adapter.get_image_embeds(face_embedding_512dim)
        # Then pass face_embeds via cross_attention_kwargs
    """

    HF_REPO = "h94/IP-Adapter-FaceID"
    WEIGHT_FILE = "ip-adapter-faceid_sdxl.bin"

    NUM_TOKENS = 4
    CROSS_ATTENTION_DIM = 2048  # SDXL

    def __init__(self, unet: nn.Module, device: str = "cuda", scale: float = 0.8):
        self.unet = unet
        self.device = device
        self.scale = scale
        self.image_proj_model = None
        self._loaded = False

    def load_weights(self, local_path: Optional[str] = None) -> bool:
        """
        Loads IP-Adapter FaceID weights from local path or HuggingFace.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if local_path and os.path.exists(local_path):
                weight_path = local_path
                logger.info(f"Loading IP-Adapter FaceID from local: {weight_path}")
            else:
                logger.info(f"Downloading IP-Adapter FaceID from HuggingFace: {self.HF_REPO}")
                weight_path = hf_hub_download(
                    repo_id=self.HF_REPO,
                    filename=self.WEIGHT_FILE,
                )

            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

            # Build and load projection model
            self.image_proj_model = MLPProjModel(
                cross_attention_dim=self.CROSS_ATTENTION_DIM,
                id_embeddings_dim=512,
                num_tokens=self.NUM_TOKENS,
            )
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
            self.image_proj_model = self.image_proj_model.to(self.device)
            self.image_proj_model.eval()

            # Inject attention processors into UNet
            self._inject_processors(state_dict["ip_adapter"])

            self._loaded = True
            logger.info("✅ IP-Adapter FaceID loaded and injected successfully.")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load IP-Adapter FaceID: {e}")
            logger.warning("IP-Adapter FaceID will be disabled for this run.")
            return False

    def _inject_processors(self, adapter_state_dict: dict):
        """UNet'in cross-attention (attn2) katmanlarına IP-Adapter işlemcisi takar."""
        attn_procs = {}
        n_ip = 0

        for idx, name in enumerate(self.unet.attn_processors.keys()):
            if not name.endswith("attn2.processor"):
                attn_procs[name] = self.unet.attn_processors[name]  # self-attention aynen kalır
                continue

            attn = self.unet.get_submodule(name[: -len(".processor")])
            hidden_size = attn.to_q.out_features
            cross_dim = attn.to_k.in_features

            k_key, v_key = f"{idx}.to_k_ip.weight", f"{idx}.to_v_ip.weight"
            if k_key not in adapter_state_dict or v_key not in adapter_state_dict:
                raise KeyError(f"IP-Adapter ağırlığı bulunamadı: {k_key} (katman: {name})")

            proc = IPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                num_tokens=self.NUM_TOKENS,
                scale=self.scale,
            )
            proc.to_k_ip.weight = nn.Parameter(adapter_state_dict[k_key].to(self.device))
            proc.to_v_ip.weight = nn.Parameter(adapter_state_dict[v_key].to(self.device))
            attn_procs[name] = proc
            n_ip += 1

        if n_ip == 0:
            raise RuntimeError("IP-Adapter: hiçbir cross-attention katmanına işlemci takılamadı.")

        self.unet.set_attn_processor(attn_procs)
        logger.info(f"Injected IPAttnProcessor into {n_ip} cross-attention layers.")


    def set_scale(self, scale: float):
        """Dynamically update IP-Adapter scale for all attention processors."""
        self.scale = scale
        for proc in self.unet.attn_processors.values():
            if isinstance(proc, IPAttnProcessor2_0):
                proc.scale = scale

    @torch.no_grad()
    def get_image_embeds(self, face_embedding: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Projects raw ArcFace embedding (1, 512) to cross-attention tokens (1, num_tokens, 2048).

        Args:
            face_embedding: (1, 512) from FaceIDExtractor.extract()

        Returns:
            (1, num_tokens, 2048) ready to be appended to prompt_embeds
        """
        if not self._loaded or self.image_proj_model is None:
            return None
        face_embedding = face_embedding.to(device=self.device, dtype=next(self.image_proj_model.parameters()).dtype)
        return self.image_proj_model(face_embedding)  # (1, 4, 2048)
