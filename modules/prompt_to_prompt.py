from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from modules.interpolation.latent_interpolation import lerp, slerp_embeddings


class PromptBlender:
    """
    Video kareleri arasında prompt geçişlerini yönetir.

    İki katman:
      1. String-level mix (Compel/WebUI) — log ve CLIP loss metni için
      2. Embedding-level SLERP — gerçek sahne geçişi (CLIP token + SDXL pooled)
    """

    @staticmethod
    def mix_prompts(prompt_a: str, prompt_b: str, alpha: float) -> str:
        """
        İki prompt'u Compel/WebUI formatında yumuşak (soft) şekilde harmanlar.
        alpha=0.0 -> Tamamen prompt_a
        alpha=1.0 -> Tamamen prompt_b
        """
        # Alpha değerini 0 ile 1 arasına sabitle (güvenlik için)
        alpha = max(0.0, min(1.0, alpha))

        if alpha == 0.0:
            return prompt_a
        if alpha == 1.0:
            return prompt_b

        # Float değerleri okunaklı tutmak için .3f (3 ondalık) kullanıyoruz
        return f"({prompt_a}:{1.0 - alpha:.3f}), ({prompt_b}:{alpha:.3f})"

    @classmethod
    def generate_prompt_sequence(
        cls,
        num_frames: int,
        identity_prompt: str,
        prompt_a: str,
        prompt_b: str,
        p2p_start: int,
        p2p_end: int
    ) -> List[str]:
        """
        Tüm video için frame frame prompt listesi oluşturur.
        """
        prompts = []
        for t in range(num_frames):
            if t < p2p_start:
                scene_prompt = prompt_a
            elif p2p_start <= t <= p2p_end:
                # p2p_start ve p2p_end aynıysa sıfıra bölünme hatasını önle
                if p2p_end == p2p_start:
                    alpha = 1.0
                else:
                    alpha = (t - p2p_start) / (p2p_end - p2p_start)
                scene_prompt = cls.mix_prompts(prompt_a, prompt_b, alpha)
            else:
                scene_prompt = prompt_b

            full_prompt = f"{identity_prompt}, {scene_prompt}" if identity_prompt else scene_prompt
            prompts.append(full_prompt)

        return prompts

    @staticmethod
    def generate_ip_weights(num_frames: int) -> List[float]:
        """
        Frame sayısına göre IP-Adapter ağırlık takvimini (schedule) dinamik olarak oluşturur.
        Kimlik başta çok güçlü (0.9), ortada normal (0.6), sonda zayıftır (0.3).
        """
        weights = []
        for t in range(num_frames):
            ratio = t / max(1, (num_frames - 1)) # 0.0 ile 1.0 arası frame ilerlemesi

            if ratio < 0.33:       # İlk %33'lük kısım
                weights.append(0.9)
            elif ratio < 0.66:     # Orta %33'lük kısım
                weights.append(0.6)
            else:                  # Son %33'lük kısım
                weights.append(0.3)

        return weights

    # ------------------------------------------------------------------
    # Çoklu keyframe prompt SLERP
    # ------------------------------------------------------------------

    @staticmethod
    def apply_easing(t: float, mode: str = "smoothstep") -> float:
        t = max(0.0, min(1.0, t))
        if mode == "smoothstep":
            return t * t * (3.0 - 2.0 * t)
        if mode == "smootherstep":
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        return t

    @staticmethod
    def compose_prompt(scene_prompt: str, identity_prompt: str = "") -> str:
        identity_prompt = (identity_prompt or "").strip()
        scene_prompt = (scene_prompt or "").strip()
        if identity_prompt and scene_prompt:
            return f"{identity_prompt}, {scene_prompt}"
        return identity_prompt or scene_prompt

    @staticmethod
    def even_keyframes(num_frames: int, prompts: Sequence[str]) -> List[Dict[str, Any]]:
        """Prompt listesini 0 .. num_frames-1 aralığına eşit aralıkta yerleştirir."""
        cleaned = [p.strip() for p in prompts if p and str(p).strip()]
        if not cleaned:
            return []
        if len(cleaned) == 1:
            return [{"frame": 0, "prompt": cleaned[0]}]

        last = max(num_frames - 1, 0)
        n = len(cleaned)
        indices: List[int] = []
        for i in range(n):
            idx = round(i * last / (n - 1))
            if indices and idx <= indices[-1]:
                idx = min(indices[-1] + 1, last)
            indices.append(idx)
        return [{"frame": idx, "prompt": prompt} for idx, prompt in zip(indices, cleaned)]

    @staticmethod
    def normalize_keyframes(
        num_frames: int,
        keyframes: Sequence[Dict[str, Any]],
        identity_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Keyframe dict'lerini sıralar, indeksleri [0, num_frames) içine sıkıştırır.
        Aynı karede birden fazla prompt varsa sonuncu kazanır.
        """
        last = max(num_frames - 1, 0)
        by_frame: Dict[int, str] = {}
        for kf in keyframes:
            prompt = str(kf.get("prompt", "")).strip()
            if not prompt:
                continue
            frame = int(kf.get("frame", 0))
            frame = max(0, min(frame, last))
            by_frame[frame] = PromptBlender.compose_prompt(prompt, identity_prompt)

        ordered = [{"frame": frame, "prompt": by_frame[frame]} for frame in sorted(by_frame)]
        return ordered

    @classmethod
    def resolve_keyframes(
        cls,
        num_frames: int,
        prompt: Union[str, Sequence[str]],
        prompt_keyframes: Optional[Sequence[Dict[str, Any]]] = None,
        identity_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Tek prompt / prompt listesi / açık keyframe listesinden canonical schedule üretir.
        Öncelik: prompt_keyframes > prompt listesi > tek prompt.
        """
        if prompt_keyframes:
            return cls.normalize_keyframes(num_frames, prompt_keyframes, identity_prompt)

        if isinstance(prompt, (list, tuple)):
            even = cls.even_keyframes(num_frames, prompt)
            return cls.normalize_keyframes(num_frames, even, identity_prompt)

        single = (prompt or "").strip()
        if not single:
            return []
        return [{"frame": 0, "prompt": cls.compose_prompt(single, identity_prompt)}]

    @classmethod
    def segment_alpha(
        cls,
        frame_index: int,
        keyframe_indices: Sequence[int],
        easing: str = "smoothstep",
    ) -> Tuple[int, int, float]:
        """
        frame_index için (sol_keyframe_idx, sağ_keyframe_idx, eased_t) döner.
        İndeksler keyframe_indices içindeki konumdur, kare numarası değil.
        """
        if not keyframe_indices:
            return 0, 0, 0.0
        if len(keyframe_indices) == 1 or frame_index <= keyframe_indices[0]:
            return 0, 0, 0.0
        if frame_index >= keyframe_indices[-1]:
            last = len(keyframe_indices) - 1
            return last, last, 0.0

        right = 1
        while right < len(keyframe_indices) and keyframe_indices[right] < frame_index:
            right += 1
        left = right - 1
        span = keyframe_indices[right] - keyframe_indices[left]
        raw = 0.0 if span <= 0 else (frame_index - keyframe_indices[left]) / span
        return left, right, cls.apply_easing(raw, easing)

    @staticmethod
    def _interp_tensor(t: float, a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
        if t <= 0.0:
            return a.clone()
        if t >= 1.0:
            return b.clone()
        if mode == "lerp":
            return lerp(t, a, b)
        return slerp_embeddings(t, a, b)

    @classmethod
    def interpolate_cfg_embeds(cls, t: float, e0: torch.Tensor, e1: torch.Tensor, mode: str = "slerp") -> torch.Tensor:
        """
        CFG stacked embeds: (2, ...) = uncond + cond.
        Uncond (negative) sabit kalır; yalnızca cond SLERP/LERP edilir.
        """
        if e0.shape[0] < 2:
            return cls._interp_tensor(t, e0, e1, mode)

        uncond = e0[:1]
        cond = cls._interp_tensor(t, e0[1:], e1[1:], mode)
        return torch.cat([uncond, cond], dim=0)

    @classmethod
    def interpolate_added_cond(
        cls,
        t: float,
        a0: Dict[str, torch.Tensor],
        a1: Dict[str, torch.Tensor],
        mode: str = "slerp",
    ) -> Dict[str, torch.Tensor]:
        text0 = a0["text_embeds"]
        text1 = a1["text_embeds"]
        if text0.shape[0] >= 2:
            pooled = torch.cat(
                [text0[:1], cls._interp_tensor(t, text0[1:], text1[1:], mode)],
                dim=0,
            )
        else:
            pooled = cls._interp_tensor(t, text0, text1, mode)

        return {
            "text_embeds": pooled,
            "time_ids": a0["time_ids"].clone(),
        }

    @classmethod
    def interpolate_embedding_schedule(
        cls,
        keyframe_indices: Sequence[int],
        keyframe_prompt_embeds: Sequence[torch.Tensor],
        keyframe_added_cond: Sequence[Dict[str, torch.Tensor]],
        num_frames: int,
        mode: str = "slerp",
        easing: str = "smoothstep",
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Keyframe embedding'lerinden kare başına prompt_embeds + added_cond_kwargs listesi üretir.
        """
        prompt_schedule: List[torch.Tensor] = []
        added_schedule: List[Dict[str, torch.Tensor]] = []

        for frame_index in range(num_frames):
            i0, i1, t = cls.segment_alpha(frame_index, keyframe_indices, easing=easing)
            prompt_schedule.append(
                cls.interpolate_cfg_embeds(t, keyframe_prompt_embeds[i0], keyframe_prompt_embeds[i1], mode=mode)
            )
            added_schedule.append(
                cls.interpolate_added_cond(t, keyframe_added_cond[i0], keyframe_added_cond[i1], mode=mode)
            )

        return prompt_schedule, added_schedule

    @classmethod
    def interpolate_text_schedule(
        cls,
        keyframes: Sequence[Dict[str, Any]],
        num_frames: int,
        easing: str = "smoothstep",
    ) -> List[str]:
        """CLIP loss / log için kare başına metin (string mix)."""
        if not keyframes:
            return [""] * num_frames
        indices = [int(kf["frame"]) for kf in keyframes]
        texts = [str(kf["prompt"]) for kf in keyframes]
        out: List[str] = []
        for frame_index in range(num_frames):
            i0, i1, t = cls.segment_alpha(frame_index, indices, easing=easing)
            out.append(cls.mix_prompts(texts[i0], texts[i1], t))
        return out
