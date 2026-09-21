from typing import List

class PromptBlender:
    """
    Video kareleri (frameler) arasında prompt ve IP-Adapter ağırlık geçişlerini yönetir.
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
                
            full_prompt = f"{identity_prompt}, {scene_prompt}"
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