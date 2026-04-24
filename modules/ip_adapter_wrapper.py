import torch
import logging
from typing import Optional
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
# diffusers.models.attention_processor içinden IP-Adapter ağırlıklarını 
# UNet'e yükleyen yardımcı fonksiyonlar kullanılır (Diffusers IP-Adapter standardı)
from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0

logger = logging.getLogger(__name__)

class IPAdapterWrapper:
    """
    Karakter kimliğini (Face/Identity) korumak için IP-Adapter'ı 
    orijinal pipeline'ı bozmadan doğrudan UNet'e bağlayan köprü.
    """
    def __init__(
        self,
        unet: torch.nn.Module,
        ip_ckpt_path: str = "h94/IP-Adapter",
        subfolder: str = "sdxl_models",
        weight_name: str = "ip-adapter_sdxl.bin",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.device = device
        self.dtype = dtype
        self.unet = unet
        
        logger.info("IP-Adapter Vision Modeli ve Ağırlıkları Yükleniyor...")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=dtype
        ).to(device)
        self.clip_image_processor = CLIPImageProcessor()
        
        # Bu aşamada UNet'in Attention katmanlarına IP-Adapter işlemcilerini (Processors) ekleriz.
        # Not: Üretim (inference) kodunda Diffusers'ın unet.load_attn_procs metodunu sarmalıyoruz.
        try:
            self.unet.load_attn_procs(ip_ckpt_path, subfolder=subfolder, weight_name=weight_name)
            logger.info("IP-Adapter UNet Attention katmanlarına başarıyla bağlandı.")
        except Exception as e:
            logger.error(f"IP-Adapter yüklenirken hata: {e}")

    def set_scale(self, scale: float):
        """
        PID Controller'dan gelen güncel ip_scale değerini UNet'in içine işler.
        Böylece her karede kimlik koruma gücü dinamik olarak değişebilir.
        """
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                attn_processor.scale = scale

    @torch.no_grad()
    def get_image_embeds(self, pil_image) -> torch.Tensor:
        """
        Sadece videonun en başında (Frame 0 öncesi) 1 kez çalışır.
        Referans kimlik (Face) fotoğrafını tensörlere gömer. 
        Her döngüde hesaplamayarak VRAM/CPU zamanından büyük tasarruf sağlar.
        """
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.dtype)
        
        image_embeds = self.image_encoder(clip_image).image_embeds
        # SDXL IP-Adapter genellikle unconditionally embeddings (sıfır matrisi) ile birleştirilir
        uncond_image_embeds = torch.zeros_like(image_embeds)
        
        return torch.cat([uncond_image_embeds, image_embeds])
