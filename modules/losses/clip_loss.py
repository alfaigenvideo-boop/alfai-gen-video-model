import logging
import torch
import torch.nn.functional as F
from typing import List

from transformers import CLIPModel, CLIPProcessor
from .base_loss import BaseLoss

logger = logging.getLogger(__name__)


class CLIPLoss(BaseLoss):
    """
    CLIP image-text similarity loss.
    Uses HuggingFace transformers CLIPModel (openai/clip-vit-base-patch32).
    """

    def __init__(self, device: str = "cuda", weight: float = 1.0):
        super().__init__(weight)
        self.device = device

        model_id = "openai/clip-vit-base-patch32"
        logger.info(f"Loading CLIP model '{model_id}' via transformers on {self.device}...")

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id, revision="refs/pr/66")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Freeze CLIP — sadece input gradyanı istiyoruz, CLIP'i eğitmiyoruz
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("CLIP model loaded and frozen successfully.")

    def forward(self, pred: torch.Tensor, target: List[str]) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Tahmin edilen görseller (B, 3, H, W), [0, 1] aralığında.
            target (List[str]): Hedef text prompt listesi.
        Returns:
            torch.Tensor: Cosine distance loss.
        """
        if pred.device.type != self.device:
            pred = pred.to(self.device)

        # Görseli PIL formatına dönüştür (processor için)
        # pred: (B, 3, H, W) float [0,1] → uint8
        pred_uint8 = (pred.clamp(0, 1) * 255).byte().cpu()
        pil_images = [
            torch.nn.functional.interpolate(
                pred_uint8[i].unsqueeze(0).float(), size=(224, 224), mode="bilinear"
            ).squeeze(0).byte()
            for i in range(pred_uint8.shape[0])
        ]

        # Tokenize text
        text_inputs = self.processor(
            text=target,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # Görsel features (processor yerine manuel normalizasyon)
        # (224,224) resize uygulayıp [0,1]→[-1,1] normalize
        pixel_values = pred.to(self.device)
        if pixel_values.shape[-2:] != (224, 224):
            pixel_values = F.interpolate(pixel_values, size=(224, 224), mode="bilinear", align_corners=False)
        # CLIP normalization: mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        pixel_values = (pixel_values - mean) / std

        # Encode (Modelin ana çıktısından sadece saf embedding tensörlerini alıyoruz)
        outputs = self.model(pixel_values=pixel_values, **text_inputs)
        
        img_feat = outputs.image_embeds
        txt_feat = outputs.text_embeds

        # Artık elimizde saf tensörler olduğu için normalize işlemi çökmeyecek
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        loss = 1.0 - (img_feat * txt_feat).sum(dim=-1).mean()

        return self.weight * loss