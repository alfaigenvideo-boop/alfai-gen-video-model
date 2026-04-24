import logging
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

class FaceAnalyzer:
    """ArcFace kullanarak yüz kimliği (Identity) benzerliğini ölçer."""
    def __init__(self, model_name: str = "buffalo_l", det_size: tuple = (512, 512)):
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
    
    def similarity(self, reference: Image.Image, target: Image.Image) -> float:
        try:
            ref_faces = self.app.get(np.array(reference))
            target_faces = self.app.get(np.array(target))
            
            if len(ref_faces) == 0 or len(target_faces) == 0:
                logger.warning("Sensör Uyarısı: Görsellerden birinde yüz tespit edilemedi.")
                return 0.0
            
            emb1 = torch.tensor(ref_faces[0].embedding)
            emb2 = torch.tensor(target_faces[0].embedding)
            return torch.cosine_similarity(emb1, emb2, dim=0).item()
        except Exception as e:
            logger.error(f"Yüz benzerlik hesabı başarısız: {e}")
            return 0.0

class CLIPScorer:
    """Görüntünün metne (Prompt) sadakatini ölçer."""
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    @torch.no_grad()
    def similarity(self, image: Image.Image, text: str) -> float:
        try:
            inputs = self.processor(text=[text], images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            score = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
            return score.item()
        except Exception as e:
            logger.error(f"CLIP skor hesabı başarısız: {e}")
            return 0.0

class TemporalMetrics:
    """Kareler arası piksel bazlı sapmayı (Flickering/Titreme) ölçer."""
    @staticmethod
    def compute_residual(prev_frame: Image.Image, curr_frame: Image.Image) -> float:
        try:
            prev_np = np.array(prev_frame, dtype=np.uint8)
            curr_np = np.array(curr_frame, dtype=np.uint8)
            
            # RGB'den BGR/GRAY formatına OpenCV standartlarında çeviriyoruz
            prev = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(curr_np, cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            h, w = flow.shape[:2]
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            
            warped = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR)
            residual = np.mean(np.abs(warped - curr)) / 255.0
            return float(residual)
        except Exception as e:
            logger.error(f"Zamansal residual hesabı başarısız: {e}")
            return 1.0 # Hata durumunda maksimum titreme varsay
