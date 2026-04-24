import os
import torch
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class VideoDatasetLoader:
    def __init__(self, data_dir: str, device: str = "cuda", resolution: tuple = (1024, 1024)):
        self.data_dir = data_dir
        self.device = device
        self.resolution = resolution

    def _load_folder(self, folder_name: str, num_frames: int):
        path = os.path.join(self.data_dir, folder_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Klasör bulunamadı: {path}")
            
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:num_frames]
        logger.info(f"{folder_name} klasöründen {len(files)} resim yükleniyor...")
        
        tensor_list = []
        for f in files:
            img = Image.open(os.path.join(path, f)).convert("RGB").resize(self.resolution)
            arr = np.array(img).transpose(2, 0, 1)
            tensor = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
            tensor_list.append(tensor.to(self.device).half())
            
        while len(tensor_list) < num_frames:
            tensor_list.append(tensor_list[-1])
            
        return tensor_list

    def load_data(self, num_frames: int = 50):
        """Pipeline için gereken tüm resim ve tensörleri yükler."""
        logger.info("Dataset okunuyor...")
        
        # Identity (Referans Yüz)
        identity_path = os.path.join(self.data_dir, "frames", "0001.jpg")
        identity_image = Image.open(identity_path).convert("RGB")
        
        # Poses and Depths
        pose_list = self._load_folder("poses", num_frames)
        depth_list = self._load_folder("depths", num_frames)
        
        return identity_image, pose_list, depth_list
