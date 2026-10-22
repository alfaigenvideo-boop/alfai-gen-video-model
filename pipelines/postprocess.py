import os
import logging
import cv2
import numpy as np
import subprocess
from PIL import Image
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class RIFEInterpolator:
    """
    Harici RIFE betiğini subprocess ile çalıştıran sınıf.
    """
    def __init__(self, config: dict):
        self.rife_path = config["rife_path"]
        self.exp = config.get("exp", 2) 
        self.fp16 = config.get("fp16", True)
        self.skip = config.get("skip", True)
        self.gpu_id = config.get("gpu_id", 0)

    def interpolate(self, input_video: str) -> str:
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Video not found: {input_video}")

        output_video = input_video.replace(".mp4", "_smooth.mp4")
        inference_script = os.path.join(self.rife_path, "inference_video.py")

        if not os.path.exists(inference_script):
            logger.warning(f"RIFE betiği bulunamadı: {inference_script}. İnterpolasyon atlanıyor.")
            return input_video

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        command = [
            "python", inference_script,
            "--video", input_video,
            "--output", output_video,
            "--exp", str(self.exp),
        ]

        if self.fp16:
            command.append("--fp16")
        if self.skip:
            command.append("--skip")

        logger.info(f"Running RIFE... Command: {' '.join(command)}")
        subprocess.run(command, check=True, env=env)

        if not os.path.exists(output_video):
            raise RuntimeError("Interpolation failed. Output file was not created.")

        logger.info(f"Interpolation completed: {output_video}")
        return output_video


class VideoPostProcessor:
    """
    Handles all post-processing steps for generated video frames.
    Pipeline: Upscale (PIL) -> Export to MP4 -> Interpolate (File) -> Final Video
    """

    def __init__(
        self, 
        output_dir: str = "outputs/videos",
        base_fps: int = 8,
        target_fps: int = 24,
        device: str = "cuda",
        rife_config: Optional[dict] = None
    ):
        self.output_dir = output_dir
        self.base_fps = base_fps
        self.target_fps = target_fps
        self.device = device
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # RIFE modeli config verildiyse initialize ediliyor
        self.interpolation_model = RIFEInterpolator(rife_config) if rife_config else None 
        self.upscale_model = None
        
        logger.info(f"VideoPostProcessor initialized. Output dir: {self.output_dir}, Target FPS: {self.target_fps}")

    def _upscale_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        if self.upscale_model is None:
            return frames
            
        logger.info("Upscaling frames...")
        # TODO: Upscale mantığı buraya eklenecek
        return frames

    def _export_to_mp4(self, frames: List[Image.Image], filename: str, fps: int) -> str:
        if not frames:
            raise ValueError("Frame list is empty. Cannot export video.")

        output_path = os.path.join(self.output_dir, filename)
        width, height = frames[0].size
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        
        logger.info(f"Encoding base video to {output_path} at {fps} FPS...")
        
        for pil_img in frames:
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            out.write(cv_img)
            
        out.release()
        return output_path

    def __call__(
        self, 
        frames: List[Image.Image], 
        filename: str = "output_video.mp4",
        apply_interpolation: bool = True,
        apply_upscale: bool = False
    ) -> Dict[str, Any]:
        logger.info("Starting post-processing pipeline...")
        
        processed_frames = frames
        
        # 1. Adım: PIL kareleri üzerinde çalışabilen işlemler (Upscale vb.)
        if apply_upscale:
            processed_frames = self._upscale_frames(processed_frames)
            
        # 2. Adım: Ham kareleri düşük FPS ile diske MP4 olarak kaydet
        raw_filename = filename.replace(".mp4", "_raw.mp4") if apply_interpolation else filename
        base_video_path = self._export_to_mp4(processed_frames, raw_filename, self.base_fps)
        
        final_video_path = base_video_path
        final_fps = self.base_fps
        
        # 3. Adım: Diskteki MP4 dosyası üzerinden interpolasyon yap (RIFE)
        if apply_interpolation:
            if self.interpolation_model:
                logger.info(f"Applying interpolation to {base_video_path}...")
                final_video_path = self.interpolation_model.interpolate(base_video_path)
                final_fps = self.target_fps
            else:
                logger.warning("Interpolation model config not provided. Skipping interpolation.")
        
        return {
            "video_path": final_video_path,
            "base_frame_count": len(processed_frames),
            "fps": final_fps,
            "resolution": processed_frames[0].size
        }