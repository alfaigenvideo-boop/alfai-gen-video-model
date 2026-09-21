import os
import argparse
import logging
import cv2
import numpy as np
import insightface
from numpy.linalg import norm

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """İki embedding arasındaki kosinüs benzerliğini hesaplar."""
    return np.dot(a, b) / (norm(a) * norm(b))

def extract_and_save_embedding(image_path: str, output_path: str, compare_path: str = None):
    """Görselden yüz vektörünü çıkarır ve kaydeder."""
    
    if not os.path.exists(image_path):
        logger.error(f"Referans görsel bulunamadı: {image_path}")
        return

    logger.info("InsightFace 'buffalo_l' modeli yükleniyor...")
    # ctx_id=0 GPU'yu, -1 CPU'yu temsil eder.
    model = insightface.app.FaceAnalysis(name='buffalo_l')
    model.prepare(ctx_id=0) 
    
    # Görseli oku ve BGR -> RGB dönüşümü yap
    img1 = cv2.imread(image_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    logger.info(f"Yüzler aranıyor: {image_path}")
    faces = model.get(img1)
    
    if len(faces) == 0:
        logger.error("Görselde hiçbir yüz tespit edilemedi!")
        return
        
    # Birden fazla yüz varsa, bounding box alanına göre en büyüğünü (ana karakteri) seç
    if len(faces) > 1:
        logger.warning(f"{len(faces)} yüz bulundu. En büyük olan ana karakter olarak seçiliyor.")
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
        
    target_embedding = faces[0].embedding
    
    # Opsiyonel: İkinci bir görselle karşılaştırma (Test amaçlı)
    if compare_path and os.path.exists(compare_path):
        img2 = cv2.imread(compare_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        faces2 = model.get(img2)
        
        if len(faces2) > 0:
            compare_embedding = faces2[0].embedding
            sim = cosine_similarity(target_embedding, compare_embedding)
            logger.info(f"Karşılaştırma Benzerlik Skoru: {sim:.4f}")
            if sim >= 0.35:
                logger.info("✓ Aynı kişi")
            else:
                logger.info("× Farklı kişi")
    
    # --------------------------------------------------
    # ⭐ SDXL PIPELINE İÇİN EMBEDDING KAYDI
    # --------------------------------------------------
    np.save(output_path, target_embedding)
    logger.info(f"✅ Embedding başarıyla kaydedildi: {output_path}")
    logger.info(f"Embedding boyutu: {target_embedding.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDXL ID Loss için yüz embedding (kimlik) çıkarıcı.")
    parser.add_argument("--image", type=str, required=True, help="Referans yüz görselinin yolu")
    parser.add_argument("--output", type=str, default="face_emb.npy", help="Kaydedilecek .npy dosyasının yolu")
    parser.add_argument("--compare", type=str, default=None, help="(Opsiyonel) Karşılaştırılacak test görselinin yolu")
    
    args = parser.parse_args()
    
    extract_and_save_embedding(args.image, args.output, args.compare)