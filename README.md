# 🎬 Alfai — SDXL Tabanlı Zamansal Video Üretim Sistemi

> **Hedef:** Stable Diffusion XL (SDXL) üzerine temporal consistency teknikleri katmanlayarak flicker'sız, kimlik ve sahne stabilitesi korunmuş **4 saniyelik kısa video sekansları** üretmek.

---

## 📋 İçindekiler
- [Mimari](#mimari)
- [Pipeline Akışı](#pipeline-akışı)
- [Bileşen Durumları](#bileşen-durumları)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Yol Haritası](#yol-haritası)

---

## Mimari

Alfai, zamansal tutarlılığı 9 ayrı tekniğin **katmanlı entegrasyonu** ile sağlar. Her katman, bir öncekinin bıraktığı artifact'ı azaltmak üzere konumlandırılmıştır.

```
┌─────────────────────────────────────────────────┐
│                  SDXL Base UNet                 │
│         (stabilityai/stable-diffusion-xl)       │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   Temporal Techniques   │
        │                         │
        │  1. DDIM Inversion      │  ← Keyframe latent anchor
        │  2. Latent SLERP        │  ← Smooth keyframe transitions
        │  3. ControlNet          │  ← Pose/Depth geometry lock
        │  4. IP-Adapter FaceID   │  ← Identity preservation
        │  5. Prompt-to-Prompt    │  ← Attention-level style control
        │  6. Optical Flow + Warp │  ← Pixel-level frame alignment
        │  7. TSM                 │  ← Temporal feature propagation
        │  8. Loss Guidance       │  ← CLIP + Perceptual + ID
        │  9. RIFE Post-process   │  ← Jitter correction
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │     Video Output        │
        │   outputs/videos/*.mp4  │
        └─────────────────────────┘
```

---

## Pipeline Akışı

### Kare-0 (I-frame / Keyframe)
```
Girdi Görseli
  → DDIM Inversion (görsel → latent uzayı)
  → IP-Adapter FaceID (kimlik ankrajı, w=0.9)          [⚠️ Yapım Aşamasında]
  → ControlNet Pose + Depth (geometri sabitleyici)      [⚠️ Yapım Aşamasında]
  → SDXL UNet + TSM (temporal shift aktif)
  → frame₀
```

### Sonraki Kareler (P-frame)
```
frame_{t-1}
  → RAFT Optical Flow (ileri + geri akış)
  → Latent Warp + Occlusion Mask
  → Residual Inpainting (sadece değişen bölgeler)
  → ControlNet (aktif)                                  [⚠️ Yapım Aşamasında]
  → IP-Adapter (azalan ağırlık, w azalıyor)             [⚠️ Yapım Aşamasında]
  → TSM katmanları (mid-block)
  → frame_{t}
  → Loss Check (CLIP + Perceptual + ID)
     └─ Eşik aşılırsa → LatentRefiner (Adam opt.)
```

### Post-process (İsteğe Bağlı)
```
[frame₀, frame₁, ..., frameₙ]
  → RIFE Frame Interpolation (jitter düzeltme)
  → VideoPostProcessor → output.mp4
```

---

## Bileşen Durumları

| # | Bileşen | Teknik | Durum | Notlar |
|---|---------|--------|-------|--------|
| 1 | **DDIM Inversion** | Partial DDIM (stop_at_t) | ✅ Tamamlandı | `modules/inversion/ddim_inversion.py` |
| 2 | **Latent İnterpolasyon** | LERP / SLERP / Spline | ❌ Eksik | Dosya hiç oluşturulmamış |
| 3 | **ControlNet** | Pose + Depth guidance | ❌ Eksik | Wrapper boş (`0 byte`) |
| 4 | **IP-Adapter FaceID** | FaceID w/ azalan ağırlık | ❌ Eksik | Wrapper boş (`0 byte`) |
| 5 | **Prompt-to-Prompt** | Soft prompt blending | ⚠️ Kısmi | String-level var; attention-level yok |
| 6 | **Optical Flow + Warp** | RAFT / GMFlow + Latent Warp | ✅ Tamamlandı | İleri-geri tutarlılık kontrolü dahil |
| 7 | **TSM** | Zero-Conv Temporal Shift | ✅ Tamamlandı | Mid-block only (doğru tasarım) |
| 8 | **Loss Guidance** | CLIP + VGG Perceptual + ResNet ID | ✅ Tamamlandı | ID loss için ArcFace önerilir |
| 9 | **RIFE Post-process** | Frame interpolation | ✅ Tamamlandı | Opsiyonel, default kapalı |

### Genel İlerleme: `5/9 bileşen tamamlandı`

```
✅✅❌❌⚠️✅✅✅✅
 1  2  3  4  5  6  7  8  9
```

---

## Kurulum

### Gereksinimler
- Python 3.10+
- CUDA 12.1 uyumlu GPU (minimum 6 GB VRAM önerilir; 4 GB için CPU offload aktif)
- PyTorch 2.5.1+cu121

### Kurulum Adımları

```bash
# 1. Ortam oluştur
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. (İsteğe bağlı) 4 GB VRAM için CPU offload ayarı
# configs/runtime.yaml içinde:
# enable_model_cpu_offload: true
```

> **Not:** SDXL ağırlıkları ilk çalıştırmada HuggingFace'ten otomatik indirilir (~6.5 GB).

---

## Kullanım

### Sanity Check (Hızlı Doğrulama Testi)

```bash
python scripts/sanity_check.py
```
- **Test 1:** TSM module forward pass (boyut koruması kontrolü)
- **Test 2:** Uçtan uca pipeline — `outputs/videos/test_output.mp4` üretir

### Video Üretimi

```bash
python main.py --prompt "A cinematic shot of a futuristic city at night, golden hour"
```

Çıktı: `outputs/videos/raw_output.mp4`

---

## Proje Yapısı

```
alfai-gen-video-model/
│
├── configs/                    # Konfigürasyon dosyaları
│   ├── paths.yaml              # Model ağırlık yolları
│   ├── runtime.yaml            # Cihaz & VRAM ayarları
│   └── sdxl_video.yaml         # Üretim parametreleri (fps, steps, vb.)
│
├── models/
│   └── unet/
│       └── unet_tsm_patch.py   # TSM'i SDXL UNet mid-block'a inject eder
│
├── modules/
│   ├── tsm.py                  # ZeroConvTSM implementasyonu
│   ├── inversion/
│   │   └── ddim_inversion.py   # DDIM Inversion & Reconstruction
│   ├── flow/
│   │   ├── raft_flow.py        # RAFT-Large optical flow
│   │   ├── gmflow_flow.py      # GMFlow alternatifi
│   │   └── latent_warp.py      # Latent warp + occlusion mask üretimi
│   ├── losses/
│   │   ├── clip_loss.py        # CLIP image-text similarity loss
│   │   ├── perceptual_loss.py  # VGG16 perceptual loss
│   │   ├── id_loss.py          # ResNet50 identity consistency loss
│   │   └── total_loss.py       # Ağırlıklı toplam loss + loglama
│   ├── refinement/
│   │   └── latent_refiner.py   # Adam tabanlı latent optimizasyonu
│   ├── interpolation/
│   │   └── rife.py             # RIFE post-process wrapper
│   ├── prompt_to_prompt.py     # Prompt blending & IP-Adapter ağırlık takvimi
│   ├── controlnet_wrapper.py   # [❌ Henüz yazılmadı]
│   └── ip_adapter_wrapper.py   # [❌ Henüz yazılmadı]
│
├── pipelines/
│   ├── video_pipeline.py       # SDXLVideoPipeline — ana pipeline __call__
│   ├── frame_generator.py      # Frame-by-frame ve TSM batch generation
│   └── postprocess.py          # VideoPostProcessor + RIFEInterpolator
│
├── inference.py                # VideoGenerator class (yükleme + encode_prompt)
├── main.py                     # CLI giriş noktası
├── requirements.txt
└── scripts/
    └── sanity_check.py         # TSM unit test + E2E pipeline testi
```

---

## Yol Haritası

### 🔴 Kritik (Çekirdek Özellik Eksikleri)
- [ ] `modules/controlnet_wrapper.py` — SDXL-ControlNet (pose + depth) implementasyonu
- [ ] `modules/ip_adapter_wrapper.py` — IP-Adapter FaceID entegrasyonu + azalan ağırlık takvimi
- [ ] `modules/interpolation/latent_interpolation.py` — SLERP / Spline keyframe interpolasyonu

### 🟡 İyileştirme (Mevcut Bileşenlerin Güçlendirilmesi)
- [ ] `modules/losses/id_loss.py` — ResNet50 yerine ArcFace/InsightFace backbone
- [ ] `modules/prompt_to_prompt.py` — Gerçek cross-attention map injection (Prompt2Prompt paper)
- [ ] `inference.py` — `null_prompt_embeds`'i DDIM Inversion'a doğru geçirmek

### 🟢 Teknik Borç / Opsiyonel
- [ ] `configs/interpolation.yaml` — RIFE model yolu konfigürasyonu
- [ ] VRAM < 6 GB için `enable_sequential_cpu_offload` otomasyonu
- [ ] Training script'in tamamlanması (`train/` dizini)

---

## Tasarım Kararları

| Karar | Neden |
|-------|-------|
| TSM mid-block only | Full-UNet TSM yüksek VRAM + veri ister; mid-block yeterli feature propagation sağlar |
| Frame-by-frame (varsayılan) | Batch temporal (full TSM) yerine daha az VRAM kullanır, aynı kalite |
| RAFT (GMFlow yerine varsayılan) | Daha stabil, sıkı geometrilerde daha az artifact |
| IP-Adapter > Prompt2Prompt (kimlik için) | Attention injection pahalı; IP-Adapter daha hafif ve etkili kimlik kilidi |
| LatentRefiner eşik tabanlı | Her kareyi optimize etmek çok yavaş; sadece kayıp eşiği aşınca tetikleme |

---

*Bu proje aktif geliştirme aşamasındadır.*
