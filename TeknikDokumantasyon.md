# Alfai — Teknik Dokümantasyon

Oct 26, 2026 · @alperen

## Genel bakış

Alfai, Stable Diffusion XL (SDXL) görüntü modelini kare kare çalıştırıp zamansal tutarlılık (temporal consistency) teknikleriyle birleştirerek 2–4 saniyelik kısa videolar üreten bir Python projesidir. Ayrı bir video modeli eğitmez; hazır SDXL ağırlıklarını kullanır ve kareler arası titremeyi optik akış, gizli uzay (latent) harmanlaması, kayıp yönlendirmesi (loss guidance) ve koşullandırma ağlarıyla bastırır.

İki üretim modu vardır:

- **Metinden videoya (T2V):** Yalnızca bir prompt verilir; kare-0 saf gürültüden üretilir.
- **Görselden videoya (I2V):** Bir referans fotoğraf verilir; kare-0 bu fotoğrafın DDIM ters çevirmesinden (DDIM inversion) başlar, böylece video fotoğrafa sadık kalır.

**Kapsam.** Bu belge GitHub'daki [alfaigenvideo-boop/alfai-gen-video-model](https://github.com/alfaigenvideo-boop/alfai-gen-video-model) deposunun 26 Ekim 2026 tarihli hâlini anlatır. Esas alınan dal `main`'dir (son commit `7c420e8`). `controlnet` dalındaki eklemeler ayrıca işaretlenmiştir; eski `hilal` dalı yalnızca dal yapısı bölümünde özetlenir. Kodun tamamı okunarak yazılmıştır; README'deki durum tablosu ile kod arasındaki farklar belirtilmiştir.

## Teknoloji yığını

Proje tamamen Python ile yazılmıştır ve PyTorch üzerinde Hugging Face `diffusers` kütüphanesine dayanır. Hiçbir model eğitilmez; aşağıdaki hazır ağırlıklar çıkarım (inference) sırasında indirilir ya da yerel dosyadan yüklenir.

### Yazılım

| Katman | Teknoloji | Sürüm | Projedeki görevi |
| --- | --- | --- | --- |
| Dil | Python | 3.10+ | Tüm kod tabanı |
| Derin öğrenme | PyTorch, torchvision | 2.5.1 + CUDA 12.1 (README) | Tensör işlemleri, GPU, hazır RAFT / VGG16 / ResNet50 |
| Difüzyon | diffusers | ≥ 0.30 (test edilen 0.35.2) | UNet, VAE, DDIM zamanlayıcı, ControlNet, T2I-Adapter sınıfları |
| Metin kodlayıcı | transformers | ≥ 4.45 | SDXL'in iki CLIP metin kodlayıcısı, CLIP kayıp modeli |
| Model yükleme | accelerate, safetensors, huggingface\_hub | ≥ 1.0 / ≥ 0.4.5 / ≥ 0.25 | CPU offload, ağırlık dosyaları, indirme |
| Yüz analizi | insightface + onnxruntime-gpu | ≥ 0.7.3 / ≥ 1.19 | ArcFace yüz gömmesi (IP-Adapter FaceID için) |
| Koşul haritaları | OpenCV, controlnet-aux | ≥ 4.10 / ≥ 0.0.9 | Canny kenar; derinlik, poz, çizgi haritası çıkarıcıları |
| Video G/Ç | imageio, imageio-ffmpeg, moviepy | ≥ 2.35 / ≥ 0.5.1 / ≥ 1.0.3 | MP4 yazma, sürücü video okuma |
| Yardımcı | numpy, scipy, einops, PyYAML, tqdm | — | Sayısal işlem, yapılandırma, ilerleme çubuğu |

### Kullanılan hazır modeller

| Model | Kaynak | Rolü |
| --- | --- | --- |
| SDXL Base 1.0 (UNet, VAE, 2 metin kodlayıcı) | `stabilityai/stable-diffusion-xl-base-1.0` | Ana üretici; UNet yerel `.safetensors` dosyasından da yüklenebilir |
| SDXL VAE (fp16 düzeltmeli) | `madebyollin/sdxl-vae-fp16-fix` | `download_models.py` ile indirilen yerel VAE |
| RAFT-Large | torchvision | Kareler arası optik akış |
| GMFlow | harici, henüz bağlı değil | RAFT alternatifi (sarıcı var, model yüklenmiyor) |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | Metin–görüntü uyum kaybı |
| VGG16 | torchvision | Algısal (perceptual) kayıp |
| ResNet50 | torchvision | Kimlik (ID) kaybı; ArcFace ile değiştirilmesi önerilmiş |
| IP-Adapter FaceID SDXL | `h94/IP-Adapter-FaceID` (`ip-adapter-faceid_sdxl.bin`) | Yüz kimliği koruma |
| InsightFace buffalo\_l | insightface | Referans fotoğraftan 512 boyutlu yüz gömmesi |
| T2I-Adapter SDXL (canny, depth, openpose, lineart, sketch) | `TencentARC/t2i-adapter-*-sdxl-1.0` | Hafif geometri kilidi |
| Annotator modelleri (MiDaS, OpenPose, Lineart, PiDiNet) | `lllyasviel/Annotators` | Koşul haritası çıkarma |
| ControlNet SDXL (depth-small, canny-small, openpose) | `diffusers/controlnet-*-sdxl-1.0-small`, `xinsir/controlnet-openpose-sdxl-1.0` | Adım bazlı geometri kilidi — **yalnızca `controlnet` dalında** |
| RIFE | harici depo (`./external/RIFE`) | İsteğe bağlı kare ara değerleme (interpolation); alt süreç (subprocess) olarak çağrılır |

## Mimari ve veri akışı

Video, video kodeklerindeki I-kare / P-kare fikriyle üretilir: kare-0 tam bir SDXL üretimidir, sonraki her kare bir öncekinin çarpıtılmış latentinden türetilir. Zamansal tutarlılığı asıl sağlayan adım vurgulanan harmanlamadır.

&#91;embedded content: kare kare üretim hattı · hazırlık, kare-0, P-kare döngüsü, çıktı\]

1. **Hazırlık.** Prompt, SDXL'in iki CLIP kodlayıcısıyla 77 × 2048 boyutlu gömmeye ve 1280 boyutlu havuzlanmış (pooled) gömmeye çevrilir; CFG için koşulsuz kopya eklenir. I2V'de referans görsel DDIM ters çevirmeyle t = 420 gürültü düzeyine götürülür; IP-Adapter için yüz gömmesi, T2I-Adapter / ControlNet için geometri haritası çıkarılır.
2. **Kare-0.** UNet, DDIM zamanlayıcısıyla (varsayılan 30 adım, CFG 5.5) gürültü giderir. I2V'de yalnızca t ≤ 420 olan adımlar çalışır, bu yüzden kare-0 referansa çok benzer.
3. **P-kare döngüsü.** Her kare için önceki kare ile hedef kare arasında RAFT akışı hesaplanır, önceki latent bu akışla çarpıtılır ve her gürültü giderme adımında maskenin dışında kalan bölgeye geri yazılır. Ardından CLIP, VGG16 ve ResNet50 kayıpları tahmini temiz görüntü üzerinden hesaplanır ve gradyanları latenti düzeltir. VAE türevi yerine 4 × 3'lük sabit bir latent→RGB matrisi kullanılır, böylece geri yayılım VRAM'e sığar.
4. **Çıktı.** Her latent, fp16 kararsızlığından kaçınmak için fp32'ye geçirilen VAE ile çözülür; kareler 8 fps MP4'e yazılır, isteğe bağlı olarak RIFE ile ara kareler eklenir.

**TSM toplu modu** (`enable_tsm=True`) bu döngüyü kullanmaz: tüm kareler tek bir (kare sayısı × 4 × H/8 × W/8) tensörde birlikte gürültüden arındırılır ve UNet orta bloğuna eklenen ZeroConvTSM, kanalların sekizde birini komşu karelerle paylaştırır. Akış, kayıp ve I2V bu modda yoktur.

## Modül başvuru kılavuzu

Kod dört katmanda düzenlenmiştir: giriş noktaları (`main.py`, `inference.py`), üretim hattı (`pipelines/`), teknik bileşenler (`modules/`) ve UNet yamaları (`models/unet/`). "Durum" sütunu dosyanın hatta gerçekten çağrılıp çağrılmadığını gösterir.

### Giriş noktaları ve hat

| Dosya | Ne yapar | Durum |
| --- | --- | --- |
| `main.py` | Komut satırı argümanlarını okur, üç YAML dosyasını birleştirir, `VideoGenerator`'ı çalıştırır | Aktif |
| `inference.py` → `VideoGenerator` | Metin kodlayıcıları, UNet'i, VAE'yi, zamanlayıcıyı yükler; prompt'u SDXL'in iki kodlayıcısıyla kodlar; I2V'de DDIM ters çevirme, yüz gömmesi ve T2I artıklarını hazırlar; MP4 yazar, isteğe bağlı RIFE çalıştırır | Aktif |
| `pipelines/video_pipeline.py` → `SDXLVideoPipeline` | `diffusers.DiffusionPipeline` alt sınıfı. İki mod: kare kare (akış + kayıp) ve TSM toplu (batch). IP-Adapter, T2I-Adapter ve (dalda) ControlNet'i yükler; VAE'yi fp32'de çözer | Aktif |
| `pipelines/frame_generator.py` → `generate_next_frame` | Tek bir P-kareyi üretir: her adımda çarpıtılmış önceki latent ile harmanlama, CFG'li UNet tahmini, kayıp yönlendirmeli gradyan adımı | Aktif |
| `pipelines/postprocess.py` | `RIFEInterpolator` (harici RIFE betiklerini alt süreçle çağırır) ve `VideoPostProcessor` (MP4 dışa aktarımı) | RIFE aktif, varsayılan kapalı |

### Teknik bileşenler (`modules/`)

| Dosya | Ne yapar | Durum |
| --- | --- | --- |
| `inversion/ddim_inversion.py` | Görüntüyü VAE ile latente çevirir, DDIM adımlarını tersine çalıştırarak `stop_at_t` (420) gürültü düzeyine kadar götürür | Aktif (I2V) |
| `flow/raft_flow.py`, `flow/gmflow_flow.py`, `flow/flow_factory.py` | Optik akış tahmini ve akışı latent çözünürlüğüne ölçekleme; fabrika (factory) deseni | RAFT aktif; GMFlow model yüklemiyor |
| `flow/latent_warp.py` → `LatentWarper` | Önceki karenin latentini akışa göre çarpıtır, sınır ve ileri–geri tutarlılık maskesi üretir | Aktif (geri akış verilmediği için yalnız sınır maskesi) |
| `interpolation/latent_interpolation.py` | LERP, SLERP, token bazında SLERP ve `noise_blend_schedule` (I2V P-kare başlangıç latentleri) | Aktif |
| `prompt_to_prompt.py` → `PromptBlender` | Prompt metin karışımı, anahtar kare (keyframe) SLERP takvimi, IP-Adapter ağırlık takvimi (0.9 → 0.6 → 0.3) | Ağırlık takvimi aktif; prompt SLERP yalnız `controlnet` dalında bağlı |
| `ip_adapter_wrapper.py` | `FaceIDExtractor` (InsightFace, CPU), `MLPProjModel` (512 → 4 × 2048 token), `IPAttnProcessor2_0` (70 çapraz dikkat katmanına enjekte) | Aktif, config ile açılır |
| `t2i_adapter_wrapper.py` | `ConditionPreprocessor` (canny/derinlik/poz/çizgi/eskiz) ve `T2IAdapterWrapper` (artıkları bir kez hesaplar, CFG ve toplu boyuta çoğaltır) | Var, config'de kapalı |
| `controlnet_wrapper.py` | `main`'de boş (0 bayt). `controlnet` dalında: çoklu ControlNet, adım penceresi, guess mode, sürücü video, EMA yumuşatma | Yalnızca dalda |
| `losses/` (`clip_loss`, `perceptual_loss`, `id_loss`, `total_loss`, `loss_functions`) | CLIP metin uyumu (ağırlık 1.0), VGG16 algısal (0.5), ResNet50 kimlik (0.2) kayıpları ve bunları toplayan `TotalLoss` | Aktif (P-karelerde) |
| `refinement/latent_refiner.py` → `LatentRefiner` | Kayıp eşiği aşılınca latenti Adam ile optimize eder | Yazılmış, hatta çağrılmıyor (yalnız sanity check) |
| `tsm.py` → `ZeroConvTSM` | Kanalların 1/8'ini önceki, 1/8'ini sonraki kareye kaydıran zamansal kaydırma; sıfır başlatılmış 1×1 evrişim | Aktif (yalnız TSM modunda) |
| `optical_flow.py`, `schedulers.py` | Boş dosyalar | Kullanılmıyor |

### UNet ve yardımcılar

| Dosya | Ne yapar | Durum |
| --- | --- | --- |
| `models/unet/unet_base.py` | diffusers UNet'inin özel kopyası; ağırlıklar resmi UNet'ten aktarılır (1680 anahtarın tamamı eşleşir) | Aktif, **depoda yok** (`.gitignore`) |
| `models/unet/unet_tsm_patch.py` | `ZeroConvTSM`'i UNet orta bloğuna (mid-block) enjekte eder | Aktif, **depoda yok** |
| `utils/model_utils.py`, `utils/video_utils.py` | Cihaz seçimi (CUDA > MPS > CPU), VRAM temizleme, MP4/GIF yazma | Aktif |
| `scripts/` | `sanity_check.py` (TSM birim testi + uçtan uca test), `download_models.py`, `prepare_face_embedding.py`, kurulum betikleri | Yardımcı |
| `evaluation/`, `train/` | Dosyalar oluşturulmuş ama içleri boş | Yapılmadı |

## Yapılandırma

Tüm ayarlar `configs/` altındaki dört YAML dosyasındadır. `main.py` ilk üçünü tek bir sözlükte birleştirir; `interpolation.yaml`'ı `inference.py` ayrıca okur.

| Dosya | Anahtar | Varsayılan | Anlamı |
| --- | --- | --- | --- |
| `paths.yaml` | `paths.base_dir` | `/home/hpc/Desktop/sdxl-video-generator` | Model yollarının kökü. `main` dalında `${paths.base_dir}` referansı çözülmez; yerel model dosyası bulunamayınca Hugging Face'e düşülür |
| `paths.yaml` | `models.sdxl_base`, `models.sdxl_vae` | `…/models/sdxl/base/minimal/sd_xl_base_1.0.safetensors`, `…/models/sdxl_vae/minimal/sdxl_vae.safetensors` | Yerel UNet ve VAE dosyaları |
| `runtime.yaml` | `runtime.device`, `precision`, `seed` | `cuda`, `fp16`, `42` | Cihaz, sayı hassasiyeti, tekrarlanabilirlik |
| `runtime.yaml` | `enable_model_cpu_offload` | `true` | UNet ve VAE'yi kullanılmadıklarında RAM'e taşır |
| `runtime.yaml` | `ip_adapter.enabled`, `scale` | `true`, `0.8` | Yüz kimliği kilidi ve gücü |
| `runtime.yaml` | `t2i_adapter.enabled`, `type`, `scale`, `source` | `false`, `canny`, `0.8`, `hybrid` | Hafif geometri kilidi; `hybrid` = kare-0 referans, sonrakiler önceki kare |
| `runtime.yaml` | `controlnet.*` (yalnız dalda) | `depth` / `small`, `scale 0.5`, pencere `0.0–0.8`, `source auto` | Adım bazlı geometri kilidi |
| `sdxl_video.yaml` | `width`, `height` | `1024`, `576` | Çıktı çözünürlüğü (16:9) |
| `sdxl_video.yaml` | `num_frames`, `fps` | `24`, `8` | 3 saniyelik video |
| `sdxl_video.yaml` | `steps`, `guidance_scale` | `30`, `5.5` | Gürültü giderme adımı, sınıflandırıcısız yönlendirme (CFG) gücü |
| `sdxl_video.yaml` | `scheduler.type` | `euler_ancestral` | **Kullanılmıyor:** kod her zaman DDIM yükler |
| `sdxl_video.yaml` | `flow_blend_end` (yalnız dalda) | `1.0` | P-karede akış harmanlamasının sürdüğü adım oranı |
| `sdxl_video.yaml` | `negative_prompt` | uzun bir liste | Bulanıklık, bozuk yüz, titreme vb. |
| `interpolation.yaml` | `enabled`, `exp`, `rife_path` | `true`, `3`, `./external/RIFE` | RIFE ile 2³ = 8 kat kare artırımı; RIFE deposu projede yok, hata olursa sessizce atlanır |

## Kurulum ve çalıştırma

Proje CUDA destekli bir NVIDIA ekran kartı ve CUDA'lı PyTorch ister; PyTorch'un CPU sürümüyle hat açılışta durur.

1. **Ortam:** `python -m venv venv`, ardından etkinleştirme (Windows: `.\venv\Scripts\Activate.ps1`).
2. **PyTorch:** CUDA'lı sürüm `requirements.txt`'den önce, ayrı kurulur: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` (sürücüye göre `cu126` de çalışır).
3. **Bağımlılıklar:** `pip install -r requirements.txt`.
4. **Modeller:** `python scripts/download_models.py` SDXL UNet'i, fp16 VAE'yi ve canny T2I-Adapter'ı `models/` altına indirir (\~7 GB). Metin kodlayıcılar, CLIP, RAFT, VGG16, ResNet50 ve IP-Adapter ilk çalıştırmada otomatik iner (\~5 GB daha).
5. **Özel UNet dosyaları:** `models/unet/unet_base.py` ve `unet_tsm_patch.py` depoda bulunmadığı için ekipten ayrıca alınmalıdır; yoksa `main.py` içe aktarmada durur.
6. **Doğrulama:** `python scripts/sanity_check.py` (TSM ileri geçiş testi + uçtan uca üretim).

### Komut satırı

```
python main.py -p "<prompt>" [-r referans.jpg] [--fast] [--steps N]
```

| Seçenek | Etkisi |
| --- | --- |
| `-p`, `--prompt` | Üretilecek videonun metni |
| `-r`, `--ref` | Referans fotoğraf; verilirse I2V moduna geçer. Dosya bulunamazsa uyarı verip T2V'ye döner |
| `--fast` | 4 kare, 5 adım, 512×512. Yalnızca hattın çalışıp çalışmadığını sınamak içindir; SDXL bu ayarda kullanılabilir görüntü üretmez |
| `--steps` | Adım sayısını yapılandırmadan bağımsız belirler |
| `-c`, `--control-video` (yalnız dalda) | ControlNet için hareket kaynağı video |
| `--prompt-end` (yalnız dalda) | Son karenin prompt'u; ilk kareden son kareye SLERP ile geçiş |

Çıktı: `outputs/raw_output.mp4` (RIFE çalışırsa ayrıca ara değerlenmiş sürüm).

## Dal yapısı ve bileşen durumları

Depoda üç dal vardır ve etkin geliştirme `main` üzerindedir. `controlnet` dalı bilerek `main`'e birleştirilmemiştir: ek modeller 6 GB VRAM'li makinelerde çalışmayı zorlaştırıyor.

| Dal | Son commit | İçerik |
| --- | --- | --- |
| `main` | `7c420e8` — 22 Ekim 2026 | Etkin hat: kare kare üretim, I2V, IP-Adapter, T2I-Adapter, prompt SLERP modülü, RIFE |
| `controlnet` | `0ed1e3c` — 26 Ekim 2026 | `main` + ControlNet, `--prompt-end` prompt geçişi, `flow_blend_end`, yol çözümü, fp16 UNet ve SDXL VAE ayarı düzeltmeleri, metin kodlayıcı boşaltma, TSM CFG hatası düzeltmesi |
| `hilal` | `44b9174` — 24 Nisan 2026 | `main`'den Nisan 2026'da ayrılmış deneysel dal. PID denetleyicili uyarlanabilir zamanlayıcı: yüz benzerliği (ArcFace), CLIP skoru ve kareler arası artık (residual) ölçülüp IP-Adapter ve ControlNet ağırlıkları kare kare ayarlanır. Sonraki `main` geliştirmelerini içermez |

### Bileşen durumu: README ile kod karşılaştırması

README'deki tablo 5/9 tamamlanmış der; kod buna göre daha ileridedir.

| # | Bileşen | README | `main`'deki gerçek durum |
| --- | --- | --- | --- |
| 1 | DDIM ters çevirme | Tamamlandı | Tamamlandı; zaman kimlikleri (time\_ids) ve boş prompt sıfır tensör veriliyor |
| 2 | Latent ara değerleme | Eksik | Yazılmış (SLERP, LERP, gürültü harmanlama takvimi) ve hatta bağlı |
| 3 | ControlNet | Eksik | `main`'de boş; `controlnet` dalında tamamlandı |
| 4 | IP-Adapter FaceID | Eksik | Yazılmış ve bağlı; azalan ağırlık takvimi var |
| 5 | Prompt-to-Prompt | Kısmi | Metin ve gömme düzeyinde SLERP var; dikkat haritası enjeksiyonu yok |
| 6 | Optik akış + çarpıtma | Tamamlandı | RAFT bağlı; geri akış hesaplanmadığı için tıkanma (occlusion) maskesi pratikte çalışmıyor |
| 7 | TSM | Tamamlandı | Var; yalnız toplu modda, `main`'de bu mod CFG boyut hatasıyla çöküyor |
| 8 | Kayıp yönlendirmesi | Tamamlandı | Bağlı; CLIP kaybına boş metin veriliyor |
| 9 | RIFE | Tamamlandı | Sarıcı var; RIFE deposu projede yok |
| — | T2I-Adapter | README'de yok | Yazılmış, varsayılan kapalı |

## Donanım, sınırlamalar ve teknik borç

Hat, SDXL'in yanına altı yardımcı model (RAFT, CLIP, VGG16, ResNet50, IP-Adapter, InsightFace; dalda ControlNet ile yedi) yüklediği için 6 GB VRAM'de çalışır ama çok yavaştır. RTX 4050 Laptop (6 GB) üzerinde ölçülen fark: düz SDXL 768×768'de adım başına 1,2 saniye; tüm yardımcı modellerle hat adım başına 26 saniye. Windows, VRAM'e sığmayan kısmı hata vermeden sistem RAM'ine aktarır ve her adım yirmi kat yavaşlar.

### Donanım

| Kaynak | En az | Önerilen | Not |
| --- | --- | --- | --- |
| GPU | NVIDIA, 6 GB VRAM | 12 GB ve üzeri | 6 GB'ta CPU offload ve yardımcı modellerin kapatılması gerekir |
| RAM | 16 GB | 32 GB | Model yüklenirken tepe kullanım \~12 GB; Windows'ta disk belleği dosyası 16 GB'a kadar büyütülmeli |
| Disk | \~20 GB boş | 40 GB | Modeller \~12 GB + venv \~6 GB; Hugging Face önbelleği (`HF_HOME`) boş alanı olan sürücüye yönlendirilmeli |

### Bilinen sınırlamalar (`main`)

1. **P-kareler önceki karenin kopyasına dönüşür.** Akış maskesi geri akış olmadan hesaplandığı için neredeyse tamamen "koru" çıkar; her adımda önceki latent geri yazılır. Ölçüm: prompt değişse bile son kare 0,57/255 değişir. Videolar bu yüzden durgundur.
2. **I2V'de gerçek bir hareket kaynağı yok.** `target_driving_frames` referans görselin 24 kopyasıdır; akış her kareyi referansa geri çeker.
3. **Yerel model yolu çözülmez.** `${paths.base_dir}` YAML tarafından genişletilmediği için yerel UNet bulunamaz ve Hugging Face'ten fp32 (\~10 GB) indirme denenir.
4. **Yerel VAE yanlış ayarla yüklenir.** `from_single_file` SD 1.5 ayarını tahmin eder; ölçekleme katsayısı 0,18215 olur (SDXL: 0,13025) ve renkler bozulur.
5. **TSM toplu modu çöker.** CFG'li embed'ler iki kez çoğaltıldığı için latent (2T) ile embed (4T) boyutları uyuşmaz.
6. **Yapılandırmanın bir kısmı yok sayılır.** `scheduler.type` okunmaz (her zaman DDIM); `--fast` modu SDXL için fazla küçüktür ve soyut desenler üretir.
7. **Metin kodlayıcılar (\~1,6 GB) ve kayıp modelleri sürekli GPU'da durur;** kayıp modelleri her P-kare adımında geri yayılım yapar.

Madde 3, 4, 5 ve 7'nin metin kodlayıcı kısmı `controlnet` dalında düzeltilmiştir; madde 1 için `flow_blend_end` ayarı eklenmiştir (varsayılan eski davranış).

### Teknik borç

- [ ] `models/unet/*.py` dosyalarını depoya eklemek; `.gitignore` düzeltmesi `controlnet` dalında hazır
- [ ] Geri akışı hesaplayıp tıkanma maskesini çalışır hâle getirmek
- [ ] DDIM ters çevirmeye gerçek `time_ids` ve boş prompt gömmesini vermek
- [ ] CLIP kaybına gerçek prompt metnini geçirmek (şu an `""`)
- [ ] Kimlik kaybında ResNet50 yerine ArcFace kullanmak
- [ ] Kayıp yönlendirmesini yapılandırmadan açılıp kapatılabilir yapmak
- [ ] `evaluation/` (CLIP skoru, kimlik skoru, zamansal tutarlılık) ve `train/` dosyalarını doldurmak
- [ ] RIFE deposunu projeye eklemek ya da `interpolation.enabled`'ı `false` yapmak
