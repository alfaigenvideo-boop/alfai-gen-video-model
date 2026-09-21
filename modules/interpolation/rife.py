# modules/interpolation/rife.py
# RIFEInterpolator gerçek implementasyonu pipelines/postprocess.py içinde tanımlanmıştır.
# Bu dosya inference.py'nin `from modules.interpolation.rife import RIFEInterpolator`
# import'unun çalışması için bir yönlendirme (re-export) katmanıdır.

from pipelines.postprocess import RIFEInterpolator

__all__ = ["RIFEInterpolator"]
