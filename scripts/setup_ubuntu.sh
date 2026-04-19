#!/usr/bin/env bash
set -euo pipefail

echo ">>> SDXL Video ortam kurulumu (Ubuntu)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

# 1) venv
if [ ! -d .venv ]; then
  $PYTHON_BIN -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) PyTorch (CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision --upgrade

# 3) Diğer gereklilikler
pip install -r requirements.txt

# 4) Klasörler
mkdir -p models/sdxl/base models/sdxl/refiner \
         models/controlnet/openpose models/controlnet/depth models/controlnet/hed \
         models/ip_adapter models/vae outputs/videos outputs/images logs/runs \
         data/inputs data/videos_raw

# 5) .env ve configs
[ -f .env ] || cp .env.example .env
[ -f configs/paths.yaml ]   || cp configs/paths.example.yaml configs/paths.yaml
[ -f configs/runtime.yaml ] || cp configs/runtime.example.yaml configs/runtime.yaml

echo ">>> Kurulum tamam. 'scripts/download_models.py' ile ağırlıkları indirebilirsin."
