param(
  [string]$PythonPath = "C:\Users\${env:USERNAME}\AppData\Local\Programs\Python\Python310\python.exe",
  [switch]$CreateVenv = $true
)

$ErrorActionPreference = "Stop"
Write-Host ">>> SDXL Video ortam kurulumu (Windows)"

# 1) Venv
if ($CreateVenv) {
  & $PythonPath -m venv .venv
  Write-Host "Venv oluşturuldu."
}
$env:VENV_PATH = (Resolve-Path .\.venv\Scripts\Activate.ps1)
. $env:VENV_PATH
Write-Host "Venv aktif."

# 2) Pip güncelle
python -m pip install --upgrade pip

# 3) PyTorch (CUDA 12.1 önerisi)
Write-Host "PyTorch kuruluyor..."
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision --upgrade

# 4) Diğer gereklilikler
pip install -r requirements.txt

# 5) Klasörler
$dirs = @("models\sdxl\base","models\sdxl\refiner","models\controlnet\openpose","models\controlnet\depth","models\controlnet\hed","models\ip_adapter","models\vae","outputs\videos","outputs\images","logs\runs","data\inputs","data\videos_raw")
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

# 6) .env ve configs
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
if (-not (Test-Path "configs\paths.yaml"))   { Copy-Item "configs\paths.example.yaml" "configs\paths.yaml" }
if (-not (Test-Path "configs\runtime.yaml")) { Copy-Item "configs\runtime.example.yaml" "configs\runtime.yaml" }

Write-Host ">>> Kurulum tamam. 'scripts\download_models.py' ile ağırlıkları indirebilirsin."
