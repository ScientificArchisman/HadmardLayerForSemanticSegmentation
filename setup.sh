#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-torch_env}"
PY_VER="3.10"
CUDA_SPEC_DEFAULT="cu121"   

echo "==> Setting up Conda env '${ENV_NAME}' with Python ${PY_VER}"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: 'conda' not found. Install Miniconda/Anaconda and re-run." >&2
    exit 1
  fi
fi


eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Conda env '${ENV_NAME}' already exists; skipping creation."
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

conda activate "${ENV_NAME}"
echo "==> Activated env '${ENV_NAME}'"

python -m pip install --upgrade pip wheel setuptools

OS_NAME="$(uname -s)"
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/dev/null 2>&1; then
    HAS_NVIDIA=1
  fi
fi

echo "==> Detecting platform for PyTorch:"
echo "    OS: ${OS_NAME}"
echo "    NVIDIA detected: ${HAS_NVIDIA}"

CUDA_SPEC="${CUDA_SPEC:-$CUDA_SPEC_DEFAULT}"

if [[ "$OS_NAME" == "Darwin" ]]; then
  echo "==> Installing PyTorch (macOS, MPS capable wheels)"
  python -m pip install torch torchvision torchaudio
elif [[ "$HAS_NVIDIA" -eq 1 ]]; then
  echo "==> Installing PyTorch (CUDA wheels: ${CUDA_SPEC})"
  python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_SPEC}"
else
  echo "==> Installing PyTorch (CPU-only wheels)"
  python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
fi

echo "==> Installing project libraries: matplotlib, segmentation-models-pytorch, timm"
python -m pip install matplotlib timm segmentation-models-pytorch

echo "==> Verifying installation…"
python - <<'PYCHECK'
import sys, torch
print(f"Python: {sys.version.split()[0]}")
print(f"Torch : {torch.__version__}")
print(f"CUDA  : {torch.cuda.is_available()} (num devices: {torch.cuda.device_count()})")
if sys.platform == "darwin":
    print(f"MPS   : {getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}")
import torchvision, matplotlib
print(f"torchvision: {torchvision.__version__}")
print(f"matplotlib : {matplotlib.__version__}")
try:
    import segmentation_models_pytorch as smp
    print(f"smp      : {getattr(smp, '__version__', 'ok')}")
except Exception as e:
    print("smp import failed:", e)
PYCHECK

echo ""
echo "✅ Setup complete."
echo "👉 To activate later: 'conda activate ${ENV_NAME}'"
