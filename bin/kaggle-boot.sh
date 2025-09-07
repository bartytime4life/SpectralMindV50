#!/usr/bin/env bash
# =============================================================================
# SpectraMind V50 — Kaggle Bootstrap Helper
# =============================================================================
# Purpose:
#   Ensure Kaggle notebook/container has the required Python packages,
#   GPU stack, and repo setup for the NeurIPS 2025 Ariel Data Challenge.
#
# Usage:
#   ./bin/kaggle-boot.sh [--no-pyg] [--cpu-only]
#
# Options:
#   --no-pyg       Skip installing PyTorch Geometric stack
#   --cpu-only     Force CPU wheel (disable CUDA deps)
#
# Notes:
#   - Kaggle kernels come with CUDA/cuDNN preinstalled but limited libs.
#   - Always run this script at the start of a Kaggle notebook.
#   - Safe to rerun (idempotent installs).
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse flags
INSTALL_PYG=true
FORCE_CPU=false
for arg in "$@"; do
  case "$arg" in
    --no-pyg) INSTALL_PYG=false ;;
    --cpu-only) FORCE_CPU=true ;;
  esac
done

echo "[BOOT] Starting Kaggle environment setup..."

# -------------------------------------------------------------------------
# Core Python deps
# -------------------------------------------------------------------------
echo "[BOOT] Installing base Python dependencies..."
pip install -q --upgrade pip wheel setuptools

# Kaggle-safe requirements
REQ_FILE="$REPO_DIR/requirements-kaggle.txt"
if [[ -f "$REQ_FILE" ]]; then
  pip install -q -r "$REQ_FILE"
else
  echo "[WARN] requirements-kaggle.txt not found, skipping..."
fi

# -------------------------------------------------------------------------
# Torch & CUDA
# -------------------------------------------------------------------------
if $FORCE_CPU; then
  echo "[BOOT] Installing CPU-only PyTorch..."
  pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  echo "[BOOT] Installing CUDA-enabled PyTorch..."
  pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# -------------------------------------------------------------------------
# Optional PyTorch Geometric
# -------------------------------------------------------------------------
if $INSTALL_PYG; then
  echo "[BOOT] Installing PyTorch Geometric (PyG) stack..."
  TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  CUDA_VER=$(python -c "import torch; print(torch.version.cuda or 'cpu')")
  CUDA_TAG="${CUDA_VER//./}"
  if [[ "$CUDA_VER" == "cpu" ]]; then CUDA_TAG="cpu"; fi

  pip install -q \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html
else
  echo "[BOOT] Skipping PyG installation (--no-pyg)"
fi

# -------------------------------------------------------------------------
# DVC (with GDrive + S3 remotes)
# -------------------------------------------------------------------------
echo "[BOOT] Installing DVC with extras..."
pip install -q "dvc[gs,s3,gdrive]"

# -------------------------------------------------------------------------
# Verify installs
# -------------------------------------------------------------------------
echo "[BOOT] Checking versions..."
python - <<'EOF'
import torch, dvc, sys
print(f"Torch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"DVC:   {dvc.__version__}")
print("Python:", sys.version)
EOF

echo "[BOOT] ✅ Kaggle environment ready."
