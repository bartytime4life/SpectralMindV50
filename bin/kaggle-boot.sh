#!/usr/bin/env bash
# =============================================================================
# SpectraMind V50 — Kaggle Bootstrap Helper (Upgraded)
# =============================================================================
# Purpose:
#   Prepare a Kaggle notebook/container with the required Python stack for the
#   NeurIPS 2025 Ariel Data Challenge — fast, idempotent, and CUDA-aware.
#
# Usage:
#   ./bin/kaggle-boot.sh [--no-pyg] [--cpu-only] [--torch <ver>] [--quiet]
#
# Options:
#   --no-pyg         Skip installing PyTorch Geometric stack
#   --cpu-only       Force CPU wheels (skip CUDA entirely)
#   --torch <ver>    Target torch version (e.g., 2.3.1). Defaults: keep existing
#   --quiet          Reduce pip verbosity
#
# Behavior:
#   - Detects Kaggle runtime, GPU, CUDA version and selects the correct wheels.
#   - Uses a pip cache in /kaggle/working/.cache to speed re-runs.
#   - Retries flaky installs; safe to re-run (idempotent where possible).
#   - If internet is disabled, exits early unless all deps already present.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ----------------------------- CLI FLAGS -------------------------------------
INSTALL_PYG=true
FORCE_CPU=false
QUIET=false
TARGET_TORCH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pyg) INSTALL_PYG=false; shift ;;
    --cpu-only) FORCE_CPU=true; shift ;;
    --torch) TARGET_TORCH="${2:-}"; shift 2 ;;
    --quiet) QUIET=true; shift ;;
    *) echo "[BOOT] Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------- PRINT UTILS ------------------------------------
c_cyan='\033[1;36m'; c_yel='\033[1;33m'; c_red='\033[1;31m'; c_grn='\033[1;32m'; c_off='\033[0m'
banner(){ printf "${c_cyan}\n==> %s${c_off}\n" "$1"; }
note(){ printf "${c_yel}[-] %s${c_off}\n" "$1"; }
ok(){ printf "${c_grn}[OK] %s${c_off}\n" "$1"; }
err(){ printf "${c_red}[ERR] %s${c_off}\n" "$1"; }

banner "SpectraMind V50 — Kaggle Bootstrap"

# --------------------------- KAGGLE CONTEXT -----------------------------------
IS_KAGGLE=0
if [[ -d /kaggle && -d /kaggle/working ]]; then IS_KAGGLE=1; fi
note "Kaggle runtime detected: ${IS_KAGGLE}"

# Enforce a local pip cache on Kaggle to speed restarts
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/kaggle/working/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR"

# Simple internet check (Kaggle often runs with internet disabled)
INTERNET_OK=0
if python - <<'PY' >/dev/null 2>&1; then
import socket
socket.gethostbyname("pypi.org")
print("ok")
PY
then INTERNET_OK=1; else INTERNET_OK=0; fi

if [[ $INTERNET_OK -eq 0 ]]; then
  note "Internet appears disabled. Will verify existing installs and exit if satisfied."
fi

# ---------------------------- HELPERS ----------------------------------------
PIP_QUIET=()
$QUIET && PIP_QUIET+=(-q)

pip_retry() {
  # pip_retry <args...>
  local n=0; local max=3
  until pip install "${PIP_QUIET[@]}" "$@"; do
    n=$((n+1))
    if [[ $n -ge $max ]]; then err "pip install failed after ${max} attempts: $*"; return 1; fi
    note "pip retry $n/$max for: $*"; sleep 2
  done
}

have_py() { python - "$@" >/dev/null 2>&1; }

torch_ready() {
  # torch_ready <cpu|cuda>
  python - "$1" <<'PY'
import sys, torch
want = sys.argv[1]
if want == "cpu":
    ok = not torch.cuda.is_available()
else:
    ok = torch.cuda.is_available()
sys.exit(0 if ok else 1)
PY
}

pyg_needed() {
  python - <<'PY'
import importlib
mods = ["torch_geometric","torch_scatter","torch_sparse","torch_cluster","torch_spline_conv"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
sys.exit(0 if not missing else 1)
PY
}

# ---------------------------- BASELINE TOOLS ---------------------------------
banner "Upgrade pip/setuptools/wheel"
if [[ $INTERNET_OK -eq 1 ]]; then
  pip_retry --upgrade pip wheel setuptools
else
  note "Offline: skipping pip bootstrap"
fi

# Install repo-pinned Kaggle requirements if present
REQ_FILE="$REPO_DIR/requirements-kaggle.txt"
if [[ -f "$REQ_FILE" ]]; then
  banner "Installing requirements-kaggle.txt"
  if [[ $INTERNET_OK -eq 1 ]]; then
    pip_retry -r "$REQ_FILE" || { err "Failed to install requirements-kaggle.txt"; exit 1; }
  else
    note "Offline: cannot install $REQ_FILE"
  fi
else
  note "requirements-kaggle.txt not found; continuing"
fi

# ------------------------------- TORCH ---------------------------------------
banner "Torch / CUDA setup"

# If torch is already installed and meets CPU/GPU intent, keep it unless user pinned a version
KEEP_EXISTING=false
if have_py 'import torch'; then
  EXISTING_VER="$(python - <<'PY'
import torch; print(getattr(torch, "__version__", ""), end="")
PY
)"
  EXISTING_CUDA="$(python - <<'PY'
import torch; print(torch.version.cuda or "cpu", end="")
PY
)"
  note "Found torch ${EXISTING_VER} (CUDA ${EXISTING_CUDA})"
  if [[ -z "$TARGET_TORCH" ]]; then
    if $FORCE_CPU; then
      torch_ready cpu && KEEP_EXISTING=true
    else
      torch_ready cuda && KEEP_EXISTING=true
    fi
  fi
fi

if $KEEP_EXISTING; then
  ok "Torch already satisfies requested mode; not reinstalling."
else
  if [[ $INTERNET_OK -eq 0 ]]; then
    err "Offline and torch not suitable — cannot proceed."; exit 1
  fi

  if $FORCE_CPU; then
    note "Installing CPU-only torch ${TARGET_TORCH:+(target $TARGET_TORCH)}"
    if [[ -n "$TARGET_TORCH" ]]; then
      pip_retry "torch==${TARGET_TORCH}" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
      pip_retry torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
  else
    # Prefer CUDA 12.1 wheels; fallback to cu118
    CUDA_URL="https://download.pytorch.org/whl/cu121"
    if command -v nvidia-smi >/dev/null 2>&1; then
      RAW="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)"
      note "nvidia-smi present (driver ${RAW:-unknown})"
    fi
    note "Installing CUDA-enabled torch ${TARGET_TORCH:+(target $TARGET_TORCH)} via ${CUDA_URL}"
    if [[ -n "$TARGET_TORCH" ]]; then
      pip_retry "torch==${TARGET_TORCH}" torchvision torchaudio --index-url "$CUDA_URL" || {
        note "cu121 failed; falling back to cu118"
        pip_retry "torch==${TARGET_TORCH}" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      }
    else
      pip_retry torch torchvision torchaudio --index-url "$CUDA_URL" || {
        note "cu121 failed; falling back to cu118"
        pip_retry torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      }
    fi
  fi
fi

# ------------------------------- PyG -----------------------------------------
if $INSTALL_PYG; then
  banner "PyTorch Geometric (PyG) stack"
  if pyg_needed; then
    if [[ $INTERNET_OK -eq 0 ]]; then
      err "Offline and PyG not installed — skipping."; INSTALL_PYG=false
    else
      TORCH_BASE="$(python - <<'PY'
import torch; print(torch.__version__.split('+')[0], end="")
PY
)"
      CUDA_VER="$(python - <<'PY'
import torch; print(torch.version.cuda or "cpu", end="")
PY
)"
      CUDA_TAG="${CUDA_VER//./}"
      [[ "$CUDA_VER" == "cpu" ]] && CUDA_TAG="cpu"
      WHEEL_INDEX="https://data.pyg.org/whl/torch-${TORCH_BASE}+${CUDA_TAG}.html"
      note "Resolved PyG wheel index: ${WHEEL_INDEX}"
      pip_retry torch-scatter torch-sparse torch-cluster torch-spline-conv -f "$WHEEL_INDEX"
      pip_retry torch-geometric
      ok "PyG installed."
    fi
  else
    ok "PyG already present; skipping."
  fi
else
  note "Skipping PyG installation (--no-pyg)"
fi

# ------------------------------- DVC -----------------------------------------
banner "DVC (gs/s3/gdrive extras)"
if have_py 'import dvc'; then
  ok "DVC already installed."
else
  if [[ $INTERNET_OK -eq 1 ]]; then
    pip_retry "dvc[gs,s3,gdrive]"
  else
    note "Offline: cannot install DVC."
  fi
fi

# ------------------------------- REPORT --------------------------------------
banner "Environment summary"
python - <<'PY'
import sys
def safe(x): return x if x else "n/a"
try:
    import torch
    tver = safe(getattr(torch, "__version__", None))
    tcuda = safe(getattr(getattr(torch, "version", None), "cuda", None)) or ("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    tver, tcuda = "n/a", "n/a"
try:
    import dvc
    dver = safe(dvc.__version__)
except Exception:
    dver = "n/a"
print(f"Torch: {tver} (CUDA {tcuda})")
print(f"DVC:   {dver}")
print("Python:", sys.version.split()[0])
PY

ok "Kaggle environment ready."
