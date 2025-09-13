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
set -o errtrace
shopt -s extglob

_on_err() {
  local exit_code=$?
  local line_no=${BASH_LINENO[0]:-?}
  printf "\033[1;31m[ERR] Bootstrap failed at line %s. Last: '%s' (exit %s)\033[0m\n" \
    "${line_no}" "${BASH_COMMAND}" "${exit_code}" >&2
  exit $exit_code
}
trap _on_err ERR

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
    *) printf "[BOOT] Unknown option: %s\n" "$1" >&2; exit 2 ;;
  esac
done

# ---------------------------- PRINT UTILS ------------------------------------
c_cyan='\033[1;36m'; c_yel='\033[1;33m'; c_red='\033[1;31m'; c_grn='\033[1;32m'; c_off='\033[0m'
banner(){ $QUIET || printf "${c_cyan}\n==> %s${c_off}\n" "$1"; }
note(){   $QUIET || printf "${c_yel}[-] %s${c_off}\n" "$1"; }
ok(){     $QUIET || printf "${c_grn}[OK] %s${c_off}\n" "$1"; }
err(){    printf "${c_red}[ERR] %s${c_off}\n" "$1"; }

banner "SpectraMind V50 — Kaggle Bootstrap"

# --------------------------- KAGGLE CONTEXT -----------------------------------
IS_KAGGLE=0
[[ -d /kaggle && -d /kaggle/working ]] && IS_KAGGLE=1
note "Kaggle runtime detected: ${IS_KAGGLE}"

# Local caches (Kaggle-safe)
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/kaggle/working/.cache/pip}"
export TMPDIR="${TMPDIR:-/kaggle/working/.tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# Pip noise & safety
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_PYTHON_VERSION_WARNING=1
export PYTHONUTF8=1

# Internet check (DNS to pypi)
INTERNET_OK=0
python - <<'PY' >/dev/null 2>&1 || true
import socket, sys
try:
    socket.gethostbyname("pypi.org")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
[[ $? -eq 0 ]] && INTERNET_OK=1
[[ $INTERNET_OK -eq 0 ]] && note "Internet appears disabled. Will verify existing installs and exit if satisfied."

# ---------------------------- HELPERS ----------------------------------------
PIP_QUIET=()
$QUIET && PIP_QUIET+=(-q)

pip_retry() {
  # pip_retry <args...>
  local n=0 max=3
  until python -m pip install "${PIP_QUIET[@]}" "$@"; do
    n=$((n+1))
    if [[ $n -ge $max ]]; then err "pip install failed after ${max} attempts: $*"; return 1; fi
    note "pip retry $n/$max: $*"; sleep 2
  done
}

have_py() { python - "$@" >/dev/null 2>&1; }

torch_ready() {
  # torch_ready <cpu|cuda>
  python - "$1" <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
want = sys.argv[1]
if want == "cpu":
    ok = (not torch.cuda.is_available())
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
raise SystemExit(0 if not missing else 1)
PY
}

# ---------------------------- BASELINE TOOLS ---------------------------------
banner "Upgrade pip/setuptools/wheel"
if [[ $INTERNET_OK -eq 1 ]]; then
  pip_retry --upgrade pip wheel setuptools
else
  note "Offline: skipping pip bootstrap"
fi

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

KEEP_EXISTING=false
EXISTING_VER=""
EXISTING_CUDA=""

if have_py 'import torch'; then
  EXISTING_VER="$(python - <<'PY'
import torch; print(getattr(torch, "__version__", ""), end="")
PY
)"
  EXISTING_CUDA="$(python - <<'PY'
import torch; print(torch.version.cuda or "cpu", end="")
PY
)"
  note "Found torch ${EXISTING_VER:-unknown} (CUDA ${EXISTING_CUDA:-unknown})"
  if [[ -z "$TARGET_TORCH" ]]; then
    if $FORCE_CPU; then
      torch_ready cpu && KEEP_EXISTING=true
    else
      torch_ready cuda && KEEP_EXISTING=true
    fi
  fi
fi

maybe_match_tv_ta() {
  # Emits extra args to co-install torchvision/torchaudio matching torch version (best-effort)
  local ver="$1"
  [[ -z "$ver" ]] && return 0
  printf " torchvision==%s torchaudio==%s" "$ver" "$ver"
}

if $KEEP_EXISTING; then
  ok "Torch already satisfies requested mode; not reinstalling."
else
  if [[ $INTERNET_OK -eq 0 ]]; then
    err "Offline and torch not suitable — cannot proceed."; exit 1
  fi

  if $FORCE_CPU; then
    note "Installing CPU-only torch ${TARGET_TORCH:+(target $TARGET_TORCH)}"
    if [[ -n "$TARGET_TORCH" ]]; then
      # Try matching TV/TA first, then fall back unpinned
      if ! eval pip_retry "\"torch==${TARGET_TORCH}\"$(maybe_match_tv_ta "$TARGET_TORCH")" --index-url https://download.pytorch.org/whl/cpu; then
        pip_retry "torch==${TARGET_TORCH}" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      fi
    else
      pip_retry torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
  else
    # Prefer CUDA 12.1 wheels; fallback to cu118
    CUDA_URL="https://download.pytorch.org/whl/cu121"
    if command -v nvidia-smi >/dev/null 2>&1; then
      RAW="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)"
      note "nvidia-smi present (driver ${RAW:-unknown})"
    else
      note "nvidia-smi not present; proceeding with CUDA wheels based on index URL"
    fi
    note "Installing CUDA-enabled torch ${TARGET_TORCH:+(target $TARGET_TORCH)} via ${CUDA_URL}"
    if [[ -n "$TARGET_TORCH" ]]; then
      if ! eval pip_retry "\"torch==${TARGET_TORCH}\"$(maybe_match_tv_ta "$TARGET_TORCH")" --index-url "$CUDA_URL"; then
        note "cu121 failed; falling back to cu118"
        eval pip_retry "\"torch==${TARGET_TORCH}\"$(maybe_match_tv_ta "$TARGET_TORCH")" --index-url https://download.pytorch.org/whl/cu118 || \
        pip_retry "torch==${TARGET_TORCH}" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      fi
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
      # Compute correct PyG wheel index (fix: include 'cu' prefix when CUDA present)
      TORCH_BASE="$(python - <<'PY'
import torch; print(torch.__version__.split('+')[0], end="")
PY
)"
      CUDA_VER="$(python - <<'PY'
import torch; print(torch.version.cuda or "cpu", end="")
PY
)"
      if [[ "$CUDA_VER" == "cpu" ]]; then
        CUDA_TAG="cpu"
      else
        CUDA_TAG="cu${CUDA_VER//./}"
      fi
      WHEEL_INDEX="https://data.pyg.org/whl/torch-${TORCH_BASE}+${CUDA_TAG}.html"
      note "Resolved PyG wheel index: ${WHEEL_INDEX}"
      # Install compiled ops first, then metapkg
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
import sys, importlib
def safe(x): return x if x else "n/a"
try:
    import torch
    tver = safe(getattr(torch, "__version__", None))
    tcuda = safe(getattr(getattr(torch, "version", None), "cuda", None)) or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
except Exception:
    tver, tcuda, gpu = "n/a", "n/a", "n/a"
def has(m):
    try: importlib.import_module(m); return "yes"
    except Exception: return "no"
try:
    import dvc
    dver = safe(dvc.__version__)
except Exception:
    dver = "n/a"
print(f"Torch: {tver} (CUDA {tcuda}) device={gpu}")
print(f"DVC:   {dver}")
print("PyG:   tg:", has("torch_geometric"), " tsct:", has("torch_scatter"), " tspr:", has("torch_sparse"),
      " tclu:", has("torch_cluster"), " tspl:", has("torch_spline_conv"))
print("Python:", sys.version.split()[0])
PY

ok "Kaggle environment ready."
