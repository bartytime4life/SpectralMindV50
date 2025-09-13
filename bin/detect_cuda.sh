#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — detect_cuda.sh
# Robust GPU detection for NVIDIA CUDA and AMD ROCm, with PyTorch probe.
# • Works without external deps (best-effort fallbacks)
# • Kaggle-aware (quieter; never assumes sudo/pkg tools)
# • Exports env vars for downstream scripts unless --dry-run
# • Optional JSON summary for CI logs
# ------------------------------------------------------------------------------
# Usage:
#   bin/detect_cuda.sh [--json] [--quiet] [--dry-run] [--strict] [-h|--help]
#
# Exports (unless --dry-run):
#   SM_HAS_CUDA=1|0
#   SM_CUDA_BACKEND=cuda|rocm|none
#   SM_GPU_COUNT=<int>
#   SM_GPU_NAMES=<comma-separated>
#   SM_TORCH_CUDA=1|0|unknown
#   SM_TORCH_VERSION=<ver|unknown>
#   CUDA_VISIBLE_DEVICES (preserved if already set; otherwise 0..n-1)
#
# Exit codes:
#   0 = ran successfully (GPU may or may not be present)
#   3 = --strict and no usable GPU detected
# ==============================================================================

set -Eeuo pipefail

# -------- Pretty printing ------------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

usage() {
  cat <<'USAGE'
detect_cuda.sh — detect NVIDIA/ROCm GPUs and PyTorch CUDA availability

Options:
  --json       Emit JSON summary to stdout (in addition to exports)
  --quiet      Suppress informational logs
  --dry-run    Do not export environment variables
  --strict     Exit non-zero if no usable GPU is detected
  -h, --help   Show this help

Exports:
  SM_HAS_CUDA, SM_CUDA_BACKEND, SM_GPU_COUNT, SM_GPU_NAMES,
  SM_TORCH_CUDA, SM_TORCH_VERSION, CUDA_VISIBLE_DEVICES (if unset)
USAGE
}

trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"' ERR

# -------- Args ----------------------------------------------------------------
EMIT_JSON=0
QUIET="${QUIET:-0}"
DRYRUN=0
STRICT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) EMIT_JSON=1; shift ;;
    --quiet) QUIET=1; shift ;;
    --dry-run) DRYRUN=1; shift ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# Kaggle awareness (only used for tone; no behavior change required)
IS_KAGGLE=0
[[ -d "/kaggle" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] && IS_KAGGLE=1

# -------- Detection state ------------------------------------------------------
SM_HAS_CUDA=0
SM_CUDA_BACKEND="none"
SM_GPU_COUNT=0
SM_GPU_NAMES=""
SM_TORCH_CUDA="unknown"
SM_TORCH_VERSION="unknown"

# -------- Helpers --------------------------------------------------------------
join_by_comma() {
  local IFS=','; echo "$*"
}

# -------- Strategy 1: NVIDIA via nvidia-smi -----------------------------------
NV_NAMES=()
if command -v nvidia-smi >/dev/null 2>&1; then
  # Query names & count; tolerate restricted drivers (no JSON)
  if mapfile -t NV_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^ *//;s/ *$//'); then
    if [[ ${#NV_NAMES[@]} -gt 0 ]]; then
      SM_GPU_COUNT=${#NV_NAMES[@]}
      SM_GPU_NAMES="$(join_by_comma "${NV_NAMES[@]}")"
      SM_CUDA_BACKEND="cuda"
      SM_HAS_CUDA=1
      ok "Detected NVIDIA GPUs (${SM_GPU_COUNT}): ${SM_GPU_NAMES}"
    fi
  fi
fi

# -------- Strategy 2: ROCm via rocm-smi / rocminfo ----------------------------
if [[ "$SM_HAS_CUDA" -eq 0 ]]; then
  ROCM_NAMES=()
  if command -v rocm-smi >/dev/null 2>&1; then
    # rocm-smi --showproductname (older) or --showproductname --json (newer)
    if mapfile -t ROCM_NAMES < <(rocm-smi --showproductname 2>/dev/null | awk -F': ' '/Card series/ {print $2}' | sed 's/^ *//;s/ *$//'); then
      if [[ ${#ROCM_NAMES[@]} -gt 0 ]]; then
        SM_GPU_COUNT=${#ROCM_NAMES[@]}
        SM_GPU_NAMES="$(join_by_comma "${ROCM_NAMES[@]}")"
        SM_CUDA_BACKEND="rocm"
        SM_HAS_CUDA=1
        ok "Detected AMD ROCm GPUs (${SM_GPU_COUNT}): ${SM_GPU_NAMES}"
      fi
    fi
  elif command -v rocminfo >/dev/null 2>&1; then
    # Fallback: parse GPU names from rocminfo
    if mapfile -t ROCM_NAMES < <(rocminfo 2>/dev/null | awk -F': ' '/Name:/ {print $2}' | sort -u); then
      if [[ ${#ROCM_NAMES[@]} -gt 0 ]]; then
        SM_GPU_COUNT=${#ROCM_NAMES[@]}
        SM_GPU_NAMES="$(join_by_comma "${ROCM_NAMES[@]}")"
        SM_CUDA_BACKEND="rocm"
        SM_HAS_CUDA=1
        ok "Detected AMD ROCm GPUs (${SM_GPU_COUNT}): ${SM_GPU_NAMES}"
      fi
    fi
  fi
fi

# -------- Strategy 3: Kernel/proc and lspci (very best-effort) ----------------
if [[ "$SM_HAS_CUDA" -eq 0 ]]; then
  # /proc/driver/nvidia/gpus/<id>/information may exist even without nvidia-smi
  if [[ -d /proc/driver/nvidia/gpus ]]; then
    mapfile -t NV_RAW < <(grep -h '^Model' /proc/driver/nvidia/gpus/*/information 2>/dev/null | sed 's/^Model: *//')
    if [[ ${#NV_RAW[@]} -gt 0 ]]; then
      SM_GPU_COUNT=${#NV_RAW[@]}
      SM_GPU_NAMES="$(join_by_comma "${NV_RAW[@]}")"
      SM_CUDA_BACKEND="cuda"
      SM_HAS_CUDA=1
      ok "Detected NVIDIA via /proc (${SM_GPU_COUNT}): ${SM_GPU_NAMES}"
    fi
  fi
fi

if [[ "$SM_HAS_CUDA" -eq 0 ]] && command -v lspci >/dev/null 2>&1; then
  # Look for NVIDIA/AMD VGA controllers
  mapfile -t VGA_LINES < <(lspci 2>/dev/null | grep -Ei 'VGA compatible controller|3D controller' || true)
  if printf "%s\n" "${VGA_LINES[@]}" | grep -qi 'nvidia'; then
    SM_CUDA_BACKEND="cuda"
    SM_HAS_CUDA=1
    SM_GPU_COUNT=$(printf "%s\n" "${VGA_LINES[@]}" | grep -i 'nvidia' | wc -l | tr -d ' ')
    SM_GPU_NAMES="NVIDIA GPU (from lspci)"
    ok "Detected NVIDIA via lspci (${SM_GPU_COUNT})"
  elif printf "%s\n" "${VGA_LINES[@]}" | grep -Eqi 'AMD|Advanced Micro Devices'; then
    SM_CUDA_BACKEND="rocm"
    SM_HAS_CUDA=1
    SM_GPU_COUNT=$(printf "%s\n" "${VGA_LINES[@]}" | grep -Ei 'AMD|Advanced Micro Devices' | wc -l | tr -d ' ')
    SM_GPU_NAMES="AMD GPU (from lspci)"
    ok "Detected AMD via lspci (${SM_GPU_COUNT})"
  fi
fi

# -------- Strategy 4: PyTorch probe (if present) ------------------------------
PYTHON_EXE="${PYTHON:-${PYTHON3:-python3}}"
if command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  PY_OUT="$("$PYTHON_EXE" - <<'PY' 2>/dev/null || true)
import json, sys
try:
    import torch
    v = getattr(torch, "__version__", "unknown")
    try:
        avail = torch.cuda.is_available()
    except Exception:
        avail = False
    try:
        devs = torch.cuda.device_count() if avail else 0
        names = [torch.cuda.get_device_name(i) for i in range(devs)] if avail else []
    except Exception:
        devs, names = 0, []
    j = {"version": v, "available": bool(avail), "count": devs, "names": names}
    print(json.dumps(j))
except Exception:
    print(json.dumps({"version": "unknown", "available": "unknown", "count": 0, "names": []}))
PY
  " || true)"

  if [[ -n "$PY_OUT" ]]; then
    # shellcheck disable=SC2001
    SM_TORCH_VERSION="$(printf "%s" "$PY_OUT" | sed -n 's/.*"version":"\([^"]*\)".*/\1/p')"
    case "$PY_OUT" in
      *'"available": true'*) SM_TORCH_CUDA="1" ;;
      *'"available": false'*) SM_TORCH_CUDA="0" ;;
      *) SM_TORCH_CUDA="unknown" ;;
    esac
    if [[ "$SM_TORCH_CUDA" == "1" && "$SM_HAS_CUDA" -eq 0 ]]; then
      # Trust PyTorch even if system tools failed (containerized scenarios)
      SM_HAS_CUDA=1
      # Derive backend heuristically (NVIDIA common on PyTorch)
      SM_CUDA_BACKEND="${SM_CUDA_BACKEND:-cuda}"
      # Extract names if present
      TORCH_NAMES="$(printf "%s" "$PY_OUT" | sed -n 's/.*"names":\[\(.*\)\].*/\1/p' | tr -d '"' )"
      [[ -n "$TORCH_NAMES" ]] && SM_GPU_NAMES="$(printf "%s" "$TORCH_NAMES" | sed 's/,/,/g')"
      # Count
      if [[ "$PY_OUT" =~ \"count\":([0-9]+) ]]; then
        SM_GPU_COUNT="${BASH_REMATCH[1]}"
      fi
    fi
    [[ "$SM_TORCH_VERSION" == "" ]] && SM_TORCH_VERSION="unknown"
    [[ "$SM_TORCH_CUDA" == "1" ]] && ok "PyTorch CUDA available (torch=${SM_TORCH_VERSION})"
  fi
else
  warn "No python3 on PATH; skipping PyTorch probe"
fi

# -------- Compose CUDA_VISIBLE_DEVICES if unset --------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$SM_HAS_CUDA" -eq 1 && "$SM_GPU_COUNT" -gt 0 ]]; then
  CVD=""
  for ((i=0; i<SM_GPU_COUNT; i++)); do
    CVD+="${i},"
  done
  CVD="${CVD%,}"
  export_cvd=1
else
  CVD="${CUDA_VISIBLE_DEVICES:-}"
  export_cvd=0
fi

# -------- Export env vars (unless --dry-run) ----------------------------------
export_line_count=0
do_export() {
  local k="$1" v="$2"
  if [[ "$DRYRUN" -eq 1 ]]; then
    printf "%s=%q\n" "$k" "$v"
  else
    export "$k=$v"
    export_line_count=$((export_line_count+1))
  fi
}

do_export SM_HAS_CUDA "$SM_HAS_CUDA"
do_export SM_CUDA_BACKEND "$SM_CUDA_BACKEND"
do_export SM_GPU_COUNT "$SM_GPU_COUNT"
do_export SM_GPU_NAMES "$SM_GPU_NAMES"
do_export SM_TORCH_CUDA "$SM_TORCH_CUDA"
do_export SM_TORCH_VERSION "$SM_TORCH_VERSION"
if [[ "$export_cvd" -eq 1 ]]; then
  do_export CUDA_VISIBLE_DEVICES "$CVD"
fi

if [[ "$DRYRUN" -eq 1 ]]; then
  log "Dry-run: not exporting. (Shown above)"
else
  ok "Detection complete: backend=${SM_CUDA_BACKEND}, has_cuda=${SM_HAS_CUDA}, gpus=${SM_GPU_COUNT}"
  [[ "$export_cvd" -eq 1 ]] && log "Set CUDA_VISIBLE_DEVICES=${CVD}"
fi

# -------- JSON summary ---------------------------------------------------------
if [[ "$EMIT_JSON" -eq 1 ]]; then
  # Escape helper
  esc() { printf "%s" "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  printf '{'
  printf '"kaggle":%s,' "$IS_KAGGLE"
  printf '"has_cuda":%s,' "$SM_HAS_CUDA"
  printf '"backend":"%s",' "$(esc "$SM_CUDA_BACKEND")"
  printf '"gpu_count":%s,' "$SM_GPU_COUNT"
  printf '"gpu_names":"%s",' "$(esc "$SM_GPU_NAMES")"
  printf '"torch_cuda":"%s",' "$(esc "$SM_TORCH_CUDA")"
  printf '"torch_version":"%s",' "$(esc "$SM_TORCH_VERSION")"
  printf '"cuda_visible_devices":"%s"' "$(esc "$CVD")"
  printf '}\n'
fi

# -------- Strict mode ----------------------------------------------------------
if [[ "$STRICT" -eq 1 && "$SM_HAS_CUDA" -ne 1 ]]; then
  err "No usable GPU detected (strict mode)"
  exit 3
fi

exit 0
