#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Train Helper
# ==============================================================================
# One-shot training wrapper that:
#   • Verifies repo root & toolchain
#   • Detects (or forces) CPU/GPU execution
#   • Composes Hydra configs / profiles (e.g., ci, kaggle, local)
#   • Runs: `python -m spectramind train ...`
#   • Captures config snapshot, metrics, and model checkpoint
#   • (Optional) tracks outputs in DVC
#
# Usage:
#   bin/sm_train.sh
#   bin/sm_train.sh --config configs/train.yaml --profile /profiles/ci
#   bin/sm_train.sh --resume path/to/ckpt.pth --tags "abl=ssm" --seed 123
#   bin/sm_train.sh --cpu          # force CPU
#   bin/sm_train.sh --kaggle       # Kaggle-safe settings
#   bin/sm_train.sh --dvc-track    # dvc add the run artifacts
#
# Notes:
#   • Aligns with DVC stages (calibrate → preprocess → train → predict → …) and
#     emits stable artifacts for caching/lineage:contentReference[oaicite:2]{index=2}.
#   • Keeps business logic in `src/spectramind/` CLI (Typer) per repo blueprint:contentReference[oaicite:3]{index=3}.
# ==============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------
CONFIG="configs/train.yaml"
PROFILE=""                 # e.g. "/profiles/ci" or "/env/kaggle"
OUTDIR=""                  # auto if empty: outputs/train/<timestamp>
RESUME=""
SEED=""
TAGS=""                    # freeform run tags/notes
FORCE_CPU=0
KAGGLE=0
DVC_TRACK=0
EXTRA=""                   # extra args pass-through to python CLI

# ------------------------------------------------------------------------------
# Args
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)   CONFIG="$2"; shift 2 ;;
    --profile)  PROFILE="$2"; shift 2 ;;     # Hydra group, e.g. "/profiles/ci"
    --outdir)   OUTDIR="$2"; shift 2 ;;
    --resume)   RESUME="$2"; shift 2 ;;
    --seed)     SEED="$2"; shift 2 ;;
    --tags)     TAGS="$2"; shift 2 ;;
    --cpu)      FORCE_CPU=1; shift ;;
    --kaggle)   KAGGLE=1; shift ;;
    --dvc-track) DVC_TRACK=1; shift ;;
    --)         shift; EXTRA="$*"; break ;;
    -h|--help)
      echo "Usage: $0 [--config FILE] [--profile HYDRA_GROUP] [--outdir DIR] [--resume CKPT]"
      echo "          [--seed N] [--tags STR] [--cpu] [--kaggle] [--dvc-track] [-- ...extra...]"
      exit 0 ;;
    *) echo "[ERR] Unknown arg: $1"; exit 1 ;;
  esac
done

# ------------------------------------------------------------------------------
# Repo-root sanity
# ------------------------------------------------------------------------------
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/spectramind" ]]; then
  echo "[ERR] Run from SpectraMind V50 repo root."
  exit 1
fi

# ------------------------------------------------------------------------------
# Env detection (CUDA/Kaggle) & settings
# ------------------------------------------------------------------------------
HAS_NVIDIA=$(command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)
if [[ $FORCE_CPU -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES=""
  DEV_NOTE="cpu-forced"
elif [[ $HAS_NVIDIA -eq 1 ]]; then
  DEV_NOTE="gpu"
else
  export CUDA_VISIBLE_DEVICES=""
  DEV_NOTE="cpu-auto"
fi

if [[ $KAGGLE -eq 1 ]]; then
  # Prefer Kaggle-safe profile (no internet, time/space constraints)
  # Expect a hydra group like configs/env/kaggle.yaml to slim deps/precision:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}.
  PROFILE="${PROFILE:-/env/kaggle}"
fi

# ------------------------------------------------------------------------------
# Outdir + run metadata
# ------------------------------------------------------------------------------
TS=$(date +"%Y%m%d_%H%M%S")
OUTDIR="${OUTDIR:-outputs/train/${TS}}"
mkdir -p "$OUTDIR"

GIT_REV=$(git rev-parse --short HEAD || echo "nogit")
echo "[INFO] outdir=$OUTDIR  git=$GIT_REV  dev=$DEV_NOTE  profile=${PROFILE:-none}"

# ------------------------------------------------------------------------------
# Build Hydra overrides
# ------------------------------------------------------------------------------
HYDRA_OVR=()
HYDRA_OVR+=("--config" "$CONFIG")
if [[ -n "$PROFILE" ]]; then
  # Let users pass a *Hydra group* path. We mimic: +defaults='[/profiles/ci]':contentReference[oaicite:6]{index=6}.
  HYDRA_OVR+=("+defaults=[${PROFILE}]")
fi
if [[ -n "$SEED" ]]; then HYDRA_OVR+=("seed=$SEED"); fi
if [[ -n "$TAGS" ]]; then HYDRA_OVR+=("run.tags='${TAGS}'"); fi
HYDRA_OVR+=("paths.run_dir=${OUTDIR}")    # honor hydra.run.dir in training code:contentReference[oaicite:7]{index=7}

# ------------------------------------------------------------------------------
# Optional resume
# ------------------------------------------------------------------------------
RESUME_ARG=()
if [[ -n "$RESUME" ]]; then
  if [[ ! -f "$RESUME" ]]; then echo "[ERR] --resume file not found"; exit 1; fi
  RESUME_ARG+=( "--resume" "$RESUME" )
fi

# ------------------------------------------------------------------------------
# Snapshot config for provenance (optional utility in repo):contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
# ------------------------------------------------------------------------------
SNAP_JSON="$OUTDIR/config_snapshot.json"
python -m spectramind.config.snapshot save \
  --config "${CONFIG}" \
  --output "${SNAP_JSON}" \
  --extra "${PROFILE}" || true

# ------------------------------------------------------------------------------
# Run training
# ------------------------------------------------------------------------------
echo "[INFO] Launching training..."
PYARGS=(
  -m spectramind train
  "${HYDRA_OVR[@]}"
  --out "${OUTDIR}"
  "${RESUME_ARG[@]}"
)

# Extra passthrough (after --)
if [[ -n "$EXTRA" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=( $EXTRA )
  PYARGS+=( "${EXTRA_ARR[@]}" )
fi

python "${PYARGS[@]}"

# ------------------------------------------------------------------------------
# Expect artifacts: model ckpt, metrics, logs (consistent with repo blueprint):contentReference[oaicite:10]{index=10}
# ------------------------------------------------------------------------------
CKPT_GLOB=("$OUTDIR"/*.pth "$OUTDIR"/*.ckpt)
METRICS_GLOB=("$OUTDIR"/metrics.json "$OUTDIR"/metrics.yaml)
FOUND_CKPT=""; for f in "${CKPT_GLOB[@]}"; do [[ -f "$f" ]] && FOUND_CKPT="$f" && break; done
FOUND_METRICS=""; for m in "${METRICS_GLOB[@]}"; do [[ -f "$m" ]] && FOUND_METRICS="$m" && break; done

if [[ -z "$FOUND_CKPT" ]]; then
  echo "[WARN] No checkpoint found in $OUTDIR (check trainer config/paths)."
else
  echo "[OK] Checkpoint: $FOUND_CKPT"
fi
if [[ -z "$FOUND_METRICS" ]]; then
  echo "[WARN] No metrics file found in $OUTDIR."
else
  echo "[OK] Metrics:    $FOUND_METRICS"
fi

# ------------------------------------------------------------------------------
# (Optional) DVC track run outputs for reproducibility/cache:contentReference[oaicite:11]{index=11}:contentReference[oaicite:12]{index=12}
# ------------------------------------------------------------------------------
if [[ $DVC_TRACK -eq 1 ]]; then
  if command -v dvc >/dev/null 2>&1; then
    echo "[INFO] DVC tracking $OUTDIR ..."
    dvc add "$OUTDIR" >/dev/null
    git add "${OUTDIR}.dvc" .gitignore >/dev/null || true
    echo "[OK] Added DVC artifact for $OUTDIR"
  else
    echo "[WARN] dvc not found; skipping --dvc-track."
  fi
fi

echo "[DONE] Training complete → $OUTDIR"

