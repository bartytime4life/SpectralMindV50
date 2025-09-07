#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — End-to-End Pipeline Runner
# -----------------------------------------------------------------------------
# Runs the canonical calibrate → train → predict → submit pipeline using the
# spectramind CLI. Safe for local dev, CI, and Kaggle kernels.
#
# Usage:
#   ./scripts/run_pipeline.sh [config_name]
#
# Example:
#   ./scripts/run_pipeline.sh train   # uses configs/train.yaml
#
# Notes:
# - Fails fast on errors (set -euo pipefail).
# - Prints each stage with timestamps for reproducibility.
# - Detects Kaggle vs local runtime for path safety.
# - All configs must live under configs/ and be Hydra-compatible.
# -----------------------------------------------------------------------------

set -euo pipefail

CFG_NAME="${1:-train}"   # default config = train
CLI="spectramind"

# --- Helpers ---------------------------------------------------------------

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

log() {
  echo -e "[ $(timestamp) ] [run_pipeline] $*"
}

detect_env() {
  if [ -d "/kaggle/input" ]; then
    echo "kaggle"
  else
    echo "local"
  fi
}

ENV_TYPE=$(detect_env)

# --- Pipeline --------------------------------------------------------------

log "Running SpectraMind V50 pipeline (env: $ENV_TYPE, config: $CFG_NAME)"

log "Step 1: Calibration"
$CLI calibrate --config-name "$CFG_NAME"

log "Step 2: Training"
$CLI train --config-name "$CFG_NAME"

log "Step 3: Prediction"
$CLI predict --config-name predict

log "Step 4: Submission packaging"
$CLI submit --config-name submit

log "Pipeline finished successfully ✅"
