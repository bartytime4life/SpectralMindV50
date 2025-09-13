#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Kaggle Submission Helper
# ==============================================================================
# Builds a validated Kaggle-ready submission bundle:
#   • Verifies repo root, config, and data paths
#   • Runs prediction stage (if needed) → submission.csv
#   • Validates schema (Frict. tableschema)
#   • Packages artifacts → submission.zip
#   • (Optional) pushes bundle to Kaggle via CLI
#
# Usage:
#   bin/sm_submit.sh --config configs/submit.yaml
#   bin/sm_submit.sh --skip-predict   # use existing predictions
#   bin/sm_submit.sh --kaggle-push    # also kaggle competitions submit
#
# Env vars:
#   KAGGLE_COMP   = ariel-data-challenge-2025 (default)
#   KAGGLE_NOTE   = "SpectraMind V50 submission"
# ==============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------
KAGGLE_COMP="${KAGGLE_COMP:-ariel-data-challenge-2025}"
KAGGLE_NOTE="${KAGGLE_NOTE:-SpectraMind V50 submission}"
CONFIG="configs/submit.yaml"
DO_PREDICT=1
DO_PUSH=0

# ------------------------------------------------------------------------------
# Parse args
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2 ;;
    --skip-predict) DO_PREDICT=0; shift ;;
    --kaggle-push) DO_PUSH=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--config FILE] [--skip-predict] [--kaggle-push]"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ------------------------------------------------------------------------------
# Ensure repo root
# ------------------------------------------------------------------------------
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/spectramind" ]]; then
  echo "[ERR] Must be run from SpectraMind V50 repo root"
  exit 1
fi

# ------------------------------------------------------------------------------
# Run predict → submission.csv
# ------------------------------------------------------------------------------
OUTDIR="outputs/submission"
mkdir -p "$OUTDIR"

if [[ $DO_PREDICT -eq 1 ]]; then
  echo "[INFO] Running prediction stage..."
  python -m spectramind predict --config "$CONFIG" \
    --out "$OUTDIR/submission.csv"
else
  echo "[INFO] Skipping predict, using existing $OUTDIR/submission.csv"
fi

# ------------------------------------------------------------------------------
# Validate schema
# ------------------------------------------------------------------------------
SCHEMA="schemas/submission.tableschema.sample_id.json"
HEADER="schemas/submission_header.csv"
if [[ ! -f "$OUTDIR/submission.csv" ]]; then
  echo "[ERR] submission.csv not found"
  exit 1
fi

echo "[INFO] Validating submission schema..."
python -m spectramind.utils.schema validate \
  --schema "$SCHEMA" \
  --header "$HEADER" \
  --csv "$OUTDIR/submission.csv"

# ------------------------------------------------------------------------------
# Package bundle
# ------------------------------------------------------------------------------
echo "[INFO] Packaging bundle..."
ZIP="$OUTDIR/submission.zip"
rm -f "$ZIP"
zip -j "$ZIP" "$OUTDIR/submission.csv" "$SCHEMA" "$HEADER" \
  > /dev/null

echo "[OK] Built bundle: $ZIP"

# ------------------------------------------------------------------------------
# Optional Kaggle push
# ------------------------------------------------------------------------------
if [[ $DO_PUSH -eq 1 ]]; then
  echo "[INFO] Submitting to Kaggle..."
  kaggle competitions submit -c "$KAGGLE_COMP" \
    -f "$ZIP" \
    -m "$KAGGLE_NOTE"
fi

