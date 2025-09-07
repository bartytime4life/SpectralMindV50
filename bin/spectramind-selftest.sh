#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Self-Test Script
# ==============================================================================
# Runs a minimal end-to-end pipeline check to validate that the repository
# is functional in the current environment (local, Kaggle, CI).
#
# This script does NOT run heavy training. It uses:
#   - fast calibration profile
#   - debug data profile (tiny subset)
#   - dry-run submission
#
# Usage:
#   bin/spectramind-selftest.sh
#
# Exit codes:
#   0  = all tests passed
#   >0 = failure in one of the pipeline stages
# ==============================================================================

set -euo pipefail

# ----------------------------- Helpers ----------------------------------------
log() { echo -e "\033[1;34m[SELFTEST]\033[0m $*"; }
fail() { echo -e "\033[1;31m[SELFTEST ERROR]\033[0m $*"; exit 1; }

# ----------------------------- Env sanity -------------------------------------
if ! command -v python &>/dev/null; then
    fail "Python not found in PATH"
fi

if ! python -m spectramind.cli --help &>/dev/null; then
    fail "spectramind CLI not importable (check PYTHONPATH)"
fi

# ----------------------------- Pipeline tests ---------------------------------
log "1. Calibrate (fast profile)"
python -m spectramind calibrate +calib=fast +data=debug +env=local || fail "calibration failed"

log "2. Train (short smoke run)"
python -m spectramind train +data=debug +env=local trainer.max_epochs=1 trainer.limit_train_batches=2 || fail "training failed"

log "3. Predict (dummy checkpoint)"
python -m spectramind predict +data=debug +env=local checkpoint=last.ckpt || fail "prediction failed"

log "4. Diagnose (metrics + report)"
python -m spectramind diagnose +data=debug +env=local report_dir=artifacts/selftest_report || fail "diagnostics failed"

log "5. Submit (dry-run package)"
python -m spectramind submit +data=debug +env=local dry_run=true || fail "submission packaging failed"

# ----------------------------- Success ----------------------------------------
log "✅ Self-test completed successfully."
exit 0
