#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Self-Test Script (Upgraded)
# ------------------------------------------------------------------------------
# What it does
#   • Runs a tiny end-to-end pipeline to validate repo wiring (imports, CLI,
#     configs, checkpoints & packaging) across Local / CI / Kaggle.
#   • Measures per-stage duration, writes colored console logs + JSONL audit.
#   • Creates an isolated working dir under artifacts/selftest/ with run hash.
#
# What it does NOT do
#   • Heavy training. This is a smoke test with tiny batches/epochs.
#
# Usage:
#   bin/spectramind-selftest.sh [--env local|kaggle|ci] [--gpu|--cpu]
#                               [--keep] [--verbose]
#
# Exit codes:
#   0  = all stages passed
#   1+ = failure (stage name is shown; see artifacts/selftest/*)
# ==============================================================================

set -euo pipefail

# ----------------------------- CLI & defaults ---------------------------------
ENV_HINT=""
FORCE_DEVICE=""
KEEP_WORKDIR=false
VERBOSE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)         ENV_HINT="${2:-}"; shift 2 ;;
    --gpu)         FORCE_DEVICE="cuda"; shift ;;
    --cpu)         FORCE_DEVICE="cpu"; shift  ;;
    --keep)        KEEP_WORKDIR=true; shift ;;
    --verbose|-v)  VERBOSE=true; shift ;;
    *) echo "[SELFTEST] Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ----------------------------- Pretty logging ---------------------------------
c_blu='\033[1;34m'; c_red='\033[1;31m'; c_grn='\033[1;32m'; c_yel='\033[1;33m'; c_dim='\033[2m'; c_off='\033[0m'
log()  { echo -e "${c_blu}[SELFTEST]${c_off} $*"; }
ok()   { echo -e "${c_grn}[SELFTEST]${c_off} $*"; }
warn() { echo -e "${c_yel}[SELFTEST]${c_off} $*"; }
fail() { echo -e "${c_red}[SELFTEST ERROR]${c_off} $*"; exit 1; }

# ----------------------------- Env detection ----------------------------------
is_kaggle=0
[[ -d /kaggle && -d /kaggle/working ]] && is_kaggle=1
is_ci="${CI:-}"
if [[ -n "$ENV_HINT" ]]; then
  env_name="$ENV_HINT"
elif [[ "$is_kaggle" -eq 1 ]]; then
  env_name="kaggle"
elif [[ -n "$is_ci" ]]; then
  env_name="ci"
else
  env_name="local"
fi
log "Environment: ${env_name}"

# Device hint
if [[ -z "${FORCE_DEVICE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    FORCE_DEVICE="cuda"
  else
    FORCE_DEVICE="cpu"
  fi
fi
log "Device preference: ${FORCE_DEVICE}"

# ----------------------------- Paths & workdir --------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTI_DIR="${ROOT_DIR}/artifacts/selftest"
mkdir -p "${ARTI_DIR}"

# Short run hash (timestamp + random)
ts="$(date -u +"%Y%m%dT%H%M%SZ")"
rand="$RANDOM"
RUN_ID="st_${ts}_${rand}"
WORKDIR="${ARTI_DIR}/${RUN_ID}"
LOG_FILE="${WORKDIR}/selftest.log"
JSONL="${WORKDIR}/events.jsonl"
mkdir -p "${WORKDIR}"

# Ensure logs also go to file
exec > >(tee -a "${LOG_FILE}") 2>&1

# ----------------------------- Pre-flight checks ------------------------------
log "Python: $(python -V 2>&1 | tr -d '\n')"
command -v python >/dev/null 2>&1 || fail "Python not found in PATH"

# Ensure the spectramind module is importable
python - <<'PY' || fail "spectramind package not importable; check PYTHONPATH/install."
import importlib, sys
mod = importlib.util.find_spec("spectramind")
cli = importlib.util.find_spec("spectramind.__main__") or importlib.util.find_spec("spectramind.cli")
print("spectramind:", "OK" if mod else "MISSING")
print("CLI entry  :", "OK" if cli else "MISSING")
assert mod is not None
PY

# Helpful config sanity (best-effort)
for f in "configs" "src/spectramind"; do
  [[ -e "${ROOT_DIR}/${f}" ]] || warn "Expected path missing: ${f} (continuing)"
done

# ----------------------------- JSONL helpers ----------------------------------
event() {
  # event <stage> <status> <seconds> [msg]
  local stage="$1"; local status="$2"; local secs="$3"; shift 3 || true
  local msg="${*:-}"
  printf '{"ts":"%s","run_id":"%s","stage":"%s","status":"%s","seconds":%s,"msg":%s}\n' \
    "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    "${RUN_ID}" \
    "${stage}" \
    "${status}" \
    "${secs}" \
    "$(python - <<PY
import json,sys
print(json.dumps(" ".join(sys.argv[1:])) if len(sys.argv)>1 else "null")
PY ${msg@Q})" \
    >> "${JSONL}"
}

run_step() {
  # run_step <stage-name> <command...>
  local stage="$1"; shift
  log "▶ ${stage}"
  local t0 t1 dt
  t0=$(python - <<'PY';import time;print(time.time());PY)
  if "$@"; then
    t1=$(python - <<'PY';import time;print(time.time());PY)
    dt=$(python - <<PY;print({}.fromkeys([None]).__class__);PY >/dev/null 2>&1; python - <<PY
import sys,math
t0=float(sys.argv[1]); t1=float(sys.argv[2])
print(round(t1-t0,3))
PY "$t0" "$t1")
    ok "✔ ${stage} (${dt}s)"
    event "${stage}" "ok" "${dt}"
    return 0
  else
    t1=$(python - <<'PY';import time;print(time.time());PY)
    dt=$(python - <<PY
import sys
t0=float(sys.argv[1]); t1=float(sys.argv[2])
print(round(t1-t0,3))
PY "$t0" "$t1")
    event "${stage}" "fail" "${dt}" "command failed"
    fail "${stage} failed — see ${LOG_FILE}"
  fi
}

# Cleanup trap
cleanup() {
  echo -e "${c_dim}[SELFTEST] logs: ${LOG_FILE}${c_off}"
  echo -e "${c_dim}[SELFTEST] jsonl: ${JSONL}${c_off}"
  $KEEP_WORKDIR || true
}
trap cleanup EXIT

# ----------------------------- Common overrides -------------------------------
# Tiny, deterministic run settings (Hydra-friendly), can be tuned later
CALIB_OVR="+calib=fast +data=debug env=${env_name}"
TRAIN_OVR="+data=debug env=${env_name} trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=1 seed=42"
PRED_OVR="+data=debug env=${env_name}"
DIAG_OVR="+data=debug env=${env_name} report_dir=${WORKDIR}/report"
SUBM_OVR="+data=debug env=${env_name} dry_run=true"

# Device override for training/predict if your CLI supports it
if [[ "${FORCE_DEVICE}" == "cpu" ]]; then
  TRAIN_OVR="${TRAIN_OVR} device=cpu"
  PRED_OVR="${PRED_OVR} device=cpu"
else
  TRAIN_OVR="${TRAIN_OVR} device=cuda"
  PRED_OVR="${PRED_OVR} device=cuda"
fi

# ----------------------------- Run stages -------------------------------------
# 1) Calibrate (fast)
run_step "calibrate" \
  python -m spectramind calibrate ${CALIB_OVR}

# 2) Train (smoke)
run_step "train" \
  python -m spectramind train ${TRAIN_OVR}

# 3) Predict (auto/latest checkpoint if supported; else last.ckpt)
# Try to auto-discover a checkpoint inside artifacts/ or similar, else fallback.
CKPT="last.ckpt"
if compgen -G "artifacts/**/last.ckpt" > /dev/null; then
  CKPT="$(ls -1 artifacts/**/last.ckpt | head -n 1)"
elif compgen -G "artifacts/**/*.ckpt" > /dev/null; then
  CKPT="$(ls -1 artifacts/**/*.ckpt | head -n 1)"
fi
log "Using checkpoint: ${CKPT}"
run_step "predict" \
  python -m spectramind predict ${PRED_OVR} checkpoint="${CKPT}"

# 4) Diagnose (HTML/JSON)
run_step "diagnose" \
  python -m spectramind diagnose ${DIAG_OVR}

# 5) Submit (dry-run)
run_step "submit" \
  python -m spectramind submit ${SUBM_OVR}

ok "✅ Self-test completed successfully."
echo "[SELFTEST] Workdir: ${WORKDIR}"
exit 0