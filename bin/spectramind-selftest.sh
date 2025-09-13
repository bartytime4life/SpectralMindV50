#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Self-Test Script (Upgraded, Precise Timing + Fallbacks)
# ------------------------------------------------------------------------------
# Runs a tiny end-to-end pipeline to validate wiring across Local / CI / Kaggle.
# Precise per-stage wall time (atomic Python wrapper), console + JSONL logging.
# Creates isolated workdir under artifacts/selftest/<run_id>.
#
# Usage:
#   ./bin/selftest.sh [--env <local|ci|kaggle>] [--gpu|--cpu]
#                     [--keep] [--verbose|-v] [--stages list]
#   where "list" is comma/space-separated subset of: calibrate,train,predict,diagnose,submit
# ==============================================================================
set -euo pipefail
set -o errtrace

# ----------------------------- CLI & defaults ---------------------------------
ENV_HINT=""
FORCE_DEVICE=""
KEEP_WORKDIR=false
VERBOSE=false
STAGES="calibrate,train,predict,diagnose,submit"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)         ENV_HINT="${2:-}"; shift 2 ;;
    --gpu)         FORCE_DEVICE="cuda"; shift ;;
    --cpu)         FORCE_DEVICE="cpu"; shift  ;;
    --keep)        KEEP_WORKDIR=true; shift ;;
    --stages)      STAGES="${2:-}"; shift 2 ;;
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

$VERBOSE && set -x

_on_err() {
  local exit_code=$?
  local line_no=${BASH_LINENO[0]:-?}
  echo -e "${c_red}[SELFTEST ERROR] Failed at line ${line_no}. Last: '${BASH_COMMAND}' (exit ${exit_code})${c_off}" >&2
  exit $exit_code
}
trap _on_err ERR

# ----------------------------- Determinism ------------------------------------
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8" 2>/dev/null || true

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

# Device hint (with auto-fallback from cuda→cpu if CUDA not actually available)
if [[ -z "${FORCE_DEVICE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then FORCE_DEVICE="cuda"; else FORCE_DEVICE="cpu"; fi
fi

# If CUDA requested but not available in torch, fall back cleanly to CPU
if python - <<'PY' >/dev/null 2>&1; then
import sys
try:
    import torch
    sys.exit(0 if torch.cuda.is_available() else 1)
except Exception:
    sys.exit(2)
PY
then
  cuda_ok=1
else
  cuda_ok=0
fi
if [[ "${FORCE_DEVICE}" == "cuda" && $cuda_ok -eq 0 ]]; then
  warn "CUDA requested, but torch reports no GPU; falling back to CPU."
  FORCE_DEVICE="cpu"
fi
log "Device preference: ${FORCE_DEVICE}"

# ----------------------------- Paths & workdir --------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTI_DIR="${ROOT_DIR}/artifacts/selftest"
mkdir -p "${ARTI_DIR}"

ts="$(date -u +"%Y%m%dT%H%M%SZ")"
# shellcheck disable=SC2034
RUN_ID="st_${ts}_$RANDOM"
WORKDIR="${ARTI_DIR}/${RUN_ID}"
LOG_FILE="${WORKDIR}/selftest.log"
JSONL="${WORKDIR}/events.jsonl"
mkdir -p "${WORKDIR}"

# Tee all output
exec > >(tee -a "${LOG_FILE}") 2>&1

# ----------------------------- Pre-flight checks ------------------------------
command -v python >/dev/null 2>&1 || fail "Python not found in PATH"
log "Python: $(python -V 2>&1 | tr -d '\n')"

python - <<'PY' || exit 1
import importlib, sys
mod = importlib.util.find_spec("spectramind")
cli = importlib.util.find_spec("spectramind.__main__") or importlib.util.find_spec("spectramind.cli")
print("spectramind:", "OK" if mod else "MISSING")
print("CLI entry  :", "OK" if cli else "MISSING")
if mod is None:
    sys.exit(2)
PY
[[ $? -eq 0 ]] || fail "spectramind package not importable; check PYTHONPATH/install."

for f in "configs" "src/spectramind"; do
  [[ -e "${ROOT_DIR}/${f}" ]] || warn "Expected path missing: ${f} (continuing)"
done

# ----------------------------- JSONL helper -----------------------------------
event() {
  # event <stage> <status> <seconds> [msg...]
  local stage="$1"; local status="$2"; local secs="${3:-0}"; shift 3 || true
  local msg="${*:-}"
  python - "$stage" "$status" "$secs" "$RUN_ID" "$env_name" "$FORCE_DEVICE" "$msg" >> "${JSONL}" <<'PY'
import json, sys, datetime, platform, os, random
stage, status, secs, run_id, env_name, device, msg = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], " ".join(sys.argv[7:])
py = platform.python_version()
rec = {
  "ts": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
  "run_id": run_id,
  "env": env_name,
  "device": device,
  "stage": stage,
  "status": status,
  "seconds": round(secs, 3),
  "python": py,
  "msg": msg or None,
}
print(json.dumps(rec, separators=(",",":")))
PY
}

# ----------------------- Precise timing wrapper (atomic) ----------------------
# Executes the command *inside Python*, measures wall time precisely,
# returns child's exit code, and prints duration to stdout for capture.
_run_timed_python() {
  # _run_timed_python <stage> <argv...>
  local stage="$1"; shift
  python - "$stage" "$@" <<'PY'
import sys, time, subprocess
stage = sys.argv[1]
cmd = sys.argv[2:]
t0 = time.monotonic()
proc = subprocess.run(cmd, check=False)
dt = time.monotonic() - t0
print(f"{dt:.6f}")
sys.exit(proc.returncode)
PY
}

run_step() {
  # run_step <stage-name> <command...>
  local stage="$1"; shift
  log "▶ ${stage}"
  local duration rc
  duration="$(_run_timed_python "$stage" "$@")"; rc=$?
  if [[ $rc -eq 0 ]]; then
    ok "✔ ${stage} (${duration}s)"
    event "${stage}" "ok" "${duration}"
    return 0
  else
    event "${stage}" "fail" "${duration:-0.0}" "command failed (rc=${rc})"
    fail "${stage} failed — see ${LOG_FILE}"
  fi
}

# Cleanup trap — delete workdir unless --keep (but keep last_selftest.* snapshots)
cleanup() {
  echo -e "${c_dim}[SELFTEST] log:   ${LOG_FILE}${c_off}"
  echo -e "${c_dim}[SELFTEST] jsonl: ${JSONL}${c_off}"
  if ! $KEEP_WORKDIR; then
    cp -f "${LOG_FILE}" "${ARTI_DIR}/last_selftest.log" || true
    cp -f "${JSONL}"    "${ARTI_DIR}/last_selftest.jsonl" || true
    rm -rf "${WORKDIR}" || true
    echo -e "${c_dim}[SELFTEST] cleaned ${WORKDIR} (kept last_selftest.*)${c_off}"
  fi
}
trap cleanup EXIT

# ----------------------------- Common overrides -------------------------------
# Force deterministic seeds for the tiny run
SEED=42
export SM_SUBMISSION_BINS="${SM_SUBMISSION_BINS:-283}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::DeprecationWarning}"

# Respect device choice in Hydra overrides
HDEV="device=${FORCE_DEVICE}"

CALIB_OVR="+calib=fast +data=debug env=${env_name}"
TRAIN_OVR="+data=debug env=${env_name} trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=1 seed=${SEED} ${HDEV}"
PRED_OVR="+data=debug env=${env_name} ${HDEV}"
DIAG_OVR="+data=debug env=${env_name} report_dir=${WORKDIR}/report"
SUBM_OVR="+data=debug env=${env_name} dry_run=true"

# Normalize stage list → array
IFS=', ' read -r -a STAGE_LIST <<< "${STAGES}"

should_run() {
  local target="$1"
  for s in "${STAGE_LIST[@]}"; do [[ "$s" == "$target" ]] && return 0; done
  return 1
}

# ----------------------------- Env snapshot -----------------------------------
log "Seeding: PYTHONHASHSEED=${PYTHONHASHSEED}, script seed=${SEED}"
python - <<'PY'
try:
    import torch, numpy as np, random, os
    random.seed(42); np.random.seed(42)
    if hasattr(torch, 'manual_seed'): torch.manual_seed(42)
    if hasattr(torch.cuda, 'manual_seed_all'): torch.cuda.manual_seed_all(42)
    cuda = torch.cuda.is_available()
    dev  = torch.cuda.get_device_name(0) if cuda else "cpu"
    print(f"Torch    : {getattr(torch,'__version__','n/a')} | CUDA={cuda} | dev={dev}")
except Exception as e:
    print(f"Torch    : not importable ({e})")
try:
    import numpy as np; print(f"NumPy    : {np.__version__}")
except Exception as e:
    print(f"NumPy    : not importable ({e})")
PY

# ----------------------------- Run stages -------------------------------------
if should_run calibrate; then
  run_step "calibrate" python -m spectramind calibrate ${CALIB_OVR}
fi

if should_run train; then
  run_step "train"     python -m spectramind train     ${TRAIN_OVR}
fi

# Checkpoint discovery (first last.ckpt, else any .ckpt)
CKPT="last.ckpt"
if command -v find >/dev/null 2>&1; then
  if CK=$(find "${ROOT_DIR}/artifacts" -type f -name 'last.ckpt' -print -quit 2>/dev/null); then
    [[ -n "$CK" ]] && CKPT="$CK"
  fi
  if [[ "$CKPT" == "last.ckpt" ]]; then
    if CK=$(find "${ROOT_DIR}/artifacts" -type f -name '*.ckpt' -print -quit 2>/dev/null); then
      [[ -n "$CK" ]] && CKPT="$CK"
    fi
  fi
fi
log "Using checkpoint: ${CKPT}"

if should_run predict; then
  run_step "predict"  python -m spectramind predict  ${PRED_OVR}  checkpoint="${CKPT}"
fi
if should_run diagnose; then
  run_step "diagnose" python -m spectramind diagnose ${DIAG_OVR}
fi
if should_run submit; then
  run_step "submit"   python -m spectramind submit   ${SUBM_OVR}
fi

ok "✅ Self-test completed successfully."
echo "[SELFTEST] Workdir: ${WORKDIR}"
exit 0
