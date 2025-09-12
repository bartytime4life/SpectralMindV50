#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Self-Test Script (Upgraded, Precise Timing)
# ------------------------------------------------------------------------------
# Runs a tiny end-to-end pipeline to validate wiring across Local / CI / Kaggle.
# Measures per-stage wall time precisely (atomic wrapper), logs to console + JSONL.
# Creates isolated workdir under artifacts/selftest/<run_id>.
# ==============================================================================
set -euo pipefail
set -o errtrace

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

$VERBOSE && set -x

_on_err() {
  local exit_code=$?
  local line_no=${BASH_LINENO[0]:-?}
  echo -e "${c_red}[SELFTEST ERROR] Failed at line ${line_no}. Last: '${BASH_COMMAND}' (exit ${exit_code})${c_off}" >&2
  exit $exit_code
}
trap _on_err ERR

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

ts="$(date -u +"%Y%m%dT%H%M%SZ")"
rand="$RANDOM"
RUN_ID="st_${ts}_${rand}"
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
  local stage="$1"; local status="$2"; local secs="$3"; shift 3 || true
  local msg="${*:-}"
  python - "$stage" "$status" "$secs" "$RUN_ID" "$msg" >> "${JSONL}" <<'PY'
import json, sys, datetime
stage, status, secs, run_id, msg = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], " ".join(sys.argv[5:])
rec = {
  "ts": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
  "run_id": run_id,
  "stage": stage,
  "status": status,
  "seconds": round(secs, 3),
  "msg": msg or None
}
print(json.dumps(rec))
PY
}

# ----------------------- Precise timing wrapper (atomic) ----------------------
# Executes the command *inside Python*, measures wall time precisely,
# returns child's exit code, and prints duration to stdout for capture.
_run_timed_python() {
  # _run_timed_python <stage> <argv...>
  local stage="$1"; shift
  python - "$stage" "$@" <<'PY'
import sys, time, subprocess, shlex
stage = sys.argv[1]
cmd = sys.argv[2:]
# Run without shell; each token is already separated by bash
t0 = time.monotonic()
proc = subprocess.run(cmd, check=False)
dt = time.monotonic() - t0
# Emit only the duration so caller can capture it
print(f"{dt:.6f}")
# Propagate child's exit code
sys.exit(proc.returncode)
PY
}

run_step() {
  # run_step <stage-name> <command...>
  local stage="$1"; shift
  log "▶ ${stage}"
  local duration rc
  # Capture duration (stdout) and exit code
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
CALIB_OVR="+calib=fast +data=debug env=${env_name}"
TRAIN_OVR="+data=debug env=${env_name} trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=1 seed=42"
PRED_OVR="+data=debug env=${env_name}"
DIAG_OVR="+data=debug env=${env_name} report_dir=${WORKDIR}/report"
SUBM_OVR="+data=debug env=${env_name} dry_run=true"

if [[ "${FORCE_DEVICE}" == "cpu" ]]; then
  TRAIN_OVR="${TRAIN_OVR} device=cpu"
  PRED_OVR="${PRED_OVR} device=cpu"
else
  TRAIN_OVR="${TRAIN_OVR} device=cuda"
  PRED_OVR="${PRED_OVR} device=cuda"
fi

# ----------------------------- Run stages -------------------------------------
run_step "calibrate" python -m spectramind calibrate ${CALIB_OVR}
run_step "train"     python -m spectramind train     ${TRAIN_OVR}

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
run_step "predict"  python -m spectramind predict  ${PRED_OVR}  checkpoint="${CKPT}"
run_step "diagnose" python -m spectramind diagnose ${DIAG_OVR}
run_step "submit"   python -m spectramind submit   ${SUBM_OVR}

ok "✅ Self-test completed successfully."
echo "[SELFTEST] Workdir: ${WORKDIR}"
exit 0