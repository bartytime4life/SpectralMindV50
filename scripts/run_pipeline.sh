#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — End-to-End Pipeline Runner (Ultra Upgraded)
# -----------------------------------------------------------------------------
# Runs calibrate → train → predict → submit via the spectramind CLI.
# Safe for local dev, CI, and Kaggle kernels; resumable; deterministic optional.
#
# Usage:
#   ./scripts/run_pipeline.sh [options] [train_cfg_name]
#
# Stage control:
#   --no-calib | --no-train | --no-predict | --no-submit   Skip stage(s)
#   --from STAGE       Start from {calibrate|train|predict|submit}
#   --to   STAGE       End at   {calibrate|train|predict|submit}
#   --resume           Skip stages already marked OK in events JSONL
#
# Behavior:
#   --dry-run          Print the plan, do nothing
#   --log-file PATH    Log file (default env-aware)
#   --predict-cfg NAME Predict config (default: predict)
#   --submit-cfg  NAME Submit  config (default: submit)
#   --retries N        Per-stage retries (default: 1)
#   --backoff S        Initial backoff (sec, default: 2; doubles per retry)
#   --timeout S        Per-stage timeout (sec, default: 0 = no timeout)
#   --env-dump         Write env snapshot into artifacts/
#   --deterministic    Export deterministic envs for CUDA/torch run
#   --tail N           Lines of log tail to print on failure (default: 120)
#
# Notes:
# - Writes JSONL events: artifacts/run_events.jsonl
# - Writes summary JSON: artifacts/run_summary.json
# - Detects Kaggle vs local for safe paths
# -----------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# ------------------------- Defaults & Globals --------------------------------
CFG_NAME="train"
PREDICT_CFG="predict"
SUBMIT_CFG="submit"

ENV_TYPE="local"
LOG_FILE=""
EVENTS_JSONL=""
SUMMARY_JSON=""
ARTIFACTS_DIR=""

DO_CALIB=1
DO_TRAIN=1
DO_PREDICT=1
DO_SUBMIT=1
DRY_RUN=0
RESUME=0
TAIL_N=120

FROM_STAGE=""
TO_STAGE=""

RETRIES=1
BACKOFF=2
TIMEOUT=0
ENV_DUMP=0
DETERMINISTIC=0

CLI="${CLI:-spectramind}"
EXTRA_ARGS="${SPECTRAMIND_EXTRA_ARGS:-}"   # pass-through to all stages

# ------------------------------ Helpers --------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
now_secs()  { date +%s; }
detect_env(){
  if [[ -d "/kaggle/input" ]]; then echo "kaggle"; else echo "local"; fi
}
have()      { command -v "$1" >/dev/null 2>&1; }
fsize()     { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

log() { echo "[ $(timestamp) ] [run_pipeline] $*"; }
err() { echo "[ $(timestamp) ] [run_pipeline][ERROR] $*" >&2; exit 1; }

print_usage() { sed -n '1,160p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

config_exists() {
  local name="$1"
  [[ -f "configs/${name}.yaml" ]] || [[ -d "configs/${name}" ]]
}

ensure_paths() {
  mkdir -p "${ARTIFACTS_DIR}/logs"
  mkdir -p "$(dirname "$EVENTS_JSONL")"
  mkdir -p "$(dirname "$SUMMARY_JSON")"
}

append_event() {
  # args: stage status start end dur cfg env try msg
  local stage="$1" status="$2" start="$3" end="$4" dur="$5" cfg="$6" env="$7" attempt="${8:-1}" msg="${9:-}"
  {
    printf '{'
    printf '"ts":"%s",' "$(timestamp)"
    printf '"stage":%q,' "$stage"
    printf '"status":%q,' "$status"
    printf '"start":%s,' "$start"
    printf '"end":%s,' "$end"
    printf '"duration_sec":%s,' "$dur"
    printf '"config":%q,' "$cfg"
    printf '"env":%q,' "$env"
    printf '"attempt":%s,' "$attempt"
    printf '"message":%q' "$msg"
    printf '}\n'
  } >> "$EVENTS_JSONL"
}

stage_index() { case "$1" in calibrate) echo 1;; train) echo 2;; predict) echo 3;; submit) echo 4;; *) echo 0;; esac; }

with_timeout_eval() {
  # usage: with_timeout_eval <timeout_seconds> "<command string>"
  local to="$1"; shift; local cmd="$*"
  if (( to > 0 )) && have timeout; then
    eval "timeout ${to}s $cmd"
  elif (( to > 0 )); then
    # Portable Python shim
    python - "$to" "$cmd" <<'PY'
import os, sys, time, subprocess, shlex
to=int(sys.argv[1]); cmd=sys.argv[2]
p=subprocess.Popen(cmd, shell=True)
t0=time.time()
while p.poll() is None:
    if time.time()-t0>to:
        try: p.terminate()
        except Exception: pass
        time.sleep(1)
        try: p.kill()
        except Exception: pass
        sys.exit(124)
    time.sleep(0.1)
sys.exit(p.returncode)
PY
  else
    eval "$cmd"
  fi
}

run_with_retries() {
  # usage: run_with_retries <stage> <cfg> <timeout> <retries> <backoff> "<cmd>"
  local stage="$1" cfg="$2" to="$3" tries="$4" back="$5" cmd="$6"
  local attempt=1 start end dur rc
  while :; do
    start="$(now_secs)"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      log "DRY-RUN: [$stage] $cmd"
      end="$start"; dur=0; append_event "$stage" "skipped(dry-run)" "$start" "$end" "$dur" "$cfg" "$ENV_TYPE" "$attempt"
      return 0
    fi
    log "▶ $stage (attempt $attempt/$tries)"
    log "cmd: $cmd"
    set +e
    with_timeout_eval "$to" "$cmd"
    rc=$?
    set -e
    end="$(now_secs)"; dur="$(( end - start ))"
    if [[ $rc -eq 0 ]]; then
      log "✓ $stage completed in ${dur}s"
      append_event "$stage" "ok" "$start" "$end" "$dur" "$cfg" "$ENV_TYPE" "$attempt"
      return 0
    fi
    log "✗ $stage failed with rc=$rc after ${dur}s"
    append_event "$stage" "failed" "$start" "$end" "$dur" "$cfg" "$ENV_TYPE" "$attempt" "rc=$rc"
    if (( attempt >= tries )); then
      return $rc
    fi
    log "Retrying in ${back}s..."
    sleep "$back"
    attempt=$((attempt+1))
    back=$((back*2))
  done
}

print_log_tail_on_error() {
  local code=$?
  if [[ $code -ne 0 && -f "$LOG_FILE" ]]; then
    echo; echo "─── LOG TAIL (last ${TAIL_N} lines) ─────────────────────────────────────"
    tail -n "$TAIL_N" "$LOG_FILE" || true
    echo "────────────────────────────────────────────────────────────────────────"
  fi
  exit $code
}

# ------------------------------ Args -----------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-calib)    DO_CALIB=0; shift ;;
    --no-train)    DO_TRAIN=0; shift ;;
    --no-predict)  DO_PREDICT=0; shift ;;
    --no-submit)   DO_SUBMIT=0; shift ;;
    --from)        FROM_STAGE="${2:-}"; shift 2 ;;
    --to)          TO_STAGE="${2:-}"; shift 2 ;;
    --resume)      RESUME=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --log-file)    LOG_FILE="${2:?}"; shift 2 ;;
    --predict-cfg) PREDICT_CFG="${2:?}"; shift 2 ;;
    --submit-cfg)  SUBMIT_CFG="${2:?}"; shift 2 ;;
    --retries)     RETRIES="${2:-1}"; shift 2 ;;
    --backoff)     BACKOFF="${2:-2}"; shift 2 ;;
    --timeout)     TIMEOUT="${2:-0}"; shift 2 ;;
    --env-dump)    ENV_DUMP=1; shift ;;
    --deterministic) DETERMINISTIC=1; shift ;;
    --tail)        TAIL_N="${2:-120}"; shift 2 ;;
    -h|--help)     print_usage ;;
    *)             CFG_NAME="$1"; shift ;;
  esac
done

# ------------------------------ Bootstrap ------------------------------------
ENV_TYPE="$(detect_env)"

if [[ -z "$LOG_FILE" ]]; then
  if [[ "$ENV_TYPE" = "kaggle" ]]; then
    ARTIFACTS_DIR="/kaggle/working/artifacts"
  else
    ARTIFACTS_DIR="artifacts"
  fi
  LOG_FILE="${ARTIFACTS_DIR}/logs/run.log"
else
  ARTIFACTS_DIR="$(dirname "$(dirname "$LOG_FILE")")"
fi

EVENTS_JSONL="${ARTIFACTS_DIR}/run_events.jsonl"
SUMMARY_JSON="${ARTIFACTS_DIR}/run_summary.json"

ensure_paths

# tee logs
exec > >(tee -a "$LOG_FILE") 2>&1

# graceful traps
trap 'code=$?; [[ $code -ne 0 ]] && log "Pipeline aborted (exit $code)"; exit $code' EXIT
trap 'print_log_tail_on_error' ERR
trap 'log "Interrupted (SIGINT)"; exit 130' INT

have "$CLI" || err "Missing required command: $CLI"

# config existence checks
for n in "$CFG_NAME" "$PREDICT_CFG" "$SUBMIT_CFG"; do
  config_exists "$n" || err "Config '$n' not found (expected configs/$n.yaml or configs/$n/)"
done

# determinism (best-effort)
if [[ "$DETERMINISTIC" -eq 1 ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONHASHSEED=0
  export TORCH_USE_CUDA_DSA=0
  export CUDA_LAUNCH_BLOCKING=1
  export TF_DETERMINISTIC_OPS=1
  log "Deterministic mode: ON (exported CUDA/torch envs)"
fi

# git/version context (best-effort)
GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GIT_REV="$(git -C "$GIT_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo "nogit")"
GIT_DIRTY="$(git -C "$GIT_ROOT" diff --quiet 2>/dev/null || echo "-dirty")"
VERSION_FILE="$GIT_ROOT/VERSION"
PROJ_VERSION="$( [[ -f "$VERSION_FILE" ]] && tr -d '[:space:]' < "$VERSION_FILE" || echo "0.0.0")"

log "Running SpectraMind V50 pipeline (env: $ENV_TYPE)"
log "Configs: calibrate/train=$CFG_NAME, predict=$PREDICT_CFG, submit=$SUBMIT_CFG"
[[ -n "$EXTRA_ARGS" ]] && log "Extra CLI args: $EXTRA_ARGS"
[[ "$DRY_RUN" -eq 1 ]] && log "Dry-run enabled (no commands will execute)"
log "Git: ${GIT_REV}${GIT_DIRTY}  Version: ${PROJ_VERSION}"
log "Artifacts dir: ${ARTIFACTS_DIR}"

# optional env snapshot
if [[ "$ENV_DUMP" -eq 1 ]]; then
  SNAP="${ARTIFACTS_DIR}/env_snapshot.txt"
  {
    echo "# $(timestamp)"
    echo "ENV: $ENV_TYPE"
    if have nvidia-smi; then nvidia-smi || true; fi
    if have python; then
      python - <<'PY'
try:
  import json, platform
  print("PY:", platform.python_version())
  try:
    import torch
    print("TORCH:", torch.__version__, "cuda:", getattr(torch.version,'cuda',None), "gpu:", torch.cuda.is_available())
  except Exception as e:
    print("TORCH probe failed:", e)
except Exception as e:
  print("PY probe failed:", e)
PY
      python -m pip freeze | sed 's/^/pip: /' || true
    fi
  } > "$SNAP"
  log "Environment snapshot → $SNAP ($(fsize "$SNAP" || echo 0) bytes)"
fi

# ------------------------------ Stage gating ---------------------------------
FROM_IDX="${FROM_STAGE:+$(stage_index "$FROM_STAGE")}"
TO_IDX="${TO_STAGE:+$(stage_index "$TO_STAGE")}"

should_run_flag() {
  local idx="$1"
  case "$idx" in
    1) [[ $DO_CALIB   -eq 1 ]] ;;
    2) [[ $DO_TRAIN   -eq 1 ]] ;;
    3) [[ $DO_PREDICT -eq 1 ]] ;;
    4) [[ $DO_SUBMIT  -eq 1 ]] ;;
  esac
}

within_window() {
  local idx="$1"
  if [[ -n "$FROM_IDX" && $idx -lt $FROM_IDX ]]; then return 1; fi
  if [[ -n "$TO_IDX"   && $idx -gt $TO_IDX   ]]; then return 1; fi
  return 0
}

resume_skip() {
  # returns 0 (skip) if RESUME=1 and last event for stage has status "ok"
  local stage="$1"
  [[ "$RESUME" -ne 1 ]] && return 1
  [[ -f "$EVENTS_JSONL" ]] || return 1
  awk -v st="$stage" '
    BEGIN{ok=0}
    /"stage":/ && $0~("\""st"\""){found=1}
    found && /"status":"ok"/{ok=1}
    END{ if(ok) exit 0; else exit 1 }
  ' "$EVENTS_JSONL"
}

# ------------------------------ Pipeline -------------------------------------
PIPE_START="$(now_secs)"
RC=0

run_stage() {
  local idx="$1" name="$2" cfg="$3" cmd="$4"
  if ! should_run_flag "$idx"; then
    log "Skipping $name (explicit --no-*)"
    append_event "$name" "skipped" "$(now_secs)" "$(now_secs)" 0 "$cfg" "$ENV_TYPE" 0
    return 0
  fi
  within_window "$idx" || { log "Skipping $name (outside window)"; append_event "$name" "skipped" "$(now_secs)" "$(now_secs)" 0 "$cfg" "$ENV_TYPE" 0; return 0; }
  if resume_skip "$name"; then
    log "Resuming: skip $name (already ok)"
    append_event "$name" "resumed-skip" "$(now_secs)" "$(now_secs)" 0 "$cfg" "$ENV_TYPE" 0
    return 0
  fi
  run_with_retries "$name" "$cfg" "$TIMEOUT" "$RETRIES" "$BACKOFF" "$cmd"
}

run_stage 1 "Calibration" "$CFG_NAME"  "$CLI calibrate --config-name \"$CFG_NAME\" $EXTRA_ARGS" || RC=$?
[[ $RC -ne 0 ]] && err "Calibration failed (rc=$RC)"

run_stage 2 "Training"    "$CFG_NAME"  "$CLI train     --config-name \"$CFG_NAME\" $EXTRA_ARGS" || RC=$?
[[ $RC -ne 0 ]] && err "Training failed (rc=$RC)"

run_stage 3 "Prediction"  "$PREDICT_CFG" "$CLI predict  --config-name \"$PREDICT_CFG\" $EXTRA_ARGS" || RC=$?
[[ $RC -ne 0 ]] && err "Prediction failed (rc=$RC)"

run_stage 4 "Submission packaging" "$SUBMIT_CFG" "$CLI submit --config-name \"$SUBMIT_CFG\" $EXTRA_ARGS" || RC=$?
[[ $RC -ne 0 ]] && err "Submission packaging failed (rc=$RC)"

PIPE_END="$(now_secs)"
PIPE_DUR="$(( PIPE_END - PIPE_START ))"

# ------------------------------ Summary --------------------------------------
{
  echo "{"
  echo "  \"ts\": \"$(timestamp)\","
  echo "  \"env\": \"${ENV_TYPE}\","
  echo "  \"git\": { \"rev\": \"${GIT_REV}${GIT_DIRTY}\", \"root\": \"${GIT_ROOT}\" },"
  echo "  \"version\": \"${PROJ_VERSION}\","
  echo "  \"configs\": { \"calibrate/train\": \"${CFG_NAME}\", \"predict\": \"${PREDICT_CFG}\", \"submit\": \"${SUBMIT_CFG}\" },"
  echo "  \"args\": { \"retries\": ${RETRIES}, \"backoff\": ${BACKOFF}, \"timeout\": ${TIMEOUT}, \"dry_run\": ${DRY_RUN}, \"resume\": ${RESUME}, \"deterministic\": ${DETERMINISTIC} },"
  echo "  \"artifacts_dir\": \"${ARTIFACTS_DIR}\","
  echo "  \"events_jsonl\": \"${EVENTS_JSONL}\","
  echo "  \"log_file\": \"${LOG_FILE}\","
  echo "  \"duration_sec\": ${PIPE_DUR}"
  echo "}"
} > "$SUMMARY_JSON"

# CI outputs (if running in GitHub Actions)
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "new_run_id=$(date +%Y%m%d%H%M%S)-${GIT_REV}"
    echo "run_summary=${SUMMARY_JSON}"
  } >> "$GITHUB_OUTPUT"
fi

log "Pipeline finished successfully ✅ (total ${PIPE_DUR}s)"
log "Events JSONL → $EVENTS_JSONL"
log "Summary JSON → $SUMMARY_JSON"