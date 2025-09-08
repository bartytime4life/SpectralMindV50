#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — End-to-End Pipeline Runner (Upgraded)
# -----------------------------------------------------------------------------
# Runs calibrate → train → predict → submit using the spectramind CLI.
# Safe for local dev, CI, and Kaggle kernels.
#
# Usage:
#   ./scripts/run_pipeline.sh [options] [config_name]
#
# Common options:
#   --no-calib | --no-train | --no-predict | --no-submit   Skip stage(s)
#   --from STAGE       Start from {calibrate|train|predict|submit}
#   --to   STAGE       End at   {calibrate|train|predict|submit}
#   --dry-run          Print the plan, do nothing
#   --log-file PATH    Log file (default: artifacts/logs/run.log)
#   --predict-cfg NAME Predict config-name (default: predict)
#   --submit-cfg  NAME Submit  config-name (default: submit)
#   --retries N        Per-stage retries on failure (default: 1; i.e., 1 try)
#   --backoff S        Initial backoff seconds (default: 2; doubles each retry)
#   --timeout S        Per-stage timeout seconds (default: 0 = no timeout)
#   --env-dump         Write env snapshot (GPU/cuda/pip list) into artifacts/
#
# Examples:
#   ./scripts/run_pipeline.sh                    # uses configs/train.yaml
#   ./scripts/run_pipeline.sh debug              # uses configs/debug.yaml
#   ./scripts/run_pipeline.sh --from predict     # only predict → submit
#   ./scripts/run_pipeline.sh --no-submit        # stop after predict
#   ./scripts/run_pipeline.sh --retries 3 --timeout 18000
#
# Notes:
# - Fails fast (set -Eeuo pipefail). All stdout/err is tee'd to a logfile.
# - Writes JSONL events: artifacts/run_events.jsonl
# - Writes summary JSON: artifacts/run_summary.json
# - Detects Kaggle vs local to keep paths safe.
# -----------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# ------------------------- Defaults & Globals --------------------------------
CFG_NAME="train"
PREDICT_CFG="predict"
SUBMIT_CFG="submit"

LOG_FILE="${LOG_FILE:-artifacts/logs/run.log}"
EVENTS_JSONL="artifacts/run_events.jsonl"
SUMMARY_JSON="artifacts/run_summary.json"

DO_CALIB=1
DO_TRAIN=1
DO_PREDICT=1
DO_SUBMIT=1
DRY_RUN=0

FROM_STAGE=""
TO_STAGE=""

RETRIES=1
BACKOFF=2
TIMEOUT=0

ENV_DUMP=0

CLI="${CLI:-spectramind}"
EXTRA_ARGS="${SPECTRAMIND_EXTRA_ARGS:-}"   # pass-through to all stages

# ------------------------------ Helpers --------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
now_secs()  { date +%s; }
detect_env(){ [[ -d "/kaggle/input" ]] && echo "kaggle" || echo "local"; }
have()      { command -v "$1" >/dev/null 2>&1; }
fsize()     { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

log() { echo "[ $(timestamp) ] [run_pipeline] $*"; }
err() { echo "[ $(timestamp) ] [run_pipeline][ERROR] $*" >&2; exit 1; }

ensure_paths() {
  mkdir -p "artifacts/logs"
  mkdir -p "$(dirname "$EVENTS_JSONL")"
}

config_exists() {
  local name="$1"
  [[ -f "configs/${name}.yaml" ]] || [[ -d "configs/${name}" ]]
}

append_event() {
  # args: stage status start end dur cfg env try msg
  local stage="$1" status="$2" start="$3" end="$4" dur="$5" cfg="$6" env="$7" attempt="$8" msg="$9"
  {
    printf '{'
    printf '"ts":"%s",' "$(timestamp)"
    printf '"stage":"%s",' "$stage"
    printf '"status":"%s",' "$status"
    printf '"start":%s,' "$start"
    printf '"end":%s,' "$end"
    printf '"duration_sec":%s,' "$dur"
    printf '"config":"%s",' "$cfg"
    printf '"env":"%s",' "$env"
    printf '"attempt":%s,' "${attempt:-1}"
    printf '"message":%q' "${msg:-""}"
    printf '}\n'
  } >> "$EVENTS_JSONL"
}

with_timeout_eval() {
  # usage: with_timeout_eval <timeout_seconds> "<command string>"
  local to="$1"; shift
  local cmd="$*"
  if (( to > 0 )) && have timeout; then
    eval "timeout ${to}s $cmd"
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

print_usage() {
  sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
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
    --dry-run)     DRY_RUN=1; shift ;;
    --log-file)    LOG_FILE="${2:?}"; shift 2 ;;
    --predict-cfg) PREDICT_CFG="${2:?}"; shift 2 ;;
    --submit-cfg)  SUBMIT_CFG="${2:?}"; shift 2 ;;
    --retries)     RETRIES="${2:-1}"; shift 2 ;;
    --backoff)     BACKOFF="${2:-2}"; shift 2 ;;
    --timeout)     TIMEOUT="${2:-0}"; shift 2 ;;
    --env-dump)    ENV_DUMP=1; shift ;;
    -h|--help)     print_usage ;;
    *)             CFG_NAME="$1"; shift ;;
  esac
done

# ------------------------------ Bootstrap ------------------------------------
ENV_TYPE="$(detect_env)"
ensure_paths

# tee logs
exec > >(tee -a "$LOG_FILE") 2>&1

# graceful exit on Ctrl+C
trap 'code=$?; [[ $code -ne 0 ]] && log "Pipeline aborted (exit $code)"; exit $code' EXIT
trap 'log "Interrupted (SIGINT)"; exit 130' INT

have "$CLI" || err "Missing required command: $CLI"

# config existence checks
for n in "$CFG_NAME" "$PREDICT_CFG" "$SUBMIT_CFG"; do
  config_exists "$n" || err "Config '$n' not found under configs/ (expected configs/$n.yaml or directory)"
done

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

# optional env snapshot
if [[ "$ENV_DUMP" -eq 1 ]]; then
  SNAP="artifacts/env_snapshot.txt"
  {
    echo "# $(timestamp)"
    echo "ENV: $ENV_TYPE"
    if have nvidia-smi; then nvidia-smi || true; fi
    if have python; then
      python - <<'PY'
try:
  import torch, json, platform
  print("PY:", platform.python_version())
  print("TORCH:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.is_available())
except Exception as e:
  print("PY/torch probe failed:", e)
PY
      python -m pip freeze | sed 's/^/pip: /' || true
    fi
  } > "$SNAP"
  log "Environment snapshot → $SNAP ($(fsize "$SNAP" || echo 0) bytes)"
fi

# ------------------------------ Stage gating ---------------------------------
stage_index() { case "$1" in calibrate) echo 1;; train) echo 2;; predict) echo 3;; submit) echo 4;; *) echo 0;; esac; }
FROM_IDX="${FROM_STAGE:+$(stage_index "$FROM_STAGE")}"
TO_IDX="${TO_STAGE:+$(stage_index "$TO_STAGE")}"

should_run() {
  local idx="$1" default_on="$2"
  # respect explicit --no-* flags first
  if   [[ $idx -eq 1 && $DO_CALIB   -eq 0 ]]; then return 1
  elif [[ $idx -eq 2 && $DO_TRAIN   -eq 0 ]]; then return 1
  elif [[ $idx -eq 3 && $DO_PREDICT -eq 0 ]]; then return 1
  elif [[ $idx -eq 4 && $DO_SUBMIT  -eq 0 ]]; then return 1; fi
  # respect --from/--to window if provided
  if [[ -n "$FROM_IDX" && $idx -lt $FROM_IDX ]]; then return 1; fi
  if [[ -n "$TO_IDX"   && $idx -gt $TO_IDX   ]]; then return 1; fi
  # default
  [[ $default_on -eq 1 ]]
}

# ------------------------------ Pipeline -------------------------------------
PIPE_START="$(now_secs)"
RC=0

# Calibration
if should_run 1 1; then
  run_with_retries "Calibration" "$CFG_NAME" "$TIMEOUT" "$RETRIES" "$BACKOFF" \
    "$CLI calibrate --config-name \"$CFG_NAME\" $EXTRA_ARGS" || RC=$?
else
  log "Skipping Calibration"
  append_event "Calibration" "skipped" "$(now_secs)" "$(now_secs)" 0 "$CFG_NAME" "$ENV_TYPE" 0
fi
[[ $RC -ne 0 ]] && err "Calibration failed (rc=$RC)"

# Training
if should_run 2 1; then
  run_with_retries "Training" "$CFG_NAME" "$TIMEOUT" "$RETRIES" "$BACKOFF" \
    "$CLI train --config-name \"$CFG_NAME\" $EXTRA_ARGS" || RC=$?
else
  log "Skipping Training"
  append_event "Training" "skipped" "$(now_secs)" "$(now_secs)" 0 "$CFG_NAME" "$ENV_TYPE" 0
fi
[[ $RC -ne 0 ]] && err "Training failed (rc=$RC)"

# Prediction
if should_run 3 1; then
  run_with_retries "Prediction" "$PREDICT_CFG" "$TIMEOUT" "$RETRIES" "$BACKOFF" \
    "$CLI predict --config-name \"$PREDICT_CFG\" $EXTRA_ARGS" || RC=$?
else
  log "Skipping Prediction"
  append_event "Prediction" "skipped" "$(now_secs)" "$(now_secs)" 0 "$PREDICT_CFG" "$ENV_TYPE" 0
fi
[[ $RC -ne 0 ]] && err "Prediction failed (rc=$RC)"

# Submission (package)
if should_run 4 1; then
  run_with_retries "Submission packaging" "$SUBMIT_CFG" "$TIMEOUT" "$RETRIES" "$BACKOFF" \
    "$CLI submit --config-name \"$SUBMIT_CFG\" $EXTRA_ARGS" || RC=$?
else
  log "Skipping Submission"
  append_event "Submission packaging" "skipped" "$(now_secs)" "$(now_secs)" 0 "$SUBMIT_CFG" "$ENV_TYPE" 0
fi
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
  echo "  \"args\": { \"retries\": ${RETRIES}, \"backoff\": ${BACKOFF}, \"timeout\": ${TIMEOUT}, \"dry_run\": ${DRY_RUN} },"
  echo "  \"duration_sec\": ${PIPE_DUR},"
  echo "  \"events_jsonl\": \"${EVENTS_JSONL}\","
  echo "  \"log_file\": \"${LOG_FILE}\""
  echo "}"
} > "$SUMMARY_JSON"

log "Pipeline finished successfully ✅ (total ${PIPE_DUR}s)"
log "Events JSONL → $EVENTS_JSONL"
log "Summary JSON → $SUMMARY_JSON"
