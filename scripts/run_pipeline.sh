#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — End-to-End Pipeline Runner
# -----------------------------------------------------------------------------
# Runs the canonical calibrate → train → predict → submit pipeline using the
# spectramind CLI. Safe for local dev, CI, and Kaggle kernels.
#
# Usage:
#   ./scripts/run_pipeline.sh [--no-calib] [--no-train] [--no-predict] [--no-submit]
#                             [--dry-run] [--log-file artifacts/logs/run.log]
#                             [--predict-cfg predict] [--submit-cfg submit]
#                             [config_name]
#
# Examples:
#   ./scripts/run_pipeline.sh                 # uses configs/train.yaml
#   ./scripts/run_pipeline.sh debug           # uses configs/debug.yaml
#   ./scripts/run_pipeline.sh --no-train      # skip training, run others
#   ./scripts/run_pipeline.sh --dry-run       # print the plan, do nothing
#
# Notes:
# - Fails fast on errors (set -euo pipefail).
# - Logs each stage with timestamps; records per-stage durations.
# - Writes JSONL audit to artifacts/run_events.jsonl.
# - Detects Kaggle vs local runtime for path safety.
# - All configs must be Hydra-compatible and live under configs/.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
CFG_NAME="train"                 # default primary config (calibrate/train)
PREDICT_CFG="predict"            # predict config-name
SUBMIT_CFG="submit"              # submit config-name
LOG_FILE="${LOG_FILE:-artifacts/logs/run.log}"
EVENTS_JSONL="artifacts/run_events.jsonl"

DO_CALIB=1
DO_TRAIN=1
DO_PREDICT=1
DO_SUBMIT=1
DRY_RUN=0

CLI="${CLI:-spectramind}"        # allow override if needed
EXTRA_ARGS="${SPECTRAMIND_EXTRA_ARGS:-}"  # pass-through flags

# --- Helpers ---------------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
now_secs() { date +%s; }

log() {
  echo -e "[ $(timestamp) ] [run_pipeline] $*"
}

err() {
  echo -e "[ $(timestamp) ] [run_pipeline][ERROR] $*" >&2
  exit 1
}

detect_env() {
  if [[ -d "/kaggle/input" ]]; then
    echo "kaggle"
  else
    echo "local"
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || err "Missing required command: $1"
}

ensure_paths() {
  mkdir -p "$(dirname "$LOG_FILE")"
  mkdir -p "$(dirname "$EVENTS_JSONL")"
  # Ensure artifacts folder exists for outputs/logs even in Kaggle
  mkdir -p artifacts
}

config_exists() {
  local name="$1"
  [[ -f "configs/${name}.yaml" ]] || [[ -d "configs/${name}" ]]
}

append_event() {
  # args: stage status start_epoch end_epoch duration_secs cfg env
  local stage="$1" status="$2" start="$3" end="$4" dur="$5" cfg="$6" env="$7"
  {
    printf '{'
    printf '"ts":"%s",' "$(timestamp)"
    printf '"stage":"%s",' "$stage"
    printf '"status":"%s",' "$status"
    printf '"start":%s,' "$start"
    printf '"end":%s,' "$end"
    printf '"duration_sec":%s,' "$dur"
    printf '"config":"%s",' "$cfg"
    printf '"env":"%s"' "$env"
    printf '}\n'
  } >> "$EVENTS_JSONL"
}

run_stage() {
  # args: stage_name command_string config_name
  local stage="$1" cmd="$2" cfg="$3"

  local start end dur
  start="$(now_secs)"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN: [$stage] $cmd"
    end="$start"; dur=0
    append_event "$stage" "skipped(dry-run)" "$start" "$end" "$dur" "$cfg" "$ENV_TYPE"
    return 0
  fi

  log "▶ $stage"
  log "cmd: $cmd"
  if eval "$cmd"; then
    end="$(now_secs)"; dur="$(( end - start ))"
    log "✓ $stage completed in ${dur}s"
    append_event "$stage" "ok" "$start" "$end" "$dur" "$cfg" "$ENV_TYPE"
  else
    end="$(now_secs)"; dur="$(( end - start ))"
    log "✗ $stage failed after ${dur}s"
    append_event "$stage" "failed" "$start" "$end" "$dur" "$cfg" "$ENV_TYPE"
    return 1
  fi
}

usage() {
  sed -n '1,60p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

# --- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-calib)   DO_CALIB=0; shift ;;
    --no-train)   DO_TRAIN=0; shift ;;
    --no-predict) DO_PREDICT=0; shift ;;
    --no-submit)  DO_SUBMIT=0; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --log-file)   LOG_FILE="${2:?}"; shift 2 ;;
    --predict-cfg) PREDICT_CFG="${2:?}"; shift 2 ;;
    --submit-cfg)  SUBMIT_CFG="${2:?}"; shift 2 ;;
    -h|--help)    usage ;;
    *)            CFG_NAME="$1"; shift ;;
  esac
done

# --- Bootstrap --------------------------------------------------------------
ENV_TYPE="$(detect_env)"
ensure_paths
require_cmd "$CLI"

# basic config checks
for n in "$CFG_NAME" "$PREDICT_CFG" "$SUBMIT_CFG"; do
  config_exists "$n" || err "Config '$n' not found under configs/ (expected configs/$n.yaml or directory)"
done

# Log to file (tee keeps console output too)
exec > >(tee -a "$LOG_FILE") 2>&1

log "Running SpectraMind V50 pipeline (env: $ENV_TYPE)"
log "Configs: calibrate/train=$CFG_NAME, predict=$PREDICT_CFG, submit=$SUBMIT_CFG"
[[ -n "$EXTRA_ARGS" ]] && log "Extra CLI args: $EXTRA_ARGS"
[[ "$DRY_RUN" -eq 1 ]] && log "Dry-run enabled (no commands will execute)"

# Trap for unexpected exit
trap 'code=$?; [[ $code -ne 0 ]] && log "Pipeline aborted (exit $code)"; exit $code' EXIT

# --- Pipeline ---------------------------------------------------------------
PIPE_START="$(now_secs)"

if [[ "$DO_CALIB" -eq 1 ]]; then
  run_stage "Calibration" "$CLI calibrate --config-name \"$CFG_NAME\" $EXTRA_ARGS" "$CFG_NAME"
else
  log "Skipping Calibration (--no-calib)"
  append_event "Calibration" "skipped" "$(now_secs)" "$(now_secs)" 0 "$CFG_NAME" "$ENV_TYPE"
fi

if [[ "$DO_TRAIN" -eq 1 ]]; then
  run_stage "Training" "$CLI train --config-name \"$CFG_NAME\" $EXTRA_ARGS" "$CFG_NAME"
else
  log "Skipping Training (--no-train)"
  append_event "Training" "skipped" "$(now_secs)" "$(now_secs)" 0 "$CFG_NAME" "$ENV_TYPE"
fi

if [[ "$DO_PREDICT" -eq 1 ]]; then
  run_stage "Prediction" "$CLI predict --config-name \"$PREDICT_CFG\" $EXTRA_ARGS" "$PREDICT_CFG"
else
  log "Skipping Prediction (--no-predict)"
  append_event "Prediction" "skipped" "$(now_secs)" "$(now_secs)" 0 "$PREDICT_CFG" "$ENV_TYPE"
fi

if [[ "$DO_SUBMIT" -eq 1 ]]; then
  run_stage "Submission packaging" "$CLI submit --config-name \"$SUBMIT_CFG\" $EXTRA_ARGS" "$SUBMIT_CFG"
else
  log "Skipping Submission (--no-submit)"
  append_event "Submission packaging" "skipped" "$(now_secs)" "$(now_secs)" 0 "$SUBMIT_CFG" "$ENV_TYPE"
fi

PIPE_END="$(now_secs)"
PIPE_DUR="$(( PIPE_END - PIPE_START ))"
log "Pipeline finished successfully ✅ (total ${PIPE_DUR}s)"
