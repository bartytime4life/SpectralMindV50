#!/usr/bin/env bash
# fast_diagnose.sh — Run SpectraMind V50 diagnostics quickly, in isolation.
#
# Examples:
#   ./scripts/fast_diagnose.sh
#   ./scripts/fast_diagnose.sh -e kaggle -r artifacts/diagnostics -o "training.lightning.fast=true"
#   ./scripts/fast_diagnose.sh -C artifacts/models/best.ckpt -o "predict.checkpoint=artifacts/models/best.ckpt"
#   ./scripts/fast_diagnose.sh --use-dvc
#
# Notes:
# - Defaults to Hydra config `diagnose`. Pass extra Hydra overrides via -o/--override.
# - If --use-dvc is given, we call `dvc repro -s diagnose` instead of python directly.
# - Keeps output terse; use --verbose for more detail.
#
# Safe bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

CONFIG_NAME="diagnose"
ENV_NAME=""
REPORTS_DIR=""
CHECKPOINT=""
DRY_RUN=false
QUIET=false
VERBOSE=false
USE_DVC=false
OVERRIDES=()

# colors
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; NC=$'\033[0m'
else
  BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; NC=""
fi

usage() {
  cat <<EOF
${BOLD}fast_diagnose.sh${NC} — Run diagnostics stage quickly (Hydra + optional DVC).

${BOLD}Usage${NC}
  ${BOLD}$0${NC} [options] [--] [extra-args...]

${BOLD}Options${NC}
  -c, --config NAME         Hydra config name (default: diagnose)
  -e, --env NAME            Env profile, e.g. local|kaggle (passed as +env=NAME)
  -r, --reports-dir PATH    Output directory for diagnostics artifacts (override-able)
  -C, --checkpoint PATH     Model checkpoint to use (you may also pass as a Hydra override)
  -o, --override STR        Hydra override (repeatable), e.g. -o "training.lightning.fast=true"
  --use-dvc                 Use 'dvc repro -s diagnose' instead of python entrypoint
  -n, --dry-run             Print the plan; do not run
  -q, --quiet               Minimal output
  -v, --verbose             Verbose output
  -h, --help                Show this help

${BOLD}Examples${NC}
  $0
  $0 -e kaggle -r artifacts/diagnostics -o "predict.batch_size=64"
  $0 -C artifacts/models/best.ckpt -o "predict.checkpoint=artifacts/models/best.ckpt"
  $0 --use-dvc

${BOLD}Notes${NC}
- This script assumes the repo structure described in the SpectraMind V50 scaffold.
- Pass any additional CLI args after '--' to forward them to the underlying command.
EOF
}

log() {
  $QUIET && return 0
  printf "%s\n" "$*"
}

log_ok()    { $QUIET || printf "%s\n" "${GREEN}$*${NC}"; }
log_warn()  { $QUIET || printf "%s\n" "${YELLOW}$*${NC}"; }
log_err()   { printf "%s\n" "${RED}$*${NC}" >&2; }

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)      CONFIG_NAME="${2:-}"; shift 2 ;;
    -e|--env)         ENV_NAME="${2:-}"; shift 2 ;;
    -r|--reports-dir) REPORTS_DIR="${2:-}"; shift 2 ;;
    -C|--checkpoint)  CHECKPOINT="${2:-}"; shift 2 ;;
    -o|--override)    OVERRIDES+=("${2:-}"); shift 2 ;;
    --use-dvc)        USE_DVC=true; shift ;;
    -n|--dry-run)     DRY_RUN=true; shift ;;
    -q|--quiet)       QUIET=true; shift ;;
    -v|--verbose)     VERBOSE=true; shift ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; break ;;
    *)                # Forward unknowns later
                      break ;;
  esac
done

EXTRA_ARGS=("$@")

cd "${REPO_ROOT}"

# Sanity checks
if ! command -v python &>/dev/null; then
  log_err "Python not found in PATH."
  exit 127
fi

if $USE_DVC && ! command -v dvc &>/dev/null; then
  log_err "--use-dvc specified but 'dvc' is not installed."
  exit 127
fi

# Build command
CMD=()
if $USE_DVC; then
  CMD=( dvc repro -s diagnose )
else
  CMD=( python -m spectramind diagnose --config-name "${CONFIG_NAME}" )
fi

# Hydra overrides
if [[ -n "${ENV_NAME}" ]]; then
  OVERRIDES+=( "+env=${ENV_NAME}" )
fi

if [[ -n "${REPORTS_DIR}" ]]; then
  # Common pattern: diagnostics/report_dir or similar; keep it generic:
  OVERRIDES+=( "diagnostics.report_dir=${REPORTS_DIR}" )
fi

if [[ -n "${CHECKPOINT}" ]]; then
  # Common override key names vary; we add two common forms safely:
  OVERRIDES+=( "predict.checkpoint=${CHECKPOINT}" )
  OVERRIDES+=( "model.checkpoint=${CHECKPOINT}" )
fi

# Expand overrides into args (for python flow)
if ! $USE_DVC; then
  for ov in "${OVERRIDES[@]:-}"; do
    CMD+=( "${ov}" )
  done
fi

# Verbosity toggles
if $VERBOSE; then
  set -x
fi

# Show plan
log "${BOLD}SpectraMind V50 — fast diagnostics${NC}"
log "Repo: ${REPO_ROOT}"
log "Config: ${CONFIG_NAME}"
[[ -n "${ENV_NAME}"     ]] && log "Env: ${ENV_NAME}"
[[ -n "${REPORTS_DIR}"  ]] && log "Reports dir: ${REPORTS_DIR}"
[[ -n "${CHECKPOINT}"   ]] && log "Checkpoint: ${CHECKPOINT}"
((${#OVERRIDES[@]:-0} > 0)) && log "Overrides: ${OVERRIDES[*]}"
((${#EXTRA_ARGS[@]:-0} > 0)) && log "Extra args: ${EXTRA_ARGS[*]}"
$USE_DVC && log "Mode: DVC repro -s diagnose"

if $DRY_RUN; then
  log_warn "[Dry-run] Command:"
  echo "${CMD[@]} ${EXTRA_ARGS[*]:-}"
  exit 0
fi

# Ensure reports dir
if [[ -n "${REPORTS_DIR}" ]]; then
  mkdir -p -- "${REPORTS_DIR}"
fi

# Timer
START_TS=$(date +%s)

# Run
set +e
"${CMD[@]}" "${EXTRA_ARGS[@]}" 2>&1 | ( $QUIET && cat > /dev/null || tee -a "${REPORTS_DIR:-.}/fast_diagnose.log" )
RC=${PIPESTATUS[0]}
set -e

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

if [[ ${RC} -ne 0 ]]; then
  log_err "Diagnostics failed (exit ${RC}) after ${ELAPSED}s."
  exit ${RC}
fi

log_ok "Diagnostics completed successfully in ${ELAPSED}s."
if [[ -n "${REPORTS_DIR}" ]]; then
  log_ok "Artifacts/logs: ${REPORTS_DIR}"
fi
