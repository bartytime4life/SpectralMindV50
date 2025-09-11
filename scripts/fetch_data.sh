#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# fast_diagnose.sh — Run SpectraMind V50 diagnostics quickly, in isolation.
# ------------------------------------------------------------------------------
# Examples:
#   ./scripts/fast_diagnose.sh
#   ./scripts/fast_diagnose.sh -e kaggle -r artifacts/diagnostics -o "training.lightning.fast=true"
#   ./scripts/fast_diagnose.sh -C artifacts/models/best.ckpt -o "predict.checkpoint=artifacts/models/best.ckpt"
#   ./scripts/fast_diagnose.sh --use-dvc
#
# Notes:
# - Defaults to Hydra config `diagnose`. Extra Hydra overrides via -o/--override.
# - If --use-dvc is given, we call `dvc repro -s diagnose` (warn if overrides set).
# - Terse by default; use --verbose for more detail.
# - Portable across Linux/macOS (BSD/GNU coreutils).
# ------------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# --- Resolve paths (script dir → repo root) ---
SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_SOURCE")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# --- Defaults ---
CONFIG_NAME="diagnose"
ENV_NAME=""
REPORTS_DIR=""
CHECKPOINT=""
DRY_RUN=false
QUIET=false
VERBOSE=false
USE_DVC=false
OVERRIDES=()
EXTRA_ARGS=()

# --- Colors (TTY only) ---
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
  -r, --reports-dir PATH    Output directory for diagnostics artifacts (default: artifacts/diagnostics)
  -C, --checkpoint PATH     Model checkpoint to use (also forwarded to common Hydra keys)
  -o, --override STR        Hydra override (repeatable), e.g. -o "training.lightning.fast=true"
  --use-dvc                 Use 'dvc repro -s diagnose' instead of python entrypoint
  -n, --dry-run             Print the plan; do not run
  -q, --quiet               Minimal output (still logs to file if reports-dir set)
  -v, --verbose             Verbose output (sets HYDRA_FULL_ERROR=1)
  -h, --help                Show this help

${BOLD}Examples${NC}
  $0
  $0 -e kaggle -r artifacts/diagnostics -o "predict.batch_size=64"
  $0 -C artifacts/models/best.ckpt -o "predict.checkpoint=artifacts/models/best.ckpt"
  $0 --use-dvc

${BOLD}Notes${NC}
- In DVC mode, overrides are NOT auto-applied unless your dvc.yaml stage consumes params; we warn if overrides are present.
- Extra args after '--' are forwarded verbatim to the underlying command.
EOF
}

log()       { $QUIET || printf "%s\n" "$*"; }
log_ok()    { $QUIET || printf "%s\n" "${GREEN}$*${NC}"; }
log_warn()  { $QUIET || printf "%s\n" "${YELLOW}$*${NC}"; }
log_err()   { printf "%s\n" "${RED}$*${NC}" >&2; }

# --- Parse args ---
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
    --)               shift; EXTRA_ARGS+=("$@"); break ;;
    *)                EXTRA_ARGS+=("$1"); shift ;;
  esac
done

cd "${REPO_ROOT}"

# --- Sanity checks ---
command -v python >/dev/null 2>&1 || { log_err "Python not found in PATH."; exit 127; }
if $USE_DVC && ! command -v dvc >/dev/null 2>&1; then
  log_err "--use-dvc specified but 'dvc' is not installed."
  exit 127
fi

# --- Reports dir & logging setup ---
if [[ -z "${REPORTS_DIR}" ]]; then
  REPORTS_DIR="artifacts/diagnostics"
fi
mkdir -p -- "${REPORTS_DIR}"

# robust timestamp (portable)
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${REPORTS_DIR}/fast_diagnose_${TS}.log"

# --- Build command ---
detect_python_entrypoint() {
  # Prefer "python -m spectramind diagnose"; fall back to "python -m spectramind.diagnose"
  if python - <<'PY' 2>/dev/null
import importlib, sys
ok = False
try:
    m = importlib.import_module("spectramind")
    # Heuristic: CLI with callable main? (best-effort)
    ok = hasattr(m, "__package__")
except Exception:
    pass
sys.exit(0 if ok else 1)
PY
  then
    printf 'spectramind_cli\n'
  elif python - <<'PY' 2>/dev/null
import importlib, sys
try:
    importlib.import_module("spectramind.diagnose")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    printf 'module_diagnose\n'
  else
    printf 'unknown\n'
  fi
}

ENTRYPOINT_KIND="$(detect_python_entrypoint)"
CMD=()
if $USE_DVC; then
  CMD=( dvc repro -s "${CONFIG_NAME}" )
else
  case "$ENTRYPOINT_KIND" in
    spectramind_cli)   CMD=( python -m spectramind diagnose --config-name "${CONFIG_NAME}" ) ;;
    module_diagnose)   CMD=( python -m spectramind.diagnose --config-name "${CONFIG_NAME}" ) ;;
    *)                 CMD=( python -m spectramind diagnose --config-name "${CONFIG_NAME}" ) ;; # best-effort default
  esac
fi

# --- Hydra overrides ---
if [[ -n "${ENV_NAME}" ]]; then
  OVERRIDES+=( "+env=${ENV_NAME}" )
fi
if [[ -n "${REPORTS_DIR}" ]]; then
  # Common override key; adapt to your config schema
  OVERRIDES+=( "diagnostics.report_dir=${REPORTS_DIR}" )
fi
if [[ -n "${CHECKPOINT}" ]]; then
  OVERRIDES+=( "predict.checkpoint=${CHECKPOINT}" "model.checkpoint=${CHECKPOINT}" )
fi

if $USE_DVC && ((${#OVERRIDES[@]} > 0)); then
  log_warn "Overrides provided but --use-dvc is set. Ensure your dvc.yaml stage '${CONFIG_NAME}' consumes params; overrides here won't auto-apply."
fi

# Only append overrides in python mode
if ! $USE_DVC; then
  for ov in "${OVERRIDES[@]:-}"; do
    CMD+=( "${ov}" )
  done
fi

# --- Verbosity ---
if $VERBOSE; then
  export HYDRA_FULL_ERROR=1
  set -x
fi

# --- Plan summary ---
log "${BOLD}SpectraMind V50 — fast diagnostics${NC}"
log "Repo: ${REPO_ROOT}"
log "Config: ${CONFIG_NAME}"
[[ -n "${ENV_NAME}"    ]] && log "Env: ${ENV_NAME}"
[[ -n "${REPORTS_DIR}" ]] && log "Reports dir: ${REPORTS_DIR}"
[[ -n "${CHECKPOINT}"  ]] && log "Checkpoint: ${CHECKPOINT}"
((${#OVERRIDES[@]:-0} > 0)) && log "Overrides: ${OVERRIDES[*]}"
((${#EXTRA_ARGS[@]:-0} > 0)) && log "Extra args: ${EXTRA_ARGS[*]}"
$USE_DVC && log "Mode: DVC repro -s ${CONFIG_NAME}"
log "Log file: ${LOG_FILE}"

if $DRY_RUN; then
  log_warn "[Dry-run] Command:"
  printf "%q " "${CMD[@]}" "${EXTRA_ARGS[@]}"; printf "\n"
  exit 0
fi

# --- Execute (capture RC reliably across pipe) ---
START_TS=$(date +%s)

# If QUIET, still write to logfile; if not, tee to both
set +e
if $QUIET; then
  "${CMD[@]}" "${EXTRA_ARGS[@]}" >> "${LOG_FILE}" 2>&1
  RC=$?
else
  # tee -a is safe even if file does not exist yet
  "${CMD[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
  RC=${PIPESTATUS[0]}
fi
set -e

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

if [[ ${RC} -ne 0 ]]; then
  log_err "Diagnostics failed (exit ${RC}) after ${ELAPSED}s."
  log_err "See: ${LOG_FILE}"
  exit ${RC}
fi

log_ok "Diagnostics completed successfully in ${ELAPSED}s."
log_ok "Artifacts & logs: ${REPORTS_DIR}"