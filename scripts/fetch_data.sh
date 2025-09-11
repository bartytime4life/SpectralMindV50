#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# fast_diagnose.sh — Run SpectraMind V50 diagnostics quickly, in isolation.
# Portable (Linux/macOS), Hydra-aware, optional DVC, great logs.
# ------------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# --- Resolve paths (script dir → repo root) ---
SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_SOURCE")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

# --- Colors (TTY only) ---
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; NC=$'\033[0m'
else
  BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; NC=""
fi

# --- Defaults ---
CONFIG_NAME="diagnose"
CONFIG_DIR=""
HYDRA_RUN_DIR=""
ENV_NAME=""
REPORTS_DIR=""
CHECKPOINT=""
ENV_FILE=""
DRY_RUN=false
QUIET=false
VERBOSE=false
USE_DVC=false
MAX_LOG_LINES=80
OVERRIDES=()
EXTRA_ARGS=()

# --- Python resolution (prefer python3) ---
if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; else echo "${RED}Python not found in PATH.${NC}" >&2; exit 127; fi

usage() {
  cat <<EOF
${BOLD}fast_diagnose.sh${NC} — quick diagnostics (Hydra + optional DVC)

${BOLD}Usage${NC}
  $0 [options] [--] [extra-args...]

${BOLD}Options${NC}
  -c, --config NAME          Hydra config name (default: diagnose)
  --config-dir DIR           Hydra config directory (passed to --config-path)
  --hydra-run-dir DIR        Set hydra.run.dir override to DIR
  -e, --env NAME             Env profile, e.g. local|kaggle (as +env=NAME)
  -r, --reports-dir PATH     Diagnostics artifacts dir (default: artifacts/diagnostics)
  -C, --checkpoint PATH      Model checkpoint (also forwards to common keys)
  -o, --override STR         Hydra override (repeatable), e.g. -o "predict.batch_size=64"
  --env-file PATH            Preload env vars from PATH (exported before run)
  --use-dvc                  Use 'dvc repro -s <config>' instead of Python entrypoint
  --max-log-lines N          Log lines to display on failure (default: 80)
  -n, --dry-run              Print command; do not run
  -q, --quiet                Minimal stdout (still logs to file)
  -v, --verbose              Verbose + HYDRA_FULL_ERROR=1
  -h, --help                 Show this help

${BOLD}Examples${NC}
  $0
  $0 -e kaggle -r artifacts/diagnostics -o "predict.batch_size=64"
  $0 -C artifacts/models/best.ckpt -o "predict.checkpoint=artifacts/models/best.ckpt"
  $0 --config-dir configs --hydra-run-dir artifacts/diag_runs/$(date +%Y%m%d)
  $0 --use-dvc

${BOLD}Notes${NC}
- In DVC mode, overrides here don't auto-apply unless your dvc.yaml stage reads params.
- Extra args after '--' are forwarded verbatim to the underlying command.
EOF
}

log()      { $QUIET || printf "%s\n" "$*"; }
log_ok()   { $QUIET || printf "%s\n" "${GREEN}$*${NC}"; }
log_warn() { $QUIET || printf "%s\n" "${YELLOW}$*${NC}"; }
log_err()  { printf "%s\n" "${RED}$*${NC}" >&2; }

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)        CONFIG_NAME="${2:-}"; shift 2 ;;
    --config-dir)       CONFIG_DIR="${2:-}"; shift 2 ;;
    --hydra-run-dir)    HYDRA_RUN_DIR="${2:-}"; shift 2 ;;
    -e|--env)           ENV_NAME="${2:-}"; shift 2 ;;
    -r|--reports-dir)   REPORTS_DIR="${2:-}"; shift 2 ;;
    -C|--checkpoint)    CHECKPOINT="${2:-}"; shift 2 ;;
    -o|--override)      OVERRIDES+=("${2:-}"); shift 2 ;;
    --env-file)         ENV_FILE="${2:-}"; shift 2 ;;
    --use-dvc)          USE_DVC=true; shift ;;
    --max-log-lines)    MAX_LOG_LINES="${2:-80}"; shift 2 ;;
    -n|--dry-run)       DRY_RUN=true; shift ;;
    -q|--quiet)         QUIET=true; shift ;;
    -v|--verbose)       VERBOSE=true; shift ;;
    -h|--help)          usage; exit 0 ;;
    --)                 shift; EXTRA_ARGS+=("$@"); break ;;
    *)                  EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# --- Optional env preload ---
if [[ -n "$ENV_FILE" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC2046
    set -a; source "$ENV_FILE"; set +a
  else
    log_warn "--env-file specified but not found: $ENV_FILE"
  fi
fi

# --- Reports dir & logging setup ---
if [[ -z "${REPORTS_DIR}" ]]; then REPORTS_DIR="artifacts/diagnostics"; fi
mkdir -p -- "${REPORTS_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${REPORTS_DIR}/fast_diagnose_${TS}.log"

# --- Error trap prints log tail for fast triage ---
on_err() {
  local lineno="$1" cmd="$2"
  log_err "Failed at line ${lineno}: ${cmd}"
  if [[ -s "${LOG_FILE}" ]]; then
    echo "${DIM}---- Last ${MAX_LOG_LINES} log lines (${LOG_FILE}) ----${NC}" >&2
    tail -n "${MAX_LOG_LINES}" "${LOG_FILE}" >&2 || true
    echo "${DIM}------------------------------------------------------${NC}" >&2
  fi
}
trap 'on_err $LINENO "$BASH_COMMAND"' ERR

# --- DVC check if requested ---
if $USE_DVC && ! command -v dvc >/dev/null 2>&1; then
  log_err "--use-dvc specified but 'dvc' is not installed."
  exit 127
fi

# --- Checkpoint existence (warn only; Hydra may handle remote/URL) ---
if [[ -n "${CHECKPOINT}" && ! -e "${CHECKPOINT}" ]]; then
  log_warn "Checkpoint path does not exist locally: ${CHECKPOINT}"
fi

# --- Prefer spectramind CLI; fallback to module form ---
detect_python_entrypoint() {
  if "$PY" - <<'PY' 2>/dev/null
import importlib, sys
try:
    importlib.import_module("spectramind")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    printf 'spectramind_cli\n'
  elif "$PY" - <<'PY' 2>/dev/null
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

# --- Build base command ---
CMD=()
if $USE_DVC; then
  CMD=( dvc repro -s "${CONFIG_NAME}" )
else
  # Enable faulthandler & show warnings for better triage
  PYFLAGS=( -X faulthandler -W default )
  if [[ -n "$CONFIG_DIR" ]]; then
    HYDRA_PATH=( --config-path "${CONFIG_DIR}" )
  else
    HYDRA_PATH=()
  fi
  case "$ENTRYPOINT_KIND" in
    spectramind_cli) CMD=( "$PY" "${PYFLAGS[@]}" -m spectramind diagnose "${HYDRA_PATH[@]}" --config-name "${CONFIG_NAME}" ) ;;
    module_diagnose) CMD=( "$PY" "${PYFLAGS[@]}" -m spectramind.diagnose "${HYDRA_PATH[@]}" --config-name "${CONFIG_NAME}" ) ;;
    *)               CMD=( "$PY" "${PYFLAGS[@]}" -m spectramind diagnose "${HYDRA_PATH[@]}" --config-name "${CONFIG_NAME}" ) ;;
  esac
fi

# --- Assemble Hydra overrides (python mode only) ---
if [[ -n "${ENV_NAME}" ]]; then
  OVERRIDES+=( "+env=${ENV_NAME}" )
fi
if [[ -n "${REPORTS_DIR}" ]]; then
  OVERRIDES+=( "diagnostics.report_dir=${REPORTS_DIR}" )
fi
if [[ -n "${HYDRA_RUN_DIR}" ]]; then
  OVERRIDES+=( "hydra.run.dir=${HYDRA_RUN_DIR}" )
fi
if [[ -n "${CHECKPOINT}" ]]; then
  OVERRIDES+=( "predict.checkpoint=${CHECKPOINT}" "model.checkpoint=${CHECKPOINT}" )
fi

if $USE_DVC && ((${#OVERRIDES[@]} > 0)); then
  log_warn "Overrides provided with --use-dvc. Ensure dvc.yaml stage '${CONFIG_NAME}' consumes params; these CLI overrides are NOT auto-applied."
fi

if ! $USE_DVC; then
  for ov in "${OVERRIDES[@]:-}"; do CMD+=( "${ov}" ); done
fi

# --- Verbosity & hydra traces ---
if $VERBOSE; then export HYDRA_FULL_ERROR=1; set -x; fi

# --- Plan summary ---
log "${BOLD}SpectraMind V50 — fast diagnostics${NC}"
log "Repo: ${REPO_ROOT}"
log "Config: ${CONFIG_NAME}"
[[ -n "${CONFIG_DIR}"   ]] && log "Config dir: ${CONFIG_DIR}"
[[ -n "${HYDRA_RUN_DIR}" ]] && log "Hydra run dir: ${HYDRA_RUN_DIR}"
[[ -n "${ENV_NAME}"     ]] && log "Env: ${ENV_NAME}"
[[ -n "${REPORTS_DIR}"  ]] && log "Reports dir: ${REPORTS_DIR}"
[[ -n "${CHECKPOINT}"   ]] && log "Checkpoint: ${CHECKPOINT}"
[[ -n "${ENV_FILE}"     ]] && log "Env file: ${ENV_FILE}"
((${#OVERRIDES[@]:-0} > 0)) && log "Overrides: ${OVERRIDES[*]}"
((${#EXTRA_ARGS[@]:-0} > 0)) && log "Extra args: ${EXTRA_ARGS[*]}"
$USE_DVC && log "Mode: DVC repro -s ${CONFIG_NAME}"
log "Log file: ${LOG_FILE}"

# --- Dry-run ---
if $DRY_RUN; then
  log_warn "[Dry-run] Command:"
  printf "%q " "${CMD[@]}" "${EXTRA_ARGS[@]}"; printf "\n"
  exit 0
fi

# --- Execute with robust RC capture ---
START_TS=$(date +%s)
set +e
if $QUIET; then
  "${CMD[@]}" "${EXTRA_ARGS[@]}" >> "${LOG_FILE}" 2>&1
  RC=$?
else
  "${CMD[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
  RC=${PIPESTATUS[0]}
fi
set -e
END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

# --- Outcome ---
if [[ ${RC} -ne 0 ]]; then
  log_err "Diagnostics failed (exit ${RC}) after ${ELAPSED}s. See ${LOG_FILE}"
  exit ${RC}
fi

log_ok "Diagnostics completed successfully in ${ELAPSED}s."
log_ok "Artifacts & logs: ${REPORTS_DIR}"