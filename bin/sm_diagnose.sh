#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — sm_diagnose.sh
# Wrapper for: python -m spectramind diagnose ...
# • Repo-root aware, Kaggle/CI-aware
# • Creates a timestamped report dir (override with --report)
# • Toggles for FFT/UMAP/SHAP; dataset split & sampling helpers
# • Pass-through Hydra overrides after `--`
# • Dry-run & JSON summary (for CI artifacts)
# ------------------------------------------------------------------------------
# Usage:
#   bin/sm_diagnose.sh [--report DIR] [--split {train|val|test}] [--limit N]
#                      [--fft 0|1] [--umap 0|1] [--shap 0|1]
#                      [--json] [--quiet] [--dry-run] [--strict]
#                      [--] <hydra overrides...>
#
# Examples:
#   # Default diagnostics (FFT+UMAP on val split) into timestamped dir
#   bin/sm_diagnose.sh
#
#   # Explicit report dir + SHAP on 128 samples with extra Hydra overrides
#   bin/sm_diagnose.sh --report outputs/reports/run_01 --split val --limit 128 --shap 1 -- \
#       model=baseline_v1 trainer.accumulate_grad_batches=2
#
# Exit codes:
#   0 = success
#   2 = bad arguments / missing python entrypoint
#   3 = diagnose run failed (or missing expected artifacts in --strict)
# ==============================================================================

set -Eeuo pipefail

# ---------- logging ------------------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

usage() {
  sed -n '1,120p' "${BASH_SOURCE[0]}" | awk '/^# ====/{flag=1;next}/^set -Eeuo/{flag=0}flag' | sed 's/^# \{0,1\}//'
}

trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"; exit 3' ERR

# ---------- args ---------------------------------------------------------------
REPORT_DIR=""                # default computed below
SPLIT="val"                  # train|val|test
LIMIT=""                     # int
DO_FFT=1
DO_UMAP=1
DO_SHAP=0
EMIT_JSON=0
QUIET="${QUIET:-0}"
DRYRUN=0
STRICT=0

# Collect pass-through after `--`
HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --report) REPORT_DIR="${2:-}"; shift 2 ;;
    --split)  SPLIT="${2:-}"; shift 2 ;;
    --limit)  LIMIT="${2:-}"; shift 2 ;;
    --fft)    DO_FFT="${2:-1}"; shift 2 ;;
    --umap)   DO_UMAP="${2:-1}"; shift 2 ;;
    --shap)   DO_SHAP="${2:-0}"; shift 2 ;;
    --json)   EMIT_JSON=1; shift ;;
    --quiet)  QUIET=1; shift ;;
    --dry-run) DRYRUN=1; shift ;;
    --strict) STRICT=1; shift ;;
    --)       shift; HYDRA_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ---------- env awareness ------------------------------------------------------
IS_KAGGLE=0
[[ -d "/kaggle" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] && IS_KAGGLE=1
IS_CI=0
[[ "${CI:-}" == "true" || "${GITHUB_ACTIONS:-}" == "true" ]] && IS_CI=1

# ---------- repo root detection ------------------------------------------------
repo_root() {
  if command -v git >/dev/null 2>&1; then
    if r="$(git rev-parse --show-toplevel 2>/dev/null || true)"; then
      [[ -n "$r" ]] && { printf "%s" "$r"; return; }
    fi
  fi
  local d="$PWD"
  while [[ "$d" != "/" ]]; do
    if [[ -e "$d/pyproject.toml" || -e "$d/dvc.yaml" || -d "$d/.git" ]]; then
      printf "%s" "$d"; return
    fi
    d="$(dirname "$d")"
  done
  printf "%s" "$PWD"
}
ROOT="$(repo_root)"
cd "$ROOT"

# ---------- optional env stack & root guard -----------------------------------
if [[ -x "bin/ensure_repo_root.sh" ]]; then bin/ensure_repo_root.sh --quiet || true; fi
if [[ -x "bin/apply_env_stack.sh" ]]; then bin/apply_env_stack.sh --quiet || true; fi

# ---------- sanity -------------------------------------------------------------
PY_EXE="${PYTHON:-${PYTHON3:-python3}}"
command -v "$PY_EXE" >/dev/null 2>&1 || { err "python3 not found"; exit 2; }

# Check entrypoint existence (best-effort)
if ! "$PY_EXE" - <<'PY' >/dev/null 2>&1; then
import importlib, sys
m = importlib.util.find_spec("spectramind")
sys.exit(1 if m is None else 0)
PY
then
  err "Python package 'spectramind' not importable (src not on PYTHONPATH?)"
  err "Tip: run from repo root, or ensure editable install: pip install -e ."
  exit 2
fi

# ---------- defaults -----------------------------------------------------------
timestamp_utc() { date -u +'%Y-%m-%dT%H-%M-%SZ'; }
if [[ -z "$REPORT_DIR" ]]; then
  REPORT_DIR="outputs/reports/diagnose_$(timestamp_utc)"
fi

# Guard split
case "$SPLIT" in
  train|val|test) : ;;
  *) warn "Unknown --split '$SPLIT' (using as-is for data.split)";;
esac

mkdir -p "$REPORT_DIR"
REPORT_DIR="$(cd "$REPORT_DIR" && pwd -P)"

log "Diagnose → $REPORT_DIR (split=$SPLIT, limit=${LIMIT:-all}, fft=$DO_FFT, umap=$DO_UMAP, shap=$DO_SHAP)"

# ---------- build hydra overrides ---------------------------------------------
OVR=()
OVR+=("diagnose.report_dir=$REPORT_DIR")
OVR+=("data.split=$SPLIT")
[[ -n "$LIMIT" ]] && OVR+=("diagnose.limit=$LIMIT")
OVR+=("diagnose.enable_fft=$DO_FFT")
OVR+=("diagnose.enable_umap=$DO_UMAP")
OVR+=("diagnose.enable_shap=$DO_SHAP")

# Prefer quiet JSON logging if config exists
if [[ -d "configs/logger" ]]; then
  OVR+=("logger=jsonl")
fi

# ---------- run (or dry-run) ---------------------------------------------------
CMD=( "$PY_EXE" -m spectramind diagnose "${OVR[@]}" "${HYDRA_ARGS[@]}" )
if [[ "$DRYRUN" -eq 1 ]]; then
  log "[dry-run] ${CMD[*]}"
  EXIT=0
else
  "${CMD[@]}"
  EXIT=$?
fi
[[ $EXIT -eq 0 ]] && ok "Diagnose run completed" || err "Diagnose returned exit code $EXIT"

# ---------- collect artifacts (best-effort) -----------------------------------
# Convention (adjusts gracefully if some are missing)
INDEX_HTML=""
FIGS=()
JSONL=""
METRICS=""
MANIFEST=""

[[ -f "$REPORT_DIR/index.html" ]] && INDEX_HTML="$REPORT_DIR/index.html"
mapfile -t maybe_figs < <(find "$REPORT_DIR" -maxdepth 2 -type f \( -name "*.png" -o -name "*.svg" \) 2>/dev/null | LC_ALL=C sort || true)
FIGS=("${maybe_figs[@]}")
[[ -f "$REPORT_DIR/events.jsonl" ]] && JSONL="$REPORT_DIR/events.jsonl"
[[ -f "$REPORT_DIR/metrics.json" ]] && METRICS="$REPORT_DIR/metrics.json"
[[ -f "$REPORT_DIR/manifest.json" ]] && MANIFEST="$REPORT_DIR/manifest.json"

# ---------- strict verification ------------------------------------------------
if (( STRICT )); then
  missing=0
  [[ -n "$INDEX_HTML" ]] || { warn "Missing index.html in report (STRICT)"; missing=1; }
  (( ${#FIGS[@]} > 0 )) || { warn "No figures produced (STRICT)"; missing=1; }
  [[ $missing -eq 0 && $EXIT -eq 0 ]] || { err "Strict mode: diagnostics incomplete"; exit 3; }
fi

# ---------- JSON summary (compact) --------------------------------------------
if (( EMIT_JSON )); then
  esc() { printf "%s" "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  printf '{'
  printf '"ok":%s,' $(( EXIT==0 ? 1 : 0 ))
  printf '"ci":%s,"kaggle":%s,' "$IS_CI" "$IS_KAGGLE"
  printf '"report_dir":"%s",' "$(esc "$REPORT_DIR")"
  printf '"index":"%s",' "$(esc "$INDEX_HTML")"
  printf '"events":"%s",' "$(esc "$JSONL")"
  printf '"metrics":"%s",' "$(esc "$METRICS")"
  printf '"manifest":"%s",' "$(esc "$MANIFEST")"
  printf '"figures":['
    for i in "${!FIGS[@]}"; do [[ $i -gt 0 ]] && printf ','; printf '"%s"' "$(esc "${FIGS[$i]}")"; done
  printf '],'
  printf '"split":"%s","limit":"%s","fft":%s,"umap":%s,"shap":%s' \
    "$(esc "$SPLIT")" "$(esc "${LIMIT:-all}")" "$DO_FFT" "$DO_UMAP" "$DO_SHAP"
  printf '}\n'
fi

exit $EXIT

