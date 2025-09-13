#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — sm_predict.sh
# Wrapper for: python -m spectramind predict ...
# • Repo-root aware, Kaggle/CI-aware
# • Timestamped output dir by default (override with --out)
# • Optional checkpoint path, split/limit, and save modes (npz/csv)
# • Optional submission CSV build + lightweight header validation
# • Pass-through Hydra overrides after `--`
# • Dry-run & compact JSON summary for CI
# ------------------------------------------------------------------------------
# Usage:
#   bin/sm_predict.sh [--out DIR] [--split {train|val|test}] [--limit N]
#                     [--ckpt PATH] [--save-npz 0|1] [--save-csv 0|1]
#                     [--submit 0|1] [--submission PATH]
#                     [--schema PATH] [--header PATH]
#                     [--json] [--quiet] [--dry-run] [--strict]
#                     [--] <hydra overrides...>
#
# Examples:
#   # Default (val split) → outputs/predict/ts... ; NPZ+CSV on; no submission
#   bin/sm_predict.sh
#
#   # Test split, explicit ckpt, write submission with header check
#   bin/sm_predict.sh --split test --ckpt ckpts/best.ckpt --submit 1 \
#       --submission outputs/submissions/sm_v50.csv \
#       --header schemas/submission_header.csv
#
# Exit codes:
#   0 = success
#   2 = bad arguments / environment problem
#   3 = prediction failed OR (STRICT) required artifacts missing / header mismatch
# ==============================================================================

set -Eeuo pipefail

# ---------- logging ------------------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

usage() {
  sed -n '1,140p' "${BASH_SOURCE[0]}" | awk '/^# ====/{flag=1;next}/^set -Eeuo/{flag=0}flag' | sed 's/^# \{0,1\}//'
}

trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"; exit 3' ERR

# ---------- args ---------------------------------------------------------------
OUT_DIR=""
SPLIT="val"                # train|val|test
LIMIT=""
CKPT=""
SAVE_NPZ=1
SAVE_CSV=1
DO_SUBMIT=0
SUBMISSION_PATH=""         # e.g., outputs/submissions/sm_v50.csv
SCHEMA_JSON=""             # frictionless schema (optional hint to model)
HEADER_CSV=""              # exact header template for validation
EMIT_JSON=0
QUIET="${QUIET:-0}"
DRYRUN=0
STRICT=0

HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)        OUT_DIR="${2:-}"; shift 2 ;;
    --split)      SPLIT="${2:-}"; shift 2 ;;
    --limit)      LIMIT="${2:-}"; shift 2 ;;
    --ckpt)       CKPT="${2:-}"; shift 2 ;;
    --save-npz)   SAVE_NPZ="${2:-1}"; shift 2 ;;
    --save-csv)   SAVE_CSV="${2:-1}"; shift 2 ;;
    --submit)     DO_SUBMIT="${2:-0}"; shift 2 ;;
    --submission) SUBMISSION_PATH="${2:-}"; shift 2 ;;
    --schema)     SCHEMA_JSON="${2:-}"; shift 2 ;;
    --header)     HEADER_CSV="${2:-}"; shift 2 ;;
    --json)       EMIT_JSON=1; shift ;;
    --quiet)      QUIET=1; shift ;;
    --dry-run)    DRYRUN=1; shift ;;
    --strict)     STRICT=1; shift ;;
    --)           shift; HYDRA_ARGS+=("$@"); break ;;
    -h|--help)    usage; exit 0 ;;
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
if [[ -x "bin/ensure_repo_root.sh" ]]; then bin/ensure_repo_root.sh --quiet || true; fi
if [[ -x "bin/apply_env_stack.sh" ]]; then bin/apply_env_stack.sh --quiet || true; fi

# ---------- sanity -------------------------------------------------------------
PY_EXE="${PYTHON:-${PYTHON3:-python3}}"
command -v "$PY_EXE" >/dev/null 2>&1 || { err "python3 not found"; exit 2; }

# Verify module import
if ! "$PY_EXE" - <<'PY' >/dev/null 2>&1; then
import importlib, sys
m = importlib.util.find_spec("spectramind")
sys.exit(1 if m is None else 0)
PY
then
  err "Python package 'spectramind' not importable (run from repo root or pip install -e .)"
  exit 2
fi

timestamp_utc() { date -u +'%Y-%m-%dT%H-%M-%SZ'; }
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="outputs/predict/run_$(timestamp_utc)"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"

# Default submission path if requested but not set
if [[ "$DO_SUBMIT" -eq 1 && -z "$SUBMISSION_PATH" ]]; then
  mkdir -p "$ROOT/outputs/submissions"
  SUBMISSION_PATH="$ROOT/outputs/submissions/sm_v50_$(timestamp_utc).csv"
fi

# ---------- build hydra overrides ---------------------------------------------
OVR=()
OVR+=("predict.output_dir=$OUT_DIR")
OVR+=("data.split=$SPLIT")
[[ -n "$LIMIT" ]] && OVR+=("predict.limit=$LIMIT")
[[ -n "$CKPT"  ]] && OVR+=("predict.checkpoint=$CKPT")
OVR+=("predict.save_npz=$SAVE_NPZ")
OVR+=("predict.save_csv=$SAVE_CSV")
# If schema/header hints exist, pass them (model may or may not use)
[[ -n "$SCHEMA_JSON" ]] && OVR+=("submission.schema=$SCHEMA_JSON")
if [[ "$DO_SUBMIT" -eq 1 ]]; then
  OVR+=("submission.enable=true")
  OVR+=("submission.path=$SUBMISSION_PATH")
else
  OVR+=("submission.enable=false")
fi
# Quiet JSON logging if available
if [[ -d "configs/logger" ]]; then
  OVR+=("logger=jsonl")
fi

# ---------- run (or dry-run) ---------------------------------------------------
CMD=( "$PY_EXE" -m spectramind predict "${OVR[@]}" "${HYDRA_ARGS[@]}" )
log "Predict → $OUT_DIR (split=$SPLIT, limit=${LIMIT:-all}, ckpt=${CKPT:-<auto>}, npz=$SAVE_NPZ, csv=$SAVE_CSV, submit=$DO_SUBMIT)"
if [[ "$DRYRUN" -eq 1 ]]; then
  log "[dry-run] ${CMD[*]}"
  EXIT=0
else
  "${CMD[@]}"
  EXIT=$?
fi
[[ $EXIT -eq 0 ]] && ok "Predict run completed" || err "Predict returned exit code $EXIT"

# ---------- collect artifacts --------------------------------------------------
PRED_NPZ=""
PRED_CSV=""
EVENTS=""
METRICS=""
MANIFEST=""
SUBMISSION=""

# best-effort discovery
mapfile -t npzs < <(find "$OUT_DIR" -maxdepth 2 -type f -name "*.npz" 2>/dev/null | LC_ALL=C sort || true)
mapfile -t csvs < <(find "$OUT_DIR" -maxdepth 2 -type f -name "*.csv" 2>/dev/null | LC_ALL=C sort || true)
[[ -f "$OUT_DIR/events.jsonl" ]] && EVENTS="$OUT_DIR/events.jsonl"
[[ -f "$OUT_DIR/metrics.json" ]] && METRICS="$OUT_DIR/metrics.json"
[[ -f "$OUT_DIR/manifest.json" ]] && MANIFEST="$OUT_DIR/manifest.json"

# choose canonical predictions (first match)
[[ ${#npzs[@]} -gt 0 ]] && PRED_NPZ="${npzs[0]}"
# Distinguish submission CSV from intermediate csv
if [[ "$DO_SUBMIT" -eq 1 && -n "$SUBMISSION_PATH" && -f "$SUBMISSION_PATH" ]]; then
  SUBMISSION="$SUBMISSION_PATH"
fi
# if no explicit submission, pick first csv in OUT_DIR as prediction csv
if [[ -z "$SUBMISSION" && ${#csvs[@]} -gt 0 ]]; then
  PRED_CSV="${csvs[0]}"
fi

# ---------- header validation (lightweight) -----------------------------------
HEADER_OK=""
HEADER_ERR=""
try_header_check() {
  local csv="$1" hdr="$2"
  [[ -f "$csv" && -f "$hdr" ]] || return 0
  python3 - "$csv" "$hdr" <<'PY' || exit 11
import sys,csv
csv_path, hdr_path = sys.argv[1], sys.argv[2]
with open(hdr_path, 'r', newline='') as fh:
    header_line = fh.readline().rstrip('\n').rstrip('\r')
exp = [c.strip() for c in header_line.split(',')]
with open(csv_path, 'r', newline='') as fc:
    reader = csv.reader(fc)
    got = next(reader, [])
if exp != got:
    print("MISMATCH", file=sys.stderr)
    sys.exit(10)
PY
}
if [[ -n "$HEADER_CSV" ]]; then
  target_csv="${SUBMISSION:-$PRED_CSV}"
  if [[ -n "$target_csv" ]]; then
    if try_header_check "$target_csv" "$HEADER_CSV"; then
      HEADER_OK="1"
      ok "Header check OK against $HEADER_CSV"
    else
      HEADER_OK="0"
      HEADER_ERR="Header mismatch vs $HEADER_CSV"
      err "$HEADER_ERR"
    fi
  fi
fi

# ---------- strict verification ------------------------------------------------
if (( STRICT )); then
  missing=0
  # Must have at least one of NPZ or CSV predictions
  if [[ "$SAVE_NPZ" -eq 1 && -z "$PRED_NPZ" && "$SAVE_CSV" -eq 0 ]]; then
    warn "STRICT: expected NPZ predictions but none found"; missing=1
  fi
  if [[ "$SAVE_CSV" -eq 1 && -z "$PRED_CSV" && "$DO_SUBMIT" -eq 0 ]]; then
    warn "STRICT: expected prediction CSV but none found"; missing=1
  fi
  if [[ "$DO_SUBMIT" -eq 1 && -z "$SUBMISSION" ]]; then
    warn "STRICT: submission requested but file not found"; missing=1
  fi
  if [[ -n "$HEADER_CSV" && "$HEADER_OK" == "0" ]]; then
    warn "STRICT: header mismatch"
    missing=1
  end_if_strict=true
  fi
  if (( missing == 1 || EXIT != 0 )); then
    err "Strict mode: prediction artifacts incomplete"
    exit 3
  fi
fi

# ---------- JSON summary -------------------------------------------------------
if (( EMIT_JSON )); then
  esc() { printf "%s" "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  printf '{'
  printf '"ok":%s,' $(( EXIT==0 ? 1 : 0 ))
  printf '"ci":%s,"kaggle":%s,' "$IS_CI" "$IS_KAGGLE"
  printf '"out_dir":"%s",' "$(esc "$OUT_DIR")"
  printf '"npz":"%s",' "$(esc "$PRED_NPZ")"
  printf '"csv":"%s",' "$(esc "$PRED_CSV")"
  printf '"submission":"%s",' "$(esc "$SUBMISSION")"
  printf '"events":"%s","metrics":"%s","manifest":"%s",' \
    "$(esc "$EVENTS")" "$(esc "$METRICS")" "$(esc "$MANIFEST")"
  printf '"split":"%s","limit":"%s","ckpt":"%s",' \
    "$(esc "$SPLIT")" "$(esc "${LIMIT:-all}")" "$(esc "${CKPT:-}")"
  printf '"save_npz":%s,"save_csv":%s,"submit":%s,' "$SAVE_NPZ" "$SAVE_CSV" "$DO_SUBMIT"
  printf '"header_check":"%s"' "$(esc "${HEADER_OK:-unknown}")"
  printf '}\n'
fi

exit $EXIT

