#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — sm_preprocess.sh
# Wrapper for: python -m spectramind preprocess ...
# • Repo-root aware, Kaggle/CI-aware
# • Timestamped output dir by default (override with --out)
# • Presets and granular toggles (normalize/detrend/binning/masks)
# • Export format control (npz/parquet), overwrite & manifest knobs
# • Pass-through Hydra overrides after `--`
# • Dry-run & compact JSON summary for CI
# ------------------------------------------------------------------------------
# Usage:
#   bin/sm_preprocess.sh
#   bin/sm_preprocess.sh [--out DIR] [--split {train|val|test}] [--limit N]
#                        [--preset fast|nominal|custom]
#                        [--format npz|parquet] [--overwrite 0|1]
#                        [--normalize 0|1] [--detrend poly|savgol|none]
#                        [--binning none|auto|FGS1] [--masks 0|1]
#                        [--workers N] [--seed N] [--manifest 0|1]
#                        [--json] [--quiet] [--dry-run] [--strict]
#                        [--] <hydra overrides...>
#
# Examples:
#   # Fast CI preset (tiny slice), NPZ export
#   bin/sm_preprocess.sh --preset fast --format npz
#
#   # Nominal run to Parquet with 8 workers, overwrite outputs
#   bin/sm_preprocess.sh --preset nominal --format parquet --workers 8 --overwrite 1
#
# Exit codes:
#   0 = success
#   2 = bad arguments / environment problem
#   3 = run failed OR (STRICT) required artifacts missing
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
SPLIT="train"                 # train|val|test
LIMIT=""
PRESET=""
FORMAT="npz"                  # npz|parquet
OVERWRITE=0
NORMALIZE=1
DETREND="poly"                # poly|savgol|none
BINNING="auto"                # none|auto|FGS1
MASKS=1
WORKERS=""
SEED=""
MANIFEST=1
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
    --preset)     PRESET="${2:-}"; shift 2 ;;
    --format)     FORMAT="${2:-}"; shift 2 ;;
    --overwrite)  OVERWRITE="${2:-0}"; shift 2 ;;
    --normalize)  NORMALIZE="${2:-1}"; shift 2 ;;
    --detrend)    DETREND="${2:-poly}"; shift 2 ;;
    --binning)    BINNING="${2:-auto}"; shift 2 ;;
    --masks)      MASKS="${2:-1}"; shift 2 ;;
    --workers)    WORKERS="${2:-}"; shift 2 ;;
    --seed)       SEED="${2:-}"; shift 2 ;;
    --manifest)   MANIFEST="${2:-1}"; shift 2 ;;
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

# Optional helpers
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
  OUT_DIR="outputs/features/preprocess_$(timestamp_utc)"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"

# ---------- build Hydra overrides ---------------------------------------------
OVR=()
OVR+=("preprocess.output_dir=$OUT_DIR")
OVR+=("data.split=$SPLIT")
[[ -n "$LIMIT"    ]] && OVR+=("preprocess.limit=$LIMIT")
[[ -n "$PRESET"   ]] && OVR+=("+preset=$PRESET")               # Hydra preset path (e.g., /preprocess/presets/fast)
OVR+=("io.format=$FORMAT")
OVR+=("io.overwrite=$OVERWRITE")
OVR+=("normalize.enable=$NORMALIZE")
case "$DETREND" in
  poly|savgol|none) OVR+=("detrend.mode=$DETREND");;
  *) warn "Unknown --detrend '$DETREND' (using as-is)"; OVR+=("detrend.mode=$DETREND");;
esac
case "$BINNING" in
  none|auto|FGS1) OVR+=("binning.mode=$BINNING");;
  *) warn "Unknown --binning '$BINNING' (using as-is)"; OVR+=("binning.mode=$BINNING");;
esac
OVR+=("masks.enable=$MASKS")
[[ -n "$WORKERS"  ]] && OVR+=("runtime.num_workers=$WORKERS")
[[ -n "$SEED"     ]] && OVR+=("seed=$SEED")
OVR+=("export.manifest.enable=$MANIFEST")

# Quiet JSON logging if config exists
if [[ -d "configs/logger" ]]; then
  OVR+=("logger=jsonl")
fi

# ---------- run (or dry-run) ---------------------------------------------------
CMD=( "$PY_EXE" -m spectramind preprocess "${OVR[@]}" "${HYDRA_ARGS[@]}" )
log "Preprocess → $OUT_DIR (split=$SPLIT, preset=${PRESET:-<none>}, fmt=$FORMAT, overwrite=$OVERWRITE)"
if [[ "$DRYRUN" -eq 1 ]]; then
  log "[dry-run] ${CMD[*]}"
  EXIT=0
else
  "${CMD[@]}"
  EXIT=$?
fi
[[ $EXIT -eq 0 ]] && ok "Preprocess run completed" || err "Preprocess returned exit code $EXIT"

# ---------- collect artifacts --------------------------------------------------
COUNT_NPZ=0
COUNT_PARQ=0
EVENTS=""
MANIFEST_PATH=""
SNAPSHOT_CFG=""

if [[ -d "$OUT_DIR" ]]; then
  COUNT_NPZ="$(find "$OUT_DIR" -type f -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')"
  COUNT_PARQ="$(find "$OUT_DIR" -type f -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')"
  [[ -f "$OUT_DIR/events.jsonl"      ]] && EVENTS="$OUT_DIR/events.jsonl"
  [[ -f "$OUT_DIR/manifest.json"     ]] && MANIFEST_PATH="$OUT_DIR/manifest.json"
  # common pattern for Hydra: .hydra/config.yaml snapshot under run dir
  if [[ -d "$OUT_DIR/.hydra" && -f "$OUT_DIR/.hydra/config.yaml" ]]; then
    SNAPSHOT_CFG="$OUT_DIR/.hydra/config.yaml"
  fi
fi

# ---------- strict verification ------------------------------------------------
if (( STRICT )); then
  missing=0
  case "$FORMAT" in
    npz)     (( COUNT_NPZ > 0 )) || { warn "STRICT: expected NPZ outputs, found none"; missing=1; } ;;
    parquet) (( COUNT_PARQ > 0 )) || { warn "STRICT: expected Parquet outputs, found none"; missing=1; } ;;
  esac
  [[ -n "$MANIFEST_PATH" ]] || { warn "STRICT: manifest.json missing"; missing=1; }
  if (( missing == 1 || EXIT != 0 )); then
    err "Strict mode: preprocess artifacts incomplete"
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
  printf '"format":"%s","overwrite":%s,' "$(esc "$FORMAT")" "$OVERWRITE"
  printf '"count_npz":%s,"count_parquet":%s,' "$COUNT_NPZ" "$COUNT_PARQ"
  printf '"events":"%s","manifest":"%s","hydra_config":"%s",' \
    "$(esc "$EVENTS")" "$(esc "$MANIFEST_PATH")" "$(esc "$SNAPSHOT_CFG")"
  printf '"split":"%s","limit":"%s","preset":"%s",' \
    "$(esc "$SPLIT")" "$(esc "${LIMIT:-all}")" "$(esc "${PRESET:-}")"
  printf '"normalize":%s,"detrend":"%s","binning":"%s","masks":%s,' \
    "$NORMALIZE" "$(esc "$DETREND")" "$(esc "$BINNING")" "$MASKS"
  printf '"workers":"%s","seed":"%s"' \
    "$(esc "${WORKERS:-auto}")" "$(esc "${SEED:-auto}")"
  printf '}\n'
fi

exit $EXIT

