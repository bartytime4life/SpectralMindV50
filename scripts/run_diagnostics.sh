#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Diagnostics Runner
# -----------------------------------------------------------------------------
# Generates diagnostics (FFT/UMAP/plots/HTML report) for a finished run.
# Works in both "pipeline-led" (runs spectramind diagnose) and "post-hoc"
# (given predictions + truth paths) modes.
#
# Usage:
#   ./scripts/run_diagnostics.sh [options] [-- ARGS...]
#
# Options:
#   -c, --config-name NAME   Hydra config for diagnose stage (default: diagnose)
#   -p, --pred PATH          Predictions CSV/Parquet to analyze (optional)
#   -t, --truth PATH         Ground-truth CSV/Parquet (optional)
#   -o, --outdir DIR         Output dir (default env-aware: artifacts/diagnostics)
#   -n, --name  BASENAME     Report base name (default: diag_YYYYmmdd_HHMMSS)
#   -N, --no-cli             Skip `spectramind diagnose` and do post-hoc only
#   -Q, --quiet              Less verbose output
#   -h, --help               Show this help
#
# Passthrough:
#   Everything after `--` is forwarded to `spectramind diagnose` as-is.
#
# Notes:
# - Fails fast on errors (set -euo pipefail).
# - Detects Kaggle & picks safe outdir (/kaggle/working/artifacts/diagnostics).
# - Tries to locate predictions automatically if not provided.
# - If available, runs `python -m spectramind.reports` to render a consolidated
#   HTML report; otherwise emits a directory of artifacts and a manifest.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ----------------------------------------------------------------
CFG_NAME="diagnose"
PRED_PATH=""
TRUTH_PATH=""
OUTDIR=""
BASENAME=""
QUIET="0"
SKIP_CLI="0"
EXTRA_ARGS=()

DEFAULT_OUTDIR_LOCAL="artifacts/diagnostics"
DEFAULT_OUTDIR_KAGGLE="/kaggle/working/artifacts/diagnostics"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { [ "$QUIET" = "1" ] || echo -e "[ $(timestamp) ] [diag] $*"; }
warn(){ echo -e "[ $(timestamp) ] [diag][WARN] $*" >&2; }
die() { echo -e "[ $(timestamp) ] [diag][ERROR] $*" >&2; exit 1; }

detect_env() {
  if [ -d "/kaggle/input" ]; then echo "kaggle"; else echo "local"; fi
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

ENV_TYPE="$(detect_env)"

usage() { sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//'; }

# --- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config-name) CFG_NAME="${2:-}"; shift 2 ;;
    -p|--pred)        PRED_PATH="${2:-}"; shift 2 ;;
    -t|--truth)       TRUTH_PATH="${2:-}"; shift 2 ;;
    -o|--outdir)      OUTDIR="${2:-}"; shift 2 ;;
    -n|--name)        BASENAME="${2:-}"; shift 2 ;;
    -N|--no-cli)      SKIP_CLI="1"; shift 1 ;;
    -Q|--quiet)       QUIET="1"; shift 1 ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; EXTRA_ARGS=("$@"); break ;;
    *)                warn "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# --- Resolve outdir/basename -------------------------------------------------
if [ -z "$OUTDIR" ]; then
  if [ "$ENV_TYPE" = "kaggle" ]; then OUTDIR="$DEFAULT_OUTDIR_KAGGLE"; else OUTDIR="$DEFAULT_OUTDIR_LOCAL"; fi
fi
mkdir -p "$OUTDIR"
if [ -z "$BASENAME" ]; then BASENAME="diag_$(date +%Y%m%d_%H%M%S)"; fi

RUN_DIR="$OUTDIR/$BASENAME"
mkdir -p "$RUN_DIR"

log "Environment: $ENV_TYPE"
log "Output dir : $RUN_DIR"

# --- Helpers to discover files ----------------------------------------------
autofind_predictions() {
  # typical places we produce artifacts
  for f in \
    "outputs/predictions.csv" \
    "outputs/predictions.parquet" \
    "artifacts/predictions.csv" \
    "artifacts/predictions.parquet" \
    "predictions/predictions.csv" \
    "predictions/predictions.parquet" \
    ; do
    [ -f "$f" ] && { echo "$f"; return; }
  done
  # fall back: first CSV or Parquet named like predictions
  local any
  any="$(find . -maxdepth 3 -type f \( -name '*pred*.csv' -o -name '*pred*.parquet' \) | head -n1 || true)"
  [ -n "$any" ] && echo "$any" || true
}

autofind_truth() {
  # Look for validation target or train labels
  for f in \
    "data/processed/val_labels.csv" \
    "data/processed/val_labels.parquet" \
    "data/processed/train_labels.csv" \
    "data/processed/train_labels.parquet" \
    ; do
    [ -f "$f" ] && { echo "$f"; return; }
  done
  # fallback search
  local any
  any="$(find data -maxdepth 3 -type f \( -name '*label*.csv' -o -name '*label*.parquet' \) | head -n1 || true)"
  [ -n "$any" ] && echo "$any" || true
}

# --- Run CLI diagnose (optional) --------------------------------------------
maybe_run_cli() {
  if [ "$SKIP_CLI" = "1" ]; then
    log "Skipping spectramind diagnose (post-hoc only)"
    return 0
  fi
  if ! has_cmd spectramind; then
    warn "spectramind CLI not found; skipping CLI diagnose"
    return 0
  fi
  log "Running spectramind diagnose --config-name $CFG_NAME ${EXTRA_ARGS[*]}"
  spectramind diagnose --config-name "$CFG_NAME" "${EXTRA_ARGS[@]}" || {
    warn "spectramind diagnose returned non-zero; continuing with post-hoc"
  }
}

# --- Lightweight CSV sanity --------------------------------------------------
csv_quick_check() {
  local f="$1"
  [ -s "$f" ] || die "File empty: $f"
  local hdr cols rows
  hdr="$(head -n1 "$f")"
  cols="$(echo "$hdr" | awk -F, '{print NF}')"
  rows="$(wc -l < "$f" | tr -d ' ')"
  log "Quick CSV check: $(basename "$f") -> cols=$cols rows=$rows"
  if [ "$rows" -lt 2 ]; then warn "CSV has header but no rows: $f"; fi
}

# --- Execute ---------------------------------------------------------------
maybe_run_cli

# If user didn’t pass predictions, try to find them
if [ -z "$PRED_PATH" ]; then
  PRED_PATH="$(autofind_predictions || true)"
fi
[ -n "$PRED_PATH" ] || warn "Predictions not provided & not found automatically."
[ -n "$PRED_PATH" ] && log "Predictions: $PRED_PATH"

# If user didn’t pass truth, try to find some label file
if [ -z "$TRUTH_PATH" ]; then
  TRUTH_PATH="$(autofind_truth || true)"
fi
[ -n "$TRUTH_PATH" ] && log "Truth     : $TRUTH_PATH"

# Quick sanity on CSVs if CSV present
case "$PRED_PATH" in *.csv) csv_quick_check "$PRED_PATH" || true ;; esac
case "$TRUTH_PATH" in *.csv) csv_quick_check "$TRUTH_PATH" || true ;; esac

# --- Post-hoc diagnostics orchestration -------------------------------------
# If your project exposes a diagnostic entrypoint, use it.
# 1) Preferred: python -m spectramind.reports generate ...
# 2) Fallback: just copy known artifacts + emit a manifest; attempt a light FFT/UMAP if helpers exist.

FOUND_REPORTS=0

run_reports_module() {
  if python -c "import spectramind.reports" >/dev/null 2>&1; then
    log "Using Python entrypoint: python -m spectramind.reports"
    set +e
    python - <<'PY'
import os, sys, json, argparse, importlib
from pathlib import Path

outdir   = os.environ.get("RUN_DIR")
pred     = os.environ.get("PRED_PATH") or ""
truth    = os.environ.get("TRUTH_PATH") or ""

# Try to import a well-known API; fallback to CLI-ish main
mod = importlib.import_module("spectramind.reports")
ok  = False
if hasattr(mod, "generate_report"):
    try:
        mod.generate_report(predictions=pred or None, ground_truth=truth or None, outdir=outdir)
        ok = True
    except Exception as e:
        print(f"[diag][WARN] generate_report failed: {e}", file=sys.stderr)

if not ok:
    if hasattr(mod, "main"):
        try:
            # Best-effort: pass common flags if supported
            argv = ["--outdir", outdir]
            if pred:  argv += ["--predictions", pred]
            if truth: argv += ["--truth", truth]
            sys.argv = ["spectramind.reports"] + argv
            mod.main()
            ok = True
        except Exception as e:
            print(f"[diag][WARN] reports.main failed: {e}", file=sys.stderr)

if not ok:
    print("[diag][WARN] reports API not compatible; returning non-zero", file=sys.stderr)
    sys.exit(1)
PY
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
      log "spectramind.reports completed"
      FOUND_REPORTS=1
    else
      warn "spectramind.reports unsuccessful"
    fi
  else
    warn "spectramind.reports not found; skipping"
  fi
}

# Export for python block
export RUN_DIR PRED_PATH TRUTH_PATH
run_reports_module || true

# fallback: best-effort copy and manifest
if [ "$FOUND_REPORTS" -eq 0 ]; then
  log "Fallback: copying artifacts & writing manifest"
  mkdir -p "$RUN_DIR/assets"
  # Copy likely plots if present
  for d in outputs artifacts; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 2 -type f \( -name '*.png' -o -name '*.svg' -o -name '*.html' \) -print0 | \
      xargs -0 -I{} cp -n "{}" "$RUN_DIR/assets/" || true
  done

  # manifest
  git_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  git_rev="$(git -C "$git_root" rev-parse --short HEAD 2>/dev/null || echo "nogit")"
  git_status="$(git -C "$git_root" status --porcelain 2>/dev/null || echo "")"
  dirty="clean"; [ -n "$git_status" ] && dirty="dirty"

  cat > "$RUN_DIR/diagnostics_manifest.json" <<EOF
{
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "environment": "$ENV_TYPE",
  "cli_config": "$CFG_NAME",
  "predictions": "$(realpath "$PRED_PATH" 2>/dev/null || echo "$PRED_PATH")",
  "ground_truth": "$(realpath "$TRUTH_PATH" 2>/dev/null || echo "$TRUTH_PATH")",
  "git": {
    "root": "$(realpath "$git_root" 2>/dev/null || echo "$git_root")",
    "revision": "$git_rev",
    "state": "$dirty"
  },
  "notes": "Fallback manifest; consider implementing spectramind.reports.generate_report() for a full HTML dashboard."
}
EOF
fi

# zip result for convenience
ZIP_PATH="$RUN_DIR.zip"
if command -v zip >/dev/null 2>&1; then
  (cd "$(dirname "$RUN_DIR")" && zip -q -r "$(basename "$ZIP_PATH")" "$(basename "$RUN_DIR")")
  log "Diagnostics bundle -> $ZIP_PATH"
else
  warn "zip not found; skipping archive"
fi

log "Diagnostics completed ✅"
