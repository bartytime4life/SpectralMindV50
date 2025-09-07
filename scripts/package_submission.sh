#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Submission Packager
# -----------------------------------------------------------------------------
# Builds a Kaggle-ready submission bundle by:
#   1) Running the submit stage (unless a CSV is provided)
#   2) Performing lightweight validation (column count & header sanity)
#   3) Emitting a manifest.json with provenance details
#   4) Packaging into a versioned zip under artifacts/
#
# Usage:
#   ./scripts/package_submission.sh [options]
#
# Options:
#   -c, --config-name NAME   Hydra config for submit stage (default: submit)
#   -f, --file PATH          Use an existing submission CSV (skip submit)
#   -o, --outdir DIR         Output directory (default: artifacts)
#   -n, --name BASENAME      Zip base name, no extension
#                            (default: submission_YYYYmmdd_HHMMSS)
#   -S, --skip-validate      Skip CSV validation checks
#   -q, --quiet              Less verbose logging
#   -h, --help               Show help and exit
#
# Notes:
# - Fails fast on any error (set -euo pipefail).
# - Detects Kaggle vs local; defaults outdir accordingly.
# - Expected CSV columns: 1 id + 283 mu_* + 283 sigma_* = 567 total.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ----------------------------------------------------------------
CFG_NAME="submit"
SUBMISSION_FILE=""
OUTDIR=""
BASENAME=""
SKIP_VALIDATE="0"
QUIET="0"

EXPECTED_COLS=567
ID_COL="sample_id"          # expected ID column name
MU_PREFIX="mu_"
SIGMA_PREFIX="sigma_"

# repo-aware defaults
DEFAULT_OUTDIR_LOCAL="artifacts"
DEFAULT_OUTDIR_KAGGLE="/kaggle/working/artifacts"

# --- Helpers -----------------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { [ "$QUIET" = "1" ] || echo -e "[ $(timestamp) ] [package_submission] $*"; }
warn() { echo -e "[ $(timestamp) ] [package_submission][WARN] $*" >&2; }
die() { echo -e "[ $(timestamp) ] [package_submission][ERROR] $*" >&2; exit 1; }

detect_env() {
  if [ -d "/kaggle/input" ]; then echo "kaggle"; else echo "local"; fi
}
has_cmd() { command -v "$1" >/dev/null 2>&1; }

usage() {
  sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//'
}

# --- Argparse ----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config-name) CFG_NAME="${2:-}"; shift 2 ;;
    -f|--file)        SUBMISSION_FILE="${2:-}"; shift 2 ;;
    -o|--outdir)      OUTDIR="${2:-}"; shift 2 ;;
    -n|--name)        BASENAME="${2:-}"; shift 2 ;;
    -S|--skip-validate) SKIP_VALIDATE="1"; shift 1 ;;
    -q|--quiet)       QUIET="1"; shift 1 ;;
    -h|--help)        usage; exit 0 ;;
    *) warn "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

ENV_TYPE="$(detect_env)"
if [ -z "$OUTDIR" ]; then
  OUTDIR="$([ "$ENV_TYPE" = "kaggle" ] && echo "$DEFAULT_OUTDIR_KAGGLE" || echo "$DEFAULT_OUTDIR_LOCAL")"
fi
mkdir -p "$OUTDIR"

if [ -z "$BASENAME" ]; then
  BASENAME="submission_$(date +%Y%m%d_%H%M%S)"
fi

# --- Locate / Produce submission.csv ----------------------------------------
if [ -z "$SUBMISSION_FILE" ]; then
  # Run submit stage
  if ! has_cmd spectramind; then
    die "spectramind CLI not found on PATH."
  fi
  log "Running submit stage with config: $CFG_NAME"
  spectramind submit --config-name "$CFG_NAME"

  # Common locations to scan for output
  CANDIDATES=(
    "$OUTDIR/submission.csv"
    "outputs/submission.csv"
    "artifacts/submission.csv"
    "submission.csv"
    "$(git rev-parse --show-toplevel 2>/dev/null || pwd)/outputs/submission.csv"
    "$(git rev-parse --show-toplevel 2>/dev/null || pwd)/artifacts/submission.csv"
  )
  FOUND=""
  for f in "${CANDIDATES[@]}"; do
    [ -f "$f" ] && FOUND="$f" && break
  done
  if [ -z "$FOUND" ]; then
    # fallback: find by name in repo
    FOUND="$(find . -maxdepth 3 -type f -name 'submission*.csv' | head -n1 || true)"
  fi
  [ -n "$FOUND" ] || die "Unable to locate submission CSV after submit stage."
  SUBMISSION_FILE="$FOUND"
else
  [ -f "$SUBMISSION_FILE" ] || die "Provided file not found: $SUBMISSION_FILE"
fi

log "Using submission CSV: $SUBMISSION_FILE"

# --- Lightweight Validation --------------------------------------------------
validate_csv() {
  local path="$1"

  # 1) Non-empty
  [ -s "$path" ] || die "Submission CSV is empty: $path"

  # 2) Header checks
  local header
  header="$(head -n1 "$path")"
  [ -n "$header" ] || die "Cannot read header from CSV."

  # Column count
  local cols
  cols="$(echo "$header" | awk -F, '{print NF}')"
  if [ "$cols" -ne "$EXPECTED_COLS" ]; then
    die "Unexpected column count in header: got $cols, expected $EXPECTED_COLS"
  fi

  # ID col name sanity
  local first_col
  first_col="$(echo "$header" | awk -F, '{print $1}')"
  if [ "$first_col" != "$ID_COL" ]; then
    warn "First column is '$first_col' (expected '$ID_COL'). Continuing, but verify your schema."
  fi

  # Prefix sanity (quick substring check)
  echo "$header" | grep -q "$MU_PREFIX" || warn "No '${MU_PREFIX}*' columns found in header."
  echo "$header" | grep -q "$SIGMA_PREFIX" || warn "No '${SIGMA_PREFIX}*' columns found in header."

  # 3) At least one data row
  local nrows
  nrows="$(wc -l < "$path" | tr -d ' ')"
  if [ "$nrows" -lt 2 ]; then
    die "CSV has header but no data rows."
  fi

  # 4) Optional: size limit sanity (prevent empty)
  local bytes
  bytes="$(wc -c < "$path" | tr -d ' ')"
  if [ "$bytes" -lt 100 ]; then
    warn "CSV file size is very small ($bytes bytes); verify contents."
  fi

  log "Validation passed: columns=$cols, rows=$nrows, size=${bytes}B"
}

if [ "$SKIP_VALIDATE" = "0" ]; then
  log "Validating CSV structure"
  validate_csv "$SUBMISSION_FILE"
else
  log "Skipping CSV validation as requested"
fi

# --- Manifest ----------------------------------------------------------------
manifest_path="$OUTDIR/${BASENAME}_manifest.json"
git_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
git_rev="$(git -C "$git_root" rev-parse --short HEAD 2>/dev/null || echo "nogit")"
git_status="$(git -C "$git_root" status --porcelain 2>/dev/null || echo "")"
dirty="clean"
[ -n "$git_status" ] && dirty="dirty"

# sha256
SHA256_TOOL="$(command -v sha256sum || command -v shasum || true)"
if [ -n "$SHA256_TOOL" ]; then
  if [[ "$SHA256_TOOL" =~ shasum$ ]]; then
    csv_sha256="$("$SHA256_TOOL" -a 256 "$SUBMISSION_FILE" | awk '{print $1}')"
  else
    csv_sha256="$("$SHA256_TOOL" "$SUBMISSION_FILE" | awk '{print $1}')"
  fi
else
  csv_sha256="unavailable"
  warn "sha256 tool not found; manifest sha256 set to 'unavailable'."
fi

log "Writing manifest → $manifest_path"
cat > "$manifest_path" <<EOF
{
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "environment": "$ENV_TYPE",
  "config_name": "$CFG_NAME",
  "csv_path": "$(realpath "$SUBMISSION_FILE" 2>/dev/null || echo "$SUBMISSION_FILE")",
  "csv_sha256": "$csv_sha256",
  "git": {
    "root": "$(realpath "$git_root" 2>/dev/null || echo "$git_root")",
    "revision": "$git_rev",
    "state": "$dirty"
  },
  "expected_columns": $EXPECTED_COLS,
  "id_column": "$ID_COL",
  "mu_prefix": "$MU_PREFIX",
  "sigma_prefix": "$SIGMA_PREFIX"
}
EOF

# --- Zip ---------------------------------------------------------------------
zip_path="$OUTDIR/${BASENAME}.zip"
log "Packaging bundle → $zip_path"

# Prefer zip; fallback to Python zipfile
if has_cmd zip; then
  ( cd "$(dirname "$SUBMISSION_FILE")" && zip -q -j "$zip_path" "$(basename "$SUBMISSION_FILE")" )
  ( cd "$(dirname "$manifest_path")" && zip -q -j "$zip_path" "$(basename "$manifest_path")" )
else
  warn "zip not found; using Python zipfile fallback."
  python - "$zip_path" "$SUBMISSION_FILE" "$manifest_path" <<'PYZ'
import sys, zipfile, os
zip_path, sub_csv, manifest = sys.argv[1], sys.argv[2], sys.argv[3]
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(sub_csv, arcname=os.path.basename(sub_csv))
    z.write(manifest, arcname=os.path.basename(manifest))
print("Wrote", zip_path)
PYZ
fi

log "Submission bundle ready ✅"
log "ZIP: $zip_path"
