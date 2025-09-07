#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Submission Packager
# -----------------------------------------------------------------------------
# Builds a Kaggle-ready submission bundle by:
#   1) Running the submit stage (unless a CSV is provided)
#   2) Performing robust validation:
#        - header shape & exact column names
#        - row arity checks (every row has the same #fields as header)
#        - numeric checks (no NaN/Inf/blank for mu_*/sigma_*)
#   3) Emitting a manifest.json with provenance details (git, VERSION, sha256, env)
#   4) Packaging into a versioned zip under artifacts/
#   5) (Optional) Uploading via Kaggle CLI
#
# Usage:
#   ./scripts/package_submission.sh [options]
#
# Options:
#   -c, --config-name NAME   Hydra config for submit stage (default: submit)
#   -f, --file PATH          Use an existing submission CSV (skip submit)
#   -o, --outdir DIR         Output directory (default: artifacts or /kaggle/working/artifacts)
#   -n, --name BASENAME      Zip base name (no extension)
#                            default: submission_<VERSION|nogit>_<YYYYmmdd_HHMMSS>
#   -S, --skip-validate      Skip CSV validation checks
#   -q, --quiet              Less verbose logging
#       --auto-upload        Upload bundle via Kaggle CLI (requires -C)
#   -C, --competition SLUG   Kaggle competition slug (e.g. ariel-data-challenge-2025)
#   -m, --message MSG        Submission message (Kaggle upload)
#   -h, --help               Show help and exit
#
# Notes:
# - Fails fast on any error (set -euo pipefail).
# - Detects Kaggle vs local; defaults outdir accordingly.
# - Expected header: 1 id + 283 mu_* + 283 sigma_* = 567 total columns.
# - Exact column names enforced: mu_000..mu_282 and sigma_000..sigma_282 by default.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ----------------------------------------------------------------
CFG_NAME="submit"
SUBMISSION_FILE=""
OUTDIR=""
BASENAME=""
SKIP_VALIDATE="0"
QUIET="0"
AUTO_UPLOAD="0"
KAGGLE_COMPETITION=""
KAGGLE_MSG="SpectraMind V50 submission"

EXPECTED_COLS=567
ID_COL="sample_id"
MU_PREFIX="mu_"
SIGMA_PREFIX="sigma_"
N_BINS=283

DEFAULT_OUTDIR_LOCAL="artifacts"
DEFAULT_OUTDIR_KAGGLE="/kaggle/working/artifacts"

# --- Helpers -----------------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { [ "$QUIET" = "1" ] || echo -e "[ $(timestamp) ] [package_submission] $*"; }
warn() { echo -e "[ $(timestamp) ] [package_submission][WARN] $*" >&2; }
die() { echo -e "[ $(timestamp) ] [package_submission][ERROR] $*" >&2; exit 1; }

detect_env() { if [[ -d "/kaggle/input" ]]; then echo "kaggle"; else echo "local"; fi; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }
realpath_f() { python - <<'PY' "$1"
import os,sys; p=sys.argv[1]; print(os.path.realpath(p) if os.path.exists(p) else p)
PY
}

usage() { sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'; }

# --- Args --------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config-name)   CFG_NAME="${2:-}"; shift 2 ;;
    -f|--file)          SUBMISSION_FILE="${2:-}"; shift 2 ;;
    -o|--outdir)        OUTDIR="${2:-}"; shift 2 ;;
    -n|--name)          BASENAME="${2:-}"; shift 2 ;;
    -S|--skip-validate) SKIP_VALIDATE="1"; shift 1 ;;
    -q|--quiet)         QUIET="1"; shift 1 ;;
        --auto-upload)  AUTO_UPLOAD="1"; shift 1 ;;
    -C|--competition)   KAGGLE_COMPETITION="${2:-}"; shift 2 ;;
    -m|--message)       KAGGLE_MSG="${2:-}"; shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    *) warn "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

ENV_TYPE="$(detect_env)"
if [[ -z "$OUTDIR" ]]; then
  OUTDIR="$([ "$ENV_TYPE" = "kaggle" ] && echo "$DEFAULT_OUTDIR_KAGGLE" || echo "$DEFAULT_OUTDIR_LOCAL")"
fi
mkdir -p "$OUTDIR" "$OUTDIR/logs"

# --- Version/Git -------------------------------------------------------------
git_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
git_rev="$(git -C "$git_root" rev-parse --short HEAD 2>/dev/null || echo "nogit")"
git_status="$(git -C "$git_root" status --porcelain 2>/dev/null || echo "")"
git_dirty=$([[ -n "$git_status" ]] && echo "dirty" || echo "clean")
version_file="$git_root/VERSION"
repo_version=$([[ -f "$version_file" ]] && tr -d ' \n' < "$version_file" || echo "0.0.0")

if [[ -z "$BASENAME" ]]; then
  BASENAME="submission_${repo_version:-nogit}_$(date +%Y%m%d_%H%M%S)"
fi

# --- Locate / Produce submission.csv ----------------------------------------
if [[ -z "$SUBMISSION_FILE" ]]; then
  has_cmd spectramind || die "spectramind CLI not found on PATH."
  log "Running submit stage with config: $CFG_NAME"
  spectramind submit --config-name "$CFG_NAME"

  # Candidates to locate the CSV
  CANDIDATES=(
    "$OUTDIR/submission.csv"
    "outputs/submission.csv"
    "artifacts/submission.csv"
    "submission.csv"
    "$(git rev-parse --show-toplevel 2>/dev/null || pwd)/outputs/submission.csv"
    "$(git rev-parse --show-toplevel 2>/dev/null || pwd)/artifacts/submission.csv"
  )
  for f in "${CANDIDATES[@]}"; do
    [[ -f "$f" ]] && SUBMISSION_FILE="$f" && break
  done
  [[ -n "${SUBMISSION_FILE:-}" ]] || SUBMISSION_FILE="$(find . -maxdepth 3 -type f -name 'submission*.csv' | head -n1 || true)"
  [[ -n "${SUBMISSION_FILE:-}" ]] || die "Unable to locate submission CSV after submit stage."
else
  [[ -f "$SUBMISSION_FILE" ]] || die "Provided file not found: $SUBMISSION_FILE"
fi

log "Using submission CSV: $SUBMISSION_FILE"

# --- Validation --------------------------------------------------------------
# Build expected header exactly: sample_id, mu_000..mu_282, sigma_000..sigma_282
build_expected_header() {
  python - "$ID_COL" "$N_BINS" <<'PY'
import sys
id_col = sys.argv[1]; n=int(sys.argv[2])
mu = [f"mu_{i:03d}" for i in range(n)]
sg = [f"sigma_{i:03d}" for i in range(n)]
print(",".join([id_col] + mu + sg))
PY
}

validate_csv() {
  local path="$1"
  [[ -s "$path" ]] || die "Submission CSV is empty: $path"

  local header; header="$(head -n1 "$path")"
  [[ -n "$header" ]] || die "Cannot read header from CSV."

  local cols; cols="$(awk -F',' 'NR==1{print NF}' "$path")"
  [[ "$cols" -eq "$EXPECTED_COLS" ]] || die "Unexpected column count: got $cols, expected $EXPECTED_COLS"

  local first_col; first_col="$(awk -F',' 'NR==1{print $1}' "$path")"
  [[ "$first_col" == "$ID_COL" ]] || warn "First column '$first_col' != expected '$ID_COL'"

  # Exact name check
  local expected; expected="$(build_expected_header)"
  if [[ "$header" != "$expected" ]]; then
    warn "Header does not exactly match expected schema."
    # Show a short diff preview (requires python3)
    if has_cmd python; then
      python - <<'PY' "$header" "$expected"
import sys, difflib
a=sys.argv[1].split(",")
b=sys.argv[2].split(",")
for i,(x,y) in enumerate(zip(a,b)):
    if x!=y:
        print(f"  col {i+1}: got '{x}' expected '{y}'")
        if i>10 and i< len(a)-10: break
PY
    fi
  fi

  # Per-row arity check (first 100 rows + all rows if small)
  awk -F',' -v n="$cols" 'NR>1 { if (NF!=n) { printf("Row %d has %d fields (expected %d)\n", NR, NF, n); exit 42 } }' "$path" \
    || die "Row arity check failed (see message above)."

  # Numeric checks for mu_*/sigma_* (first 1000 rows)
  # Allow scientific notation; disallow empty/NaN/Inf.
  awk -F',' -v n="$cols" '
    BEGIN{ ok=1; }
    NR==1 { next }
    NR>1001 { exit }  # sample first 1000 lines
    {
      for(i=2;i<=n;i++){
        x=$i
        if (x=="" || x=="NaN" || x=="nan" || x=="INF" || x=="inf" || x=="+inf" || x=="-inf") { printf("Bad numeric at row %d col %d: '%s'\n", NR, i, x); ok=0; exit }
        # regex numeric: optional sign, digits, optional decimal, optional exponent
        if (x !~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) { printf("Non-numeric at row %d col %d: '%s'\n", NR, i, x); ok=0; exit }
      }
    }
    END{ if (!ok) exit 43 }
  ' "$path" || die "Numeric validation failed."

  # At least one data row
  local nrows; nrows="$(wc -l < "$path" | tr -d ' ')"
  [[ "$nrows" -ge 2 ]] || die "CSV has header but no data rows."

  # Soft size sanity
  local bytes; bytes="$(wc -c < "$path" | tr -d ' ')"
  [[ "$bytes" -ge 100 ]] || warn "CSV file size is very small ($bytes bytes); verify contents."

  log "Validation passed: columns=$cols, rows=$nrows, size=${bytes}B"
}

if [[ "$SKIP_VALIDATE" = "0" ]]; then
  log "Validating CSV structure and numerics"
  validate_csv "$SUBMISSION_FILE"
else
  log "Skipping CSV validation as requested"
fi

# --- Manifest ----------------------------------------------------------------
manifest_path="$OUTDIR/${BASENAME}_manifest.json"
sha_tool="$(command -v sha256sum || command -v shasum || true)"
if [[ -n "$sha_tool" ]]; then
  if [[ "$sha_tool" =~ shasum$ ]]; then
    csv_sha256="$("$sha_tool" -a 256 "$SUBMISSION_FILE" | awk '{print $1}')"
  else
    csv_sha256="$("$sha_tool" "$SUBMISSION_FILE" | awk '{print $1}')"
  fi
else
  csv_sha256="unavailable"; warn "sha256 tool not found; manifest sha256 set to 'unavailable'."
fi

log "Writing manifest → $manifest_path"
cat > "$manifest_path" <<EOF
{
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "environment": "$ENV_TYPE",
  "submit_config": "$CFG_NAME",
  "csv_path": "$(realpath_f "$SUBMISSION_FILE")",
  "csv_sha256": "$csv_sha256",
  "expected_columns": $EXPECTED_COLS,
  "id_column": "$ID_COL",
  "mu_prefix": "$MU_PREFIX",
  "sigma_prefix": "$SIGMA_PREFIX",
  "n_bins": $N_BINS,
  "git": {
    "root": "$(realpath_f "$git_root")",
    "revision": "$git_rev",
    "state": "$git_dirty"
  },
  "version_file": {
    "path": "$(realpath_f "$version_file")",
    "value": "$repo_version"
  }
}
EOF

# --- Zip ---------------------------------------------------------------------
zip_path="$OUTDIR/${BASENAME}.zip"
log "Packaging bundle → $zip_path"

if has_cmd zip; then
  ( cd "$(dirname "$SUBMISSION_FILE")" && zip -q -j "$zip_path" "$(basename "$SUBMISSION_FILE")" )
  ( cd "$(dirname "$manifest_path")"   && zip -q -j "$zip_path" "$(basename "$manifest_path")" )
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

# --- Optional Kaggle Upload --------------------------------------------------
if [[ "$AUTO_UPLOAD" = "1" ]]; then
  [[ -n "$KAGGLE_COMPETITION" ]] || die "--auto-upload requires --competition <slug>"
  has_cmd kaggle || die "Kaggle CLI not found."
  # Ensure credentials exist (env or ~/.kaggle/kaggle.json)
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
      die "Kaggle credentials not found (env KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)."
    fi
  fi
  log "Uploading to Kaggle competition: $KAGGLE_COMPETITION"
  kaggle competitions submit -c "$KAGGLE_COMPETITION" -f "$zip_path" -m "$KAGGLE_MSG"
  log "Upload submitted to Kaggle."
fi
