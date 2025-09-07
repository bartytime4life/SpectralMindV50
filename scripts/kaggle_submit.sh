#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Kaggle Submission Helper
# -----------------------------------------------------------------------------
# Builds (if needed) and submits the submission ZIP to the Kaggle competition.
# Safe for local dev and CI runners.
#
# Usage:
#   scripts/kaggle_submit.sh [--zip artifacts/submission.zip]
#                            [--csv artifacts/submission.csv]
#                            [--comp neurips-2025-ariel-data-challenge]
#                            [--title "v0.1.2 run"] [--msg "notes..."]
#                            [--dry-run] [--no-build] [--no-upload]
#
# Environment overrides:
#   KAGGLE_COMP                Competition slug (default below)
#   KAGGLE_ZIP                 Path to submission zip
#   KAGGLE_CSV                 Path to submission csv (for sanity checks)
#
# Notes:
# - Requires Kaggle CLI for upload (unless --no-upload).
# - If --no-build is omitted and the ZIP is missing, we invoke:
#       python -m spectramind submit --config-name submit
#   which should create artifacts/submission.zip and (optionally) CSV.
# - Produces a small JSONL audit at artifacts/kaggle_submit_events.jsonl.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Pretty logging -----------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
info()  { echo "[INFO ] $(timestamp) $*"; }
warn()  { echo "[WARN ] $(timestamp) $*" >&2; }
error() { echo "[ERROR] $(timestamp) $*" >&2; }
die()   { error "$@"; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die "No sha256 tool found (sha256sum/shasum)."
  fi
}

# --- Defaults ----------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts"
EVENTS_JSONL="${ARTIFACTS_DIR}/kaggle_submit_events.jsonl"

DEFAULT_COMP="${KAGGLE_COMP:-neurips-2025-ariel-data-challenge}"
DEFAULT_ZIP="${KAGGLE_ZIP:-${ARTIFACTS_DIR}/submission.zip}"
DEFAULT_CSV="${KAGGLE_CSV:-${ARTIFACTS_DIR}/submission.csv}"
VERSION_FILE="${ROOT_DIR}/VERSION"

# --- Args --------------------------------------------------------------------
ZIP_PATH="$DEFAULT_ZIP"
CSV_PATH="$DEFAULT_CSV"
COMP="$DEFAULT_COMP"
TITLE=""
MSG=""
DRY_RUN=0
NO_BUILD=0
NO_UPLOAD=0

usage() {
  sed -n '1,70p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip)         ZIP_PATH="$2"; shift 2 ;;
    --csv)         CSV_PATH="$2"; shift 2 ;;
    --comp)        COMP="$2"; shift 2 ;;
    --title)       TITLE="$2"; shift 2 ;;
    --msg|--message) MSG="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --no-build)    NO_BUILD=1; shift ;;
    --no-upload)   NO_UPLOAD=1; shift ;;
    -h|--help)     usage ;;
    *)             die "Unknown argument: $1" ;;
  esac
done

mkdir -p "$ARTIFACTS_DIR"

# --- Discover repo context ----------------------------------------------------
GIT_REF="$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo 'nogit')"
GIT_DIRTY="$(git -C "$ROOT_DIR" diff --quiet 2>/dev/null || echo '-dirty')"
VERSION="$( [[ -f "$VERSION_FILE" ]] && cat "$VERSION_FILE" || echo '0.0.0')"

if [[ -z "$TITLE" ]]; then
  TITLE="SpectraMind V50 ${VERSION} (${GIT_REF}${GIT_DIRTY})"
fi

# Compose default message if none provided
if [[ -z "$MSG" ]]; then
  MSG="auto: ${TITLE} @ $(timestamp)"
fi

# --- Preflight checks ---------------------------------------------------------
# Core tools
need_cmd python
need_cmd zip

if [[ $NO_UPLOAD -eq 0 ]]; then
  need_cmd kaggle
  # Kaggle API credentials
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    # Kaggle CLI also accepts ~/.kaggle/kaggle.json; we warn if env vars not set.
    if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
      die "Kaggle credentials not found (set KAGGLE_USERNAME/KAGGLE_KEY or create ~/.kaggle/kaggle.json)."
    fi
  fi
fi

# --- Build (if needed) --------------------------------------------------------
if [[ $NO_BUILD -eq 0 ]]; then
  if [[ ! -f "$ZIP_PATH" ]]; then
    info "Submission ZIP not found → building via spectramind submit…"
    ( cd "$ROOT_DIR"
      python -m spectramind submit --config-name submit
    )
    if [[ ! -f "$ZIP_PATH" ]]; then
      # Try a common fallback
      if [[ -f "${ARTIFACTS_DIR}/submission.zip" ]]; then
        ZIP_PATH="${ARTIFACTS_DIR}/submission.zip"
      else
        die "Build step completed but submission ZIP not found at: $ZIP_PATH"
      fi
    fi
  else
    info "Using existing ZIP: $ZIP_PATH"
  fi
else
  info "--no-build set; will not build if ZIP is missing."
fi

# --- Sanity checks on artifacts ----------------------------------------------
[[ -f "$ZIP_PATH" ]] || die "ZIP not found: $ZIP_PATH"

if [[ -f "$CSV_PATH" ]]; then
  # Quick CSV header sanity (optional, non-fatal if missing awk)
  if command -v awk >/dev/null 2>&1; then
    header_cols=$(awk -F',' 'NR==1{print NF}' "$CSV_PATH")
    if [[ "$header_cols" -lt 3 ]]; then
      warn "CSV appears to have very few columns (NF=$header_cols). Ensure it matches challenge schema."
    fi
  fi
else
  warn "CSV not found at $CSV_PATH (ok if your ZIP contains the CSV)."
fi

ZIP_SHA256="$(sha256_file "$ZIP_PATH")"
ZIP_SIZE=$(stat -c%s "$ZIP_PATH" 2>/dev/null || stat -f%z "$ZIP_PATH")
info "ZIP sha256: $ZIP_SHA256"
info "ZIP size  : ${ZIP_SIZE} bytes"

# --- Submit to Kaggle ---------------------------------------------------------
SUBMIT_CMD=(kaggle competitions submit -c "$COMP" -f "$ZIP_PATH" -m "$MSG")

if [[ $NO_UPLOAD -eq 1 ]]; then
  info "--no-upload set; skipping Kaggle upload."
elif [[ $DRY_RUN -eq 1 ]]; then
  info "[DRY-RUN] Would run: ${SUBMIT_CMD[*]}"
else
  info "Submitting to Kaggle competition: $COMP"
  "${SUBMIT_CMD[@]}"
  info "Submitted. Check the Kaggle competition submissions page for status."
fi

# --- Audit trail --------------------------------------------------------------
# Minimal JSONL event (no jq dependency required)
{
  printf '{'
  printf '"ts":"%s",'   "$(timestamp)"
  printf '"version":"%s",' "$VERSION"
  printf '"git_ref":"%s",' "$GIT_REF$GIT_DIRTY"
  printf '"competition":"%s",' "$COMP"
  printf '"zip_path":"%s",' "$ZIP_PATH"
  printf '"zip_sha256":"%s",' "$ZIP_SHA256"
  printf '"zip_size_bytes":%s,' "${ZIP_SIZE:-0}"
  printf '"csv_path":"%s",' "$CSV_PATH"
  printf '"title":%q,' "$TITLE"
  printf '"message":%q,' "$MSG"
  printf '"dry_run":%s,' "$DRY_RUN"
  printf '"no_build":%s,' "$NO_BUILD"
  printf '"no_upload":%s' "$NO_UPLOAD"
  printf '}\n'
} >> "$EVENTS_JSONL"

# --- Summary ------------------------------------------------------------------
echo
echo "──────────────────────────────────────────────────────────────────────────────"
echo " Kaggle submission summary"
echo "  • Competition : $COMP"
echo "  • Title       : $TITLE"
echo "  • Message     : $MSG"
echo "  • ZIP         : $ZIP_PATH"
echo "  • ZIP SHA256  : $ZIP_SHA256"
echo "  • Events log  : $EVENTS_JSONL"
if [[ $NO_UPLOAD -eq 1 ]]; then
  echo "  • Upload      : SKIPPED (--no-upload)"
elif [[ $DRY_RUN -eq 1 ]]; then
  echo "  • Upload      : DRY-RUN (not executed)"
else
  echo "  • Upload      : DONE (check Kaggle for result)"
fi
echo "──────────────────────────────────────────────────────────────────────────────"
