#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Kaggle Submission Helper (Upgraded)
# -----------------------------------------------------------------------------
# Builds (if needed), validates, and submits the submission ZIP to Kaggle.
# Optional polling of submission status until a score appears (or times out).
#
# Usage:
#   scripts/kaggle_submit.sh [--zip artifacts/submission.zip]
#                            [--csv artifacts/submission.csv]
#                            [--comp neurips-2025-ariel-data-challenge]
#                            [--title "v0.1.2 run"] [--msg "notes..."]
#                            [--dry-run] [--no-build] [--no-upload]
#                            [--poll] [--interval 30] [--timeout 1800]
#
# Environment overrides:
#   KAGGLE_COMP     Competition slug (default below)
#   KAGGLE_ZIP      Path to submission zip
#   KAGGLE_CSV      Path to submission csv (for sanity checks)
#
# Notes:
# - Requires Kaggle CLI for upload (unless --no-upload).
# - If ZIP is missing and --no-build is not set, we invoke:
#       python -m spectramind submit --config-name submit
# - Produces JSONL audit: artifacts/kaggle_submit_events.jsonl
# - Performs quick schema guards (567 columns = 1 id + 283 mu + 283 sigma).
# - Can poll Kaggle submissions until status != pending (or timeout).
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
    echo "unavailable"
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

# Expected schema quick-check (1 id + 283 mu_* + 283 sigma_* = 567 columns)
EXPECTED_COLS=567
ID_COL="sample_id"
MU_PREFIX="mu_"
SIGMA_PREFIX="sigma_"

# --- Args --------------------------------------------------------------------
ZIP_PATH="$DEFAULT_ZIP"
CSV_PATH="$DEFAULT_CSV"
COMP="$DEFAULT_COMP"
TITLE=""
MSG=""
DRY_RUN=0
NO_BUILD=0
NO_UPLOAD=0
POLL=0
POLL_INTERVAL=30       # seconds
POLL_TIMEOUT=1800      # seconds

usage() {
  sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip)           ZIP_PATH="$2"; shift 2 ;;
    --csv)           CSV_PATH="$2"; shift 2 ;;
    --comp)          COMP="$2"; shift 2 ;;
    --title)         TITLE="$2"; shift 2 ;;
    --msg|--message) MSG="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --no-build)      NO_BUILD=1; shift ;;
    --no-upload)     NO_UPLOAD=1; shift ;;
    --poll)          POLL=1; shift ;;
    --interval)      POLL_INTERVAL="${2:-30}"; shift 2 ;;
    --timeout)       POLL_TIMEOUT="${2:-1800}"; shift 2 ;;
    -h|--help)       usage ;;
    *)               die "Unknown argument: $1" ;;
  esac
done

mkdir -p "$ARTIFACTS_DIR"

# --- Discover repo context ----------------------------------------------------
GIT_REF="$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo 'nogit')"
GIT_DIRTY="$(git -C "$ROOT_DIR" diff --quiet 2>/dev/null || echo '-dirty')"
VERSION="$( [[ -f "$VERSION_FILE" ]] && tr -d '[:space:]' < "$VERSION_FILE" || echo '0.0.0')"

if [[ -z "$TITLE" ]]; then
  TITLE="SpectraMind V50 ${VERSION} (${GIT_REF}${GIT_DIRTY})"
fi
if [[ -z "$MSG" ]]; then
  MSG="auto: ${TITLE} @ $(timestamp)"
fi

# --- Preflight checks ---------------------------------------------------------
need_cmd python
need_cmd zip

if [[ $NO_UPLOAD -eq 0 ]]; then
  need_cmd kaggle
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
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
      # Try common fallback location
      if [[ -f "${ARTIFACTS_DIR}/submission.zip" ]]; then
        ZIP_PATH="${ARTIFACTS_DIR}/submission.zip"
      else
        die "Build completed but submission ZIP not found at: $ZIP_PATH"
      fi
    fi
  else
    info "Using existing ZIP: $ZIP_PATH"
  fi
else
  info "--no-build set; will not build if ZIP is missing."
fi

[[ -f "$ZIP_PATH" ]] || die "ZIP not found: $ZIP_PATH"

# --- Derive CSV path if missing ----------------------------------------------
# If CSV is not present where expected, try to list ZIP to find a plausible CSV.
if [[ ! -f "$CSV_PATH" ]]; then
  if command -v unzip >/dev/null 2>&1; then
    CAND="$(unzip -Z1 "$ZIP_PATH" 2>/dev/null | grep -E '\.csv$' | head -n1 || true)"
    if [[ -n "$CAND" ]]; then
      warn "CSV not found at $CSV_PATH; ZIP contains: $CAND (will use for quick checks via unzip -p)"
      CSV_IN_ZIP="$CAND"
    else
      warn "No CSV alongside nor inside ZIP; skipping CSV sanity."
    fi
  else
    warn "unzip not found; cannot inspect ZIP contents for CSV."
  fi
fi

# --- Quick CSV schema guards --------------------------------------------------
csv_quick_check_file() {
  local path="$1"
  [[ -s "$path" ]] || die "CSV empty: $path"
  local header cols nrows
  header="$(head -n1 "$path")"
  cols="$(awk -F',' 'NR==1{print NF}' "$path")"
  nrows="$(wc -l < "$path" | tr -d ' ')"
  info "CSV: $(basename "$path") -> cols=$cols rows=$nrows"
  if [[ "$cols" -ne "$EXPECTED_COLS" ]]; then
    warn "Unexpected column count: got $cols, expected $EXPECTED_COLS"
  fi
  local first_col; first_col="$(echo "$header" | awk -F',' '{print $1}')"
  if [[ "$first_col" != "$ID_COL" ]]; then
    warn "First column is '$first_col' (expected '$ID_COL')."
  fi
  if ! echo "$header" | grep -q "$MU_PREFIX"; then
    warn "No '$MU_PREFIX*' columns detected in header."
  fi
  if ! echo "$header" | grep -q "$SIGMA_PREFIX"; then
    warn "No '$SIGMA_PREFIX*' columns detected in header."
  fi
}

csv_quick_check_zip() {
  local zip="$1" inner="$2"
  local header cols
  header="$(unzip -p "$zip" "$inner" | head -n1 || true)"
  [[ -n "$header" ]] || { warn "Cannot read header from CSV inside ZIP ($inner)"; return 0; }
  cols="$(echo "$header" | awk -F',' '{print NF}')"
  info "CSV(in-zip): $inner -> cols=$cols"
  if [[ "$cols" -ne "$EXPECTED_COLS" ]]; then
    warn "Unexpected column count (in-zip): got $cols, expected $EXPECTED_COLS"
  fi
}

if [[ -f "$CSV_PATH" ]]; then
  csv_quick_check_file "$CSV_PATH"
elif [[ -n "${CSV_IN_ZIP:-}" ]]; then
  csv_quick_check_zip "$ZIP_PATH" "$CSV_IN_ZIP"
else
  warn "Skipping CSV sanity checks."
fi

# --- ZIP stats ---------------------------------------------------------------
ZIP_SHA256="$(sha256_file "$ZIP_PATH")"
ZIP_SIZE=$(stat -c%s "$ZIP_PATH" 2>/dev/null || stat -f%z "$ZIP_PATH")
info "ZIP sha256: $ZIP_SHA256"
info "ZIP size  : ${ZIP_SIZE} bytes"

# --- Compose Kaggle command ---------------------------------------------------
SUBMIT_CMD=(kaggle competitions submit -c "$COMP" -f "$ZIP_PATH" -m "$MSG")

# --- Upload logic -------------------------------------------------------------
if [[ $NO_UPLOAD -eq 1 ]]; then
  info "--no-upload set; skipping Kaggle upload."
elif [[ $DRY_RUN -eq 1 ]]; then
  info "[DRY-RUN] Would run: ${SUBMIT_CMD[*]}"
else
  info "Submitting to Kaggle competition: $COMP"
  "${SUBMIT_CMD[@]}"
  info "Submitted. Check Kaggle submissions page for status."
fi

# --- Optional: Poll status until scored --------------------------------------
if [[ $POLL -eq 1 && $NO_UPLOAD -eq 0 && $DRY_RUN -eq 0 ]]; then
  need_cmd awk
  need_cmd grep
  info "Polling for score (interval=${POLL_INTERVAL}s, timeout=${POLL_TIMEOUT}s)…"
  START_TS=$(date +%s)
  while :; do
    # Kaggle CLI: list submissions as CSV
    SUBS_CSV="$(mktemp)"
    kaggle competitions submissions -c "$COMP" --csv > "$SUBS_CSV" || true
    # Try to match our message (exact), then take the newest row
    ROW="$(awk -F',' -v msg="$MSG" 'NR==1{header=$0; next} $0 ~ msg {print NR, $0}' "$SUBS_CSV" | tail -n1 | cut -d' ' -f2- || true)"
    # Fallback: just take latest row if message match fails
    if [[ -z "$ROW" ]]; then
      ROW="$(tail -n1 "$SUBS_CSV" || true)"
    fi

    STATUS="$(echo "$ROW" | awk -F',' '{print $6}' | tr -d '"' || true)"
    SCORE="$(echo "$ROW"  | awk -F',' '{print $7}' | tr -d '"' || true)"
    SUBMIT_DT="$(echo "$ROW" | awk -F',' '{print $2}' | tr -d '"' || true)"

    [[ -n "$STATUS" ]] || STATUS="unknown"
    info "Latest: status=$STATUS score=${SCORE:-NA} submitted_at=${SUBMIT_DT:-NA}"

    if [[ "$STATUS" != "pending" && "$STATUS" != "queued" && "$STATUS" != "unknown" ]]; then
      info "Evaluation finished: status=$STATUS score=${SCORE:-NA}"
      rm -f "$SUBS_CSV"
      break
    fi

    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))
    if (( ELAPSED >= POLL_TIMEOUT )); then
      warn "Poll timeout (${POLL_TIMEOUT}s) reached; exiting poll loop."
      rm -f "$SUBS_CSV"
      break
    fi
    sleep "$POLL_INTERVAL"
    rm -f "$SUBS_CSV" || true
  done
fi

# --- Audit trail --------------------------------------------------------------
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
  printf '"no_upload":%s,' "$NO_UPLOAD"
  printf '"poll":%s,' "$POLL"
  printf '"interval":%s,' "$POLL_INTERVAL"
  printf '"timeout":%s' "$POLL_TIMEOUT"
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
  echo "  • Upload      : DONE (see Kaggle for result)"
fi
if [[ $POLL -eq 1 && $NO_UPLOAD -eq 0 && $DRY_RUN -eq 0 ]]; then
  echo "  • Polling     : ENABLED (interval=${POLL_INTERVAL}s, timeout=${POLL_TIMEOUT}s)"
else
  echo "  • Polling     : DISABLED"
fi
echo "──────────────────────────────────────────────────────────────────────────────"
