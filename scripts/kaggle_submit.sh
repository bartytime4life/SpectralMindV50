#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Kaggle Submission Helper (Ultra Upgraded)
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
#                            [--ensure-single-csv] [--retries 3] [--backoff 2]
#
# Env overrides:
#   KAGGLE_COMP  → competition slug (default below)
#   KAGGLE_ZIP   → path to submission zip
#   KAGGLE_CSV   → path to submission csv (for sanity checks)
#
# Notes:
# - Requires Kaggle CLI for upload (unless --no-upload).
# - If ZIP is missing and --no-build is not set, we invoke:
#       python -m spectramind submit --config-name submit
# - Writes JSONL audit: artifacts/kaggle_submit_events.jsonl
# - Quick schema guards (567 columns = 1 id + 283 mu + 283 sigma).
# - Polls Kaggle submissions until status resolves (or timeout).
# - macOS/BSD + GNU compatible.
# -----------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# --- Pretty logging -----------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
info()  { printf "[INFO ] %s %s\n"  "$(timestamp)" "$*"; }
warn()  { printf "[WARN ] %s %s\n"  "$(timestamp)" "$*" >&2; }
error() { printf "[ERROR] %s %s\n"  "$(timestamp)" "$*" >&2; }
die()   { error "$*"; exit 1; }
trap 'error "Failed at line $LINENO: $BASH_COMMAND"; exit 1' ERR

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}';
  else echo "unavailable"; fi
}

stat_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }

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
POLL_INTERVAL=30
POLL_TIMEOUT=1800
ENSURE_SINGLE_CSV=0
RETRIES=3
BACKOFF=2

usage() {
  sed -n '1,140p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip)              ZIP_PATH="$2"; shift 2 ;;
    --csv)              CSV_PATH="$2"; shift 2 ;;
    --comp)             COMP="$2"; shift 2 ;;
    --title)            TITLE="$2"; shift 2 ;;
    --msg|--message)    MSG="$2"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    --no-build)         NO_BUILD=1; shift ;;
    --no-upload)        NO_UPLOAD=1; shift ;;
    --poll)             POLL=1; shift ;;
    --interval)         POLL_INTERVAL="${2:-30}"; shift 2 ;;
    --timeout)          POLL_TIMEOUT="${2:-1800}"; shift 2 ;;
    --ensure-single-csv)ENSURE_SINGLE_CSV=1; shift ;;
    --retries)          RETRIES="${2:-3}"; shift 2 ;;
    --backoff)          BACKOFF="${2:-2}"; shift 2 ;;
    -h|--help)          usage ;;
    *)                  die "Unknown argument: $1" ;;
  esac
done

mkdir -p "$ARTIFACTS_DIR"

# --- Discover repo context ----------------------------------------------------
GIT_REF="$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo 'nogit')"
GIT_DIRTY="$(git -C "$ROOT_DIR" diff --quiet 2>/dev/null || echo '-dirty')"
VERSION="$( [[ -f "$VERSION_FILE" ]] && tr -d '[:space:]' < "$VERSION_FILE" || echo '0.0.0')"

[[ -n "$TITLE" ]] || TITLE="SpectraMind V50 ${VERSION} (${GIT_REF}${GIT_DIRTY})"
[[ -n "$MSG"   ]] || MSG="auto: ${TITLE} @ $(timestamp)"

# --- Preflight checks ---------------------------------------------------------
need_cmd python
need_cmd zip

if [[ $NO_UPLOAD -eq 0 ]]; then
  need_cmd kaggle
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    [[ -f "${HOME}/.kaggle/kaggle.json" ]] || die "Kaggle creds not found (set KAGGLE_USERNAME/KAGGLE_KEY or create ~/.kaggle/kaggle.json)"
  fi
fi

# --- Retry wrapper ------------------------------------------------------------
run_with_retries() {
  local tries="$1"; shift
  local backoff="$1"; shift
  local attempt=1
  until "$@"; do
    local rc=$?
    if (( attempt >= tries )); then return "$rc"; fi
    warn "Command failed (rc=$rc). Retry $attempt/$tries in ${backoff}s: $*"
    sleep "$backoff"
    attempt=$((attempt+1))
    backoff=$((backoff*2))
  done
}

# --- Build (if needed) --------------------------------------------------------
if [[ $NO_BUILD -eq 0 ]]; then
  if [[ ! -f "$ZIP_PATH" ]]; then
    info "Submission ZIP not found → building via spectramind submit…"
    ( cd "$ROOT_DIR" && python -m spectramind submit --config-name submit )
    if [[ ! -f "$ZIP_PATH" ]]; then
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

# --- ZIP contents sanity ------------------------------------------------------
ZIP_LIST="$(mktemp)"; trap 'rm -f "$ZIP_LIST"' EXIT
if command -v unzip >/dev/null 2>&1; then
  unzip -Z1 "$ZIP_PATH" > "$ZIP_LIST" || die "Failed to list ZIP contents"
  CSV_COUNT="$(grep -Ec '\.csv$' "$ZIP_LIST" || true)"
  if (( ENSURE_SINGLE_CSV == 1 )); then
    (( CSV_COUNT == 1 )) || die "Expected exactly one CSV in ZIP, found $CSV_COUNT"
  fi
else
  warn "unzip not found; skipping ZIP content listing"
fi

# If no external CSV provided, try to pick first CSV from ZIP for checks
CSV_IN_ZIP=""
if [[ ! -f "$CSV_PATH" && -s "$ZIP_LIST" ]]; then
  CSV_IN_ZIP="$(grep -E '\.csv$' "$ZIP_LIST" | head -n1 || true)"
  [[ -n "$CSV_IN_ZIP" ]] && warn "CSV not found at $CSV_PATH; will sanity-check in-zip file: $CSV_IN_ZIP"
fi

# --- Quick CSV schema guards --------------------------------------------------
csv_check_file_py() {
  python - <<PY || return 1
import csv, sys, os
path=sys.argv[1]
with open(path, newline='') as f:
    r=csv.reader(f)
    header=next(r, [])
cols=len(header)
print("HEADER:", ",".join(header))
print("COLS:", cols)
print("FIRST_COL:", header[0] if header else "")
PY
}

csv_check_zip_py() {
  python - <<PY || return 1
import csv, sys, subprocess, io
zip_path, inner = sys.argv[1], sys.argv[2]
p = subprocess.run(["unzip","-p",zip_path, inner], check=True, capture_output=True)
s = io.StringIO(p.stdout.decode("utf-8", errors="ignore"))
r=csv.reader(s)
header=next(r, [])
print("HEADER:", ",".join(header))
print("COLS:", len(header))
print("FIRST_COL:", header[0] if header else "")
PY
}

csv_quick_check_file() {
  local path="$1"
  [[ -s "$path" ]] || die "CSV empty: $path"
  local out; out="$(csv_check_file_py "$path" 2>/dev/null || true)"
  if [[ -n "$out" ]]; then
    info "$out" | sed 's/^/[CSV ] /'
    local cols; cols="$(printf "%s\n" "$out" | awk -F': ' '/^COLS:/{print $2}')"
    local first_col; first_col="$(printf "%s\n" "$out" | awk -F': ' '/^FIRST_COL:/{print $2}')"
    if [[ "$cols" -ne "$EXPECTED_COLS" ]]; then warn "Unexpected column count: got $cols, expected $EXPECTED_COLS"; fi
    if [[ "$first_col" != "$ID_COL" ]]; then warn "First column is '$first_col' (expected '$ID_COL')"; fi
  else
    # Fallback (approx)
    local cols; cols="$(awk -F',' 'NR==1{print NF}' "$path")"
    info "[CSV ] Fallback header cols=$cols"
    [[ "$cols" -eq "$EXPECTED_COLS" ]] || warn "Unexpected column count: got $cols, expected $EXPECTED_COLS"
  fi
}

csv_quick_check_zip() {
  local zip="$1" inner="$2"
  local out; out="$(csv_check_zip_py "$zip" "$inner" 2>/dev/null || true)"
  if [[ -n "$out" ]]; then
    info "$out" | sed 's/^/[CSV-ZIP] /'
    local cols; cols="$(printf "%s\n" "$out" | awk -F': ' '/^COLS:/{print $2}')"
    if [[ "$cols" -ne "$EXPECTED_COLS" ]]; then warn "Unexpected column count (in-zip): got $cols, expected $EXPECTED_COLS"; fi
  else
    warn "Could not inspect CSV inside ZIP (need unzip+python)."
  fi
}

if [[ -f "$CSV_PATH" ]]; then
  csv_quick_check_file "$CSV_PATH"
elif [[ -n "$CSV_IN_ZIP" ]]; then
  csv_quick_check_zip "$ZIP_PATH" "$CSV_IN_ZIP"
else
  warn "Skipping CSV sanity checks."
fi

# --- ZIP stats ---------------------------------------------------------------
ZIP_SHA256="$(sha256_file "$ZIP_PATH")"
ZIP_SIZE="$(stat_size "$ZIP_PATH")"
info "ZIP sha256: $ZIP_SHA256"
info "ZIP size  : ${ZIP_SIZE} bytes"

# --- Compose Kaggle command ---------------------------------------------------
SUBMIT_CMD=(kaggle competitions submit -c "$COMP" -f "$ZIP_PATH" -m "$MSG")

# --- Upload logic with retries ------------------------------------------------
UPLOAD_RAN=0
if [[ $NO_UPLOAD -eq 1 ]]; then
  info "--no-upload set; skipping Kaggle upload."
elif [[ $DRY_RUN -eq 1 ]]; then
  info "[DRY-RUN] Would run: ${SUBMIT_CMD[*]}"
else
  info "Submitting to Kaggle competition: $COMP"
  if run_with_retries "$RETRIES" "$BACKOFF" "${SUBMIT_CMD[@]}"; then
    info "Submitted to Kaggle."
    UPLOAD_RAN=1
  else
    die "Kaggle submit failed after $RETRIES attempt(s)."
  fi
fi

# --- Optional: Poll status until scored --------------------------------------
poll_once_py() {
  python - <<'PY'
import csv, os, sys, subprocess
comp=os.environ.get("SM_KAGGLE_COMP","")
msg=os.environ.get("SM_SUBMIT_MSG","")
if not comp:
  print("ERR: missing SM_KAGGLE_COMP", file=sys.stderr); sys.exit(2)
p=subprocess.run(["kaggle","competitions","submissions","-c",comp,"--csv"], capture_output=True, check=False, text=True)
if p.returncode!=0:
  print("ERR: kaggle CLI returned", p.returncode, file=sys.stderr); sys.exit(3)
rows=list(csv.DictReader(p.stdout.splitlines()))
# Prefer row whose Description contains our exact message (can be truncated by Kaggle UI; CLI usually keeps full)
cand=[r for r in rows if msg and msg in (r.get("Description","") or "")]
row=(cand[-1] if cand else (rows[-1] if rows else {}))
status=(row.get("Status") or row.get("status") or "").lower()
score=(row.get("PublicScore") or row.get("publicScore") or row.get("Score") or "")
submitted_at=row.get("Date","")
print(f"STATUS:{status}")
print(f"SCORE:{score}")
print(f"DATE:{submitted_at}")
PY
}

if [[ $POLL -eq 1 && $NO_UPLOAD -eq 0 && $DRY_RUN -eq 0 ]]; then
  need_cmd awk
  need_cmd grep
  info "Polling for score (interval=${POLL_INTERVAL}s, timeout=${POLL_TIMEOUT}s)…"
  export SM_KAGGLE_COMP="$COMP"
  export SM_SUBMIT_MSG="$MSG"
  START_TS=$(date +%s)
  while :; do
    OUT="$(poll_once_py 2>/dev/null || true)"
    STATUS="$(printf "%s\n" "$OUT" | awk -F':' '/^STATUS:/{print $2}')"
    SCORE="$( printf "%s\n" "$OUT" | awk -F':' '/^SCORE:/{print $2}')"
    DATEF="$( printf "%s\n" "$OUT" | awk -F':' '/^DATE:/{print $2}')"
    [[ -n "$STATUS" ]] || STATUS="unknown"
    info "Latest: status=$STATUS score=${SCORE:-NA} submitted_at=${DATEF:-NA}"
    if [[ "$STATUS" != "pending" && "$STATUS" != "queued" && "$STATUS" != "unknown" ]]; then
      info "Evaluation finished: status=$STATUS score=${SCORE:-NA}"
      break
    fi
    NOW=$(date +%s)
    (( NOW - START_TS < POLL_TIMEOUT )) || { warn "Poll timeout reached"; break; }
    sleep "$POLL_INTERVAL"
  done
fi

# --- Audit trail --------------------------------------------------------------
mkdir -p "$(dirname "$EVENTS_JSONL")"
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
  printf '"upload_attempted":%s,' "$UPLOAD_RAN"
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
