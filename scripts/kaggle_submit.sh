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
# - macOS/BSD + GNU compatible; no hard dependency on `unzip`.
# -----------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# --- Pretty logging -----------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
tty_ok=false; [[ -t 1 ]] && tty_ok=true
if $tty_ok; then BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; NC=$'\033[0m'; else BOLD=""; GREEN=""; YELLOW=""; RED=""; NC=""; fi
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
  sed -n '1,200p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip)               ZIP_PATH="${2:?}"; shift 2 ;;
    --csv)               CSV_PATH="${2:?}"; shift 2 ;;
    --comp)              COMP="${2:?}"; shift 2 ;;
    --title)             TITLE="${2:-}"; shift 2 ;;
    --msg|--message)     MSG="${2:-}"; shift 2 ;;
    --dry-run)           DRY_RUN=1; shift ;;
    --no-build)          NO_BUILD=1; shift ;;
    --no-upload)         NO_UPLOAD=1; shift ;;
    --poll)              POLL=1; shift ;;
    --interval)          POLL_INTERVAL="${2:-30}"; shift 2 ;;
    --timeout)           POLL_TIMEOUT="${2:-1800}"; shift 2 ;;
    --ensure-single-csv) ENSURE_SINGLE_CSV=1; shift ;;
    --retries)           RETRIES="${2:-3}"; shift 2 ;;
    --backoff)           BACKOFF="${2:-2}"; shift 2 ;;
    -h|--help)           usage ;;
    *)                   die "Unknown argument: $1" ;;
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
# Python is needed for validation & (optional) build
if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; else die "Python not found"; fi
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
    warn "Command failed (rc=$rc). Retry ${attempt}/${tries} in ${backoff}s: $*"
    sleep "$backoff"
    attempt=$((attempt+1))
    backoff=$((backoff*2))
  done
}

# --- Build (if needed) --------------------------------------------------------
if [[ $NO_BUILD -eq 0 ]]; then
  if [[ ! -f "$ZIP_PATH" ]]; then
    info "Submission ZIP not found → building via spectramind submit…"
    ( cd "$ROOT_DIR" && "$PY" -m spectramind submit --config-name submit )
    if [[ ! -f "$ZIP_PATH" && -f "${ARTIFACTS_DIR}/submission.zip" ]]; then
      ZIP_PATH="${ARTIFACTS_DIR}/submission.zip"
    fi
  else
    info "Using existing ZIP: $ZIP_PATH"
  fi
else
  info "--no-build set; will not build if ZIP is missing."
fi

[[ -f "$ZIP_PATH" ]] || die "ZIP not found: $ZIP_PATH"

# --- ZIP contents sanity ------------------------------------------------------
# Works with or without `unzip`; prefers Python zipfile for portability.
ZIP_LIST="$(mktemp)"; trap 'rm -f "$ZIP_LIST"' EXIT
if command -v unzip >/dev/null 2>&1; then
  unzip -Z1 "$ZIP_PATH" > "$ZIP_LIST" || die "Failed to list ZIP contents"
else
  "$PY" - <<PY > "$ZIP_LIST" || die "Failed to list ZIP contents via python"
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], 'r') as z:
    for n in z.namelist():
        print(n)
PY
"$ZIP_PATH"
fi

CSV_COUNT="$(grep -Ec '\.csv$' "$ZIP_LIST" || true)"
CSV_IN_ZIP="$(grep -E '\.csv$' "$ZIP_LIST" | head -n1 || true)"
if (( ENSURE_SINGLE_CSV == 1 )); then
  (( CSV_COUNT == 1 )) || die "Expected exactly one CSV in ZIP, found $CSV_COUNT"
fi

# If no external CSV provided, try the first CSV from ZIP for checks
if [[ ! -f "$CSV_PATH" && -n "$CSV_IN_ZIP" ]]; then
  warn "CSV not found at $CSV_PATH; will sanity-check inner file: $CSV_IN_ZIP"
fi

# --- Quick CSV schema guards --------------------------------------------------
csv_check_file_py() {
  "$PY" - <<PY || return 1
import csv, sys
p=sys.argv[1]
with open(p, newline='') as f:
    r=csv.reader(f)
    hdr=next(r, [])
cols=len(hdr)
print("COLS:", cols)
print("FIRST:", hdr[0] if hdr else "")
PY
  "$1"
}

csv_check_zip_py() {
  if command -v unzip >/dev/null 2>&1; then
    "$PY" - <<'PY' "$ZIP_PATH" "$CSV_IN_ZIP" || return 1
import csv, io, subprocess, sys
zip_path, inner = sys.argv[1], sys.argv[2]
p = subprocess.run(["unzip","-p",zip_path, inner], check=True, capture_output=True)
r = csv.reader(io.StringIO(p.stdout.decode("utf-8", "ignore")))
hdr = next(r, [])
print("COLS:", len(hdr))
print("FIRST:", hdr[0] if hdr else "")
PY
  else
    "$PY" - <<'PY' "$ZIP_PATH" "$CSV_IN_ZIP" || return 1
import csv, io, sys, zipfile
zip_path, inner = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(inner, 'r') as f:
        s = io.TextIOWrapper(f, encoding='utf-8', errors='ignore')
        r = csv.reader(s)
        hdr = next(r, [])
        print("COLS:", len(hdr))
        print("FIRST:", hdr[0] if hdr else "")
PY
  fi
}

if [[ -f "$CSV_PATH" ]]; then
  OUT="$(csv_check_file_py "$CSV_PATH" 2>/dev/null || true)"
  if [[ -n "$OUT" ]]; then
    info "[CSV ] $OUT"
    COLS="$(awk -F': ' '/^COLS:/{print $2}' <<<"$OUT")"
    FIRST="$(awk -F': ' '/^FIRST:/{print $2}' <<<"$OUT")"
    [[ "$COLS" == "$EXPECTED_COLS" ]] || warn "Unexpected column count: got $COLS, expected $EXPECTED_COLS"
    [[ "$FIRST" == "$ID_COL" ]] || warn "First column is '$FIRST' (expected '$ID_COL')"
  else
    warn "CSV header check failed for $CSV_PATH"
  fi
elif [[ -n "$CSV_IN_ZIP" ]]; then
  OUT="$(csv_check_zip_py 2>/dev/null || true)"
  if [[ -n "$OUT" ]]; then
    info "[CSV-ZIP] $OUT"
    COLS="$(awk -F': ' '/^COLS:/{print $2}' <<<"$OUT")"
    [[ "$COLS" == "$EXPECTED_COLS" ]] || warn "Unexpected in-zip column count: got $COLS, expected $EXPECTED_COLS"
  else
    warn "Could not inspect CSV inside ZIP."
  fi
else
  warn "Skipping CSV sanity checks (no CSV provided or found in ZIP)."
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
  "$PY" - <<'PY'
import csv, os, sys, subprocess
comp=os.environ.get("SM_KAGGLE_COMP","")
msg=os.environ.get("SM_SUBMIT_MSG","")
if not comp:
  print("ERR: missing SM_KAGGLE_COMP", file=sys.stderr); sys.exit(2)
p=subprocess.run(["kaggle","competitions","submissions","-c",comp,"--csv"], capture_output=True, check=False, text=True)
if p.returncode!=0:
  print("ERR: kaggle CLI returned", p.returncode, file=sys.stderr); sys.exit(3)
rows=list(csv.DictReader(p.stdout.splitlines()))
# Prefer row whose Description contains our message (CLI usually keeps full text)
row=None
if msg:
  cand=[r for r in rows if msg in (r.get("Description","") or "")]
  if cand: row=cand[-1]
row=row or (rows[-1] if rows else {})
status=(row.get("Status") or row.get("status") or "").strip().lower()
score=(row.get("PublicScore") or row.get("publicScore") or row.get("Score") or "").strip()
submitted_at=(row.get("Date") or "").strip()
submission_id=(row.get("ref") or row.get("id") or "").strip()
print(f"STATUS:{status}")
print(f"SCORE:{score}")
print(f"DATE:{submitted_at}")
print(f"SUBID:{submission_id}")
PY
}

if [[ $POLL -eq 1 && $NO_UPLOAD -eq 0 && $DRY_RUN -eq 0 ]]; then
  need_cmd awk
  info "Polling for score (interval=${POLL_INTERVAL}s, timeout=${POLL_TIMEOUT}s)…"
  export SM_KAGGLE_COMP="$COMP"
  export SM_SUBMIT_MSG="$MSG"
  START_TS=$(date +%s)
  while :; do
    OUT="$(poll_once_py 2>/dev/null || true)"
    STATUS="$(awk -F':' '/^STATUS:/{print $2}' <<<"$OUT")"
    SCORE="$( awk -F':' '/^SCORE:/{print $2}' <<<"$OUT")"
    DATEF="$( awk -F':' '/^DATE:/{print $2}'  <<<"$OUT")"
    SUBID="$( awk -F':' '/^SUBID:/{print $2}' <<<"$OUT")"
    [[ -n "$STATUS" ]] || STATUS="unknown"
    info "Latest: status=$STATUS score=${SCORE:-NA} date=${DATEF:-NA} id=${SUBID:-NA}"
    case "$STATUS" in
      pending|queued|unknown|"") : ;;
      *) info "Evaluation finished: status=$STATUS score=${SCORE:-NA}"; break ;;
    esac
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