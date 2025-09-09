#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Submission Packager (Upgraded)
# -----------------------------------------------------------------------------
# Builds a Kaggle-ready submission bundle by:
#   1) Running the submit stage (unless a CSV is provided)
#   2) Performing robust validation:
#        - header shape & exact column names (id + 283 mu + 283 sigma = 567)
#        - full-file row arity checks (every row has same #fields as header)
#        - numeric checks (no NaN/Inf/blank for mu_*/sigma_*, scientific ok)
#   3) Emitting manifest.json with provenance (git, VERSION, sha256, env) and
#      validation stats (rows, header hash, first/last IDs, sample checksums)
#   4) Packaging into a versioned zip under artifacts/
#   5) (Optional) Uploading via Kaggle CLI with retries/backoff
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
#       --dry-run            Show actions and exit (no writes/uploads)
#       --auto-upload        Upload bundle via Kaggle CLI (requires -C)
#   -C, --competition SLUG   Kaggle competition slug
#   -m, --message MSG        Submission message (Kaggle upload)
#       --retries N          Retries for Kaggle upload (default: 3)
#       --backoff SEC        Initial backoff seconds (default: 2, doubles each retry)
#       --allow-missing-cols Allow missing trailing columns (rare; for schema changes)
#       --id-col NAME        Override id column name (default: sample_id)
#       --bins N             Number of spectral bins (default: 283)
#   -h, --help               Show help and exit
#
# Notes:
# - Fails fast on any error (set -Eeuo pipefail).
# - Detects Kaggle vs local; defaults outdir accordingly.
# - Expected header: 1 id + N mu_* + N sigma_* → 2N+1 total columns.
# - Exact column names enforced: mu_000..mu_(N-1) and sigma_000..sigma_(N-1).
# -----------------------------------------------------------------------------

set -Eeuo pipefail
LC_ALL=C
IFS=$'\n\t'

# --- Defaults ----------------------------------------------------------------
CFG_NAME="submit"
SUBMISSION_FILE=""
OUTDIR=""
BASENAME=""
SKIP_VALIDATE="0"
QUIET="0"
DRY_RUN="0"
AUTO_UPLOAD="0"
KAGGLE_COMPETITION=""
KAGGLE_MSG="SpectraMind V50 submission"
RETRIES=3
BACKOFF=2
ALLOW_MISSING_COLS=0
ID_COL="sample_id"
N_BINS=283

DEFAULT_OUTDIR_LOCAL="artifacts"
DEFAULT_OUTDIR_KAGGLE="/kaggle/working/artifacts"

# --- Helpers -----------------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { [[ "$QUIET" = "1" ]] || printf "[ %s ] [package_submission] %s\n" "$(timestamp)" "$*"; }
warn() { printf "[ %s ] [package_submission][WARN] %s\n"  "$(timestamp)" "$*" >&2; }
die() {  printf "[ %s ] [package_submission][ERROR] %s\n" "$(timestamp)" "$*" >&2; exit 1; }

detect_env() { [[ -d "/kaggle/input" ]] && echo "kaggle" || echo "local"; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }
realpath_f() { python - "$1" <<'PY'
import os,sys; p=sys.argv[1]; print(os.path.realpath(p) if os.path.exists(p) else p)
PY
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else echo "unavailable"; fi
}

stat_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }

usage() { sed -n '1,160p' "$0" | sed 's/^# \{0,1\}//'; }

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

# --- Args --------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config-name)   CFG_NAME="${2:-}"; shift 2 ;;
    -f|--file)          SUBMISSION_FILE="${2:-}"; shift 2 ;;
    -o|--outdir)        OUTDIR="${2:-}"; shift 2 ;;
    -n|--name)          BASENAME="${2:-}"; shift 2 ;;
    -S|--skip-validate) SKIP_VALIDATE="1"; shift ;;
    -q|--quiet)         QUIET="1"; shift ;;
        --dry-run)      DRY_RUN="1"; shift ;;
        --auto-upload)  AUTO_UPLOAD="1"; shift ;;
    -C|--competition)   KAGGLE_COMPETITION="${2:-}"; shift 2 ;;
    -m|--message)       KAGGLE_MSG="${2:-}"; shift 2 ;;
        --retries)      RETRIES="${2:-3}"; shift 2 ;;
        --backoff)      BACKOFF="${2:-2}"; shift 2 ;;
        --allow-missing-cols) ALLOW_MISSING_COLS=1; shift ;;
        --id-col)       ID_COL="${2:-sample_id}"; shift 2 ;;
        --bins)         N_BINS="${2:-283}"; shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    *) warn "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

ENV_TYPE="$(detect_env)"
if [[ -z "$OUTDIR" ]]; then
  OUTDIR="$([ "$ENV_TYPE" = "kaggle" ] && echo "$DEFAULT_OUTDIR_KAGGLE" || echo "$DEFAULT_OUTDIR_LOCAL")"
fi
[[ "$DRY_RUN" = "1" ]] || mkdir -p "$OUTDIR" "$OUTDIR/logs"

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
  log "No CSV provided; running submit stage with config: $CFG_NAME"
  if [[ "$DRY_RUN" = "0" ]]; then
    if has_cmd spectramind; then
      spectramind submit --config-name "$CFG_NAME"
    else
      log "spectramind CLI not found; trying python -m spectramind submit"
      python -m spectramind submit --config-name "$CFG_NAME"
    fi
  else
    log "[dry-run] would run: spectramind submit --config-name $CFG_NAME"
  fi

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
EXPECTED_COLS=$((2 * N_BINS + 1))

validate_csv_streaming() {
  local path="$1"
  python - "$path" "$ID_COL" "$N_BINS" "$EXPECTED_COLS" "$ALLOW_MISSING_COLS" <<'PY'
import csv, sys, hashlib, re, math, platform
from itertools import islice

path, id_col = sys.argv[1], sys.argv[2]
n_bins = int(sys.argv[3]); expected_cols = int(sys.argv[4])
allow_missing = int(sys.argv[5]) == 1

def is_number(x:str)->bool:
    if x == "" or x.lower() in {"nan","+nan","-nan","inf","+inf","-inf"}:
        return False
    return bool(re.match(r'^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$', x))

with open(path, newline='') as f:
    r = csv.reader(f)
    header = next(r, None)
    if not header:
        print("ERR: CSV has no header", file=sys.stderr); sys.exit(2)

    # exact header check
    exp = [id_col] + [f"mu_{i:03d}" for i in range(n_bins)] + [f"sigma_{i:03d}" for i in range(n_bins)]
    if header != exp:
        if allow_missing and header == exp[:len(header)]:
            pass
        else:
            print("ERR: Header mismatch", file=sys.stderr)
            for i,(a,b) in enumerate(zip(header,exp), start=1):
                if a!=b:
                    print(f"  col {i}: got '{a}' expected '{b}'", file=sys.stderr)
                    break
            sys.exit(3)

    cols = len(header)
    if cols != expected_cols and not (allow_missing and cols <= expected_cols):
        print(f"ERR: Unexpected column count: got {cols}, expected {expected_cols}", file=sys.stderr)
        sys.exit(4)

    # Validate rows
    total = 0
    bad = None
    id_first = None
    id_last = None
    sample_hash = hashlib.sha256()
    for row in r:
        total += 1
        if len(row) != cols:
            bad = f"Row {total+1} fields={len(row)} expected={cols}"; break
        if id_first is None: id_first = row[0]
        id_last = row[0]
        for v in row[1:cols]:
            if not is_number(v):
                bad = f"Row {total+1} has non-numeric/invalid value '{v}'"; break
        if bad: break
        if total <= 10:
            sample_hash.update(",".join(row).encode())

    if bad:
        print("ERR:", bad, file=sys.stderr); sys.exit(5)

    # stats out
    print("OK")
    print(f"COLS:{cols}")
    print(f"ROWS:{total}")
    print(f"ID_FIRST:{id_first}")
    print(f"ID_LAST:{id_last}")
    print(f"SAMPLE_SHA256:{sample_hash.hexdigest()}")
PY
}

if [[ "$SKIP_VALIDATE" = "0" ]]; then
  log "Validating CSV (streaming full-file checks)…"
  if [[ "$DRY_RUN" = "1" ]]; then
    log "[dry-run] would validate: $SUBMISSION_FILE"
    COLS="$EXPECTED_COLS"; ROWS="-1"; ID_FIRST=""; ID_LAST=""; SAMPLE_SHA256=""
  else
    OUT="$(validate_csv_streaming "$SUBMISSION_FILE" 2>&1 || true)"
    if ! grep -q "^OK$" <<<"$OUT"; then
      printf "%s\n" "$OUT" >&2
      die "CSV validation failed."
    fi
    COLS="$(awk -F: '/^COLS:/{print $2}' <<<"$OUT")"
    ROWS="$(awk -F: '/^ROWS:/{print $2}' <<<"$OUT")"
    ID_FIRST="$(awk -F: '/^ID_FIRST:/{print $2}' <<<"$OUT")"
    ID_LAST="$(awk -F: '/^ID_LAST:/{print $2}' <<<"$OUT")"
    SAMPLE_SHA256="$(awk -F: '/^SAMPLE_SHA256:/{print $2}' <<<"$OUT")"
    log "Validation passed: cols=$COLS rows=$ROWS first_id=$ID_FIRST last_id=$ID_LAST"
  fi
else
  log "Skipping CSV validation as requested"
  COLS="$EXPECTED_COLS"; ROWS="-1"; ID_FIRST=""; ID_LAST=""; SAMPLE_SHA256=""
fi

# --- Manifest ----------------------------------------------------------------
manifest_path="$OUTDIR/${BASENAME}_manifest.json"
csv_sha256="$(sha256_file "$SUBMISSION_FILE")"
csv_size="$([[ -f "$SUBMISSION_FILE" ]] && stat_size "$SUBMISSION_FILE" || echo 0)"

# header hash (exact bytes of first line)
HEADER_SHA256="$(
  if [[ "$DRY_RUN" = "1" ]]; then echo "dry-run";
  else head -n1 "$SUBMISSION_FILE" | { sha256_file /dev/stdin || echo "unavailable"; }; fi
)"

log "Writing manifest → $manifest_path"
if [[ "$DRY_RUN" = "1" ]]; then
  log "[dry-run] would write manifest: $manifest_path"
else
  python - "$manifest_path" <<PY
import json, os, platform, sys, subprocess, datetime
out = sys.argv[1]
def shell(cmd):
    try: return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception: return ""
data = {
  "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
  "environment": "${ENV_TYPE}",
  "submit_config": "${CFG_NAME}",
  "schema": {
    "id_column": "${ID_COL}",
    "n_bins": ${N_BINS},
    "expected_columns": ${EXPECTED_COLS},
    "allow_missing_cols": ${ALLOW_MISSING_COLS}
  },
  "csv": {
    "path": "$(realpath_f "$SUBMISSION_FILE")",
    "sha256": "${csv_sha256}",
    "size_bytes": ${csv_size},
    "header_sha256": "${HEADER_SHA256}",
    "validated": ${SKIP_VALIDATE:+0}${SKIP_VALIDATE:+"0"},
    "cols": ${COLS},
    "rows": ${ROWS},
    "id_first": "${ID_FIRST}",
    "id_last": "${ID_LAST}",
    "sample_sha256": "${SAMPLE_SHA256}"
  },
  "git": {
    "root": "$(realpath_f "$git_root")",
    "revision": "${git_rev}",
    "state": "${git_dirty}"
  },
  "version_file": {
    "path": "$(realpath_f "$version_file")",
    "value": "${repo_version}"
  },
  "runtime": {
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "exe": sys.executable
  }
}
with open(out, "w") as f: json.dump(data, f, indent=2, sort_keys=False)
print(out)
PY
fi

# --- Zip ---------------------------------------------------------------------
zip_path="$OUTDIR/${BASENAME}.zip"
log "Packaging bundle → $zip_path"

if [[ "$DRY_RUN" = "1" ]]; then
  log "[dry-run] would create zip with: $(basename "$SUBMISSION_FILE") and $(basename "$manifest_path")"
else
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
fi

ZIP_SHA256="$([[ -f "$zip_path" ]] && sha256_file "$zip_path" || echo "dry-run")"
ZIP_SIZE="$([[ -f "$zip_path" ]] && stat_size "$zip_path" || echo 0)"
log "Submission bundle ready ✅"
log "ZIP: $zip_path"
log "ZIP sha256: $ZIP_SHA256  size: ${ZIP_SIZE} bytes"

# --- Optional Kaggle Upload --------------------------------------------------
if [[ "$AUTO_UPLOAD" = "1" ]]; then
  [[ "$DRY_RUN" = "1" ]] && { log "[dry-run] would upload to Kaggle"; AUTO_UPLOAD="0"; }
fi

if [[ "$AUTO_UPLOAD" = "1" ]]; then
  [[ -n "$KAGGLE_COMPETITION" ]] || die "--auto-upload requires --competition <slug>"
  has_cmd kaggle || die "Kaggle CLI not found."
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    [[ -f "${HOME}/.kaggle/kaggle.json" ]] || die "Kaggle credentials not found (env KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)."
  fi
  CMD=(kaggle competitions submit -c "$KAGGLE_COMPETITION" -f "$zip_path" -m "$KAGGLE_MSG")
  log "Uploading to Kaggle competition: $KAGGLE_COMPETITION"
  if run_with_retries "$RETRIES" "$BACKOFF" "${CMD[@]}"; then
    log "Upload submitted to Kaggle."
  else
    die "Kaggle submit failed after $RETRIES attempt(s)."
  fi
fi

# --- Summary ------------------------------------------------------------------
echo
echo "──────────────────────────────────────────────────────────────────────────────"
echo " Submission packaging summary"
echo "  • CSV         : $SUBMISSION_FILE"
echo "  • CSV SHA256  : $csv_sha256"
echo "  • CSV Size    : ${csv_size} bytes"
echo "  • ZIP         : $zip_path"
echo "  • ZIP SHA256  : $ZIP_SHA256"
echo "  • ZIP Size    : ${ZIP_SIZE} bytes"
if [[ "$AUTO_UPLOAD" = "1" ]]; then
  echo "  • Upload      : DONE (see Kaggle for result)"
else
  echo "  • Upload      : SKIPPED (use --auto-upload)"
fi
if [[ "$DRY_RUN" = "1" ]]; then
  echo "  • Dry-run     : YES (no files written)"
fi
echo "──────────────────────────────────────────────────────────────────────────────"
