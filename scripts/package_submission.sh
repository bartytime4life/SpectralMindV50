#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Submission Packager (Upgraded, Deterministic, Portable)
# -----------------------------------------------------------------------------
# Builds a Kaggle-ready submission zip:
#   1) Run submit stage (unless --file CSV provided)
#   2) Validate CSV (header, row counts, finite numeric)
#   3) Emit manifest.json (git/version/env/provenance + validation stats)
#   4) Package deterministically into artifacts/<name>.zip
#   5) Optional Kaggle upload with retries/backoff
# -----------------------------------------------------------------------------

set -Eeuo pipefail
LC_ALL=C
IFS=$'\n\t'

# ---------- Defaults ----------
CFG_NAME="submit"
SUBMISSION_FILE=""
OUTDIR=""
BASENAME=""
SKIP_VALIDATE=0
QUIET=0
DRY_RUN=0
AUTO_UPLOAD=0
KAGGLE_COMPETITION=""
KAGGLE_MSG="SpectraMind V50 submission"
RETRIES=3
BACKOFF=2
ALLOW_MISSING_COLS=0
ID_COL="sample_id"
N_BINS=283

DEFAULT_OUTDIR_LOCAL="artifacts"
DEFAULT_OUTDIR_KAGGLE="/kaggle/working/artifacts"
: "${SOURCE_DATE_EPOCH:=946684800}"   # 2000-01-01T00:00:00Z for deterministic zips

# ---------- Helpers ----------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
detect_env() { [[ -d "/kaggle/input" ]] && echo "kaggle" || echo "local"; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }
log()   { [[ "$QUIET" -eq 1 ]] || printf "[ %s ] [package_submission] %s\n" "$(timestamp)" "$*"; }
warn()  { printf "[ %s ] [package_submission][WARN] %s\n"  "$(timestamp)" "$*" >&2; }
die()   { printf "[ %s ] [package_submission][ERROR] %s\n" "$(timestamp)" "$*" >&2; exit 1; }

abspath_py() {
  python3 - "$1" <<'PY' 2>/dev/null || python - "$1" <<'PY'
import os, sys; p=sys.argv[1]; print(os.path.abspath(p))
PY
}
sha256_any() {
  if [ "$1" = "-" ]; then
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 - | awk '{print $1}'; else sha256sum - | awk '{print $1}'; fi
  else
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
    elif command -v shasum >/div/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}';
    else echo "unavailable"; fi
  fi
}
stat_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }

usage() { sed -n '1,200p' "$0" | sed 's/^# \{0,1\}//'; }

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

# ---------- Args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config-name)   CFG_NAME="${2:-}"; shift 2 ;;
    -f|--file)          SUBMISSION_FILE="${2:-}"; shift 2 ;;
    -o|--outdir)        OUTDIR="${2:-}"; shift 2 ;;
    -n|--name)          BASENAME="${2:-}"; shift 2 ;;
    -S|--skip-validate) SKIP_VALIDATE=1; shift ;;
    -q|--quiet)         QUIET=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --auto-upload)  AUTO_UPLOAD=1; shift ;;
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
[[ "$DRY_RUN" -eq 1 ]] || mkdir -p "$OUTDIR" "$OUTDIR/logs"

# ---------- Version/Git ----------
git_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
git_rev="$(git -C "$git_root" rev-parse --short HEAD 2>/dev/null || echo "nogit")"
git_dirty="$([ -n "$(git -C "$git_root" status --porcelain 2>/dev/null || true)" ] && echo "dirty" || echo "clean")"
version_file="$git_root/VERSION"
repo_version="$([ -f "$version_file" ] && tr -d ' \n' < "$version_file" || echo "0.0.0")"

if [[ -z "$BASENAME" ]]; then
  BASENAME="submission_${repo_version:-nogit}_$(date +%Y%m%d_%H%M%S)"
fi

# ---------- Locate / Produce submission.csv ----------
if [[ -z "$SUBMISSION_FILE" ]]; then
  log "No CSV provided; running submit stage (config: $CFG_NAME)"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    if has_cmd spectramind; then spectramind submit --config-name "$CFG_NAME";
    else python -m spectramind submit --config-name "$CFG_NAME"; fi
  else
    log "[dry-run] spectramind submit --config-name $CFG_NAME"
  fi
  # Common locations
  for f in \
    "$OUTDIR/submission.csv" \
    "outputs/submission.csv" \
    "artifacts/submission.csv" \
    "submission.csv" \
    "$(abspath_py "$git_root")/outputs/submission.csv" \
    "$(abspath_py "$git_root")/artifacts/submission.csv"; do
    [[ -f "$f" ]] && SUBMISSION_FILE="$f" && break
  done
  [[ -n "$SUBMISSION_FILE" ]] || SUBMISSION_FILE="$(find . -maxdepth 3 -type f -name 'submission*.csv' | head -n1 || true)"
  [[ -n "$SUBMISSION_FILE" ]] || die "Unable to locate submission CSV after submit stage."
else
  [[ -f "$SUBMISSION_FILE" ]] || die "Provided file not found: $SUBMISSION_FILE"
fi
log "Using submission CSV: $SUBMISSION_FILE"

# ---------- Validation ----------
EXPECTED_COLS=$((2 * N_BINS + 1))

validate_csv_streaming() {
  local path="$1"
  python - "$path" "$ID_COL" "$N_BINS" "$EXPECTED_COLS" "$ALLOW_MISSING_COLS" <<'PY'
import csv, sys, re, math, hashlib
path, id_col = sys.argv[1], sys.argv[2]
n_bins = int(sys.argv[3]); expected_cols = int(sys.argv[4])
allow_missing = int(sys.argv[5]) == 1

num_re = re.compile(r'^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$')
def finite_num(s):
    if not s or s.lower() in {"nan","+nan","-nan","inf","+inf","-inf"}: return False
    if not num_re.match(s): return False
    try:
        x = float(s)
        return math.isfinite(x)
    except Exception:
        return False

with open(path, newline='') as f:
    r = csv.reader(f)
    header = next(r, None)
    if not header: print("ERR: no header", file=sys.stderr); sys.exit(2)

    exp = [id_col] + [f"mu_{i:03d}" for i in range(n_bins)] + [f"sigma_{i:03d}" for i in range(n_bins)]
    if header != exp:
        if allow_missing and header == exp[:len(header)]:
            pass
        else:
            print("ERR: header mismatch", file=sys.stderr)
            # show first mismatch for speed
            for i,(a,b) in enumerate(zip(header,exp), start=1):
                if a!=b:
                    print(f"  col {i}: got '{a}' expected '{b}'", file=sys.stderr)
                    break
            sys.exit(3)

    cols = len(header)
    if cols != expected_cols and not (allow_missing and cols <= expected_cols):
        print(f"ERR: column count {cols} != expected {expected_cols}", file=sys.stderr)
        sys.exit(4)

    total = 0
    id_first = id_last = ""
    sample_hash = hashlib.sha256()
    for row in r:
        total += 1
        if len(row) != cols:
            print(f"ERR: row {total+1} length {len(row)} != header {cols}", file=sys.stderr); sys.exit(5)
        if total == 1: id_first = row[0]
        id_last = row[0]
        for v in row[1:cols]:
            if not finite_num(v):
                print(f"ERR: row {total+1} invalid numeric '{v}'", file=sys.stderr); sys.exit(6)
        if total <= 10:
            sample_hash.update(",".join(row).encode())

    print("OK")
    print(f"COLS:{cols}")
    print(f"ROWS:{total}")
    print(f"ID_FIRST:{id_first}")
    print(f"ID_LAST:{id_last}")
    print(f"SAMPLE_SHA256:{sample_hash.hexdigest()}")
PY
}

if [[ "$SKIP_VALIDATE" -eq 0 ]]; then
  log "Validating CSV…"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "[dry-run] would validate: $SUBMISSION_FILE"
    COLS="$EXPECTED_COLS"; ROWS="-1"; ID_FIRST=""; ID_LAST=""; SAMPLE_SHA256=""
  else
    OUT="$(validate_csv_streaming "$SUBMISSION_FILE" 2>&1 || true)"
    if ! grep -qx "OK" <<<"$OUT"; then printf "%s\n" "$OUT" >&2; die "CSV validation failed."; fi
    COLS="$(awk -F: '/^COLS:/{print $2}' <<<"$OUT")"
    ROWS="$(awk -F: '/^ROWS:/{print $2}' <<<"$OUT")"
    ID_FIRST="$(awk -F: '/^ID_FIRST:/{print $2}' <<<"$OUT")"
    ID_LAST="$(awk -F: '/^ID_LAST:/{print $2}' <<<"$OUT")"
    SAMPLE_SHA256="$(awk -F: '/^SAMPLE_SHA256:/{print $2}' <<<"$OUT")"
    log "Validation passed: cols=$COLS rows=$ROWS first_id=$ID_FIRST last_id=$ID_LAST"
  fi
else
  log "Skipping CSV validation as requested."
  COLS="$EXPECTED_COLS"; ROWS="-1"; ID_FIRST=""; ID_LAST=""; SAMPLE_SHA256=""
fi

# ---------- Manifest ----------
manifest_path="$OUTDIR/${BASENAME}_manifest.json"
csv_sha256="$(sha256_any "$SUBMISSION_FILE")"
csv_size="$([[ -f "$SUBMISSION_FILE" ]] && stat_size "$SUBMISSION_FILE" || echo 0)"
HEADER_SHA256="$(
  if [[ "$DRY_RUN" -eq 1 ]]; then echo "dry-run"; else
    head -n1 "$SUBMISSION_FILE" | sha256_any -; fi
)"

log "Writing manifest → $manifest_path"
if [[ "$DRY_RUN" -eq 1 ]]; then
  log "[dry-run] manifest not written"
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
    "path": "${SUBMISSION_FILE}",
    "sha256": "${csv_sha256}",
    "size_bytes": ${csv_size},
    "header_sha256": "${HEADER_SHA256}",
    "validated": ${SKIP_VALIDATE==0},
    "cols": ${COLS},
    "rows": ${ROWS},
    "id_first": "${ID_FIRST}",
    "id_last": "${ID_LAST}",
    "sample_sha256": "${SAMPLE_SHA256}"
  },
  "git": {
    "root": "${git_root}",
    "revision": "${git_rev}",
    "state": "${git_dirty}"
  },
  "version_file": {
    "path": "${version_file}",
    "value": "${repo_version}"
  },
  "runtime": {
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "platform": platform.platform()
  }
}
with open(out, "w") as f: json.dump(data, f, indent=2)
print(out)
PY
fi

# ---------- Package (deterministic) ----------
zip_path="$OUTDIR/${BASENAME}.zip"
log "Packaging bundle → $zip_path"
if [[ "$DRY_RUN" -eq 1 ]]; then
  log "[dry-run] would add: $(basename "$SUBMISSION_FILE"), $(basename "$manifest_path")"
else
  python3 - "$zip_path" "$SUBMISSION_FILE" "$manifest_path" "$SOURCE_DATE_EPOCH" <<'PYZ' 2>/dev/null || \
  python - "$zip_path" "$SUBMISSION_FILE" "$manifest_path" "$SOURCE_DATE_EPOCH" <<'PYZ'
import os, sys, time, zipfile
dst, csvf, manifest, epoch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def add(zf, src):
    zi = zipfile.ZipInfo(os.path.basename(src))
    zi.date_time = time.gmtime(epoch)[:6]
    # 0644 regular file
    zi.external_attr = (0o100644 & 0xFFFF) << 16
    with open(src, 'rb') as f: zf.writestr(zi, f.read(), compress_type=zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(dst, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    add(z, csvf); add(z, manifest)
print(dst)
PYZ
fi

ZIP_SHA256="$([[ -f "$zip_path" ]] && sha256_any "$zip_path" || echo "dry-run")"
ZIP_SIZE="$([[ -f "$zip_path" ]] && stat_size "$zip_path" || echo 0)"
log "ZIP sha256: $ZIP_SHA256  size: ${ZIP_SIZE} bytes"

# ---------- Optional Kaggle Upload ----------
if [[ "$AUTO_UPLOAD" -eq 1 ]]; then
  [[ -n "$KAGGLE_COMPETITION" ]] || die "--auto-upload requires --competition <slug>"
  has_cmd kaggle || die "Kaggle CLI not found."
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    [[ -f "${HOME}/.kaggle/kaggle.json" ]] || die "Kaggle creds missing (env or ~/.kaggle/kaggle.json)."
  fi
  CMD=(kaggle competitions submit -c "$KAGGLE_COMPETITION" -f "$zip_path" -m "$KAGGLE_MSG")
  log "Uploading to Kaggle competition: $KAGGLE_COMPETITION"
  if run_with_retries "$RETRIES" "$BACKOFF" "${CMD[@]}"; then
    log "Upload submitted to Kaggle."
  else
    die "Kaggle submit failed after $RETRIES attempt(s)."
  fi
fi

# ---------- Summary ----------
echo
echo "──────────────────────────────────────────────────────────────────────────────"
echo " Submission packaging summary"
echo "  • CSV         : $SUBMISSION_FILE"
echo "  • CSV SHA256  : $csv_sha256"
echo "  • CSV Size    : ${csv_size} bytes"
echo "  • ZIP         : $zip_path"
echo "  • ZIP SHA256  : $ZIP_SHA256"
echo "  • ZIP Size    : ${ZIP_SIZE} bytes"
if [[ "$AUTO_UPLOAD" -eq 1 ]]; then
  echo "  • Upload      : DONE (see Kaggle for result)"
else
  echo "  • Upload      : SKIPPED (use --auto-upload)"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  • Dry-run     : YES (no files written)"
fi
echo "──────────────────────────────────────────────────────────────────────────────"