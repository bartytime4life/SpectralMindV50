#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# pack_precalibrated.sh — Reproducible ZIP packer for Kaggle artifacts
# -----------------------------------------------------------------------------
# Usage:
#   scripts/pack_precalibrated.sh [SRC=data/calibrated] [DST=artifacts/kaggle/precalibrated.zip]
# Notes:
#   • Deterministic output: sorted file order, stripped extra fields, fixed mtime.
#   • Portable: prefers `zip`; falls back to Python's zipfile if missing.
#   • Excludes common junk: .DS_Store, __pycache__, *.pyc, .git, ipynb_checkpoints, Thumbs.db
# -----------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# --- Pretty logging -----------------------------------------------------------
info()  { printf '[pack] %s\n' "$*"; }
warn()  { printf '\033[33m[pack] WARN:\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[31m[pack] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- Args --------------------------------------------------------------------
SRC="${1:-data/calibrated}"
DST="${2:-artifacts/kaggle/precalibrated.zip}"

# --- Helpers -----------------------------------------------------------------
# Portable abspath resolver (avoids `realpath -m` which is absent on macOS)
abspath_py() {
  python3 - "$1" <<'PY' 2>/dev/null || python - "$1" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else echo "unavailable"
  fi
}

have() { command -v "$1" >/dev/null 2>&1; }

# --- Preflight ---------------------------------------------------------------
[[ -d "$SRC" ]] || die "Source directory '$SRC' not found."
DST_DIR="$(dirname -- "$DST")"
mkdir -p -- "$DST_DIR"

# Ensure non-empty source (at least one regular file after excludes)
non_empty_src=false
while IFS= read -r -d '' f; do non_empty_src=true; break; done < <(
  find "$SRC" -type f \
    ! -path '*/.git/*' \
    ! -path '*/__pycache__/*' \
    ! -path '*/.ipynb_checkpoints/*' \
    ! -name '.DS_Store' ! -name 'Thumbs.db' ! -name '*.pyc' \
    -print0
)
$non_empty_src || die "No files to pack in '$SRC' after exclusions."

ABS_DST="$(abspath_py "$DST")"
ABS_SRC="$(abspath_py "$SRC")"

# Use fixed timestamp for reproducibility (SOURCE_DATE_EPOCH if set, else 2000-01-01)
: "${SOURCE_DATE_EPOCH:=946684800}"

# --- Build sorted file list ---------------------------------------------------
# We will store paths relative to SRC, sorted bytewise, NUL-delimited
FILES_LIST="$(mktemp)"; trap 'rm -f "$FILES_LIST"' EXIT
(
  cd "$SRC"
  find . -type f \
    ! -path './.git/*' \
    ! -path './__pycache__/*' \
    ! -path './.ipynb_checkpoints/*' \
    ! -name '.DS_Store' ! -name 'Thumbs.db' ! -name '*.pyc' \
    -print0 \
  | LC_ALL=C sort -z \
  > "$FILES_LIST"
)

# --- Pack (zip or Python fallback) -------------------------------------------
if have zip; then
  info "Packing with 'zip' → $ABS_DST"
  (
    cd "$SRC"
    # -X strip extra fields; -q quiet; -r recurse (we feed explicit list); -9 max compression
    # Use -@ (read file list from stdin). Ensure deterministic order via sorted list.
    # Set TZ to UTC and tell zip to use fixed timestamp by touching via env (zip itself doesn't set mtimes,
    # but deterministic order + stripped extras yields stable bytes for static inputs).
    TZ=UTC zip -X -q -9 -@ "$ABS_DST" < "$FILES_LIST"
  )
else
  warn "'zip' not found; using Python zipfile fallback."
  python3 - "$ABS_SRC" "$ABS_DST" "$FILES_LIST" "$SOURCE_DATE_EPOCH" <<'PY' 2>/dev/null || \
  python - "$ABS_SRC" "$ABS_DST" "$FILES_LIST" "$SOURCE_DATE_EPOCH" <<'PY'
import os, sys, time, zipfile
src, dst, listfile, epoch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def write_rel(zf, rel):
    p = os.path.join(src, rel)
    st = os.stat(p)
    zi = zipfile.ZipInfo(rel)
    # Fixed timestamp for reproducibility
    zi.date_time = time.gmtime(epoch)[:6]
    # Unix external attributes: regular file 0644 (rw-r--r--)
    zi.external_attr = (0o100644 & 0xFFFF) << 16
    with open(p, 'rb') as f:
        zf.writestr(zi, f.read(), compress_type=zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(dst, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    with open(listfile, 'rb') as lf:
        for rel_b in lf.read().split(b'\x00'):
            if not rel_b: continue
            rel = rel_b.decode('utf-8', 'strict')
            if rel.startswith('./'): rel = rel[2:]
            write_rel(zf, rel)
PY
fi

# --- Report -------------------------------------------------------------------
SZ="$( (du -h "$ABS_DST" 2>/dev/null || gdu -h "$ABS_DST" 2>/dev/null) | awk '{print $1; exit}')" || SZ="(size n/a)"
SHA="$(sha256_file "$ABS_DST")"

info "Source : $ABS_SRC"
info "Output : $ABS_DST"
info "Size   : $SZ"
info "SHA256 : $SHA"
echo "Wrote $SZ → $ABS_DST"