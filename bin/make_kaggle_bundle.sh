#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — make_kaggle_bundle.sh
# Build a deterministic, Kaggle-safe ZIP bundle with the minimal project payload.
# • Repo-root aware, fail-fast, dry-run, JSON summary for CI
# • Deterministic: fixed mtimes/permissions/sorted entries + SHA256 manifest
# • Offline-safe: excludes heavy/dev/test/provenance content
# • Optional runner notebook/script and pinned requirements
# ------------------------------------------------------------------------------
# Usage:
#   bin/make_kaggle_bundle.sh
#   bin/make_kaggle_bundle.sh --out outputs/bundles/sm_v50_bundle.zip
#   bin/make_kaggle_bundle.sh --req requirements-kaggle.txt --nb notebooks/runner.ipynb
#   bin/make_kaggle_bundle.sh --name SpectraMindV50 --version 0.1.0 --json
#
# Options:
#   --out PATH            Output ZIP path (default: outputs/bundles/sm_v50_bundle.zip)
#   --name STR            Logical package name (default: repo dir name)
#   --version STR         Version string (default: VERSION file or git describe)
#   --req PATH            Use this requirements file instead of auto-min
#   --nb PATH             Include a runner notebook (.ipynb) or .py
#   --extra LIST          Comma-separated extra globs to include (e.g. "README.md,LICENSE,schemas/*.json")
#   --epoch TS            SOURCE_DATE_EPOCH override (POSIX seconds; default: 2000-01-01T00:00:00Z)
#   --quiet               Suppress informational logs
#   --dry-run             Print actions without writing ZIP
#   --json                Emit compact JSON summary to stdout
#   -h, --help            Show this help
#
# Outputs:
#   • <out>.zip           Deterministic ZIP archive for Kaggle
#   • <out>.zip.MANIFEST.json   File list, sizes, SHA256, build meta
#
# Excludes (by default):
#   .git/ .github/ .dvc/ data/ outputs/ dist/ build/ site/ docs/_build/ .venv/ __pycache__/ tests/ *.svg *.png *.pdf
#   notebooks/* (except if explicitly passed via --nb), assets/ (except assets/diagrams/*.mmd)
# ==============================================================================

set -Eeuo pipefail

# ---------- pretty printing ----------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

usage() { sed -n '1,120p' "$0" | awk '/^# ====/{flag=1;next}/^set -Eeuo/{flag=0}flag' | sed 's/^# \{0,1\}//'; }

trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"; exit 4' ERR

# ---------- args ---------------------------------------------------------------
OUT="outputs/bundles/sm_v50_bundle.zip"
NAME=""
VERSION=""
REQ_OVERRIDE=""
RUNNER_NOTEBOOK=""
EXTRA_GLOBS=""
QUIET="${QUIET:-0}"
DRYRUN=0
EMIT_JSON=0
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-946684800}" # 2000-01-01 UTC

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)     OUT="${2:-}"; shift 2 ;;
    --name)    NAME="${2:-}"; shift 2 ;;
    --version) VERSION="${2:-}"; shift 2 ;;
    --req)     REQ_OVERRIDE="${2:-}"; shift 2 ;;
    --nb)      RUNNER_NOTEBOOK="${2:-}"; shift 2 ;;
    --extra)   EXTRA_GLOBS="${2:-}"; shift 2 ;;
    --epoch)   SOURCE_DATE_EPOCH="${2:-}"; shift 2 ;;
    --quiet)   QUIET=1; shift ;;
    --dry-run) DRYRUN=1; shift ;;
    --json)    EMIT_JSON=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done
export SOURCE_DATE_EPOCH

# ---------- repo root detection ------------------------------------------------
repo_root() {
  if command -v git >/dev/null 2>&1; then
    if r="$(git rev-parse --show-toplevel 2>/dev/null || true)"; then
      [[ -n "$r" ]] && { printf "%s" "$r"; return 0; }
    fi
  fi
  local d="$PWD"
  while [[ "$d" != "/" ]]; do
    if [[ -e "$d/pyproject.toml" || -e "$d/dvc.yaml" || -d "$d/.git" ]]; then
      printf "%s" "$d"; return 0
    fi
    d="$(dirname "$d")"
  done
  printf "%s" "$PWD"
}
ROOT="$(repo_root)"
cd "$ROOT"

[[ -z "$NAME" ]] && NAME="$(basename "$ROOT")"
if [[ -z "$VERSION" ]]; then
  if [[ -f "$ROOT/VERSION" ]]; then
    VERSION="$(sed -n '1p' "$ROOT/VERSION" | tr -d '[:space:]')"
  else
    VERSION="$(git describe --tags --always 2>/dev/null || echo "0.0.0")"
  fi
fi

OUT_ABS="$(python3 - <<PY
import os,sys
p=sys.argv[1]
print(os.path.abspath(p))
PY
"$OUT")"
OUT_DIR="$(dirname "$OUT_ABS")"
mkdir -p "$OUT_DIR"

# ---------- staging dir (clean) -----------------------------------------------
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/smv50_bundle.XXXXXX")"
CLEANUP() { rm -rf "$STAGE"; }
trap CLEANUP EXIT

log "Building Kaggle bundle for $NAME@$VERSION"
log "Root: $ROOT"
log "Stage: $STAGE"
log "Out:   $OUT_ABS"

# ---------- default include/exclude rules -------------------------------------
# Includes (minimal runnable payload):
INCLUDE_PATTERNS=(
  "pyproject.toml"
  "README.md"
  "LICENSE*"
  "src/**"
  "configs/**"
  "schemas/**"
  "assets/diagrams/*.mmd"
  "bin/sm_preprocess.sh"
  "bin/sm_train.sh"
  "bin/sm_predict.sh"
  "bin/sm_diagnose.sh"
  "bin/sm_submit.sh"
)

# Extras (user-specified globs)
IFS=',' read -r -a EXTRA_ARR <<< "${EXTRA_GLOBS:-}"
for g in "${EXTRA_ARR[@]}"; do
  [[ -n "$g" ]] && INCLUDE_PATTERNS+=("$g")
done

# Excludes (heavy/dev/offline-unsafe)
EXCLUDE_PATTERNS=(
  ".git/**" ".github/**" ".dvc/**" ".mypy_cache/**" ".ruff_cache/**" ".pytest_cache/**" ".venv/**" "__pycache__/**"
  "outputs/**" "data/**" "dist/**" "build/**" "site/**" "docs/_build/**" "tests/**" "notebooks/**"
  "*.svg" "*.png" "*.jpg" "*.jpeg" "*.gif" "*.pdf"
  "assets/**"  # blanket, but we add back diagrams/*.mmd above
)

# If a runner notebook/script is provided, whitelist it explicitly
if [[ -n "$RUNNER_NOTEBOOK" ]]; then
  # copy later explicitly
  :
fi

# ---------- rsync copy with include/exclude -----------------------------------
if ! command -v rsync >/dev/null 2>&1; then
  err "rsync is required"; exit 3
fi

# Build rsync filter file
FILTER_FILE="$STAGE/.rsync-filter"
{
  for p in "${INCLUDE_PATTERNS[@]}"; do echo "+ $p"; done
  for p in "${EXCLUDE_PATTERNS[@]}"; do echo "- $p"; done
  echo "- *"
} > "$FILTER_FILE"

log "Syncing files → staging (rsync + deterministic perms)"
rsync -a --delete --prune-empty-dirs --filter="merge $FILTER_FILE" ./ "$STAGE/"

# Runner notebook/script explicit copy (if provided)
if [[ -n "$RUNNER_NOTEBOOK" ]]; then
  if [[ -f "$RUNNER_NOTEBOOK" ]]; then
    mkdir -p "$STAGE/notebooks"
    cp -p "$RUNNER_NOTEBOOK" "$STAGE/notebooks/"
    ok "Included runner: $RUNNER_NOTEBOOK"
  else
    warn "Runner not found: $RUNNER_NOTEBOOK (skipped)"
  fi
fi

# Requirements: use override or attempt to select minimal
REQ_OUT="$STAGE/requirements.txt"
if [[ -n "$REQ_OVERRIDE" ]]; then
  cp -p "$REQ_OVERRIDE" "$REQ_OUT"
else
  # Prefer a curated minimal file if present
  if [[ -f "requirements-kaggle.txt" ]]; then
    cp -p "requirements-kaggle.txt" "$REQ_OUT"
  elif [[ -f "requirements.txt" ]]; then
    # Use project reqs but strip obvious dev tools
    grep -Ev '^(black|flake8|mypy|ruff|pytest|ipykernel|mkdocs|graphviz|pygraphviz)\b' requirements.txt > "$REQ_OUT" || true
  else
    # Fallback: infer from pyproject (best-effort)
    python3 - <<'PY' > "$REQ_OUT" || true
import tomllib,sys,io,os
p="pyproject.toml"
req=[]
if os.path.exists(p):
    with open(p,"rb") as f:
        t=tomllib.load(f)
    deps = t.get("project",{}).get("dependencies",[])
    for d in deps: print(d)
PY
  fi
fi
[[ -s "$REQ_OUT" ]] || echo "# minimal" > "$REQ_OUT"

# ---------- normalize perms & mtimes for determinism ---------------------------
# Files 0644, executables 0755, fixed mtime = SOURCE_DATE_EPOCH
find "$STAGE" -type f -exec chmod 0644 {} +
# Make shell wrappers executable
find "$STAGE" -type f \( -name "*.sh" -o -path "*/bin/*" \) -exec chmod 0755 {} + || true
# Normalize mtime
touch_ref="$(python3 - <<PY
import os,sys,time,datetime
print(datetime.datetime.utcfromtimestamp(int(os.environ.get("SOURCE_DATE_EPOCH","946684800"))).strftime("%Y-%m-%d %H:%M:%S"))
PY
)"
find "$STAGE" -exec touch -d "$touch_ref" {} +

# ---------- build MANIFEST with SHA256 -----------------------------------------
MANIFEST="$OUT_ABS.MANIFEST.json"
python3 - "$STAGE" "$NAME" "$VERSION" "$MANIFEST" "$OUT_ABS" "$SOURCE_DATE_EPOCH" <<'PY'
import sys, os, hashlib, json, time
root, name, version, manifest_path, out_zip, epoch = sys.argv[1:]
files=[]
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        rp = os.path.relpath(p, root)
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1<<20), b''):
                h.update(chunk)
        files.append({"path": rp.replace("\\","/"), "size": os.path.getsize(p), "sha256": h.hexdigest()})
files.sort(key=lambda x: x["path"])
meta = {
  "name": name, "version": version,
  "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(epoch))),
  "source_date_epoch": int(epoch),
  "count": len(files),
  "out_zip": out_zip,
  "files": files,
}
with open(manifest_path, "w") as f:
    json.dump(meta, f, separators=(",",":"))
print(manifest_path)
PY

ok "Wrote MANIFEST: $MANIFEST"

# ---------- create deterministic ZIP (sorted, fixed timestamps) ----------------
if [[ "$DRYRUN" -eq 1 ]]; then
  log "[dry-run] Would write ZIP → $OUT_ABS"
else
  python3 - "$STAGE" "$OUT_ABS" "$SOURCE_DATE_EPOCH" <<'PY'
import sys, os, time, zipfile, stat, datetime
src, outp, epoch = sys.argv[1], sys.argv[2], int(sys.argv[3])
# Ensure parent
os.makedirs(os.path.dirname(outp), exist_ok=True)
# Collect paths sorted
paths=[]
for dp, _, fns in os.walk(src):
    for fn in fns:
        rp = os.path.relpath(os.path.join(dp, fn), src).replace("\\","/")
        paths.append(rp)
paths.sort()
dt = time.gmtime(epoch)
fixed_dt = (dt.tm_year, dt.tm_mon, dt.tm_mday, dt.tm_hour, dt.tm_min, dt.tm_sec)
with zipfile.ZipFile(outp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for rp in paths:
        ap = os.path.join(src, rp)
        zi = zipfile.ZipInfo(rp, date_time=fixed_dt)
        # 0644 by default; mark executables (scripts) as 0755
        mode = 0o100644
        if rp.endswith(".sh") or "/bin/" in rp:
            mode = 0o100755
        zi.external_attr = (mode & 0xFFFF) << 16
        with open(ap, "rb") as f:
            zf.writestr(zi, f.read())
PY
  ok "Wrote ZIP: $OUT_ABS"
fi

# ---------- summary / size -----------------------------------------------------
BYTES=0
if [[ -f "$OUT_ABS" ]]; then
  BYTES="$(wc -c < "$OUT_ABS" | tr -d ' ')"
  log "Bundle size: $BYTES bytes"
fi

# ---------- JSON summary -------------------------------------------------------
if [[ "$EMIT_JSON" -eq 1 ]]; then
  esc() { printf "%s" "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  printf '{'
  printf '"ok":true,'
  printf '"name":"%s",' "$(esc "$NAME")"
  printf '"version":"%s",' "$(esc "$VERSION")"
  printf '"out_zip":"%s",' "$(esc "$OUT_ABS")"
  printf '"manifest":"%s",' "$(esc "$MANIFEST")"
  printf '"bytes":%s,' "${BYTES:-0}"
  printf '"epoch":%s' "$SOURCE_DATE_EPOCH"
  printf '}\n'
fi

ok "Kaggle bundle ready."
exit 0

