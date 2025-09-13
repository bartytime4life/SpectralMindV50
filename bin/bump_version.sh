#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Version Bump Script
#
# Safely bump the project version across the repo and create a signed/annotated
# git tag (optional). Portable across Linux/macOS (BSD/GNU sed not required).
#
# Highlights
#   • Accepts either bump type (major|minor|patch) or explicit version X.Y.Z
#   • Validates SemVer; enforces monotonic increase unless --force
#   • Updates the following when present:
#       - VERSION                                 (single line)
#       - pyproject.toml                          -> version = "X.Y.Z"
#       - src/spectramind/__init__.py             -> __version__ = "X.Y.Z"
#       - src/spectramind/version.py              -> __version__ = "X.Y.Z"
#       - package.json                            -> "version": "X.Y.Z"
#       - conda/meta.yaml                         -> version: "X.Y.Z"
#       - docs/version.md                         -> single line X.Y.Z
#   • Creates a single commit; optional signed/annotated tag vX.Y.Z
#   • Dry-run support; clear logs; exits on first error
#
# Usage
#   bin/bump_version.sh patch
#   bin/bump_version.sh minor --tag
#   bin/bump_version.sh 1.3.0 --tag --sign
#   bin/bump_version.sh major --no-commit --dry-run
#
# Options
#   --tag          Create a tag vX.Y.Z (annotated)
#   --sign         GPG-sign the tag (implies --tag)
#   --no-commit    Do not git commit; just update files
#   --force        Allow non-monotonic bump (override guard)
#   --allow-dirty  Skip dirty working-tree check
#   --dry-run      Print actions; do not modify files or git
#   -h|--help      Show help
#
# Notes
#   • Requires git. Uses python3 for robust, cross-platform file edits.
#   • If python3 is unavailable, the script aborts with a helpful message.
# ==============================================================================

set -Eeuo pipefail

# ---------- colored logs ----------
if [[ -t 1 ]]; then
  C_RESET="\033[0m"; C_DIM="\033[2m"; C_GREEN="\033[32m"; C_YELLOW="\033[33m"; C_RED="\033[31m"; C_BLUE="\033[34m"
else
  C_RESET=""; C_DIM=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""
fi
log()   { printf "%b%s%b\n"   "${C_DIM}"   "$*" "${C_RESET}"; }
info()  { printf "%bℹ %s%b\n" "${C_BLUE}"   "$*" "${C_RESET}"; }
ok()    { printf "%b✓ %s%b\n" "${C_GREEN}"  "$*" "${C_RESET}"; }
warn()  { printf "%b! %s%b\n" "${C_YELLOW}" "$*" "${C_RESET}"; }
err()   { printf "%b✗ %s%b\n" "${C_RED}"    "$*" "${C_RESET}" >&2; }

# ---------- repo root ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null); then :; else ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"; fi
cd "${ROOT_DIR}"

# ---------- defaults ----------
DO_TAG="0"
DO_SIGN="0"
NO_COMMIT="0"
FORCE="0"
ALLOW_DIRTY="0"
DRY_RUN="0"

usage() {
  sed -n '1,120p' "${BASH_SOURCE[0]}" | sed -n '1,80p'
}

# ---------- parse args ----------
NEW_SPEC="${1:-}"
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)         DO_TAG="1"; shift ;;
    --sign)        DO_TAG="1"; DO_SIGN="1"; shift ;;
    --no-commit)   NO_COMMIT="1"; shift ;;
    --force)       FORCE="1"; shift ;;
    --allow-dirty) ALLOW_DIRTY="1"; shift ;;
    --dry-run)     DRY_RUN="1"; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) err "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "${NEW_SPEC}" ]]; then
  err "Specify a bump type (major|minor|patch) or explicit version X.Y.Z"
  usage; exit 2
fi

command -v git >/dev/null 2>&1 || { err "git not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { err "python3 is required"; exit 1; }

# ---------- helpers ----------
semver_ok() { [[ "$1" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; }
semver_cmp() { # echo -1,0,1 for a<b, a==b, a>b
  python3 - "$1" "$2" <<'PY'
import sys
def parse(s): 
    x=s.strip().split('.'); 
    if len(x)!=3: raise SystemExit(2)
    return tuple(map(int,x))
a,b = parse(sys.argv[1]), parse(sys.argv[2])
print((-1 if a<b else (1 if a>b else 0)))
PY
}

read_version_file() {
  if [[ -f VERSION ]]; then
    tr -d '\r\n ' < VERSION
  else
    echo "0.0.0"
  fi
}

compute_bump() {
  local cur="$1" spec="$2"
  python3 - "$cur" "$spec" <<'PY'
import sys,re
cur, spec = sys.argv[1], sys.argv[2]
m = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)', cur)
if not m: 
    print("0.0.0"); sys.exit(0)
x,y,z = map(int, m.groups())
if spec in ("major","minor","patch"):
    if spec=="major": x,y,z = x+1,0,0
    elif spec=="minor": y,z = y+1,0
    else: z += 1
    print(f"{x}.{y}.{z}")
else:
    if not re.fullmatch(r'\d+\.\d+\.\d+', spec):
        print("INVALID"); sys.exit(0)
    print(spec)
PY
}

write_file() {
  local path="$1" content="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    info "Would write ${path}: ${content}"
  else
    mkdir -p "$(dirname "${path}")"
    printf "%s\n" "${content}" > "${path}"
  fi
}

update_with_python() {
  local path="$1" pat="$2" repl="$3" mode="${4:-regex}" create_if_absent="${5:-0}"
  [[ -f "${path}" ]] || {
    if [[ "${create_if_absent}" == "1" ]]; then
      if [[ "${DRY_RUN}" != "1" ]]; then : > "${path}"; fi
    else
      return 0
    fi
  }
  if [[ "${DRY_RUN}" == "1" ]]; then
    info "Would update ${path} (${mode}): ${pat} -> ${repl}"
    return 0
  fi
  python3 - "$path" "$pat" "$repl" "$mode" <<'PY'
import sys,re,io,os,json
p, pat, repl, mode = sys.argv[1:5]
with open(p,'r',encoding='utf-8',errors='ignore') as fp:
    s=fp.read()
changed=False
if mode=="regex":
    ns, n = re.subn(pat, repl, s, flags=re.M)
    if n>0: s, changed = ns, True
elif mode=="json":
    # naive: update top-level "version"
    try:
        o=json.loads(s)
        if isinstance(o,dict) and "version" in o:
            o["version"]=repl
            s=json.dumps(o,indent=2,ensure_ascii=False)+"\n"
            changed=True
    except Exception: pass
else:
    # whole file replace
    s, changed = repl, True
if changed:
    with open(p,'w',encoding='utf-8') as fp: fp.write(s)
print("OK")
PY
}

# ---------- preflight ----------
CUR_VER="$(read_version_file)"
NEW_VER="$(compute_bump "${CUR_VER}" "${NEW_SPEC}")"

if [[ "${NEW_VER}" == "INVALID" ]]; then
  err "Invalid explicit version: ${NEW_SPEC} (expected X.Y.Z)"
  exit 2
fi
semver_ok "${NEW_VER}" || { err "Computed version is invalid: ${NEW_VER}"; exit 2; }

info "Current version : ${CUR_VER}"
info "Target version  : ${NEW_VER}"

if [[ "${FORCE}" != "1" && "${CUR_VER}" != "0.0.0" ]]; then
  cmp="$(semver_cmp "${NEW_VER}" "${CUR_VER}")"
  if [[ "${cmp}" -le 0 ]]; then
    err "Refusing non-incremental bump (${NEW_VER} <= ${CUR_VER}). Use --force to override."
    exit 2
  fi
fi

# working tree clean?
if [[ "${ALLOW_DIRTY}" != "1" ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    err "Working tree is dirty. Commit/stash changes or use --allow-dirty."
    exit 2
  fi
fi

# ---------- planned updates ----------
FILES_UPDATED=()

# 1) VERSION (single line)
write_file "VERSION" "${NEW_VER}"
FILES_UPDATED+=("VERSION")

# 2) pyproject.toml  version = "X.Y.Z"
update_with_python "pyproject.toml" \
  '^(version\s*=\s*)"(?:\d+\.\d+\.\d+)"' \
  "\g<1>\"${NEW_VER}\""
FILES_UPDATED+=("pyproject.toml")

# 3) __init__.py  __version__ = "X.Y.Z"
update_with_python "src/spectramind/__init__.py" \
  '^(__version__\s*=\s*)"(?:\d+\.\d+\.\d+)"' \
  "\g<1>\"${NEW_VER}\""
FILES_UPDATED+=("src/spectramind/__init__.py")

# 4) version.py  __version__ = "X.Y.Z"
update_with_python "src/spectramind/version.py" \
  '^(__version__\s*=\s*)"(?:\d+\.\d+\.\d+)"' \
  "\g<1>\"${NEW_VER}\""
FILES_UPDATED+=("src/spectramind/version.py")

# 5) package.json  "version": "X.Y.Z"
update_with_python "package.json" "" "${NEW_VER}" "json"
FILES_UPDATED+=("package.json")

# 6) conda/meta.yaml  version: "X.Y.Z" or version: X.Y.Z
update_with_python "conda/meta.yaml" \
  '^(version:\s*)"?(?:\d+\.\d+\.\d+)"?' \
  "\g<1>\"${NEW_VER}\""
FILES_UPDATED+=("conda/meta.yaml")

# 7) docs/version.md  single line
if [[ -f "docs/version.md" ]]; then
  write_file "docs/version.md" "${NEW_VER}"
  FILES_UPDATED+=("docs/version.md")
fi

# ---------- git commit & tag ----------
if [[ "${NO_COMMIT}" == "1" ]]; then
  warn "Skipping git commit (--no-commit)."
else
  if [[ "${DRY_RUN}" == "1" ]]; then
    info "Would git add/commit changed files."
  else
    # add only existing, modified paths
    add_list=()
    for f in "${FILES_UPDATED[@]}"; do
      [[ -f "$f" ]] && add_list+=("$f")
    done
    if [[ ${#add_list[@]} -gt 0 ]]; then
      git add "${add_list[@]}" || true
    fi
    if [[ -n "$(git diff --cached --name-only)" ]]; then
      git commit -m "chore: bump version ${CUR_VER} → ${NEW_VER}"
      ok "Committed version bump."
    else
      warn "No changes staged; nothing to commit."
    fi
  fi
fi

if [[ "${DO_TAG}" == "1" ]]; then
  TAG="v${NEW_VER}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    info "Would create tag ${TAG} (signed=${DO_SIGN})."
  else
    if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
      warn "Tag ${TAG} already exists; skipping."
    else
      if [[ "${DO_SIGN}" == "1" ]]; then
        git tag -s "${TAG}" -m "Release ${TAG}"
      else
        git tag -a "${TAG}" -m "Release ${TAG}"
      fi
      ok "Created tag ${TAG}."
    fi
  fi
fi

ok "Version bump complete → ${NEW_VER}"
