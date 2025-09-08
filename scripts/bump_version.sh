#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# SpectraMind V50 — Version Bump Script (Upgraded)
# ------------------------------------------------------------------------------
# Safely bump the project version across the repo and create a signed/annotated
# git tag. Portable across Linux/macOS (BSD/GNU sed).
#
# Highlights:
# - Strict shell safety + failure traps with line numbers
# - Accepts either explicit version (e.g., 0.2.0) or bump type (major|minor|patch)
# - Validates SemVer and enforces monotonic increases (unless --force)
# - Updates VERSION and (if present) common version fields in:
#     * pyproject.toml                  → version = "X.Y.Z"
#     * src/spectramind/__init__.py     → __version__ = "X.Y.Z"
#     * src/spectramind/version.py      → __version__ = "X.Y.Z"
#     * package.json                    → "version": "X.Y.Z"
#     * conda/meta.yaml                 → version: "X.Y.Z"
#     * docs/version.md                 → (single line) X.Y.Z
#     * CHANGELOG.md                    → inserts "## vX.Y.Z — YYYY-MM-DD" if present
# - Refuses to proceed if the working tree is dirty (unless --allow-dirty)
# - Refuses to clobber existing tag; fetches tags first (unless --no-fetch)
# - Optional GPG tag signing via GPG_SIGN=1
# - Optional push via --push (pushes commits + tags)
# - CI output: prints the new version and writes to $GITHUB_OUTPUT if set
#
# Usage:
#   scripts/bump_version.sh 0.2.0 [--push]
#   scripts/bump_version.sh --bump minor [--push]
#
# Options:
#   --bump {major|minor|patch}  Derive next version from current VERSION
#   --allow-dirty               Skip clean working tree check
#   --no-fetch                  Do not fetch tags before checking collisions
#   --no-tag                    Update files/commit only (no tag)
#   --no-commit                 Update files only (no commit, no tag)
#   --force                     Allow same/older version (not recommended)
#   --push                      git push && git push --tags (if tag created)
#   -h|--help                   Show help
#
# Examples:
#   scripts/bump_version.sh 1.3.0 --push
#   scripts/bump_version.sh --bump patch
#
# Notes:
#   - Requires git in PATH and a repo with commit rights.
#   - For macOS sed compatibility we use in-place backups then rm *.bak.
# ------------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- Pretty logging ----------
info() { printf '[bump-version] %s\n' "$*"; }
warn() { printf '\033[33m[bump-version] WARN:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[bump-version] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

trap 'err "Failed at line $LINENO (command: $BASH_COMMAND)"' ERR

# ---------- Helpers ----------
usage() { sed -n '1,120p' "$0" | sed -n '1,/^set -Eeuo pipefail/p' | sed 's/^# \{0,1\}//g'; }

have() { command -v "$1" >/dev/null 2>&1; }

semver_ok() {
  [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+([\-+][A-Za-z0-9\.-]+)?$ ]]
}

# returns 0 if v1 < v2, 1 if v1 == v2, 2 if v1 > v2
semver_cmp() {
  # strip pre-release/build for ordering here; rely on sort -V
  local a b
  a="${1%%[-+]*}"
  b="${2%%[-+]*}"
  if [[ "$a" == "$b" ]]; then echo 1; return; fi
  if [[ "$(printf '%s\n%s\n' "$a" "$b" | sort -V | head -1)" == "$a" ]]; then
    echo 0
  else
    echo 2
  fi
}

# portable sed replace (creates .bak then removes)
sed_inplace() {
  local pattern="$1" file="$2"
  sed -E -i.bak "$pattern" "$file" && rm -f "${file}.bak"
}

update_if_exists() {
  local file="$1" pattern_desc="$2" pattern="$3"
  if [[ -f "$file" ]]; then
    sed_inplace "$pattern" "$file"
    info "Updated $pattern_desc in $file"
    UPDATED_FILES+=("$file")
  fi
}

insert_changelog_entry() {
  local file="CHANGELOG.md" ver="$1" today
  [[ -f "$file" ]] || return 0
  today="$(date +%Y-%m-%d)"
  # Insert after the first H1 or at top if none
  awk -v ver="$ver" -v date="$today" '
    NR==1{
      print; 
      print "";
      print "## v" ver " — " date;
      print "";
      print "- Placeholder: summarize changes.";
      next
    }
    { print }
  ' "$file" > "${file}.tmp"
  mv "${file}.tmp" "$file"
  info "Inserted CHANGELOG entry for v$ver"
  UPDATED_FILES+=("$file")
}

# ---------- Parse args ----------
NEW_VERSION=""
BUMP_TYPE=""
ALLOW_DIRTY=0
NO_FETCH=0
NO_TAG=0
NO_COMMIT=0
FORCE=0
DO_PUSH=0

while (( "$#" )); do
  case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    --bump) BUMP_TYPE="${2:-}"; shift ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --no-fetch) NO_FETCH=1 ;;
    --no-tag) NO_TAG=1 ;;
    --no-commit) NO_COMMIT=1 ;;
    --force) FORCE=1 ;;
    --push) DO_PUSH=1 ;;
    *) 
       if [[ -z "$NEW_VERSION" && "$1" != "" ]]; then
         NEW_VERSION="$1"
       else
         err "Unknown argument: $1"
       fi
       ;;
  esac
  shift || true
done

# ---------- Git preflight ----------
have git || err "git not found in PATH"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || err "Not inside a git repository"

if [[ $ALLOW_DIRTY -ne 1 ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    err "Working tree is dirty. Commit or stash changes, or use --allow-dirty."
  fi
fi

# ---------- Determine current version ----------
CURRENT_VERSION="0.0.0"
if [[ -f VERSION ]]; then
  CURRENT_VERSION="$(tr -d ' \t\r\n' < VERSION || echo "0.0.0")"
fi
if ! semver_ok "$CURRENT_VERSION"; then
  warn "Existing VERSION '$CURRENT_VERSION' does not look like SemVer; treating as 0.0.0"
  CURRENT_VERSION="0.0.0"
fi

# ---------- Derive new version if --bump used ----------
if [[ -n "$BUMP_TYPE" && -z "$NEW_VERSION" ]]; then
  IFS='.' read -r MAJ MIN PAT <<< "${CURRENT_VERSION%%[-+]*}"
  case "$BUMP_TYPE" in
    major) MAJ=$((MAJ+1)); MIN=0; PAT=0 ;;
    minor) MIN=$((MIN+1)); PAT=0 ;;
    patch) PAT=$((PAT+1)) ;;
    *) err "--bump must be one of: major, minor, patch" ;;
  esac
  NEW_VERSION="${MAJ}.${MIN}.${PAT}"
fi

[[ -n "$NEW_VERSION" ]] || err "Provide new version or use --bump {major|minor|patch}"

semver_ok "$NEW_VERSION" || err "Invalid version format: '$NEW_VERSION' (expected SemVer, e.g. 1.2.3 or 1.2.3-rc1)"

if [[ $FORCE -ne 1 ]]; then
  case "$(semver_cmp "$CURRENT_VERSION" "$NEW_VERSION")" in
    0) : ;;        # current < new → OK
    1) err "New version equals current ($CURRENT_VERSION). Use --force to allow." ;;
    2) err "New version ($NEW_VERSION) is less than current ($CURRENT_VERSION). Use --force to allow." ;;
  esac
fi

# ---------- Check tag collisions ----------
TAG="v$NEW_VERSION"
if [[ $NO_TAG -ne 1 && $NO_FETCH -ne 1 ]]; then
  info "Fetching tags to check for collisions..."
  git fetch --tags --quiet || warn "Tag fetch failed; continuing"
fi
if [[ $NO_TAG -ne 1 ]]; then
  if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    err "Tag $TAG already exists."
  fi
fi

# ---------- Update files ----------
UPDATED_FILES=()

# Always update VERSION
printf '%s\n' "$NEW_VERSION" > VERSION
info "Updated VERSION → $NEW_VERSION"
UPDATED_FILES+=("VERSION")

# Optional/common locations
update_if_exists "pyproject.toml" "pyproject version" \
  "s/^(version\s*=\s*\")[0-9]+\.[0-9]+\.[0-9]+([\-+][A-Za-z0-9\.-]+)?(\")/\1$NEW_VERSION\3/"

update_if_exists "src/spectramind/__init__.py" "__version__" \
  "s/^(__version__\s*=\s*\")[^\"]*(\")/\1$NEW_VERSION\2/"

update_if_exists "src/spectramind/version.py" "__version__" \
  "s/^(__version__\s*=\s*\")[^\"]*(\")/\1$NEW_VERSION\2/"

update_if_exists "package.json" "npm version" \
  "s/(\"version\"\s*:\s*\")[0-9]+\.[0-9]+\.[0-9]+([\-+][A-Za-z0-9\.-]+)?(\")/\1$NEW_VERSION\3/"

update_if_exists "conda/meta.yaml" "conda version" \
  "s/^(version:\s*\")[0-9]+\.[0-9]+\.[0-9]+([\-+][A-Za-z0-9\.-]+)?(\")/\1$NEW_VERSION\3/"

if [[ -f "docs/version.md" ]]; then
  echo "$NEW_VERSION" > docs/version.md
  info "Updated docs/version.md"
  UPDATED_FILES+=("docs/version.md")
fi

insert_changelog_entry "$NEW_VERSION"

# ---------- Commit / Tag ----------
if [[ $NO_COMMIT -eq 1 ]]; then
  info "Skipping git commit (per --no-commit)."
else
  git add "${UPDATED_FILES[@]}"
  git commit -m "chore(version): bump to $NEW_VERSION"
  info "Created commit for version bump."
fi

if [[ $NO_TAG -eq 1 ]]; then
  info "Skipping git tag (per --no-tag)."
else
  if [[ "${GPG_SIGN:-0}" == "1" ]]; then
    git tag -s "$TAG" -m "Release $NEW_VERSION"
  else
    git tag -a "$TAG" -m "Release $NEW_VERSION"
  fi
  info "Created git tag $TAG"
fi

# ---------- Push (optional) ----------
if [[ $DO_PUSH -eq 1 ]]; then
  info "Pushing commit..."
  git push
  if [[ $NO_TAG -ne 1 ]]; then
    info "Pushing tags..."
    git push --tags
  fi
fi

# ---------- CI outputs ----------
echo "$NEW_VERSION"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "new_version=$NEW_VERSION"
    echo "tag=$TAG"
  } >> "$GITHUB_OUTPUT"
fi

info "Done. Current version: $NEW_VERSION"
