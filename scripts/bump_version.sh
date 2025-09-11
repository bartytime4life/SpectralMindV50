#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# SpectraMind V50 — Version Bump Script (Upgraded, Portable)
# ------------------------------------------------------------------------------
# Safely bump the project version across the repo and create a signed/annotated
# git tag. Portable across Linux/macOS (BSD/GNU sed). No 'sort -V' dependency.
#
# Highlights:
# - Strict shell safety + failure traps with line numbers
# - Accepts explicit version (e.g., 0.2.0) or bump type (major|minor|patch)
# - Validates SemVer (X.Y.Z with optional -prerelease/+build)
# - Monotonic increase enforced (unless --force)
# - Updates (if present):
#     * VERSION                          → X.Y.Z
#     * pyproject.toml                   → version = "X.Y.Z"  (robust regex)
#     * src/spectramind/__init__.py      → __version__ = "X.Y.Z" / 'X.Y.Z'
#     * src/spectramind/version.py       → __version__ = "X.Y.Z" / 'X.Y.Z'
#     * package.json                     → "version": "X.Y.Z" (uses jq if available)
#     * conda/meta.yaml                  → version: "X.Y.Z" | version: X.Y.Z
#     * docs/version.md                  → single line X.Y.Z
#     * CHANGELOG.md                     → insert below "## [Unreleased]" or top
# - Refuses to proceed if working tree is dirty (unless --allow-dirty)
# - Tag collision check (fetches tags unless --no-fetch)
# - Optional GPG tag signing via GPG_SIGN=1
# - Optional push via --push (commits + tags if created)
# - CI output: prints new version and writes to $GITHUB_OUTPUT if set
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
#   --force                     Allow same/older version
#   --push                      git push && git push --tags (if tag created)
#   -h|--help                   Show help
# ------------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- Pretty logging ----------
info() { printf '[bump-version] %s\n' "$*"; }
warn() { printf '\033[33m[bump-version] WARN:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[bump-version] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

trap 'err "Failed at line $LINENO (command: $BASH_COMMAND)"' ERR

# ---------- Helpers ----------
usage() { sed -n '1,160p' "$0" | sed -n '1,/^set -Eeuo pipefail/p' | sed 's/^# \{0,1\}//g'; }
have()  { command -v "$1" >/dev/null 2>&1; }

semver_ok() {
  # X.Y.Z with optional -prerelease or +build (but we only compare X.Y.Z)
  [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+([\-+][A-Za-z0-9\.-]+)?$ ]]
}

# Compare only the numeric core X.Y.Z, portable across macOS/Linux.
# echoes: 0 if v1 < v2, 1 if equal, 2 if v1 > v2
semver_cmp_core() {
  local a b A B C D E F
  a="${1%%[-+]*}"; b="${2%%[-+]*}"
  IFS='.' read -r A B C <<<"$a"
  IFS='.' read -r D E F <<<"$b"
  A=${A:-0}; B=${B:-0}; C=${C:-0}
  D=${D:-0}; E=${E:-0}; F=${F:-0}
  if (( A < D )); then echo 0; return; fi
  if (( A > D )); then echo 2; return; fi
  if (( B < E )); then echo 0; return; fi
  if (( B > E )); then echo 2; return; fi
  if (( C < F )); then echo 0; return; fi
  if (( C > F )); then echo 2; return; fi
  echo 1
}

# portable sed replace (creates .bak then removes)
sed_inplace() {
  local pattern="$1" file="$2"
  sed -E -i.bak "$pattern" "$file" && rm -f "${file}.bak"
}

update_if_exists_sed() {
  local file="$1" desc="$2" pattern="$3"
  if [[ -f "$file" ]]; then
    sed_inplace "$pattern" "$file"
    info "Updated $desc in $file"
    UPDATED_FILES+=("$file")
  fi
}

update_package_json_version() {
  local file="package.json" ver="$1"
  [[ -f "$file" ]] || return 0
  if have jq; then
    # Preserve formatting minimally: write via jq then move
    tmp="${file}.tmp"
    jq --arg v "$ver" '.version = $v' "$file" > "$tmp" || err "jq failed on $file"
    mv "$tmp" "$file"
  else
    # Fallback sed: tolerate spacing
    sed_inplace "s/(\"version\"[[:space:]]*:[[:space:]]*\")[0-9]+\.[0-9]+\.[0-9]+([\-+][A-Za-z0-9\.-]+)?(\")/\1$ver\3/" "$file"
  fi
  info "Updated npm version in $file"
  UPDATED_FILES+=("$file")
}

insert_changelog_entry() {
  local file="CHANGELOG.md" ver="$1" today head
  [[ -f "$file" ]] || return 0
  today="$(date +%Y-%m-%d)"

  # If we find "## [Unreleased]" (Keep a Changelog), insert below it; else insert after the first header line
  if grep -qE '^##[[:space:]]*\[?Unreleased\]?' "$file"; then
    awk -v ver="$ver" -v date="$today" '
      BEGIN{ins=0}
      /^##[[:space:]]*\[?Unreleased\]?/ && ins==0 {
        print; print ""; print "## v" ver " — " date; print ""; print "- Placeholder: summarize changes."; ins=1; next
      }
      { print }
    ' "$file" > "${file}.tmp"
  else
    # Insert near top (after first line, which is usually # Changelog)
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
  fi
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
      if [[ -z "$NEW_VERSION" && -n "${1:-}" ]]; then
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
semver_ok "$NEW_VERSION" || err "Invalid version format: '$NEW_VERSION' (expected X.Y.Z or X.Y.Z-rc1)"

if [[ $FORCE -ne 1 ]]; then
  case "$(semver_cmp_core "$CURRENT_VERSION" "$NEW_VERSION")" in
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
declare -a UPDATED_FILES=()

# Always update VERSION
printf '%s\n' "$NEW_VERSION" > VERSION
info "Updated VERSION → $NEW_VERSION"
UPDATED_FILES+=("VERSION")

# pyproject.toml: tolerate whitespace and quotes, anchor on key
update_if_exists_sed "pyproject.toml" "pyproject version" \
  "s/^[[:space:]]*version[[:space:]]*=[[:space:]]*\"[0-9]+\.[0-9]+\.[0-9]+([\-+][A-Za-z0-9\.-]+)?\"/[version = \"$NEW_VERSION\"]/"

# src/spectramind/__init__.py or version.py: allow ' or "
update_if_exists_sed "src/spectramind/__init__.py" "__version__" \
  "s/^[[:space:]]*__version__[[:space:]]*=[[:space:]]*['\"][^'\"]*['\"]/__version__ = \"$NEW_VERSION\"/"

update_if_exists_sed "src/spectramind/version.py" "__version__" \
  "s/^[[:space:]]*__version__[[:space:]]*=[[:space:]]*['\"][^'\"]*['\"]/__version__ = \"$NEW_VERSION\"/"

# package.json (jq preferred)
update_package_json_version "$NEW_VERSION"

# conda/meta.yaml: quoted or unquoted
if [[ -f "conda/meta.yaml" ]]; then
  sed_inplace "s/^[[:space:]]*version:[[:space:]]*\"?[0-9]+\.[0-9]+\.[0-9]+([\-+][A-Za-z0-9\.-]+)?\"?/version: \"$NEW_VERSION\"/" "conda/meta.yaml"
  info "Updated conda version in conda/meta.yaml"
  UPDATED_FILES+=("conda/meta.yaml")
fi

# docs/version.md
if [[ -f "docs/version.md" ]]; then
  printf '%s\n' "$NEW_VERSION" > docs/version.md
  info "Updated docs/version.md"
  UPDATED_FILES+=("docs/version.md")
fi

# CHANGELOG.md
insert_changelog_entry "$NEW_VERSION"

# ---------- Commit / Tag ----------
if [[ $NO_COMMIT -eq 1 ]]; then
  info "Skipping git commit (per --no-commit)."
else
  if ((${#UPDATED_FILES[@]}==0)); then
    warn "No files updated; skipping commit."
  else
    git add "${UPDATED_FILES[@]}"
    git commit -m "chore(version): bump to $NEW_VERSION"
    info "Created commit for version bump."
  fi
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