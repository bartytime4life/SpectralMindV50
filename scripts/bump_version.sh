#!/usr/bin/env bash
# ----------------------------------------------------------------------
# SpectraMind V50 — Version Bump Script
# ----------------------------------------------------------------------
# Safely bump the project version across the repo.
#
# Features:
# - Strict shell safety (set -euo pipefail).
# - Updates VERSION file.
# - Creates a git commit and annotated tag.
# - Validates semantic version format.
# - Optionally signs the tag if GPG_SIGN=1 is exported.
# - Outputs the new version for CI/CD pipelines.
#
# Usage:
#   ./scripts/bump_version.sh 0.2.0
#
# ----------------------------------------------------------------------

set -euo pipefail

# --- Helpers -----------------------------------------------------------
err() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[bump-version] $*"; }

# --- Input -------------------------------------------------------------
NEW_VERSION="${1:-}"
[[ -z "$NEW_VERSION" ]] && err "Usage: $0 <new-version>"

# Validate semver (basic)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9\.-]+)?$ ]]; then
  err "Invalid version format: '$NEW_VERSION' (expected semver, e.g. 1.2.3 or 1.2.3-rc1)"
fi

# --- Update VERSION file ----------------------------------------------
echo "$NEW_VERSION" > VERSION
info "Updated VERSION file → $NEW_VERSION"

# --- Git commit -------------------------------------------------------
git add VERSION
git commit -m "chore(version): bump to $NEW_VERSION"

# --- Git tag ----------------------------------------------------------
TAG="v$NEW_VERSION"
if [[ "${GPG_SIGN:-0}" == "1" ]]; then
  git tag -s "$TAG" -m "Release $NEW_VERSION"
else
  git tag -a "$TAG" -m "Release $NEW_VERSION"
fi
info "Created git tag $TAG"

# --- Output for CI/CD -------------------------------------------------
echo "$NEW_VERSION"
