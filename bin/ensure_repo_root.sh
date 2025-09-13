#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — ensure_repo_root.sh
# Find the repository root (git top-level or marker files) and optionally chdir.
# • Repo-root aware from any subdir
# • Deterministic marker search with configurable set
# • Exports SM_REPO_ROOT for downstream scripts
# • JSON summary for CI, quiet mode for scripting
# • Safe to source (affects caller shell) or execute (affects subshell)
# ------------------------------------------------------------------------------
# Usage:
#   bin/ensure_repo_root.sh [--no-chdir] [--markers list] [--json] [--quiet]
#                           [--strict] [-h|--help]
#
# Examples:
#   # Typical (cd to root, export SM_REPO_ROOT, print path)
#   bin/ensure_repo_root.sh
#
#   # CI JSON summary, no chdir (just detection + export)
#   bin/ensure_repo_root.sh --no-chdir --json --quiet
#
#   # Custom markers (comma-separated)
#   bin/ensure_repo_root.sh --markers pyproject.toml,dvc.yaml,.git
#
# Exit codes:
#   0 = success
#   2 = bad arguments
#   3 = repo root not found (when --strict), or unexpected failure
# ==============================================================================

set -Eeuo pipefail

# -------- pretty printing ------------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

usage() {
  cat <<'USAGE'
ensure_repo_root.sh — locate SpectraMind V50 repo root and optionally chdir

Options:
  --no-chdir           Do not chdir to the detected root (default: chdir)
  --markers LIST       Comma-separated marker files/dirs to detect the root.
                       Default: pyproject.toml,dvc.yaml,.git
  --json               Emit a JSON summary to stdout
  --quiet              Suppress non-error logs
  --strict             Exit non-zero if root is not found
  -h, --help           Show this help

Behavior:
  • First tries: `git rev-parse --show-toplevel`
  • Fallback: ascend from $PWD until any marker exists
  • On success: exports SM_REPO_ROOT and (by default) cd "$SM_REPO_ROOT"
  • When sourced ('. bin/ensure_repo_root.sh'), the directory change
    affects the caller shell; when executed, it affects only this process.
USAGE
}

trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"; exit 3' ERR

# -------- args ----------------------------------------------------------------
CHDIR=1
MARKERS="pyproject.toml,dvc.yaml,.git"
EMIT_JSON=0
QUIET="${QUIET:-0}"
STRICT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-chdir) CHDIR=0; shift ;;
    --markers)  MARKERS="${2:-}"; shift 2 ;;
    --json)     EMIT_JSON=1; shift ;;
    --quiet)    QUIET=1; shift ;;
    --strict)   STRICT=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# -------- split markers --------------------------------------------------------
IFS=',' read -r -a MARKER_ARR <<< "$MARKERS"
if [[ ${#MARKER_ARR[@]} -eq 0 ]]; then
  err "Empty marker set"; exit 2
fi

# -------- detection ------------------------------------------------------------
FOUND_ROOT=""
if command -v git >/dev/null 2>&1; then
  if git_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"; then
    if [[ -n "$git_root" && -d "$git_root" ]]; then
      FOUND_ROOT="$git_root"
    fi
  fi
fi

if [[ -z "$FOUND_ROOT" ]]; then
  # Ascend from PWD to /
  d="$PWD"
  while [[ "$d" != "/" ]]; do
    for m in "${MARKER_ARR[@]}"; do
      if [[ -e "$d/$m" ]]; then
        FOUND_ROOT="$d"
        break 2
      fi
    done
    d="$(dirname "$d")"
  done
fi

if [[ -z "$FOUND_ROOT" ]]; then
  warn "Repository root not found (markers: $MARKERS)"
  if [[ "$STRICT" -eq 1 ]]; then
    [[ "$EMIT_JSON" -eq 1 ]] && printf '{"ok":false,"reason":"not_found"}\n'
    exit 3
  else
    # Export empty to make intent explicit
    export SM_REPO_ROOT=""
    [[ "$EMIT_JSON" -eq 1 ]] && printf '{"ok":false,"reason":"not_found"}\n'
    exit 0
  fi
fi

# Normalize path
FOUND_ROOT="$(cd "$FOUND_ROOT" && pwd -P)"
export SM_REPO_ROOT="$FOUND_ROOT"

# -------- chdir if requested ---------------------------------------------------
CHDIR_RESULT="skipped"
if [[ "$CHDIR" -eq 1 ]]; then
  cd "$SM_REPO_ROOT"
  CHDIR_RESULT="changed"
fi

ok "Repo root: $SM_REPO_ROOT (chdir: $CHDIR_RESULT)"

# -------- json summary ---------------------------------------------------------
if [[ "$EMIT_JSON" -eq 1 ]]; then
  esc() { printf "%s" "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  printf '{'
  printf '"ok":true,'
  printf '"root":"%s",' "$(esc "$SM_REPO_ROOT")"
  printf '"chdir":"%s",' "$CHDIR_RESULT"
  printf '"markers":"%s",' "$(esc "$MARKERS")"
  printf '"cwd":"%s"' "$(esc "$PWD")"
  printf '}\n'
fi

exit 0

