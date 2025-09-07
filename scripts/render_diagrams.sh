#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Mermaid Diagram Renderer
# -----------------------------------------------------------------------------
# Renders all Mermaid diagrams (*.mmd, *.mermaid) to SVG (and optionally PNG).
# Uses local mermaid-cli (mmdc) if available, else falls back to Docker.
#
# Sources:
#   - assets/diagrams/
#   - docs/diagrams/
#
# Usage:
#   ./scripts/render_diagrams.sh [options]
#
# Options:
#   -o, --outdir DIR      Output directory (default: same as source file)
#   -t, --theme THEME     Mermaid theme (default: default) [default|dark|forest|neutral]
#   -b, --background HEX  Background color (e.g., transparent, #ffffff) (default: transparent)
#   -p, --png             Also produce PNG next to SVG
#   -c, --concurrency N   Parallel renders (default: 4 if xargs supports -P)
#   -s, --source DIR      Additional source directory (can repeat)
#   -q, --quiet           Less verbose output
#   -h, --help            Show help and exit
#
# Notes:
# - Fails fast on errors (set -euo pipefail).
# - Auto-detects Kaggle; Docker fallback is disabled in Kaggle.
# - Requires mermaid-cli (npm i -g @mermaid-js/mermaid-cli) or Docker.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ----------------------------------------------------------------
THEME="default"
BGCOL="transparent"
RENDER_PNG="0"
OUTDIR=""                      # empty = same directory as source
CONCURRENCY="4"
QUIET="0"

# Default sources matching repo layout
SOURCES=(
  "assets/diagrams"
  "docs/diagrams"
)

# --- Helpers -----------------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { [ "$QUIET" = "1" ] || echo -e "[ $(timestamp) ] [render_diagrams] $*"; }
warn() { echo -e "[ $(timestamp) ] [render_diagrams][WARN] $*" >&2; }
die() { echo -e "[ $(timestamp) ] [render_diagrams][ERROR] $*" >&2; exit 1; }

detect_env() {
  if [ -d "/kaggle/input" ]; then
    echo "kaggle"
  else
    echo "local"
  fi
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

MMDC_BIN=""
DOCKER_IMAGE="ghcr.io/mermaid-js/mermaid-cli:latest"  # official image
ENV_TYPE="$(detect_env)"

pick_renderer() {
  if has_cmd mmdc; then
    MMDC_BIN="mmdc"
    log "Using local mermaid-cli (mmdc)"
    return 0
  fi

  if has_cmd docker && [ "$ENV_TYPE" != "kaggle" ]; then
    MMDC_BIN="docker"
    log "Using Docker mermaid-cli image: $DOCKER_IMAGE"
    return 0
  fi

  die "No mermaid-cli found. Install with 'npm i -g @mermaid-js/mermaid-cli' or ensure Docker is available (not in Kaggle)."
}

usage() {
  sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//'
}

# --- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--outdir)      OUTDIR="${2:-}"; shift 2 ;;
    -t|--theme)       THEME="${2:-}"; shift 2 ;;
    -b|--background)  BGCOL="${2:-}"; shift 2 ;;
    -p|--png)         RENDER_PNG="1"; shift 1 ;;
    -c|--concurrency) CONCURRENCY="${2:-}"; shift 2 ;;
    -s|--source)      SOURCES+=("${2:-}"); shift 2 ;;
    -q|--quiet)       QUIET="1"; shift 1 ;;
    -h|--help)        usage; exit 0 ;;
    *) warn "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

# --- Resolve renderer --------------------------------------------------------
pick_renderer

# --- Collect input files -----------------------------------------------------
FILES=()
for dir in "${SOURCES[@]}"; do
  if [ -d "$dir" ]; then
    while IFS= read -r -d '' f; do FILES+=("$f"); done < <(find "$dir" -type f \( -name '*.mmd' -o -name '*.mermaid' \) -print0)
  else
    warn "Source directory not found: $dir"
  fi
done

if [ "${#FILES[@]}" -eq 0 ]; then
  warn "No Mermaid files found in sources: ${SOURCES[*]}"
  exit 0
fi

log "Found ${#FILES[@]} Mermaid file(s) in: ${SOURCES[*]}"

# --- Rendering ---------------------------------------------------------------
render_one() {
  src="$1"
  base="$(basename "$src")"
  stem="${base%.*}"

  # Decide output directory
  if [ -n "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
    svg_out="$OUTDIR/${stem}.svg"
    png_out="$OUTDIR/${stem}.png"
  else
    svg_out="$(dirname "$src")/${stem}.svg"
    png_out="$(dirname "$src")/${stem}.png"
  fi

  if [ "$MMDC_BIN" = "mmdc" ]; then
    # local CLI path
    mmdc -i "$src" -o "$svg_out" -t "$THEME" -b "$BGCOL" >/dev/null 2>&1 || {
      echo "[render_diagrams][ERROR] mmdc failed: $src" >&2; return 1;
    }
    if [ "$RENDER_PNG" = "1" ]; then
      mmdc -i "$src" -o "$png_out" -t "$THEME" -b "$BGCOL" >/dev/null 2>&1 || {
        echo "[render_diagrams][ERROR] mmdc (png) failed: $src" >&2; return 1;
      }
    fi
  else
    # Docker path
    # Mount repo root to /work; run mmdc inside the container
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    rel_in="${src#$repo_root/}"
    rel_svg="${svg_out#$repo_root/}"
    rel_png="${png_out#$repo_root/}"

    docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
      "$DOCKER_IMAGE" \
      mmdc -i "$rel_in" -o "$rel_svg" -t "$THEME" -b "$BGCOL" >/dev/null 2>&1 || {
        echo "[render_diagrams][ERROR] docker mmdc failed: $src" >&2; return 1;
      }

    if [ "$RENDER_PNG" = "1" ]; then
      docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
        "$DOCKER_IMAGE" \
        mmdc -i "$rel_in" -o "$rel_png" -t "$THEME" -b "$BGCOL" >/dev/null 2>&1 || {
          echo "[render_diagrams][ERROR] docker mmdc (png) failed: $src" >&2; return 1;
        }
    fi
  fi

  [ "$QUIET" = "1" ] || echo "[render_diagrams] ✔ ${src} → ${svg_out}$( [ "$RENDER_PNG" = "1" ] && echo ", ${png_out}" )"
  return 0
}

# Parallel if available
run_parallel() {
  if xargs --help >/dev/null 2>&1; then
    # shellcheck disable=SC2016
    printf '%s\0' "${FILES[@]}" | xargs -0 -n1 -P "${CONCURRENCY}" bash -lc 'render_one "$0"'
  else
    # Fallback: serial
    for f in "${FILES[@]}"; do render_one "$f"; done
  fi
}

log "Rendering with theme='$THEME', bg='$BGCOL', png=$RENDER_PNG, outdir='${OUTDIR:-<src dir>}'"
run_parallel

log "All diagrams rendered successfully ✅"
