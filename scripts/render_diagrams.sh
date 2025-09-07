#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Mermaid Diagram Renderer (Upgraded)
# -----------------------------------------------------------------------------
# Renders Mermaid (*.mmd, *.mermaid) to SVG (and optionally PNG).
# Prefers local mermaid-cli (mmdc); falls back to Docker (disabled on Kaggle).
# Incremental by default: only re-renders if source is newer than output.
#
# Sources:
#   - assets/diagrams/
#   - docs/diagrams/
#
# Usage:
#   ./scripts/render_diagrams.sh [options]
#
# Options:
#   -o, --outdir DIR         Output directory (default: alongside source)
#   -s, --source DIR         Additional source directory (can repeat)
#   -t, --theme THEME        Mermaid theme [default|dark|forest|neutral] (default: default)
#   -b, --background HEX     Background (transparent | #ffffff ...) (default: transparent)
#   -p, --png                Also produce PNG next to SVG
#   -c, --concurrency N      Parallel renders (default: 4 when xargs supports -P)
#   -f, --force              Force rebuild (ignore mtimes)
#   -l, --list               List diagrams that would be rendered and exit
#   -k, --keep-going         Keep going on errors (default: fail fast)
#   -q, --quiet              Less verbose output
#   -h, --help               Show help and exit
#
# Advanced (passed to mmdc when applicable):
#       --width N            Page width (px)
#       --height N           Page height (px)
#       --theme-vars FILE    --themeVariables FILE (JSON)
#       --theme-css  FILE    --themeCSS FILE (CSS)
#
# Notes:
# - Fails fast on errors (set -euo pipefail). Use -k/--keep-going to continue on errors.
# - Auto-detects Kaggle; Docker fallback is disabled in Kaggle.
# - Requires mermaid-cli (npm i -g @mermaid-js/mermaid-cli) or Docker.
# - Emits a manifest JSON at artifacts/diagrams_manifest.json for CI diffing.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ----------------------------------------------------------------
THEME="default"
BGCOL="transparent"
RENDER_PNG="0"
OUTDIR=""                      # empty => same directory as source
CONCURRENCY="4"
QUIET="0"
FORCE="0"
LIST_ONLY="0"
KEEP_GOING="0"

# Advanced mmdc knobs (empty when unset)
MD_WIDTH=""
MD_HEIGHT=""
MD_THEME_VARS=""
MD_THEME_CSS=""

# Default sources matching repo layout
SOURCES=(
  "assets/diagrams"
  "docs/diagrams"
)

# Manifest location
MANIFEST_DIR="artifacts"
MANIFEST_PATH="$MANIFEST_DIR/diagrams_manifest.json"

# --- Helpers -----------------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { [ "$QUIET" = "1" ] || echo -e "[ $(timestamp) ] [render_diagrams] $*"; }
warn() { echo -e "[ $(timestamp) ] [render_diagrams][WARN] $*" >&2; }
die() { echo -e "[ $(timestamp) ] [render_diagrams][ERROR] $*" >&2; exit 1; }

detect_env() {
  if [ -d "/kaggle/input" ]; then echo "kaggle"; else echo "local"; fi
}
ENV_TYPE="$(detect_env)"

has_cmd() { command -v "$1" >/dev/null 2>&1; }

MMDC_BIN=""
DOCKER_IMAGE="ghcr.io/mermaid-js/mermaid-cli:latest"  # official image

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
  die "No mermaid renderer found. Install 'mmdc' (npm i -g @mermaid-js/mermaid-cli) or enable Docker (not in Kaggle)."
}

usage() {
  sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'
}

# --- Parse args --------------------------------------------------------------
ADV_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--outdir)      OUTDIR="${2:-}"; shift 2 ;;
    -t|--theme)       THEME="${2:-}"; shift 2 ;;
    -b|--background)  BGCOL="${2:-}"; shift 2 ;;
    -p|--png)         RENDER_PNG="1"; shift 1 ;;
    -c|--concurrency) CONCURRENCY="${2:-}"; shift 2 ;;
    -s|--source)      SOURCES+=("${2:-}"); shift 2 ;;
    -f|--force)       FORCE="1"; shift 1 ;;
    -l|--list)        LIST_ONLY="1"; shift 1 ;;
    -k|--keep-going)  KEEP_GOING="1"; shift 1 ;;
    -q|--quiet)       QUIET="1"; shift 1 ;;
    --width)          MD_WIDTH="${2:-}"; shift 2 ;;
    --height)         MD_HEIGHT="${2:-}"; shift 2 ;;
    --theme-vars)     MD_THEME_VARS="${2:-}"; shift 2 ;;
    --theme-css)      MD_THEME_CSS="${2:-}"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) warn "Unknown arg: $1"; usage; exit 1 ;;
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

# --- Build final mmdc args (advanced) ---------------------------------------
build_mmdc_args() {
  local extra=()
  [ -n "$MD_WIDTH" ]       && extra+=("--width" "$MD_WIDTH")
  [ -n "$MD_HEIGHT" ]      && extra+=("--height" "$MD_HEIGHT")
  [ -n "$MD_THEME_VARS" ]  && extra+=("--themeVariables" "$MD_THEME_VARS")
  [ -n "$MD_THEME_CSS" ]   && extra+=("--themeCSS" "$MD_THEME_CSS")
  printf '%s\0' "${extra[@]}" | xargs -0
}

MMDC_ADV_ARGS=()
# shellcheck disable=SC2207
MMDC_ADV_ARGS=($(build_mmdc_args))

# --- Incremental logic -------------------------------------------------------
# Return 0 if rebuild required, 1 if up-to-date
needs_rebuild() {
  local src="$1" svg="$2" png="$3"
  if [ "$FORCE" = "1" ]; then return 0; fi
  # If SVG missing, rebuild
  [ -f "$svg" ] || return 0
  # If PNG requested and missing, rebuild
  if [ "$RENDER_PNG" = "1" ] && [ ! -f "$png" ]; then return 0; fi
  # If src newer than outputs, rebuild
  local src_mtime; src_mtime=$(stat -c %Y "$src" 2>/dev/null || stat -f %m "$src")
  local svg_mtime; svg_mtime=$(stat -c %Y "$svg" 2>/dev/null || stat -f %m "$svg")
  [ "$src_mtime" -gt "$svg_mtime" ] && return 0
  if [ "$RENDER_PNG" = "1" ] && [ -f "$png" ]; then
    local png_mtime; png_mtime=$(stat -c %Y "$png" 2>/dev/null || stat -f %m "$png")
    [ "$src_mtime" -gt "$png_mtime" ] && return 0
  fi
  return 1
}

# --- Rendering ---------------------------------------------------------------
render_one() {
  local src="$1"
  local base stem dir svg_out png_out
  base="$(basename "$src")"
  stem="${base%.*}"

  # Decide output directory
  if [ -n "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
    svg_out="$OUTDIR/${stem}.svg"
    png_out="$OUTDIR/${stem}.png"
  else
    dir="$(dirname "$src")"
    svg_out="${dir}/${stem}.svg"
    png_out="${dir}/${stem}.png"
  fi

  if [ "$LIST_ONLY" = "1" ]; then
    echo "$src"
    return 0
  fi

  if needs_rebuild "$src" "$svg_out" "$png_out"; then
    log "⇢ Render: $src"
  else
    log "↻ Skip (up-to-date): $src"
    return 0
  fi

  if [ "$MMDC_BIN" = "mmdc" ]; then
    # local CLI path
    if ! mmdc -i "$src" -o "$svg_out" -t "$THEME" -b "$BGCOL" "${MMDC_ADV_ARGS[@]}" >/dev/null 2>&1; then
      if [ "$KEEP_GOING" = "1" ]; then warn "mmdc failed: $src"; return 0; else echo "[render_diagrams][ERROR] mmdc failed: $src" >&2; return 1; fi
    fi
    if [ "$RENDER_PNG" = "1" ]; then
      if ! mmdc -i "$src" -o "$png_out" -t "$THEME" -b "$BGCOL" "${MMDC_ADV_ARGS[@]}" >/dev/null 2>&1; then
        if [ "$KEEP_GOING" = "1" ]; then warn "mmdc (png) failed: $src"; return 0; else echo "[render_diagrams][ERROR] mmdc (png) failed: $src" >&2; return 1; fi
      fi
    fi
  else
    # Docker path
    local repo_root rel_in rel_svg rel_png
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    rel_in="${src#$repo_root/}"
    rel_svg="${svg_out#$repo_root/}"
    rel_png="${png_out#$repo_root/}"

    if ! docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
        "$DOCKER_IMAGE" mmdc -i "$rel_in" -o "$rel_svg" -t "$THEME" -b "$BGCOL" "${MMDC_ADV_ARGS[@]}" >/dev/null 2>&1; then
      if [ "$KEEP_GOING" = "1" ]; then warn "docker mmdc failed: $src"; return 0; else echo "[render_diagrams][ERROR] docker mmdc failed: $src" >&2; return 1; fi
    fi

    if [ "$RENDER_PNG" = "1" ]; then
      if ! docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
          "$DOCKER_IMAGE" mmdc -i "$rel_in" -o "$rel_png" -t "$THEME" -b "$BGCOL" "${MMDC_ADV_ARGS[@]}" >/dev/null 2>&1; then
        if [ "$KEEP_GOING" = "1" ]; then warn "docker mmdc (png) failed: $src"; return 0; else echo "[render_diagrams][ERROR] docker mmdc (png) failed: $src" >&2; return 1; fi
      fi
    fi
  fi

  [ "$QUIET" = "1" ] || echo "[render_diagrams] ✔ ${src} → ${svg_out}$( [ "$RENDER_PNG" = "1" ] && echo ", ${png_out}" )"
  return 0
}

# Parallel if available
run_parallel() {
  if [ "$LIST_ONLY" = "1" ]; then
    for f in "${FILES[@]}"; do render_one "$f"; done
    exit 0
  fi
  if xargs --help >/dev/null 2>&1; then
    # shellcheck disable=SC2016
    printf '%s\0' "${FILES[@]}" | xargs -0 -n1 -P "${CONCURRENCY}" bash -lc 'render_one "$0"'
  else
    # Fallback: serial
    for f in "${FILES[@]}"; do render_one "$f"; done
  fi
}

log "Rendering with theme='$(printf "%s" "$THEME")', bg='$(printf "%s" "$BGCOL")', png=$RENDER_PNG, outdir='${OUTDIR:-<src dir>}'"
[ "${#MMDC_ADV_ARGS[@]}" -gt 0 ] && log "Advanced mmdc args: ${MMDC_ADV_ARGS[*]}"

run_parallel

# --- Write manifest for CI ---------------------------------------------------
mkdir -p "$MANIFEST_DIR"
# Collect outputs
OUTS=()
for dir in "${SOURCES[@]}"; do
  [ -d "$dir" ] || continue
  while IFS= read -r -d '' f; do OUTS+=("$f"); done < <(find "$dir" -type f \( -name '*.svg' -o -name '*.png' \) -print0 2>/dev/null || true)
done
if [ -n "$OUTDIR" ] && [ -d "$OUTDIR" ]; then
  while IFS= read -r -d '' f; do OUTS+=("$f"); done < <(find "$OUTDIR" -type f \( -name '*.svg' -o -name '*.png' \) -print0 2>/dev/null || true)
fi

# sha256 helper
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; \
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'; \
  else echo "unavailable"; fi
}

{
  echo "{"
  echo "  \"generated_at\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\","
  echo "  \"environment\": \"${ENV_TYPE}\","
  echo "  \"theme\": \"${THEME}\","
  echo "  \"background\": \"${BGCOL}\","
  echo "  \"png\": ${RENDER_PNG},"
  echo "  \"outdir\": \"${OUTDIR}\","
  echo "  \"sources\": ["
  for i in "${!SOURCES[@]}"; do
    s="${SOURCES[$i]}"; printf "    \"%s\"%s\n" "$s" $([ "$i" -lt $((${#SOURCES[@]}-1)) ] && echo "," || true)
  done
  echo "  ],"
  echo "  \"outputs\": ["
  for i in "${!OUTS[@]}"; do
    f="${OUTS[$i]}"
    sum="$(sha256 "$f")"
    printf "    {\"path\": \"%s\", \"sha256\": \"%s\"}%s\n" "$(printf "%s" "$f")" "$sum" $([ "$i" -lt $((${#OUTS[@]}-1)) ] && echo "," || true)
  done
  echo "  ]"
  echo "}"
} > "$MANIFEST_PATH"

log "Manifest -> $MANIFEST_PATH"
log "All diagrams rendered successfully ✅"
