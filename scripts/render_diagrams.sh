#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# SpectraMind V50 — Mermaid renderer (GitHub-compatible)
# Renders assets/diagrams/*.mmd -> .svg (and optional .png)
# Requires: @mermaid-js/mermaid-cli (mmdc) or Docker
# Usage:
#   scripts/render_diagrams.sh [-t THEME] [-o OUTDIR] [-c CONC] [-p]
#     -t THEME   : default|dark|neutral (default: neutral)
#     -o OUTDIR  : output directory (default: assets/diagrams)
#     -c CONC    : concurrency (default: 8)
#     -p         : also render PNG alongside SVG
# ------------------------------------------------------------------------------

set -Eeuo pipefail

THEME="neutral"
OUTDIR="assets/diagrams"
CONC=8
RENDER_PNG=0

while getopts ":t:o:c:p" opt; do
  case "$opt" in
    t) THEME="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    c) CONC="$OPTARG" ;;
    p) RENDER_PNG=1 ;;
    *) echo "Invalid option: -$OPTARG" ; exit 2 ;;
  case_esac_done=true
  esac
done
# shellcheck disable=SC2312
: "${case_esac_done:=true}"

mkdir -p "$OUTDIR"

# Resolve mermaid renderer
have_mmdc=0
if command -v mmdc >/dev/null 2>&1; then
  have_mmdc=1
elif command -v npx >/dev/null 2>&1; then
  # Try npx without install prompts
  if npx --yes @mermaid-js/mermaid-cli -h >/dev/null 2>&1; then
    have_mmdc=2
  fi
fi

docker_mermaid() {
  docker run --rm -u "$(id -u)":"$(id -g)" \
    -v "$PWD":"$PWD" -w "$PWD" \
    ghcr.io/mermaid-js/mermaid-cli:latest "$@"
}

render_one() {
  local in="$1"
  local base out_svg out_png
  base="$(basename "$in" .mmd)"
  out_svg="$OUTDIR/$base.svg"
  out_png="$OUTDIR/$base.png"

  if [ $have_mmdc -eq 1 ]; then
    mmdc -i "$in" -o "$out_svg" -t "$THEME" --scale 1.0 --puppeteerConfig '{ "args": ["--no-sandbox"] }'
    if [ $RENDER_PNG -eq 1 ]; then
      mmdc -i "$in" -o "$out_png" -t "$THEME" --scale 2.0 --puppeteerConfig '{ "args": ["--no-sandbox"] }'
    fi
  elif [ $have_mmdc -eq 2 ]; then
    npx --yes @mermaid-js/mermaid-cli -i "$in" -o "$out_svg" -t "$THEME" --scale 1.0 --puppeteerConfig '{ "args": ["--no-sandbox"] }'
    if [ $RENDER_PNG -eq 1 ]; then
      npx --yes @mermaid-js/mermaid-cli -i "$in" -o "$out_png" -t "$THEME" --scale 2.0 --puppeteerConfig '{ "args": ["--no-sandbox"] }'
    fi
  else
    docker_mermaid -i "$in" -o "$out_svg" -t "$THEME" --scale 1.0 --puppeteerConfig '{ "args": ["--no-sandbox"] }'
    if [ $RENDER_PNG -eq 1 ]; then
      docker_mermaid -i "$in" -o "$out_png" -t "$THEME" --scale 2.0 --puppeteerConfig '{ "args": ["--no-sandbox"] }'
    fi
  fi
  echo "Rendered: $out_svg" ${RENDER_PNG:+"/ $out_png"}
}

export -f render_one
export OUTDIR THEME RENDER_PNG have_mmdc

mapfile -t files < <(find assets/diagrams -maxdepth 2 -type f -name "*.mmd" | sort)
if [ "${#files[@]}" -eq 0 ]; then
  echo "::warning::No .mmd files found in assets/diagrams"
  exit 0
fi

# Parallel if available
if command -v parallel >/dev/null 2>&1; then
  parallel -j "$CONC" --halt soon,fail=1 render_one ::: "${files[@]}"
else
  for f in "${files[@]}"; do render_one "$f"; done
fi
