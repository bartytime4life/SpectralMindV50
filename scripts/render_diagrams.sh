#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Mermaid Diagram Renderer (Ultra Upgraded)
# -----------------------------------------------------------------------------
# Renders Mermaid (*.mmd, *.mermaid) to SVG (and optionally PNG).
# Prefers local mermaid-cli (mmdc); falls back to Docker (disabled on Kaggle).
# Incremental by default: only re-renders if source is newer than output.
#
# Sources searched (by default):
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
#       --since REF          Only render files changed since git ref/branch/tag
#       --check              Validate only (no outputs written)
#
# Advanced (passed to mmdc when applicable):
#       --width N            Page width (px)
#       --height N           Page height (px)
#       --theme-vars FILE    --themeVariables FILE (JSON)
#       --theme-css  FILE    --themeCSS FILE (CSS)
#       --config FILE        --configFile FILE (mmdc JSON config)
#       --puppeteer FILE     --puppeteerConfigFile FILE
#
# Per-file overrides:
#   - YAML front-matter block at top of file:
#       ---
#       theme: dark
#       background: "#0b1021"
#       width: 1600
#       height: 900
#       png: true
#       ---
#   - Or header comments (first ~10 lines):  // @theme: dark
#
# Notes:
# - Fails fast on errors (set -Eeuo pipefail). Use -k/--keep-going to continue.
# - Auto-detects Kaggle; Docker fallback is disabled in Kaggle.
# - Requires mermaid-cli (npm i -g @mermaid-js/mermaid-cli) or Docker.
# - Emits manifest JSON at artifacts/diagrams_manifest.json for CI diffing.
# -----------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

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
CHECK_ONLY="0"
SINCE_REF=""

# Advanced mmdc knobs (empty when unset)
MD_WIDTH=""
MD_HEIGHT=""
MD_THEME_VARS=""
MD_THEME_CSS=""
MD_CONFIG_FILE=""
MD_PPTR_FILE=""

# Default sources matching repo layout
SOURCES=("assets/diagrams" "docs/diagrams")

# Manifest location
MANIFEST_DIR="artifacts"
MANIFEST_PATH="$MANIFEST_DIR/diagrams_manifest.json"

# Docker image
DOCKER_IMAGE="ghcr.io/mermaid-js/mermaid-cli:latest"

# --- Helpers -----------------------------------------------------------------
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { [[ "$QUIET" = "1" ]] || printf "[ %s ] [render_diagrams] %s\n" "$(timestamp)" "$*"; }
warn(){ printf "[ %s ] [render_diagrams][WARN] %s\n"  "$(timestamp)" "$*" >&2; }
die() { printf "[ %s ] [render_diagrams][ERROR] %s\n" "$(timestamp)" "$*" >&2; exit 1; }

detect_env(){ [[ -d "/kaggle/input" ]] && echo "kaggle" || echo "local"; }
ENV_TYPE="$(detect_env)"

has_cmd(){ command -v "$1" >/dev/null 2>&1; }

# BSD/GNU stat
mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"; }
fsize() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

sha256() {
  if has_cmd sha256sum; then sha256sum "$1" | awk '{print $1}';
  elif has_cmd shasum; then shasum -a 256 "$1" | awk '{print $1}';
  else echo "unavailable"; fi
}

usage(){ sed -n '1,180p' "$0" | sed 's/^# \{0,1\}//'; }

trap 'echo "[render_diagrams][ERROR] failed at line $LINENO: $BASH_COMMAND" >&2' ERR

# --- Parse args --------------------------------------------------------------
ADV_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--outdir)      OUTDIR="${2:-}"; shift 2 ;;
    -t|--theme)       THEME="${2:-}"; shift 2 ;;
    -b|--background)  BGCOL="${2:-}"; shift 2 ;;
    -p|--png)         RENDER_PNG="1"; shift ;;
    -c|--concurrency) CONCURRENCY="${2:-}"; shift 2 ;;
    -s|--source)      SOURCES+=("${2:-}"); shift 2 ;;
    -f|--force)       FORCE="1"; shift ;;
    -l|--list)        LIST_ONLY="1"; shift ;;
    -k|--keep-going)  KEEP_GOING="1"; shift ;;
    -q|--quiet)       QUIET="1"; shift ;;
        --since)      SINCE_REF="${2:-}"; shift 2 ;;
        --check)      CHECK_ONLY="1"; shift ;;
        --width)      MD_WIDTH="${2:-}"; shift 2 ;;
        --height)     MD_HEIGHT="${2:-}"; shift 2 ;;
        --theme-vars) MD_THEME_VARS="${2:-}"; shift 2 ;;
        --theme-css)  MD_THEME_CSS="${2:-}"; shift 2 ;;
        --config)     MD_CONFIG_FILE="${2:-}"; shift 2 ;;
        --puppeteer)  MD_PPTR_FILE="${2:-}"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) warn "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# --- Choose renderer ----------------------------------------------------------
MMDC_BIN=""
pick_renderer() {
  if has_cmd mmdc; then
    MMDC_BIN="mmdc"; log "Using local mermaid-cli (mmdc)"; return 0
  fi
  if has_cmd docker && [[ "$ENV_TYPE" != "kaggle" ]]; then
    MMDC_BIN="docker"; log "Using Docker mermaid-cli image: $DOCKER_IMAGE"; return 0
  fi
  die "No mermaid renderer found. Install 'mmdc' (npm i -g @mermaid-js/mermaid-cli) or enable Docker (not in Kaggle)."
}
pick_renderer

# --- Build extra mmdc args ---------------------------------------------------
build_mmdc_args() {
  local extra=()
  [[ -n "$MD_WIDTH"      ]] && extra+=("--width" "$MD_WIDTH")
  [[ -n "$MD_HEIGHT"     ]] && extra+=("--height" "$MD_HEIGHT")
  [[ -n "$MD_THEME_VARS" ]] && extra+=("--themeVariables" "$MD_THEME_VARS")
  [[ -n "$MD_THEME_CSS"  ]] && extra+=("--themeCSS" "$MD_THEME_CSS")
  [[ -n "$MD_CONFIG_FILE" ]] && extra+=("--configFile" "$MD_CONFIG_FILE")
  [[ -n "$MD_PPTR_FILE"   ]] && extra+=("--puppeteerConfigFile" "$MD_PPTR_FILE")
  printf '%s\0' "${extra[@]}" | xargs -0
}
# shellcheck disable=SC2207
MMDC_ADV_ARGS=($(build_mmdc_args))

# --- Collect candidate files --------------------------------------------------
collect_files() {
  local arr=()
  for dir in "${SOURCES[@]}"; do
    [[ -d "$dir" ]] || { warn "Source directory not found: $dir"; continue; }
    while IFS= read -r -d '' f; do arr+=("$f"); done < <(find "$dir" -type f \( -name '*.mmd' -o -name '*.mermaid' \) -print0)
  done
  printf '%s\n' "${arr[@]}"
}

apply_since_filter() {
  if [[ -z "$SINCE_REF" ]]; then cat; return; fi
  if ! has_cmd git || ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    warn "--since specified but not in a git repo; ignoring"; cat; return;
  fi
  while IFS= read -r f; do
    if git diff --name-only "$SINCE_REF"... -- "$f" | grep -q .; then echo "$f"; fi
  done
}

mapfile -t FILES < <(collect_files | apply_since_filter || true)

if [[ ${#FILES[@]} -eq 0 ]]; then
  warn "No Mermaid files found in sources: ${SOURCES[*]}"
  exit 0
fi

log "Found ${#FILES[@]} Mermaid file(s) in: ${SOURCES[*]}${SINCE_REF:+ (changed since '$SINCE_REF')}"

# --- Incremental logic -------------------------------------------------------
needs_rebuild() {
  local src="$1" svg="$2" png="$3"
  [[ "$FORCE" = "1" ]] && return 0
  [[ "$CHECK_ONLY" = "1" ]] && return 0
  [[ -f "$svg" ]] || return 0
  if [[ "$RENDER_PNG" = "1" && ! -f "$png" ]]; then return 0; fi
  local sm tm; sm="$(mtime "$src")"; tm="$(mtime "$svg")"
  [[ "$sm" -gt "$tm" ]] && return 0
  if [[ "$RENDER_PNG" = "1" && -f "$png" ]]; then
    local pm; pm="$(mtime "$png")"; [[ "$sm" -gt "$pm" ]] && return 0
  fi
  return 1
}

# --- Per-file overrides -------------------------------------------------------
# Parse yaml-front-matter (--- ... ---) or // @key: value style within first 12 lines
read_overrides() {
  local src="$1"
  python - "$src" <<'PY'
import io,sys,re,yaml,os
p=sys.argv[1]
try:
  with open(p,'r',encoding='utf-8',errors='ignore') as f:
    head="".join([next(f) for _ in range(12)])
except StopIteration:
  pass
except Exception:
  head=""

cfg={}
if head.startswith("---"):
  try:
    end=head.find("\n---",3)
    if end!=-1:
      block=head[3:end]
      y=yaml.safe_load(block) or {}
      if isinstance(y,dict):
        cfg.update(y)
  except Exception:
    pass

for m in re.finditer(r'^\s*//\s*@([a-zA-Z0-9_\-]+)\s*:\s*(.+)\s*$', head, re.M):
  cfg[m.group(1).strip().lower()]=m.group(2).strip()

def out(k): print(cfg.get(k,""))
for k in ("theme","background","width","height","png"):
  out(k)
PY
}

# --- Rendering ---------------------------------------------------------------
render_one() {
  local src="$1"
  local base stem dir svg_out png_out
  base="$(basename "$src")"
  stem="${base%.*}"

  # default outputs
  if [[ -n "$OUTDIR" ]]; then
    mkdir -p "$OUTDIR"
    svg_out="$OUTDIR/${stem}.svg"
    png_out="$OUTDIR/${stem}.png"
  else
    dir="$(dirname "$src")"
    svg_out="${dir}/${stem}.svg"
    png_out="${dir}/${stem}.png"
  fi

  if [[ "$LIST_ONLY" = "1" ]]; then
    echo "$src"
    return 0
  fi

  # per-file overrides
  local ov_theme ov_bg ov_w ov_h ov_png
  mapfile -t OV < <(read_overrides "$src")
  ov_theme="${OV[0]}"; ov_bg="${OV[1]}"; ov_w="${OV[2]}"; ov_h="${OV[3]}"; ov_png="${OV[4]}"
  local theme="${ov_theme:-$THEME}"
  local bg="${ov_bg:-$BGCOL}"
  local want_png="$RENDER_PNG"; [[ "${ov_png,,}" == "true" ]] && want_png="1"
  local extra_args=("${MMDC_ADV_ARGS[@]}")
  [[ -n "$ov_w" ]] && extra_args+=("--width" "$ov_w")
  [[ -n "$ov_h" ]] && extra_args+=("--height" "$ov_h")

  if needs_rebuild "$src" "$svg_out" "$png_out"; then
    [[ "$CHECK_ONLY" = "1" ]] && log "✓ Check: $src" || log "⇢ Render: $src"
  else
    log "↻ Skip (up-to-date): $src"
    echo "$src|$svg_out|0|skip" >> "$TMP_TIMINGS"
    return 0
  fi

  local t0 t1 dur rc
  t0=$(date +%s%3N 2>/dev/null || date +%s)

  # mmdc invocation (local or docker)
  if [[ "$MMDC_BIN" = "mmdc" ]]; then
    if [[ "$CHECK_ONLY" = "1" ]]; then
      # Validate by attempting render to temp, then remove
      local tmp_svg; tmp_svg="$(mktemp).svg"
      set +e
      mmdc -i "$src" -o "$tmp_svg" -t "$theme" -b "$bg" "${extra_args[@]}" 2> "$TMP_ERR"
      rc=$?
      rm -f "$tmp_svg"
      set -e
    else
      set +e
      mmdc -i "$src" -o "$svg_out" -t "$theme" -b "$bg" "${extra_args[@]}" 2> "$TMP_ERR"
      rc=$?
      set -e
    fi
    if [[ $rc -ne 0 ]]; then
      if [[ "$KEEP_GOING" = "1" ]]; then warn "mmdc failed: $src"; cat "$TMP_ERR" >&2; return 0; else
        echo "[render_diagrams][ERROR] mmdc failed: $src" >&2; cat "$TMP_ERR" >&2; return 1; fi
    fi
    if [[ "$want_png" = "1" && "$CHECK_ONLY" = "0" ]]; then
      set +e
      mmdc -i "$src" -o "$png_out" -t "$theme" -b "$bg" "${extra_args[@]}" 2>> "$TMP_ERR"
      rc=$?; set -e
      if [[ $rc -ne 0 ]]; then
        if [[ "$KEEP_GOING" = "1" ]]; then warn "mmdc (png) failed: $src"; cat "$TMP_ERR" >&2; return 0; else
          echo "[render_diagrams][ERROR] mmdc (png) failed: $src" >&2; cat "$TMP_ERR" >&2; return 1; fi
      fi
    fi
  else
    local repo_root rel_in rel_svg rel_png
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    rel_in="${src#$repo_root/}"
    rel_svg="${svg_out#$repo_root/}"
    rel_png="${png_out#$repo_root/}"
    if [[ "$CHECK_ONLY" = "1" ]]; then
      local tmp_svg; tmp_svg="$(mktemp --suffix=.svg 2>/dev/null || mktemp).svg"
      set +e
      docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
        "$DOCKER_IMAGE" mmdc -i "$rel_in" -o "$tmp_svg" -t "$theme" -b "$bg" "${extra_args[@]}" 2> "$TMP_ERR"
      rc=$?
      rm -f "$tmp_svg"; set -e
    else
      set +e
      docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
        "$DOCKER_IMAGE" mmdc -i "$rel_in" -o "$rel_svg" -t "$theme" -b "$bg" "${extra_args[@]}" 2> "$TMP_ERR"
      rc=$?; set -e
    fi
    if [[ $rc -ne 0 ]]; then
      if [[ "$KEEP_GOING" = "1" ]]; then warn "docker mmdc failed: $src"; cat "$TMP_ERR" >&2; return 0; else
        echo "[render_diagrams][ERROR] docker mmdc failed: $src" >&2; cat "$TMP_ERR" >&2; return 1; fi
    fi
    if [[ "$want_png" = "1" && "$CHECK_ONLY" = "0" ]]; then
      set -e
      docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
        "$DOCKER_IMAGE" mmdc -i "$rel_in" -o "$rel_png" -t "$theme" -b "$bg" "${extra_args[@]}" 2>> "$TMP_ERR" || {
          if [[ "$KEEP_GOING" = "1" ]]; then warn "docker mmdc (png) failed: $src"; cat "$TMP_ERR" >&2; return 0; else
            echo "[render_diagrams][ERROR] docker mmdc (png) failed: $src" >&2; cat "$TMP_ERR" >&2; return 1; fi
        }
    fi
  fi

  t1=$(date +%s%3N 2>/dev/null || date +%s)
  dur=$(( t1 - t0 ))
  local out_size="0"
  [[ "$CHECK_ONLY" = "0" && -f "$svg_out" ]] && out_size="$(fsize "$svg_out" 2>/dev/null || echo 0)"

  [[ "$QUIET" = "1" ]] || echo "[render_diagrams] ✔ ${src} → ${svg_out}${want_png:+, ${png_out}}  (${dur}ms)"
  echo "$src|$svg_out|$dur|ok" >> "$TMP_TIMINGS"
  return 0
}

# --- Parallel driver ----------------------------------------------------------
TMP_ERR="$(mktemp)"; trap 'rm -f "$TMP_ERR" "$TMP_TIMINGS"' EXIT
TMP_TIMINGS="$(mktemp)"

run_all() {
  if [[ "$LIST_ONLY" = "1" ]]; then
    for f in "${FILES[@]}"; do render_one "$f"; done; exit 0
  fi
  if xargs --help >/dev/null 2>&1; then
    printf '%s\0' "${FILES[@]}" | xargs -0 -n1 -P "${CONCURRENCY}" bash -lc 'render_one "$0"'
  else
    for f in "${FILES[@]}"; do render_one "$f"; done
  fi
}
log "Rendering with theme='${THEME}', bg='${BGCOL}', png=${RENDER_PNG}, outdir='${OUTDIR:-<src dir>}'${CHECK_ONLY:+ (check mode)}"
[[ ${#MMDC_ADV_ARGS[@]} -gt 0 ]] && log "Advanced mmdc args: ${MMDC_ADV_ARGS[*]}"
run_all

# --- Manifest for CI ----------------------------------------------------------
mkdir -p "$MANIFEST_DIR"

# Collect outputs
OUTS=()
collect_outputs() {
  local arr=()
  for dir in "${SOURCES[@]}"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r -d '' f; do arr+=("$f"); done < <(find "$dir" -type f \( -name '*.svg' -o -name '*.png' \) -print0 2>/dev/null || true)
  done
  if [[ -n "$OUTDIR" && -d "$OUTDIR" ]]; then
    while IFS= read -r -d '' f; do arr+=("$f"); done < <(find "$OUTDIR" -type f \( -name '*.svg' -o -name '*.png' \) -print0 2>/dev/null || true)
  fi
  printf '%s\n' "${arr[@]}"
}
mapfile -t OUTS < <(collect_outputs)

# Build simple timing map
declare -A TIME_MAP
while IFS='|' read -r src out dur status; do
  [[ -n "$src" ]] || continue
  TIME_MAP["$out"]="$dur"
done < "$TMP_TIMINGS"

{
  echo "{"
  echo "  \"generated_at\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\","
  echo "  \"environment\": \"${ENV_TYPE}\","
  echo "  \"check_only\": ${CHECK_ONLY},"
  echo "  \"theme\": \"${THEME}\","
  echo "  \"background\": \"${BGCOL}\","
  echo "  \"png\": ${RENDER_PNG},"
  echo "  \"outdir\": \"${OUTDIR}\","
  echo "  \"since\": \"${SINCE_REF}\","
  echo "  \"sources\": ["
  for i in "${!SOURCES[@]}"; do
    s="${SOURCES[$i]}"; printf "    \"%s\"%s\n" "$s" $([[ "$i" -lt $((${#SOURCES[@]}-1)) ]] && echo "," || true)
  done
  echo "  ],"
  echo "  \"outputs\": ["
  for i in "${!OUTS[@]}"; do
    f="${OUTS[$i]}"; h="$(sha256 "$f")"; sz="$(fsize "$f" 2>/dev/null || echo 0)"; t="${TIME_MAP[$f]:-0}"
    printf "    {\"path\": \"%s\", \"sha256\": \"%s\", \"size\": %s, \"ms\": %s}%s\n" "$(printf "%s" "$f")" "$h" "$sz" "$t" $([[ "$i" -lt $((${#OUTS[@]}-1)) ]] && echo "," || true)
  done
  echo "  ]"
  echo "}"
} > "$MANIFEST_PATH"

log "Manifest -> $MANIFEST_PATH"
log "Diagram pass complete ✅"
