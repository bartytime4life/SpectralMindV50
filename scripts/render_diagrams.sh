#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Mermaid Diagram Renderer (Ultra Upgraded, Deterministic)
# -----------------------------------------------------------------------------
# Renders Mermaid (*.mmd, *.mermaid) → SVG (and optionally PNG).
# Prefers local mermaid-cli (mmdc); falls back to Docker (disabled on Kaggle).
# Incremental by default:
#   • content-hash cache (source + options)   ← primary
#   • mtime fallback                           ← secondary
#
# Default sources:
#   - assets/diagrams/
#   - docs/diagrams/
#
# Usage:
#   ./scripts/render_diagrams.sh [options]
#
# Options:
#   -o, --outdir DIR         Output directory (default: alongside source)
#   -s, --source DIR         Additional source directory (repeatable)
#   -t, --theme THEME        [default|dark|forest|neutral] (default: default)
#   -b, --background HEX     Background (transparent|#RRGGBB), default transparent
#   -p, --png                Also produce PNG next to SVG
#   -c, --concurrency N      Parallel renders (default: auto cores, min 1)
#   -f, --force              Force rebuild (ignore caches/mtimes)
#   -l, --list               List diagrams that would be rendered and exit
#   -k, --keep-going         Keep going on errors (default: fail fast)
#   -q, --quiet              Less verbose output
#       --since REF          Only files changed since git ref/branch/tag
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
# Per-file overrides (front-matter or header hints):
#   YAML front-matter (PyYAML optional; ignored if unavailable)
#       ---
#       theme: dark
#       background: "#0b1021"
#       width: 1600
#       height: 900
#       png: true
#       ---
#   // @theme: dark
#
# Emits manifest JSON at artifacts/diagrams_manifest.json for CI diffing.
# -----------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- Defaults ----------
THEME="default"
BGCOL="transparent"
RENDER_PNG="0"
OUTDIR=""
CONCURRENCY=""
QUIET="0"
FORCE="0"
LIST_ONLY="0"
KEEP_GOING="0"
CHECK_ONLY="0"
SINCE_REF=""

MD_WIDTH=""
MD_HEIGHT=""
MD_THEME_VARS=""
MD_THEME_CSS=""
MD_CONFIG_FILE=""
MD_PPTR_FILE=""

SOURCES=("assets/diagrams" "docs/diagrams")
MANIFEST_DIR="artifacts"
MANIFEST_PATH="$MANIFEST_DIR/diagrams_manifest.json"
CACHE_DIR=".cache/mermaid"       # stores content-hash files per output
DOCKER_IMAGE="ghcr.io/mermaid-js/mermaid-cli:latest"

# ---------- Helpers ----------
timestamp(){ date +"%Y-%m-%d %H:%M:%S"; }
log()     { [[ "$QUIET" = "1" ]] || printf "[ %s ] [render_diagrams] %s\n" "$(timestamp)" "$*"; }
warn()    { printf "[ %s ] [render_diagrams][WARN] %s\n"  "$(timestamp)" "$*" >&2; }
die()     { printf "[ %s ] [render_diagrams][ERROR] %s\n" "$(timestamp)" "$*" >&2; exit 1; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }
is_kaggle(){ [[ -d "/kaggle/input" ]]; }

# BSD/GNU stat wrappers
mtime(){ stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"; }
fsize(){ stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

sha256() {
  if has_cmd sha256sum; then sha256sum "$1" | awk '{print $1}';
  elif has_cmd shasum; then shasum -a 256 "$1" | awk '{print $1}';
  else printf "unavailable"; fi
}

# Detect cores for default concurrency
detect_cores() {
  if has_cmd nproc; then nproc;
  elif [[ "$OSTYPE" == "darwin"* ]]; then sysctl -n hw.ncpu;
  else printf "1"; fi
}

usage(){ sed -n '1,220p' "$0" | sed 's/^# \{0,1\}//'; }
trap 'echo "[render_diagrams][ERROR] failed at line $LINENO: $BASH_COMMAND" >&2' ERR

# ---------- Parse args ----------
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
[[ -n "$CONCURRENCY" ]] || CONCURRENCY="$(detect_cores)"; [[ "$CONCURRENCY" -ge 1 ]] || CONCURRENCY=1

# ---------- Choose renderer ----------
MMDC_MODE=""
if has_cmd mmdc; then
  MMDC_MODE="local"; log "Using mermaid-cli (mmdc)"
elif has_cmd docker && ! is_kaggle; then
  MMDC_MODE="docker"; log "Using Docker mermaid-cli image: $DOCKER_IMAGE"
else
  die "No mermaid renderer found. Install 'mmdc' (npm i -g @mermaid-js/mermaid-cli) or Docker (Docker disabled on Kaggle)."
fi

# ---------- Build mmdc args ----------
MMDC_ADV_ARGS=()
[[ -n "$MD_WIDTH"       ]] && MMDC_ADV_ARGS+=(--width "$MD_WIDTH")
[[ -n "$MD_HEIGHT"      ]] && MMDC_ADV_ARGS+=(--height "$MD_HEIGHT")
[[ -n "$MD_THEME_VARS"  ]] && MMDC_ADV_ARGS+=(--themeVariables "$MD_THEME_VARS")
[[ -n "$MD_THEME_CSS"   ]] && MMDC_ADV_ARGS+=(--themeCSS "$MD_THEME_CSS")
[[ -n "$MD_CONFIG_FILE" ]] && MMDC_ADV_ARGS+=(--configFile "$MD_CONFIG_FILE")
[[ -n "$MD_PPTR_FILE"   ]] && MMDC_ADV_ARGS+=(--puppeteerConfigFile "$MD_PPTR_FILE")

# ---------- Collect sources ----------
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
    warn "--since given but no git repo found; ignoring"; cat; return
  fi
  while IFS= read -r f; do
    if git diff --name-only "$SINCE_REF"... -- "$f" | grep -q .; then echo "$f"; fi
  done
}
mapfile -t FILES < <(collect_files | apply_since_filter || true)
[[ ${#FILES[@]} -gt 0 ]] || { warn "No Mermaid files found in: ${SOURCES[*]}"; exit 0; }

# ---------- Cache + incremental ----------
mkdir -p "$CACHE_DIR"
needs_rebuild() {
  local src="$1" svg="$2" png="$3" theme="$4" bg="$5" width="$6" height="$7" adv="$8" want_png="$9"
  [[ "$FORCE" = "1" ]] && return 0
  [[ "$CHECK_ONLY" = "1" ]] && return 0

  # content-hash cache: src bytes + options
  local sigfile="${svg}.sha256"
  local curhash
  curhash="$( (cat "$src"; printf "\n---\n%s|%s|%s|%s|%s|%s\n" "$theme" "$bg" "$width" "$height" "$adv" "$want_png") | sha256sum 2>/dev/null | awk '{print $1}' )" \
    || curhash="$( (cat "$src"; printf "\n---\n%s|%s|%s|%s|%s|%s\n" "$theme" "$bg" "$width" "$height" "$adv" "$want_png") | shasum -a 256 | awk '{print $1}' )"
  [[ -z "$curhash" ]] && return 0

  if [[ -f "$svg" && -f "$sigfile" ]] && grep -qx "$curhash" "$sigfile"; then
    # fallback to mtime check in case sig drifts
    local sm tm; sm="$(mtime "$src")"; tm="$(mtime "$svg")"
    [[ "$sm" -le "$tm" ]] && return 1
  fi
  return 0
}
write_sig() {
  local svg="$1" curhash="$2"
  [[ -n "$curhash" ]] || return 0
  printf "%s\n" "$curhash" > "${svg}.sha256" || true
}

# ---------- Per-file overrides (PyYAML optional) ----------
# Extract front-matter and // @key: value from first 12 lines.
read_overrides() {
  local src="$1"
  python - "$src" <<'PY'
import sys,re
p=sys.argv[1]
head=""
try:
  with open(p,'r',encoding='utf-8',errors='ignore') as f:
    for _ in range(12): head+=next(f)
except Exception: pass

cfg={}
# YAML block
if head.startswith('---'):
  try:
    end=head.find("\n---",3)
    if end!=-1:
      block=head[3:end]
      try:
        import yaml
        y=yaml.safe_load(block) or {}
        if isinstance(y,dict): cfg.update({str(k).lower():str(v) for k,v in y.items()})
      except Exception:
        # graceful: ignore if PyYAML absent or invalid
        pass
  except Exception:
    pass
# Header hints
for m in re.finditer(r'^\s*//\s*@([a-zA-Z0-9_\-]+)\s*:\s*(.+)\s*$', head, re.M):
  cfg[m.group(1).strip().lower()]=m.group(2).strip()

def out(k): print(cfg.get(k,""))
for k in ("theme","background","width","height","png"):
  out(k)
PY
}

# ---------- Render driver ----------
TMP_ERR="$(mktemp)"; TMP_TIMINGS="$(mktemp)"
trap 'rm -f "$TMP_ERR" "$TMP_TIMINGS"' EXIT

render_one() {
  local src="$1"
  local base stem dir svg_out png_out
  base="$(basename "$src")"; stem="${base%.*}"; dir="$(dirname "$src")"

  if [[ -n "$OUTDIR" ]]; then
    mkdir -p "$OUTDIR"; svg_out="$OUTDIR/${stem}.svg"; png_out="$OUTDIR/${stem}.png"
  else
    svg_out="${dir}/${stem}.svg"; png_out="${dir}/${stem}.png"
  fi

  if [[ "$LIST_ONLY" = "1" ]]; then echo "$src"; return 0; fi

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

  # cache signature
  local adv_sig; adv_sig="$(printf "%s" "${extra_args[*]}")"
  local curhash; curhash="$( (cat "$src"; printf "\n---\n%s|%s|%s|%s|%s|%s\n" "$theme" "$bg" "${ov_w:-$MD_WIDTH}" "${ov_h:-$MD_HEIGHT}" "$adv_sig" "$want_png") | \
      (sha256sum 2>/dev/null || shasum -a 256) | awk '{print $1}' )" || curhash=""

  if needs_rebuild "$src" "$svg_out" "$png_out" "$theme" "$bg" "${ov_w:-$MD_WIDTH}" "${ov_h:-$MD_HEIGHT}" "$adv_sig" "$want_png"; then
    [[ "$CHECK_ONLY" = "1" ]] && log "✓ Check: $src" || log "⇢ Render: $src"
  else
    log "↻ Skip (cached): $src"
    echo "$src|$svg_out|0|skip" >> "$TMP_TIMINGS"
    return 0
  fi

  local t0 t1 dur rc
  t0=$(date +%s%3N 2>/dev/null || date +%s)

  if [[ "$MMDC_MODE" = "local" ]]; then
    if [[ "$CHECK_ONLY" = "1" ]]; then
      local tmp_svg; tmp_svg="$(mktemp).svg"
      set +e; mmdc -i "$src" -o "$tmp_svg" -t "$theme" -b "$bg" "${extra_args[@]}" 2> "$TMP_ERR"; rc=$?; set -e
      rm -f "$tmp_svg"
    else
      set +e; mmdc -i "$src" -o "$svg_out" -t "$theme" -b "$bg" "${extra_args[@]}" 2> "$TMP_ERR"; rc=$?; set -e
    fi
    if [[ $rc -ne 0 ]]; then
      if [[ "$KEEP_GOING" = "1" ]]; then warn "mmdc failed: $src"; cat "$TMP_ERR" >&2; return 0; else
        echo "[render_diagrams][ERROR] mmdc failed: $src" >&2; cat "$TMP_ERR" >&2; return 1; fi
    fi
    if [[ "$want_png" = "1" && "$CHECK_ONLY" = "0" ]]; then
      set +e; mmdc -i "$src" -o "$png_out" -t "$theme" -b "$bg" "${extra_args[@]}" 2>> "$TMP_ERR"; rc=$?; set -e
      if [[ $rc -ne 0 ]]; then
        if [[ "$KEEP_GOING" = "1" ]]; then warn "mmdc (png) failed: $src"; cat "$TMP_ERR" >&2; return 0; else
          echo "[render_diagrams][ERROR] mmdc (png) failed: $src" >&2; cat "$TMP_ERR" >&2; return 1; fi
      fi
    fi
  else # docker
    local repo_root rel_in rel_svg rel_png
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    rel_in="${src#$repo_root/}"
    rel_svg="${svg_out#$repo_root/}"
    rel_png="${png_out#$repo_root/}"
    if [[ "$CHECK_ONLY" = "1" ]]; then
      local tmp_svg; tmp_svg="$(mktemp --suffix=.svg 2>/dev/null || mktemp).svg"
      set +e
      docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
        "$DOCKER_IMAGE" mmdc -i "$rel_in" -o "$tmp_svg" -t "$theme" -b "$bg" "${extra_args[@]}" 2> "$TMP_ERR"; rc=$?
      set -e; rm -f "$tmp_svg"
    else
      set +e
      docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
        "$DOCKER_IMAGE" mmdc -i "$rel_in" -o "$rel_svg" -t "$theme" -b "$bg" "${extra_args[@]}" 2> "$TMP_ERR"; rc=$?
      set -e
    fi
    if [[ $rc -ne 0 ]]; then
      if [[ "$KEEP_GOING" = "1" ]]; then warn "docker mmdc failed: $src"; cat "$TMP_ERR" >&2; return 0; else
        echo "[render_diagrams][ERROR] docker mmdc failed: $src" >&2; cat "$TMP_ERR" >&2; return 1; fi
    fi
    if [[ "$want_png" = "1" && "$CHECK_ONLY" = "0" ]]; then
      set +e
      docker run --rm -u "$(id -u):$(id -g)" -v "$repo_root":/work -w /work \
        "$DOCKER_IMAGE" mmdc -i "$rel_in" -o "$rel_png" -t "$theme" -b "$bg" "${extra_args[@]}" 2>> "$TMP_ERR"; rc=$?
      set -e
      if [[ $rc -ne 0 ]]; then
        if [[ "$KEEP_GOING" = "1" ]]; then warn "docker mmdc (png) failed: $src"; cat "$TMP_ERR" >&2; return 0; else
          echo "[render_diagrams][ERROR] docker mmdc (png) failed: $src" >&2; cat "$TMP_ERR" >&2; return 1; fi
      fi
    fi
  fi

  t1=$(date +%s%3N 2>/dev/null || date +%s); dur=$(( t1 - t0 ))
  [[ "$CHECK_ONLY" = "0" && -n "$curhash" && -f "$svg_out" ]] && write_sig "$svg_out" "$curhash"
  echo "$src|$svg_out|$dur|ok" >> "$TMP_TIMINGS"
  [[ "$QUIET" = "1" ]] || echo "[render_diagrams] ✔ ${src} → ${svg_out}${want_png:+, ${png_out}}  (${dur}ms)"
  return 0
}

# ---------- Execute (parallel if possible) ----------
log "Found ${#FILES[@]} Mermaid file(s) (theme='${THEME}', bg='${BGCOL}', png=${RENDER_PNG}, outdir='${OUTDIR:-<src dir>}')${CHECK_ONLY:+ [check-only]}"
[[ ${#MMDC_ADV_ARGS[@]} -gt 0 ]] && log "mmdc args: ${MMDC_ADV_ARGS[*]}"
mkdir -p "$MANIFEST_DIR" "$CACHE_DIR"

if [[ "$LIST_ONLY" = "1" ]]; then
  printf "%s\n" "${FILES[@]}"
  exit 0
fi

# export function for xargs
export -f render_one read_overrides has_cmd mtime fsize sha256
export THEME BGCOL RENDER_PNG OUTDIR MMDC_MODE DOCKER_IMAGE CHECK_ONLY KEEP_GOING QUIET
export MD_WIDTH MD_HEIGHT
# shellcheck disable=SC2046
if xargs --help >/dev/null 2>&1; then
  printf '%s\0' "${FILES[@]}" | xargs -0 -n1 -P "$CONCURRENCY" bash -lc 'render_one "$0"'
else
  for f in "${FILES[@]}"; do render_one "$f"; done
fi

# ---------- Manifest ----------
# Gather outputs from both source dirs and OUTDIR (if set)
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

declare -A TIME_MAP
while IFS='|' read -r src out dur status; do
  [[ -n "$out" ]] && TIME_MAP["$out"]="$dur"
done < "$TMP_TIMINGS"

# write manifest via Python for safe JSON escaping
python - "$MANIFEST_PATH" <<PY || { warn "Failed to write manifest"; exit 0; }
import json, os, sys, hashlib, time
out=sys.argv[1]
def sha256(p):
  try:
    h=hashlib.sha256()
    with open(p,'rb') as f:
      for chunk in iter(lambda: f.read(1<<20), b''):
        h.update(chunk)
    return h.hexdigest()
  except Exception:
    return "unavailable"
def size(p):
  try: return os.path.getsize(p)
  except Exception: return 0
# receive OUTS/TIME_MAP from env via a temp file? no, re-scan in shell and pass via stdin is complex,
# but we already have OUTS on disk; just rebuild minimal info:
manifest = {
  "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "environment": "kaggle" if os.path.isdir("/kaggle/input") else "local",
  "check_only": ${CHECK_ONLY},
  "theme": "${THEME}",
  "background": "${BGCOL}",
  "png": ${RENDER_PNG},
  "outdir": "${OUTDIR}",
  "since": "${SINCE_REF}",
  "sources": ${SOURCES!r},
  "outputs": []
}
# pull file list from a temp text file created by shell:
outs = ${OUTS!r}
time_map = {${", ".join(f'"{k}": {v or 0}' for k,v in TIME_MAP.items())}}
for p in outs:
  manifest["outputs"].append({
    "path": p,
    "sha256": sha256(p),
    "size": size(p),
    "ms": time_map.get(p, 0)
  })
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,"w") as f: json.dump(manifest, f, indent=2)
print(out)
PY

log "Manifest -> $MANIFEST_PATH"
log "Diagram pass complete ✅"