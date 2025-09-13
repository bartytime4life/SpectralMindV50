#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — render_mermaid.sh
# Render Mermaid .mmd diagrams to SVG (and optional PNG), with a pre-lint pass.
# • Repo-root aware; Kaggle/CI aware
# • Lint warns about GitHub-incompatible patterns (e.g., '|' in labels)
# • Deterministic outputs; dry-run; compact JSON summary for CI
# • Uses Mermaid CLI (mmdc). Optional fallback to npx mmdc if available.
# ------------------------------------------------------------------------------
# Usage:
#   bin/render_mermaid.sh [--src assets/diagrams] [--dst docs/assets/diagrams]
#                         [--formats svg,png] [--bg transparent|white]
#                         [--theme default|dark|forest|neutral]
#                         [--scale 1..4] [--lint] [--strict]
#                         [--json] [--quiet] [--dry-run]
#
# Examples:
#   # Default: assets/diagrams/*.mmd → docs/assets/diagrams/*.svg
#   bin/render_mermaid.sh --lint
#
#   # Also produce PNG, white background, 2x scale
#   bin/render_mermaid.sh --formats svg,png --bg white --scale 2
#
# Exit codes:
#   0 = success
#   2 = bad args / missing tools
#   3 = lint errors with --strict OR rendering failures
# ==============================================================================

set -Eeuo pipefail

# ---------- logging ------------------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

usage() {
  sed -n '1,200p' "${BASH_SOURCE[0]}" | awk '/^# ====/{flag=1;next}/^set -Eeuo/{flag=0}flag' | sed 's/^# \{0,1\}//'
}

trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"; exit 3' ERR

# ---------- args ---------------------------------------------------------------
SRC_DIR="assets/diagrams"
DST_DIR="docs/assets/diagrams"
FORMATS="svg"
BG="transparent"         # transparent | white | #RRGGBB
THEME="default"          # default|dark|forest|neutral
SCALE="1"                # integer 1..4 (mmdc scale factor for raster)
DO_LINT=0
STRICT=0
EMIT_JSON=0
QUIET="${QUIET:-0}"
DRYRUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)     SRC_DIR="${2:-}"; shift 2 ;;
    --dst)     DST_DIR="${2:-}"; shift 2 ;;
    --formats) FORMATS="${2:-}"; shift 2 ;;
    --bg)      BG="${2:-}"; shift 2 ;;
    --theme)   THEME="${2:-}"; shift 2 ;;
    --scale)   SCALE="${2:-}"; shift 2 ;;
    --lint)    DO_LINT=1; shift ;;
    --strict)  STRICT=1; shift ;;
    --json)    EMIT_JSON=1; shift ;;
    --quiet)   QUIET=1; shift ;;
    --dry-run) DRYRUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ---------- env awareness ------------------------------------------------------
IS_KAGGLE=0
[[ -d "/kaggle" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] && IS_KAGGLE=1
IS_CI=0
[[ "${CI:-}" == "true" || "${GITHUB_ACTIONS:-}" == "true" ]] && IS_CI=1

# ---------- repo root detection ------------------------------------------------
repo_root() {
  if command -v git >/dev/null 2>&1; then
    if r="$(git rev-parse --show-toplevel 2>/dev/null || true)"; then
      [[ -n "$r" ]] && { printf "%s" "$r"; return; }
    fi
  fi
  local d="$PWD"
  while [[ "$d" != "/" ]]; do
    if [[ -e "$d/pyproject.toml" || -e "$d/dvc.yaml" || -d "$d/.git" ]]; then
      printf "%s" "$d"; return
    fi
    d="$(dirname "$d")"
  done
  printf "%s" "$PWD"
}
ROOT="$(repo_root)"
cd "$ROOT"

# ---------- tool resolution ----------------------------------------------------
need() { command -v "$1" >/dev/null 2>&1; }

MMDC_CMD=""
if need mmdc; then
  MMDC_CMD="mmdc"
elif need npx; then
  # Fallback to npx mmdc if available locally
  if npx --yes --quiet @mermaid-js/mermaid-cli -V >/dev/null 2>&1; then
    MMDC_CMD="npx @mermaid-js/mermaid-cli"
  fi
fi

if [[ -z "$MMDC_CMD" ]]; then
  warn "Mermaid CLI (mmdc) not found. Install via: npm i -g @mermaid-js/mermaid-cli"
  # We still continue if only linting was requested
  if [[ "$DO_LINT" -eq 0 ]]; then
    err "Cannot render without mmdc"; exit 2
  fi
fi

# ---------- formats parsing ----------------------------------------------------
IFS=',' read -r -a FMT_ARR <<< "$FORMATS"
declare -A WANT=()
for f in "${FMT_ARR[@]}"; do
  case "$f" in
    svg|png) WANT["$f"]=1 ;;
    *) warn "Unknown format '$f' ignored";;
  esac
done
[[ ${#WANT[@]} -eq 0 ]] && WANT["svg"]=1

# ---------- input discovery ----------------------------------------------------
if [[ ! -d "$SRC_DIR" ]]; then
  warn "Source dir not found: $SRC_DIR"
  [[ "$STRICT" -eq 1 ]] && exit 3 || exit 0
fi

mapfile -t MMD_FILES < <(find "$SRC_DIR" -maxdepth 1 -type f -name "*.mmd" | LC_ALL=C sort || true)
if (( ${#MMD_FILES[@]} == 0 )); then
  warn "No .mmd files found in $SRC_DIR"
  [[ "$STRICT" -eq 1 ]] && exit 3 || exit 0
fi

# Ensure destination directory
[[ "$DRYRUN" -eq 1 ]] || mkdir -p "$DST_DIR"

# ---------- Lint pass (GitHub-compatible heuristics) ---------------------------
# We flag common breakages:
#  • Vertical bar '|' inside node labels → Mermaid parser 'PIPE' errors on GitHub
#  • Mixed fencing / non-mermaid content (naive check)
#  • Recommend <br/> over raw newlines in labels
lint_errors=0
declare -a LINT_WARNINGS=()

lint_file() {
  local f="$1"
  local base; base="$(basename "$f")"
  # Heuristic: any '|' that is not part of class definitions or comments
  # We exempt lines starting with '%%' (comment) and classDef/links
  if grep -nE '^[[:space:]]*[^%].*\|.*' "$f" >/dev/null 2>&1; then
    LINT_WARNINGS+=("$base: '|' found in label text — replace with '/', '·', or line break <br/> for GitHub Mermaid.")
    ((lint_errors++))
  fi
  # Recommend <br/> instead of literal HTML <br> (GitHub prefers <br/>)
  if grep -nE '<br[^/]*>' "$f" >/dev/null 2>&1; then
    LINT_WARNINGS+=("$base: use '<br/>' not '<br>' for GitHub renderer consistency.")
  fi
  # Encourage flowchart keyword
  if grep -qiE '^\s*graph\s' "$f"; then
    LINT_WARNINGS+=("$base: prefer 'flowchart TD|LR' over 'graph' for clarity.")
  fi
}

if (( DO_LINT )); then
  for m in "${MMD_FILES[@]}"; do lint_file "$m"; done
  for w in "${LINT_WARNINGS[@]}"; do warn "$w"; done
  if (( lint_errors > 0 )) && (( STRICT )); then
    err "Lint found $lint_errors error(s) with --strict"
    [[ "$EMIT_JSON" -eq 1 ]] && printf '{"ok":false,"reason":"lint","errors":%s}\n' "$lint_errors"
    exit 3
  fi
fi

# ---------- Rendering ----------------------------------------------------------
render_count=0
fail_count=0
declare -a RENDERED=()
declare -a FAILED=()

render_one() {
  local in="$1" base out_svg out_png
  base="$(basename "${in%.*}")"
  out_svg="$DST_DIR/${base}.svg"
  out_png="$DST_DIR/${base}.png"

  # SVG (vector) — preferred for GitHub/MkDocs
  if [[ -n "${WANT[svg]:-}" ]]; then
    if [[ "$DRYRUN" -eq 1 ]]; then
      log "[dry-run] $MMDC_CMD -i $in -o $out_svg --backgroundColor $BG --theme $THEME"
    else
      if $MMDC_CMD -i "$in" -o "$out_svg" --backgroundColor "$BG" --theme "$THEME"; then
        RENDERED+=("$out_svg"); ((render_count++))
      else
        FAILED+=("$in → $out_svg"); ((fail_count++))
      fi
    fi
  fi

  # PNG (raster) — optional for places that need bitmaps
  if [[ -n "${WANT[png]:-}" ]]; then
    if [[ "$DRYRUN" -eq 1 ]]; then
      log "[dry-run] $MMDC_CMD -i $in -o $out_png --backgroundColor $BG --theme $THEME --scale $SCALE"
    else
      if $MMDC_CMD -i "$in" -o "$out_png" --backgroundColor "$BG" --theme "$THEME" --scale "$SCALE"; then
        RENDERED+=("$out_png"); ((render_count++))
      else
        FAILED+=("$in → $out_png"); ((fail_count++))
      fi
    fi
  fi
}

log "Rendering ${#MMD_FILES[@]} Mermaid file(s) → $DST_DIR (formats: ${!WANT[*]})"
for m in "${MMD_FILES[@]}"; do render_one "$m"; done

if (( fail_count == 0 )); then
  ok "Rendered $render_count artifact(s)"
else
  err "Rendered $render_count artifact(s); $fail_count failed"
  (( STRICT )) && exit 3
fi

# ---------- JSON summary -------------------------------------------------------
if (( EMIT_JSON )); then
  esc() { printf "%s" "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  printf '{'
  printf '"ok":%s,' $(( fail_count==0 ? 1 : 0 ))
  printf '"kaggle":%s,"ci":%s,' "$IS_KAGGLE" "$IS_CI"
  printf '"rendered_count":%s,' "$render_count"
  printf '"failed_count":%s,' "$fail_count"
  printf '"rendered":['
    for i in "${!RENDERED[@]}"; do [[ $i -gt 0 ]] && printf ','; printf '"%s"' "$(esc "${RENDERED[$i]}")"; done
  printf '],'
  printf '"failed":['
    for i in "${!FAILED[@]}"; do [[ $i -gt 0 ]] && printf ','; printf '"%s"' "$(esc "${FAILED[$i]}")"; done
  printf '],'
  printf '"lint_errors":%s,' "$lint_errors"
  printf '"dst":"%s","bg":"%s","theme":"%s","formats":"%s","scale":%s' \
    "$(esc "$DST_DIR")" "$(esc "$BG")" "$(esc "$THEME")" "$(esc "$FORMATS")" "$SCALE"
  printf '}\n'
fi

ok "Mermaid render pipeline finished."
exit $(( fail_count>0 && STRICT ? 3 : 0 ))
