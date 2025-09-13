#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — render_docs.sh
# Render Mermaid diagrams and build/serve MkDocs documentation.
# • Repo-root aware; Kaggle/CI aware; dry-run + JSON summary
# • Mermaid: assets/diagrams/*.mmd → docs/assets/diagrams/*.svg[.png]
# • Deterministic: stable output paths; optional clean site dir
# • MkDocs: build (--strict) or serve (-a host:port)
# ------------------------------------------------------------------------------
# Usage:
#   bin/render_docs.sh [--build] [--serve] [--strict] [--clean]
#                      [--formats svg,png] [--site-dir site]
#                      [--host 127.0.0.1] [--port 8000]
#                      [--open] [--json] [--quiet] [--dry-run]
#
# Examples:
#   # Render diagrams to SVG and build site strictly
#   bin/render_docs.sh --build --strict
#
#   # Render diagrams (SVG+PNG), then serve docs locally and open browser
#   bin/render_docs.sh --serve --formats svg,png --open
#
# Exit codes:
#   0 = success
#   2 = bad args / missing tools
#   3 = operation failed (render/build/serve)
# ==============================================================================

set -Eeuo pipefail

# ---------- pretty printing ----------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

usage() {
  sed -n '1,200p' "${BASH_SOURCE[0]}" | awk '/^# ====/{flag=1;next}/^set -Eeuo/{flag=0}flag' | sed 's/^# \{0,1\}//'
}

trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"; exit 3' ERR

# ---------- args ---------------------------------------------------------------
DO_BUILD=0
DO_SERVE=0
STRICT=0
CLEAN=0
FORMATS="svg"
SITE_DIR="site"
HOST="127.0.0.1"
PORT="8000"
OPEN=0
EMIT_JSON=0
QUIET="${QUIET:-0}"
DRYRUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)    DO_BUILD=1; shift ;;
    --serve)    DO_SERVE=1; shift ;;
    --strict)   STRICT=1; shift ;;
    --clean)    CLEAN=1; shift ;;
    --formats)  FORMATS="${2:-}"; shift 2 ;;
    --site-dir) SITE_DIR="${2:-}"; shift 2 ;;
    --host)     HOST="${2:-}"; shift 2 ;;
    --port)     PORT="${2:-}"; shift 2 ;;
    --open)     OPEN=1; shift ;;
    --json)     EMIT_JSON=1; shift ;;
    --quiet)    QUIET=1; shift ;;
    --dry-run)  DRYRUN=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# Can't both build+serve and cleanly exit; serving is long-running
if (( DO_BUILD == 1 && DO_SERVE == 1 )); then
  err "Choose either --build or --serve (not both)"; exit 2
fi

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

# ---------- tool checks --------------------------------------------------------
need() { command -v "$1" >/dev/null 2>&1; }

if ! need mmdc; then
  warn "Mermaid CLI (mmdc) not found — diagram rendering will be skipped"
fi
if ! need mkdocs; then
  if (( DO_BUILD==1 || DO_SERVE==1 )); then
    err "mkdocs is required for --build/--serve"; exit 2
  else
    warn "mkdocs not found — skipping doc site steps"
  fi
fi

# ---------- paths --------------------------------------------------------------
SRC_DIR="assets/diagrams"
DST_DIR="docs/assets/diagrams"
[[ -d "$DST_DIR" ]] || { [[ "$DRYRUN" -eq 1 ]] || mkdir -p "$DST_DIR"; }

# ---------- parse formats ------------------------------------------------------
IFS=',' read -r -a FMT_ARR <<< "$FORMATS"
declare -A WANT=()
for f in "${FMT_ARR[@]}"; do
  case "$f" in
    svg|png) WANT["$f"]=1 ;;
    *) warn "Unknown format '$f' ignored";;
  esac
done
[[ ${#WANT[@]} -eq 0 ]] && WANT["svg"]=1

# ---------- render Mermaid diagrams -------------------------------------------
render_count=0
declare -a RENDERED=()
declare -a SKIPPED=()

render_one() {
  local in="$1" base out_svg out_png
  base="$(basename "${in%.*}")"
  out_svg="$DST_DIR/${base}.svg"
  out_png="$DST_DIR/${base}.png"

  if ! need mmdc; then
    SKIPPED+=("$in")
    return 0
  fi

  # Render SVG (preferred for GitHub & MkDocs)
  if [[ -n "${WANT[svg]:-}" ]]; then
    if [[ "$DRYRUN" -eq 1 ]]; then
      log "[dry-run] mmdc -i $in -o $out_svg"
    else
      mmdc -i "$in" -o "$out_svg" --backgroundColor transparent
      RENDERED+=("$out_svg"); ((render_count++))
    fi
  fi
  # Optional PNG
  if [[ -n "${WANT[png]:-}" ]]; then
    if [[ "$DRYRUN" -eq 1 ]]; then
      log "[dry-run] mmdc -i $in -o $out_png"
    else
      mmdc -i "$in" -o "$out_png" --backgroundColor transparent
      RENDERED+=("$out_png"); ((render_count++))
    fi
  fi
}

if [[ -d "$SRC_DIR" ]]; then
  mapfile -t MMD_FILES < <(find "$SRC_DIR" -maxdepth 1 -type f -name "*.mmd" | LC_ALL=C sort || true)
  if (( ${#MMD_FILES[@]} == 0 )); then
    warn "No .mmd files found in $SRC_DIR"
  else
    log "Rendering ${#MMD_FILES[@]} Mermaid file(s) → $DST_DIR (formats: ${!WANT[*]})"
    for m in "${MMD_FILES[@]}"; do render_one "$m"; done
    ok "Rendered $render_count artifact(s)"
  fi
else
  warn "Diagram source dir not found: $SRC_DIR (skipping Mermaid rendering)"
fi

# ---------- mkdocs build/serve -------------------------------------------------
SITE_ARGS=()
[[ "$STRICT" -eq 1 ]] && SITE_ARGS+=( "--strict" )
SITE_ARGS+=( "--site-dir" "$SITE_DIR" )

if (( DO_BUILD == 1 )); then
  if need mkdocs; then
    if [[ "$CLEAN" -eq 1 && "$DRYRUN" -eq 0 ]]; then
      log "Cleaning site dir: $SITE_DIR"
      rm -rf "$SITE_DIR"
    fi
    if [[ "$DRYRUN" -eq 1 ]]; then
      log "[dry-run] mkdocs build ${SITE_ARGS[*]}"
    else
      mkdocs build "${SITE_ARGS[@]}"
      ok "MkDocs build complete → $SITE_DIR"
    fi
  fi
fi

if (( DO_SERVE == 1 )); then
  if [[ "$IS_KAGGLE" -eq 1 ]]; then
    err "Serving docs is not supported on Kaggle environment"; exit 2
  fi
  if need mkdocs; then
    addr="${HOST}:${PORT}"
    if [[ "$DRYRUN" -eq 1 ]]; then
      log "[dry-run] mkdocs serve -a $addr ${SITE_ARGS[*]}"
    else
      # Optionally open browser
      if (( OPEN == 1 )); then
        url="http://${addr}"
        if command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" || true
        elif command -v open >/dev/null 2>&1; then open "$url" || true
        fi
      fi
      log "Serving docs at http://$addr (Ctrl+C to stop)"
      mkdocs serve -a "$addr" "${SITE_ARGS[@]}"
      exit 0
    fi
  fi
fi

# ---------- JSON summary -------------------------------------------------------
if (( EMIT_JSON )); then
  esc() { printf "%s" "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  printf '{'
  printf '"ok":true,'
  printf '"kaggle":%s,"ci":%s,' "$IS_KAGGLE" "$IS_CI"
  printf '"rendered_count":%s,' "$render_count"
  printf '"rendered":['
    for i in "${!RENDERED[@]}"; do [[ $i -gt 0 ]] && printf ','; printf '"%s"' "$(esc "${RENDERED[$i]}")"; done
  printf '],'
  printf '"skipped":['
    for i in "${!SKIPPED[@]}"; do [[ $i -gt 0 ]] && printf ','; printf '"%s"' "$(esc "${SKIPPED[$i]}")"; done
  printf '],'
  printf '"built":%s,' "$DO_BUILD"
  printf '"served":%s,' "$DO_SERVE"
  printf '"strict":%s,' "$STRICT"
  printf '"site_dir":"%s"' "$(esc "$SITE_DIR")"
  printf '}\n'
fi

ok "Docs pipeline finished."
exit 0

