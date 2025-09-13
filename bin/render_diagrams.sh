#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Mermaid Diagram Renderer
# Renders all .mmd files under assets/diagrams/ into .svg (incremental by default)
#
# Features:
#   • Incremental rebuild (only if input is newer than output)
#   • Force rebuild (--all), dry-run (--dry-run), watch mode (--watch) if `entr` exists
#   • Parallelism (--jobs auto|N), theme selection (--theme), custom in/out dirs
#   • Mermaid CLI discovery: mmdc > npx @mermaid-js/mermaid-cli > error
#   • CI/Kaggle safe: no network required unless using `npx` (node/npm must exist)
#
# Usage:
#   bin/render_diagrams.sh [--all] [--dry-run] [--watch] [--jobs N|auto]
#                          [--in assets/diagrams] [--out assets/diagrams]
#                          [--theme default|neutral|dark|forest|base]
#
# Examples:
#   bin/render_diagrams.sh
#   bin/render_diagrams.sh --all --jobs auto --theme neutral
#   bin/render_diagrams.sh --watch
#
# Notes:
#   • Default dirs mirror the repo scaffold (assets/diagrams):contentReference[oaicite:2]{index=2}.
#   • Mermaid syntax rendered like GitHub’s, but locally via CLI:contentReference[oaicite:3]{index=3}.
# ==============================================================================

set -Eeuo pipefail

# ------------- styling -------------
if [[ -t 1 ]]; then
  C_RESET="\033[0m"; C_DIM="\033[2m"; C_GREEN="\033[32m"; C_YELLOW="\033[33m"; C_RED="\033[31m"; C_BLUE="\033[34m"
else
  C_RESET=""; C_DIM=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""
fi

log()   { printf "%b%s%b\n"   "${C_DIM}"   "$*" "${C_RESET}"; }
info()  { printf "%bℹ %s%b\n" "${C_BLUE}"   "$*" "${C_RESET}"; }
ok()    { printf "%b✓ %s%b\n" "${C_GREEN}"  "$*" "${C_RESET}"; }
warn()  { printf "%b! %s%b\n" "${C_YELLOW}" "$*" "${C_RESET}"; }
err()   { printf "%b✗ %s%b\n" "${C_RED}"    "$*" "${C_RESET}" >&2; }

# ------------- repo root -------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Try git root, else two levels up from bin/
if ROOT_DIR=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null); then
  :
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"

# ------------- defaults -------------
IN_DIR="assets/diagrams"
OUT_DIR="assets/diagrams"
THEME="${MMD_THEME:-default}" # default|neutral|dark|forest|base
JOBS="1"
FORCE_ALL="0"
DRY_RUN="0"
WATCH="0"

usage() {
  sed -n '1,80p' "${BASH_SOURCE[0]}" | sed -n '1,60p' | sed 's/^# \{0,1\}//'
}

# ------------- parse args -------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --in)      IN_DIR="${2:?}"; shift 2 ;;
    --out)     OUT_DIR="${2:?}"; shift 2 ;;
    --theme)   THEME="${2:?}"; shift 2 ;;
    --jobs)    JOBS="${2:?}"; shift 2 ;;
    --all)     FORCE_ALL="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --watch)   WATCH="1"; shift ;;
    *) err "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

# ------------- preflight -------------
if [[ ! -d "${IN_DIR}" ]]; then
  err "Input directory not found: ${IN_DIR}"
  exit 1
fi
mkdir -p "${OUT_DIR}"

# Resolve jobs
if [[ "${JOBS}" == "auto" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    JOBS="$(sysctl -n hw.ncpu || echo 4)"
  else
    JOBS="4"
  fi
fi

# Find Mermaid CLI
MERMAID_BIN=""
if command -v mmdc >/dev/null 2>&1; then
  MERMAID_BIN="mmdc"
elif command -v npx >/dev/null 2>&1; then
  MERMAID_BIN="npx -y @mermaid-js/mermaid-cli"
else
  err "Mermaid CLI not found. Install one of:
    • npm i -g @mermaid-js/mermaid-cli     (provides 'mmdc'), or
    • use: npx -y @mermaid-js/mermaid-cli
  "
  exit 1
fi

# mmdc supports themes; we pass via --theme.
info "Renderer: ${MERMAID_BIN} | theme: ${THEME} | jobs: ${JOBS}"
info "In: ${IN_DIR}  Out: ${OUT_DIR}"

# ------------- collect tasks -------------
mapfile -t INPUTS < <(find "${IN_DIR}" -type f -name '*.mmd' | sort)
if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  warn "No .mmd files found in ${IN_DIR}."
  exit 0
fi

# Build list (input|output) respecting incremental rebuild
TASKS=()
for inpath in "${INPUTS[@]}"; do
  base="$(basename "${inpath}" .mmd)"
  outpath="${OUT_DIR}/${base}.svg"
  if [[ "${FORCE_ALL}" == "1" ]]; then
    TASKS+=("${inpath}|${outpath}")
  else
    if [[ ! -f "${outpath}" || "${inpath}" -nt "${outpath}" ]]; then
      TASKS+=("${inpath}|${outpath}")
    fi
  fi
done

TOTAL="${#INPUTS[@]}"
NEEDED="${#TASKS[@]}"

if [[ "${NEEDED}" -eq 0 ]]; then
  ok "All ${TOTAL} diagram(s) are up-to-date."
  exit 0
fi

info "Rendering ${NEEDED}/${TOTAL} diagram(s)…"

# ------------- render function -------------
render_one() {
  local pair="$1"
  local inpath="${pair%%|*}"
  local outpath="${pair##*|}"

  printf "%b↻ %s → %s%b\n" "${C_DIM}" "${inpath}" "${outpath}" "${C_RESET}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  # Ensure out dir exists (support nested mirrors if needed later)
  mkdir -p "$(dirname "${outpath}")"

  # Execute mermaid-cli
  # -t|--theme    : theme name
  # -i|--input    : input .mmd
  # -o|--output   : output .svg
  # --scale 1     : default scale (tweak if fonts look small)
  # --backgroundColor transparent : avoid white boxes on dark themes
  # Timeout via node flags can be added if puppeteer stalls.
  ${MERMAID_BIN} \
    -t "${THEME}" \
    -i "${inpath}" \
    -o "${outpath}" \
    --scale 1 \
    --backgroundColor transparent \
    >/dev/null

  local rc=$?
  if [[ $rc -ne 0 ]]; then
    err "Failed: ${inpath}"
    return $rc
  fi
  ok "Rendered: ${outpath}"
}

export -f render_one
export MERMAID_BIN THEME C_DIM C_RESET C_GREEN C_RED C_YELLOW C_BLUE

# ------------- watch mode -------------
if [[ "${WATCH}" == "1" ]]; then
  if ! command -v entr >/dev/null 2>&1; then
    err "--watch requested but 'entr' not found. Install: brew install entr (macOS) / apt-get install entr"
    exit 2
  fi
  info "Entering watch mode (rebuild on change)…"
  # Watch both inputs and existing outputs
  find "${IN_DIR}" -type f -name '*.mmd' -print0 | \
    entr -p sh -c '
      echo "== Change detected ==";
      '"${BASH_SOURCE[0]}"' --in "'"${IN_DIR}"'" --out "'"${OUT_DIR}"'" --theme "'"${THEME}"'" --jobs "'"${JOBS}"'"
    '
  exit 0
fi

# ------------- parallel render -------------
# Build a null-delimited stream for robust filenames
printf "%s\0" "${TASKS[@]}" | \
  xargs -0 -n1 -P "${JOBS}" bash -c 'render_one "$@"' _

ok "Done."
