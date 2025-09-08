#!/usr/bin/env bash
# fetch_data.sh — Fetch and stage data for SpectraMind V50 (Ariel Challenge)
#
# Supports:
#   • Kaggle competition download (default: ariel-data-challenge-2025)
#   • Kaggle dataset download (user/dataset:version or dataset ref)
#   • DVC pull from configured remote
#
# Examples:
#   ./scripts/fetch_data.sh                               # Kaggle competition → data/raw
#   ./scripts/fetch_data.sh -t dataset -k "yourname/ariel-precalibrated:latest"
#   ./scripts/fetch_data.sh -t dvc
#   ./scripts/fetch_data.sh -o data/raw -n                # dry-run
#
# Repo expectations (per scaffold):
#   data/
#     raw/
#     interim/
#     processed/
#     external/
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

# ---------- defaults ----------
TYPE="competition"         # competition | dataset | dvc
KAGGLE_REF="ariel-data-challenge-2025"          # competition slug by default
KAGGLE_DATASET_REF=""      # e.g. user/dataset:version
OUT_DIR="data/raw"
FORCE=false
DRY_RUN=false
QUIET=false
UNZIP=true

# colors
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; NC=$'\033[0m'
else
  BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; NC=""
fi

usage() {
  cat <<EOF
${BOLD}fetch_data.sh${NC} — Fetch Ariel data into repo-standard folders.

${BOLD}Usage${NC}
  $0 [options]

${BOLD}Options${NC}
  -t, --type TYPE         Source type: competition | dataset | dvc (default: competition)
  -k, --kaggle REF        Kaggle ref:
                            • if TYPE=competition  → competition slug (default: ariel-data-challenge-2025)
                            • if TYPE=dataset      → dataset ref e.g. user/dataset[:version]
  -o, --out DIR           Destination folder (default: data/raw)
  -f, --force             Overwrite existing files (re-download/unzip)
  -n, --dry-run           Print plan only; do not execute
  -q, --quiet             Minimal output
  --no-unzip              Do not unzip archives after download
  -h, --help              Show this help

${BOLD}Examples${NC}
  $0
  $0 -t dataset -k "yourname/ariel-precalibrated:latest" -o data/raw
  $0 -t dvc

${BOLD}Notes${NC}
- Requires Kaggle CLI configured (~/.kaggle/kaggle.json) for Kaggle flows.
- DVC flow assumes 'dvc remote' already set up (or local cache present).
- Safe and idempotent; use --force to re-download/unpack.
EOF
}

log()      { $QUIET || printf "%s\n" "$*"; }
log_ok()   { $QUIET || printf "%s\n" "${GREEN}$*${NC}"; }
log_warn() { $QUIET || printf "%s\n" "${YELLOW}$*${NC}"; }
log_err()  { printf "%s\n" "${RED}$*${NC}" >&2; }

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--type)       TYPE="${2:-}"; shift 2 ;;
    -k|--kaggle)     KAGGLE_REF="${2:-}"; KAGGLE_DATASET_REF="${2:-}"; shift 2 ;;
    -o|--out)        OUT_DIR="${2:-}"; shift 2 ;;
    -f|--force)      FORCE=true; shift ;;
    -n|--dry-run)    DRY_RUN=true; shift ;;
    -q|--quiet)      QUIET=true; shift ;;
    --no-unzip)      UNZIP=false; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               log_err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ---------- helpers ----------
need_cmd() {
  command -v "$1" &>/dev/null || { log_err "Missing required command: $1"; exit 127; }
}

plan() {
  log "${BOLD}Fetch plan${NC}"
  log "  Type      : ${TYPE}"
  if [[ "${TYPE}" == "competition" ]]; then
    log "  Kaggle    : competition/${KAGGLE_REF}"
  elif [[ "${TYPE}" == "dataset" ]]; then
    log "  Kaggle    : dataset/${KAGGLE_DATASET_REF}"
  fi
  log "  Out dir   : ${OUT_DIR}"
  log "  Force     : ${FORCE}"
  log "  Unzip     : ${UNZIP}"
}

ensure_dirs() {
  mkdir -p -- "${OUT_DIR}"
}

download_competition() {
  need_cmd kaggle
  # download to OUT_DIR
  local args=(competitions download -c "${KAGGLE_REF}" -p "${OUT_DIR}")
  $FORCE && args+=( --force )
  log "kaggle ${args[*]}"
  $DRY_RUN || kaggle "${args[@]}"
}

download_dataset() {
  need_cmd kaggle
  if [[ -z "${KAGGLE_DATASET_REF}" ]]; then
    log_err "TYPE=dataset requires --kaggle user/dataset[:version]"
    exit 2
  fi
  local args=(datasets download -d "${KAGGLE_DATASET_REF}" -p "${OUT_DIR}")
  $FORCE && args+=( --force )
  log "kaggle ${args[*]}"
  $DRY_RUN || kaggle "${args[@]}"
}

maybe_unzip_all() {
  $UNZIP || { log_warn "Skipping unzip (--no-unzip)"; return 0; }
  shopt -s nullglob
  local zips=("${OUT_DIR}"/*.zip)
  local tars=("${OUT_DIR}"/*.tar.gz "${OUT_DIR}"/*.tgz)
  if (( ${#zips[@]} == 0 && ${#tars[@]} == 0 )); then
    log_warn "No archives found to unzip in ${OUT_DIR}"
    return 0
  fi
  for zf in "${zips[@]}"; do
    local mark="${zf}.unzipped"
    if $FORCE || [[ ! -f "${mark}" ]]; then
      log "unzip -o '${zf}' -d '${OUT_DIR}'"
      $DRY_RUN || unzip -o -q "${zf}" -d "${OUT_DIR}"
      $DRY_RUN || touch "${mark}"
    else
      log "Skip unzip (already done): ${zf}"
    fi
  done
  for tf in "${tars[@]}"; do
    local mark="${tf}.untarred"
    if $FORCE || [[ ! -f "${mark}" ]]; then
      log "tar -xzf '${tf}' -C '${OUT_DIR}'"
      $DRY_RUN || tar -xzf "${tf}" -C "${OUT_DIR}"
      $DRY_RUN || touch "${mark}"
    else
      log "Skip untar (already done): ${tf}"
    fi
  done
}

pull_dvc() {
  need_cmd dvc
  # If OUT_DIR not default, we still pull whole workspace (fast if cached)
  log "dvc pull"
  $DRY_RUN || dvc pull
}

post_checks() {
  # Basic sanity: ensure OUT_DIR not empty
  if [[ -d "${OUT_DIR}" ]]; then
    local count
    count=$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')
    if [[ "${count}" -eq 0 ]]; then
      log_warn "No files present in ${OUT_DIR} yet."
    else
      log_ok "Data present in ${OUT_DIR}."
    fi
  fi
  # Convenience symlink for Kaggle kernels or code expecting canonical names (optional)
  # ln -snf "${OUT_DIR}" "data/raw"  # uncomment if you want to force canonical link
}

main() {
  plan
  $DRY_RUN && exit 0
  ensure_dirs

  case "${TYPE}" in
    competition)
      download_competition
      maybe_unzip_all
      ;;
    dataset)
      download_dataset
      maybe_unzip_all
      ;;
    dvc)
      pull_dvc
      ;;
    *)
      log_err "Unknown --type '${TYPE}'"
      exit 2
      ;;
  esac

  post_checks
  log_ok "Fetch completed."
}

main
