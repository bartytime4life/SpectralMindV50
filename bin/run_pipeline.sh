#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — End-to-End Pipeline Orchestrator
#
# Runs the physics-informed, dual-encoder pipeline:
#   calibrate → preprocess → train → predict → diagnose → submit
#
# Features
#   • DVC-aware: use `dvc repro` when available, or call CLI stages directly
#   • Partial runs: --from/--to/--only; skip stages with --skip STAGE
#   • Hydra overrides passthrough after `--` (e.g., +env=kaggle +training.max_epochs=1)
#   • Kaggle/offline-safe: auto-disables DVC network actions, respects env defaults
#   • Dry-run, force rebuild, and pretty logging
#
# Usage
#   bin/run_pipeline.sh                      # full pipeline
#   bin/run_pipeline.sh --from preprocess --to predict
#   bin/run_pipeline.sh --only train
#   bin/run_pipeline.sh --no-dvc             # bypass DVC, call CLI directly
#   bin/run_pipeline.sh --force              # force rebuild (DVC or CLI)
#   bin/run_pipeline.sh -- env=kaggle        # shorthand: forwards as +env=kaggle
#   bin/run_pipeline.sh -- +env=kaggle +training.fast=true
#
# Stages (canonical order)
#   calibrate | preprocess | train | predict | diagnose | submit
#
# Notes
#   • CLI entry is `python -m spectramind <stage>`; expects configs/ to be present.
#   • DVC stage names should mirror these six for smooth operation.
# ==============================================================================

set -Eeuo pipefail

# ------------- colors & logging -------------
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
if ROOT_DIR=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null); then :; else ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"; fi
cd "${ROOT_DIR}"

# ------------- defaults -------------
STAGES_ALL=(calibrate preprocess train predict diagnose submit)
RUN_FROM=""
RUN_TO=""
RUN_ONLY=""
SKIP_STAGES=()
USE_DVC="auto"         # auto|yes|no
FORCE="0"
DRY_RUN="0"
PY="python3"           # or python
# Hydra overrides after '--'
HYDRA_OVERRIDES=()

usage() {
  cat <<'USAGE'
SpectraMind V50 — End-to-End Pipeline Orchestrator

Usage:
  bin/run_pipeline.sh [options] [-- <Hydra overrides>]

Options:
  --from STAGE       Start from this stage (inclusive)
  --to STAGE         Stop after this stage (inclusive)
  --only STAGE       Run exactly this stage
  --skip STAGE       Skip a stage (can repeat)
  --dvc              Force using DVC (dvc repro)
  --no-dvc           Bypass DVC; run stages via CLI
  --force            Force rebuild (DVC: --force; CLI: add --rebuild when supported)
  --dry-run          Show what would run without executing
  --help             Show help

Stages:
  calibrate | preprocess | train | predict | diagnose | submit

Hydra overrides:
  Everything after '--' is forwarded to Hydra as '+key=value' pairs.
  Examples:
    bin/run_pipeline.sh -- +env=kaggle
    bin/run_pipeline.sh -- +env=local +training.max_epochs=1
USAGE
}

# ------------- parse args -------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --from)     RUN_FROM="${2:?}"; shift 2 ;;
    --to)       RUN_TO="${2:?}"; shift 2 ;;
    --only)     RUN_ONLY="${2:?}"; shift 2 ;;
    --skip)     SKIP_STAGES+=("${2:?}"); shift 2 ;;
    --dvc)      USE_DVC="yes"; shift ;;
    --no-dvc)   USE_DVC="no"; shift ;;
    --force)    FORCE="1"; shift ;;
    --dry-run)  DRY_RUN="1"; shift ;;
    --)         shift; HYDRA_OVERRIDES=("$@"); break ;;
    *)          err "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

# ------------- traps -------------
cleanup() { :; }
trap cleanup EXIT

# ------------- helpers -------------
have_cmd()   { command -v "$1" >/dev/null 2>&1; }
in_array()   { local x="$1"; shift; for i in "$@"; do [[ "$i" == "$x" ]] && return 0; done; return 1; }
stage_index(){ local s="$1"; local i; for i in "${!STAGES_ALL[@]}"; do [[ "${STAGES_ALL[$i]}" == "$s" ]] && { echo "$i"; return 0; }; done; echo -1; }

# ------------- determine runner mode -------------
is_kaggle="0"
if [[ -d "/kaggle" || "${KAGGLE_KERNEL_RUN_TYPE:-}" != "" ]]; then is_kaggle="1"; fi

if [[ "${USE_DVC}" == "auto" ]]; then
  if have_cmd dvc; then
    USE_DVC="yes"
  else
    USE_DVC="no"
  fi
fi

# Kaggle often offline; do not attempt network pulls/gc here.
if [[ "${is_kaggle}" == "1" && "${USE_DVC}" == "yes" ]]; then
  info "Kaggle environment detected; using DVC for repro only (no network)."
fi

# ------------- compute plan -------------
PLAN=()
if [[ -n "${RUN_ONLY}" ]]; then
  PLAN=("${RUN_ONLY}")
else
  # derive slice [from..to]
  local_from=0
  local_to=$((${#STAGES_ALL[@]} - 1))

  if [[ -n "${RUN_FROM}" ]]; then
    idx=$(stage_index "${RUN_FROM}")
    [[ $idx -lt 0 ]] && { err "Unknown --from stage: ${RUN_FROM}"; exit 2; }
    local_from=$idx
  fi
  if [[ -n "${RUN_TO}" ]]; then
    idx=$(stage_index "${RUN_TO}")
    [[ $idx -lt 0 ]] && { err "Unknown --to stage: ${RUN_TO}"; exit 2; }
    local_to=$idx
  fi

  for i in $(seq "${local_from}" "${local_to}"); do
    PLAN+=("${STAGES_ALL[$i]}")
  done
fi

# apply skips
if [[ ${#SKIP_STAGES[@]} -gt 0 ]]; then
  filtered=()
  for s in "${PLAN[@]}"; do
    if in_array "$s" "${SKIP_STAGES[@]}"; then
      warn "Skipping stage: ${s}"
      continue
    fi
    filtered+=("$s")
  done
  PLAN=("${filtered[@]}")
fi

if [[ ${#PLAN[@]} -eq 0 ]]; then
  warn "Nothing to do (empty plan)."
  exit 0
fi

# ------------- print plan -------------
info "Runner     : $([[ "${USE_DVC}" == "yes" ]] && echo 'DVC repro' || echo 'CLI direct')"
info "Environment: $([[ "${is_kaggle}" == "1" ]] && echo 'Kaggle' || echo 'Local/CI')"
info "Stages plan: ${PLAN[*]}"
if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
  info "Hydra overrides: ${HYDRA_OVERRIDES[*]}"
fi
[[ "${FORCE}" == "1" ]] && info "Force rebuild: enabled"
[[ "${DRY_RUN}" == "1" ]] && info "Dry-run: enabled"

# ------------- DVC runner -------------
run_dvc_stage() {
  local s="$1"
  local dvc_args=()
  [[ "${FORCE}" == "1" ]] && dvc_args+=(--force)
  # If user provided Hydra overrides, forward them via params file/env if your DVC stages support it.
  # Otherwise they apply only to CLI mode.
  printf "%b↻ DVC: %s%b\n" "${C_DIM}" "${s}" "${C_RESET}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Would run: dvc repro ${dvc_args[*]} ${s}"
    return 0
  fi
  dvc repro "${dvc_args[@]}" "${s}"
}

# ------------- CLI runner -------------
# Map stage -> CLI command
cli_cmd_for() {
  local s="$1"
  case "$s" in
    calibrate) printf "%s -m spectramind calibrate" "${PY}" ;;
    preprocess) printf "%s -m spectramind preprocess" "${PY}" ;;
    train) printf "%s -m spectramind train" "${PY}" ;;
    predict) printf "%s -m spectramind predict" "${PY}" ;;
    diagnose) printf "%s -m spectramind diagnose" "${PY}" ;;
    submit) printf "%s -m spectramind submit" "${PY}" ;;
    *) err "Unknown stage for CLI: $s"; return 1 ;;
  esac
}

run_cli_stage() {
  local s="$1"
  local cmd; cmd="$(cli_cmd_for "${s}")" || return 1
  local args=()

  # Honor --force where supported by your CLI; we pass a generic flag if available.
  # If your Typer commands use a different flag (e.g., --rebuild/--overwrite), change here:
  if [[ "${FORCE}" == "1" ]]; then
    # args+=(--rebuild)  # uncomment if your CLI implements it
    :
  fi

  printf "%b↻ CLI: %s%b\n" "${C_DIM}" "${s}" "${C_RESET}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Would run: ${cmd} ${args[*]} -- ${HYDRA_OVERRIDES[*]}"
    return 0
  fi

  # Run command with hydra overrides
  # If you use Typer + Hydra, this pattern forwards overrides cleanly.
  if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
    eval "${cmd}" "${args[@]}" -- "${HYDRA_OVERRIDES[@]}"
  else
    eval "${cmd}" "${args[@]}"
  fi
}

# ------------- execute plan -------------
for s in "${PLAN[@]}"; do
  if [[ "${USE_DVC}" == "yes" ]]; then
    run_dvc_stage "${s}"
  else
    run_cli_stage "${s}"
  fi
done

ok "Pipeline complete."
