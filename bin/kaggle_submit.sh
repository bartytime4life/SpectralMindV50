#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Kaggle Notebook Builder & Uploader
#
# Builds a self-contained Kaggle Notebook project (code + configs + metadata),
# declares inputs (competition & datasets), pushes via Kaggle CLI, and (optional)
# fetches kernel outputs (e.g., submission.csv) for local packaging.
#
# What it does
#   • Creates a sandbox folder: kaggle/nb/<slug>/
#   • Writes kernel-metadata.json with data_sources (competition, datasets)
#   • Copies src/spectramind, configs/, and a minimal kaggle.ipynb
#   • Pushes kernel (kaggle kernels push -p) and optionally polls status
#   • (Optional) downloads outputs (kaggle kernels output …)
#
# Kaggle specifics
#   • Competition inputs (auto-mounted at /kaggle/input/<name>/):contentReference[oaicite:3]{index=3}
#   • Internet disabled by default; rely on attached data + included code:contentReference[oaicite:4]{index=4}
#   • Ensure ~/.kaggle/kaggle.json exists with mode 600:contentReference[oaicite:5]{index=5}
#
# Usage
#   bin/kaggle_submit.sh --slug spectramind-v50-submission \
#                        --title "SpectraMind V50 Submission" \
#                        --competition ariel-data-challenge-2025 \
#                        --gpu --private --push --wait --fetch
#
# Common flags
#   --slug <id>              Kernel slug (required, lowercase, no spaces)
#   --title "<title>"        Kernel title (required)
#   --competition <id>       Competition ID (default: ariel-data-challenge-2025)
#   --dataset <owner/name>   Extra dataset dependency (repeatable)
#   --src src/spectramind    Code dir to include (default)
#   --configs configs         Config dir to include (default)
#   --nb kaggle.ipynb        Notebook filename (default name)
#   --path kaggle/nb         Base folder for kernels (default)
#   --private|--public       Visibility (default: --private)
#   --gpu|--no-gpu           Enable GPU (default: off)
#   --internet|--no-internet Enable internet (default: no-internet)
#   --push                   Push kernel to Kaggle
#   --wait                   Poll kernel status until finished
#   --fetch                  Download kernel outputs into artifacts/kaggle_output/<slug>
#   --dry-run                Print actions only
#
# Notes
#   • The generated notebook runs: `python -m spectramind predict +env=kaggle`
#     and writes submission.csv to /kaggle/working (Kaggle’s writable workspace):contentReference[oaicite:6]{index=6}.
#   • If your pipeline needs a different entry, tweak NOTEBOOK_CMD below.
# ==============================================================================

set -Eeuo pipefail

# ------------- colors -------------
if [[ -t 1 ]]; then
  C_RESET="\033[0m"; C_DIM="\033[2m"; C_GREEN="\033[32m"; C_YELLOW="\033[33m"; C_RED="\033[31m"; C_BLUE="\033[34m"
else
  C_RESET=""; C_DIM=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""
fi
info(){ printf "%bℹ %s%b\n" "${C_BLUE}" "$*" "${C_RESET}"; }
ok() { printf "%b✓ %s%b\n" "${C_GREEN}" "$*" "${C_RESET}"; }
warn(){ printf "%b! %s%b\n" "${C_YELLOW}" "$*" "${C_RESET}"; }
err(){ printf "%b✗ %s%b\n" "${C_RED}" "$*" "${C_RESET}" >&2; }

# ------------- defaults -------------
SLUG=""
TITLE=""
COMPETITION="ariel-data-challenge-2025"
DATASETS=()              # repeatable --dataset owner/name
SRC_DIR="src/spectramind"
CONFIGS_DIR="configs"
NB_NAME="kaggle.ipynb"
BASE_PATH="kaggle/nb"
VISIBILITY="private"     # private|public
GPU="false"
INTERNET="false"
DO_PUSH="0"
DO_WAIT="0"
DO_FETCH="0"
DRY_RUN="0"

usage(){
  sed -n '1,120p' "$0" | sed -n '1,80p'
}

# ------------- args -------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug) SLUG="${2:?}"; shift 2 ;;
    --title) TITLE="${2:?}"; shift 2 ;;
    --competition) COMPETITION="${2:?}"; shift 2 ;;
    --dataset) DATASETS+=("${2:?}"); shift 2 ;;
    --src) SRC_DIR="${2:?}"; shift 2 ;;
    --configs) CONFIGS_DIR="${2:?}"; shift 2 ;;
    --nb) NB_NAME="${2:?}"; shift 2 ;;
    --path) BASE_PATH="${2:?}"; shift 2 ;;
    --private) VISIBILITY="private"; shift ;;
    --public) VISIBILITY="public"; shift ;;
    --gpu) GPU="true"; shift ;;
    --no-gpu) GPU="false"; shift ;;
    --internet) INTERNET="true"; shift ;;
    --no-internet) INTERNET="false"; shift ;;
    --push) DO_PUSH="1"; shift ;;
    --wait) DO_WAIT="1"; shift ;;
    --fetch) DO_FETCH="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ------------- checks -------------
[[ -z "${SLUG}" || -z "${TITLE}" ]] && { err "Provide --slug and --title"; exit 2; }
command -v kaggle >/dev/null 2>&1 || { err "kaggle CLI not found. Install and auth (kaggle.json)"; exit 1; }

# Kaggle credentials (mode 600)
if [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  chmod 600 "${HOME}/.kaggle/kaggle.json" || true
else
  warn "~/.kaggle/kaggle.json not found — Kaggle CLI will fail auth if missing"  # User can still dry-run
fi

# ------------- project layout -------------
KDIR="${BASE_PATH}/${SLUG}"
CODE_DIR="${KDIR}/code"
mkdir -p "${CODE_DIR}"

info "Kernel dir      : ${KDIR}"
info "Kernel slug     : ${SLUG}"
info "Title           : ${TITLE}"
info "Competition     : ${COMPETITION}"
if [[ ${#DATASETS[@]} -gt 0 ]]; then info "Extra datasets  : ${DATASETS[*]}"; fi
info "GPU             : ${GPU}"
info "Internet        : ${INTERNET}"
info "Visibility      : ${VISIBILITY}"

# ------------- copy code & configs -------------
copy_path(){
  local src="$1" dst="$2"
  if [[ ! -e "${src}" ]]; then
    warn "Skip missing: ${src}"
    return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    info "Would copy ${src} -> ${dst}"
    return
  fi
  mkdir -p "$(dirname "${dst}")"
  rsync -a --delete "${src}" "${dst%/*}/"
}

copy_path "${SRC_DIR}"   "${CODE_DIR}/$(basename "${SRC_DIR}")"
copy_path "${CONFIGS_DIR}" "${CODE_DIR}/$(basename "${CONFIGS_DIR}")"

# Optionally include a slim requirements-kaggle.txt if you rely on preinstalled libs only.
# (Kaggle is offline for comps; avoid pip installs):contentReference[oaicite:7]{index=7}.

# ------------- write minimal notebook -------------
# This notebook:
#  • Sets Python path to include code/ (so `python -m spectramind ...` resolves)
#  • Runs your predict stage with Kaggle env config
#  • Writes submission.csv to /kaggle/working
NOTEBOOK_CMD='!python -m spectramind predict +env=kaggle'
NOTEBOOK_JSON="$(cat <<'NB'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SpectraMind V50 — Kaggle Submission Notebook\n",
    "Pipeline: `predict` with offline Kaggle inputs.\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {"execution":{"timeout":0}},
   "source": [
    "import sys, os\n",
    "sys.path.insert(0, os.path.join('/kaggle', 'working', 'code'))\n",
    "print('PYTHONPATH set for embedded code')\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {"execution":{"timeout":0}},
   "source": [
    "import os, subprocess, json, textwrap\n",
    "print('Running SpectraMind V50 predict...')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
  "language_info": {"name":"python","pygments_lexer":"ipython3","version":"3.x"}
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
NB
)"

if [[ "${DRY_RUN}" == "1" ]]; then
  info "Would write notebook ${KDIR}/${NB_NAME}"
else
  printf '%s' "${NOTEBOOK_JSON}" > "${KDIR}/${NB_NAME}"
  # Append execution cell with the actual command (runtime-injected to avoid escaping headaches)
  python3 - <<PY
import json,sys
p="${KDIR}/${NB_NAME}"
with open(p,'r+',encoding='utf-8') as fp:
    nb=json.load(fp)
    nb['cells'].append({
      "cell_type":"code","metadata":{"execution":{"timeout":0}},
      "source":[${NOTEBOOK_CMD!@A}]
    })
    fp.seek(0); json.dump(nb,fp,indent=1,ensure_ascii=False); fp.truncate()
PY
fi

# ------------- kernel metadata -------------
# See: kernel-metadata.json schema — data_sources allow competition + datasets
# data_sources entries format:
#   "competition-<id>" or "<owner>/<dataset>" (CLI accepts both forms)
DATA_SOURCES=("competition-${COMPETITION}")
for ds in "${DATASETS[@]}"; do DATA_SOURCES+=("${ds}"); done

# bools/visibility -> Kaggle expected strings
K_GPU=$([[ "${GPU}" == "true" ]] && echo "true" || echo "false")
K_INTERNET=$([[ "${INTERNET}" == "true" ]] && echo "true" || echo "false")
K_PRIV=$([[ "${VISIBILITY}" == "private" ]] && echo "true" || echo "false")

if [[ "${DRY_RUN}" != "1" ]]; then
  python3 - "$KDIR" "$SLUG" "$TITLE" "$K_PRIV" "$K_GPU" "$K_INTERNET" "${DATA_SOURCES[@]}" <<'PY'
import sys,json,os
kdir,slug,title,priv,gpu,net,*sources=sys.argv[1:]
meta={
  "id": slug,
  "title": title,
  "code_file": "kaggle.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": (priv=="true"),
  "enable_gpu": (gpu=="true"),
  "enable_internet": (net=="true"),
  "dataset_sources": [s for s in sources if '/' in s and not s.startswith('competition-')],
  "competition_sources": [s.replace('competition-','') for s in sources if s.startswith('competition-')],
  "kernel_sources": []
}
with open(os.path.join(kdir,'kernel-metadata.json'),'w',encoding='utf-8') as fp:
  json.dump(meta,fp,indent=2)
PY
else
  info "Would write kernel-metadata.json with data_sources: ${DATA_SOURCES[*]}"
fi

ok "Kaggle project prepared at: ${KDIR}"

# ------------- push kernel -------------
if [[ "${DO_PUSH}" == "1" ]]; then
  info "Pushing kernel to Kaggle…"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Would run: kaggle kernels push -p ${KDIR}"
  else
    kaggle kernels push -p "${KDIR}"
  fi
fi

# ------------- wait/poll -------------
if [[ "${DO_WAIT}" == "1" ]]; then
  # Need <username>/<slug> to poll. username is from kaggle config or API whoami
  USERNAME="$(kaggle config view 2>/dev/null | awk -F= '/username/ {print $2}' | tr -d ' ' || true)"
  if [[ -z "${USERNAME}" ]]; then
    USERNAME="$(kaggle datasets list -s "" -p 1 2>/dev/null >/dev/null; echo "${KAGGLE_USERNAME:-}")"
  fi
  if [[ -z "${USERNAME}" ]]; then
    warn "Could not infer Kaggle username; skipping status polling."
  else
    KREF="${USERNAME}/${SLUG}"
    info "Polling kernel status: ${KREF}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "Would run: kaggle kernels status ${KREF}"
    else
      # Simple poll loop
      for i in $(seq 1 60); do
        out="$(kaggle kernels status "${KREF}" 2>&1 || true)"
        echo "${out}"
        echo "${out}" | grep -qiE 'complete|success|error|failed' && break
        sleep 10
      done
    fi
  fi
fi

# ------------- fetch outputs -------------
if [[ "${DO_FETCH}" == "1" ]]; then
  USERNAME="${USERNAME:-$(kaggle config view 2>/dev/null | awk -F= '/username/ {print $2}' | tr -d ' ')}"
  if [[ -z "${USERNAME}" ]]; then
    warn "Could not infer Kaggle username; skipping fetch."
  else
    KREF="${USERNAME}/${SLUG}"
    OUTDIR="artifacts/kaggle_output/${SLUG}"
    info "Fetching kernel outputs → ${OUTDIR}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "Would run: kaggle kernels output ${KREF} -p ${OUTDIR}"
    else
      mkdir -p "${OUTDIR}"
      kaggle kernels output "${KREF}" -p "${OUTDIR}"
      ok "Outputs fetched. Inspect ${OUTDIR}/ for submission.csv."
    fi
  fi
fi

ok "Done."
