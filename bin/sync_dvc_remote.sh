#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — DVC Remote Sync Helper
# -----------------------------------------------------------------------------
# Safely sync DVC artifacts to/from a configured remote.
# - Kaggle-aware (disables push/gc; offline-safe)
# - Concurrency-safe (flock)
# - Repo-root auto-detection
# - Helpful flags and dry-run mode
#
# Usage examples:
#   bin/sync_dvc_remote.sh --status
#   bin/sync_dvc_remote.sh --pull --remote my-s3
#   bin/sync_dvc_remote.sh --push --jobs 8
#   bin/sync_dvc_remote.sh --fetch --repro
#   bin/sync_dvc_remote.sh --list-remotes
#
# Environment overrides:
#   DVC_REMOTE       default remote name (overridden by --remote)
#   DVC_JOBS         default jobs if --jobs not supplied
#   DVC_DRY_RUN      "1" to simulate commands without executing
# -----------------------------------------------------------------------------

set -Eeuo pipefail

# ---- small helpers -----------------------------------------------------------
ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(ts)] $*"; }
err() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }

is_kaggle() {
  [[ -d "/kaggle/input" ]] || [[ "${KAGGLE_KERNEL_RUN_TYPE:-}" != "" ]]
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# ---- usage -------------------------------------------------------------------
usage() {
  cat <<'EOF'
DVC Remote Sync Helper

Flags:
  --remote NAME        Use specific DVC remote (default: $DVC_REMOTE or repo default)
  --push               dvc push (disabled on Kaggle)
  --pull               dvc pull
  --fetch              dvc fetch
  --status             dvc status --cloud
  --gc                 dvc gc -c  (disabled on Kaggle)
  --repro              dvc repro  (recompute pipeline stages)
  --list-remotes       dvc remote list
  --jobs N             Parallel jobs for transfer (default: $DVC_JOBS or 4)
  --dry-run            Print commands without executing (or set DVC_DRY_RUN=1)
  --all-branches       Include all branches for gc (careful!)
  -h|--help            Show this help

Notes:
  * On Kaggle, push/gc are blocked to keep runs offline-safe.
  * The script uses a simple flock to avoid concurrent DVC operations.
EOF
}

# ---- defaults ----------------------------------------------------------------
REMOTE="${DVC_REMOTE:-}"
DO_PUSH="0"
DO_PULL="0"
DO_FETCH="0"
DO_STATUS="0"
DO_GC="0"
DO_REPRO="0"
DO_LIST="0"
JOBS="${DVC_JOBS:-4}"
DRY_RUN="${DVC_DRY_RUN:-0}"
ALL_BRANCHES="0"

# ---- parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)        REMOTE="${2:-}"; shift ;;
    --push)          DO_PUSH="1" ;;
    --pull)          DO_PULL="1" ;;
    --fetch)         DO_FETCH="1" ;;
    --status)        DO_STATUS="1" ;;
    --gc)            DO_GC="1" ;;
    --repro)         DO_REPRO="1" ;;
    --list-remotes)  DO_LIST="1" ;;
    --jobs)          JOBS="${2:-}"; shift ;;
    --dry-run)       DRY_RUN="1" ;;
    --all-branches)  ALL_BRANCHES="1" ;;
    -h|--help)       usage; exit 0 ;;
    *)               err "Unknown argument: $1 (use --help)" ;;
  esac
  shift
done

# ---- locate repo root --------------------------------------------------------
if have_cmd git; then
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
else
  REPO_ROOT=""
fi

if [[ -z "$REPO_ROOT" ]]; then
  # fallback: search upward for dvc.yaml
  REPO_ROOT="$(pwd)"
  while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/dvc.yaml" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
  done
fi

[[ -f "$REPO_ROOT/dvc.yaml" ]] || err "Could not find dvc.yaml. Run from inside repo or set REPO_ROOT."

cd "$REPO_ROOT"

# ---- sanity checks -----------------------------------------------------------
have_cmd dvc || err "dvc not found. Please install DVC (pip install dvc[<remote>])"
have_cmd flock || log "flock not found; continuing without interprocess lock (low risk)."

# ---- resolve default remote if not provided ----------------------------------
if [[ -z "$REMOTE" ]]; then
  # try reading default remote from dvc config
  REMOTE="$(dvc remote list --quiet 2>/dev/null | awk 'NR==1{print $1}')"
  REMOTE="${REMOTE%%:*}"
fi

# ---- Kaggle safety gates -----------------------------------------------------
if is_kaggle; then
  if [[ "$DO_PUSH" == "1" || "$DO_GC" == "1" ]]; then
    err "Push/GC is disabled on Kaggle (offline environment). Use pull/fetch/status only."
  fi
fi

# ---- run helper with dry-run support -----------------------------------------
_run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] $*"
  else
    eval "$@"
  fi
}

# ---- lock to prevent concurrent DVC ops -------------------------------------
LOCK_DIR="${REPO_ROOT}/.dvc_sync_lock"
mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/lockfile"

with_lock() {
  if have_cmd flock; then
    flock -w 60 200 || err "Could not acquire DVC sync lock within 60s"
    "$@"
    # flock will be released when FD 200 closes
  else
    "$@"
  fi
}

# ---- operations --------------------------------------------------------------
list_remotes() {
  log "DVC remotes:"
  _run "dvc remote list"
}

status_cloud() {
  local rflag=""
  [[ -n "$REMOTE" ]] && rflag="--remote ${REMOTE}"
  log "Checking DVC cloud status ${REMOTE:+(remote: $REMOTE)}"
  _run "dvc status --cloud ${rflag}"
}

do_fetch() {
  local rflag=""
  [[ -n "$REMOTE" ]] && rflag="--remote ${REMOTE}"
  log "Fetching DVC objects ${REMOTE:+from $REMOTE} (jobs=${JOBS})"
  _run "dvc fetch ${rflag} -j ${JOBS}"
}

do_pull() {
  local rflag=""
  [[ -n "$REMOTE" ]] && rflag="--remote ${REMOTE}"
  log "Pulling DVC objects ${REMOTE:+from $REMOTE} (jobs=${JOBS})"
  _run "dvc pull ${rflag} -j ${JOBS}"
}

do_push() {
  local rflag=""
  [[ -n "$REMOTE" ]] && rflag="--remote ${REMOTE}"
  log "Pushing DVC objects ${REMOTE:+to $REMOTE} (jobs=${JOBS})"
  _run "dvc push ${rflag} -j ${JOBS}"
}

do_gc() {
  local base="dvc gc -c"
  [[ "$ALL_BRANCHES" == "1" ]] && base+=" --all-branches --all-tags"
  log "Garbage collecting unused DVC objects (cloud) ${ALL_BRANCHES:+[all branches/tags]}"
  _run "$base"
}

do_repro() {
  log "Reproducing pipeline stages (dvc repro)"
  _run "dvc repro"
}

# ---- main dispatch -----------------------------------------------------------
main() {
  # If no verb given, default to --status
  if [[ "$DO_PUSH$DO_PULL$DO_FETCH$DO_STATUS$DO_GC$DO_REPRO$DO_LIST" == "0000000" ]]; then
    DO_STATUS="1"
  fi

  {
    exec 200>"$LOCK_FILE"
    with_lock bash -c '
      set -Eeuo pipefail

      [[ "'"$DO_LIST"'" == "1" ]]   && list_remotes
      [[ "'"$DO_STATUS"'" == "1" ]] && status_cloud
      [[ "'"$DO_FETCH"'" == "1" ]]  && do_fetch
      [[ "'"$DO_PULL"'" == "1" ]]   && do_pull
      [[ "'"$DO_PUSH"'" == "1" ]]   && do_push
      [[ "'"$DO_GC"'" == "1" ]]     && do_gc
      [[ "'"$DO_REPRO"'" == "1" ]]  && do_repro
    '
  }
  log "Done."
}

main "$@"
