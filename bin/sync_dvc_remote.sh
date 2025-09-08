#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — DVC Remote Sync Helper (Upgraded)
# -----------------------------------------------------------------------------
# Safely sync DVC artifacts to/from a configured remote.
# • Kaggle-aware (disables push/gc; offline-safe)
# • Concurrency-safe (flock)
# • Repo-root auto-detection
# • Retries with exponential backoff on network-y DVC ops
# • Dry-run (no effect) and optional JSON summary
#
# Usage:
#   bin/sync_dvc_remote.sh --status
#   bin/sync_dvc_remote.sh --pull --remote my-s3 --jobs 8
#   bin/sync_dvc_remote.sh --fetch --repro
#   bin/sync_dvc_remote.sh --list-remotes
#
# Env overrides:
#   DVC_REMOTE     default remote name (overridden by --remote)
#   DVC_JOBS       default jobs if --jobs not supplied (default 4)
#   DVC_DRY_RUN    "1" to simulate commands without executing
#   DVC_JSON       "1" to emit a JSON summary to stdout at the end
# -----------------------------------------------------------------------------

set -Eeuo pipefail

# --------- small helpers ------------------------------------------------------
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }
warn() { echo "[$(ts)] WARN: $*" >&2; }
die() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }

is_kaggle() { [[ -d "/kaggle/input" ]] || [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

# --------- usage --------------------------------------------------------------
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
  --all-branches       Include all branches/tags for gc (dangerous)
  --dry-run            Print commands without executing (or set DVC_DRY_RUN=1)
  --json               Emit JSON summary to stdout (or set DVC_JSON=1)
  -h|--help            Show this help

Notes:
  * On Kaggle, push/gc are blocked for safety; use pull/fetch/status only.
  * Uses flock to avoid concurrent DVC operations.
  * Exits non-zero if any selected operation fails.
EOF
}

# --------- defaults -----------------------------------------------------------
REMOTE="${DVC_REMOTE:-}"
DO_PUSH=0; DO_PULL=0; DO_FETCH=0; DO_STATUS=0; DO_GC=0; DO_REPRO=0; DO_LIST=0
JOBS="${DVC_JOBS:-4}"
DRY_RUN="${DVC_DRY_RUN:-0}"
ALL_BRANCHES=0
WANT_JSON="${DVC_JSON:-0}"

# --------- parse args ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)        REMOTE="${2:-}"; shift 2 ;;
    --push)          DO_PUSH=1; shift ;;
    --pull)          DO_PULL=1; shift ;;
    --fetch)         DO_FETCH=1; shift ;;
    --status)        DO_STATUS=1; shift ;;
    --gc)            DO_GC=1; shift ;;
    --repro)         DO_REPRO=1; shift ;;
    --list-remotes)  DO_LIST=1; shift ;;
    --jobs)          JOBS="${2:-}"; shift 2 ;;
    --all-branches)  ALL_BRANCHES=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --json)          WANT_JSON=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               die "Unknown argument: $1 (use --help)";;
  esac
done

# Validate jobs
[[ "$JOBS" =~ ^[0-9]+$ ]] || die "--jobs expects an integer"

# --------- repo root detection ------------------------------------------------
if have_cmd git; then
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
else
  REPO_ROOT=""
fi
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(pwd)"
  while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/dvc.yaml" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
  done
fi
[[ -f "$REPO_ROOT/dvc.yaml" ]] || die "Could not find dvc.yaml. Run from inside repo or set DVC working dir."
cd "$REPO_ROOT"

# --------- sanity checks ------------------------------------------------------
have_cmd dvc || die "dvc not found. Install with: pip install 'dvc[<remote>]'"
have_cmd flock || warn "flock not found; proceeding without interprocess lock (low risk)."

# --------- resolve remote (if not provided) -----------------------------------
if [[ -z "$REMOTE" ]]; then
  # first remote listed becomes default
  REMOTE="$(dvc remote list --quiet 2>/dev/null | awk 'NR==1{print $1}' | sed 's/://')"
fi

# Validate explicit remote exists (when specified)
if [[ -n "$REMOTE" ]]; then
  if ! dvc remote list --quiet 2>/dev/null | awk -F: '{print $1}' | grep -qx "$REMOTE"; then
    die "Remote '$REMOTE' not defined in this repo (dvc remote list)"
  fi
fi

# --------- Kaggle safety gates ------------------------------------------------
if is_kaggle; then
  if (( DO_PUSH==1 || DO_GC==1 )); then
    die "Push/GC is disabled on Kaggle (offline safety). Use --pull/--fetch/--status."
  fi
fi

# --------- exec helpers (no eval) --------------------------------------------
dry_echo() { echo "[DRY-RUN]" "$@"; }

run_cmd() {
  # run_cmd <arg1> <arg2> ...
  if (( DRY_RUN==1 )); then
    dry_echo "$@"
  else
    "$@"
  fi
}

# Retry wrapper for networked DVC ops
retry_run() {
  # retry_run <max_tries> -- <command...>
  local tries="$1"; shift
  [[ "$1" == "--" ]] && shift
  local attempt=1 backoff=2
  while true; do
    if run_cmd "$@"; then
      return 0
    fi
    if (( DRY_RUN==1 )); then
      # in dry-run we don't actually fail retries; just echo once
      return 0
    fi
    if (( attempt >= tries )); then
      return 1
    fi
    warn "Retry $attempt/$tries for: $*"
    sleep "$backoff"; backoff=$(( backoff*2 )); attempt=$(( attempt+1 ))
  done
}

# --------- locking ------------------------------------------------------------
LOCK_DIR="${REPO_ROOT}/.dvc_sync_lock"
mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/lockfile"

with_lock() {
  if have_cmd flock; then
    # shellcheck disable=SC2094
    flock -w 60 "$LOCK_FILE" "$@"
  else
    "$@"
  fi
}

# --------- ops ---------------------------------------------------------------
status_cloud() {
  local args=(status --cloud)
  [[ -n "$REMOTE" ]] && args+=(--remote "$REMOTE")
  args+=(-v)
  log "dvc ${args[*]}"
  retry_run 3 -- dvc "${args[@]}"
}

do_fetch() {
  local args=(fetch -j "$JOBS")
  [[ -n "$REMOTE" ]] && args+=(--remote "$REMOTE")
  log "dvc ${args[*]}"
  retry_run 3 -- dvc "${args[@]}"
}

do_pull() {
  local args=(pull -j "$JOBS")
  [[ -n "$REMOTE" ]] && args+=(--remote "$REMOTE")
  log "dvc ${args[*]}"
  retry_run 3 -- dvc "${args[@]}"
}

do_push() {
  local args=(push -j "$JOBS")
  [[ -n "$REMOTE" ]] && args+=(--remote "$REMOTE")
  log "dvc ${args[*]}"
  retry_run 3 -- dvc "${args[@]}"
}

do_gc() {
  local args=(gc -c)
  (( ALL_BRANCHES==1 )) && args+=(--all-branches --all-tags)
  log "dvc ${args[*]}"
  run_cmd dvc "${args[@]}"
}

do_repro() {
  local args=(repro)
  log "dvc ${args[*]}"
  run_cmd dvc "${args[@]}"
}

list_remotes() {
  log "dvc remote list"
  run_cmd dvc remote list
}

# --------- dispatch -----------------------------------------------------------
# default verb: --status
if (( DO_PUSH==0 && DO_PULL==0 && DO_FETCH==0 && DO_STATUS==0 && DO_GC==0 && DO_REPRO==0 && DO_LIST==0 )); then
  DO_STATUS=1
fi

START_TS="$(ts)"
START_EPOCH="$(date +%s)"
FAIL=0

with_lock bash -c '
  set -Eeuo pipefail
  # Export functions & vars to subshell for readability
  '"$(typeset -f log warn run_cmd retry_run status_cloud do_fetch do_pull do_push do_gc do_repro list_remotes)"'
  DO_LIST='"$DO_LIST"'
  DO_STATUS='"$DO_STATUS"'
  DO_FETCH='"$DO_FETCH"'
  DO_PULL='"$DO_PULL"'
  DO_PUSH='"$DO_PUSH"'
  DO_GC='"$DO_GC"'
  DO_REPRO='"$DO_REPRO"'

  (( DO_LIST==1 ))   && list_remotes || true
  (( DO_STATUS==1 )) && status_cloud || true
  (( DO_FETCH==1 ))  && status_cloud && do_fetch || true
  (( DO_PULL==1 ))   && do_pull || true
  (( DO_PUSH==1 ))   && do_push || true
  (( DO_GC==1 ))     && do_gc || true
  (( DO_REPRO==1 ))  && do_repro || true
' || FAIL=$?

END_TS="$(ts)"
END_EPOCH="$(date +%s)"
DUR=$(( END_EPOCH - START_EPOCH ))

# --------- summary ------------------------------------------------------------
if (( WANT_JSON==1 )); then
  # JSON summary to stdout (no jq required)
  cat <<JSON
{
  "ts_start": "${START_TS}",
  "ts_end": "${END_TS}",
  "duration_sec": ${DUR},
  "kaggle": $( is_kaggle && echo true || echo false ),
  "remote": "$(printf '%s' "${REMOTE}")",
  "jobs": ${JOBS},
  "ops": {
    "list": $( ((DO_LIST)) && echo true || echo false ),
    "status": $( ((DO_STATUS)) && echo true || echo false ),
    "fetch": $( ((DO_FETCH)) && echo true || echo false ),
    "pull": $( ((DO_PULL)) && echo true || echo false ),
    "push": $( ((DO_PUSH)) && echo true || echo false ),
    "gc": $( ((DO_GC)) && echo true || echo false ),
    "repro": $( ((DO_REPRO)) && echo true || echo false )
  },
  "dry_run": $( ((DRY_RUN)) && echo true || echo false ),
  "exit_code": ${FAIL}
}
JSON
else
  log "Done in ${DUR}s (exit=${FAIL})."
fi

exit "${FAIL}"