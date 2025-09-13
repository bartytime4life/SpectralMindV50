#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — apply_env_stack.sh
# Compose environment overlays (base → local/ci/kaggle) and export variables.
# • Repo-root aware (works from anywhere inside the repo)
# • Kaggle-aware (auto-picks kaggle overlay if running on Kaggle)
# • Deterministic merge order with last-one-wins semantics
# • Optional write-out to .env (for IDEs, Docker, CI)
# • Optional JSON summary to stdout
# • Dry-run mode prints what would be exported
# ------------------------------------------------------------------------------
# Usage:
#   bin/apply_env_stack.sh [--preset {local|ci|kaggle}] [--extra foo,bar] \
#       [--write .env.generated] [--json] [--quiet] [--dry-run] [-h|--help]
#
# Examples:
#   bin/apply_env_stack.sh --preset local --write .env.generated
#   bin/apply_env_stack.sh --preset ci --extra dev,secrets --json
#   bin/apply_env_stack.sh --preset kaggle --quiet
#
# Overlays search path (in order, only if present):
#   env/.env.base
#   env/.env.<preset>
#   env/.env.<extra1>, env/.env.<extra2>, ...
#   .env.local (last, optional developer overrides; ignored on Kaggle/CI)
#
# Conventions:
#   • Lines are KEY=VALUE (no spaces around '='). '#' starts a comment.
#   • Values may contain ${VAR} which will be expanded during merge.
#   • Keys limited to [A-Z0-9_]; invalid lines are ignored with a warning.
# ==============================================================================

set -Eeuo pipefail

# -------- Pretty printing ------------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

# -------- Usage ----------------------------------------------------------------
usage() {
  cat <<'USAGE'
apply_env_stack.sh — compose env overlays for SpectraMind V50

Options:
  --preset {local|ci|kaggle}  Choose primary overlay (defaults: auto-detect).
  --extra name1,name2         Additional overlays in env/.env.<name>.
  --write PATH                Write merged variables to PATH as KEY=VALUE.
  --json                      Emit JSON summary of resolved variables to stdout.
  --quiet                     Suppress informational logs.
  --dry-run                   Print actions without exporting to current shell.
  -h, --help                  Show this help.

Auto-detection:
  • Kaggle if /kaggle exists or KAGGLE_KERNEL_RUN_TYPE is set ⇒ preset=kaggle
  • CI if CI=true ⇒ preset=ci
  • Else ⇒ preset=local

Merge order (last wins):
  env/.env.base → env/.env.<preset> → env/.env.<extra...> → .env.local*

  * .env.local is skipped on Kaggle/CI to prevent accidental leakage.

Examples:
  bin/apply_env_stack.sh --preset local --write .env.generated
  bin/apply_env_stack.sh --extra secrets,dev --json
USAGE
}

# -------- Fail trap with line numbers -----------------------------------------
trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"' ERR

# -------- Resolve repo root ----------------------------------------------------
repo_root() {
  # Try git first, then ascend until we find markers
  if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    printf "%s" "$git_root"; return 0
  fi
  local d="$PWD"
  while [[ "$d" != "/" ]]; do
    if [[ -e "$d/pyproject.toml" || -e "$d/dvc.yaml" || -d "$d/.git" ]]; then
      printf "%s" "$d"; return 0
    fi
    d="$(dirname "$d")"
  done
  err "Unable to locate repo root (looked for pyproject.toml/dvc.yaml/.git)"
  exit 2
}

ROOT="$(repo_root)"
cd "$ROOT"

# -------- Defaults / args ------------------------------------------------------
PRESET=""
EXTRA=""
WRITE_PATH=""
EMIT_JSON=0
QUIET="${QUIET:-0}"
DRYRUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)   PRESET="${2:-}"; shift 2 ;;
    --extra)    EXTRA="${2:-}"; shift 2 ;;
    --write)    WRITE_PATH="${2:-}"; shift 2 ;;
    --json)     EMIT_JSON=1; shift ;;
    --quiet)    QUIET=1; shift ;;
    --dry-run)  DRYRUN=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# Auto-detect preset if not provided
if [[ -z "${PRESET:-}" ]]; then
  if [[ -d "/kaggle" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
    PRESET="kaggle"
  elif [[ "${CI:-}" == "true" || "${GITHUB_ACTIONS:-}" == "true" ]]; then
    PRESET="ci"
  else
    PRESET="local"
  fi
fi

case "$PRESET" in
  local|ci|kaggle) : ;;
  *) err "Invalid --preset '$PRESET' (expected local|ci|kaggle)"; exit 2 ;;
esac

# Parse extras into array
IFS=',' read -r -a EXTRA_ARR <<< "${EXTRA:-}"

ENV_DIR="$ROOT/env"
declare -a CANDIDATES=()
CANDIDATES+=("$ENV_DIR/.env.base")
CANDIDATES+=("$ENV_DIR/.env.${PRESET}")

for name in "${EXTRA_ARR[@]}"; do
  [[ -n "$name" ]] && CANDIDATES+=("$ENV_DIR/.env.${name}")
done

# Only include .env.local for local preset and when not CI/Kaggle
if [[ "$PRESET" == "local" && "${CI:-}" != "true" && -z "${KAGGLE_KERNEL_RUN_TYPE:-}" && ! -d "/kaggle" ]]; then
  CANDIDATES+=("$ROOT/.env.local")
fi

# -------- Helpers to parse .env fragments -------------------------------------
#  • Allow KEY=VALUE (no spaces around '=')
#  • Ignore empty lines and comments
#  • Only accept [A-Z0-9_]+ keys
#  • Expand ${VAR} in VALUE using already merged map
declare -A MERGED=()

is_valid_key() { [[ "$1" =~ ^[A-Z0-9_]+$ ]]; }

expand_value() {
  local val="$1"
  # Expand ${VAR} using current MERGED or process env
  # shellcheck disable=SC2016
  while [[ "$val" =~ (\$\{[A-Z0-9_]+\}) ]]; do
    local ref="${BASH_REMATCH[1]}"
    local key="${ref:2:${#ref}-3}"
    local repl="${MERGED[$key]:-${!key:-}}"
    val="${val//$ref/$repl}"
  done
  printf "%s" "$val"
}

merge_file() {
  local f="$1"
  [[ -f "$f" ]] || { warn "Skip missing env fragment: $f"; return 0; }
  log "Merging: $f"
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    # trim leading/trailing spaces
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    if [[ "$line" =~ ^([A-Z0-9_]+)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
      if ! is_valid_key "$key"; then
        warn "Invalid key (ignored): $key in $f"
        continue
      fi
      # Strip surrounding quotes if present
      if [[ "$val" =~ ^\".*\"$ ]]; then val="${val:1:${#val}-2}"; fi
      if [[ "$val" =~ ^\'.*\'$ ]]; then val="${val:1:${#val}-2}"; fi
      val="$(expand_value "$val")"
      MERGED["$key"]="$val"
    else
      warn "Invalid line (ignored): $line"
    fi
  done < "$f"
}

# -------- Merge all fragments --------------------------------------------------
for f in "${CANDIDATES[@]}"; do
  merge_file "$f"
done

# -------- Always-on safe defaults (only if not already set) -------------------
defaults=(
  "PYTHONUTF8=1"
  "PIP_DISABLE_PIP_VERSION_CHECK=1"
  "HYDRA_FULL_ERROR=1"
  "NUMEXPR_MAX_THREADS=8"
)
for kv in "${defaults[@]}"; do
  k="${kv%%=*}"; v="${kv#*=}"
  [[ -n "${MERGED[$k]:-}" ]] || MERGED["$k"]="$v"
done

# Kaggle-specific safe knobs
if [[ "$PRESET" == "kaggle" ]]; then
  MERGED["SM_OFFLINE"]="1"
  MERGED["DVC_OFFLINE"]="1"
  MERGED["PIP_NO_CACHE_DIR"]="1"
fi

# CI-specific hints
if [[ "$PRESET" == "ci" ]]; then
  MERGED["PYTHONHASHSEED"]="${MERGED["PYTHONHASHSEED"]:-0}"
fi

# -------- Write-out merged .env if requested ----------------------------------
if [[ -n "${WRITE_PATH:-}" ]]; then
  out="$(realpath -m "$WRITE_PATH")"
  log "Writing merged env → $out"
  {
    echo "# Autogenerated by bin/apply_env_stack.sh on $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "# Preset=${PRESET}  Extras=${EXTRA:-<none>}  Root=${ROOT}"
    for k in "${!MERGED[@]}"; do
      printf "%s=%s\n" "$k" "${MERGED[$k]}"
    done | LC_ALL=C sort
  } > "$out"
  ok "Wrote $out"
fi

# -------- Export (unless dry-run) ---------------------------------------------
if [[ "$DRYRUN" -eq 1 ]]; then
  log "Dry-run: not exporting to current shell. Resolved variables:"
  for k in "${!MERGED[@]}"; do printf "%s=%q\n" "$k" "${MERGED[$k]}"; done | LC_ALL=C sort
else
  for k in "${!MERGED[@]}"; do export "$k=${MERGED[$k]}"; done
  ok "Exported $(printf "%s\n" "${!MERGED[@]}" | wc -l | tr -d ' ') variables (preset=$PRESET)"
fi

# -------- Optional JSON summary ------------------------------------------------
if [[ "$EMIT_JSON" -eq 1 ]]; then
  # Minimal JSON emitter (no jq dependency)
  printf '{'
  printf '"preset":"%s","extras":[' "$PRESET"
  first=1
  for name in "${EXTRA_ARR[@]}"; do
    [[ -z "$name" ]] && continue
    if [[ $first -eq 1 ]]; then first=0; else printf ','; fi
    printf '"%s"' "$name"
  done
  printf '],"vars":{'
  keys_sorted=($(printf "%s\n" "${!MERGED[@]}" | LC_ALL=C sort))
  for i in "${!keys_sorted[@]}"; do
    k="${keys_sorted[$i]}"; v="${MERGED[$k]}"
    [[ $i -gt 0 ]] && printf ','
    # Escape backslashes and quotes
    esc="${v//\\/\\\\}"; esc="${esc//\"/\\\"}"
    printf '"%s":"%s"' "$k" "$esc"
  done
  printf '}}'
  [[ "$DRYRUN" -eq 1 || "$QUIET" -eq 1 ]] || echo
fi

exit 0

