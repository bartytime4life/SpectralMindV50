#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — preflight_check.sh
# Run repo preflight gates (tools, lint/type sanity, config presence, CUDA probe)
# • Repo-root aware; Kaggle/CI aware
# • Fast by default (sanity gates); --full for deeper checks
# • Optional --fix for ruff/black; optional JSON summary for CI
# • Never mutates project except when --fix is set
# ------------------------------------------------------------------------------
# Usage:
#   bin/preflight_check.sh [--quick|--full] [--fix] [--json] [--quiet] [--strict]
#
# Examples:
#   bin/preflight_check.sh --quick                # fast checks (default)
#   bin/preflight_check.sh --full --json --quiet  # deep checks + CI JSON
#   bin/preflight_check.sh --fix                  # format with ruff/black if avail
#
# Exit codes:
#   0 = all selected gates passed
#   1 = at least one selected gate failed (non-strict mode)
#   2 = bad arguments / script error
#   3 = strict mode and a gate failed
# ==============================================================================

set -Eeuo pipefail

# ---------- pretty printing ----------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

usage() {
  cat <<'USAGE'
preflight_check.sh — sanity gates for SpectraMind V50

Options:
  --quick       Fast checks (tools present, Python, minimal lint) [default]
  --full        Deeper checks (pytest smoke, mypy if config, mkdocs build --strict dry-run)
  --fix         Allow auto-fixes by ruff/black if available (non-destructive beyond formatting)
  --json        Emit compact JSON summary to stdout
  --quiet       Suppress informational logs (errors still shown)
  --strict      Exit non-zero (3) if any gate fails
  -h, --help    Show this help

Notes:
  • On Kaggle, heavy/online steps are skipped automatically.
  • Gates auto-detect tools; missing optional tools are reported as SKIP, not FAIL.
USAGE
}

trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"; exit 2' ERR

# ---------- args ----------------------------------------------------------------
MODE="quick"   # quick|full
FIX=0
EMIT_JSON=0
QUIET="${QUIET:-0}"
STRICT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE="quick"; shift ;;
    --full)  MODE="full"; shift ;;
    --fix)   FIX=1; shift ;;
    --json)  EMIT_JSON=1; shift ;;
    --quiet) QUIET=1; shift ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ---------- environment awareness ----------------------------------------------
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

# ---------- gate runner helpers ------------------------------------------------
declare -a GATE_NAME=()
declare -a GATE_RESULT=() # PASS|FAIL|SKIP
declare -a GATE_INFO=()

add_result() {
  GATE_NAME+=("$1"); GATE_RESULT+=("$2"); GATE_INFO+=("$3")
  case "$2" in
    PASS) ok "$1 — PASS ${3:+($3)}" ;;
    SKIP) warn "$1 — SKIP ${3:+($3)}" ;;
    FAIL) err "$1 — FAIL ${3:+($3)}" ;;
  esac
}

need() { command -v "$1" >/dev/null 2>&1; }

fast_py_ver_check() {
  local min="$1"
  if ! need python3; then
    echo "python3 missing"; return 2
  fi
  local ver
  ver="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"
  # shellcheck disable=SC2206
  local a=(${min//./ }); local b=(${ver//./ })
  if (( b[0] > a[0] || (b[0]==a[0] && (b[1] > a[1] || (b[1]==a[1] && b[2] >= a[2])) ) )); then
    echo "$ver"; return 0
  else
    echo "$ver"; return 1
  fi
}

# ---------- Gates --------------------------------------------------------------

# Gate 1: Repository skeleton present
gate_repo_skeleton() {
  local miss=()
  for p in "pyproject.toml" "src/spectramind" "configs" "dvc.yaml" ; do
    [[ -e "$p" ]] || miss+=("$p")
  done
  if (( ${#miss[@]} == 0 )); then
    add_result "Repo skeleton" "PASS" "$ROOT"
  else
    add_result "Repo skeleton" "FAIL" "missing: ${miss[*]}"
  fi
}

# Gate 2: Python >= 3.11
gate_python_version() {
  local ver; ver="$(fast_py_ver_check "3.11.0")" || {
    local code=$?
    if [[ $code -eq 2 ]]; then
      add_result "Python >= 3.11" "FAIL" "python3 not found"
    else
      add_result "Python >= 3.11" "FAIL" "found $ver"
    fi
    return
  }
  add_result "Python >= 3.11" "PASS" "found $ver"
}

# Gate 3: Core tools present
gate_core_tools() {
  local req=( git dvc )
  local opt=( ruff black mypy pytest mkdocs mmdc syft grype )
  local miss=() ; local opt_miss=()
  for t in "${req[@]}"; do need "$t" || miss+=("$t"); done
  for t in "${opt[@]}"; do need "$t" || opt_miss+=("$t"); done
  if (( ${#miss[@]} > 0 )); then
    add_result "Core tools" "FAIL" "missing: ${miss[*]}; optional missing: ${opt_miss[*]:-none}"
  else
    add_result "Core tools" "PASS" "optional missing: ${opt_miss[*]:-none}"
  fi
}

# Gate 4: Lint/format (fast; does not modify unless --fix)
gate_lint_fast() {
  local info=()
  local any_fail=0

  if need ruff; then
    if (( FIX )); then
      if ruff check . --fix >/dev/null 2>&1; then info+=("ruff fix ok"); else any_fail=1; info+=("ruff fix failed"); fi
    else
      if ruff check . >/dev/null 2>&1; then info+=("ruff ok"); else any_fail=1; info+=("ruff issues"); fi
    fi
  else
    info+=("ruff SKIP")
  fi

  if need black; then
    if (( FIX )); then
      if black . >/dev/null 2>&1; then info+=("black reformatted/ok"); else any_fail=1; info+=("black failed"); fi
    else
      if black --check . >/dev/null 2>&1; then info+=("black ok"); else any_fail=1; info+=("black needs format"); fi
    fi
  else
    info+=("black SKIP")
  fi

  if (( any_fail==0 )); then
    add_result "Lint/format" "PASS" "${info[*]}"
  else
    add_result "Lint/format" "FAIL" "${info[*]}"
  fi
}

# Gate 5: Type check (only if mypy.ini/pyproject has tool.mypy and mypy exists)
gate_typecheck() {
  [[ "$MODE" == "full" ]] || { add_result "Type check" "SKIP" "use --full"; return; }
  if ! need mypy; then add_result "Type check" "SKIP" "mypy not installed"; return; fi
  local has_cfg=0
  grep -q '^\[tool.mypy\]' pyproject.toml 2>/dev/null && has_cfg=1
  [[ -f "mypy.ini" ]] && has_cfg=1
  [[ $has_cfg -eq 0 ]] && { add_result "Type check" "SKIP" "no mypy config"; return; }
  if mypy . >/dev/null 2>&1; then
    add_result "Type check" "PASS" "mypy clean"
  else
    add_result "Type check" "FAIL" "mypy errors"
  fi
}

# Gate 6: Pytest smoke (only if tests/ exists and pytest present)
gate_pytest_smoke() {
  [[ "$MODE" == "full" ]] || { add_result "Pytest smoke" "SKIP" "use --full"; return; }
  [[ -d "tests" ]] || { add_result "Pytest smoke" "SKIP" "tests/ missing"; return; }
  need pytest || { add_result "Pytest smoke" "SKIP" "pytest not installed"; return; }
  # Prefer a "smoke" marker/keyword if present; else run -q with short timeout
  if pytest -q -k "smoke" >/dev/null 2>&1 || pytest -q >/dev/null 2>&1; then
    add_result "Pytest smoke" "PASS" "pytest ran"
  else
    add_result "Pytest smoke" "FAIL" "pytest failures"
  fi
}

# Gate 7: Mermaid toolchain presence (renderer separate)
gate_mermaid_tooling() {
  if need mmdc; then
    add_result "Mermaid CLI" "PASS" "mmdc $(mmdc --version 2>/dev/null | tr -d '\n')"
  else
    add_result "Mermaid CLI" "SKIP" "mmdc not installed (docs rendering skipped)"
  fi
}

# Gate 8: MkDocs strict build (dry run)
gate_mkdocs_strict() {
  [[ "$MODE" == "full" ]] || { add_result "MkDocs strict" "SKIP" "use --full"; return; }
  need mkdocs || { add_result "MkDocs strict" "SKIP" "mkdocs not installed"; return; }
  if [[ -f "mkdocs.yml" || -f "mkdocs.yaml" ]]; then
    if mkdocs build --strict -q >/dev/null 2>&1; then
      add_result "MkDocs strict" "PASS" "site build ok"
    else
      add_result "MkDocs strict" "FAIL" "mkdocs errors"
    fi
  else
    add_result "MkDocs strict" "SKIP" "no mkdocs.yml"
  fi
}

# Gate 9: DVC status (offline-safe)
gate_dvc_status() {
  need dvc || { add_result "DVC status" "SKIP" "dvc not installed"; return; }
  # Avoid network; check local metafiles
  if dvc status -q >/dev/null 2>&1; then
    add_result "DVC status" "PASS" "workspace consistent"
  else
    add_result "DVC status" "FAIL" "dvc reports changes"
  fi
}

# Gate 10: CUDA probe (best-effort)
gate_cuda_probe() {
  local script="bin/detect_cuda.sh"
  if [[ -x "$script" ]]; then
    local out; out="$("$script" --json --quiet || true)"
    if [[ "$out" == "" ]]; then add_result "CUDA probe" "SKIP" "no output"; return; fi
    if grep -q '"has_cuda":1' <<<"$out"; then
      add_result "CUDA probe" "PASS" "GPU present"
    else
      add_result "CUDA probe" "SKIP" "no GPU (ok)"
    fi
  else
    add_result "CUDA probe" "SKIP" "bin/detect_cuda.sh not found"
  fi
}

# Gate 11: Submission schema presence
gate_submission_schema() {
  local have_json=0
  ls schemas/submission*.json >/dev/null 2>&1 && have_json=1
  if (( have_json )); then
    add_result "Submission schema" "PASS" "schemas present"
  else
    add_result "Submission schema" "FAIL" "schemas/submission*.json missing"
  fi
}

# ---------- run selected gates -------------------------------------------------
log "Preflight — mode=$MODE fix=$FIX ci=$IS_CI kaggle=$IS_KAGGLE root=$ROOT"

gate_repo_skeleton
gate_python_version
gate_core_tools
gate_lint_fast
gate_typecheck
gate_pytest_smoke
gate_mermaid_tooling
gate_mkdocs_strict
gate_dvc_status
gate_cuda_probe
gate_submission_schema

# ---------- summarize ----------------------------------------------------------
FAILS=0; PASSES=0; SKIPS=0
for r in "${GATE_RESULT[@]}"; do
  case "$r" in
    PASS) ((PASSES++)) ;;
    FAIL) ((FAILS++)) ;;
    SKIP) ((SKIPS++)) ;;
  esac
done

log "----------------------------------------------------------------"
if (( FAILS == 0 )); then
  ok "Preflight summary — PASS=$PASSES SKIP=$SKIPS FAIL=$FAILS"
else
  err "Preflight summary — PASS=$PASSES SKIP=$SKIPS FAIL=$FAILS"
fi

# ---------- JSON summary (compact) --------------------------------------------
if (( EMIT_JSON )); then
  printf '{'
  printf '"mode":"%s","ci":%s,"kaggle":%s,' "$MODE" "$IS_CI" "$IS_KAGGLE"
  printf '"pass":%s,"skip":%s,"fail":%s,' "$PASSES" "$SKIPS" "$FAILS"
  printf '"gates":['
  for i in "${!GATE_NAME[@]}"; do
    [[ $i -gt 0 ]] && printf ','
    # minimal escaping
    n="${GATE_NAME[$i]//\"/\\\"}"
    res="${GATE_RESULT[$i]}"
    inf="${GATE_INFO[$i]//\"/\\\"}"
    printf '{"name":"%s","result":"%s","info":"%s"}' "$n" "$res" "$inf"
  done
  printf ']}\n'
fi

# ---------- exit policy --------------------------------------------------------
if (( FAILS > 0 )); then
  if (( STRICT )); then exit 3; else exit 1; fi
fi
exit 0

