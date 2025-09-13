#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — generate_sbom.sh
# Create Software Bill of Materials (SBOM) for the repo (dir: .), optionally for
# a container image, in SPDX and CycloneDX formats using Syft. Optionally run
# Grype for vulnerability scanning. Kaggle/CI/offline aware.
#
# Outputs (by default under outputs/sbom/):
#   • sbom-spdx.json           (SPDX JSON)
#   • sbom-cyclonedx.json      (CycloneDX JSON)
#   • vulns-grype.json         (Grype results; optional)
#   • _meta.json               (tool/run metadata)
#
# Usage:
#   bin/generate_sbom.sh
#   bin/generate_sbom.sh --out-dir outputs/sbom --formats spdx-json,cyclonedx-json
#   bin/generate_sbom.sh --image ghcr.io/org/app:latest --with-vulns
#   bin/generate_sbom.sh --json --quiet
#
# Options:
#   --out-dir PATH          Output directory (default: outputs/sbom)
#   --source PATH           Source directory to scan (default: repo root)
#   --image NAME[:TAG]      Additionally scan container image (requires local pull)
#   --formats LIST          Comma list of formats: spdx-json,cyclonedx-json (default both)
#   --name STR              Component name override (default: repo dir name)
#   --version STR           Version override (default: from VERSION or git describe)
#   --with-vulns            Run Grype vulnerability scan (offline-safe)
#   --no-vulns              Force skip Grype
#   --offline               Force offline mode (implies GRYPE_DB_AUTO_UPDATE=0)
#   --json                  Emit a JSON summary to stdout
#   --quiet                 Suppress informational logs
#   --dry-run               Print actions without executing
#   -h, --help              Show help
#
# Tooling:
#   • Requires: syft (https://github.com/anchore/syft)
#   • Optional: grype (https://github.com/anchore/grype), jq (for pretty JSON)
#
# Exit codes:
#   0 = success; 2 = bad args; 3 = required tool missing; 4 = failures during run
# ==============================================================================

set -Eeuo pipefail

# ---------- log helpers --------------------------------------------------------
log()  { [[ "${QUIET:-0}" == "1" ]] || printf "%b\n" "$*"; }
err()  { printf "\e[31m[ERR]\e[0m %b\n" "$*" >&2; }
ok()   { [[ "${QUIET:-0}" == "1" ]] || printf "\e[32m[OK]\e[0m %b\n" "$*"; }
warn() { [[ "${QUIET:-0}" == "1" ]] || printf "\e[33m[WARN]\e[0m %b\n" "$*"; }

usage() {
  sed -n '1,120p' "${BASH_SOURCE[0]}" | sed -n '1,120p' | awk '/^# ====/{flag=1;next}/^set -Eeuo/{flag=0}flag' | sed 's/^# \{0,1\}//'
}

trap 'err "Failure at ${BASH_SOURCE[0]}:${LINENO} (exit=$?)"; exit 4' ERR

# ---------- defaults / args ----------------------------------------------------
OUT_DIR="outputs/sbom"
SOURCE=""
IMAGE=""
FORMATS="spdx-json,cyclonedx-json"
NAME=""
VERSION=""
DO_VULNS="auto"   # auto|yes|no
EMIT_JSON=0
QUIET="${QUIET:-0}"
DRYRUN=0
FORCE_OFFLINE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)   OUT_DIR="${2:-}"; shift 2 ;;
    --source)    SOURCE="${2:-}"; shift 2 ;;
    --image)     IMAGE="${2:-}"; shift 2 ;;
    --formats)   FORMATS="${2:-}"; shift 2 ;;
    --name)      NAME="${2:-}"; shift 2 ;;
    --version)   VERSION="${2:-}"; shift 2 ;;
    --with-vulns) DO_VULNS="yes"; shift ;;
    --no-vulns)   DO_VULNS="no"; shift ;;
    --offline)    FORCE_OFFLINE=1; shift ;;
    --json)       EMIT_JSON=1; shift ;;
    --quiet)      QUIET=1; shift ;;
    --dry-run)    DRYRUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ---------- repo root detection ------------------------------------------------
repo_root() {
  if git_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"; then
    [[ -n "$git_root" ]] && { printf "%s" "$git_root"; return 0; }
  fi
  local d="$PWD"
  while [[ "$d" != "/" ]]; do
    if [[ -e "$d/pyproject.toml" || -e "$d/dvc.yaml" || -d "$d/.git" ]]; then
      printf "%s" "$d"; return 0
    fi
    d="$(dirname "$d")"
  done
  printf "%s" "$PWD"
}
ROOT="$(repo_root)"

# ---------- source path, name, version ----------------------------------------
if [[ -z "$SOURCE" ]]; then
  SOURCE="$ROOT"
fi
if [[ -z "$NAME" ]]; then
  NAME="$(basename "$ROOT")"
fi
if [[ -z "$VERSION" ]]; then
  if [[ -f "$ROOT/VERSION" ]]; then
    VERSION="$(sed -n '1p' "$ROOT/VERSION" | tr -d '[:space:]')"
  else
    VERSION="$(git -C "$ROOT" describe --tags --always 2>/dev/null || echo "0.0.0")"
  fi
fi

# ---------- environment & offline knobs ---------------------------------------
IS_KAGGLE=0
[[ -d "/kaggle" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] && IS_KAGGLE=1
IS_CI=0
[[ "${CI:-}" == "true" || "${GITHUB_ACTIONS:-}" == "true" ]] && IS_CI=1
OFFLINE=$(( FORCE_OFFLINE || IS_KAGGLE ))

# Do not auto-update vulnerability DBs offline
export SYFT_CHECK_FOR_APP_UPDATE="false"
export GRYPE_CHECK_FOR_APP_UPDATE="false"
[[ "$OFFLINE" -eq 1 ]] && export GRYPE_DB_AUTO_UPDATE="0"

# ---------- tool availability --------------------------------------------------
need() { command -v "$1" >/dev/null 2>&1 || { err "Missing required tool: $1"; exit 3; }; }
need syft
if [[ "${DO_VULNS}" == "yes" ]]; then
  need grype
elif [[ "${DO_VULNS}" == "auto" ]]; then
  if command -v grype >/dev/null 2>&1; then DO_VULNS="yes"; else DO_VULNS="no"; fi
fi

# ---------- output dir ---------------------------------------------------------
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"

# ---------- parse formats ------------------------------------------------------
IFS=',' read -r -a F_ARR <<< "$FORMATS"
declare -A F_SET=()
for f in "${F_ARR[@]}"; do
  case "$f" in
    spdx-json|cyclonedx-json) F_SET["$f"]=1 ;;
    *) warn "Unknown format '$f' (ignored)";;
  esac
done
[[ ${#F_SET[@]} -eq 0 ]] && { err "No valid formats selected"; exit 2; }

# ---------- syft invocation helper --------------------------------------------
syft_dir() {
  local src="$1" fmt="$2" out="$3"
  if [[ "$DRYRUN" -eq 1 ]]; then
    log "[dry-run] syft dir:$src -o $fmt --file $out --select-catalogers all --source-name \"$NAME\" --source-version \"$VERSION\""
    return 0
  fi
  syft "dir:$src" -o "$fmt" --file "$out" \
    --select-catalogers all \
    --source-name "$NAME" --source-version "$VERSION"
}

syft_image() {
  local img="$1" fmt="$2" out="$3"
  if [[ "$DRYRUN" -eq 1 ]]; then
    log "[dry-run] syft $img -o $fmt --file $out --select-catalogers all --source-name \"$NAME\" --source-version \"$VERSION\""
    return 0
  fi
  syft "$img" -o "$fmt" --file "$out" \
    --select-catalogers all \
    --source-name "$NAME" --source-version "$VERSION"
}

# ---------- grype helper -------------------------------------------------------
run_grype() {
  local subject="$1" out="$2"
  local extra=()
  [[ "$OFFLINE" -eq 1 ]] && extra+=( "--only-fixed=false" )
  if [[ "$DRYRUN" -eq 1 ]]; then
    log "[dry-run] grype $subject -o json ${extra[*]} > $out"
    return 0
  fi
  # Avoid DB update offline
  if [[ "$OFFLINE" -eq 1 ]]; then
    GRYPE_DB_AUTO_UPDATE=0 grype "$subject" -o json "${extra[@]}" > "$out"
  else
    grype "$subject" -o json "${extra[@]}" > "$out"
  fi
}

# ---------- generate SBOM(s) for directory ------------------------------------
SPDX_PATH="$OUT_DIR/sbom-spdx.json"
CYCLO_PATH="$OUT_DIR/sbom-cyclonedx.json"
META_PATH="$OUT_DIR/_meta.json"
VULN_PATH="$OUT_DIR/vulns-grype.json"
IMG_SPDX=""
IMG_CYCL=""

log "SBOM generation for: $NAME @ $VERSION"
log "Source: $SOURCE"
log "Output dir: $OUT_DIR"

if [[ -n "${F_SET[spdx-json]:-}" ]]; then
  syft_dir "$SOURCE" "spdx-json" "$SPDX_PATH"
  ok "Wrote $SPDX_PATH"
fi

if [[ -n "${F_SET[cyclonedx-json]:-}" ]]; then
  syft_dir "$SOURCE" "cyclonedx-json" "$CYCLO_PATH"
  ok "Wrote $CYCLO_PATH"
fi

# ---------- optional image scan ------------------------------------------------
if [[ -n "$IMAGE" ]]; then
  log "Image scan requested: $IMAGE"
  if [[ -n "${F_SET[spdx-json]:-}" ]]; then
    IMG_SPDX="$OUT_DIR/sbom-image-spdx.json"
    syft_image "$IMAGE" "spdx-json" "$IMG_SPDX"
    ok "Wrote $IMG_SPDX"
  fi
  if [[ -n "${F_SET[cyclonedx-json]:-}" ]]; then
    IMG_CYCL="$OUT_DIR/sbom-image-cyclonedx.json"
    syft_image "$IMAGE" "cyclonedx-json" "$IMG_CYCL"
    ok "Wrote $IMG_CYCL"
  fi
fi

# ---------- vulnerabilities (grype) -------------------------------------------
if [[ "$DO_VULNS" == "yes" ]]; then
  # Prefer image subject if provided; else directory
  if [[ -n "$IMAGE" ]]; then
    run_grype "$IMAGE" "$VULN_PATH"
  else
    run_grype "dir:$SOURCE" "$VULN_PATH"
  fi
  ok "Wrote $VULN_PATH"
else
  VULN_PATH=""
  log "Vulnerability scan skipped"
fi

# ---------- meta summary -------------------------------------------------------
NOW_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
if [[ "$DRYRUN" -eq 1 ]]; then
  log "[dry-run] Would write metadata to $META_PATH"
else
  {
    printf '{'
    printf '"name":%q,'    "$NAME"
    printf '"version":%q,' "$VERSION"
    printf '"source":%q,'  "$SOURCE"
    printf '"image":%q,'   "$IMAGE"
    printf '"formats":%q,' "$FORMATS"
    printf '"paths":{'
      printf '"spdx":%q,'        "${SPDX_PATH:-}"
      printf '"cyclonedx":%q,'   "${CYCLO_PATH:-}"
      printf '"image_spdx":%q,'  "${IMG_SPDX:-}"
      printf '"image_cyclonedx":%q,' "${IMG_CYCL:-}"
      printf '"vulns":%q'        "${VULN_PATH:-}"
    printf '},'
    printf '"env":{"ci":%s,"kaggle":%s,"offline":%s},' "$IS_CI" "$IS_KAGGLE" "$OFFLINE"
    printf '"tooling":{"syft":%q,"grype":%q},' "$(syft version 2>/dev/null | head -n1 | awk '{print $3}')" "$(command -v grype >/dev/null 2>&1 && grype version 2>/dev/null | head -n1 | awk '{print $3}' || echo "absent")"
    printf '"created":"%s"' "$NOW_UTC"
    printf '}\n'
  } > "$META_PATH"
  ok "Wrote $META_PATH"
fi

# ---------- CI-friendly JSON summary ------------------------------------------
if [[ "$EMIT_JSON" -eq 1 ]]; then
  esc() { printf "%s" "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  printf '{'
  printf '"ok":true,'
  printf '"name":"%s",' "$(esc "$NAME")"
  printf '"version":"%s",' "$(esc "$VERSION")"
  printf '"spdx":"%s",' "$(esc "${SPDX_PATH:-}")"
  printf '"cyclonedx":"%s",' "$(esc "${CYCLO_PATH:-}")"
  printf '"image_spdx":"%s",' "$(esc "${IMG_SPDX:-}")"
  printf '"image_cyclonedx":"%s",' "$(esc "${IMG_CYCL:-}")"
  printf '"vulns":"%s",' "$(esc "${VULN_PATH:-}")"
  printf '"out_dir":"%s",' "$(esc "$OUT_DIR")"
  printf '"created":"%s",' "$(esc "$NOW_UTC")"
  printf '"offline":%s' "$OFFLINE"
  printf '}\n'
fi

ok "SBOM generation complete."
exit 0

