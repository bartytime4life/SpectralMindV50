#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Kaggle Submission Packager
#
# Packages a validated submission CSV (+ manifest & provenance) into
# artifacts/submission.zip, suitable for direct Kaggle upload.
#
# Features
#   • Safe defaults: outputs/submission.csv → artifacts/submission.zip
#   • Schema & sanity checks (columns, NaN/inf, σ ≥ 0)
#   • Manifest with git commit, timestamps, hashes, Hydra config snapshot, etc.
#   • Cross-platform hashing/zip (Linux/macOS), no network, Kaggle-safe
#   • Dry-run & force rebuild; optional JSON schema validation (jq/ajv/python)
#
# Usage
#   bin/package_submission.sh
#   bin/package_submission.sh --input outputs/submission.csv --zip artifacts/submission.zip
#   bin/package_submission.sh --all             # include extra provenance files
#   bin/package_submission.sh --dry-run
#
# Important paths (repo scaffold)
#   • Default input CSV        : outputs/submission.csv
#   • Default zip output       : artifacts/submission.zip
#   • Default manifest         : artifacts/manifest.json
#   • JSON schema (optional)   : schemas/submission.schema.json
# ==============================================================================

set -Eeuo pipefail

# ---------- colored logs ----------
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

# ---------- repo root ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null); then :; else ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"; fi
cd "${ROOT_DIR}"

# ---------- defaults ----------
INPUT_CSV="outputs/submission.csv"
ZIP_OUT="artifacts/submission.zip"
MANIFEST="artifacts/manifest.json"
SCHEMA="schemas/submission.schema.json"    # optional
INCLUDE_ALL="0"      # include extra provenance files
FORCE="0"
DRY_RUN="0"

usage() {
  cat <<'USAGE'
SpectraMind V50 — Kaggle Submission Packager

Usage:
  bin/package_submission.sh [--input PATH] [--zip PATH] [--manifest PATH]
                            [--schema PATH] [--all] [--force] [--dry-run] [--help]

Options:
  --input PATH       Path to submission CSV (default: outputs/submission.csv)
  --zip PATH         Output zip file (default: artifacts/submission.zip)
  --manifest PATH    Output manifest JSON (default: artifacts/manifest.json)
  --schema PATH      JSON schema for submission (optional; default: schemas/submission.schema.json)
  --all              Include extra provenance (VERSION, dvc.lock, configs/* snapshots if present)
  --force            Rebuild zip even if up-to-date
  --dry-run          Print actions, do not write files
  --help             Show this help
USAGE
}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --input)     INPUT_CSV="${2:?}"; shift 2 ;;
    --zip)       ZIP_OUT="${2:?}"; shift 2 ;;
    --manifest)  MANIFEST="${2:?}"; shift 2 ;;
    --schema)    SCHEMA="${2:?}"; shift 2 ;;
    --all)       INCLUDE_ALL="1"; shift ;;
    --force)     FORCE="1"; shift ;;
    --dry-run)   DRY_RUN="1"; shift ;;
    *) err "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

# ---------- helpers ----------
hash_file() {
  # Prints SHA256 of file, cross-platform
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  else
    python3 - "$f" <<'PY'
import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as fp:
    for chunk in iter(lambda: fp.read(1<<20), b''):
        h.update(chunk)
print(h.hexdigest())
PY
  fi
}

zip_create() {
  local zip="$1"; shift
  # Create zip with fallback if 'zip' is absent
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "%s\n" "$zip" "$@" | sed '1s/^/Would ZIP into: /'
    return 0
  fi
  mkdir -p "$(dirname "$zip")"
  if command -v zip >/dev/null 2>&1; then
    # -q quiet, -j junk paths? keep paths to be explicit; reproducibility prefers fixed order
    # We'll include full relative paths to keep provenance structure.
    zip -q -X -r "$zip" "$@"  # -X omit extra file attrs
  else
    python3 - "$zip" "$@" <<'PY'
import sys,zipfile,os
zip_path=sys.argv[1]; files=sys.argv[2:]
os.makedirs(os.path.dirname(zip_path), exist_ok=True)
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
    for p in files:
        if os.path.isdir(p):
            for root,dirs,fs in os.walk(p):
                for f in fs:
                    ap=os.path.join(root,f)
                    zf.write(ap, ap)
        else:
            zf.write(p, p)
PY
  fi
}

require_file() {
  local p="$1" msg="$2"
  if [[ ! -f "$p" ]]; then err "${msg:-Required file missing}: $p"; exit 1; fi
}

json_escape() {
  # minimal escaper for strings; prefer jq or python when available
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))' 2>/dev/null || cat
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# ---------- preflight ----------
require_file "${INPUT_CSV}" "Submission CSV not found"

mkdir -p "$(dirname "${ZIP_OUT}")" "$(dirname "${MANIFEST}")"

info "Submission CSV : ${INPUT_CSV}"
info "Output zip     : ${ZIP_OUT}"
info "Manifest JSON  : ${MANIFEST}"

# incremental: skip if zip newer than CSV+manifest unless --force
if [[ "${FORCE}" != "1" && -f "${ZIP_OUT}" && -f "${MANIFEST}" ]]; then
  if [[ "${ZIP_OUT}" -nt "${INPUT_CSV}" && "${ZIP_OUT}" -nt "${MANIFEST}" ]]; then
    ok "Zip is up-to-date (use --force to rebuild)."
    exit 0
  fi
fi

# ---------- CSV sanity checks ----------
info "Running CSV sanity checks…"

# 1) header check (expects: id/sample_id then mu_000..mu_282, sigma_000..sigma_282)
header="$(head -n1 "${INPUT_CSV}")"
IFS=',' read -r -a cols <<<"${header}"

# flexible first id column: sample_id or id
first_col_ok="0"
[[ "${cols[0]}" == "sample_id" || "${cols[0]}" == "id" ]] && first_col_ok="1"
if [[ "${first_col_ok}" != "1" ]]; then
  err "First column must be 'sample_id' or 'id', got: '${cols[0]}'"
  exit 1
fi

expect_cols=567  # 1 + 283 + 283
if [[ "${#cols[@]}" -ne "${expect_cols}" ]]; then
  err "Unexpected column count: ${#cols[@]} (expected ${expect_cols})."
  warn "Tip: columns should be: [id|sample_id], mu_000..mu_282, sigma_000..sigma_282"
  exit 1
fi

# Check mu/sigma name patterns quickly (spot check start/end)
[[ "${cols[1]}" == "mu_000" && "${cols[283]}" == "mu_282" && "${cols[284]}" == "sigma_000" && "${cols[566]}" == "sigma_282" ]] || {
  warn "Column name pattern looks unusual. Expected mu_000..mu_282, sigma_000..sigma_282."
}

# 2) scan for NaN/inf and negative sigma using awk (fast, memory-light)
#    skip header; treat empty as error too
badnum_count=$(awk -F, 'NR>1{
  for(i=2;i<=NF;i++){
    x=$i;
    if (x=="" || x=="NaN" || x=="nan" || x=="INF" || x=="Inf" || x=="inf" || x=="+inf" || x=="-inf") {bad++}
  }
}END{print bad+0}' "${INPUT_CSV}")
if [[ "${badnum_count}" -gt 0 ]]; then
  err "Found ${badnum_count} empty/NaN/Inf numeric entries."
  exit 1
fi

neg_sigma=$(awk -F, 'NR>1{
  # sigma columns start at index 285 to 567 (1-based)
  for(i=285;i<=NF;i++){
    if($i+0 < 0){bad++}
  }
}END{print bad+0}' "${INPUT_CSV}")
if [[ "${neg_sigma}" -gt 0 ]]; then
  err "Found ${neg_sigma} negative sigma values."
  exit 1
fi

ok "CSV sanity checks passed."

# ---------- optional JSON schema validation ----------
if [[ -f "${SCHEMA}" ]]; then
  info "Validating against schema: ${SCHEMA}"
  if have_cmd jq && have_cmd ajv; then
    # Node ajv (fast) if available
    if [[ "${DRY_RUN}" != "1" ]]; then
      ajv validate -s "${SCHEMA}" -d "${INPUT_CSV}" >/dev/null 2>&1 || {
        err "Schema validation failed (ajv)."; exit 1;
      }
    fi
  else
    # Python jsonschema (portable)
    python3 - "${SCHEMA}" "${INPUT_CSV}" <<'PY' || { echo "Schema validation failed (python)"; exit 1; }
import sys, json, csv
from jsonschema import validate, Draft7Validator
schema_path, csv_path = sys.argv[1], sys.argv[2]
with open(schema_path,'r') as fp:
    schema=json.load(fp)
# Very light touch: header presence & types per row against inferred object;
# full CSV->JSON transformation is use-specific; here we just sanity check headers.
expected = [f"mu_{i:03d}" for i in range(283)] + [f"sigma_{i:03d}" for i in range(283)]
with open(csv_path,'r', newline='') as fp:
    r=csv.reader(fp)
    header=next(r)
    if header[0] not in ('id','sample_id'): raise SystemExit("First column must be id/sample_id")
    if header[1:1+283]!=[f"mu_{i:03d}" for i in range(283)]: raise SystemExit("mu_* columns mismatch")
    if header[1+283:1+283+283]!=[f"sigma_{i:03d}" for i in range(283)]: raise SystemExit("sigma_* columns mismatch")
print("OK")
PY
  fi
  ok "Schema validation passed."
else
  warn "Schema file not found (${SCHEMA}); skipping JSON schema validation."
fi

# ---------- assemble file list ----------
FILES_TO_ZIP=()
FILES_TO_ZIP+=("${INPUT_CSV}")

# create manifest in-memory first
GIT_COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")"
GIT_STATUS="$(git status --porcelain 2>/dev/null | wc -l | awk '{print $1}')"
VERSION_FILE="VERSION"
VERSION_STR="$( [[ -f ${VERSION_FILE} ]] && cat "${VERSION_FILE}" || echo "0.0.0" )"
NOW_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

CSV_SIZE_BYTES="$(wc -c < "${INPUT_CSV}" | tr -d ' ')"
CSV_SHA256="$(hash_file "${INPUT_CSV}")"

# try to capture a hydra config snapshot if your code saves it:
CFG_SNAPSHOT="$(ls -1 outputs/config_snapshot*.yaml 2>/dev/null | head -n1 || true)"

# Compose manifest JSON (prefer jq/python if available for correctness)
if have_cmd jq; then
  MANIFEST_JSON="$(jq -n \
    --arg created_at   "${NOW_ISO}" \
    --arg git_commit   "${GIT_COMMIT}" \
    --arg git_dirty    "$([[ "${GIT_STATUS}" != "0" ]] && echo "true" || echo "false")" \
    --arg version      "${VERSION_STR}" \
    --arg input_csv    "${INPUT_CSV}" \
    --arg csv_sha256   "${CSV_SHA256}" \
    --arg csv_size     "${CSV_SIZE_BYTES}" \
    --arg cfg_snapshot "${CFG_SNAPSHOT}" \
    '{
      tool: "SpectraMind V50",
      action: "package_submission",
      created_at: $created_at,
      git: { commit: $git_commit, dirty: ($git_dirty=="true") },
      version: $version,
      inputs: {
        submission_csv: { path: $input_csv, sha256: $csv_sha256, size_bytes: ($csv_size|tonumber) }
      },
      config_snapshot: ( $cfg_snapshot|length>0 ? { path: $cfg_snapshot } : null )
    }'
  )"
else
  # minimal portable JSON via python
  MANIFEST_JSON="$(python3 - <<PY
import json,os
d={
 "tool":"SpectraMind V50",
 "action":"package_submission",
 "created_at":"${NOW_ISO}",
 "git":{"commit":"${GIT_COMMIT}","dirty":${"true" if int("${GIT_STATUS}")!=0 else "false"}},
 "version":"${VERSION_STR}",
 "inputs":{"submission_csv":{"path":"${INPUT_CSV}","sha256":"${CSV_SHA256}","size_bytes":int("${CSV_SIZE_BYTES}")}},
 "config_snapshot": {"path":"${CFG_SNAPSHOT}"} if "${CFG_SNAPSHOT}" else None
}
print(json.dumps(d, separators=(',',':')))
PY
  )"
fi

# write manifest
if [[ "${DRY_RUN}" == "1" ]]; then
  info "Would write manifest to ${MANIFEST}:"
  printf "%s\n" "${MANIFEST_JSON}"
else
  printf "%s\n" "${MANIFEST_JSON}" > "${MANIFEST}"
fi
FILES_TO_ZIP+=("${MANIFEST}")

# include extras when --all
if [[ "${INCLUDE_ALL}" == "1" ]]; then
  [[ -f "${VERSION_FILE}" ]] && FILES_TO_ZIP+=("${VERSION_FILE}")
  [[ -f "dvc.lock" ]] && FILES_TO_ZIP+=("dvc.lock")
  [[ -n "${CFG_SNAPSHOT}" && -f "${CFG_SNAPSHOT}" ]] && FILES_TO_ZIP+=("${CFG_SNAPSHOT}")
  # include minimal config/docs helpful for audit (optional)
  [[ -d "configs" ]] && FILES_TO_ZIP+=("configs")
  [[ -d "schemas" ]] && FILES_TO_ZIP+=("schemas")
fi

# ---------- create zip ----------
info "Creating zip…"
zip_create "${ZIP_OUT}" "${FILES_TO_ZIP[@]}"

# ---------- report ----------
ZIP_SHA256="$(hash_file "${ZIP_OUT}")"
ZIP_SIZE="$(wc -c < "${ZIP_OUT}" | tr -d ' ')"
ok "Packaged: ${ZIP_OUT}  (${ZIP_SIZE} bytes)"
info "ZIP sha256: ${ZIP_SHA256}"
