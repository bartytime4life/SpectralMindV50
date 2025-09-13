#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — Submission Validator
# ==============================================================================
# Validates a submission CSV against:
#   • Canonical header order (schemas/submission_header.csv)
#   • Frictionless Table Schema (schemas/submission.tableschema.sample_id.json)
#   • Structural/semantic checks: unique sample_id, numeric casts, NaNs/infs
#   • Physics sanity: mu_* >= 0, sigma_* > 0 (warn/error in --strict)
#
# Usage:
#   bin/validate_submission.sh --csv outputs/submission/submission.csv
#   bin/validate_submission.sh --csv path.csv --strict --json
#
# Exits non-zero on failures. Prints a short report (or JSON with --json).
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------
CSV=""
SCHEMA="${SCHEMA:-schemas/submission.tableschema.sample_id.json}"
HEADER="${HEADER:-schemas/submission_header.csv}"
STRICT=0
JSON=0
MAX_ROWS="${MAX_ROWS:-}"   # optional limit (e.g., for spot checks)
PYBIN="${PYBIN:-python}"   # override if needed

# ------------------------------------------------------------------------------
# Args
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)    CSV="$2"; shift 2 ;;
    --schema) SCHEMA="$2"; shift 2 ;;
    --header) HEADER="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    --json)   JSON=1; shift ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --csv FILE [--schema FILE] [--header FILE] [--strict] [--json] [--max-rows N]"
      exit 0 ;;
    *) echo "[ERR] Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ------------------------------------------------------------------------------
# Repo-root & files
# ------------------------------------------------------------------------------
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/spectramind" ]]; then
  echo "[ERR] Run from SpectraMind V50 repo root." >&2
  exit 1
fi
[[ -z "$CSV" ]] && { echo "[ERR] --csv required"; exit 1; }
[[ -f "$CSV" ]] || { echo "[ERR] csv not found: $CSV"; exit 1; }
[[ -f "$SCHEMA" ]] || { echo "[ERR] schema not found: $SCHEMA"; exit 1; }
[[ -f "$HEADER" ]] || { echo "[ERR] header not found: $HEADER"; exit 1; }

# ------------------------------------------------------------------------------
# 1) Header exact match (order/columns)
# ------------------------------------------------------------------------------
csv_header="$(head -n1 "$CSV" | tr -d '\r')"
ref_header="$(tr -d '\r' < "$HEADER")"
if [[ "$csv_header" != "$ref_header" ]]; then
  echo "[ERR] Header mismatch with $HEADER"
  echo "      Expected: $ref_header"
  echo "      Found:    $csv_header"
  exit 2
fi

# Count columns
COLS=$(awk -F',' 'NR==1{print NF}' "$CSV")
REF_COLS=$(awk -F',' 'NR==1{print NF}' "$HEADER")
if [[ "$COLS" -ne "$REF_COLS" ]]; then
  echo "[ERR] Column count mismatch ($COLS != $REF_COLS)"; exit 2
fi

# ------------------------------------------------------------------------------
# 2) Table Schema validation (frictionless)
# ------------------------------------------------------------------------------
# Expect repository Python module to provide validator wrapper:
#   python -m spectramind.utils.schema validate --schema SCHEMA --header HEADER --csv CSV
$PYBIN -m spectramind.utils.schema validate \
  --schema "$SCHEMA" \
  --header "$HEADER" \
  --csv "$CSV"

# ------------------------------------------------------------------------------
# 3) Structural checks: unique ids, NaNs/Infs, numeric casting
# ------------------------------------------------------------------------------
tmp_report="$(mktemp)"
trap 'rm -f "$tmp_report"' EXIT

$PYBIN - <<'PYCODE' "$CSV" "$HEADER" "$STRICT" "$MAX_ROWS" > "$tmp_report"
import csv, sys, math, json
from collections import Counter

csv_path, header_path, strict_s, max_rows_s = sys.argv[1:5]
STRICT = bool(int(strict_s))
MAX_ROWS = int(max_rows_s) if max_rows_s else None

def is_float(s):
    try:
        float(s)
        return True
    except Exception:
        return False

with open(header_path, newline='') as fh:
    header_ref = next(csv.reader(fh))
mu_cols = [h for h in header_ref if h.startswith("mu_")]
sg_cols = [h for h in header_ref if h.startswith("sigma_")]

id_seen = Counter()
n_rows = 0
err = 0
warn = 0

with open(csv_path, newline='') as fh:
    rdr = csv.DictReader(fh)
    for i, row in enumerate(rdr, start=1):
        if MAX_ROWS and i > MAX_ROWS: break
        sid = row['sample_id']
        id_seen[sid] += 1
        if id_seen[sid] > 1:
            print(f"[ERR] Duplicate sample_id at row {i}: {sid}")
            err += 1

        # numeric cast & inf/nan check
        for k,v in row.items():
            if k == 'sample_id': continue
            if not is_float(v):
                print(f"[ERR] Non-numeric value at row {i}, col {k}: {v!r}")
                err += 1; continue
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                print(f"[ERR] NaN/Inf at row {i}, col {k}: {v}")
                err += 1

        # physics sanity: mu>=0, sigma>0
        for k in mu_cols:
            x = float(row[k])
            if x < 0.0:
                lvl = "[ERR]" if STRICT else "[WARN]"
                print(f"{lvl} Negative mu at row {i}, col {k}: {x}")
                if STRICT: err += 1
                else: warn += 1
        for k in sg_cols:
            x = float(row[k])
            if x <= 0.0:
                lvl = "[ERR]" if STRICT else "[WARN]"
                print(f"{lvl} Non-positive sigma at row {i}, col {k}: {x}")
                if STRICT: err += 1
                else: warn += 1

        n_rows += 1

# summary
res = {"rows_checked": n_rows, "dupe_ids": sum(c>1 for c in id_seen.values()),
       "errors": err, "warnings": warn}
print(f"[SUMMARY] rows={n_rows} dupes={res['dupe_ids']} errors={err} warnings={warn}")
print("::JSON::" + json.dumps(res))
PYCODE

# ------------------------------------------------------------------------------
# 4) Emit report / JSON
# ------------------------------------------------------------------------------
rows=$(grep -Eo '\[SUMMARY\].*' "$tmp_report" || true)
json_line=$(grep '^::JSON::' "$tmp_report" | sed 's/^::JSON:://' || true)

if [[ $JSON -eq 1 ]]; then
  if [[ -z "$json_line" ]]; then
    echo '{"errors":1,"message":"internal error: missing summary JSON"}'
    exit 3
  fi
  echo "$json_line"
else
  cat "$tmp_report"
fi

# Fail on errors
errs=$(echo "$json_line" | awk -F'"errors":' '{print $2}' | awk -F',' '{print $1}' 2>/dev/null || echo "")
if [[ -z "$errs" ]]; then errs=1; fi
if (( errs > 0 )); then
  echo "[FAIL] Submission failed validation."
  exit 4
fi

echo "[OK] Submission passed validation."

