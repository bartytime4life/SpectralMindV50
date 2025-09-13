#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# SpectraMind V50 — Kaggle submission packager (schema-aware)
# Collects one submission CSV, enforces header, validates, and zips bundle.
# Usage:
#   scripts/package_submission.sh [--input path/to/preds.csv]
# Outputs:
#   artifacts/submission.zip
# ------------------------------------------------------------------------------

set -Eeuo pipefail

INPUT=""
OUTDIR="${SUBMISSION_DIR:-artifacts}"
OUTZIP="${SUBMISSION_ZIP:-$OUTDIR/submission.zip}"
HEADER_FILE="schemas/submission_header.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

mkdir -p "$OUTDIR"

# Choose input: explicit file or latest CSV under common dirs
if [[ -z "$INPUT" ]]; then
  mapfile -t CANDIDATES < <(ls -1t \
    artifacts/predictions/*.csv 2>/dev/null || true; \
    ls -1t predictions/*.csv 2>/dev/null || true; \
    ls -1t outputs/*.csv 2>/dev/null || true)
  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "::error::No candidate CSV found (artifacts/predictions|predictions|outputs)."
    exit 1
  fi
  INPUT="${CANDIDATES[0]}"
fi

[[ -f "$INPUT" ]] || { echo "::error::Missing input CSV: $INPUT"; exit 1; }
[[ -f "$HEADER_FILE" ]] || { echo "::error::Missing schema header: $HEADER_FILE"; exit 1; }

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

python - "$INPUT" "$HEADER_FILE" "$TMPDIR/reordered.csv" << 'PY'
import sys, csv, math, re
from pathlib import Path

inp, header_file, outp = map(Path, sys.argv[1:4])

# Load desired header order
with header_file.open("r", encoding="utf-8", newline="") as f:
    rdr = csv.reader(f)
    desired = next(rdr)

required = set(desired)

# Utility: numeric check (float) and nonneg
def as_float(x):
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Non-numeric value: {x!r}")

# Read input and inspect columns
with inp.open("r", encoding="utf-8", newline="") as f:
    rdr = csv.DictReader(f)
    cols = rdr.fieldnames or []
    have = set(cols)

    missing = [c for c in desired if c not in have]
    if missing:
        raise SystemExit(f"ERROR: Missing required columns: {missing}")

    # Reorder; ignore extra columns by keeping them after required columns
    extras = [c for c in cols if c not in required]
    ordered = desired + extras

    mu_cols = [c for c in desired if c.startswith("mu_")]
    sg_cols = [c for c in desired if c.startswith("sigma_")]
    if len(mu_cols) != 283 or len(sg_cols) != 283:
        raise SystemExit("ERROR: Expected 283 mu_* and 283 sigma_* columns.")

    with outp.open("w", encoding="utf-8", newline="") as g:
        w = csv.DictWriter(g, fieldnames=ordered)
        w.writeheader()
        n = 0
        for row in rdr:
            # Validate required numeric fields
            for c in mu_cols + sg_cols:
                v = row.get(c, "")
                if v == "" or v is None:
                    raise SystemExit(f"ERROR: Empty value in required column {c}")
                val = as_float(v)
                if c.startswith("sigma_") and val < 0:
                    raise SystemExit(f"ERROR: Negative sigma in {c}: {val}")
                # NaN checks
                if math.isnan(val) or math.isinf(val):
                    raise SystemExit(f"ERROR: Non-finite value in {c}: {val}")
            w.writerow({k: row.get(k, "") for k in ordered})
            n += 1
        if n == 0:
            raise SystemExit("ERROR: Input CSV appears to have no rows.")
print("OK")
PY

cp "$TMPDIR/reordered.csv" "$OUTDIR/submission.csv"

# Zip (flat)
( cd "$OUTDIR" && zip -q -j "$OUTZIP" submission.csv )

[[ -f "$OUTZIP" ]] || { echo "::error::Failed to create $OUTZIP"; exit 1; }
echo "::notice::Submission bundle -> $OUTZIP"
