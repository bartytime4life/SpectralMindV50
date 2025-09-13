#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# SpectraMind V50 — Kaggle submission packager (schema-aware, reproducible)
# • Validates against header CSV (required) and optional JSON Schema
# • Enforces column order; drops extras unless --keep-extras
# • Deterministic ZIP via Python (no system `zip` dependency)
# • Emits artifacts/submission.csv + artifacts/submission.zip + manifest.json
#
# Usage:
#   scripts/package_submission.sh \
#     [--input path/to/preds.csv] \
#     [--header schemas/submission_header.csv] \
#     [--schema schemas/submission.schema.json] \
#     [--outdir artifacts] \
#     [--zipname submission.zip] \
#     [--keep-extras]
# ------------------------------------------------------------------------------

set -Eeuo pipefail

INPUT=""
OUTDIR="${SUBMISSION_DIR:-artifacts}"
ZIPNAME="${SUBMISSION_ZIP:-$OUTDIR/submission.zip}"
HEADER_FILE="schemas/submission_header.csv"
SCHEMA_FILE=""
KEEP_EXTRAS=0

# --- args ---------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)  INPUT="${2:?}"; shift 2 ;;
    --header) HEADER_FILE="${2:?}"; shift 2 ;;
    --schema) SCHEMA_FILE="${2:?}"; shift 2 ;;
    --outdir) OUTDIR="${2:?}"; shift 2 ;;
    --zipname) ZIPNAME="${2:?}"; shift 2 ;;
    --keep-extras) KEEP_EXTRAS=1; shift ;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0 ;;
    *) echo "::error::Unknown arg: $1"; exit 2 ;;
  esac
done

mkdir -p "$OUTDIR"

# --- choose input if not provided ---------------------------------------------
if [[ -z "$INPUT" ]]; then
  mapfile -t CANDIDATES < <(
    ls -1t artifacts/predictions/*.csv 2>/dev/null || true
    ls -1t predictions/*.csv          2>/dev/null || true
    ls -1t outputs/*.csv              2>/dev/null || true
  )
  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "::error::No candidate CSV found (artifacts/predictions|predictions|outputs)."
    exit 1
  fi
  INPUT="${CANDIDATES[0]}"
fi

[[ -f "$INPUT" ]]       || { echo "::error::Missing input CSV: $INPUT"; exit 1; }
[[ -f "$HEADER_FILE" ]] || { echo "::error::Missing header CSV: $HEADER_FILE"; exit 1; }
if [[ -n "$SCHEMA_FILE" && ! -f "$SCHEMA_FILE" ]]; then
  echo "::error::Schema file not found: $SCHEMA_FILE"; exit 1
fi

OUT_CSV="$OUTDIR/submission.csv"
MANIFEST="$OUTDIR/submission.manifest.json"

# --- run validator/packager in Python -----------------------------------------
python - "$INPUT" "$HEADER_FILE" "$OUT_CSV" "$ZIPNAME" "$MANIFEST" "$SCHEMA_FILE" "$KEEP_EXTRAS" << 'PY'
import sys, csv, math, re, json, hashlib, time, os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo
from datetime import datetime, timezone

inp, header_file, out_csv, out_zip, manifest, schema_file, keep_extras = sys.argv[1:8]
keep_extras = bool(int(keep_extras))

inp, header_file, out_csv, out_zip, manifest = map(Path, (inp, header_file, out_csv, out_zip, manifest))
schema = None
if schema_file:
    schema = json.loads(Path(schema_file).read_text(encoding="utf-8"))

# ---------- desired header (order) ----------
with open(header_file, "r", encoding="utf-8", newline="") as f:
    rdr = csv.reader(f)
    desired = next(rdr)
required = set(desired)

# convenience: detect mu_/sigma_ sets
mu_cols = [c for c in desired if c.startswith("mu_")]
sg_cols = [c for c in desired if c.startswith("sigma_")]
if len(mu_cols) != 283 or len(sg_cols) != 283:
    raise SystemExit("ERROR: Expected 283 mu_* and 283 sigma_* columns in header CSV.")

# optional schema constraints
sample_id_pattern = None
sigma_min = 0.0
if schema:
    # try to load sample_id pattern if present
    try:
        for field in schema.get("schema", {}).get("fields", []):
            if field.get("name") == "sample_id":
                pat = field.get("constraints", {}).get("pattern")
                if pat:
                    sample_id_pattern = re.compile(pat)
    except Exception:
        pass

# ---------- validate & reorder streamingly ----------
rows = 0
seen_ids = set()
with open(inp, "r", encoding="utf-8", newline="") as f_in, \
     open(out_csv, "w", encoding="utf-8", newline="") as f_out:
    rdr = csv.DictReader(f_in)
    cols = rdr.fieldnames or []
    if len(cols) != len(set(cols)):
        dupes = [c for c in cols if cols.count(c) > 1]
        raise SystemExit(f"ERROR: Duplicate column names in input: {sorted(set(dupes))}")

    have = set(cols)
    missing = [c for c in desired if c not in have]
    if missing:
        raise SystemExit(f"ERROR: Missing required columns: {missing}")

    extras = [c for c in cols if c not in required]
    ordered = desired + (extras if keep_extras else [])

    w = csv.DictWriter(f_out, fieldnames=ordered, lineterminator="\n")
    w.writeheader()

    for row in rdr:
        # sample_id checks (if present)
        sid = row.get("sample_id")
        if "sample_id" in required:
            if sid is None or sid == "":
                raise SystemExit("ERROR: Empty sample_id")
            if sample_id_pattern and not sample_id_pattern.fullmatch(sid):
                raise SystemExit(f"ERROR: sample_id violates pattern: {sid!r}")
            if sid in seen_ids:
                raise SystemExit(f"ERROR: duplicate sample_id: {sid!r}")
            seen_ids.add(sid)

        # numeric validations
        for c in mu_cols + sg_cols:
            v = row.get(c, "")
            if v == "" or v is None:
                raise SystemExit(f"ERROR: Empty value in required column {c}")
            try:
                val = float(v)
            except Exception:
                raise SystemExit(f"ERROR: Non-numeric value in {c}: {v!r}")
            if math.isnan(val) or math.isinf(val):
                raise SystemExit(f"ERROR: Non-finite value in {c}: {val}")
            if c.startswith("sigma_") and val < sigma_min:
                raise SystemExit(f"ERROR: Negative sigma in {c}: {val}")

        # write ordered row (dropping extras unless requested)
        out_row = {k: row.get(k, "") for k in ordered}
        w.writerow(out_row)
        rows += 1

if rows == 0:
    raise SystemExit("ERROR: Input CSV appears to have no rows.")

# ---------- sha256 + manifest ----------
h = hashlib.sha256()
with open(out_csv, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
sha256 = h.hexdigest()

created_at = datetime.now(timezone.utc).isoformat()

manifest_data = {
    "source_csv": str(inp),
    "output_csv": str(out_csv),
    "zip_path": str(out_zip),
    "rows": rows,
    "columns": len(desired),
    "keep_extras": keep_extras,
    "sha256": sha256,
    "created_at": created_at,
    "header": desired,
    "schema": str(schema_file) if schema_file else None,
}
Path(manifest).write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

# ---------- deterministic ZIP (no external zip; stable timestamps) ----------
# Use 1980-01-01 00:00:00 DOS epoch to make zip byte-identical across runs
DOS_EPOCH = (1980, 1, 1, 0, 0, 0)
out_zip.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
    zi = ZipInfo(filename="submission.csv", date_time=DOS_EPOCH)
    # normalize external attributes (permissions) for determinism: 0644
    zi.external_attr = (0o100644 & 0xFFFF) << 16
    with open(out_csv, "rb") as fh:
        zf.writestr(zi, fh.read())

print(f"OK rows={rows} sha256={sha256}")
PY

echo "::notice::Wrote $OUT_CSV"
echo "::notice::Manifest -> $MANIFEST"
[[ -f "$ZIPNAME" ]] || { echo "::error::Failed to create $ZIPNAME"; exit 1; }
echo "::notice::Submission bundle -> $ZIPNAME"
