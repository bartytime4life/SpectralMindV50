# src/spectramind/submit/validate.py
"""
SpectraMind V50 — Submission Validation Utilities
=================================================

Validates a submission CSV against the repository JSON schema and additional
semantic checks tailored for the Ariel Data Challenge:

Supported layouts
-----------------
1) Narrow: columns ['id'|'sample_id', 'mu', 'sigma'] where mu/sigma are JSON arrays
           (or Python-list literals) of length N_BINS (default 283).
2) Wide:   columns ['id'|'sample_id'] + mu_000..mu_{N_BINS-1} + sigma_000..sigma_{N_BINS-1}

Extras in this upgrade
----------------------
- Robust schema resolution + optional user-supplied schema path
- Accepts 'id' or 'sample_id' (configurable), with optional strict-uniqueness
- Mask-safe numeric checks; sigma >= 0; finite μ/σ; vector length == N_BINS
- Chunked validation for low memory environments (Kaggle/CI)
- Optional CSV gz/bz2/zip inferencing via pandas
- Optional report writing (JSON) with summary stats & first-N errors
- Utility to normalize to *wide* format (`coerce_to_wide`) for diagnostics
- CLI entry-point: `python -m spectramind.submit.validate <csv> [--schema ...]`

Notes
-----
- Deterministic behavior; no network access.
- Requires `pandas` and `jsonschema`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import pandas as pd
from jsonschema import ValidationError, validate

# ==============================================================================
# Constants / schema loader
# ==============================================================================

N_BINS_DEFAULT = int(os.environ.get("SM_SUBMISSION_BINS", 283))
_SUB_SCHEMA_CACHE: Optional[Dict] = None


def _find_schema_file(explicit: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the submission schema path robustly.

    Precedence:
      0) explicit (if provided)
      1) ./schemas/submission.schema.json
      2) walk upwards from this file for <repo_root>/schemas/submission.schema.json
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Schema file not found: {p}")
        return p

    local = Path("schemas/submission.schema.json")
    if local.exists():
        return local

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent.parent.parent / "schemas" / "submission.schema.json"
        if candidate.exists():
            return candidate

    # Fallback to original relative path (will raise later if missing)
    return local


def _load_schema(explicit: Optional[Union[str, Path]] = None) -> Dict:
    global _SUB_SCHEMA_CACHE
    # Cache only if using default path; explicit schemas might differ across calls
    if explicit is None and _SUB_SCHEMA_CACHE is not None:
        return _SUB_SCHEMA_CACHE

    schema_path = _find_schema_file(explicit)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Submission schema not found. Looked for: {schema_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema: {schema_path} :: {e}") from e

    if explicit is None:
        _SUB_SCHEMA_CACHE = schema
    return schema


# ==============================================================================
# Helpers
# ==============================================================================

def _is_finite_number(x: object) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def _try_parse_json_array(val: object) -> Optional[List[float]]:
    """
    Parse value as JSON array -> Python list[float].
    Returns None if not parseable / not a JSON array.
    """
    if isinstance(val, (list, tuple)):
        try:
            return [float(v) for v in val]
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)):
                return [float(v) for v in parsed]
        except Exception:
            return None
    return None


def _collect_wide_columns(row: pd.Series, prefix: str, n_bins: int) -> Optional[List[float]]:
    """
    Support wide format: mu_000..mu_XXX , sigma_000..sigma_XXX
    """
    values: List[float] = []
    for i in range(n_bins):
        col = f"{prefix}_{i:03d}"
        if col not in row:
            return None
        v = row[col]
        try:
            values.append(float(v))
        except Exception:
            return None
    return values


def _extract_vectors(
    row: pd.Series, n_bins: int
) -> Tuple[List[float], List[float], Optional[str]]:
    """
    Detect and extract mu/sigma vectors from either:
      - JSON string/list columns: 'mu', 'sigma'
      - wide columns: 'mu_000'.., 'sigma_000'..
    Returns (mu, sigma, error_message?) where error_message is None on success.
    """
    mu = _try_parse_json_array(row.get("mu"))
    sigma = _try_parse_json_array(row.get("sigma"))

    # Path 2: wide columns
    if mu is None or sigma is None:
        mu_wide = _collect_wide_columns(row, "mu", n_bins)
        sigma_wide = _collect_wide_columns(row, "sigma", n_bins)
        if mu is None:
            mu = mu_wide
        if sigma is None:
            sigma = sigma_wide

    if mu is None or sigma is None:
        return [], [], "mu/sigma could not be parsed (neither JSON arrays nor wide columns found)"

    if len(mu) != n_bins or len(sigma) != n_bins:
        return [], [], f"vector length mismatch: got mu={len(mu)}, sigma={len(sigma)}, expected={n_bins}"

    if not all(_is_finite_number(v) for v in mu):
        return [], [], "mu contains non-finite values"
    if not all(_is_finite_number(v) and float(v) >= 0.0 for v in sigma):
        return [], [], "sigma contains negative or non-finite values"

    return mu, sigma, None


def _validate_record(rec: Dict, schema: Dict, n_bins: int) -> Optional[str]:
    """
    Validate a single record dict against JSON schema and additional semantic checks.
    Returns error string if any, else None.
    """
    try:
        validate(rec, schema)
    except ValidationError as e:
        return f"jsonschema validation error: {e.message}"

    mu = rec.get("mu") or []
    sigma = rec.get("sigma") or []
    if len(mu) != n_bins or len(sigma) != n_bins:
        return f"expected length {n_bins}, got mu={len(mu)}, sigma={len(sigma)}"
    if any(not _is_finite_number(x) for x in mu):
        return "mu contains non-finite values"
    if any((not _is_finite_number(s) or float(s) < 0.0) for s in sigma):
        return "sigma contains negative or non-finite values"
    return None


# ==============================================================================
# Public API
# ==============================================================================

@dataclass
class CSVValidationResult:
    ok: bool
    errors: List[str]
    n_rows: int
    n_valid: int


def coerce_to_wide(df: pd.DataFrame, *, n_bins: int = N_BINS_DEFAULT, id_field: str = "id") -> pd.DataFrame:
    """
    Return a normalized *wide* dataframe with columns:
       [id_field, mu_000.., mu_XXX, sigma_000.., sigma_XXX]

    If rows are already wide, they are passed through. For narrow rows, 'mu'/'sigma' JSON arrays are expanded.
    """
    out_cols = [id_field] + [f"mu_{i:03d}" for i in range(n_bins)] + [f"sigma_{i:03d}" for i in range(n_bins)]
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rid = row.get(id_field)
        mu, sigma, err = _extract_vectors(row, n_bins)
        if err is not None:
            # keep row but leave missing cols as NaN
            rows.append({id_field: rid})
            continue
        rec = {id_field: rid}
        rec.update({f"mu_{i:03d}": mu[i] for i in range(n_bins)})
        rec.update({f"sigma_{i:03d}": sigma[i] for i in range(n_bins)})
        rows.append(rec)
    return pd.DataFrame(rows, columns=out_cols)


def validate_csv(
    csv_path: Union[str, Path],
    *,
    n_bins: int = N_BINS_DEFAULT,
    strict_ids: bool = True,
    chunksize: Optional[int] = None,
    id_field: str = "id",
    allow_alt_id: bool = True,
    schema_path: Optional[Union[str, Path]] = None,
    write_report: Optional[Union[str, Path]] = None,
    max_errors_in_report: int = 200,
) -> CSVValidationResult:
    """
    Validate a submission CSV against schema + SpectraMind semantic checks.

    CSV layouts supported:
      1) Narrow: ['id'|'sample_id', 'mu', 'sigma'] (mu/sigma JSON arrays)
      2) Wide:   ['id'|'sample_id'] + mu_000.. + sigma_000..

    Parameters
    ----------
    csv_path : str | Path
        Path to CSV file to validate. (Supports gz/bz2/zip via pandas infer.)
    n_bins : int
        Expected number of spectral bins per id (default from env or 283).
    strict_ids : bool
        Enforce non-empty ids and (if True) uniqueness.
    chunksize : int | None
        If provided, validates the CSV in chunks to reduce memory usage.
    id_field : str
        Name of id column to expect (default 'id').
    allow_alt_id : bool
        Allow 'sample_id' as an alternative id column if present.
    schema_path : str | Path | None
        Optional explicit path to the submission JSON schema.
    write_report : str | Path | None
        If set, write a JSON report (summary + first N errors) to this path.
    max_errors_in_report : int
        Cap number of stored error strings in the output report.

    Returns
    -------
    CSVValidationResult
        ok=True if all rows validate, else ok=False with per-row error messages.

    Raises
    ------
    FileNotFoundError
        If csv_path does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    schema = _load_schema(schema_path)
    errors: List[str] = []
    n_rows = 0
    n_valid = 0
    seen_ids: set = set()

    alt_field = "sample_id"
    # We will dynamically decide which id column is present
    resolved_id_field = id_field

    def _process_frame(frame: pd.DataFrame) -> None:
        nonlocal n_rows, n_valid, resolved_id_field
        # Resolve id column once per chunk (first chunk determines resolution)
        if resolved_id_field not in frame.columns and allow_alt_id and alt_field in frame.columns:
            resolved_id_field = alt_field

        for idx, row in frame.iterrows():
            n_rows += 1
            rid = row.get(resolved_id_field)
            # ID checks
            if strict_ids:
                if pd.isna(rid) or (isinstance(rid, str) and rid.strip() == ""):
                    errors.append(f"row {idx}: missing/empty {resolved_id_field}")
                    continue
                if rid in seen_ids:
                    errors.append(f"row {idx}: duplicate {resolved_id_field} '{rid}'")
                    continue
                seen_ids.add(rid)

            mu_vec, sigma_vec, parse_err = _extract_vectors(row, n_bins)
            if parse_err is not None:
                errors.append(f"row {idx} ({resolved_id_field}={rid}): {parse_err}")
                continue

            rec = {resolved_id_field: rid, "mu": mu_vec, "sigma": sigma_vec}
            # For schema validation, ensure property name is 'sample_id' or 'id' as schema expects.
            # We pass both; schema allows one (preferred 'sample_id' in newer schema).
            if resolved_id_field == "sample_id":
                rec["id"] = rid
            else:
                rec["sample_id"] = rid

            err = _validate_record(rec, schema, n_bins)
            if err is not None:
                errors.append(f"row {idx} ({resolved_id_field}={rid}): {err}")
            else:
                n_valid += 1

    read_kwargs = dict(dtype=object, keep_default_na=False)
    # pandas infers compression by extension; keep dtype=object to avoid coercion
    if chunksize and chunksize > 0:
        for chunk in pd.read_csv(path, chunksize=chunksize, **read_kwargs):  # type: ignore[arg-type]
            _process_frame(chunk)
    else:
        df = pd.read_csv(path, **read_kwargs)  # type: ignore[arg-type]
        _process_frame(df)

    ok = len(errors) == 0
    result = CSVValidationResult(ok=ok, errors=errors, n_rows=n_rows, n_valid=n_valid)

    # Optional report
    if write_report:
        report = {
            "ok": ok,
            "n_rows": n_rows,
            "n_valid": n_valid,
            "n_errors": len(errors),
            "first_errors": errors[:max_errors_in_report],
            "bins_expected": n_bins,
            "id_field_used": resolved_id_field,
            "csv": str(path),
            "schema": str(_find_schema_file(schema_path)),
        }
        Path(write_report).parent.mkdir(parents=True, exist_ok=True)
        Path(write_report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return result


# ==============================================================================
# CLI
# ==============================================================================

def _main() -> None:
    ap = argparse.ArgumentParser(description="Validate SpectraMind V50 submission CSV.")
    ap.add_argument("csv", type=str, help="Path to submission CSV (supports .csv.gz).")
    ap.add_argument("--schema", type=str, default=None, help="Path to submission.schema.json (optional).")
    ap.add_argument("--bins", type=int, default=N_BINS_DEFAULT, help=f"Number of spectral bins (default {N_BINS_DEFAULT}).")
    ap.add_argument("--no-strict-ids", action="store_true", help="Do not enforce unique/non-empty ids.")
    ap.add_argument("--id-field", type=str, default="id", help="Name of id column to use (default 'id').")
    ap.add_argument("--no-alt-id", action="store_true", help="Do not accept 'sample_id' as an alternative id.")
    ap.add_argument("--chunksize", type=int, default=None, help="Validate in chunks (memory friendly).")
    ap.add_argument("--report", type=str, default=None, help="Write JSON report to this path.")
    args = ap.parse_args()

    res = validate_csv(
        args.csv,
        n_bins=args.bins,
        strict_ids=(not args.no_strict_ids),
        chunksize=args.chunksize,
        id_field=args.id_field,
        allow_alt_id=(not args.no_alt_id),
        schema_path=args.schema,
        write_report=args.report,
    )

    print(json.dumps(asdict(res), indent=2, sort_keys=True))
    if not res.ok:
        raise SystemExit(2)


if __name__ == "__main__":  # pragma: no cover
    _main()