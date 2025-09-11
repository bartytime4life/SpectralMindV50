# src/spectramind/submit/validate.py
"""
SpectraMind V50 — Submission Validation Utilities
=================================================

Validates a submission CSV/DF against the repository JSON schema and additional
semantic checks tailored for the Ariel Data Challenge.

Supported layouts
-----------------
1) Narrow: columns ['id'|'sample_id', 'mu', 'sigma'] where mu/sigma are JSON arrays
           (or Python-list literals) of length N_BINS (default 283).
2) Wide:   columns ['id'|'sample_id'] + mu_000..mu_{N_BINS-1} + sigma_000..sigma_{N_BINS-1}

Extras in this upgrade
----------------------
- Robust schema resolution + optional user-supplied schema path
- Accepts 'id' or 'sample_id' (configurable), with optional strict-uniqueness
- Mask-safe numeric checks; μ ≥ 0; σ ≥ 0; finite μ/σ; vector length == N_BINS
- Chunked validation for low memory environments (Kaggle/CI)
- Optional CSV gz/bz2/zip inferencing via pandas
- Optional report writing (JSON) with summary stats & first-N errors
- Utility to normalize to *wide* format (`coerce_to_wide`) for diagnostics
- Back-compat helpers: `validate_row_dict`, `validate_dataframe`
- CLI entry point: `python -m spectramind.submit.validate <csv> [--schema ...]`

Notes
-----
- Deterministic behavior; no network access.
- `pandas` required; `jsonschema` optional.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

try:
    # jsonschema is optional; if absent, schema validation becomes a no-op
    from jsonschema import ValidationError, validate as _js_validate  # type: ignore
except Exception:  # pragma: no cover
    ValidationError = Exception  # type: ignore
    _js_validate = None  # type: ignore

# ==============================================================================
# Constants / schema loader
# ==============================================================================

N_BINS_DEFAULT = int(os.environ.get("SM_SUBMISSION_BINS", 283))
_SUB_SCHEMA_CACHE: Optional[Dict] = None


def _write_json_pretty(out_path: Union[str, Path], obj: Any) -> None:
    """
    Atomic, pretty JSON writer: <path>.tmp then os.replace().
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)


def _find_schema_file(explicit: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the submission schema path robustly.

    Precedence:
      0) explicit (if provided)
      1) ./schemas/submission.schema.json (from CWD)
      2) walk upwards from CWD to find a 'schemas/submission.schema.json'
      3) walk upwards from this file to find the same
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Schema file not found: {p}")
        return p

    # try current working dir first
    local = Path("schemas/submission.schema.json")
    if local.exists():
        return local

    # walk upward from CWD
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / "schemas" / "submission.schema.json"
        if candidate.exists():
            return candidate

    # walk upward from this file's location
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / "schemas" / "submission.schema.json"
        if candidate.exists():
            return candidate

    # fallback (will error when opened)
    return local


def _load_schema(explicit: Optional[Union[str, Path]] = None) -> Dict:
    """
    Load and cache the schema JSON; no-op if jsonschema not available.
    """
    global _SUB_SCHEMA_CACHE
    if explicit is None and _SUB_SCHEMA_CACHE is not None:
        return _SUB_SCHEMA_CACHE

    schema_path = _find_schema_file(explicit)
    try:
        text = schema_path.read_text(encoding="utf-8")
        schema = json.loads(text)
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

    # physics & numeric guards
    if not all(_is_finite_number(v) and float(v) >= 0.0 for v in mu):
        return [], [], "mu contains negative or non-finite values"
    if not all(_is_finite_number(v) and float(v) >= 0.0 for v in sigma):
        return [], [], "sigma contains negative or non-finite values"

    return mu, sigma, None


def _validate_record(rec: Dict, schema: Dict, n_bins: int) -> Optional[str]:
    """
    Validate a single record dict against JSON schema and additional semantic checks.
    Returns error string if any, else None.
    """
    # Schema may be loaded even if jsonschema isn't installed; short-circuit
    if _js_validate is not None:
        try:
            _js_validate(rec, schema)  # type: ignore
        except ValidationError as e:  # type: ignore
            return f"jsonschema validation error: {e.message}"

    mu = rec.get("mu") or []
    sigma = rec.get("sigma") or []
    if len(mu) != n_bins or len(sigma) != n_bins:
        return f"expected length {n_bins}, got mu={len(mu)}, sigma={len(sigma)}"
    if any((not _is_finite_number(x) or float(x) < 0.0) for x in mu):
        return "mu contains negative or non-finite values"
    if any((not _is_finite_number(s) or float(s) < 0.0) for s in sigma):
        return "sigma contains negative or non-finite values"
    return None


def _expected_wide_columns(n_bins: int, id_field: str) -> List[str]:
    return [id_field] + [f"mu_{i:03d}" for i in range(n_bins)] + [f"sigma_{i:03d}" for i in range(n_bins)]


# ==============================================================================
# Public API (CSV-focused) + Back-compat helpers
# ==============================================================================

@dataclass
class CSVValidationResult:
    ok: bool
    errors: List[str]
    n_rows: int
    n_valid: int


def coerce_to_wide(
    df: pd.DataFrame, *, n_bins: int = N_BINS_DEFAULT, id_field: str = "id"
) -> pd.DataFrame:
    """
    Return a normalized *wide* dataframe with columns:
       [id_field, mu_000.., mu_XXX, sigma_000.., sigma_XXX]

    If rows are already wide, they are passed through. For narrow rows, 'mu'/'sigma' JSON arrays are expanded.
    """
    out_cols = _expected_wide_columns(n_bins, id_field)
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rid = row.get(id_field)
        mu, sigma, err = _extract_vectors(row, n_bins)
        if err is not None:
            rows.append({id_field: rid})
            continue
        rec = {id_field: rid}
        rec.update({f"mu_{i:03d}": mu[i] for i in range(n_bins)})
        rec.update({f"sigma_{i:03d}": sigma[i] for i in range(n_bins)})
        rows.append(rec)
    return pd.DataFrame(rows, columns=out_cols)


def validate_row_dict(
    row: Dict[str, Any],
    *,
    n_bins: int = N_BINS_DEFAULT,
    schema_path: Optional[Union[str, Path]] = None,
    id_field: str = "sample_id",
) -> Optional[str]:
    """
    Back-compat single-record validator. Expects a dict with 'mu'/'sigma' lists and an id.
    Returns error string if any else None.
    """
    schema = _load_schema(schema_path)
    rid = row.get(id_field) or row.get("id")
    rec = {"mu": row.get("mu"), "sigma": row.get("sigma"), "id": rid, "sample_id": rid}
    return _validate_record(rec, schema, n_bins)


def validate_dataframe(
    df: pd.DataFrame,
    *,
    n_bins: int = N_BINS_DEFAULT,
    strict_order: bool = True,
    check_unique_ids: bool = True,
    id_field: str = "id",
    allow_alt_id: bool = True,
) -> CSVValidationResult:
    """
    Back-compat DF validator for wide DataFrames (or narrow with 'mu'/'sigma').
    """
    errors: List[str] = []
    n_rows = 0
    n_valid = 0
    seen_ids: set = set()
    resolved_id = id_field if id_field in df.columns else ("sample_id" if allow_alt_id and "sample_id" in df.columns else id_field)

    # If it's obviously wide, optionally check order
    maybe_wide = all(f"mu_{i:03d}" in df.columns for i in range(min(n_bins, 5))) and all(
        f"sigma_{i:03d}" in df.columns for i in range(min(n_bins, 5))
    )
    if strict_order and maybe_wide:
        expected = _expected_wide_columns(n_bins, resolved_id)
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing:
            errors.append(f"Missing expected columns (wide): {missing[:10]}{'...' if len(missing)>10 else ''}")
        if extra:
            errors.append(f"Unexpected columns (wide): {extra[:10]}{'...' if len(extra)>10 else ''}")

    schema = _load_schema(None)
    for idx, row in df.iterrows():
        n_rows += 1
        rid = row.get(resolved_id)
        if check_unique_ids:
            if pd.isna(rid) or (isinstance(rid, str) and rid.strip() == ""):
                errors.append(f"row {idx}: missing/empty {resolved_id}")
                continue
            if rid in seen_ids:
                errors.append(f"row {idx}: duplicate {resolved_id} '{rid}'")
                continue
            seen_ids.add(rid)

        mu_vec, sigma_vec, parse_err = _extract_vectors(row, n_bins)
        if parse_err is not None:
            errors.append(f"row {idx} ({resolved_id}={rid}): {parse_err}")
            continue

        rec = {resolved_id: rid, "mu": mu_vec, "sigma": sigma_vec, "id": rid, "sample_id": rid}
        err = _validate_record(rec, schema, n_bins)
        if err is not None:
            errors.append(f"row {idx} ({resolved_id}={rid}): {err}")
        else:
            n_valid += 1

    return CSVValidationResult(ok=(len(errors) == 0), errors=errors, n_rows=n_rows, n_valid=n_valid)


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
    strict_wide_order: bool = False,
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
    strict_wide_order : bool
        If True and the file is wide, enforce exact wide column order.

    Returns
    -------
    CSVValidationResult
        ok=True if all rows validate, else ok=False with per-row error messages.
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
    resolved_id_field = id_field  # may flip to 'sample_id' if found

    def _maybe_check_wide_order(frame: pd.DataFrame) -> None:
        if not strict_wide_order:
            return
        nonlocal resolved_id_field
        if resolved_id_field not in frame.columns and allow_alt_id and alt_field in frame.columns:
            resolved_id_field = alt_field
        # If looks wide, enforce order
        maybe_wide = all(f"mu_{i:03d}" in frame.columns for i in range(min(n_bins, 5))) and all(
            f"sigma_{i:03d}" in frame.columns for i in range(min(n_bins, 5))
        )
        if maybe_wide:
            expected = _expected_wide_columns(n_bins, resolved_id_field)
            missing = [c for c in expected if c not in frame.columns]
            extra = [c for c in frame.columns if c not in expected]
            if missing:
                errors.append(f"Missing expected columns (wide): {missing[:10]}{'...' if len(missing)>10 else ''}")
            if extra:
                errors.append(f"Unexpected columns (wide): {extra[:10]}{'...' if len(extra)>10 else ''}")

    def _process_frame(frame: pd.DataFrame) -> None:
        nonlocal n_rows, n_valid, resolved_id_field
        # Resolve id column once per chunk (first chunk determines resolution)
        if resolved_id_field not in frame.columns and allow_alt_id and alt_field in frame.columns:
            resolved_id_field = alt_field

        _maybe_check_wide_order(frame)

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
            # Ensure both id keys exist for schema variants
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
    if chunksize and chunksize > 0:
        for chunk in pd.read_csv(path, chunksize=chunksize, **read_kwargs):  # type: ignore[arg-type]
            _process_frame(chunk)
    else:
        df = pd.read_csv(path, **read_kwargs)  # type: ignore[arg-type]
        _process_frame(df)

    ok = len(errors) == 0
    result = CSVValidationResult(ok=ok, errors=errors, n_rows=n_rows, n_valid=n_valid)

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
        _write_json_pretty(write_report, report)

    return result


# ==============================================================================
# CLI
# ==============================================================================

def _main() -> None:
    ap = argparse.ArgumentParser(description="Validate SpectraMind V50 submission CSV.")
    ap.add_argument("csv", type=str, help="Path to submission CSV (supports .csv.gz/.bz2/.zip).")
    ap.add_argument("--schema", type=str, default=None, help="Path to submission.schema.json (optional).")
    ap.add_argument("--bins", type=int, default=N_BINS_DEFAULT, help=f"Number of spectral bins (default {N_BINS_DEFAULT}).")
    ap.add_argument("--no-strict-ids", action="store_true", help="Do not enforce unique/non-empty ids.")
    ap.add_argument("--id-field", type=str, default="id", help="Name of id column to use (default 'id').")
    ap.add_argument("--no-alt-id", action="store_true", help="Do not accept 'sample_id' as an alternative id.")
    ap.add_argument("--chunksize", type=int, default=None, help="Validate in chunks (memory friendly).")
    ap.add_argument("--report", type=str, default=None, help="Write JSON report to this path.")
    ap.add_argument("--strict-wide-order", action="store_true", help="Enforce exact wide column order if detected.")
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
        strict_wide_order=args.strict_wide_order,
    )

    print(json.dumps(asdict(res), indent=2, sort_keys=True))
    if not res.ok:
        raise SystemExit(2)


if __name__ == "__main__":  # pragma: no cover
    _main()
