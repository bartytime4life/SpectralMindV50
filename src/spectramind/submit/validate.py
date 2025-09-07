# src/spectramind/submit/validate.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import pandas as pd
from jsonschema import ValidationError, validate

# ==============================================================================
# Constants / schema loader
# ==============================================================================

N_BINS_DEFAULT = 283
_SUB_SCHEMA_CACHE: Optional[Dict] = None


def _find_schema_file() -> Path:
    """
    Resolve the submission schema path robustly, regardless of current working dir.

    It will try the following locations (in order):
      1) ./schemas/submission.schema.json
      2) <repo_root>/schemas/submission.schema.json (by walking upwards)
    """
    local = Path("schemas/submission.schema.json")
    if local.exists():
        return local

    # Walk upwards to locate a 'schemas/submission.schema.json'
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent.parent.parent / "schemas" / "submission.schema.json"
        if candidate.exists():
            return candidate

    # Fallback to original relative path (will raise later if missing)
    return local


def _load_schema() -> Dict:
    global _SUB_SCHEMA_CACHE
    if _SUB_SCHEMA_CACHE is None:
        schema_path = _find_schema_file()
        try:
            _SUB_SCHEMA_CACHE = json.loads(schema_path.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Submission schema not found. Looked for: {schema_path}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema: {schema_path} :: {e}") from e
    return _SUB_SCHEMA_CACHE


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
            return [float(v) for v in val]  # ensure numeric
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


def _collect_wide_columns(
    row: pd.Series, prefix: str, n_bins: int
) -> Optional[List[float]]:
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
    # Path 1: JSON array or list in columns 'mu' and 'sigma'
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
    if not all(_is_finite_number(v) for v in sigma):
        return [], [], "sigma contains non-finite values"

    return mu, sigma, None


def _validate_record(
    rec: Dict, schema: Dict, n_bins: int
) -> Optional[str]:
    """
    Validate a single record dict against JSON schema and additional semantic checks.
    Returns error string if any, else None.
    """
    # JSON schema
    try:
        validate(rec, schema)
    except ValidationError as e:
        return f"jsonschema validation error: {e.message}"

    # Semantic checks
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


def validate_csv(
    csv_path: Union[str, Path],
    *,
    n_bins: int = N_BINS_DEFAULT,
    strict_ids: bool = True,
    chunksize: Optional[int] = None,
) -> CSVValidationResult:
    """
    Validate a submission CSV against schema + SpectraMind semantic checks.

    CSV layouts supported:
      1) Narrow: columns ['id', 'mu', 'sigma'] where mu/sigma are JSON arrays or Python lists.
      2) Wide:   columns ['id'] + mu_000..mu_{n_bins-1} + sigma_000..sigma_{n_bins-1}

    Parameters
    ----------
    csv_path : str | Path
        Path to CSV file to validate.
    n_bins : int
        Expected number of spectral bins per id (default 283).
    strict_ids : bool
        Enforce non-empty ids and (if True) uniqueness.
    chunksize : int | None
        If provided, validates the CSV in chunks to reduce memory usage.

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

    schema = _load_schema()
    errors: List[str] = []
    n_rows = 0
    n_valid = 0
    seen_ids: set = set()

    def _process_frame(frame: pd.DataFrame) -> None:
        nonlocal n_rows, n_valid
        # Try to keep raw JSON strings intact
        for idx, row in frame.iterrows():
            n_rows += 1
            rid = row.get("id")
            if strict_ids:
                if pd.isna(rid) or (isinstance(rid, str) and rid.strip() == ""):
                    errors.append(f"row {idx}: missing/empty id")
                    continue
                if rid in seen_ids:
                    errors.append(f"row {idx}: duplicate id '{rid}'")
                    continue
                seen_ids.add(rid)

            mu_vec, sigma_vec, parse_err = _extract_vectors(row, n_bins)
            if parse_err is not None:
                errors.append(f"row {idx} (id={rid}): {parse_err}")
                continue

            rec = {"id": rid, "mu": mu_vec, "sigma": sigma_vec}
            err = _validate_record(rec, schema, n_bins)
            if err is not None:
                errors.append(f"row {idx} (id={rid}): {err}")
            else:
                n_valid += 1

    # Read CSV (with optional chunking)
    read_kwargs = dict(dtype=object, keep_default_na=False)
    if chunksize and chunksize > 0:
        for chunk in pd.read_csv(path, chunksize=chunksize, **read_kwargs):  # type: ignore[arg-type]
            _process_frame(chunk)
    else:
        df = pd.read_csv(path, **read_kwargs)  # type: ignore[arg-type]
        _process_frame(df)

    ok = len(errors) == 0
    return CSVValidationResult(ok=ok, errors=errors, n_rows=n_rows, n_valid=n_valid)
