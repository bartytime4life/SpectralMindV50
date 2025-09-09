# src/spectramind/submit/validate.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# ======================================================================================
# Env-configurable knobs
# ======================================================================================

def _bins_from_env(default: int = 283) -> int:
    try:
        v = int(os.environ.get("SM_SUBMISSION_BINS", str(default)))
        return v if v > 0 else default
    except Exception:
        return default

SM_ENFORCE_ORDER = os.environ.get("SM_ENFORCE_SUBMISSION_ORDER", "0") == "1"
SM_ENFORCE_NONNEG_MU = os.environ.get("SM_ENFORCE_NONNEG_MU", "0") == "1"
SM_ID_PATTERN = os.environ.get("SM_ID_PATTERN", r"^[A-Za-z0-9_\-]+$")
SM_ID_REGEX = re.compile(SM_ID_PATTERN)

# ======================================================================================
# Public result object
# ======================================================================================

@dataclass(slots=True)
class ValidationResult:
    ok: bool
    n_rows: int
    n_valid: int
    errors: List[str]

# ======================================================================================
# Helpers (wide schema)
# ======================================================================================

def _expected_columns(n_bins: int) -> List[str]:
    cols = ["sample_id"]
    cols += [f"mu_{i:03d}" for i in range(n_bins)]
    cols += [f"sigma_{i:03d}" for i in range(n_bins)]
    return cols

def _finite_numeric_block(df: pd.DataFrame, cols: Sequence[str]) -> Optional[str]:
    sub = df.loc[:, cols]
    # dtype check (allow nullable floats but block objects)
    non_num = [c for c in cols if not np.issubdtype(sub[c].dtype, np.number)]
    if non_num:
        return f"Non-numeric values in columns: {non_num[:10]}{' …' if len(non_num) > 10 else ''}"
    vals = sub.to_numpy(copy=False)
    if not np.isfinite(vals).all():
        return "NaN/Inf detected in μ/σ columns"
    return None

def _validate_structure_wide(df: pd.DataFrame, n_bins: int, enforce_order: bool) -> List[str]:
    errors: List[str] = []
    expected = _expected_columns(n_bins)

    got = list(df.columns)
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    if missing:
        errors.append(f"Missing columns: {missing[:10]}{' …' if len(missing) > 10 else ''}")
    if extra:
        errors.append(f"Extra columns: {extra[:10]}{' …' if len(extra) > 10 else ''}")

    if enforce_order and not errors and got != expected:
        errors.append("Column order mismatch with canonical submission ordering")

    if errors:
        return errors

    mu_cols = [f"mu_{i:03d}" for i in range(n_bins)]
    sigma_cols = [f"sigma_{i:03d}" for i in range(n_bins)]

    msg = _finite_numeric_block(df, mu_cols + sigma_cols)
    if msg:
        errors.append(msg)
        return errors

    sig = df[sigma_cols].to_numpy(copy=False)
    if not (sig > 0.0).all():
        errors.append("σ columns must be strictly positive (no zeros/negatives)")

    if SM_ENFORCE_NONNEG_MU:
        mu = df[mu_cols].to_numpy(copy=False)
        if (mu < 0.0).any():
            errors.append("μ columns must be non-negative (SM_ENFORCE_NONNEG_MU=1)")

    # sample_id checks
    if df["sample_id"].isna().any():
        errors.append("sample_id contains missing values")
    bad_pat = df[~df["sample_id"].astype(str).str.match(SM_ID_REGEX)]
    if not bad_pat.empty:
        examples = bad_pat["sample_id"].astype(str).head(5).tolist()
        errors.append(f"sample_id values violate pattern {SM_ID_PATTERN!r}, e.g. {examples}")
    if df["sample_id"].duplicated().any():
        dups = df.loc[df["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        errors.append(f"Duplicate sample_id values found, e.g. {dups}")

    return errors

# ======================================================================================
# Helpers (narrow schema)
# ======================================================================================

def _parse_array_cell(cell: object) -> Optional[List[float]]:
    """
    Accept JSON arrays or stringified arrays; return list[float] or None on failure.
    """
    if isinstance(cell, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in cell]
        except Exception:
            return None
    s = str(cell)
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return [float(x) for x in arr]
    except Exception:
        # Try forgiving bracket trimming like "[1,2,3]" without strict JSON
        if s.startswith("[") and s.endswith("]"):
            try:
                return [float(x.strip()) for x in s[1:-1].split(",") if x.strip() != ""]
            except Exception:
                return None
    return None

def _validate_structure_narrow(df: pd.DataFrame, n_bins: int) -> List[str]:
    errors: List[str] = []
    required = {"sample_id", "mu", "sigma"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
        return errors

    if df["sample_id"].isna().any():
        errors.append("sample_id contains missing values")
    bad_pat = df[~df["sample_id"].astype(str).str.match(SM_ID_REGEX)]
    if not bad_pat.empty:
        examples = bad_pat["sample_id"].astype(str).head(5).tolist()
        errors.append(f"sample_id values violate pattern {SM_ID_PATTERN!r}, e.g. {examples}")
    if df["sample_id"].duplicated().any():
        dups = df.loc[df["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        errors.append(f"Duplicate sample_id values found, e.g. {dups}")

    # row-wise array checks (streamed to avoid huge memory)
    row_errors = 0
    for i, row in df.iterrows():
        mu = _parse_array_cell(row["mu"])
        sg = _parse_array_cell(row["sigma"])
        sid = row["sample_id"]
        if mu is None or sg is None:
            errors.append(f"row {i} (id={sid!s}): mu/sigma not valid arrays")
            row_errors += 1
            if row_errors > 50:
                errors.append("Too many row errors; stopping early.")
                break
            continue
        if len(mu) != n_bins or len(sg) != n_bins:
            errors.append(f"row {i} (id={sid!s}): array lengths (mu={len(mu)}, sigma={len(sg)}) != n_bins={n_bins}")
            row_errors += 1
            if row_errors > 50:
                errors.append("Too many row errors; stopping early.")
                break
            continue
        if not np.isfinite(mu).all() or not np.isfinite(sg).all():
            errors.append(f"row {i} (id={sid!s}): NaN/Inf in mu/sigma arrays")
            row_errors += 1
            if row_errors > 50:
                errors.append("Too many row errors; stopping early.")
                break
        if (np.asarray(sg) <= 0.0).any():
            errors.append(f"row {i} (id={sid!s}): sigma must be strictly > 0")
            row_errors += 1
            if row_errors > 50:
                errors.append("Too many row errors; stopping early.")
                break
        if SM_ENFORCE_NONNEG_MU and (np.asarray(mu) < 0.0).any():
            errors.append(f"row {i} (id={sid!s}): mu must be non-negative (SM_ENFORCE_NONNEG_MU=1)")
            row_errors += 1
            if row_errors > 50:
                errors.append("Too many row errors; stopping early.")
                break

    return errors

# ======================================================================================
# Public validators
# ======================================================================================

def _read_input(inp: Union[PathLike, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(inp, (str, Path)):
        p = Path(inp)
        if not p.exists():
            raise FileNotFoundError(p)
        return pd.read_csv(p)
    if isinstance(inp, pd.DataFrame):
        return inp.copy()
    raise TypeError(f"Unsupported input type: {type(inp)!r}")

def _detect_format(df: pd.DataFrame) -> str:
    has_wide = any(c.startswith("mu_") for c in df.columns) and any(c.startswith("sigma_") for c in df.columns)
    has_narrow = {"mu", "sigma"}.issubset(df.columns)
    if has_wide and has_narrow:
        # Prefer wide if both present (ambiguous CSV)
        return "wide"
    if has_wide:
        return "wide"
    if has_narrow:
        return "narrow"
    return "unknown"

def validate_submission(inp: Union[PathLike, pd.DataFrame]) -> Tuple[bool, List[str]]:
    """
    Flexible in-memory validator for tests and ad-hoc checks.

    Returns:
        (ok: bool, errors: list[str])
    """
    try:
        df = _read_input(inp)
    except Exception as e:
        return False, [f"read-error: {type(e).__name__}: {e}"]

    n_bins = _bins_from_env(283)
    fmt = _detect_format(df)
    if fmt == "wide":
        errors = _validate_structure_wide(df, n_bins=n_bins, enforce_order=SM_ENFORCE_ORDER)
    elif fmt == "narrow":
        errors = _validate_structure_narrow(df, n_bins=n_bins)
    else:
        errors = ["Unrecognized schema: expected either wide (mu_***/sigma_***) or narrow (mu, sigma arrays)."]
    return (len(errors) == 0), errors

# ======================================================================================
# CSV streaming validator (used by CLI)
# ======================================================================================

def validate_csv(
    csv_path: PathLike,
    n_bins: int = _bins_from_env(283),
    strict_ids: bool = True,
    chunksize: Optional[int] = None,
) -> ValidationResult:
    """
    Validate a CSV file on disk. Supports both wide and narrow formats.
    Streams with chunksize to keep memory bounded.

    Args
    ----
    csv_path : str | Path
    n_bins   : expected number of spectral bins
    strict_ids : if True, enforce ID regex & duplicates check
    chunksize : if provided, read in chunks of this many rows

    Returns
    -------
    ValidationResult(ok, n_rows, n_valid, errors)
    """
    p = Path(csv_path)
    if not p.exists():
        return ValidationResult(False, 0, 0, [f"File not found: {p}"])

    errors: List[str] = []
    n_rows = 0
    n_valid = 0

    # Read header first to decide format quickly
    try:
        head = pd.read_csv(p, nrows=5)
    except Exception as e:
        return ValidationResult(False, 0, 0, [f"read-error: {type(e).__name__}: {e}"])

    fmt = _detect_format(head)
    if fmt == "unknown":
        return ValidationResult(False, 0, 0, ["Unrecognized schema: need wide(mu_***/sigma_***) or narrow(mu,sigma)."])

    if chunksize is None:
        # Single-shot
        try:
            df = pd.read_csv(p)
        except Exception as e:
            return ValidationResult(False, 0, 0, [f"read-error: {type(e).__name__}: {e}"])
        n_rows = len(df)
        if fmt == "wide":
            errs = _validate_structure_wide(df, n_bins=n_bins, enforce_order=SM_ENFORCE_ORDER)
        else:
            errs = _validate_structure_narrow(df, n_bins=n_bins)
        if not strict_ids:
            errs = [e for e in errs if not e.startswith("sample_id ")]
        n_valid = 0 if errs else n_rows
        return ValidationResult(len(errs) == 0, n_rows, n_valid, errs)

    # Chunked
    try:
        for chunk in pd.read_csv(p, chunksize=chunksize):
            n_rows += len(chunk)
            if fmt == "wide":
                errs = _validate_structure_wide(chunk, n_bins=n_bins, enforce_order=False)
            else:
                errs = _validate_structure_narrow(chunk, n_bins=n_bins)
            if not strict_ids:
                errs = [e for e in errs if not e.startswith("sample_id ")]
            if errs:
                # Scope errors to chunk (sample examples are included inside)
                errors.extend(errs[:50])  # keep error list bounded
            else:
                n_valid += len(chunk)
    except Exception as e:
        errors.append(f"stream-read-error: {type(e).__name__}: {e}")

    ok = len(errors) == 0
    return ValidationResult(ok, n_rows, n_valid, errors)

# ======================================================================================
# Optional JSON-schema hook (used by CLI if present)
# ======================================================================================

def validate_against_schema(csv_path: PathLike, schema_path: PathLike) -> Tuple[bool, str]:
    """
    If jsonschema is available and the user supplied a schema, run a lightweight check.
    Returns (ok, message). Never raises if jsonschema is missing.
    """
    try:
        import jsonschema  # type: ignore
    except Exception:
        return True, "jsonschema not installed; skipping schema check"

    try:
        import json
        p = Path(csv_path)
        df = pd.read_csv(p)
        # Convert wide to narrow JSON-like rows for generic schema friendliness
        if "mu" in df.columns and "sigma" in df.columns:
            rows = df.to_dict(orient="records")
        else:
            n_bins = _bins_from_env(283)
            mu_cols = [f"mu_{i:03d}" for i in range(n_bins)]
            sg_cols = [f"sigma_{i:03d}" for i in range(n_bins)]
            rows = []
            for _, r in df.iterrows():
                rows.append({
                    "sample_id": r["sample_id"],
                    "mu": [float(r[c]) for c in mu_cols],
                    "sigma": [float(r[c]) for c in sg_cols],
                })
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        for i, row in enumerate(rows):
            jsonschema.validate(row, schema)
        return True, ""
    except jsonschema.ValidationError as e:  # type: ignore
        return False, f"Schema validation failed at instance path {list(e.path)}: {e.message}"
    except Exception as e:
        return False, f"Schema validation crashed: {type(e).__name__}: {e}"
