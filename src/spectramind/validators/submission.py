from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _bins_from_env(default: int = 283) -> int:
    try:
        v = int(os.environ.get("SM_SUBMISSION_BINS", str(default)))
        return v if v > 0 else default
    except Exception:
        return default


def _expected_columns(n_bins: int) -> List[str]:
    cols = ["sample_id"]
    cols += [f"mu_{i:03d}" for i in range(n_bins)]
    cols += [f"sigma_{i:03d}" for i in range(n_bins)]
    return cols


def _read_input(inp: Union[PathLike, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(inp, (str, Path)):
        p = Path(inp)
        if not p.exists():
            raise FileNotFoundError(p)
        # Let pandas infer; CSV is canonical
        return pd.read_csv(p)
    if isinstance(inp, pd.DataFrame):
        return inp.copy()
    raise TypeError(f"Unsupported input type: {type(inp)!r}")


def _finite_numeric(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    sub = df.loc[:, cols]
    # dtypes numeric?
    non_num = [c for c in cols if not np.issubdtype(sub[c].dtype, np.number)]
    if non_num:
        errs.append(f"Non-numeric values in columns: {non_num[:10]}{' …' if len(non_num) > 10 else ''}")
        return False, errs
    # finite?
    vals = sub.to_numpy(copy=False)
    if not np.isfinite(vals).all():
        errs.append("NaN/Inf detected in μ/σ columns")
        return False, errs
    return True, errs


def _validate_structure(df: pd.DataFrame, n_bins: int, enforce_order: bool) -> List[str]:
    errors: List[str] = []
    expected = _expected_columns(n_bins)

    # Columns presence
    got = list(df.columns)
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    if missing:
        errors.append(f"Missing columns: {missing[:10]}{' …' if len(missing) > 10 else ''}")
    if extra:
        errors.append(f"Extra columns: {extra[:10]}{' …' if len(extra) > 10 else ''}")

    # Order (optional)
    if enforce_order and not errors:  # only check order if sets matched
        if got != expected:
            errors.append("Column order mismatch with canonical submission ordering")

    # Basic value checks only if we have the columns
    if not errors:
        mu_cols = [f"mu_{i:03d}" for i in range(n_bins)]
        sigma_cols = [f"sigma_{i:03d}" for i in range(n_bins)]
        ok, errs = _finite_numeric(df, mu_cols + sigma_cols)
        errors.extend(errs)
        if ok:
            sig = df[sigma_cols].to_numpy(copy=False)
            if not (sig > 0.0).all():
                errors.append("σ columns must be strictly positive (no zeros/negatives)")

        if df["sample_id"].duplicated().any():
            errors.append("Duplicate sample_id values found")

    return errors


def validate_submission(inp: Union[PathLike, pd.DataFrame]) -> Tuple[bool, List[str]]:
    """
    Flexible validator for tests and CLI.

    Returns:
        (ok: bool, errors: list[str])
    """
    try:
        df = _read_input(inp)
    except Exception as e:
        return False, [f"read-error: {type(e).__name__}: {e}"]

    n_bins = _bins_from_env(283)
    enforce_order = os.environ.get("SM_ENFORCE_SUBMISSION_ORDER", "0") == "1"
    errors = _validate_structure(df, n_bins=n_bins, enforce_order=enforce_order)
    return (len(errors) == 0), errors


@dataclass(slots=True)
class SubmissionValidator:
    """Class-style API wrapper."""

    n_bins: int = _bins_from_env(283)
    enforce_order: bool = os.environ.get("SM_ENFORCE_SUBMISSION_ORDER", "0") == "1"

    def validate(self, inp: Union[PathLike, pd.DataFrame]) -> List[str]:
        df = _read_input(inp)
        return _validate_structure(df, n_bins=self.n_bins, enforce_order=self.enforce_order)
