# tests/unit/test_submission_validator_property.py
"""
Property-based tests for submission validator.

Covers schema compliance, sigma positivity, NaN/Inf checks, ID uniqueness,
and strict column order using Hypothesis. Aligned with SpectraMind V50 and
the Kaggle submission schema.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any, List

import importlib
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #
N_BINS = 283
ID_COLUMN = "sample_id"
MU_PREFIX, SIGMA_PREFIX = "mu_", "sigma_"
EXPECTED_COLUMNS: List[str] = (
    [ID_COLUMN]
    + [f"{MU_PREFIX}{i:03d}" for i in range(N_BINS)]
    + [f"{SIGMA_PREFIX}{i:03d}" for i in range(N_BINS)]
)

# ----------------------------------------------------------------------------- #
# Typing
# ----------------------------------------------------------------------------- #
ValidatorFn = Callable[[Union[str, Path, pd.DataFrame]], Tuple[bool, List[str]]]

# ----------------------------------------------------------------------------- #
# Validator resolution
# ----------------------------------------------------------------------------- #
def _resolve_validator() -> Optional[ValidatorFn]:
    """
    Locate a submission validator (function or class with .validate).
    Accepts return shapes:
      - (bool, [errors...])
      - [errors...] → coerced to (len==0, list)
      - bool       → coerced to (bool, [])
    """
    candidates = [
        "spectramind.validators.submission:validate_submission",
        "spectramind.validation.submission:validate_submission",
        "spectramind.submission:validate_submission",
        "spectramind.validators.submission:SubmissionValidator",
        "spectramind.validation.submission:SubmissionValidator",
    ]

    def _import_by_path(spec: str) -> Any:
        mod, name = spec.split(":")
        module = importlib.import_module(mod)
        return getattr(module, name)

    for spec in candidates:
        try:
            obj = _import_by_path(spec)
        except Exception:
            continue

        # Function-style API
        if callable(obj) and getattr(obj, "__name__", "").startswith("validate_"):
            def _fn(inp: Union[str, Path, pd.DataFrame]) -> Tuple[bool, List[str]]:
                out = obj(inp)
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
                    return out
                if isinstance(out, list):
                    return (len(out) == 0, out)
                if isinstance(out, bool):
                    return (out, [] if out else ["validation failed"])
                return (False, [f"unexpected return: {out!r}"])
            return _fn

        # Class-style API
        if hasattr(obj, "validate"):
            try:
                inst = obj()
            except Exception:
                continue

            def _fn(inp: Union[str, Path, pd.DataFrame]) -> Tuple[bool, List[str]]:
                out = inst.validate(inp)
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
                    return out
                if isinstance(out, list):
                    return (len(out) == 0, out)
                if isinstance(out, bool):
                    return (out, [] if out else ["validation failed"])
                return (False, [f"unexpected return: {out!r}"])
            return _fn

    return None


@pytest.fixture(scope="session")
def validator() -> ValidatorFn:
    fn = _resolve_validator()
    if fn is None:
        pytest.skip("Submission validator API not found.")
    return fn

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def _df_from_arrays(ids, mus, sigmas) -> pd.DataFrame:
    data = {ID_COLUMN: ids}
    for i in range(N_BINS):
        data[f"{MU_PREFIX}{i:03d}"] = mus[:, i]
    for i in range(N_BINS):
        data[f"{SIGMA_PREFIX}{i:03d}"] = sigmas[:, i]
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "sub.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def _allowed_id() -> st.SearchStrategy[str]:
    # Alphanumerics, underscore, hyphen — keep it simple and schema-friendly.
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return st.text(alphabet=alphabet, min_size=1, max_size=32)


# ----------------------------------------------------------------------------- #
# Hypothesis strategies
# ----------------------------------------------------------------------------- #
st_row_count = st.integers(min_value=1, max_value=7)

def st_valid_df() -> st.SearchStrategy[pd.DataFrame]:
    """Strategy: generate valid submission-like DataFrames with stable RNG."""
    rng = np.random.default_rng(2025)

    @st.composite
    def _mk(draw) -> pd.DataFrame:
        n = draw(st_row_count)
        # Unique, valid IDs
        # Start with deterministic base IDs and optionally replace with random allowed strings
        ids = np.array([f"row_{i}" for i in range(n)], dtype=object)
        # With small probability, replace some with random allowed IDs
        repl_idx = draw(st.lists(st.integers(min_value=0, max_value=n - 1), unique=True, max_size=max(0, n // 2)))
        for i in repl_idx:
            ids[i] = draw(_allowed_id())

        mus = rng.normal(0.0, 0.2, size=(n, N_BINS))
        sigmas = rng.uniform(1e-3, 0.3, size=(n, N_BINS))
        return _df_from_arrays(ids, mus, sigmas)

    return _mk()


# ----------------------------------------------------------------------------- #
# Properties
# ----------------------------------------------------------------------------- #
@settings(max_examples=30, deadline=None)
@given(df=st_valid_df())
def test_valid_frames_always_pass(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    # Path input
    p = _write_csv(tmp_path, df, "ok.csv")
    ok, errs = validator(p)
    assert ok, f"Expected pass (path), got: {errs}"

    # Path-like (string) input
    ok2, errs2 = validator(str(p))
    assert ok2, f"Expected pass (str), got: {errs2}"

    # Direct DataFrame input (if supported)
    ok3, errs3 = validator(df)
    # If API doesn't support DataFrame directly, it will likely return False or error text.
    # We accept either 'pass' or a clear message indicating file-like is required.
    if not ok3:
        joined = " ".join(errs3).lower()
        assert ("dataframe" in joined or "data frame" in joined or "path" in joined or "csv" in joined) or len(errs3) == 0


@settings(max_examples=25, deadline=None)
@given(
    df=st_valid_df(),
    idx=st.integers(min_value=0, max_value=N_BINS - 1),
    bad=st.sampled_from([0.0, -1e-12, -1e-3]),
)
def test_sigma_non_positive_fails(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame, idx: int, bad: float):
    df.at[0, f"{SIGMA_PREFIX}{idx:03d}"] = bad
    p = _write_csv(tmp_path, df, "sigma_nonpos.csv")
    ok, errs = validator(p)
    assert not ok
    assert any(("sigma" in e.lower() and ("pos" in e.lower() or ">" in e or "non" in e.lower())) for e in errs), f"errors={errs}"


@settings(max_examples=25, deadline=None)
@given(df=st_valid_df(), which=st.sampled_from(["nan_mu", "inf_mu", "nan_sigma", "inf_sigma"]))
def test_nan_inf_fail(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame, which: str):
    if which == "nan_mu":
        df.at[0, f"{MU_PREFIX}010"] = np.nan
    elif which == "inf_mu":
        df.at[0, f"{MU_PREFIX}010"] = np.inf
    elif which == "nan_sigma":
        df.at[0, f"{SIGMA_PREFIX}010"] = np.nan
    else:
        df.at[0, f"{SIGMA_PREFIX}010"] = np.inf
    p = _write_csv(tmp_path, df, "nan_inf.csv")
    ok, errs = validator(p)
    assert not ok
    joined = " | ".join(errs).lower()
    assert "nan" in joined or "inf" in joined or "finite" in joined or "valid number" in joined


@settings(max_examples=20, deadline=None)
@given(df=st_valid_df())
def test_missing_or_extra_columns_fail(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    # Remove a known column and add an unknown one
    df = df.drop(columns=[f"{MU_PREFIX}000"])
    df["mystery"] = 123
    p = _write_csv(tmp_path, df, "schema.csv")
    ok, errs = validator(p)
    assert not ok
    j = " | ".join(errs).lower()
    assert ("missing" in j or "schema" in j or "column" in j or "expected" in j) and ("extra" in j or "unknown" in j)


@settings(max_examples=20, deadline=None)
@given(df=st_valid_df())
def test_strict_column_order_required(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    # Permute columns but keep the same set → expect failure if validator requires exact order
    perm = df.columns.tolist()
    if len(perm) > 3:
        # Move a few columns around deterministically
        perm[1], perm[2] = perm[2], perm[1]
        perm[-1], perm[-2] = perm[-2], perm[-1]
    df_perm = df[perm]
    assert list(df_perm.columns) != EXPECTED_COLUMNS  # ensure we actually changed order

    p = _write_csv(tmp_path, df_perm, "order.csv")
    ok, errs = validator(p)
    # Most Kaggle validators require exact order; if your validator is order-invariant,
    # you can relax this assertion to accept ok==True.
    assert not ok, f"Validator accepted permuted column order; errs={errs}"
    j = " | ".join(errs).lower()
    assert "order" in j or "schema" in j or "columns must match" in j or "expected" in j


@settings(max_examples=20, deadline=None)
@given(df=st_valid_df())
def test_id_column_presence_and_uniqueness(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    # Duplicate an ID → should fail
    df_dup = df.copy()
    if len(df_dup) >= 2:
        df_dup.at[1, ID_COLUMN] = df_dup.at[0, ID_COLUMN]
    p_dup = _write_csv(tmp_path, df_dup, "dup_ids.csv")
    ok_dup, errs_dup = validator(p_dup)
    assert not ok_dup
    jd = " | ".join(errs_dup).lower()
    assert "duplicate" in jd or "unique" in jd or "id" in jd

    # Remove ID column entirely → should fail
    df_noid = df.drop(columns=[ID_COLUMN])
    p_noid = _write_csv(tmp_path, df_noid, "no_id.csv")
    ok_noid, errs_noid = validator(p_noid)
    assert not ok_noid
    ji = " | ".join(errs_noid).lower()
    assert "sample_id" in ji or "id" in ji or "missing" in ji or "schema" in ji
