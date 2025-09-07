# tests/unit/test_submission_validator_property.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

N_BINS = 283
ID_COLUMN = "sample_id"
MU_PREFIX = "mu_"
SIGMA_PREFIX = "sigma_"
EXPECTED_COLUMNS = (
    [ID_COLUMN]
    + [f"{MU_PREFIX}{i:03d}" for i in range(N_BINS)]
    + [f"{SIGMA_PREFIX}{i:03d}" for i in range(N_BINS)]
)

ValidatorFn = callable

def _resolve_validator() -> Optional[callable]:
    candidates = [
        "spectramind.validators.submission:validate_submission",
        "spectramind.validation.submission:validate_submission",
        "spectramind.submission:validate_submission",
        "spectramind.validators.submission:SubmissionValidator",
        "spectramind.validation.submission:SubmissionValidator",
    ]
    def _import_by_path(spec: str) -> Any:
        mod, name = spec.split(":")
        module = __import__(mod, fromlist=[name])
        return getattr(module, name)

    for spec in candidates:
        try:
            obj = _import_by_path(spec)
        except Exception:
            continue
        if callable(obj) and getattr(obj, "__name__", "").startswith("validate_"):
            def _fn(inp):
                out = obj(inp)
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
                    return out
                if isinstance(out, list):
                    return (len(out) == 0, out)
                if isinstance(out, bool):
                    return (out, [] if out else ["validation failed"])
                return (False, [f"unexpected return: {out!r}"])
            return _fn
        if hasattr(obj, "validate"):
            try:
                inst = obj()
            except Exception:
                continue
            def _fn(inp):
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
        pytest.skip("Submission validator API not found (see unit/test_submission_validator.py for expected paths).")
    return fn

def _df_from_arrays(ids, mus, sigmas) -> pd.DataFrame:
    data = {ID_COLUMN: ids}
    for i in range(N_BINS):
        data[f"{MU_PREFIX}{i:03d}"] = mus[:, i]
    for i in range(N_BINS):
        data[f"{SIGMA_PREFIX}{i:03d}"] = sigmas[:, i]
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)

# ------------------------ Hypothesis strategies ------------------------ #
st_row_count = st.integers(min_value=1, max_value=7)

def st_valid_df():
    def _mk(n):
        rng = np.random.default_rng(2025)
        ids = np.array([f"row_{i}" for i in range(n)])
        mus = rng.normal(0.0, 0.2, size=(n, N_BINS))
        sigmas = rng.uniform(1e-3, 0.3, size=(n, N_BINS))
        return _df_from_arrays(ids, mus, sigmas)
    return st_row_count.map(_mk)

# ----------------------------- Properties ------------------------------ #
@settings(max_examples=30, deadline=None)
@given(df=st_valid_df())
def test_valid_frames_always_pass(tmp_path: Path, validator, df: pd.DataFrame):
    p = tmp_path / "ok.csv"
    df.to_csv(p, index=False)
    ok, errs = validator(p)
    assert ok, f"Expected pass, got: {errs}"

@settings(max_examples=25, deadline=None)
@given(df=st_valid_df(), idx=st.integers(min_value=0, max_value=N_BINS-1))
def test_sigma_non_positive_fails(tmp_path, validator, df, idx):
    # choose a random row and flip sign/zero
    r = 0
    df[f"{SIGMA_PREFIX}{idx:03d}"].iloc[r] = st.sampled_from([0.0, -1e-3]).example()
    p = tmp_path / "sigma_nonpos.csv"
    df.to_csv(p, index=False)
    ok, errs = validator(p)
    assert not ok
    assert any("sigma" in e.lower() or "positive" in e.lower() for e in errs)

@settings(max_examples=25, deadline=None)
@given(df=st_valid_df(), which=st.sampled_from(["nan_mu", "inf_mu", "nan_sigma", "inf_sigma"]))
def test_nan_inf_fail(tmp_path, validator, df, which):
    r, c = 0, 10
    if which == "nan_mu":
        df[f"{MU_PREFIX}{c:03d}"].iloc[r] = np.nan
    elif which == "inf_mu":
        df[f"{MU_PREFIX}{c:03d}"].iloc[r] = np.inf
    elif which == "nan_sigma":
        df[f"{SIGMA_PREFIX}{c:03d}"].iloc[r] = np.nan
    else:
        df[f"{SIGMA_PREFIX}{c:03d}"].iloc[r] = np.inf
    p = tmp_path / "nan_inf.csv"
    df.to_csv(p, index=False)
    ok, errs = validator(p)
    assert not ok
    joined = " | ".join(errs).lower()
    assert "nan" in joined or "inf" in joined or "finite" in joined

@settings(max_examples=20, deadline=None)
@given(df=st_valid_df())
def test_missing_or_extra_columns_fail(tmp_path, validator, df):
    # Remove one column, add an unknown one
    df = df.drop(columns=[f"{MU_PREFIX}000"])
    df["mystery"] = 123
    p = tmp_path / "schema.csv"
    df.to_csv(p, index=False)
    ok, errs = validator(p)
    assert not ok
    j = " | ".join(errs).lower()
    assert ("missing" in j or "schema" in j or "column" in j) and ("extra" in j or "unknown" in j)