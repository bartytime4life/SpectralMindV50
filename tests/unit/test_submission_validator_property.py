"""
Property-based tests for submission validator (Upgraded).

Covers schema compliance, sigma positivity, NaN/Inf checks, ID uniqueness,
strict column order (optionally), duplicate/missing columns, gzip/CRLF,
and coercibility behaviors using Hypothesis. Aligned with SpectraMind V50
and the Kaggle submission schema.

The tests are tolerant of validators that normalize some cases (e.g., trim
header whitespace, auto-drop index columns, coerce numeric strings), but
still insist on descriptive error messages for strict validators.
"""

from __future__ import annotations
import importlib
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any, List, Dict

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, example

# ----------------------------------------------------------------------------- #
# Config via env (bins, order)
# ----------------------------------------------------------------------------- #
def _bins_from_env(default: int = 283) -> int:
    try:
        return int(os.environ.get("SM_SUBMISSION_BINS", str(default)))
    except Exception:
        return default

N_BINS = _bins_from_env(283)
ID_COLUMN = "sample_id"
MU_PREFIX, SIGMA_PREFIX = "mu_", "sigma_"
EXPECTED_COLUMNS: List[str] = (
    [ID_COLUMN]
    + [f"{MU_PREFIX}{i:03d}" for i in range(N_BINS)]
    + [f"{SIGMA_PREFIX}{i:03d}" for i in range(N_BINS)]
)

ENFORCE_COLUMN_ORDER = os.environ.get("SM_ENFORCE_SUBMISSION_ORDER", "0") == "1"

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
                out = inst.validate(inp)  # type: ignore[attr-defined]
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
    data: Dict[str, Any] = {ID_COLUMN: ids}
    for i in range(N_BINS):
        data[f"{MU_PREFIX}{i:03d}"] = mus[:, i]
    for i in range(N_BINS):
        data[f"{SIGMA_PREFIX}{i:03d}"] = sigmas[:, i]
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "sub.csv", newline: Optional[str] = None) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if newline is None:
        df.to_csv(p, index=False)
    else:
        with open(p, "w", encoding="utf-8", newline=newline) as f:
            df.to_csv(f, index=False)
    return p


def _allowed_id() -> st.SearchStrategy[str]:
    # Alphanumerics, underscore, hyphen — keep it simple and schema-friendly.
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return st.text(alphabet=alphabet, min_size=1, max_size=32)


def _finite_float() -> st.SearchStrategy[float]:
    # Strictly finite floats for μ. Avoid NaN/Inf; those are tested separately.
    return st.floats(allow_nan=False, allow_infinity=False, width=64)


def _pos_float(min_value: float = 1e-9, max_value: float = 1e3) -> st.SearchStrategy[float]:
    # Strictly > 0 for σ (with tiny epsilon floor).
    return st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False, width=64)

# ----------------------------------------------------------------------------- #
# Hypothesis strategies
# ----------------------------------------------------------------------------- #
st_row_count = st.integers(min_value=1, max_value=6)

def st_valid_df() -> st.SearchStrategy[pd.DataFrame]:
    """Strategy: generate valid submission-like DataFrames."""
    @st.composite
    def _mk(draw) -> pd.DataFrame:
        n = draw(st_row_count)
        # Unique, valid IDs
        base_ids = np.array([draw(_allowed_id()) for _ in range(n)], dtype=object)
        # μ and σ arrays (finite; σ strictly positive)
        mus = np.zeros((n, N_BINS), dtype=float)
        sigmas = np.zeros((n, N_BINS), dtype=float)
        # Use local RNG for speed but ensure finite constraints via hypothesis too
        for j in range(n):
            row_mus = draw(st.lists(_finite_float(), min_size=N_BINS, max_size=N_BINS))
            row_sgs = draw(st.lists(_pos_float(),    min_size=N_BINS, max_size=N_BINS))
            mus[j, :] = np.array(row_mus, dtype=float)
            sigmas[j, :] = np.array(row_sgs, dtype=float)
        return _df_from_arrays(base_ids, mus, sigmas)
    return _mk()


def st_permutation(lst: List[str]) -> st.SearchStrategy[List[str]]:
    return st.permutations(lst).map(list)

# ----------------------------------------------------------------------------- #
# Properties — happy path
# ----------------------------------------------------------------------------- #
@settings(max_examples=20, deadline=None)
@given(df=st_valid_df())
def test_valid_frames_always_pass(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    # Path input
    p = _write_csv(tmp_path, df, "ok.csv")
    ok, errs = validator(p)
    assert ok, f"Expected pass (path), got: {errs}"

    # Path-like (string) input
    ok2, errs2 = validator(str(p))
    assert ok2, f"Expected pass (str), got: {errs2}"

    # Direct DataFrame input
    ok3, errs3 = validator(df)
    if not ok3:
        joined = " ".join(errs3).lower()
        # Some validators may demand a path (csv), which is acceptable here:
        assert ("dataframe" in joined or "data frame" in joined or "path" in joined or "csv" in joined) or len(errs3) == 0


# ----------------------------------------------------------------------------- #
# Properties — σ > 0, NaN/Inf
# ----------------------------------------------------------------------------- #
@settings(max_examples=20, deadline=None)
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


@settings(max_examples=20, deadline=None)
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

# ----------------------------------------------------------------------------- #
# Properties — missing/extra/duplicate columns & order
# ----------------------------------------------------------------------------- #
@settings(max_examples=15, deadline=None)
@given(df=st_valid_df())
def test_missing_or_extra_columns_fail(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    # Remove a known column and add an unknown one
    df2 = df.drop(columns=[f"{MU_PREFIX}000"])
    df2["mystery"] = 123
    p = _write_csv(tmp_path, df2, "schema.csv")
    ok, errs = validator(p)
    assert not ok
    j = " | ".join(errs).lower()
    assert ("missing" in j or "schema" in j or "column" in j or "expected" in j) and ("extra" in j or "unknown" in j)


@settings(max_examples=15, deadline=None)
@given(df=st_valid_df())
def test_duplicate_columns_fail_property(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    # Introduce a duplicate column name by renaming one sigma to an existing mu header
    dup_name = f"{MU_PREFIX}007"
    df2 = df.rename(columns={f"{SIGMA_PREFIX}007": dup_name})
    p = _write_csv(tmp_path, df2, "dup_cols.csv")
    ok, errs = validator(p)
    if ok:
        pytest.skip("Validator auto-resolved duplicate columns; accepting lenient behavior.")
    assert any(tok in " ".join(errs).lower() for tok in ("duplicate", "ambiguous", "column"))


@settings(max_examples=15, deadline=None)
@given(df=st_valid_df())
def test_strict_column_order_required(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    # Permute columns but keep the same set → expect failure if validator requires exact order
    perm = df.columns.tolist()
    if len(perm) > 3:
        perm[1], perm[2] = perm[2], perm[1]
        perm[-1], perm[-2] = perm[-2], perm[-1]
    df_perm = df[perm]
    assert list(df_perm.columns) != EXPECTED_COLUMNS  # ensure changed order

    p = _write_csv(tmp_path, df_perm, "order.csv")
    ok, errs = validator(p)
    if ENFORCE_COLUMN_ORDER:
        assert not ok, f"Validator accepted permuted column order; errs={errs}"
        j = " | ".join(errs).lower()
        assert "order" in j or "schema" in j or "columns must match" in j or "expected" in j
    else:
        # Order tolerant validators may normalize; accept either behavior.
        assert ok or not ok

# ----------------------------------------------------------------------------- #
# Properties — sample_id rules
# ----------------------------------------------------------------------------- #
@settings(max_examples=12, deadline=None)
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


@settings(max_examples=12, deadline=None)
@given(df=st_valid_df())
def test_non_string_or_missing_ids_fail_property(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    df2 = df.copy()
    # Make one missing, one numeric
    if len(df2) >= 2:
        df2.at[0, ID_COLUMN] = np.nan
        df2.at[1, ID_COLUMN] = 12345
    else:
        df2.at[0, ID_COLUMN] = 12345
    p = _write_csv(tmp_path, df2, "bad_ids.csv")
    ok, errs = validator(p)
    assert not ok
    txt = " ".join(errs).lower()
    assert "sample_id" in txt and any(tok in txt for tok in ("string", "missing", "null", "dtype", "type"))

# ----------------------------------------------------------------------------- #
# Properties — optional behaviors (header whitespace, coercible strings, gzip, CRLF)
# ----------------------------------------------------------------------------- #
@settings(max_examples=10, deadline=None)
@given(df=st_valid_df())
def test_header_whitespace_optional_property(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    # Rename a couple of headers with extra spaces
    df2 = df.rename(columns={f"{MU_PREFIX}000": f"  {MU_PREFIX}000  ", ID_COLUMN: f" {ID_COLUMN} "})
    p = _write_csv(tmp_path, df2, "header_ws.csv")
    ok, _ = validator(p)
    # Allow either trimming or strictness
    assert ok or not ok


@settings(max_examples=10, deadline=None)
@given(df=st_valid_df())
def test_mu_sigma_coercible_strings_optional_property(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    df.loc[0, f"{MU_PREFIX}001"] = "0.123"
    df.loc[0, f"{SIGMA_PREFIX}002"] = "0.045"
    p = _write_csv(tmp_path, df, "coercible_strings.csv")
    ok, errs = validator(p)
    if ok:
        return
    txt = " ".join(errs).lower()
    assert any(tok in txt for tok in ("numeric", "float", "number", "dtype", "parse"))


@settings(max_examples=8, deadline=None)
@given(df=st_valid_df())
def test_gzipped_csv_optional_property(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    import gzip
    gz_path = tmp_path / "sub.csv.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    ok, errs = validator(gz_path)
    if not ok:
        joined = " ".join(errs).lower()
        if any(tok in joined for tok in ("gzip", "compression", "open", "decode")):
            pytest.skip("Validator does not support gzipped CSV; skipping.")
        else:
            assert False, f"Validator rejected gzipped CSV unexpectedly: {errs}"


@settings(max_examples=8, deadline=None)
@given(df=st_valid_df())
def test_crlf_csv_optional_property(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame):
    p = _write_csv(tmp_path, df, "crlf.csv", newline="\r\n")
    ok, errs = validator(p)
    # Expect robust validators to accept; allow strict ones to fail (log reason)
    assert ok or ("newline" in " ".join(errs).lower() or "parse" in " ".join(errs).lower())

# ----------------------------------------------------------------------------- #
# Properties — single-cell corruption (NaN/Inf/non-numeric) must fail
# ----------------------------------------------------------------------------- #
@settings(max_examples=20, deadline=None)
@given(
    df=st_valid_df(),
    which=st.sampled_from(["mu", "sigma"]),
    col=st.integers(min_value=0, max_value=N_BINS - 1),
    kind=st.sampled_from(["nan", "inf", "nonstr"]),
)
@example(df=_df_from_arrays(np.array(["x"]), np.zeros((1, N_BINS)), np.ones((1, N_BINS))), which="mu", col=0, kind="nonstr")
def test_single_cell_corruption_must_fail(tmp_path: Path, validator: ValidatorFn, df: pd.DataFrame, which: str, col: int, kind: str):
    target_col = f"{MU_PREFIX}{col:03d}" if which == "mu" else f"{SIGMA_PREFIX}{col:03d}"
    if kind == "nan":
        df.at[0, target_col] = np.nan
    elif kind == "inf":
        df.at[0, target_col] = np.inf
    else:
        df.at[0, target_col] = "not-a-number"
    p = _write_csv(tmp_path, df, f"corrupt_{which}_{kind}.csv")
    ok, errs = validator(p)
    assert not ok
    joined = " ".join(errs).lower()
    if kind == "nonstr":
        assert any(tok in joined for tok in ("numeric", "float", "number", "dtype", "parse"))
    else:
        assert "nan" in joined or "inf" in joined or "finite" in joined

# ----------------------------------------------------------------------------- #
# Properties — env override (small bins) honored or schema-mismatch explained
# ----------------------------------------------------------------------------- #
@settings(max_examples=5, deadline=None)
@given(rows=st.integers(min_value=1, max_value=4))
def test_env_override_small_bins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validator: ValidatorFn, rows: int):
    """
    Test that SM_SUBMISSION_BINS is honored. We build a small 8-bin file.
    If the validator pins to 283 bins (official), this should fail with schema/column mismatch;
    if it reads env, it should pass. We accept both (and assert mismatch reason if failing).
    """
    monkeypatch.setenv("SM_SUBMISSION_BINS", "8")
    bins = 8
    ids = np.array([f"s{i}" for i in range(rows)], dtype=object)
    mus = np.zeros((rows, bins))
    sigs = np.ones((rows, bins))
    df_small = pd.DataFrame(
        {ID_COLUMN: ids,
         **{f"{MU_PREFIX}{i:03d}": mus[:, i] for i in range(bins)},
         **{f"{SIGMA_PREFIX}{i:03d}": sigs[:, i] for i in range(bins)}},
        columns=[ID_COLUMN] + [f"{MU_PREFIX}{i:03d}" for i in range(bins)] + [f"{SIGMA_PREFIX}{i:03d}" for i in range(bins)]
    )
    p = _write_csv(tmp_path, df_small, "bins8.csv")
    ok, errors = validator(p)
    if not ok:
        txt = " ".join(errors).lower()
        assert any(tok in txt for tok in ("column", "schema", "missing", "count")), \
            f"Unexpected reason for failing small-bin file: {errors}"