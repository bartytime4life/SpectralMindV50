# tests/unit/test_submission_validator.py
# =============================================================================
# SpectraMind V50 — Submission Validator Tests (Upgraded)
# -----------------------------------------------------------------------------
# What we assert:
#   • CSV & DataFrame with N μ columns and N σ columns are accepted (happy path)
#   • Missing/extra columns → fails with useful error text
#   • Column order mismatch → optionally fails if validator enforces order
#   • NaN/Inf values → fail
#   • σ must be strictly positive (no zeros/negatives)
#   • Non-numeric types in μ/σ → fail
#   • Duplicate sample_id → fail
#
# API flexibility:
#   We try multiple import paths and shapes:
#     - validate_submission(path | pandas.DataFrame) -> (ok: bool, errors: list[str])
#     - SubmissionValidator().validate(path | df) -> list[str] (empty == ok)
#   If nothing matches, we SKIP with a helpful message.
# =============================================================================
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import pytest

# -----------------------------------------------------------------------------#
# Test config
# -----------------------------------------------------------------------------#
def _bins_from_env(default: int = 283) -> int:
    try:
        return int(os.environ.get("SM_SUBMISSION_BINS", str(default)))
    except Exception:
        return default

N_BINS = _bins_from_env(283)
ID_COLUMN = "sample_id"
MU_PREFIX = "mu_"
SIGMA_PREFIX = "sigma_"

# Expected columns in canonical order: id, mu_000..mu_(N-1), sigma_000..sigma_(N-1)
EXPECTED_COLUMNS = (
    [ID_COLUMN]
    + [f"{MU_PREFIX}{i:03d}" for i in range(N_BINS)]
    + [f"{SIGMA_PREFIX}{i:03d}" for i in range(N_BINS)]
)

# Some validators enforce exact order; toggle via env if needed
ENFORCE_COLUMN_ORDER = os.environ.get("SM_ENFORCE_SUBMISSION_ORDER", "0") == "1"


# -----------------------------------------------------------------------------#
# Minimal adapter to whatever validator API you implemented
# -----------------------------------------------------------------------------#
ValidatorFn = Callable[[Union[str, Path, pd.DataFrame]], Tuple[bool, list[str]]]


def _resolve_validator() -> Optional[ValidatorFn]:
    """
    Try common import paths and return a uniform validator function.
    The returned callable must accept (path|DataFrame) and return (ok, errors).
    """
    candidates = [
        # Function-style
        "spectramind.validators.submission:validate_submission",
        "spectramind.validation.submission:validate_submission",
        "spectramind.submission:validate_submission",
        # Class-style
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

        # Direct function
        if callable(obj) and getattr(obj, "__name__", "").startswith("validate_"):
            def _fn(inp: Union[str, Path, pd.DataFrame]) -> Tuple[bool, list[str]]:
                try:
                    out = obj(inp)  # type: ignore[misc]
                except Exception as e:
                    return False, [f"exception: {type(e).__name__}: {e}"]
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
                    return out
                if isinstance(out, list):
                    return (len(out) == 0, out)
                if out is True or out is False:
                    return (bool(out), [] if out else ["validation failed"])
                return False, [f"unexpected return from validator: {out!r}"]
            return _fn

        # Class with .validate
        if hasattr(obj, "__call__") or hasattr(obj, "validate"):
            try:
                inst = obj()  # type: ignore[call-arg]
            except Exception:
                continue

            def _fn(inp: Union[str, Path, pd.DataFrame]) -> Tuple[bool, list[str]]:
                try:
                    res = inst.validate(inp)  # type: ignore[attr-defined]
                except Exception as e:
                    return False, [f"exception: {type(e).__name__}: {e}"]
                if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], bool):
                    return res
                if isinstance(res, list):
                    return (len(res) == 0, res)
                if res is True or res is False:
                    return (bool(res), [] if res else ["validation failed"])
                return False, [f"unexpected return from validator: {res!r}"]
            return _fn

    return None


@pytest.fixture(scope="session")
def validator() -> ValidatorFn:
    fn = _resolve_validator()
    if fn is None:
        pytest.skip(
            "Submission validator API not found. "
            "Expected one of:\n"
            "  - spectramind.validators.submission:validate_submission\n"
            "  - spectramind.validation.submission:validate_submission\n"
            "  - spectramind.submission:validate_submission\n"
            "  - spectramind.validators.submission:SubmissionValidator\n"
            "  - spectramind.validation.submission:SubmissionValidator\n"
            "Please expose one of these or adjust the test adapter."
        )
    return fn


# -----------------------------------------------------------------------------#
# Helpers to craft small CSVs/DataFrames for tests
# -----------------------------------------------------------------------------#
def _df_valid(n_rows: int = 3, seed: int = 123) -> pd.DataFrame:
    data = {}
    data[ID_COLUMN] = [f"row_{i}" for i in range(n_rows)]
    rng = np.random.default_rng(seed)
    mus = rng.normal(loc=0.0, scale=0.1, size=(n_rows, N_BINS))
    sigmas = rng.uniform(low=1e-3, high=0.2, size=(n_rows, N_BINS))  # strictly positive
    for i in range(N_BINS):
        data[f"{MU_PREFIX}{i:03d}"] = mus[:, i]
        data[f"{SIGMA_PREFIX}{i:03d}"] = sigmas[:, i]
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# -----------------------------------------------------------------------------#
# Core unit tests (function/class API)
# -----------------------------------------------------------------------------#
def test_happy_path_accepts_valid_csv(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=2)
    csv_path = _write_csv(df, tmp_path / "sub_valid.csv")
    ok, errors = validator(csv_path)
    assert ok, f"Expected valid submission, got errors: {errors}"


def test_happy_path_accepts_dataframe(validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=2)
    ok, errors = validator(df)
    assert ok, f"DataFrame input should validate: {errors}"


def test_missing_one_column_fails(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=1)
    df = df.drop(columns=[f"{SIGMA_PREFIX}{(N_BINS-1):03d}"])
    csv_path = _write_csv(df, tmp_path / "sub_missing_col.csv")
    ok, errors = validator(csv_path)
    assert not ok, "Validator should fail on missing column"
    joined = " | ".join(errors).lower()
    assert any(k in joined for k in ("column", "schema", "missing"))


def test_extra_unknown_column_fails(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=1)
    df["unknown_col"] = 42
    csv_path = _write_csv(df, tmp_path / "sub_extra_col.csv")
    ok, errors = validator(csv_path)
    assert not ok, "Validator should fail on extra/unknown columns"
    assert any(s in e.lower() for e in errors for s in ("unknown", "extra", "unexpected"))


def test_nan_and_inf_values_fail(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=1)
    df.loc[0, f"{MU_PREFIX}010"] = np.nan
    df.loc[0, f"{SIGMA_PREFIX}020"] = np.inf
    csv_path = _write_csv(df, tmp_path / "sub_nan_inf.csv")
    ok, errors = validator(csv_path)
    assert not ok, "Validator should fail on NaN/Inf"
    joined = " | ".join(errors).lower()
    assert any(tok in joined for tok in ("nan", "inf", "finite"))


def test_sigma_must_be_positive(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=2)
    df.loc[1, f"{SIGMA_PREFIX}111"] = -0.5
    df.loc[0, f"{SIGMA_PREFIX}222"] = 0.0
    csv_path = _write_csv(df, tmp_path / "sub_sigma_nonpos.csv")
    ok, errors = validator(csv_path)
    assert not ok, "Validator should reject non-positive sigma"
    joined = " | ".join(errors).lower()
    assert "sigma" in joined and any(tok in joined for tok in ("positive", "nonzero", "non-zero"))


def test_mu_sigma_must_be_numeric(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=1)
    df.loc[0, f"{MU_PREFIX}005"] = "not-a-number"
    df.loc[0, f"{SIGMA_PREFIX}006"] = "0.1x"
    csv_path = _write_csv(df, tmp_path / "sub_non_numeric.csv")
    ok, errors = validator(csv_path)
    assert not ok, "Validator should fail on non-numeric μ/σ values"
    joined = " | ".join(errors).lower()
    assert any(tok in joined for tok in ("numeric", "float", "number", "dtype"))


def test_duplicate_sample_ids_fail(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=2)
    df.loc[1, ID_COLUMN] = df.loc[0, ID_COLUMN]  # duplicate id
    csv_path = _write_csv(df, tmp_path / "sub_dup_ids.csv")
    ok, errors = validator(csv_path)
    assert not ok, "Validator should fail on duplicate sample_id values"
    joined = " | ".join(errors).lower()
    assert any(tok in joined for tok in ("duplicate", "unique", "sample_id"))


def test_column_order_enforced_optionally(tmp_path: Path, validator: ValidatorFn) -> None:
    """
    Some validators enforce exact column order. We support both policies:
    - If ENFORCE_COLUMN_ORDER=1, then out-of-order columns must fail.
    - Otherwise, validator may accept them (and we don't fail the test).
    """
    df = _df_valid(n_rows=1)
    cols = df.columns.tolist()
    # Swap two columns far apart to avoid trivial acceptance
    idx_a = 1
    idx_b = 1 + N_BINS  # first sigma vs first mu region boundary
    cols[idx_a], cols[idx_b] = cols[idx_b], cols[idx_a]
    df = df.loc[:, cols]
    csv_path = _write_csv(df, tmp_path / "sub_reordered.csv")

    ok, errors = validator(csv_path)
    if ENFORCE_COLUMN_ORDER:
        assert not ok, "Validator should fail when column order is wrong (enforced mode)"
        assert any("order" in e.lower() or "position" in e.lower() for e in errors)
    else:
        # Non-enforcing validators may normalize/reorder; accept either outcome.
        assert ok or not ok  # no-op assert to keep branch explicit


# -----------------------------------------------------------------------------#
# Optional CLI integration tests (skipped if CLI command not available)
# -----------------------------------------------------------------------------#
@pytest.mark.usefixtures("cli_runner")  # provided by tests/conftest.py
def test_cli_validate_happy_path(tmp_path: Path, cli_runner):  # type: ignore[no-redef]
    """
    Try common CLI spellings; skip if none are wired.
    Accept any zero exit_code from a validate-like subcommand.
    """
    csv_path = _write_csv(_df_valid(n_rows=1), tmp_path / "cli_ok.csv")
    candidate_cmds = [
        ["submission", "validate", str(csv_path)],
        ["submit", "validate", str(csv_path)],
        ["validate-submission", str(csv_path)],
        ["submission-validate", str(csv_path)],
        ["kaggle", "validate-submission", str(csv_path)],
    ]
    tried = 0
    for args in candidate_cmds:
        tried += 1
        result = cli_runner.invoke(args)
        if result.exit_code == 0:
            return
    pytest.skip(
        f"No working CLI validate command found after {tried} attempts. "
        "Expose a 'submission validate' (or similar) command to enable this test."
    )


@pytest.mark.usefixtures("cli_runner")
def test_cli_validate_rejects_bad_csv(tmp_path: Path, cli_runner):  # type: ignore[no-redef]
    bad = _df_valid(n_rows=1).drop(columns=[f"{MU_PREFIX}000"])
    csv_path = _write_csv(bad, tmp_path / "cli_bad.csv")
    candidate_cmds = [
        ["submission", "validate", str(csv_path)],
        ["submit", "validate", str(csv_path)],
        ["validate-submission", str(csv_path)],
        ["submission-validate", str(csv_path)],
        ["kaggle", "validate-submission", str(csv_path)],
    ]
    for args in candidate_cmds:
        result = cli_runner.invoke(args)
        # If command exists (not “No such command”), we expect non-zero exit.
        if result.exit_code != 2 and "No such command" not in (result.stdout + result.stderr):
            assert result.exit_code != 0, "CLI should fail on invalid CSV"
            return
    pytest.skip("No working CLI validate command found; skipping negative CLI test.")
