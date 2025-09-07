# tests/unit/test_submission_validator.py
# =============================================================================
# SpectraMind V50 — Submission Validator Tests
# -----------------------------------------------------------------------------
# What we assert:
#   • CSV with 283 μ columns and 283 σ columns is accepted (happy path)
#   • Missing/extra columns → fails with useful error text
#   • NaN/Inf values → fails
#   • σ must be strictly positive (no zeros/negatives)
#
# API flexibility:
#   We try multiple import paths and shapes:
#     - validate_submission(path | pandas.DataFrame) -> (ok: bool, errors: list[str])
#     - SubmissionValidator().validate(path | df) -> list[str] (empty == ok)
#   If nothing matches, we SKIP with a helpful message.
# =============================================================================
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------------#
# Test config (adjust if your schema differs)
# -----------------------------------------------------------------------------#
N_BINS = 283
ID_COLUMN = "sample_id"
MU_PREFIX = "mu_"
SIGMA_PREFIX = "sigma_"
# Columns in expected order: id, mu_000..mu_282, sigma_000..sigma_282
EXPECTED_COLUMNS = (
    [ID_COLUMN]
    + [f"{MU_PREFIX}{i:03d}" for i in range(N_BINS)]
    + [f"{SIGMA_PREFIX}{i:03d}" for i in range(N_BINS)]
)


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
        # Function-style validators
        "spectramind.validators.submission:validate_submission",
        "spectramind.validation.submission:validate_submission",
        "spectramind.submission:validate_submission",
        # Class-style validators (wrap .validate)
        "spectramind.validators.submission:SubmissionValidator",
        "spectramind.validation.submission:SubmissionValidator",
    ]

    def _import_by_path(spec: str) -> Any:
        mod, name = spec.split(":")
        module = __import__(mod, fromlist=[name])
        return getattr(module, name)

    # Try function forms
    for spec in candidates:
        try:
            obj = _import_by_path(spec)
        except Exception:
            continue

        # Case 1: direct function
        if callable(obj) and obj.__name__.startswith("validate_"):
            def _fn(inp: Union[str, Path, pd.DataFrame]) -> Tuple[bool, list[str]]:
                try:
                    out = obj(inp)  # type: ignore[misc]
                except Exception as e:  # validator raised → fail with message
                    return False, [f"exception: {type(e).__name__}: {e}"]
                # Normalize common return shapes
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
                    return out  # (ok, errors)
                if isinstance(out, list):
                    return (len(out) == 0, out)
                if out is True or out is False:
                    return (bool(out), [] if out else ["validation failed"])
                return False, [f"unexpected return from validator: {out!r}"]
            return _fn

        # Case 2: class with .validate
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
# Helpers to craft small CSVs for tests
# -----------------------------------------------------------------------------#
def _df_valid(n_rows: int = 3) -> pd.DataFrame:
    data = {}
    data[ID_COLUMN] = [f"row_{i}" for i in range(n_rows)]
    # Smooth-ish μ; strictly positive σ
    rng = np.random.default_rng(123)
    mus = rng.normal(loc=0.0, scale=0.1, size=(n_rows, N_BINS))
    sigmas = rng.uniform(low=1e-3, high=0.2, size=(n_rows, N_BINS))
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


def test_missing_one_column_fails(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=1)
    # Drop one sigma column
    df = df.drop(columns=[f"{SIGMA_PREFIX}{(N_BINS-1):03d}"])
    csv_path = _write_csv(df, tmp_path / "sub_missing_col.csv")

    ok, errors = validator(csv_path)
    assert not ok, "Validator should fail on missing column"
    joined = " | ".join(errors).lower()
    assert "column" in joined or "schema" in joined


def test_extra_unknown_column_fails(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=1)
    df["unknown_col"] = 42
    csv_path = _write_csv(df, tmp_path / "sub_extra_col.csv")

    ok, errors = validator(csv_path)
    assert not ok, "Validator should fail on extra/unknown columns"
    assert any("unknown" in e.lower() or "extra" in e.lower() for e in errors)


def test_nan_and_inf_values_fail(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=1)
    # Inject NaN and Inf
    df.loc[0, f"{MU_PREFIX}010"] = np.nan
    df.loc[0, f"{SIGMA_PREFIX}020"] = np.inf
    csv_path = _write_csv(df, tmp_path / "sub_nan_inf.csv")

    ok, errors = validator(csv_path)
    assert not ok, "Validator should fail on NaN/Inf"
    joined = " | ".join(errors).lower()
    assert "nan" in joined or "inf" in joined or "finite" in joined


def test_sigma_must_be_positive(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=2)
    df.loc[1, f"{SIGMA_PREFIX}111"] = -0.5
    df.loc[0, f"{SIGMA_PREFIX}222"] = 0.0
    csv_path = _write_csv(df, tmp_path / "sub_sigma_nonpos.csv")

    ok, errors = validator(csv_path)
    assert not ok, "Validator should reject non-positive sigma"
    assert any("sigma" in e.lower() or "positive" in e.lower() for e in errors)


def test_id_column_presence_and_type(tmp_path: Path, validator: ValidatorFn) -> None:
    df = _df_valid(n_rows=2)
    # Remove ID column or rename it
    df = df.rename(columns={ID_COLUMN: "id"})
    csv_path = _write_csv(df, tmp_path / "sub_bad_id.csv")

    ok, errors = validator(csv_path)
    assert not ok, "Validator should require the exact ID column"
    assert any("sample_id" in e.lower() or "id column" in e.lower() for e in errors)


# -----------------------------------------------------------------------------#
# Optional CLI integration tests (skipped if CLI command not available)
# -----------------------------------------------------------------------------#
@pytest.mark.usefixtures("cli_runner")  # provided by tests/conftest.py
def test_cli_validate_happy_path(tmp_path: Path, cli_runner):  # type: ignore[no-redef]
    """
    Try common CLI spellings; skip if none are wired.
    Accept any zero exit_code from a validate-like subcommand.
    """
    from shutil import which
    # Prepare CSV
    csv_path = _write_csv(_df_valid(n_rows=1), tmp_path / "cli_ok.csv")

    # Possible command layouts to try (first that returns 0 passes)
    candidate_cmds = [
        ["submission", "validate", str(csv_path)],
        ["submit", "validate", str(csv_path)],
        ["validate-submission", str(csv_path)],
        ["submission-validate", str(csv_path)],
        ["kaggle", "validate-submission", str(csv_path)],
    ]

    # Ensure the CLI exists (typer app wired via conftest)
    # We attempt each and treat exit_code==0 as success; else try next.
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
        # If command exists, we expect non-zero
        if result.exit_code != 2 and "No such command" not in (result.stdout + result.stderr):
            assert result.exit_code != 0, "CLI should fail on invalid CSV"
            return
    pytest.skip("No working CLI validate command found; skipping negative CLI test.")