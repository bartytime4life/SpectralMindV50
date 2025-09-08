from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import pytest

# Path to canonical golden submission artifact
GOLDEN = (Path(__file__).parent / "submission_valid.csv").resolve()

# Candidate CLI validate entrypoints to probe; keep most-specific first
_CLI_VALIDATORS: tuple[Sequence[str], ...] = (
    ("submission", "validate"),
    ("submit", "validate"),
    ("validate-submission",),
    ("submission-validate",),
    ("kaggle", "validate-submission"),
)


def _find_working_cli_validate(cli_runner, candidate_cmds: Iterable[Sequence[str]]) -> Optional[Sequence[str]]:
    """
    Return the first CLI command prefix that accepts '... <path>' and exits 0 on validation success.
    """
    for prefix in candidate_cmds:
        res = cli_runner.invoke([*prefix, str(GOLDEN)])
        if res.exit_code == 0:
            return prefix
    return None


@pytest.mark.require_golden
def test_golden_file_exists_and_is_valid(validator):
    """
    The repo must include a golden, schema-valid submission CSV so downstream regression tests
    and CLI smoke tests have a stable baseline.
    """
    assert GOLDEN.exists(), "Add tests/golden/submission_valid.csv to the repo"
    ok, errs = validator(GOLDEN)
    assert ok, f"Golden submission should be valid, but validator reported: {errs}"


@pytest.mark.usefixtures("cli_runner")
@pytest.mark.require_golden
def test_cli_accepts_golden(cli_runner):
    """
    Ensure at least one exposed CLI validate entrypoint can validate the golden artifact.
    """
    assert GOLDEN.exists(), "Add tests/golden/submission_valid.csv to the repo"
    prefix = _find_working_cli_validate(cli_runner, _CLI_VALIDATORS)
    if prefix is None:
        pytest.skip(
            "No working CLI validate command found. "
            "Expose one of: "
            + ", ".join(" ".join(c) for c in _CLI_VALIDATORS)
        )
    # Sanity: the discovered prefix should succeed with the golden
    res = cli_runner.invoke([*prefix, str(GOLDEN)])
    assert res.exit_code == 0, f"CLI validator `{ ' '.join(prefix) }` should accept golden: {res.output}"


@pytest.mark.usefixtures("cli_runner")
@pytest.mark.require_golden
def test_cli_validates_copy_in_tempdir(cli_runner, tmp_path: Path):
    """
    Validation should work for any path, not only the canonical golden location.
    """
    assert GOLDEN.exists(), "Add tests/golden/submission_valid.csv to the repo"
    copied = tmp_path / "submission_copy.csv"
    shutil.copy2(GOLDEN, copied)
    # schema validator
    ok, errs = pytest.lazy_fixture("validator")(copied)  # use same schema adapter as other tests
    assert ok, f"Copied submission should validate via schema: {errs}"

    # CLI validator (if available)
    prefix = _find_working_cli_validate(cli_runner, _CLI_VALIDATORS)
    if prefix is None:
        pytest.skip("No working CLI validate command found for temp path check.")
    res = cli_runner.invoke([*prefix, str(copied)])
    assert res.exit_code == 0, f"CLI validator should accept copied submission: {res.output}"


@pytest.mark.usefixtures("cli_runner")
@pytest.mark.require_golden
def test_corrupted_submission_is_rejected(cli_runner, tmp_path: Path, validator):
    """
    Tamper with the header to create an invalid submission and ensure both the schema validator
    and CLI validator (if present) reject it.
    """
    assert GOLDEN.exists(), "Add tests/golden/submission_valid.csv to the repo"

    bad = tmp_path / "submission_bad.csv"
    # Create a minimally corrupted CSV: break the first header token
    with GOLDEN.open("r", encoding="utf-8") as f_in, bad.open("w", encoding="utf-8") as f_out:
        header = f_in.readline()
        # e.g., change 'sample_id' -> 'sampleId' (or otherwise ensure it's not recognized)
        tokens = [t.strip() for t in header.rstrip("\n").split(",")]
        if tokens:
            tokens[0] = tokens[0].replace("_", "") or "id"
        f_out.write(",".join(tokens) + "\n")
        # Copy the rest of the file unchanged
        shutil.copyfileobj(f_in, f_out)

    # Schema validator must fail
    ok, errs = validator(bad)
    assert not ok, "Corrupted submission should be invalid via schema validator"
    assert errs, "Expect validator to provide at least one error message for corrupted CSV"

    # CLI validator should also fail, if available
    prefix = _find_working_cli_validate(cli_runner, _CLI_VALIDATORS)
    if prefix is None:
        pytest.skip("No working CLI validate command found for negative test.")
    res = cli_runner.invoke([*prefix, str(bad)])
    assert res.exit_code != 0, "CLI validator must reject corrupted CSV"
    # Some feedback in stdout/stderr is useful for users; assert we see something
    assert res.output.strip(), "CLI validator should emit an error message for corrupted CSV"
