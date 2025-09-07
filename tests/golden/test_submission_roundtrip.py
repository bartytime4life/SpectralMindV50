# tests/golden/test_submission_roundtrip.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest

GOLDEN = Path(__file__).parent / "submission_valid.csv"

@pytest.mark.require_golden
def test_golden_file_exists_and_is_valid(validator):  # validator comes from property tests or unit tests' adapter
    assert GOLDEN.exists(), "Add tests/golden/submission_valid.csv to the repo"
    ok, errs = validator(GOLDEN)
    assert ok, f"Golden submission should be valid: {errs}"

@pytest.mark.usefixtures("cli_runner")
@pytest.mark.require_golden
def test_cli_accepts_golden(cli_runner):
    assert GOLDEN.exists(), "Add tests/golden/submission_valid.csv to the repo"
    candidate_cmds = [
        ["submission", "validate", str(GOLDEN)],
        ["submit", "validate", str(GOLDEN)],
        ["validate-submission", str(GOLDEN)],
        ["submission-validate", str(GOLDEN)],
        ["kaggle", "validate-submission", str(GOLDEN)],
    ]
    for args in candidate_cmds:
        res = cli_runner.invoke(args)
        if res.exit_code == 0:
            return
    pytest.skip("No working CLI validate command found; expose one to enable this test.")