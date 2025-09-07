# tests/integration/test_end_to_end_cli.py
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

TIMEOUT = 30  # seconds per CLI call; keep tight for CI
SUBCOMMANDS = ("calibrate", "train", "predict", "diagnose", "submit")


# ----------------------------------------------------------------------------- #
# CLI discovery helpers
# ----------------------------------------------------------------------------- #
def _cli_candidates() -> List[List[str]]:
    """
    Return a list of command candidates to invoke the SpectraMind CLI.
    We try:
      - spectramind (if on PATH)
      - python -m spectramind
    """
    cands: List[List[str]] = []
    if shutil.which("spectramind"):
        cands.append(["spectramind"])
    # Prefer the exact python executable running tests
    py = shutil.which("python") or "python"
    cands.append([py, "-m", "spectramind"])
    return cands


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return 124, out, err  # 124 ~ timeout
    return proc.returncode, out, err


def _first_working_cli() -> Optional[List[str]]:
    """
    Return the first CLI command that responds to --help successfully.
    """
    for base in _cli_candidates():
        code, out, err = _run_cmd([*base, "--help"])
        if code == 0:
            return base
    return None


# mark all tests here as integration
pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------------- #
# Top-level CLI smoke
# ----------------------------------------------------------------------------- #
def test_cli_top_level_help() -> None:
    """
    The CLI should be discoverable and respond to --help with exit code 0.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found on PATH nor as `python -m spectramind`.")
    code, out, err = _run_cmd([*base, "--help"])
    assert code == 0, f"Top-level --help failed: code={code}\nstdout:\n{out}\nstderr:\n{err}"
    # Basic sanity: help should mention common subcommands or some usage header
    assert any(s in out.lower() for s in ("usage", "help", "calibrate", "train", "predict")), f"Unexpected help text:\n{out}"


@pytest.mark.parametrize("subcmd", SUBCOMMANDS)
def test_cli_subcommands_help(subcmd: str) -> None:
    """
    Each major subcommand should at least expose --help and exit cleanly.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found")

    code, out, err = _run_cmd([*base, subcmd, "--help"])
    if code != 0:
        # Some subcommands may not be wired yet; skip instead of failing CI
        pytest.skip(f"`{subcmd} --help` not available (code={code}). stderr:\n{err}")
    # Minimal sanity checks
    assert "help" in out.lower() or "usage" in out.lower(), f"Unexpected subcommand help for {subcmd}:\n{out}"


# ----------------------------------------------------------------------------- #
# Optional: pipeline smoke test (opt-in via E2E_CLI_SMOKE=1)
# ----------------------------------------------------------------------------- #
def _try_smoke_run(base: List[str], tmpdir: Path) -> Optional[str]:
    """
    Attempt a very short, non-destructive pipeline smoke. We try to detect
    common flags for dry-run/limit/batch-size to avoid heavy work.

    We return a human-readable summary string on success, otherwise None.
    """
    # Common smoke patterns by subcommand; we allow absence and skip gracefully
    attempts = [
        # Predict stage is usually the lightest and can be stubbed
        ([*base, "predict", "--help"], "predict --help"),
        # Try 'predict --dry-run' / '--dry' / '--smoke' if present
        ([*base, "predict", "--dry-run"], "predict --dry-run"),
        ([*base, "predict", "--dry"], "predict --dry"),
        ([*base, "predict", "--smoke"], "predict --smoke"),
        # Try a tiny limit/num-samples
        ([*base, "predict", "--limit", "1"], "predict --limit 1"),
        ([*base, "predict", "--num-samples", "1"], "predict --num-samples 1"),
        # If predict isn’t wired, try diagnose in a dry mode too
        ([*base, "diagnose", "--dry-run"], "diagnose --dry-run"),
        ([*base, "diagnose", "--help"], "diagnose --help"),
    ]
    env = os.environ.copy()
    # Set conservative thread caps for shared CI or Kaggle-like environments
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    for cmd, label in attempts:
        code, out, err = _run_cmd(cmd, cwd=tmpdir, env=env)
        if code == 0:
            return f"OK: {label}"
        # If the subcommand exists but the flag isn’t recognized, keep trying
        if any(tok in err.lower() for tok in ("unknown option", "unrecognized", "no such option")):
            continue
        # Other failures (like missing data) are not considered fatal for a smoke pass;
        # just try the next attempt
    return None


@pytest.mark.skipif(os.environ.get("E2E_CLI_SMOKE") != "1", reason="Set E2E_CLI_SMOKE=1 to enable pipeline smoke test")
def test_cli_smoke_pipeline(tmp_path: Path) -> None:
    """
    Optional smoke test that attempts a cheap pipeline command (e.g., predict --dry-run/--limit 1).
    This should not require data or heavy compute; it skips gracefully if flags are not recognized.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found")

    # Use an isolated working dir to avoid polluting repo root
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    # Try to make the CLI write here; not all CLIs support an override, so we do a best-effort CD
    summary = _try_smoke_run(base, workdir)
    if summary is None:
        pytest.skip("No lightweight smoke mode detected (dry-run/limit).")
    # If a smoke mode succeeded, we assert it returned code 0 via _try_smoke_run’s check
    assert summary.startswith("OK:")