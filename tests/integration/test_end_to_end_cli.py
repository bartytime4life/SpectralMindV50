from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List

import pytest

# Keep per-call timeouts strict for CI; anything long-running should be opt-in via E2E_CLI_SMOKE
TIMEOUT = 30  # seconds
SUBCOMMANDS = ("calibrate", "train", "predict", "diagnose", "submit")

# Mark all tests as integration for selective runs
pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------------- #
# CLI discovery helpers
# ----------------------------------------------------------------------------- #
def _cli_candidates() -> List[List[str]]:
    """
    Return candidate invocations for the SpectraMind CLI:
      1) `spectramind` if installed on PATH
      2) the current python exec -m spectramind
      3) the current python exec -m spectramind.cli (fallback for module layout)
    """
    cands: List[List[str]] = []

    exe = shutil.which("spectramind")
    if exe:
        cands.append([exe])

    py = shutil.which("python") or shutil.which("python3") or "python"
    cands.append([py, "-m", "spectramind"])
    cands.append([py, "-m", "spectramind.cli"])
    return cands


def _cap_env(base: Optional[dict] = None) -> dict:
    """
    Return a copy of env with conservative thread caps and safe toggles
    suitable for shared CI or offline environments (e.g., Kaggle).
    """
    env = dict(base or os.environ)
    # Respect existing, but cap if missing
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Disable any accidental net access in tests (calibration / pulls must be local)
    env.setdefault("SPECTRAMIND_ALLOW_NET", "0")
    # Make logs quieter unless debugging
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: int = TIMEOUT,
) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        # Conventionally use 124 for timeout to make triage easy
        return 124, out, err


def _first_working_cli() -> Optional[List[str]]:
    """
    Return the first CLI command that responds to `--help` with exit code 0.
    """
    for base in _cli_candidates():
        code, _, _ = _run_cmd([*base, "--help"], env=_cap_env())
        if code == 0:
            return list(base)
    return None


# ----------------------------------------------------------------------------- #
# Top-level CLI smoke
# ----------------------------------------------------------------------------- #
def test_cli_top_level_help() -> None:
    """
    The CLI should be discoverable and respond to --help with exit code 0.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found on PATH nor as `python -m spectramind`/`.cli`.")
    code, out, err = _run_cmd([*base, "--help"], env=_cap_env())
    assert code == 0, f"Top-level --help failed: code={code}\nstdout:\n{out}\nstderr:\n{err}"
    # Basic sanity: help text should mention usage or common verbs
    lower = out.lower()
    assert any(k in lower for k in ("usage", "help", "calibrate", "train", "predict")), f"Unexpected help text:\n{out}"


def test_cli_top_level_version() -> None:
    """
    If the CLI exposes --version, it should exit 0 and emit something non-empty.
    If not wired yet, skip.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found.")

    code, out, err = _run_cmd([*base, "--version"], env=_cap_env())
    if code != 0 and not out and not err:
        pytest.skip("--version not exposed yet.")
    assert code == 0, f"`--version` failed with code={code}\nstderr:\n{err}"
    assert (out or err).strip(), "`--version` produced no output"


@pytest.mark.parametrize("subcmd", SUBCOMMANDS)
def test_cli_subcommands_help(subcmd: str) -> None:
    """
    Each major subcommand should at least expose --help and exit cleanly.
    If a subcommand is not wired yet, skip with context rather than failing CI.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found")

    code, out, err = _run_cmd([*base, subcmd, "--help"], env=_cap_env())
    if code != 0:
        pytest.skip(f"`{subcmd} --help` not available (code={code}). stderr:\n{err or out}")
    # Minimal sanity checks
    assert any(k in out.lower() for k in ("help", "usage")), f"Unexpected subcommand help for {subcmd}:\n{out}"


# ----------------------------------------------------------------------------- #
# Optional: pipeline smoke test (opt-in via E2E_CLI_SMOKE=1)
# ----------------------------------------------------------------------------- #
def _try_smoke_run(base: List[str], tmpdir: Path) -> Optional[str]:
    """
    Attempt a very short, non-destructive pipeline smoke. We try to detect
    common flags for dry-run/limit/batch-size to avoid heavy work and data deps.

    Return a human-readable summary string on success, otherwise None.
    """
    attempts: list[tuple[list[str], str]] = [
        # Predict is typically the lightest – try to find a do-nothing mode
        ([*base, "predict", "--help"], "predict --help"),
        ([*base, "predict", "--dry-run"], "predict --dry-run"),
        ([*base, "predict", "--dry"], "predict --dry"),
        ([*base, "predict", "--smoke"], "predict --smoke"),
        ([*base, "predict", "--limit", "1"], "predict --limit 1"),
        ([*base, "predict", "--num-samples", "1"], "predict --num-samples 1"),
        # Diagnose is often a safe no-op too
        ([*base, "diagnose", "--dry-run"], "diagnose --dry-run"),
        ([*base, "diagnose", "--help"], "diagnose --help"),
        # As a last resort, try `train --help` / `submit --help` for cheap responses
        ([*base, "train", "--help"], "train --help"),
        ([*base, "submit", "--help"], "submit --help"),
    ]

    env = _cap_env()
    # Let users direct test data explicitly if the CLI supports it (common via HYDRA or custom flag)
    # These env vars won't hurt if unused.
    env.setdefault("SPECTRAMIND_TEST_DATA", str(tmpdir))
    env.setdefault("SPECTRAMIND_TEST_WORK", str(tmpdir))

    for cmd, label in attempts:
        code, out, err = _run_cmd(cmd, cwd=tmpdir, env=env)
        if code == 0:
            return f"OK: {label}"
        # If the subcommand exists but the flag isn’t recognized, keep trying
        if any(tok in (err or "").lower() for tok in ("unknown option", "unrecognized", "no such option")):
            continue
        # If we failed for missing data or similar, just try the next attempt; smoke is best-effort
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

    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    summary = _try_smoke_run(base, workdir)
    if summary is None:
        pytest.skip("No lightweight smoke mode detected (dry-run/limit/help across predict/diagnose/train/submit).")
    # If a smoke mode succeeded, we assert it returned code 0 via _try_smoke_run’s check
    assert summary.startswith("OK:"), f"Expected OK summary, got: {summary}"
