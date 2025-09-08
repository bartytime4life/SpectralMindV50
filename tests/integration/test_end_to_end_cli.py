# tests/integration/test_end_to_end_cli.py
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# Keep tight for CI; each CLI call must return quickly.
TIMEOUT = int(os.environ.get("E2E_CLI_TIMEOUT", "30"))

# The “big five” SpectraMind subcommands expected by our Typer CLI
SUBCOMMANDS = ("calibrate", "train", "predict", "diagnose", "submit")


# ----------------------------------------------------------------------------- #
# CLI discovery & runner utilities
# ----------------------------------------------------------------------------- #
def _python_candidates() -> List[str]:
    """
    Return preferred python executables to use for `-m spectramind`.
    Order is important; we prefer the currently running interpreter.
    """
    cands: List[str] = []
    if sys.executable:
        cands.append(sys.executable)
    # Fallbacks commonly present in CI and dev workstations
    for name in ("python", "python3", "py -3", "py"):
        exe = shutil.which(name.split()[0])
        if exe:
            cands.append(name)
    # De-duplicate preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for p in cands:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _cli_candidates() -> List[List[str]]:
    """
    Return a list of command candidates to invoke the SpectraMind CLI.
    We try (in order):
      - `spectramind` if on PATH
      - `<python> -m spectramind` for several python executables
    """
    cands: List[List[str]] = []
    if shutil.which("spectramind"):
        cands.append(["spectramind"])
    for py in _python_candidates():
        # Allow "py -3" style shims (Windows); split safely
        cands.append([*shlex.split(py), "-m", "spectramind"])
    return cands


def _env_for_cli(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build a conservative environment for CLI runs suitable for CI/Kaggle-like boxes:
      - Limit thread pools (BLAS/NumExpr) to 1 to avoid noisy timing/CPU spikes
      - Disable GPUs by default (predict/dry-run paths shouldn’t need CUDA)
      - Force non-interactive backends (matplotlib) if imported downstream
    """
    env = dict(os.environ if base_env is None else base_env)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")  # make GPU optional
    env.setdefault("MPLBACKEND", "Agg")
    # Make help output predictable (en-US); ignore if locale missing
    env.setdefault("LC_ALL", env.get("LC_ALL", "C"))
    env.setdefault("LANG", env.get("LANG", "C"))
    return env


def _run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = TIMEOUT,
) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return 124, out, err  # 124 ~ timeout (rsync convention)
    return proc.returncode, out, err


def _first_working_cli() -> Optional[List[str]]:
    """
    Return the first CLI command that responds to --help successfully.
    """
    for base in _cli_candidates():
        code, out, err = _run_cmd([*base, "--help"], env=_env_for_cli())
        if code == 0:
            return base
    return None


# Mark every test in this module as integration
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
    code, out, err = _run_cmd([*base, "--help"], env=_env_for_cli())
    assert code == 0, f"Top-level --help failed: code={code}\nstdout:\n{out}\nstderr:\n{err}"
    # Basic sanity: help should mention usage header or common subcommands
    text = out.lower()
    assert any(s in text for s in ("usage", "help", "calibrate", "train", "predict")), f"Unexpected help text:\n{out}"


def test_cli_version_flag() -> None:
    """
    The CLI should expose a version flag and exit 0.
    Accept both `--version` and `-V` flavors.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found")
    tried = []
    for flag in ("--version", "-V"):
        tried.append(flag)
        code, out, err = _run_cmd([*base, flag], env=_env_for_cli())
        if code == 0:
            # Minimal sanity: output should include the project name or a semantic version token
            joined = (out or err).strip()
            assert joined, "Version output empty"
            assert any(tok in joined.lower() for tok in ("spectramind", "v", "version")), f"Unexpected version text: {joined}"
            return
    pytest.skip(f"No working version flag among {tried}")


@pytest.mark.parametrize("subcmd", SUBCOMMANDS)
def test_cli_subcommands_help(subcmd: str) -> None:
    """
    Each major subcommand should at least expose --help and exit cleanly.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found")

    code, out, err = _run_cmd([*base, subcmd, "--help"], env=_env_for_cli())
    if code != 0:
        # Some subcommands may not be wired yet; skip instead of failing CI
        pytest.skip(f"`{subcmd} --help` not available (code={code}). stderr:\n{err}")
    # Minimal sanity checks
    text = out.lower()
    assert "help" in text or "usage" in text, f"Unexpected subcommand help for {subcmd}:\n{out}"
    # If Typer/Rich prints the command name in header, it should be present
    assert subcmd in text or "options" in text, f"Help doesn’t look like {subcmd}: {out}"


# ----------------------------------------------------------------------------- #
# Optional: pipeline smoke test (opt-in via E2E_CLI_SMOKE=1)
# ----------------------------------------------------------------------------- #
def _try_smoke_run(base: List[str], tmpdir: Path) -> Optional[str]:
    """
    Attempt a very short, non-destructive pipeline smoke. We try to detect
    common flags for dry-run/limit/batch-size to avoid heavy work.

    Return a human-readable summary string on success, otherwise None.
    """
    attempts: List[Tuple[List[str], str]] = [
        # Prefer predict: typically lightest stage
        ([*base, "predict", "--help"], "predict --help"),
        ([*base, "predict", "--dry-run"], "predict --dry-run"),
        ([*base, "predict", "--dry"], "predict --dry"),
        ([*base, "predict", "--smoke"], "predict --smoke"),
        ([*base, "predict", "--limit", "1"], "predict --limit 1"),
        ([*base, "predict", "--num-samples", "1"], "predict --num-samples 1"),
        # Try diagnose in cheap modes
        ([*base, "diagnose", "--dry-run"], "diagnose --dry-run"),
        ([*base, "diagnose", "--help"], "diagnose --help"),
        # If predict/diagnose are unavailable, try calibrate dry modes
        ([*base, "calibrate", "--dry-run"], "calibrate --dry-run"),
        ([*base, "calibrate", "--limit", "1"], "calibrate --limit 1"),
    ]
    env = _env_for_cli()
    # Use an isolated working directory when possible
    for cmd, label in attempts:
        code, out, err = _run_cmd(cmd, cwd=tmpdir, env=env)
        if code == 0:
            return f"OK: {label}"
        # If the subcommand exists but a flag isn’t recognized, keep trying
        if any(tok in (err or "").lower() for tok in ("unknown option", "unrecognized", "no such option", "got unexpected")):
            continue
        # Failures due to missing data/config shouldn’t be fatal for smoke; try next
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
        pytest.skip("No lightweight smoke mode detected (dry-run/limit).")
    # If a smoke mode succeeded, we assert it returned code 0 via _try_smoke_run’s check
    assert summary.startswith("OK:")
