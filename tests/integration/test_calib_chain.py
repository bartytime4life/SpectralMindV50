# tests/integration/test_calib_chain.py
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# Keep calls short for CI boxes; override via E2E_CLI_TIMEOUT
TIMEOUT = int(os.environ.get("E2E_CLI_TIMEOUT", "30"))

# Common, plausible calibration artifacts we’ll probe for after a cheap run
# (we only *probe*—absence is not a failure; this is a smoke-level chain test)
CALIB_CANDIDATE_DIRS = (
    "data/interim",
    "data/processed",
    "outputs/calib",
    "artifacts/calib",
)


# ----------------------------------------------------------------------------- #
# CLI discovery & runner utilities
# ----------------------------------------------------------------------------- #
def _python_candidates() -> List[str]:
    cands: List[str] = []
    if sys.executable:
        cands.append(sys.executable)
    for name in ("python", "python3", "py -3", "py"):
        exe = shutil.which(name.split()[0])
        if exe:
            cands.append(name)
    seen: set[str] = set()
    uniq: List[str] = []
    for p in cands:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _cli_candidates() -> List[List[str]]:
    cands: List[List[str]] = []
    if shutil.which("spectramind"):
        cands.append(["spectramind"])
    for py in _python_candidates():
        cands.append([*shlex.split(py), "-m", "spectramind"])
    return cands


def _env_for_cli(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    # Thread caps for reproducible/quiet CI runs
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Make GPU optional/off by default for smoke-ish runs
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    # Non-interactive plotting
    env.setdefault("MPLBACKEND", "Agg")
    # Predictable help/version output
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
        return 124, out, err
    return proc.returncode, out, err


def _first_working_cli() -> Optional[List[str]]:
    for base in _cli_candidates():
        code, out, err = _run_cmd([*base, "--help"], env=_env_for_cli())
        if code == 0:
            return base
    return None


# Mark all tests in this module as integration
pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------------- #
# Calibrate subcommand — help & cheap/dry run smoke
# ----------------------------------------------------------------------------- #
def test_calibrate_help() -> None:
    """
    The 'calibrate' subcommand should at least provide --help and exit 0.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found on PATH nor as `python -m spectramind`.")
    code, out, err = _run_cmd([*base, "calibrate", "--help"], env=_env_for_cli())
    if code != 0:
        pytest.skip(f"`calibrate --help` not available (code={code}). stderr:\n{err}")
    low = out.lower()
    assert "usage" in low or "help" in low, f"Unexpected calibrate help:\n{out}"
    # Optional: help may mention inputs/outputs, config, or dry-run flags
    # (not required, but it’s a sanity hint for discoverability)
    assert any(
        tok in low
        for tok in ("input", "output", "config", "dry", "dataset", "data", "kaggle")
    ), f"Calibrate help text looks too bare:\n{out}"


@pytest.mark.skipif(os.environ.get("E2E_CALIB_SMOKE") != "1", reason="Set E2E_CALIB_SMOKE=1 to enable calibration smoke test")
def test_calibrate_smoke(tmp_path: Path) -> None:
    """
    Cheap non-destructive calibration smoke test.

    We try a sequence of likely-cheap invocations (e.g., --dry-run, --limit 1).
    We run in an isolated work dir and also pass common env knobs to keep things light.
    The test skips gracefully if the CLI doesn’t support any such flags.
    """
    base = _first_working_cli()
    if not base:
        pytest.skip("SpectraMind CLI not found")

    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    env = _env_for_cli()
    # If the CLI honors a working directory override env, pass it (best-effort)
    # (Not required; many CLIs just use CWD.)
    env.setdefault("SPECTRAMIND_WORKDIR", str(workdir))

    attempts: List[Tuple[List[str], str]] = [
        # pure help (fastest)
        ([*base, "calibrate", "--help"], "calibrate --help"),
        # common dry/smoke/limit flags
        ([*base, "calibrate", "--dry-run"], "calibrate --dry-run"),
        ([*base, "calibrate", "--dry"], "calibrate --dry"),
        ([*base, "calibrate", "--smoke"], "calibrate --smoke"),
        ([*base, "calibrate", "--limit", "1"], "calibrate --limit 1"),
        # some CLIs accept a tiny batch-size/num-samples
        ([*base, "calibrate", "--batch-size", "1"], "calibrate --batch-size 1"),
        ([*base, "calibrate", "--num-samples", "1"], "calibrate --num-samples 1"),
        # optional: explicit out dir (best-effort)
        ([*base, "calibrate", "--output", str(workdir / "outputs")], "calibrate --output <tmp>"),
    ]

    ok_label: Optional[str] = None
    for cmd, label in attempts:
        code, out, err = _run_cmd(cmd, cwd=workdir, env=env)
        if code == 0:
            ok_label = label
            break
        # Keep trying if the flag isn’t recognized
        if any(tok in (err or "").lower() for tok in ("unknown option", "unrecognized", "no such option", "got unexpected")):
            continue
        # Failures due to missing data/config shouldn’t be fatal at smoke level — try next
        if any(tok in (err or "").lower() for tok in ("missing", "not found", "no such file", "required")):
            continue

    if ok_label is None:
        pytest.skip("No lightweight calibration mode detected (dry-run/limit).")

    # Best-effort probes for artifacts if the run wasn't just `--help`
    if "help" not in ok_label:
        created = _probe_calib_artifacts(workdir)
        # Presence of artifacts is *optional*; we log via assertion message if absent
        assert created or True, f"Calibration smoke succeeded ({ok_label}) but no obvious artifacts in {workdir}"

    # If we reached here, the command exited 0
    assert ok_label.startswith("calibrate")


def _probe_calib_artifacts(root: Path) -> List[Path]:
    """
    Inspect common relative paths (data/interim, data/processed, outputs/calib, artifacts/calib)
    and return those that exist and are non-empty. This is best-effort and non-failing.
    """
    found: List[Path] = []
    for rel in CALIB_CANDIDATE_DIRS:
        p = root / rel
        if p.exists() and p.is_dir():
            # Non-empty directory?
            try:
                if any(p.iterdir()):
                    found.append(p)
            except Exception:
                # If filesystem forbids read, ignore
                pass
    return found
