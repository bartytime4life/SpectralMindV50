# tests/conftest.py
# =============================================================================
# SpectraMind V50 — Test Bootstrap (pytest)
# -----------------------------------------------------------------------------
# Goals
#   • Deterministic tests (seed, hash seed, thread caps)
#   • Safe tmp workdirs (no accidental writes into repo)
#   • CLI runner for `spectramind` Typer app
#   • Hydra-friendly (per-test runtime dirs, full error traces)
#   • Kaggle/offline guardrails (no net, stable threads)
#   • Device selection fixture (CUDA/CPU/MPS auto, with env override)
#
# Usage
#   pytest -q
#   pytest -q -k "train" SPECTRAMIND_DEVICE=cuda:0
#   pytest -q --maxfail=1 --disable-warnings
# =============================================================================

from __future__ import annotations

import contextlib
import json
import os
import random
import socket
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Optional deps (gracefully degrade if absent)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from typer.testing import CliRunner  # type: ignore
except Exception:  # pragma: no cover
    CliRunner = None  # type: ignore

# Headless/quiet plotting for tests that import plotting modules
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

# Hydra full tracebacks; confine run dirs to CWD (we chdir per test)
os.environ.setdefault("HYDRA_FULL_ERROR", "1")
os.environ.setdefault("HYDRA_RUN_DIR", ".")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Ascend from `start` (or CWD) until a repo root marker is found."""
    start = (start or Path.cwd()).resolve()
    markers = {".git", "pyproject.toml", "setup.cfg", "spectramind.toml"}
    cur = start
    while True:
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            return start  # fallback to original start
        cur = cur.parent


def _add_src_to_syspath(repo_root: Path) -> None:
    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _pin_threads() -> None:
    """Reasonable defaults for CI/Kaggle stability; override in env if needed."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


_pin_threads()

# ──────────────────────────────────────────────────────────────────────────────
# Session-scoped fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Path to the repository root (heuristic: .git/ or pyproject.toml)."""
    root = _find_repo_root()
    _add_src_to_syspath(root)
    return root


@pytest.fixture(scope="session", autouse=True)
def _session_env(repo_root: Path) -> None:
    """
    Session-level environment hygiene:
      • Ensure PYTHONHASHSEED set (deterministic hashing)
      • Point XDG-ish caches inside repo/.pytest_cache
      • Default to offline unless explicitly allowed
    """
    os.environ.setdefault("PYTHONHASHSEED", "0")
    cache_base = (repo_root / ".pytest_cache").resolve()
    cache_base.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_base))
    os.environ.setdefault("SPECTRAMIND_ALLOW_NET", "0")
    # Nicer numpy printing for debugging failures
    np.set_printoptions(precision=6, suppress=True, linewidth=140)


# ──────────────────────────────────────────────────────────────────────────────
# Function-scoped fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def rng_seed() -> int:
    """Default seed value (override via env SPECTRAMIND_TEST_SEED)."""
    return int(os.environ.get("SPECTRAMIND_TEST_SEED", "1337"))


@pytest.fixture(autouse=True)
def seeded(rng_seed: int) -> None:
    """
    Deterministic seeding across Python, NumPy, and (optionally) PyTorch.
    Also enforces PYTHONHASHSEED for stable hashing in this process.
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    os.environ["PYTHONHASHSEED"] = str(rng_seed)
    if torch is not None:
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():  # pragma: no cover (gpu not in CI)
            torch.cuda.manual_seed_all(rng_seed)
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        except Exception:
            pass  # older torch / CPU-only


@pytest.fixture
def tmp_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Per-test isolated working directory:
      • chdir to it (Hydra run dir lives here)
      • standard artifacts subfolders
    """
    monkeypatch.chdir(tmp_path)
    for d in ["artifacts", "outputs", "logs", "cache"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    # Matplotlib cache inside tmp to avoid cross-test contention
    (tmp_path / "cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(tmp_path / "cache" / "matplotlib"))
    return tmp_path


@pytest.fixture
def no_net(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Block outbound network calls by monkeypatching socket.
    Allow localhost and Unix sockets. Opt-out via SPECTRAMIND_ALLOW_NET=1.
    """
    if os.environ.get("SPECTRAMIND_ALLOW_NET", "0") == "1":
        return

    real_socket = socket.socket

    class _BlockedSocket(socket.socket):  # type: ignore[misc]
        def connect(self, address):  # type: ignore[override]
            host, *_ = address if isinstance(address, tuple) else (address,)
            allow = {"127.0.0.1", "::1", "localhost"}
            if isinstance(host, str) and (host in allow or host.startswith("/")):
                return real_socket.connect(self, address)
            raise RuntimeError(
                f"Network access blocked during tests (attempted {address}). "
                "Set SPECTRAMIND_ALLOW_NET=1 to allow."
            )

    monkeypatch.setattr(socket, "socket", _BlockedSocket, raising=True)


@pytest.fixture
def kaggle_like_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Emulate Kaggle runtime constraints where helpful:
      • No internet
      • Thread caps already set by _pin_threads
      • Non-interactive, no display
    """
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("SPECTRAMIND_ALLOW_NET", "0")


@pytest.fixture
def torch_device() -> str:
    """
    Device selection:
      • Env override: SPECTRAMIND_DEVICE="cuda:0"|"cpu"|"mps"
      • Else: cuda if available, else mps (Apple), else cpu
    """
    override = os.environ.get("SPECTRAMIND_DEVICE")
    if override:
        return override
    if torch is not None:
        if torch.cuda.is_available():  # pragma: no cover
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover
            return "mps"
    return "cpu"


@pytest.fixture
def events_log(tmp_workdir: Path) -> Path:
    """Path to a JSONL test event log (created lazily by tests)."""
    p = tmp_workdir / "logs" / "events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def write_event(events_log: Path):
    """
    Helper to append a JSON event to the per-test events.jsonl.
    Usage:
        write_event({"type": "train_start", "epoch": 0})
    """
    def _write(evt: dict) -> None:
        with events_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    return _write


@pytest.fixture
def cli_runner(repo_root: Path):
    """
    Typer CLI runner bound to the `spectramind` app.
    Example:
        result = cli_runner.invoke(["--help"])
        assert result.exit_code == 0
    """
    if CliRunner is None:
        pytest.skip("typer is not installed")
    _add_src_to_syspath(repo_root)

    # Import under repo_root to pick up local package
    with _temporarily_cwd(repo_root):
        try:
            # Your project exposes the Typer app as `cli_app` (not `app`)
            from spectramind.cli import cli_app  # type: ignore
        except Exception as e:  # pragma: no cover
            pytest.skip(f"CLI not available: {e}")

    runner = CliRunner(mix_stderr=False)

    class _Runner:
        def invoke(self, args: list[str] | tuple[str, ...], **kwargs):
            return runner.invoke(cli_app, list(args), **kwargs)

    return _Runner()


@contextlib.contextmanager
def _temporarily_cwd(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


# ──────────────────────────────────────────────────────────────────────────────
# Optional: asyncio event loop fixture (only if you use pytest-asyncio)
# -----------------------------------------------------------------------------
# import asyncio
# @pytest.fixture
# def event_loop():
#     loop = asyncio.new_event_loop()
#     yield loop
#     loop.close()
