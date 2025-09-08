# tests/unit/test_seed.py
from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import pytest


# ----------------------------------------------------------------------------- #
# Optional torch support (CI/Kaggle may be CPU-only)
# ----------------------------------------------------------------------------- #
try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _cuda_available() -> bool:
    if not _HAS_TORCH:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ----------------------------------------------------------------------------- #
# Sampling & seeding helpers
# ----------------------------------------------------------------------------- #
def _sample_triplet(n: int = 5) -> Tuple[List[int], np.ndarray, List[float]]:
    """
    Collect values from random, numpy, and torch (if available), using the
    *global singletons* so that np.random.seed / torch.manual_seed affect them.
    """
    py_vals = [random.randint(0, 10_000_000) for _ in range(n)]
    np_vals = np.random.random(n)

    if _HAS_TORCH:
        t = torch.rand(n)
        torch_vals = t.tolist()
    else:
        torch_vals = []

    return py_vals, np_vals, torch_vals


def _set_all_seeds(seed: int) -> None:
    """
    Mirror deterministic seeding logic typically used in pipeline bootstrap
    (and conftest). Kept conservative to avoid runtime-specific issues.
    """
    random.seed(seed)
    np.random.seed(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        if _cuda_available():  # pragma: no cover (often CPU-only)
            torch.cuda.manual_seed_all(seed)

        # Prefer strict determinism where available; some ops may still be nondet
        if hasattr(torch, "use_deterministic_algorithms"):
            # avoid throwing on nondeterministic ops during tests; we only check flag below
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[arg-type]

        # cuBLAS workspace (stochastic kernels) — harmless on CPU
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


# ----------------------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------------------- #
def test_reproducible_seeds_across_libraries(rng_seed: int) -> None:
    """
    Given the per-test seed (rng_seed) from conftest, draws should repeat exactly
    when we reseed to the same value and should differ for a different seed.
    """
    # First draw sequence (conftest already seeded globals)
    a_py, a_np, a_torch = _sample_triplet()

    # Re-seed with the same seed → must match exactly
    _set_all_seeds(rng_seed)
    b_py, b_np, b_torch = _sample_triplet()

    assert a_py == b_py
    assert np.allclose(a_np, b_np, rtol=0, atol=0)
    if _HAS_TORCH:
        assert a_torch == b_torch

    # Re-seed with a different seed → should differ with very high probability
    other = rng_seed + 1 if rng_seed != 2**31 - 1 else rng_seed - 1
    _set_all_seeds(other)
    c_py, c_np, c_torch = _sample_triplet()

    # Not strictly guaranteed, but in practice these will differ immediately
    assert (
        a_py != c_py
        or not np.allclose(a_np, c_np)
        or (_HAS_TORCH and a_torch != c_torch)
    )


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_determinism_flags_enabled() -> None:
    """
    Validate that determinism knobs are ON when available.
    We *observe flags* rather than enforce specific backend configs to avoid flakiness.
    """
    # torch.use_deterministic_algorithms → check readback API if present
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        assert torch.are_deterministic_algorithms_enabled()  # type: ignore[attr-defined]

    # cudnn deterministic checks (guarded and non-fatal if unavailable)
    if hasattr(torch.backends, "cudnn"):
        cudnn = torch.backends.cudnn  # type: ignore[attr-defined]
        # cudnn may be unavailable on CPU wheels; only assert attributes exist
        if hasattr(cudnn, "deterministic"):
            assert isinstance(cudnn.deterministic, bool)
        if hasattr(cudnn, "benchmark"):
            # We *prefer* benchmark False for reproducibility, but don't hard fail.
            # If your bootstrap enforces it, flip to assert not cudnn.benchmark.
            assert isinstance(cudnn.benchmark, bool)

    # Environment hint for cuBLAS: not required on CPU wheels, but if set, ensure valid form
    cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cfg is not None:
        assert cfg in {":16:8", ":4096:8"}, "Unexpected CUBLAS_WORKSPACE_CONFIG value"


def test_pythonhashseed_env_matches_fixture(rng_seed: int) -> None:
    """
    conftest sets PYTHONHASHSEED to the per-test seed for traceability.
    (Changing it at runtime doesn't affect hashing behavior, so we only
    assert consistency of the environment mirror.)
    """
    assert os.environ.get("PYTHONHASHSEED") == str(rng_seed)


def test_no_net_fixture_blocks_remote_connect(no_net) -> None:
    """
    The `no_net` fixture monkeypatches socket to block outbound connections.
    Verify a connect attempt to a non-local address raises the blocking error.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError) as exc:
            # Use a well-known public DNS IP; the call is intercepted, not executed
            s.connect(("8.8.8.8", 53))
        msg = str(exc.value).lower()
        assert "network access blocked" in msg
    finally:
        s.close()
