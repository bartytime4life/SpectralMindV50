# tests/unit/test_seed.py
from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import pytest


try:  # torch is optional in Kaggle/CI
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _sample_triplet(n: int = 5) -> Tuple[List[int], np.ndarray, List[float]]:
    """Collect a few values from random, numpy, and torch (if available)."""
    py_vals = [random.randint(0, 10_000_000) for _ in range(n)]
    np_vals = np.random.random(n)

    if _HAS_TORCH:
        t = torch.rand(n)
        torch_vals = t.tolist()
    else:
        torch_vals = []

    return py_vals, np_vals, torch_vals


def _set_all_seeds(seed: int) -> None:
    """Mirror the deterministic seeding logic used by conftest."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover (CI often CPU-only)
            torch.cuda.manual_seed_all(seed)
        # enable deterministic algorithms if available
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        # guard for cuBLAS workspace config selection
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def test_reproducible_seeds_across_libraries(rng_seed: int) -> None:
    """
    Given the per-test seed (rng_seed) from conftest, draws should repeat exactly
    when we reseed to the same value and should differ for a different seed.
    """
    # First draw sequence (uses conftest seeding)
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
    assert a_py != c_py or not np.allclose(a_np, c_np) or ( _HAS_TORCH and a_torch != c_torch )


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_determinism_toggled_on() -> None:
    """
    Our test bootstrap aims to force deterministic ops in torch,
    so validate the flag is enabled when available.
    """
    # In new torch versions this is the API; older versions will not have the attr.
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        assert torch.are_deterministic_algorithms_enabled()  # type: ignore[attr-defined]


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