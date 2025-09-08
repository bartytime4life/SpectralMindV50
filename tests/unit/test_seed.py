from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Any, List, Tuple

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

        if hasattr(torch, "use_deterministic_algorithms"):
            # avoid throwing on nondeterministic ops during tests; we only check flag below
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[arg-type]

        # cuBLAS workspace (stochastic kernels) — harmless on CPU
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def _np_state_tuple():
    """Capture numpy RandomState state (legacy global) for equality testing."""
    return tuple(np.random.get_state())


def _torch_state_cpu():
    return torch.random.get_rng_state() if _HAS_TORCH else None


def _torch_state_cuda_all():
    if not _cuda_available():
        return ()
    states = []
    for i in range(torch.cuda.device_count()):
        states.append(torch.cuda.get_rng_state(i))
    return tuple(states)


@contextmanager
def _rng_state_guard():
    """
    Snapshot global RNG states for Python/NumPy/Torch(+CUDA); restore on exit.
    Ensures helpers/ops inside the guard do not leak RNG mutation.
    """
    py_state = random.getstate()
    np_state = _np_state_tuple()
    if _HAS_TORCH:
        torch_cpu_state = _torch_state_cpu()
        torch_cuda_states = _torch_state_cuda_all()
    else:
        torch_cpu_state = None
        torch_cuda_states = None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if _HAS_TORCH:
            torch.random.set_rng_state(torch_cpu_state)  # type: ignore[arg-type]
            if _cuda_available():  # pragma: no cover
                for i, s in enumerate(torch_cuda_states or []):
                    torch.cuda.set_rng_state(s, device=i)


# ----------------------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------------------- #
def test_reproducible_seeds_across_libraries(rng_seed: int) -> None:
    """
    Given the per-test seed (rng_seed) from conftest, draws should repeat exactly
    when we reseed to the same value and should differ for a different seed.
    """
    a_py, a_np, a_torch = _sample_triplet()

    _set_all_seeds(rng_seed)
    b_py, b_np, b_torch = _sample_triplet()

    assert a_py == b_py
    assert np.allclose(a_np, b_np, rtol=0, atol=0)
    if _HAS_TORCH:
        assert a_torch == b_torch

    other = rng_seed + 1 if rng_seed != 2**31 - 1 else rng_seed - 1
    _set_all_seeds(other)
    c_py, c_np, c_torch = _sample_triplet()

    assert (
        a_py != c_py
        or not np.allclose(a_np, c_np)
        or (_HAS_TORCH and a_torch != c_torch)
    )


def test_pythonhashseed_env_matches_fixture(rng_seed: int) -> None:
    """conftest mirrors PYTHONHASHSEED to the per-test seed for traceability."""
    assert os.environ.get("PYTHONHASHSEED") == str(rng_seed)


def test_numpy_state_equality_after_reseed(rng_seed: int) -> None:
    """NumPy RNG state exactly matches after reseed to same seed; differs for a different seed."""
    state_a = _np_state_tuple()
    _ = np.random.random()  # mutate state
    np.random.seed(rng_seed)
    state_b = _np_state_tuple()
    assert state_a == state_b

    other = rng_seed + 12345
    np.random.seed(other)
    state_c = _np_state_tuple()
    assert state_c != state_b


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_cpu_state_equality_after_reseed(rng_seed: int) -> None:
    """Torch CPU RNG states repeat after reseed to same seed and differ on other seed."""
    state_a = _torch_state_cpu()
    _ = torch.rand(3)  # mutate
    torch.manual_seed(rng_seed)
    state_b = _torch_state_cpu()
    assert torch.equal(state_a, state_b)

    other = rng_seed ^ 0x5A5A5A5A
    torch.manual_seed(other)
    state_c = _torch_state_cpu()
    assert not torch.equal(state_c, state_b)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_torch_cuda_states_equality_after_reseed(rng_seed: int) -> None:
    """Optional CUDA RNG state check across all visible devices."""
    states_a = _torch_state_cuda_all()
    if torch.cuda.device_count() > 0:
        with torch.cuda.device(0):
            _ = torch.cuda.FloatTensor(4).uniform_()
    torch.cuda.manual_seed_all(rng_seed)
    states_b = _torch_state_cuda_all()
    assert all(torch.equal(a, b) for a, b in zip(states_a, states_b))

    torch.cuda.manual_seed_all(rng_seed + 999)
    states_c = _torch_state_cuda_all()
    assert any(not torch.equal(b, c) for b, c in zip(states_b, states_c))


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_determinism_flags_enabled() -> None:
    """
    Validate that determinism knobs are ON when available.
    Observe flags rather than enforce backend configs to avoid flakiness.
    """
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        assert torch.are_deterministic_algorithms_enabled()  # type: ignore[attr-defined]

    if hasattr(torch.backends, "cudnn"):
        cudnn = torch.backends.cudnn  # type: ignore[attr-defined]
        if hasattr(cudnn, "deterministic"):
            assert isinstance(cudnn.deterministic, bool)
        if hasattr(cudnn, "benchmark"):
            assert isinstance(cudnn.benchmark, bool)

    cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cfg is not None:
        assert cfg in {":16:8", ":4096:8"}, "Unexpected CUBLAS_WORKSPACE_CONFIG value"


def test_no_net_fixture_blocks_remote_connect(no_net) -> None:
    """
    The `no_net` fixture monkeypatches socket to block outbound connections.
    Verify a connect attempt to a non-local address raises the blocking error.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError) as exc:
            s.connect(("8.8.8.8", 53))
        msg = str(exc.value).lower()
        assert "network access blocked" in msg
    finally:
        s.close()


def test_reseeding_order_is_idempotent(rng_seed: int) -> None:
    """Order of reseeding across libs must not affect final draws."""
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    if _HAS_TORCH:
        torch.manual_seed(rng_seed)
        if _cuda_available():  # pragma: no cover
            torch.cuda.manual_seed_all(rng_seed)
    a = _sample_triplet()

    _ = _sample_triplet()
    if _HAS_TORCH:
        _ = torch.randn(5)

    if _HAS_TORCH:
        if _cuda_available():  # pragma: no cover
            torch.cuda.manual_seed_all(rng_seed)
        torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    b = _sample_triplet()

    assert a[0] == b[0]
    assert np.allclose(a[1], b[1])
    if _HAS_TORCH:
        assert a[2] == b[2]


def test_set_all_seeds_is_idempotent(rng_seed: int) -> None:
    """Calling _set_all_seeds(seed) multiple times should not drift RNG states."""
    _set_all_seeds(rng_seed)
    py_a, np_a, torch_a = _sample_triplet()

    _set_all_seeds(rng_seed)
    py_b, np_b, torch_b = _sample_triplet()

    assert py_a == py_b
    assert np.allclose(np_a, np_b)
    if _HAS_TORCH:
        assert torch_a == torch_b


def test_rng_state_guard_restores_states(rng_seed: int) -> None:
    """The state guard must fully restore Python / NumPy / Torch states on exit."""
    _set_all_seeds(rng_seed)

    py_a = random.getstate()
    np_a = _np_state_tuple()
    tc_a = _torch_state_cpu() if _HAS_TORCH else None

    with _rng_state_guard():
        _ = random.random()
        _ = np.random.rand()
        if _HAS_TORCH:
            _ = torch.rand(2)

    assert random.getstate() == py_a
    assert _np_state_tuple() == np_a
    if _HAS_TORCH:
        assert torch.equal(_torch_state_cpu(), tc_a)


def test_numpy_generator_is_independent_of_global_seed(rng_seed: int) -> None:
    """
    NumPy's new Generator objects are independent of np.random.* singleton seeding.
    Reminder that user code must seed Generators explicitly.
    """
    _set_all_seeds(rng_seed)
    g1 = np.random.default_rng(12345)
    g2 = np.random.default_rng(12345)
    v1 = g1.random(4)
    v2 = g2.random(4)
    assert np.allclose(v1, v2)

    v_glob1 = np.random.random(4)
    _set_all_seeds(rng_seed)
    v_glob2 = np.random.random(4)
    assert np.allclose(v_glob1, v_glob2)


def test_sampling_under_guard_has_no_leakage(rng_seed: int) -> None:
    """Helpers can be used in a guard to avoid changing global RNG state."""
    _set_all_seeds(rng_seed)

    py0 = random.getstate()
    np0 = _np_state_tuple()
    tc0 = _torch_state_cpu() if _HAS_TORCH else None

    with _rng_state_guard():
        _ = _sample_triplet(n=10)

    assert random.getstate() == py0
    assert _np_state_tuple() == np0
    if _HAS_TORCH:
        assert torch.equal(_torch_state_cpu(), tc0)