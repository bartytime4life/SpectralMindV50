# src/spectramind/utils/seed.py
"""
SpectraMind V50 — Random Seed Utility
-------------------------------------
Centralized control of randomness for reproducibility.

Usage
-----
from spectramind.utils.seed import set_global_seed, temp_seed

# Set seed globally
set_global_seed(42)

# Use as context manager
with temp_seed(1234):
    run_model()

Notes
-----
- Covers Python's `random`, NumPy, PyTorch, TensorFlow, JAX, and Numba (if available).
- Disables hash randomization for stable runs.
- Designed for integration with Hydra configs & CLI reproducibility flags.
- Provides context manager for temporary seeding.
"""

from __future__ import annotations

import os
import random
import logging
import contextlib
from typing import Optional, Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Optional deps
_HAS_TORCH = False
_HAS_TF = False
_HAS_JAX = False
_HAS_NUMBA = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    pass

try:
    import tensorflow as tf  # type: ignore
    _HAS_TF = True
except Exception:
    pass

try:
    import jax
    import jax.numpy as jnp  # noqa: F401
    _HAS_JAX = True
except Exception:
    pass

try:
    import numba
    _HAS_NUMBA = True
except Exception:
    pass


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """
    Set global random seed across libraries.

    Parameters
    ----------
    seed : int
        Seed value to use.
    deterministic_torch : bool, default=True
        If True, configures PyTorch for deterministic ops.
    """
    if seed is None:
        logger.warning("set_global_seed called with None — reproducibility not guaranteed.")
        return

    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # TensorFlow
    if _HAS_TF:
        try:
            tf.random.set_seed(seed)
        except Exception:
            logger.debug("TensorFlow set_seed failed", exc_info=True)

    # JAX
    if _HAS_JAX:
        jax.config.update("jax_enable_x64", True)
        # Users should call jax.random.PRNGKey(seed) explicitly in pipelines

    # Numba
    if _HAS_NUMBA:
        try:
            numba.random.seed(seed)
        except Exception:
            logger.debug("Numba seed failed", exc_info=True)

    logger.info(f"[Seed] Global random seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """
    Initialize worker seed for PyTorch DataLoader workers.

    Ensures each worker has a unique but reproducible seed.
    """
    seed = (np.random.get_state()[1][0] + worker_id) % (2**32)
    set_global_seed(seed, deterministic_torch=False)


@contextlib.contextmanager
def temp_seed(seed: int) -> Iterator[None]:
    """
    Context manager to temporarily set a seed and restore RNG states.

    Example
    -------
    >>> with temp_seed(123):
    ...     np.random.rand()
    """
    state_random = random.getstate()
    state_numpy = np.random.get_state()
    state_torch = torch.get_rng_state() if _HAS_TORCH else None

    set_global_seed(seed)
    try:
        yield
    finally:
        random.setstate(state_random)
        np.random.set_state(state_numpy)
        if _HAS_TORCH and state_torch is not None:
            torch.set_rng_state(state_torch)
