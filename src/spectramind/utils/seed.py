# src/spectramind/utils/seed.py
"""
SpectraMind V50 — Random Seed Utility
-------------------------------------
Centralized control of randomness for reproducibility.

Usage
-----
from spectramind.utils.seed import set_global_seed

# Set seed globally
set_global_seed(42)

Notes
-----
- Covers Python's `random`, NumPy, and PyTorch (if available).
- Disables hash randomization by default for stable runs.
- Designed to integrate with Hydra configs & CLI reproducibility flags.
"""

from __future__ import annotations

import os
import random
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import torch if available
try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """
    Set global random seed across libraries.

    Parameters
    ----------
    seed : int
        The seed value to use.
    deterministic_torch : bool, default=True
        If True, configures PyTorch for deterministic ops (slower but reproducible).
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

    logger.info(f"[Seed] Global random seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """
    Initialize worker seed for DataLoader workers (PyTorch-compatible).

    Ensures each worker has a unique but reproducible seed.
    """
    seed = (np.random.get_state()[1][0] + worker_id) % (2**32)
    set_global_seed(seed, deterministic_torch=False)