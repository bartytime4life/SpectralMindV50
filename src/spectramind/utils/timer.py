# src/spectramind/utils/timer.py
"""
SpectraMind V50 — Timer Utility
--------------------------------
Mission-grade lightweight timer for benchmarking and diagnostics.

Usage
-----
from spectramind.utils.timer import Timer, timeit

# As context manager
with Timer("Calibration"):
    run_calibration()

# As decorator
@timeit("Model training")
def train_model():
    ...

Notes
-----
- Uses monotonic clock (safe against system time changes).
- Logs human-friendly runtime messages.
- Designed for integration with CLI + logging utils.
"""

from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, label: str = "", *, log_fn: Optional[Callable[[str], None]] = None):
        self.label = label
        self.log_fn = log_fn or (lambda msg: logger.info(msg))
        self.start: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start is None:
            return
        self.elapsed = time.perf_counter() - self.start
        self.log_fn(f"[Timer] {self.label} finished in {self.elapsed:.3f} s")


def timeit(label: str = "") -> Callable:
    """
    Decorator to measure execution time of functions.

    Example
    -------
    @timeit("Expensive function")
    def foo():
        ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with Timer(label or func.__name__):
                return func(*args, **kwargs)

        return wrapper

    return decorator