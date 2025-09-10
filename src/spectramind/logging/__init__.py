# src/spectramind/logging/__init__.py
"""
SpectraMind V50 — Logging Package
=================================

Provides mission-grade logging utilities:
  • Structured JSONL event logging
  • Run context and directory initialization
  • Metrics aggregation (CSV + streaming)
  • Event logging with schema compliance
  • Hashing and serialization helpers

All loggers are designed for Kaggle/CI-safe reproducibility.
"""

from __future__ import annotations

from .jsonl import JSONLLogger
from .run_context import RunContext, init_run_dir
from .metrics import MetricsAggregator, CSVWriter
from .events import EventLogger
from .utils import flatten_dict, hash_text, hash_json, to_serializable

__all__ = [
    # Core loggers
    "JSONLLogger",
    "RunContext",
    "init_run_dir",
    "MetricsAggregator",
    "CSVWriter",
    "EventLogger",
    # Utils
    "flatten_dict",
    "hash_text",
    "hash_json",
    "to_serializable",
    # Factory
    "get_logger",
]


def get_logger(
    run_dir: str | None = None,
    with_metrics: bool = True,
    with_events: bool = True,
) -> dict[str, object]:
    """
    Convenience factory returning a logger suite.

    Args:
        run_dir: Optional path to initialize run artifacts/logging.
        with_metrics: If True, attach a MetricsAggregator + CSVWriter.
        with_events: If True, attach an EventLogger.

    Returns:
        Dictionary of active loggers keyed by name.
    """
    run_ctx = RunContext(run_dir) if run_dir else None
    loggers: dict[str, object] = {
        "jsonl": JSONLLogger(run_ctx.run_dir if run_ctx else None),
    }

    if with_metrics:
        loggers["metrics"] = MetricsAggregator()
        loggers["csv"] = CSVWriter(run_ctx.run_dir if run_ctx else None)

    if with_events:
        loggers["events"] = EventLogger()

    return loggers
