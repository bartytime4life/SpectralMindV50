# src/spectramind/utils/logging.py
"""
SpectraMind V50 — Unified Logging Utility
-----------------------------------------

Provides mission-grade, structured logging with:
- Rich console output (colorized, human-friendly).
- JSONL structured logs (machine-readable).
- Environment detection (local, Kaggle, CI).
- Hydra/DVC compatibility for config-as-code reproducibility.

Usage
-----
from spectramind.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Pipeline stage started", extra={"stage": "calibrate"})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None


class JSONLFormatter(logging.Formatter):
    """Formatter that emits one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)
        return json.dumps(base, ensure_ascii=False)


def _detect_env() -> str:
    """Detect runtime environment (local, kaggle, ci)."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists():
        return "kaggle"
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return "ci"
    return "local"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    jsonl: bool = True,
) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: Logger name (usually __name__).
        level: Logging level.
        log_file: Optional file path to write logs.
        jsonl: Whether to write JSONL structured logs.

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # prevent double configuration
        return logger

    logger.setLevel(level)

    # Console handler (pretty if Rich is available)
    if RichHandler and sys.stderr.isatty():
        console_handler = RichHandler(rich_tracebacks=True, show_time=False)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    # File handler (JSONL or plain text)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        if jsonl:
            fh.setFormatter(JSONLFormatter())
        else:
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(fh)

    # Environment tag
    logger = logging.LoggerAdapter(logger, {"extra": {"env": _detect_env()}})
    return logger