"""
SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge pipeline.

Mission-grade, CLI-first, Hydra-driven, DVC-tracked repository for multi-sensor
fusion (FGS1 + AIRS) producing calibrated μ/σ over 283 spectral bins.

Public API
----------
- __version__: str         → package version from installed metadata
- get_version(): str       → safe accessor for the version
- get_logger(name): Logger → project-scoped logger (Rich if available)
- open_resource(relpath):  → context manager returning a binary file handle
- read_text(relpath): str  → read small, text resources packaged with the module
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import resources as _resources
from importlib.metadata import PackageNotFoundError, version as _pkg_version
import logging
from typing import Iterator, Optional

__all__ = [
    "__version__",
    "get_version",
    "get_logger",
    "open_resource",
    "read_text",
]

_PKG_NAME = "spectramind-v50"

# --------------------------------------------------------------------------- #
# Version (prefer package metadata; fall back to hardcoded value if needed)
# --------------------------------------------------------------------------- #
try:
    __version__ = _pkg_version(_PKG_NAME)
except PackageNotFoundError:  # pragma: no cover - during local dev without install
    # Keep in sync with pyproject.toml [project].version
    __version__ = "0.1.0"


def get_version() -> str:
    """Return the installed package version."""
    return __version__


# --------------------------------------------------------------------------- #
# Logging (project-scoped; colorized if 'rich' is available)
# --------------------------------------------------------------------------- #
_LOGGER_CONFIGURED = False


def _configure_logging_once() -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    try:
        from rich.logging import RichHandler  # type: ignore

        handler = RichHandler(
            show_time=True,
            show_path=False,
            rich_tracebacks=False,
            markup=True,
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[handler],
        )
    except Exception:
        # Fallback to basic logging if Rich is unavailable
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    _LOGGER_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a namespaced logger configured for SpectraMind.

    Parameters
    ----------
    name : str | None
        Logger name. If None, returns the package-root logger.

    Notes
    -----
    - Uses Rich console output if available.
    - Avoids duplicate configuration across multiple imports.
    """
    _configure_logging_once()
    return logging.getLogger(name or "spectramind")


# --------------------------------------------------------------------------- #
# Resource helpers (for packaged configs/schemas under src/spectramind/)
# --------------------------------------------------------------------------- #
@contextmanager
def open_resource(relpath: str) -> Iterator[object]:
    """
    Open a bundled resource (binary) under the 'spectramind' package.

    Examples
    --------
    >>> with open_resource("schemas/submission.schema.json") as f:
    ...     payload = f.read()

    Raises
    ------
    FileNotFoundError
        If the resource does not exist in an installed wheel.
    """
    try:
        res = _resources.files(__package__).joinpath(relpath)
        with res.open("rb") as fp:
            yield fp
    except FileNotFoundError:
        raise


def read_text(relpath: str, encoding: str = "utf-8") -> str:
    """
    Read a small text resource from the installed package.

    Examples
    --------
    >>> schema = read_text("schemas/submission.schema.json")
    >>> cfg = read_text("configs/train.yaml")
    """
    res = _resources.files(__package__).joinpath(relpath)
    return res.read_text(encoding=encoding)
