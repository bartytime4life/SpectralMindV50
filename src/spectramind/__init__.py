# src/spectramind/__init__.py
"""
SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge pipeline.

Mission-grade, CLI-first, Hydra-driven, DVC-tracked repository for multi-sensor
fusion (FGS1 + AIRS) producing calibrated μ/σ over 283 spectral bins.

Public API
----------
- __version__: str         → package version from installed metadata (or fallback)
- get_version(): str       → safe accessor for the version
- get_logger(name): Logger → project-scoped logger (Rich if available)
- open_resource(relpath):  → context manager returning a binary file handle
- read_text(relpath): str  → read small, text resources packaged with the module
- package_root(): Path     → path to installed package root
- repo_root(): Path        → best-effort path to repository root (source checkouts)
"""
from __future__ import annotations

from contextlib import contextmanager
from importlib import resources as _resources
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import io
import logging
import os
import sys
from typing import BinaryIO, Iterator, Optional

__all__ = [
    "__version__",
    "get_version",
    "get_logger",
    "open_resource",
    "read_text",
    "package_root",
    "repo_root",
    "PKG_NAME",
]

# --------------------------------------------------------------------------------------
# Package & version resolution
# --------------------------------------------------------------------------------------
# Keep this aligned with pyproject.toml [project].name
PKG_NAME = "spectramind-v50"


def package_root() -> Path:
    """Return the installed package root (…/site-packages/spectramind)."""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """
    Best-effort location of the repository root. In editable installs / source checkouts
    this is typically 2 levels up from this file (…/repo/src/spectramind/__init__.py).
    Falls back to package_root() if heuristics fail.
    """
    p = package_root()
    # Common source layout: repo/src/spectramind/__init__.py
    candidate = p.parent.parent
    if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
        return candidate
    return p


def _read_version_file() -> Optional[str]:
    # Search common locations for a VERSION file (repo root or package root)
    candidates = [
        repo_root() / "VERSION",   # repo root when running from source
        package_root() / "VERSION" # package directory (rare)
    ]
    for c in candidates:
        try:
            if c.exists():
                v = c.read_text(encoding="utf-8").strip()
                if v:
                    return v
        except Exception:
            pass
    return None


def _resolve_version() -> str:
    # 1) Installed metadata (wheel/sdist)
    try:
        return _pkg_version(PKG_NAME)
    except PackageNotFoundError:
        pass
    # 2) VERSION file during local dev
    v = _read_version_file()
    if v:
        return v
    # 3) Explicit env override (useful in Kaggle/CI)
    v = os.environ.get("SPECTRAMIND_VERSION")
    if v:
        return v
    # 4) Last resort—keep in sync with pyproject for sanity
    return "0.1.0"


__version__ = _resolve_version()


def get_version() -> str:
    """Return the best-known package version."""
    return __version__


# --------------------------------------------------------------------------------------
# Logging (Rich if available), idempotent configuration
# --------------------------------------------------------------------------------------
_LOGGER_CONFIGURED = False


def _desired_level_from_env(default: str = "INFO") -> int:
    level_str = os.environ.get("SPECTRAMIND_LOGLEVEL", default).upper()
    return {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "TRACE": logging.DEBUG,  # stdlib lacks TRACE; map to DEBUG
    }.get(level_str, logging.INFO)


def _configure_logging_once() -> None:
    """
    Configure logging exactly once.

    Rules:
    - If the root logger already has handlers, only set its level (do not add).
    - Prefer RichHandler when available; otherwise use stdlib basicConfig.
    - Honors env SPECTRAMIND_LOGLEVEL.
    """
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    level = _desired_level_from_env()
    root = logging.getLogger()
    if root.handlers:
        # Respect existing handlers (e.g., notebooks, host apps, pytest)
        root.setLevel(level)
        _LOGGER_CONFIGURED = True
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
            level=level,
            format="%(message)s",
            handlers=[handler],
        )
    except Exception:
        # Fallback to plain logging
        logging.basicConfig(
            level=level,
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
    - Honors env SPECTRAMIND_LOGLEVEL (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    - Rich console output when 'rich' is available; otherwise stdlib logging.
    - Idempotent: will not stack duplicate handlers across calls/imports.
    """
    _configure_logging_once()
    return logging.getLogger(name or "spectramind")


# --------------------------------------------------------------------------------------
# Resource helpers (work both installed and from source tree)
# --------------------------------------------------------------------------------------
def _resource_path(relpath: str) -> Path:
    """
    Resolve a resource path whether running from an installed wheel or from
    a source checkout. Prefers importlib.resources, then falls back to FS paths.
    """
    # Attempt importlib.resources (installed packages and editable installs)
    try:
        res = _resources.files(__package__).joinpath(relpath)  # type: ignore[arg-type]
        try:
            # As of 3.11, Traversable supports .is_file/.exists; guard anyway
            if res.is_file() or res.exists():
                return Path(res)
        except Exception:
            # If we cannot probe, still return the virtual path; callers will use as_file()
            return Path(res)
    except Exception:
        pass

    # Fallback to filesystem paths relative to package directory
    return (package_root() / relpath).resolve()


@contextmanager
def open_resource(relpath: str) -> Iterator[BinaryIO]:
    """
    Open a bundled resource (binary) under the 'spectramind' package.

    Examples
    --------
    >>> with open_resource("schemas/submission.schema.json") as f:
    ...     payload = f.read()

    Raises
    ------
    FileNotFoundError
        If the resource does not exist.
    """
    # 1) Use importlib.resources for packages/zip-imports
    try:
        traversable = _resources.files(__package__).joinpath(relpath)  # type: ignore[arg-type]
        with _resources.as_file(traversable) as p:
            with open(p, "rb") as fp:
                yield fp
        return
    except Exception:
        # 2) Fall back to direct FS path when running from source
        fs_path = _resource_path(relpath)
        if not fs_path.exists():
            raise FileNotFoundError(f"Resource not found: {relpath} ({fs_path})")
        with fs_path.open("rb") as fp:
            yield fp


def read_text(relpath: str, encoding: str = "utf-8") -> str:
    """
    Read a small text resource from the installed package (or source tree).

    Examples
    --------
    >>> schema = read_text("schemas/submission.schema.json")
    >>> cfg = read_text("configs/train.yaml")
    """
    # Prefer importlib.resources for zip/egg safety
    try:
        traversable = _resources.files(__package__).joinpath(relpath)  # type: ignore[arg-type]
        # Traversable.read_text exists in 3.11+, but use a robust path in case
        try:
            return traversable.read_text(encoding=encoding)  # type: ignore[attr-defined]
        except Exception:
            with _resources.as_file(traversable) as p:
                return Path(p).read_text(encoding=encoding)
    except Exception:
        # Fallback to FS
        fs_path = _resource_path(relpath)
        return fs_path.read_text(encoding=encoding)
