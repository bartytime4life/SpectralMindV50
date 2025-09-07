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
"""
from __future__ import annotations

from contextlib import contextmanager
from importlib import resources as _resources
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import logging
import os
from typing import BinaryIO, Iterator, Optional

__all__ = [
    "__version__",
    "get_version",
    "get_logger",
    "open_resource",
    "read_text",
    "PKG_NAME",
]

# --------------------------------------------------------------------------------------
# Package & version resolution
# --------------------------------------------------------------------------------------
# Keep this aligned with pyproject.toml [project].name
PKG_NAME = "spectramind-v50"

def _read_version_file() -> Optional[str]:
    # Search common locations for a VERSION file (repo root or package root)
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "VERSION",  # repo root when running from source
        Path(__file__).resolve().parent / "VERSION",                # package directory (rare)
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
    # 1) installed metadata (wheel/sdist)
    try:
        return _pkg_version(PKG_NAME)
    except PackageNotFoundError:
        pass
    # 2) VERSION file during local dev
    v = _read_version_file()
    if v:
        return v
    # 3) explicit env override (useful in Kaggle/CI)
    v = os.environ.get("SPECTRAMIND_VERSION")
    if v:
        return v
    # 4) last resort—keep in sync with pyproject for sanity
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
        "TRACE": logging.DEBUG,  # no TRACE in stdlib; map to DEBUG
    }.get(level_str, logging.INFO)

def _configure_logging_once() -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    level = _desired_level_from_env()
    try:
        from rich.logging import RichHandler  # type: ignore
        # Avoid double handlers if user code pre-configured logging.basicConfig
        root = logging.getLogger()
        if not any(isinstance(h, RichHandler) for h in root.handlers):
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
        else:
            root.setLevel(level)
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
        # On some Python versions, .is_file() may need a try/except due to lazy traversals
        try:
            if res.is_file() or res.exists():
                return Path(res)
        except Exception:
            # If we cannot probe, still try to open via the context manager caller
            return Path(res)
    except Exception:
        pass

    # Fallback to filesystem paths relative to package directory
    pkg_dir = Path(__file__).resolve().parent
    fs_path = (pkg_dir / relpath).resolve()
    return fs_path

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
    # Try via importlib.resources first (works for packages and zip-imports)
    try:
        with _resources.as_file(_resources.files(__package__).joinpath(relpath)) as p:  # type: ignore[arg-type]
            with open(p, "rb") as fp:
                yield fp
            return
    except Exception:
        # Fall back to direct FS path when running from source
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
    # Go through importlib.resources first for zip/egg safety
    try:
        return _resources.files(__package__).joinpath(relpath).read_text(encoding=encoding)  # type: ignore[arg-type]
    except Exception:
        # Fall back to FS
        fs_path = _resource_path(relpath)
        return fs_path.read_text(encoding=encoding)
