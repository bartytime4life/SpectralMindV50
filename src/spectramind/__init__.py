"""
SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge pipeline.

Mission-grade, CLI-first, Hydra-driven, DVC-tracked repository for multi-sensor
fusion (FGS1 + AIRS) producing calibrated μ/σ over 283 spectral bins.

Public API
----------
- __version__: str         → package version from installed metadata (or fallback)
- get_version(): str       → safe accessor for the version
- get_logger(name): Logger → project-scoped logger (Rich if available; idempotent)
- open_resource(relpath):  → context manager returning a binary file handle
- read_text(relpath): str  → read small, text resources packaged with the module
- read_binary(relpath): bytes → load small binary resources packaged with the module
- resource_exists(relpath): bool → check for presence of a packaged resource
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
    "read_binary",
    "resource_exists",
    "package_root",
    "repo_root",
    "PKG_DIST_NAME",
]

# --------------------------------------------------------------------------------------
# Package & version resolution
# --------------------------------------------------------------------------------------
# Keep this aligned with pyproject.toml [project].name (distribution name, not module name)
PKG_DIST_NAME = "spectramind-v50"  # e.g., pip show spectramind-v50 → Version: x.y.z


def package_root() -> Path:
    """Return the installed package root (…/site-packages/spectramind)."""
    return Path(__file__).resolve().parent


def _is_repo_root(p: Path) -> bool:
    return (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / ".hg").exists()


def repo_root(max_up: int = 6) -> Path:
    """
    Best-effort location of the repository root. In editable installs / source checkouts
    this is typically a few levels up from this file. Falls back to package_root()
    when heuristics fail.

    We climb up to `max_up` levels looking for common VCS and build markers.
    """
    p = package_root()
    for _ in range(max_up):
        if _is_repo_root(p.parent):
            return p.parent
        p = p.parent
    return package_root()


def _read_version_file() -> Optional[str]:
    # Search common locations for a VERSION file (repo root or package root)
    candidates = [
        repo_root() / "VERSION",    # repo root when running from source
        package_root() / "VERSION", # package directory (rare)
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
        return _pkg_version(PKG_DIST_NAME)
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
        # stdlib lacks TRACE; treat as DEBUG
        "TRACE": logging.DEBUG,
    }.get(level_str, logging.INFO)


def _configure_logging_once() -> None:
    """
    Configure logging exactly once.

    Rules:
    - If the root logger already has handlers, only set its level (do not add).
    - Prefer RichHandler when available; otherwise use stdlib basicConfig.
    - Honors env SPECTRAMIND_LOGLEVEL.
    - Add a NullHandler to the package logger to avoid "No handler found" warnings.
    """
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    level = _desired_level_from_env()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        # avoid "No handler found" warnings for library importers
        logging.getLogger("spectramind").addHandler(logging.NullHandler())
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
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    logging.getLogger("spectramind").addHandler(logging.NullHandler())
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
    a source checkout. Prefers importlib.resources (as_file), then FS paths.
    """
    # Attempt importlib.resources for packages/zip-imports
    try:
        traversable = _resources.files(__package__).joinpath(relpath)  # type: ignore[arg-type]
        try:
            # If the resource is backed by the filesystem, we can probe directly.
            if traversable.is_file():  # type: ignore[attr-defined]
                # as_file() returns a context manager; don't materialize here
                pass
        except Exception:
            pass
        # Fall through: the caller should use open_resource/read_text helpers
    except Exception:
        pass

    # Fallback to filesystem paths relative to source tree
    # (try repo_root first to support editable installs)
    repo_candidate = (repo_root() / relpath).resolve()
    if repo_candidate.exists():
        return repo_candidate
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
        try:
            return traversable.read_text(encoding=encoding)  # type: ignore[attr-defined]
        except Exception:
            with _resources.as_file(traversable) as p:
                return Path(p).read_text(encoding=encoding)
    except Exception:
        # Fallback to FS
        fs_path = _resource_path(relpath)
        return fs_path.read_text(encoding=encoding)


def read_binary(relpath: str) -> bytes:
    """Read a small binary resource from the installed package (or source tree)."""
    with open_resource(relpath) as f:
        return f.read()


def resource_exists(relpath: str) -> bool:
    """Check whether a resource is available (installed or source tree)."""
    try:
        traversable = _resources.files(__package__).joinpath(relpath)  # type: ignore[arg-type]
        try:
            # If it's a packaged resource, .is_file() or existence checks may be limited;
            # try materializing as a file in a temp location to probe.
            with _resources.as_file(traversable) as p:
                return Path(p).exists()
        except Exception:
            pass
    except Exception:
        pass
    return _resource_path(relpath).exists()
