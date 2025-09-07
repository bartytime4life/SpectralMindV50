# src/spectramind/utils/hashing.py
"""
SpectraMind V50 — Hashing Utilities
-----------------------------------
Deterministic, stable hashing helpers for reproducibility:
- Hash bytes/strings/JSON/NumPy arrays to hex digests.
- Hash files and directories (with ignore patterns, chunked IO).
- Stable JSON hashing (sorted keys, compact separators).
- Snapshot helpers for configs/artifacts.

Default algorithm: SHA-256 (widely available, fast, secure).
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:  # optional numpy support
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------

_HASH_ALG = "sha256"
_CHUNK = 1024 * 1024  # 1 MiB


def _new(algo: str = _HASH_ALG) -> "hashlib._Hash":
    try:
        return hashlib.new(algo)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Unknown hash algorithm: {algo}") from e


def hash_bytes(b: bytes, *, algo: str = _HASH_ALG) -> str:
    """Hex digest of a bytes object."""
    h = _new(algo)
    h.update(b)
    return h.hexdigest()


def hash_str(s: str, *, algo: str = _HASH_ALG, encoding: str = "utf-8") -> str:
    """Hex digest of a string."""
    return hash_bytes(s.encode(encoding), algo=algo)


def stable_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON string for hashing:
    - sort_keys=True for canonical order
    - compact separators to avoid whitespace variance
    - ensure_ascii=False to keep UTF-8 intact
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_json(obj: Any, *, algo: str = _HASH_ALG) -> str:
    """Hash an in-memory Python object as stable JSON."""
    return hash_str(stable_json_dumps(obj), algo=algo)


# ---------------------------------------------------------------------
# NumPy support
# ---------------------------------------------------------------------

def hash_numpy_array(arr: "np.ndarray", *, algo: str = _HASH_ALG) -> str:
    """
    Hash a NumPy array content + dtype + shape deterministically.
    """
    if not _HAS_NUMPY:  # pragma: no cover
        raise RuntimeError("NumPy is not installed.")
    h = _new(algo)
    # include metadata that changes interpretation
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    # hash raw bytes in C-order for determinism
    h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return h.hexdigest()


# ---------------------------------------------------------------------
# File and directory hashing
# ---------------------------------------------------------------------

def hash_file(path: str | Path, *, algo: str = _HASH_ALG) -> str:
    """Chunked file hashing (handles large files)."""
    p = Path(path)
    h = _new(algo)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(
    root: Path,
    *,
    ignore: Optional[Sequence[str]] = None,
    include_hidden: bool = False,
    follow_symlinks: bool = False,
) -> Iterator[Path]:
    """
    Yield files under root in a deterministic order, applying ignore patterns.
    - ignore: list of fnmatch-style patterns (applied to relative POSIX path strings)
    """
    root = root.resolve()
    patterns = list(ignore or [])

    def is_ignored(rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, pat) for pat in patterns)

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        # sort to maintain deterministic traversal order
        dirnames.sort()
        filenames.sort()

        # filter hidden if requested
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]

        base = Path(dirpath)
        for name in filenames:
            fp = base / name
            rel = fp.relative_to(root).as_posix()
            if is_ignored(rel):
                continue
            yield fp


def hash_dir(
    path: str | Path,
    *,
    algo: str = _HASH_ALG,
    ignore: Optional[Sequence[str]] = None,
    include_hidden: bool = False,
    follow_symlinks: bool = False,
) -> str:
    """
    Hash directory contents deterministically.

    Strategy:
      - Walk files in sorted order
      - For each file: update with relative path + file hash
      - Ignores: fnmatch-style patterns against relative POSIX paths
    """
    root = Path(path).resolve()
    h = _new(algo)

    for fp in _iter_files(
        root,
        ignore=ignore,
        include_hidden=include_hidden,
        follow_symlinks=follow_symlinks,
    ):
        rel = fp.relative_to(root).as_posix().encode("utf-8")
        h.update(rel + b"\0")  # path boundary
        h.update(hash_file(fp, algo=algo).encode("utf-8"))
        h.update(b"\0")        # record boundary

    return h.hexdigest()


# ---------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------

def config_snapshot_hash(config: Dict[str, Any], *, algo: str = _HASH_ALG) -> str:
    """
    Compute a stable hash of a composed config dict.
    Useful for tagging runs/artifacts (e.g., in DVC/CI manifests).
    """
    return hash_json(config, algo=algo)


def files_manifest(
    paths: Sequence[str | Path],
    *,
    algo: str = _HASH_ALG,
    base: Optional[str | Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Produce a manifest {rel_path: {hash, size}} for a set of files/dirs.
    Directories are expanded to contained files recursively.
    """
    base_path = Path(base).resolve() if base else None
    entries: Dict[str, Dict[str, Any]] = {}

    def add_file(fp: Path) -> None:
        rel = fp
        if base_path:
            rel = fp.resolve().relative_to(base_path)
        rel_key = rel.as_posix()
        entries[rel_key] = {
            "hash": hash_file(fp, algo=algo),
            "size": fp.stat().st_size,
        }

    for src in paths:
        p = Path(src)
        if p.is_dir():
            for fp in _iter_files(p):
                add_file(fp)
        elif p.is_file():
            add_file(p)
        else:
            # skip missing sources silently; caller can validate upstream
            continue

    return entries