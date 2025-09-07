# src/spectramind/utils/io.py
"""
SpectraMind V50 — I/O Utilities
-------------------------------
Lightweight, safe, and dependency-tolerant I/O helpers:

- Path utils: expand env/user, ensure directories
- Atomic writes to prevent partial files on crash
- Read/Write: JSON, JSONL, YAML (optional), CSV, NumPy (.npy/.npz)
- Optional pandas/pyyaml support when available

Usage
-----
from spectramind.utils.io import (
    p, ensure_dir, atomic_write, read_json, write_json,
    read_yaml, write_yaml, read_jsonl, append_jsonl,
    read_csv, write_csv, save_npy, load_npy
)
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np

# Optional deps
try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    _HAS_PANDAS = False


# -------------------------------------------------------------------
# Path helpers
# -------------------------------------------------------------------

def p(path: Union[str, Path]) -> Path:
    """Normalize a path: expand env vars and ~, return Path."""
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists; returns the directory as Path."""
    d = p(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


# -------------------------------------------------------------------
# Atomic write
# -------------------------------------------------------------------

@contextmanager
def atomic_write(dest: Union[str, Path], mode: str = "w", encoding: Optional[str] = "utf-8"):
    """
    Atomically write to file: write to temp in same dir, then replace.
    Guarantees destination is either old file or fully written new file.
    """
    dest_path = p(dest)
    ensure_dir(dest_path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=dest_path.name + ".", dir=dest_path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, mode, encoding=encoding) as f:
            yield f
        tmp_path.replace(dest_path)  # atomic on POSIX; safe on modern Windows
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


# -------------------------------------------------------------------
# JSON / JSONL
# -------------------------------------------------------------------

def read_json(path: Union[str, Path]) -> Any:
    with p(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Union[str, Path], *, indent: int = 2, sort_keys: bool = True) -> Path:
    dest = p(path)
    with atomic_write(dest, "w", "utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    return dest


def read_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    with p(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(record: Dict[str, Any], path: Union[str, Path]) -> Path:
    dest = p(path)
    ensure_dir(dest.parent)
    with dest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")
    return dest


# -------------------------------------------------------------------
# YAML (optional)
# -------------------------------------------------------------------

def read_yaml(path: Union[str, Path]) -> Any:
    if not _HAS_YAML:
        raise RuntimeError("PyYAML is not installed.")
    with p(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore


def write_yaml(obj: Any, path: Union[str, Path]) -> Path:
    if not _HAS_YAML:
        raise RuntimeError("PyYAML is not installed.")
    dest = p(path)
    with atomic_write(dest, "w", "utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)  # type: ignore
    return dest


# -------------------------------------------------------------------
# CSV
# -------------------------------------------------------------------

def read_csv(path: Union[str, Path]) -> Any:
    """
    Read CSV. Uses pandas if available (returns DataFrame),
    else returns a list of dict rows.
    """
    src = p(path)
    if _HAS_PANDAS:  # pragma: no cover
        return pd.read_csv(src)  # type: ignore
    rows: List[Dict[str, str]] = []
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(rows: Union[Sequence[Dict[str, Any]], "pd.DataFrame"], path: Union[str, Path]) -> Path:
    dest = p(path)
    ensure_dir(dest.parent)
    if _HAS_PANDAS and hasattr(rows, "to_csv"):  # pragma: no cover
        # type: ignore[attr-defined]
        rows.to_csv(dest, index=False)  # type: ignore
        return dest
    # Fallback: list[dict]
    rows_list = list(rows)  # copy iterables
    if not rows_list:
        # write empty file
        with atomic_write(dest, "w", "utf-8") as f:
            pass
        return dest
    fieldnames = list(rows_list[0].keys())
    with atomic_write(dest, "w", "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)
    return dest


# -------------------------------------------------------------------
# NumPy arrays
# -------------------------------------------------------------------

def save_npy(array: np.ndarray, path: Union[str, Path]) -> Path:
    dest = p(path)
    ensure_dir(dest.parent)
    # atomic write via temp; np.save requires filename
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    np.save(tmp, array)
    tmp.replace(dest)
    return dest


def load_npy(path: Union[str, Path]) -> np.ndarray:
    return np.load(p(path))


def save_npz(path: Union[str, Path], **arrays: np.ndarray) -> Path:
    dest = p(path)
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(dest)
    return dest


def load_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    with np.load(p(path)) as data:
        return {k: data[k] for k in data.files}


# -------------------------------------------------------------------
# Text / Binary helpers
# -------------------------------------------------------------------

def read_text(path: Union[str, Path]) -> str:
    with p(path).open("r", encoding="utf-8") as f:
        return f.read()


def write_text(text: str, path: Union[str, Path]) -> Path:
    dest = p(path)
    with atomic_write(dest, "w", "utf-8") as f:
        f.write(text)
    return dest


def read_bytes(path: Union[str, Path]) -> bytes:
    with p(path).open("rb") as f:
        return f.read()


def write_bytes(data: bytes, path: Union[str, Path]) -> Path:
    dest = p(path)
    ensure_dir(dest.parent)
    # atomic for binary too
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", dir=dest.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        tmp_path.replace(dest)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
    return dest