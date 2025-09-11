# src/spectramind/submit/utils.py
"""
SpectraMind V50 — Submission Utilities
======================================

Helpers for robust JSON & NDJSON serialization used in submission packaging,
manifests, and event logs. Designed for deterministic diffs and safe writes
in CI/Kaggle environments.

Features
--------
- Pretty & compact JSON (atomic write, sorted keys, UTF-8, trailing newline)
- NDJSON writers (atomic, stream-to-temp), and appenders (flush+fsync)
- Streaming and eager NDJSON readers
- Gzip-aware read/write for .json.gz / .ndjson.gz

Example
-------
>>> from spectramind.submit.utils import (
...     write_json_pretty, read_json,
...     write_ndjson, iter_ndjson, append_ndjson
... )
>>> write_json_pretty("manifest.json", {"run_id": "v50.1"})
>>> list(iter_ndjson("events.ndjson"))
[]
>>> append_ndjson("events.ndjson", {"ev": 1})
>>> list(iter_ndjson("events.ndjson"))
[{'ev': 1}]
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

Pathish = Union[str, Path]

# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

def _is_gz(path: Path) -> bool:
    return Path(path).suffix == ".gz"


def _atomic_write_bytes(path: Path, data: bytes, *, gz: bool = False) -> None:
    """
    Write bytes atomically: write to a temp file in same directory, then replace.
    If gz=True, write a gzip file atomically (temp .gz -> target .gz).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Choose a consistent temp file name adjacent to target
    suffix = path.suffix + ".tmp" if not gz else path.suffix + ".tmp"
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=suffix)
    tmp = Path(tmp_name)
    try:
        if gz:
            with open(fd, "wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb") as gzfh:
                    gzfh.write(data)
                    gzfh.flush()
                raw.flush()
                os.fsync(raw.fileno())
        else:
            with open(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        # In rare races tmp may already be moved; ignore missing.
        if tmp.exists():
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _open_text_for_read(path: Path) -> io.TextIOBase:
    if _is_gz(path):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_text_for_write_atomic(path: Path, *, gz: bool = False) -> tuple[io.TextIOBase, Path]:
    """
    Open a temp text file for atomic writing; caller must .flush/.close then atomic replace.
    Returns (handle, temp_path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp" + (".gz" if gz else "")
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=suffix)
    tmp = Path(tmp_name)
    if gz:
        raw = os.fdopen(fd, "wb")
        fh = gzip.open(raw, mode="wt", encoding="utf-8")  # type: ignore[assignment]
    else:
        fh = os.fdopen(fd, "w", encoding="utf-8")
    return fh, tmp


def safe_json_dumps(
    obj: Any,
    *,
    pretty: bool = False,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
) -> str:
    """
    Deterministic JSON dump:
      - pretty controls indent vs compact (with deterministic separators).
      - sort_keys ensures stable ordering.
      - allow_nan=False by default to prevent NaN/Inf sneaking into artifacts.
      - trailing newline not included (added by writers).
    """
    if pretty:
        return json.dumps(
            obj, indent=2, sort_keys=sort_keys, ensure_ascii=ensure_ascii, allow_nan=allow_nan
        )
    # compact with stable separators
    return json.dumps(
        obj, separators=(",", ":"), sort_keys=sort_keys, ensure_ascii=ensure_ascii, allow_nan=allow_nan
    )

# -------------------------------------------------------------------------
# JSON (object) helpers
# -------------------------------------------------------------------------

def write_json_pretty(
    path: Pathish,
    data: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
) -> None:
    """
    Write a JSON value (dict/list/etc.) prettily; atomic; UTF-8; trailing newline.
    """
    path = Path(path)
    text = safe_json_dumps(data, pretty=True, sort_keys=sort_keys, ensure_ascii=ensure_ascii, allow_nan=allow_nan) + "\n"
    _atomic_write_bytes(path, text.encode("utf-8"), gz=_is_gz(path))


def write_json_compact(
    path: Pathish,
    data: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
) -> None:
    """
    Write compact JSON (no extra spaces/indents); atomic; UTF-8; trailing newline.
    """
    path = Path(path)
    text = safe_json_dumps(data, pretty=False, sort_keys=sort_keys, ensure_ascii=ensure_ascii, allow_nan=allow_nan) + "\n"
    _atomic_write_bytes(path, text.encode("utf-8"), gz=_is_gz(path))


def read_json(path: Pathish) -> Any:
    """
    Load JSON (optionally gzipped) and return parsed object.
    """
    path = Path(path)
    with _open_text_for_read(path) as f:
        return json.load(f)

# -------------------------------------------------------------------------
# NDJSON (newline-delimited JSON) helpers
# -------------------------------------------------------------------------

def write_ndjson(
    path: Pathish,
    rows: Iterable[Any],
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
) -> None:
    """
    Atomically write an iterable of JSON-serializable rows to NDJSON (one JSON per line).
    Writes to a temp file (optionally gzipped) and replaces atomically.
    """
    path = Path(path)
    buf = io.StringIO()
    for r in rows:
        buf.write(safe_json_dumps(r, pretty=False, sort_keys=sort_keys, ensure_ascii=ensure_ascii, allow_nan=allow_nan))
        buf.write("\n")
    _atomic_write_bytes(path, buf.getvalue().encode("utf-8"), gz=_is_gz(path))


def append_ndjson(
    path: Pathish,
    row: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
) -> None:
    """
    Append a single JSON object as one NDJSON line.

    Notes:
        - Plain-file append only; if you need gzipped NDJSON, use write_ndjson().
        - Appends with '\n' termination.
        - fsync is used to harden against crashes (best-effort).
        - Not atomic across processes; for high-contention logs, prefer a
          logging subsystem or per-process files merged later.
    """
    path = Path(path)
    if _is_gz(path):
        # Appending to gz is tricky; require caller to use write_ndjson for gzip targets.
        raise ValueError("append_ndjson does not support .gz targets; use write_ndjson() instead.")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = safe_json_dumps(row, pretty=False, sort_keys=sort_keys, ensure_ascii=ensure_ascii, allow_nan=allow_nan) + "\n"
    with path.open("ab") as f:
        f.write(line.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def iter_ndjson(path: Pathish) -> Iterator[Any]:
    """
    Stream NDJSON records as Python objects (generator). Supports .gz.

    Skips empty/whitespace-only lines. Raises JSONDecodeError on malformed rows.
    """
    path = Path(path)
    with _open_text_for_read(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def read_ndjson(path: Pathish) -> List[Any]:
    """
    Eagerly load all NDJSON records into a list. Supports .gz.
    """
    return list(iter_ndjson(path))
