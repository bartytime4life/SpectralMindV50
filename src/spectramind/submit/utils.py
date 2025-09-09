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
- NDJSON writers (atomic batch) and appenders (flush+fsync)
- Streaming and eager NDJSON readers

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

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Union

Pathish = Union[str, Path]

# -------------------------------------------------------------------------
# JSON (object) helpers
# -------------------------------------------------------------------------

def write_json_pretty(
    path: Pathish,
    data: Mapping[str, Any],
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> None:
    """
    Write a dictionary to a JSON file with pretty formatting.

    Deterministic: sorted keys, trailing newline.
    Atomic: temp file → rename.

    Args:
        path: Target file path.
        data: Mapping to serialize.
        sort_keys: Deterministic key ordering (default True).
        ensure_ascii: Escape non-ASCII chars (default False).
    """
    path = Path(path)
    text = json.dumps(data, indent=2, sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n"
    _atomic_write(path, text.encode("utf-8"))


def write_json_compact(
    path: Pathish,
    data: Mapping[str, Any],
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> None:
    """
    Write compact JSON (no extra spaces/indents). Atomic; UTF-8; trailing newline.
    """
    path = Path(path)
    text = json.dumps(data, separators=(",", ":"), sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n"
    _atomic_write(path, text.encode("utf-8"))


def read_json(path: Pathish) -> Any:
    """
    Load JSON file and return parsed object.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------------------------------------------------------
# NDJSON (newline-delimited JSON) helpers
# -------------------------------------------------------------------------

def write_ndjson(
    path: Pathish,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> None:
    """
    Atomically write an iterable of dict-like rows to NDJSON (one JSON per line).

    Args:
        path: Output .ndjson path.
        rows: Iterable of mappings to serialize line-by-line.
        sort_keys: Deterministic key ordering (default True).
        ensure_ascii: Escape non-ASCII chars (default False).
    """
    path = Path(path)
    # Render in-memory to bytes to preserve atomicity.
    buf = io.StringIO()
    dump = lambda obj: json.dumps(obj, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    for r in rows:
        buf.write(dump(r))
        buf.write("\n")
    _atomic_write(path, buf.getvalue().encode("utf-8"))


def append_ndjson(
    path: Pathish,
    row: Mapping[str, Any],
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> None:
    """
    Append a single JSON object as one NDJSON line.

    Notes:
        - Appends with '\n' termination.
        - fsync is used to harden against crashes (best-effort).
        - Not atomic across processes; for high-contention logs, prefer a
          logging subsystem or use per-process files merged later.

    Args:
        path: Output .ndjson path (will be created if missing).
        row: Mapping to serialize as a line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n"
    with path.open("ab") as f:
        f.write(line.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def iter_ndjson(path: Pathish) -> Iterator[Any]:
    """
    Stream NDJSON records as Python objects (generator).

    Skips empty/whitespace-only lines. Raises JSONDecodeError on malformed rows.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def read_ndjson(path: Pathish) -> List[Any]:
    """
    Eagerly load all NDJSON records into a list.
    """
    return list(iter_ndjson(path))

# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

def _atomic_write(path: Path, data: bytes) -> None:
    """
    Write bytes atomically: write to a temp file in the same directory, then rename.
    Readers never observe a partially written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.tmp-")
    tmp = Path(tmp_name)
    try:
        with open(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        # In rare races tmp may already be moved; ignore missing.
        if tmp.exists():
            tmp.unlink(missing_ok=True)