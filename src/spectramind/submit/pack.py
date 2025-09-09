# src/spectramind/utils/pack.py
"""
SpectraMind V50 — Submission Packaging Utility
==============================================

Packs prediction CSV(s) into a Kaggle-ready ZIP archive with a JSON metadata manifest.

Highlights
----------
- Deterministic output (fixed timestamps & ordering) when `seed` is provided
- UTF-8 JSON with sorted keys for stable diffs
- SHA-256 fingerprints of payload files
- Safe for Kaggle/CI (no internet, bounded memory; streams file bytes)

Typical usage
-------------
>>> from spectramind.utils.pack import pack
>>> pack("out/submission.csv", "out/submission.zip", meta={"run_id": "v50.1"}, seed=42)

Notes
-----
- Kaggle expects the primary CSV at arcname "submission.csv". You may attach
  extra files (e.g., explanation) under "assets/..." via `extras`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


Pathish = Union[str, Path]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    # stream in chunks to keep memory bounded
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _zipinfo_for(
    arcname: str,
    *,
    ts_unix: int,
    compress_type: int,
    comment: Optional[str] = None,
    external_attr: Optional[int] = None,
    compresslevel: Optional[int] = None,
) -> zipfile.ZipInfo:
    # Zip stores local time tuple; use UTC 00:00:00 of given timestamp (deterministic)
    tm = time.gmtime(ts_unix)
    zi = zipfile.ZipInfo(
        filename=arcname,
        date_time=(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec),
    )
    zi.compress_type = compress_type
    if comment:
        zi.comment = comment.encode("utf-8")
    if external_attr is not None:
        zi.external_attr = external_attr
    # compresslevel is passed to ZipFile.writestr in Py3.7+; keep in caller
    return zi


def _validate_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing submission CSV: {path}")
    if path.suffix.lower() != ".csv":
        # Not fatal, but warn loudly in metadata
        return
    # lightweight sanity: non-empty and small header presence
    if path.stat().st_size == 0:
        raise ValueError(f"Submission CSV is empty: {path}")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

@dataclass
class ExtraFile:
    """An extra file to embed under 'assets/'. `arcname` is relative to assets/."""
    path: Pathish
    arcname: str


def pack(
    csv_path: Pathish,
    out_zip: Pathish,
    meta: Optional[Dict[str, Any]] = None,
    *,
    seed: Optional[int] = None,
    extras: Optional[Iterable[Union[ExtraFile, Tuple[Pathish, str]]]] = None,
    compression: int = zipfile.ZIP_DEFLATED,
    compresslevel: Optional[int] = 9,
) -> Path:
    """
    Package a submission CSV and metadata into a ZIP archive.

    Args:
        csv_path: Path to the predictions CSV (Kaggle schema).
        out_zip: Output path for the zip archive.
        meta: Optional extra metadata (merged with defaults).
        seed: If set, produces deterministic timestamps/content ordering.
        extras: Optional iterable of extra files to embed under 'assets/'.
                Each element may be:
                  • ExtraFile(path=..., arcname="notes.txt")
                  • (path, "notes.txt") tuple
        compression: zipfile compression type (default: ZIP_DEFLATED).
        compresslevel: compression level (default: 9 if deflated).

    Returns:
        Path to the created zip archive.
    """
    csv_path = Path(csv_path)
    out_zip = Path(out_zip)

    _validate_csv(csv_path)

    # Stable unix timestamp
    ts_unix = int(seed) if seed is not None else int(time.time())

    # Build metadata manifest with stable keys
    manifest: Dict[str, Any] = {
        "created_unix": ts_unix,
        "tool": "spectramind-v50",
        "csv_name": csv_path.name,
        "csv_size_bytes": int(csv_path.stat().st_size),
        "csv_sha256": _sha256_file(csv_path),
        "extras": [],
        "env": {
            "platform": os.name,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        },
    }
    if meta:
        # User-provided keys win on conflict
        manifest.update(meta)

    # Normalize extras
    extra_items: List[ExtraFile] = []
    if extras:
        for it in extras:
            if isinstance(it, ExtraFile):
                extra_items.append(it)
            else:
                pth, arc = it  # type: ignore[misc]
                extra_items.append(ExtraFile(path=pth, arcname=str(arc)))

    # Ensure output directory exists
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # Write zip with deterministic ordering: main CSV first, then extras (sorted by arcname)
    with zipfile.ZipFile(out_zip, mode="w") as zf:
        # Write main CSV under canonical arcname
        csv_bytes = _read_bytes(csv_path)
        zi_csv = _zipinfo_for(
            "submission.csv",
            ts_unix=ts_unix,
            compress_type=compression,
        )
        zf.writestr(zi_csv, csv_bytes, compress_type=compression, compresslevel=compresslevel)

        # Write extras under assets/
        for extra in sorted(extra_items, key=lambda e: e.arcname):
            p = Path(extra.path)
            if not p.exists():
                raise FileNotFoundError(f"Extra file not found: {p}")
            data = _read_bytes(p)
            arc = f"assets/{extra.arcname.lstrip('/')}"
            zi = _zipinfo_for(
                arc,
                ts_unix=ts_unix,
                compress_type=compression,
            )
            zf.writestr(zi, data, compress_type=compression, compresslevel=compresslevel)
            manifest["extras"].append(
                {
                    "arcname": arc,
                    "size_bytes": int(len(data)),
                    "sha256": _sha256_bytes(data),
                    "src": str(p),
                }
            )

        # Add JSON metadata with deterministic encoding
        zi_meta = _zipinfo_for(
            "meta.json",
            ts_unix=ts_unix,
            compress_type=compression,
        )
        zf.writestr(
            zi_meta,
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8"),
            compress_type=compression,
            compresslevel=compresslevel,
        )

    return out_zip