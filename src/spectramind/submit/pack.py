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
- Stable Unix-style permissions (0644) on all members
- Zip64 enabled; chunked hashing to bound memory
- Optional schema validation via spectramind.submit.validate

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
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

Pathish = Union[str, Path]

# -----------------------------------------------------------------------------#
# Internal helpers                                                             #
# -----------------------------------------------------------------------------#

_CHUNK = 1024 * 1024  # 1 MiB read chunks


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _zipinfo_for(
    arcname: str,
    *,
    ts_unix: int,
    compress_type: int,
    external_attr: int = (0o644 << 16),  # -rw-r--r--
) -> zipfile.ZipInfo:
    """
    Build a ZipInfo with stable UTC timestamp and perms (0644).
    """
    tm = time.gmtime(ts_unix)  # UTC tuple
    zi = zipfile.ZipInfo(
        filename=arcname,
        date_time=(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec),
    )
    zi.compress_type = compress_type
    zi.external_attr = external_attr
    return zi


def _validate_csv_light(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing submission CSV: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Submission CSV is empty: {path}")
    # Extension check is advisory only; allow .CSV etc.
    if path.suffix.lower() != ".csv":
        # Non-fatal: keep going. Kaggle requires submission.csv arcname anyway.
        return


def _try_strict_validate(csv_path: Path) -> Optional[str]:
    """
    If spectramind.submit.validate is available, run strict validation and return an error
    message on failure; otherwise return None.
    """
    try:
        from spectramind.submit import validate_csv  # type: ignore
    except Exception:
        return None

    report = validate_csv(csv_path)
    if not report.ok:
        return "Strict validation failed:\n- " + "\n- ".join(report.errors)
    return None


def _platform_info() -> Dict[str, Any]:
    return {
        "platform": os.name,
        "sys_platform": sys.platform,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


# -----------------------------------------------------------------------------#
# Public API                                                                   #
# -----------------------------------------------------------------------------#

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
    strict_validate: bool = False,
) -> Path:
    """
    Package a submission CSV and metadata into a ZIP archive.

    Args:
        csv_path: Path to the predictions CSV (Kaggle schema).
        out_zip: Output path for the zip archive.
        meta: Optional extra metadata (merged with defaults).
        seed: If set, produces deterministic timestamps/content ordering (UTC).
        extras: Optional iterable of extra files to embed under 'assets/'.
                Each element may be:
                  • ExtraFile(path=..., arcname="notes.txt")
                  • (path, "notes.txt") tuple
        compression: zipfile compression type (default: ZIP_DEFLATED).
        compresslevel: compression level (default: 9 if deflated).
        strict_validate: if True, use spectramind.submit.validate_csv (when available).

    Returns:
        Path to the created zip archive.
    """
    csv_path = Path(csv_path)
    out_zip = Path(out_zip)

    _validate_csv_light(csv_path)
    if strict_validate:
        err = _try_strict_validate(csv_path)
        if err:
            raise ValueError(err)

    # Stable unix timestamp (if seed provided) else "now"
    ts_unix = int(seed) if seed is not None else int(time.time())

    # Build metadata manifest with stable keys
    manifest: Dict[str, Any] = {
        "created_unix": ts_unix,
        "tool": "spectramind-v50",
        "csv_name": csv_path.name,
        "csv_size_bytes": int(csv_path.stat().st_size),
        "csv_sha256": _sha256_file(csv_path),
        "extras": [],
        "env": _platform_info(),
    }
    if meta:
        # User-provided keys win on conflict
        manifest.update(meta)

    # Normalize extras -> List[ExtraFile]
    extra_items: List[ExtraFile] = []
    if extras:
        for it in extras:
            if isinstance(it, ExtraFile):
                extra_items.append(it)
            else:  # tuple-like
                pth, arc = it  # type: ignore[misc]
                extra_items.append(ExtraFile(path=pth, arcname=str(arc)))

    # Ensure output directory exists
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # Zip64 for big files; deterministic order: main CSV first, then extras (sorted by arcname)
    with zipfile.ZipFile(out_zip, mode="w", allowZip64=True) as zf:
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

        # Add JSON metadata with deterministic encoding (UTF-8, sorted keys)
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