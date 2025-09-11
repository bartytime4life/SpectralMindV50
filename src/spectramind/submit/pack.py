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


def _platform_info() -> Dict[str, Any]:
    return {
        "platform": os.name,
        "sys_platform": sys.platform,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def _validate_csv_header_exact(path: Path) -> Optional[str]:
    """
    Fast header validation (no pandas): ensure exact submission header order.
    Returns an error string or None if ok.
    """
    try:
        from spectramind.submit.format import submission_columns  # local import
    except Exception:
        return None  # if module not available, skip strict header check

    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            header_line = fh.readline().rstrip("\r\n")
        header = header_line.split(",") if header_line else []
    except Exception as e:
        return f"failed to read CSV header: {type(e).__name__}: {e}"

    expected = submission_columns()
    if header != expected:
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        return (
            "CSV header mismatch with required submission header.\n"
            f"  Expected {len(expected)} columns, got {len(header)}.\n"
            f"  Missing: {missing[:5]}{'...' if len(missing)>5 else ''}\n"
            f"  Extra: {extra[:5]}{'...' if len(extra)>5 else ''}"
        )
    return None


def _validate_csv_light(path: Path) -> None:
    """
    Lightweight safety checks: file exists, non-empty, optional header order check.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing submission CSV: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Submission CSV is empty: {path}")

    # Advisory extension check (do not block .CSV)
    if path.suffix.lower() != ".csv":
        pass

    # If we can check the exact header cheaply, do so
    hdr_err = _validate_csv_header_exact(path)
    if hdr_err:
        raise ValueError(hdr_err)


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
        return "Strict validation failed:\n- " + "\n- ".join(report.errors[:10])
    return None


def _stream_file_into_zip(
    zf: zipfile.ZipFile,
    zi: zipfile.ZipInfo,
    src_path: Path,
    *,
    compress_type: int,
    compresslevel: Optional[int],
) -> None:
    """
    Stream file into a zip entry without loading whole file into memory.
    """
    with zf.open(zi, mode="w", force_zip64=True) as dst, src_path.open("rb") as src:
        if compresslevel is not None and hasattr(dst, "compresslevel"):
            # no direct API to set compresslevel on open handle; zf.writestr handles it.
            # For streaming we rely on compressor defaults in Python's zipfile.
            pass
        for chunk in iter(lambda: src.read(_CHUNK), b""):
            dst.write(chunk)


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
        meta: Optional extra metadata (merged with defaults; user keys win).
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

    # Basic checks + header enforcement if available
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
        manifest.update(meta)  # user keys win

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

    # Zip64 for big files; deterministic order: main CSV first, then extras sorted, then meta.json
    with zipfile.ZipFile(out_zip, mode="w", allowZip64=True) as zf:
        # submission.csv (canonical arcname)
        zi_csv = _zipinfo_for("submission.csv", ts_unix=ts_unix, compress_type=compression)
        _stream_file_into_zip(zf, zi_csv, csv_path, compress_type=compression, compresslevel=compresslevel)

        # extras under assets/
        for extra in sorted(extra_items, key=lambda e: e.arcname):
            p = Path(extra.path)
            if not p.exists():
                raise FileNotFoundError(f"Extra file not found: {p}")
            arc = f"assets/{extra.arcname.lstrip('/')}"
            zi = _zipinfo_for(arc, ts_unix=ts_unix, compress_type=compression)
            _stream_file_into_zip(zf, zi, p, compress_type=compression, compresslevel=compresslevel)

            # Hash lazily (once) for manifest; chunked I/O
            manifest["extras"].append(
                {
                    "arcname": arc,
                    "size_bytes": int(p.stat().st_size),
                    "sha256": _sha256_file(p),
                    "src": str(p),
                }
            )

        # meta.json last (deterministic encoding)
        meta_bytes = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        zi_meta = _zipinfo_for("meta.json", ts_unix=ts_unix, compress_type=compression)
        zf.writestr(zi_meta, meta_bytes, compress_type=compression, compresslevel=compresslevel)

    return out_zip
