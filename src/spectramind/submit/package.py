# src/spectramind/submit/package.py
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from .validate import validate_dataframe, N_BINS_DEFAULT
from .utils import write_json_pretty


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_CHUNK = 1024 * 1024  # 1 MiB


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _zipinfo_for(arcname: str, *, ts_unix: int, compress_type: int) -> zipfile.ZipInfo:
    """Create a ZipInfo with stable UTC timestamp and -rw-r--r-- perms."""
    tm = datetime.utcfromtimestamp(ts_unix)
    zi = zipfile.ZipInfo(
        filename=arcname,
        date_time=(tm.year, tm.month, tm.day, tm.hour, tm.minute, tm.second),
    )
    zi.compress_type = compress_type
    zi.external_attr = (0o644 & 0xFFFF) << 16  # Unix perms 0644
    return zi


def _platform_info() -> Dict[str, Any]:
    return {
        "platform": os.name,
        "sys_platform": sys.platform,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def package_submission(
    df_or_csv: Union[pd.DataFrame, str, Path],
    out_dir: Union[str, Path],
    filename: str = "submission.csv",
    make_zip: bool = True,
    zip_name: str = "submission.zip",
    n_bins: int = N_BINS_DEFAULT,
    extra_meta: Optional[Dict[str, Any]] = None,
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    compresslevel: Optional[int] = 9,
    seed: Optional[int] = None,
    strict_validate: bool = True,
) -> Path:
    """
    Validate and package a submission into CSV (+ manifest.json and optional ZIP).

    Args:
      df_or_csv: pandas DataFrame or path to CSV to validate & package.
      out_dir: destination directory to write artifacts.
      filename: on-disk CSV filename (inside zip it is always 'submission.csv').
      make_zip: if True, also write {zip_name}.
      zip_name: name for zip archive.
      n_bins: number of spectral bins (defaults to 283).
      extra_meta: additional manifest fields (user keys override defaults).
      compression: zipfile compression (default: ZIP_DEFLATED).
      compresslevel: compression level (default 9 for deflate).
      seed: if provided, fixes timestamps for deterministic outputs.
      strict_validate: run schema/physics checks before writing.

    Returns:
      Path to the ZIP (if make_zip=True) else the CSV path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load/normalize input
    if isinstance(df_or_csv, (str, Path)):
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv

    # Validate
    if strict_validate:
        report = validate_dataframe(df, n_bins=n_bins, strict_order=True, check_unique_ids=True)
        report.raise_if_failed()

    # Write CSV to disk (external filename can be custom)
    csv_path = out_dir / filename
    df.to_csv(csv_path, index=False)

    # Build manifest (stable keys, user can override with extra_meta)
    created = datetime.now(timezone.utc)
    created_unix = int(seed) if seed is not None else int(created.timestamp())
    manifest: Dict[str, Any] = {
        "created_utc": created.isoformat(),
        "created_unix": created_unix,
        "git_commit": _git_commit(),
        "n_bins": n_bins,
        "csv": csv_path.name,
        "csv_size_bytes": int(csv_path.stat().st_size),
        "csv_sha256": _sha256_of_file(csv_path),
        "env": _platform_info(),
    }
    if extra_meta:
        manifest.update(extra_meta)

    write_json_pretty(out_dir / "manifest.json", manifest)

    if not make_zip:
        return csv_path

    # Build ZIP deterministically (if seed provided)
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, mode="w", allowZip64=True) as zf:
        ts = created_unix
        # submission.csv inside the zip MUST have canonical arcname
        zi_csv = _zipinfo_for("submission.csv", ts_unix=ts, compress_type=compression)
        with csv_path.open("rb") as f:
            zf.writestr(zi_csv, f.read(), compress_type=compression, compresslevel=compresslevel)

        # manifest.json inside the zip (keep exactly as written on disk)
        manifest_bytes = (out_dir / "manifest.json").read_bytes()
        zi_manifest = _zipinfo_for("manifest.json", ts_unix=ts, compress_type=compression)
        zf.writestr(zi_manifest, manifest_bytes, compress_type=compression, compresslevel=compresslevel)

    return zip_path