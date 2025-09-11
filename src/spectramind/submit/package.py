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
from typing import Any, Dict, Optional, Union, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .validate import (
    validate_dataframe,
    validate_csv,
    N_BINS_DEFAULT,
)
from .format import submission_columns
from .utils import write_json_pretty

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_CHUNK = 1024 * 1024  # 1 MiB


def _env_or_none(key: str) -> Optional[str]:
    v = os.getenv(key)
    return v if v not in (None, "") else None


def _git_commit() -> Optional[str]:
    """
    Cheap best-effort git commit hash (falls back to env).
    Avoids subprocess if SM_GIT_SHA is present. Returns short SHA if possible.
    """
    env_sha = _env_or_none("SM_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _zipinfo_for(arcname: str, *, ts_unix: int, compress_type: int) -> zipfile.ZipInfo:
    """
    Create a ZipInfo with stable UTC timestamp and -rw-r--r-- perms.
    """
    dt = datetime.utcfromtimestamp(ts_unix)
    zi = zipfile.ZipInfo(
        filename=arcname,
        date_time=(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second),
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


def _atomic_copy(src: Path, dst: Path) -> None:
    """
    Atomically copy src->dst (same filesystem): read fully and write to tmp, then replace.
    Suitable for mid-sized CSVs typical in Kaggle.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def _write_csv_atomic(df: "pd.DataFrame", out_path: Path, *, n_bins: int) -> None:
    """
    Write a DataFrame as CSV atomically with enforced column order.
    """
    cols = submission_columns(n_bins)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    # ensure order even if DF has correct columns
    df[cols].to_csv(tmp, index=False)
    os.replace(tmp, out_path)


def _seed_to_datetime(seed: int) -> datetime:
    return datetime.fromtimestamp(int(seed), tz=timezone.utc)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def package_submission(
    df_or_csv: Union["pd.DataFrame", str, Path],
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
    manifest_schema: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Validate and package a submission into CSV (+ manifest.json and optional ZIP).

    Args:
      df_or_csv: pandas DataFrame or path to CSV to validate & package.
      out_dir: destination directory for artifacts.
      filename: output CSV filename (inside ZIP it is always 'submission.csv').
      make_zip: also write {zip_name} containing the CSV + manifest.
      zip_name: name for zip archive.
      n_bins: number of spectral bins (defaults to 283).
      extra_meta: additional manifest fields (user keys override defaults).
      compression: zipfile compression (default: ZIP_DEFLATED).
      compresslevel: compression level (default 9 for deflate).
      seed: if provided, fix timestamps for deterministic outputs (seconds since epoch).
      strict_validate: run schema/physics checks before writing.
      manifest_schema: optional JSON schema for manifest validation (currently informational).

    Returns:
      Path to ZIP (if make_zip=True) else CSV path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine created timestamp: use seed if supplied to be deterministic.
    created_dt = _seed_to_datetime(seed) if (seed is not None) else datetime.now(timezone.utc)
    created_unix = int(created_dt.timestamp())
    created_iso = created_dt.isoformat()

    # Prepare CSV path
    csv_path = out_dir / filename

    # Case A: DataFrame input
    if pd is not None and isinstance(df_or_csv, pd.DataFrame):
        df = df_or_csv
        if strict_validate:
            report = validate_dataframe(
                df, n_bins=n_bins, strict_order=True, check_unique_ids=True
            )
            report.raise_if_failed()
        # Write atomically ensuring exact header order
        _write_csv_atomic(df, csv_path, n_bins=n_bins)

    # Case B: CSV path input
    elif isinstance(df_or_csv, (str, Path)):
        src_csv = Path(df_or_csv)
        if strict_validate:
            report = validate_csv(
                src_csv,
                n_bins=n_bins,
                strict_ids=True,
                strict_wide_order=True,  # enforce exact order if wide
            )
            if not report.ok:
                # Print first few errors to stderr before raising
                print(
                    f"[package_submission] validation failed ({len(report.errors)} errors). "
                    f"First 5:\n  - " + "\n  - ".join(report.errors[:5]),
                    file=sys.stderr,
                )
                raise ValueError("CSV validation failed")
        # Copy atomically to desired filename/location
        _atomic_copy(src_csv, csv_path)

    else:
        raise TypeError("df_or_csv must be a pandas DataFrame or a CSV path")

    # Validate header order on the final CSV (safety net)
    cols = submission_columns(n_bins)
    try:
        # cheap header read
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            header_line = fh.readline().rstrip("\n\r")
        header = header_line.split(",") if header_line else []
        if header != cols:
            raise ValueError("Output CSV header does not match required submission header.")
    except Exception as e:
        raise ValueError(f"Failed to validate output CSV header: {e}") from e

    # Build manifest (stable keys, user can override with extra_meta)
    manifest: Dict[str, Any] = {
        "created_utc": created_iso,
        "created_unix": created_unix,
        "git_commit": _git_commit(),
        "run_id": _env_or_none("SM_RUN_ID"),
        "config_hash": _env_or_none("SM_CONFIG_HASH"),
        "n_bins": int(n_bins),
        "csv": csv_path.name,
        "csv_size_bytes": int(csv_path.stat().st_size),
        "csv_sha256": _sha256_of_file(csv_path),
        "env": _platform_info(),
    }
    if extra_meta:
        manifest.update(extra_meta)

    # (Optional) manifest schema validation hook — for future use
    # If you define a schemas/manifest.schema.json, you can validate it here.

    write_json_pretty(out_dir / "manifest.json", manifest)

    # If only CSV requested, we’re done
    if not make_zip:
        return csv_path

    # Build deterministic ZIP with stable timestamps/perms
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, mode="w", allowZip64=True) as zf:
        # submission.csv
        zi_csv = _zipinfo_for("submission.csv", ts_unix=created_unix, compress_type=compression)
        with csv_path.open("rb") as f:
            zf.writestr(zi_csv, f.read(), compress_type=compression, compresslevel=compresslevel)

        # manifest.json
        manifest_bytes = (out_dir / "manifest.json").read_bytes()
        zi_manifest = _zipinfo_for("manifest.json", ts_unix=created_unix, compress_type=compression)
        zf.writestr(zi_manifest, manifest_bytes, compress_type=compression, compresslevel=compresslevel)

    return zip_path
