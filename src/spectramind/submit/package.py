# src/spectramind/submit/package.py
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from .validate import validate_dataframe, N_BINS_DEFAULT
from .utils import write_json_pretty


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
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_submission(
    df_or_csv: Union[pd.DataFrame, str, Path],
    out_dir: Union[str, Path],
    filename: str = "submission.csv",
    make_zip: bool = True,
    zip_name: str = "submission.zip",
    n_bins: int = N_BINS_DEFAULT,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Validate and package a submission into CSV (+ manifest.json and optional ZIP).

    Returns:
      Path to the ZIP (if make_zip=True) else the CSV path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load/validate input
    if isinstance(df_or_csv, (str, Path)):
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv

    report = validate_dataframe(df, n_bins=n_bins, strict_order=True, check_unique_ids=True)
    report.raise_if_failed()

    # Write CSV
    csv_path = out_dir / filename
    df.to_csv(csv_path, index=False)

    # Build manifest
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "n_bins": n_bins,
        "csv": str(csv_path.name),
        "csv_sha256": _sha256_of_file(csv_path),
    }
    if extra_meta:
        manifest.update(extra_meta)

    write_json_pretty(out_dir / "manifest.json", manifest)

    # Optional ZIP
    if make_zip:
        zip_path = out_dir / zip_name
        # Ensure deterministic zip content order by zipping directory with just the files we created
        # To avoid dependency on `zip`, use shutil.make_archive
        tmp_dir = out_dir / "_pkg_tmp"
        tmp_dir.mkdir(exist_ok=True)
        try:
            shutil.copy2(csv_path, tmp_dir / csv_path.name)
            shutil.copy2(out_dir / "manifest.json", tmp_dir / "manifest.json")
            archive_base = shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=tmp_dir)
            return Path(archive_base)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        return csv_path