# src/spectramind/utils/pack.py
"""
SpectraMind V50 — Submission Packaging Utility
==============================================

Packs prediction CSVs into a Kaggle-ready ZIP archive with a JSON metadata manifest.
- Deterministic (UTF-8, sorted keys, reproducible timestamps if seeded).
- Safe for Kaggle/CI (no internet, bounded resources).
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict


def pack(csv_path: str | Path, out_zip: str | Path, meta: Dict[str, Any] | None = None) -> Path:
    """
    Package a submission CSV and metadata into a ZIP archive.

    Args:
        csv_path: Path to the predictions CSV (must have Kaggle schema).
        out_zip: Output path for the zip archive.
        meta: Optional extra metadata (merged with defaults).

    Returns:
        Path to the created zip archive.
    """
    csv_path = Path(csv_path)
    out_zip = Path(out_zip)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing submission CSV: {csv_path}")

    # Default metadata
    manifest: Dict[str, Any] = {
        "created_unix": int(time.time()),
        "tool": "spectramind-v50",
        "csv_size_bytes": csv_path.stat().st_size,
        "csv_name": csv_path.name,
    }
    if meta:
        manifest.update(meta)

    # Ensure output directory exists
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # Write zip
    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add CSV with canonical name
        zf.write(csv_path, arcname="submission.csv")

        # Add JSON metadata with deterministic encoding
        zf.writestr(
            "meta.json",
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        )

    return out_zip
