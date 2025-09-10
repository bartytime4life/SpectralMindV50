# src/spectramind/diagnostics/report.py
# =============================================================================
# SpectraMind V50 — Diagnostics Report Generator
# -----------------------------------------------------------------------------
# Produces JSON or HTML reports from diagnostic results.
# • JSON: machine-readable, indented, reproducible
# • HTML: styled, human-readable with optional title and metadata
#
# Designed for CI/Kaggle safety: no external deps, deterministic output.
# =============================================================================

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any


def generate_diagnostics_report(
    results: Dict[str, Any],
    out_path: Path,
    title: str = "SpectraMind V50 — Diagnostics Report",
) -> Path:
    """
    Generate a JSON or HTML diagnostics report.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary of results from diagnostics modules.
    out_path : Path
        Output path (.json or .html).
    title : str, optional
        Title for HTML report (default: "SpectraMind V50 — Diagnostics Report").

    Returns
    -------
    Path
        Path to the saved report file.

    Raises
    ------
    ValueError
        If the output extension is unsupported.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".json":
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    elif out_path.suffix == ".html":
        html = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'>",
            f"<title>{title}</title>",
            "<style>",
            "body {font-family: system-ui, Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 16px;}",
            "h1 {color: #0366d6; font-size: 1.6em; margin-bottom: 0.4em;}",
            "pre {background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto;}",
            "</style></head><body>",
            f"<h1>{title}</h1>",
            "<h2>Results</h2>",
            "<pre>",
            json.dumps(results, indent=2),
            "</pre>",
            "</body></html>",
        ]
        out_path.write_text("\n".join(html), encoding="utf-8")

    else:
        raise ValueError(f"Unsupported extension for report: {out_path.suffix}")

    return out_path
