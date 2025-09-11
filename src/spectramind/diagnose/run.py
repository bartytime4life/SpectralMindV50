from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .io import read_predictions_any, read_truth_any
from .metrics import (
    compute_sanity_checks,
    compute_smoothness_score,
    compute_coverage,
    compute_gll_simple,
)


def run_diagnostics(
    *,
    preds_path: Path,
    truth_path: Optional[Path] = None,
    out_dir: Path,
    report_name: str = "report.html",
) -> Dict[str, float]:
    """
    High-level diagnostics runner:
      - Reads predictions (coerces to wide: id + mu_*** + sigma_***)
      - Optionally reads truth (coerces to narrow)
      - Computes sanity checks, smoothness; if truth available: coverage + simple GLL
      - Writes summary.json (and returns the dict)
      - If rich report module is available, writes an HTML/MD report alongside
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = read_predictions_any(Path(preds_path))
    truth = read_truth_any(Path(truth_path)) if truth_path else None

    summary: Dict[str, float] = {}

    # Sanity checks & smoothness (work without truth)
    summary.update(compute_sanity_checks(preds))
    summary["smoothness_score"] = compute_smoothness_score(preds)

    # Metrics requiring truth
    summary["coverage_k1"] = compute_coverage(preds, truth, k=1.0)
    summary["coverage_k2"] = compute_coverage(preds, truth, k=2.0)
    summary["gll_simple"] = compute_gll_simple(preds, truth)

    # Persist summary
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Try to generate the richer report (non-fatal if deps/file missing)
    try:
        from spectramind.diagnostics.reports import generate_report  # type: ignore
        report_path = generate_report(
            run_id=f"diag-{Path(preds_path).stem}",
            artifacts_dir=out_dir,
            predictions_csv=preds_path,
            history_csv=None,
            metrics_json=str(out_dir / "summary.json"),
            config_path=None,
            notes="Diagnostics report",
            filename=report_name,
        )
        _ = report_path  # just to avoid lints
    except Exception:
        pass

    return summary
