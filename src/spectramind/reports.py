# src/spectramind/reports.py
from __future__ import annotations

import base64
import io
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Lightweight, Kaggle/CI-safe deps
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # optional but nicer visuals
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

try:
    import jsonschema  # optional schema validation
    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False


BIN_COUNT = 283  # enforced spectral bins
DEFAULT_ID_COL = "id"
DEFAULT_BIN_COL = "bin"
DEFAULT_MU_COL = "mu"
DEFAULT_SIGMA_COL = "sigma"
DEFAULT_TARGET_COL = "target"


class ReportError(RuntimeError):
    """Typed error for report generation failures."""


@dataclass
class LineageEvent:
    """Represents a single JSONL event entry for lineage aggregation."""
    timestamp: str
    stage: Optional[str] = None
    level: Optional[str] = None
    message: Optional[str] = None
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None
    artifact_digest: Optional[str] = None
    extras: Dict[str, object] = field(default_factory=dict)


@dataclass
class LineageSummary:
    """Aggregated snapshot for lineage and reproducibility."""
    git_commits: List[str] = field(default_factory=list)
    config_hashes: List[str] = field(default_factory=list)
    artifact_digests: List[str] = field(default_factory=list)
    stages: List[str] = field(default_factory=list)
    first_event_ts: Optional[str] = None
    last_event_ts: Optional[str] = None
    counts_by_level: Dict[str, int] = field(default_factory=dict)


@dataclass
class Predictions:
    """Structured predictions table."""
    df: pd.DataFrame
    id_col: str = DEFAULT_ID_COL
    bin_col: str = DEFAULT_BIN_COL
    mu_col: str = DEFAULT_MU_COL
    sigma_col: str = DEFAULT_SIGMA_COL

    def validate(self) -> None:
        required = {self.id_col, self.bin_col, self.mu_col, self.sigma_col}
        missing = required - set(self.df.columns)
        if missing:
            raise ReportError(f"Predictions missing columns: {sorted(missing)}")

        # Ensure correct dtypes where possible
        for c in (self.bin_col, self.mu_col, self.sigma_col):
            if c in self.df:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        # Validate bins are 0..282 or 1..283; normalize to 0-based internally
        unique_bins = np.sort(self.df[self.bin_col].dropna().unique())
        if len(unique_bins) == BIN_COUNT and unique_bins[0] == 0 and unique_bins[-1] == BIN_COUNT - 1:
            pass
        elif len(unique_bins) == BIN_COUNT and unique_bins[0] == 1 and unique_bins[-1] == BIN_COUNT:
            # normalize to 0-based
            self.df[self.bin_col] = self.df[self.bin_col].astype(int) - 1
        else:
            raise ReportError(
                f"Predictions must cover exactly {BIN_COUNT} contiguous bins "
                f"(0..{BIN_COUNT-1} or 1..{BIN_COUNT}). Found range: {unique_bins[:3]}...{unique_bins[-3:]}"
            )

        # Validate sigma > 0 and finite
        bad_sigma = self.df[(~np.isfinite(self.df[self.sigma_col])) | (self.df[self.sigma_col] <= 0)]
        if not bad_sigma.empty:
            raise ReportError(f"Nonpositive or nonfinite sigma rows: {len(bad_sigma)}")

        # Basic group coverage: each id should have all bins
        counts = self.df.groupby(self.id_col)[self.bin_col].nunique()
        incomplete = counts[counts != BIN_COUNT]
        if not incomplete.empty:
            bad_ids = list(incomplete.index[:8])
            raise ReportError(
                f"Some ids do not have {BIN_COUNT} bins (e.g., {bad_ids}...). "
                f"Total incomplete IDs: {len(incomplete)}"
            )


@dataclass
class Targets:
    """Structured targets table (optional for validation/evaluation)."""
    df: pd.DataFrame
    id_col: str = DEFAULT_ID_COL
    bin_col: str = DEFAULT_BIN_COL
    target_col: str = DEFAULT_TARGET_COL

    def validate(self) -> None:
        required = {self.id_col, self.bin_col, self.target_col}
        missing = required - set(self.df.columns)
        if missing:
            raise ReportError(f"Targets missing columns: {sorted(missing)}")
        for c in (self.bin_col, self.target_col):
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        # Normalize bin indexing similarly to predictions
        unique_bins = np.sort(self.df[self.bin_col].dropna().unique())
        if len(unique_bins) == BIN_COUNT and unique_bins[0] == 0 and unique_bins[-1] == BIN_COUNT - 1:
            pass
        elif len(unique_bins) == BIN_COUNT and unique_bins[0] == 1 and unique_bins[-1] == BIN_COUNT:
            self.df[self.bin_col] = self.df[self.bin_col].astype(int) - 1
        else:
            # Be permissive for partial targets (e.g., val split); just ensure 0 or 1 based contiguous
            pass


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ReportError(f"Failed to read CSV: {path} :: {e}") from e


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except Exception as e:
        raise ReportError(f"Failed to read JSONL: {path} :: {e}") from e
    return rows


def _to_b64_png(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _sns() -> None:
    if _HAS_SEABORN:
        sns.set_context("talk")
        sns.set_style("whitegrid")


def _binwise_stats(pred: Predictions, tgt: Optional[Targets]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if tgt is None:
        return out
    merged = pred.df.merge(
        tgt.df[[tgt.id_col, tgt.bin_col, tgt.target_col]],
        left_on=[pred.id_col, pred.bin_col],
        right_on=[tgt.id_col, tgt.bin_col],
        how="inner",
        suffixes=("", "_t"),
    )
    if merged.empty:
        return out
    resid = merged[pred.mu_col] - merged[tgt.target_col]
    out["mse"] = float(np.mean(resid**2))
    out["mae"] = float(np.mean(np.abs(resid)))
    # NLL under N(mu, sigma^2)
    nll = 0.5 * (np.log(2 * math.pi * (merged[pred.sigma_col] ** 2)) + (resid**2) / (merged[pred.sigma_col] ** 2))
    out["nll"] = float(np.mean(nll))
    # Per-bin residual stats
    per_bin = (
        merged.assign(residual=resid)
        .groupby(pred.bin_col)
        .agg(mse=("residual", lambda x: float(np.mean(x**2))),
             mae=("residual", lambda x: float(np.mean(np.abs(x)))),
             mean_resid=("residual", "mean"),
             sigma=("sigma", "mean"))
        .reset_index()
        .sort_values(pred.bin_col)
    )
    out["per_bin"] = per_bin
    return out


def _plot_per_bin_mse(stats: Dict[str, object]) -> Optional[str]:
    per_bin = stats.get("per_bin")
    if per_bin is None or len(per_bin) == 0:
        return None
    _sns()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(per_bin["bin"], per_bin["mse"], lw=1.5)
    ax.set_title("Per-bin MSE")
    ax.set_xlabel("bin")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    return _to_b64_png(fig)


def _plot_sigma_profile(pred: Predictions) -> str:
    _sns()
    prof = (
        pred.df.groupby(pred.bin_col)[pred.sigma_col]
        .mean()
        .reset_index()
        .sort_values(pred.bin_col)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prof[pred.bin_col], prof[pred.sigma_col], lw=1.5, color="#8c4eff")
    ax.set_title("Per-bin σ profile (mean across IDs)")
    ax.set_xlabel("bin")
    ax.set_ylabel("σ")
    ax.grid(True, alpha=0.3)
    return _to_b64_png(fig)


def _plot_calibration_pit(pred: Predictions, tgt: Optional[Targets]) -> Optional[str]:
    if tgt is None:
        return None
    merged = pred.df.merge(
        tgt.df[[tgt.id_col, tgt.bin_col, tgt.target_col]],
        left_on=[pred.id_col, pred.bin_col],
        right_on=[tgt.id_col, tgt.bin_col],
        how="inner",
        suffixes=("", "_t"),
    )
    if merged.empty:
        return None
    # Probability Integral Transform under Normal CDF
    z = (merged[tgt.target_col] - merged[pred.mu_col]) / merged[pred.sigma_col]
    pit = 0.5 * (1.0 + erf(z / math.sqrt(2)))
    _sns()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pit, bins=30, color="#29b6f6", alpha=0.85, edgecolor="white")
    ax.set_title("Calibration — PIT Histogram (ideal: uniform)")
    ax.set_xlabel("PIT")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    return _to_b64_png(fig)


def erf(x: np.ndarray | float) -> np.ndarray | float:
    """Vectorized error function using numpy for scalars / arrays."""
    return np.erf(x)  # numpy provides erf in recent versions


def _plot_residual_scatter(pred: Predictions, tgt: Optional[Targets]) -> Optional[str]:
    if tgt is None:
        return None
    merged = pred.df.merge(
        tgt.df[[tgt.id_col, tgt.bin_col, tgt.target_col]],
        left_on=[pred.id_col, pred.bin_col],
        right_on=[tgt.id_col, tgt.bin_col],
        how="inner",
        suffixes=("", "_t"),
    )
    if merged.empty:
        return None
    resid = merged[tgt.target_col] - merged[pred.mu_col]
    _sns()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged[pred.mu_col], resid, s=6, alpha=0.5, color="#607d8b")
    ax.axhline(0, color="red", lw=1, alpha=0.6)
    ax.set_title("Residuals vs μ")
    ax.set_xlabel("μ")
    ax.set_ylabel("target − μ")
    ax.grid(True, alpha=0.3)
    return _to_b64_png(fig)


def _plot_inject_recover_delta(
    base: Predictions, pert: Predictions, label_base: str = "base", label_pert: str = "perturbed"
) -> str:
    # Compare (mean across IDs) μ profiles
    prof_base = base.df.groupby(base.bin_col)[base.mu_col].mean().reset_index().sort_values(base.bin_col)
    prof_pert = pert.df.groupby(pert.bin_col)[pert.mu_col].mean().reset_index().sort_values(pert.bin_col)
    merged = prof_base.merge(prof_pert, on=base.bin_col, suffixes=("_b", "_p"))
    merged["delta"] = merged[f"{base.mu_col}_p"] - merged[f"{base.mu_col}_b"]
    _sns()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(merged[base.bin_col], merged["delta"], lw=1.5, color="#ff7043")
    ax.set_title(f"Inject-&-Recover: Δμ profile ({label_pert} − {label_base})")
    ax.set_xlabel("bin")
    ax.set_ylabel("Δμ")
    ax.grid(True, alpha=0.3)
    return _to_b64_png(fig)


def _validate_submission_schema(pred: Predictions, schema_path: Optional[Path]) -> Optional[str]:
    if schema_path is None or not schema_path.exists():
        return None
    if not _HAS_JSONSCHEMA:
        return "jsonschema not available — skipped JSON schema validation."
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"Failed to read schema: {e}"
    # Apply a minimal row-wise validation if schema defines properties
    # For performance, just validate the first 100 rows
    sample = pred.df.head(100).to_dict(orient="records")
    try:
        for row in sample:
            jsonschema.validate(instance=row, schema=schema)
        return "Submission schema validation passed (sample)."
    except jsonschema.ValidationError as e:
        return f"Submission schema validation failed: {e.message}"
    except Exception as e:
        return f"Submission schema validation error: {e}"


def _read_events(events_path: Optional[Path]) -> Tuple[List[LineageEvent], Optional[LineageSummary]]:
    if events_path is None or not events_path.exists():
        return [], None
    raw = _read_jsonl(events_path)
    events: List[LineageEvent] = []
    for r in raw:
        ev = LineageEvent(
            timestamp=str(r.get("timestamp", "")),
            stage=r.get("stage"),
            level=r.get("level"),
            message=r.get("message"),
            git_commit=r.get("git_commit"),
            config_hash=r.get("config_hash"),
            artifact_digest=r.get("artifact_digest"),
            extras={k: v for k, v in r.items() if k not in {
                "timestamp", "stage", "level", "message", "git_commit", "config_hash", "artifact_digest"
            }},
        )
        events.append(ev)
    # Aggregate
    commits = [e.git_commit for e in events if e.git_commit]
    cfgs = [e.config_hash for e in events if e.config_hash]
    arts = [e.artifact_digest for e in events if e.artifact_digest]
    stages = [e.stage for e in events if e.stage]
    levels: Dict[str, int] = {}
    for e in events:
        if e.level:
            levels[e.level] = levels.get(e.level, 0) + 1
    ts_sorted = [e.timestamp for e in events if e.timestamp]
    ts_sorted.sort()
    summary = LineageSummary(
        git_commits=sorted(set(commits)),
        config_hashes=sorted(set(cfgs)),
        artifact_digests=sorted(set(arts)),
        stages=sorted(set(stages)),
        first_event_ts=ts_sorted[0] if ts_sorted else None,
        last_event_ts=ts_sorted[-1] if ts_sorted else None,
        counts_by_level=levels,
    )
    return events, summary


def _inline_css() -> str:
    return """
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", sans-serif; margin: 20px; color: #222; }
  h1,h2,h3 { margin-top: 1.2em; }
  .kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 10px; }
  .kpi { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; }
  .kpi .label { color: #6a737d; font-size: 12px; }
  .kpi .value { font-size: 20px; font-weight: 600; }
  .section { border-top: 1px solid #eaecef; padding-top: 16px; margin-top: 16px; }
  .plot { margin: 10px 0; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
  table { border-collapse: collapse; }
  th, td { padding: 6px 8px; border: 1px solid #e1e4e8; }
  .ok { color: #2e7d32; }
  .warn { color: #ef6c00; }
  .err { color: #c62828; }
</style>
    """.strip()


def _html_img(b64png: Optional[str], alt: str) -> str:
    if not b64png:
        return f"<div class='warn'>Plot not available: {alt}</div>"
    return f"<img class='plot' alt='{alt}' src='data:image/png;base64,{b64png}' />"


def generate_report(
    pred_path: Path,
    out_html: Path,
    *,
    targets_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
    submission_schema_path: Optional[Path] = None,
    base_pred_path: Optional[Path] = None,
    report_title: str = "SpectraMind V50 — Diagnostics Dashboard",
    id_col: str = DEFAULT_ID_COL,
    bin_col: str = DEFAULT_BIN_COL,
    mu_col: str = DEFAULT_MU_COL,
    sigma_col: str = DEFAULT_SIGMA_COL,
    target_col: str = DEFAULT_TARGET_COL,
) -> Path:
    """
    Generate a self-contained HTML diagnostics dashboard.

    Parameters
    ----------
    pred_path : Path
        CSV with predictions (id, bin, mu, sigma). Must cover exactly 283 bins per id.
    out_html : Path
        Output HTML path.
    targets_path : Optional[Path]
        Optional CSV with targets (id, bin, target). Enables evaluation metrics / calibration plots.
    events_path : Optional[Path]
        Optional JSONL events file for lineage aggregation.
    submission_schema_path : Optional[Path]
        Optional JSON schema file to validate submission rows (sample).
    base_pred_path : Optional[Path]
        Optional CSV for Inject-&-Recover comparison; compare baseline vs pred_path (perturbed).
    report_title : str
        Title string for the report.
    id_col, bin_col, mu_col, sigma_col, target_col : str
        Column names for inputs.

    Returns
    -------
    Path
        The written HTML file path.

    Raises
    ------
    ReportError
        For validation and IO faults.
    """
    # Load and validate predictions
    pred_df = _read_csv(pred_path)
    pred = Predictions(pred_df, id_col=id_col, bin_col=bin_col, mu_col=mu_col, sigma_col=sigma_col)
    pred.validate()

    # Optional targets
    tgt: Optional[Targets] = None
    if targets_path:
        tgt_df = _read_csv(targets_path)
        tgt = Targets(tgt_df, id_col=id_col, bin_col=bin_col, target_col=target_col)
        tgt.validate()

    # Optional baseline for delta plot
    base_pred_b64: Optional[str] = None
    if base_pred_path:
        base_df = _read_csv(base_pred_path)
        base = Predictions(base_df, id_col=id_col, bin_col=bin_col, mu_col=mu_col, sigma_col=sigma_col)
        base.validate()
        base_pred_b64 = _plot_inject_recover_delta(base, pred, label_base="base", label_pert="perturbed")

    # Lineage aggregation
    events, lineage = _read_events(events_path)

    # Stats and plots
    stats = _binwise_stats(pred, tgt)
    mse = stats.get("mse")
    mae = stats.get("mae")
    nll = stats.get("nll")

    p_bin_mse_b64 = _plot_per_bin_mse(stats)
    sigma_profile_b64 = _plot_sigma_profile(pred)
    pit_b64 = _plot_calibration_pit(pred, tgt)
    resid_scatter_b64 = _plot_residual_scatter(pred, tgt)

    # JSON schema validation (sample)
    schema_msg = _validate_submission_schema(pred, submission_schema_path)

    # Compose HTML
    html = io.StringIO()
    html.write(f"<!doctype html><html><head><meta charset='utf-8' />")
    html.write(f"<title>{report_title}</title>")
    html.write(_inline_css())
    html.write("</head><body>")
    html.write(f"<h1>{report_title}</h1>")

    # KPI grid
    html.write("<div class='kpi-grid'>")
    def _kpi(label: str, value: Optional[float | str]) -> None:
        if value is None:
            return
        if isinstance(value, float):
            txt = f"{value:.6g}"
        else:
            txt = str(value)
        html.write(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{txt}</div></div>")

    _kpi("IDs", pred.df[id_col].nunique())
    _kpi("Rows", len(pred.df))
    _kpi("Bins/ID (expected)", BIN_COUNT)
    if mse is not None:
        _kpi("MSE", float(mse))
    if mae is not None:
        _kpi("MAE", float(mae))
    if nll is not None:
        _kpi("NLL", float(nll))
    html.write("</div>")

    # Plots
    html.write("<div class='section'><h2>Per-bin Diagnostics</h2>")
    html.write(_html_img(p_bin_mse_b64, "Per-bin MSE"))
    html.write(_html_img(sigma_profile_b64, "Sigma profile"))
    html.write("</div>")

    html.write("<div class='section'><h2>Calibration & Residuals</h2>")
    if tgt is None:
        html.write("<div class='warn'>Targets not provided — calibration plots disabled.</div>")
    html.write(_html_img(pit_b64, "PIT histogram"))
    html.write(_html_img(resid_scatter_b64, "Residuals vs mu"))
    html.write("</div>")

    if base_pred_b64:
        html.write("<div class='section'><h2>Inject-&-Recover Comparison</h2>")
        html.write(_html_img(base_pred_b64, "Δμ profile (perturbed − base)"))
        html.write("</div>")

    # Schema validation
    html.write("<div class='section'><h2>Submission Schema Validation</h2>")
    if schema_msg:
        cls = "ok" if "passed" in schema_msg.lower() else "warn" if "skipped" in schema_msg.lower() else "err"
        html.write(f"<div class='{cls}'>{schema_msg}</div>")
    else:
        html.write("<div class='warn'>No schema validation performed.</div>")
    html.write("</div>")

    # Lineage
    html.write("<div class='section'><h2>Lineage & Run Manifest</h2>")
    if lineage:
        html.write("<table>")
        if lineage.git_commits:
            html.write(f"<tr><th>Git commits</th><td class='mono'>{', '.join(lineage.git_commits)}</td></tr>")
        if lineage.config_hashes:
            html.write(f"<tr><th>Config hashes</th><td class='mono'>{', '.join(lineage.config_hashes)}</td></tr>")
        if lineage.artifact_digests:
            html.write(f"<tr><th>Artifact digests</th><td class='mono'>{', '.join(lineage.artifact_digests)}</td></tr>")
        if lineage.stages:
            html.write(f"<tr><th>Stages</th><td class='mono'>{', '.join(lineage.stages)}</td></tr>")
        if lineage.first_event_ts or lineage.last_event_ts:
            html.write(f"<tr><th>Time span</th><td class='mono'>{lineage.first_event_ts} → {lineage.last_event_ts}</td></tr>")
        if lineage.counts_by_level:
            counts = ", ".join(f"{k}:{v}" for k, v in lineage.counts_by_level.items())
            html.write(f"<tr><th>Event levels</th><td class='mono'>{counts}</td></tr>")
        html.write("</table>")
    else:
        html.write("<div class='warn'>No events JSONL provided — lineage section is empty.</div>")
    html.write("</div>")

    # Footer
    html.write("<div class='section'><h2>Provenance</h2>")
    html.write("<ul>")
    html.write(f"<li>Predictions: <span class='mono'>{pred_path}</span></li>")
    if targets_path:
        html.write(f"<li>Targets: <span class='mono'>{targets_path}</span></li>")
    if base_pred_path:
        html.write(f"<li>Baseline predictions (inject&recover): <span class='mono'>{base_pred_path}</span></li>")
    if events_path:
        html.write(f"<li>Events JSONL: <span class='mono'>{events_path}</span></li>")
    if submission_schema_path:
        html.write(f"<li>Submission schema: <span class='mono'>{submission_schema_path}</span></li>")
    html.write("</ul></div>")

    html.write("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html.getvalue(), encoding="utf-8")
    return out_html
```

### How to call it from your CLI

Hook this into your Typer command (e.g., `spectramind diagnose report …`):

```python
# src/spectramind/cli/diagnose.py
import typer
from pathlib import Path
from spectramind.reports import generate_report, ReportError

app = typer.Typer(help="Diagnostics & reporting")

@app.command("report")
def report(
    pred: Path = typer.Argument(..., exists=True, help="Predictions CSV"),
    out: Path = typer.Option(Path("artifacts/reports/diagnostics_dashboard.html"), help="Output HTML"),
    targets: Path = typer.Option(None, exists=True, help="Optional targets CSV"),
    events: Path = typer.Option(None, exists=True, help="Optional events JSONL"),
    schema: Path = typer.Option(None, exists=True, help="Optional submission schema (.json)"),
    base: Path = typer.Option(None, exists=True, help="Optional baseline predictions CSV for inject&recover"),
    title: str = typer.Option("SpectraMind V50 — Diagnostics Dashboard", help="Report title"),
):
    try:
        out_path = generate_report(
            pred_path=pred,
            out_html=out,
            targets_path=targets,
            events_path=events,
            submission_schema_path=schema,
            base_pred_path=base,
            report_title=title,
        )
        typer.secho(f"Report written: {out_path}", fg=typer.colors.GREEN)
    except ReportError as e:
        typer.secho(f"[report] error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
```

### Notes & Extensions

* **Schema validation:** samples 100 rows to keep it fast (full validation can be added behind a flag).
* **PIT & calibration:** expects targets; if not provided, section is disabled gracefully.
* **Inject-&-Recover:** Provide `--base` baseline predictions to visualize Δμ. For Gaussian band-injection, you can generate a perturbed submission with your calibration module and compare.
* **CI:** add a `make report` target and upload the HTML as a CI artifact.
* **Security:** everything is offline, no remote scripts, suitable for Kaggle.
