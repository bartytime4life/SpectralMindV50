# src/spectramind/diagnostics/reports.py
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional dependencies (all handled gracefully)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape  # type: ignore
except Exception:  # pragma: no cover
    Environment = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:  # pragma: no cover
    plt = None  # type: ignore


# -----------------------
# Datamodels
# -----------------------

@dataclass
class RunInfo:
    run_id: str
    timestamp_utc: str
    repo_version: Optional[str] = None
    git_commit: Optional[str] = None
    dvc_lock_hash: Optional[str] = None
    env: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None


@dataclass
class FileDigest:
    path: str
    size_bytes: int
    sha256: str


@dataclass
class MetricsBlock:
    scalars: Dict[str, float] = field(default_factory=dict)
    tables: Dict[str, "pd.DataFrame"] = field(default_factory=dict)  # if pandas available
    figures: Dict[str, str] = field(default_factory=dict)  # name -> data URI (PNG)


@dataclass
class ConfigSnapshot:
    config_path: Optional[str]
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportData:
    title: str
    run: RunInfo
    config: ConfigSnapshot
    metrics: MetricsBlock
    digests: List[FileDigest] = field(default_factory=list)


# -----------------------
# Utilities
# -----------------------

def _iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _gather_digests(paths: List[Path]) -> List[FileDigest]:
    out: List[FileDigest] = []
    for p in paths:
        if p.is_file():
            out.append(FileDigest(path=str(p), size_bytes=p.stat().st_size, sha256=_sha256_file(p)))
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    out.append(FileDigest(path=str(child), size_bytes=child.stat().st_size, sha256=_sha256_file(child)))
    return out


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path or yaml is None:
        return {}
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)  # type: ignore
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_table_any(path: Optional[Path]) -> Optional["pd.DataFrame"]:
    if pd is None or not path or not path.exists():
        return None
    try:
        if path.suffix.lower() in {".csv"}:
            return pd.read_csv(path)  # type: ignore
        if path.suffix.lower() in {".parquet"}:
            return pd.read_parquet(path)  # type: ignore
        if path.suffix.lower() in {".json"}:
            return pd.read_json(path)  # type: ignore
    except Exception:
        return None
    return None


def _fig_to_data_uri(fig) -> str:
    """Return a PNG data URI for a matplotlib fig (no custom colors for safety)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# -----------------------
# Plot helpers
# -----------------------

def plot_training_curves(history: Optional["pd.DataFrame"]) -> Optional[str]:
    if plt is None or pd is None or history is None or history.empty:
        return None
    cols = [c for c in history.columns if c.lower() not in {"epoch", "step", "time"}]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot all numeric columns
    for c in cols:
        try:
            ax.plot(history.index if "epoch" not in history.columns else history["epoch"], history[c], label=c)
        except Exception:
            continue
    ax.set_title("Training Curves")
    ax.set_xlabel("epoch" if "epoch" in history.columns else "index")
    ax.set_ylabel("value")
    ax.legend()
    uri = _fig_to_data_uri(fig)
    plt.close(fig)
    return uri


def plot_spectrum(mu_sigma_df: Optional["pd.DataFrame"]) -> Optional[str]:
    """Plot a representative predicted spectrum (mu +/- sigma) if available."""
    if plt is None or pd is None or mu_sigma_df is None or mu_sigma_df.empty:
        return None
    # Expect columns like mu_000 ... mu_282 and sigma_000 ... sigma_282 (283 bins)
    mu_cols = [c for c in mu_sigma_df.columns if c.startswith("mu_")]
    sigma_cols = [c for c in mu_sigma_df.columns if c.startswith("sigma_")]
    if not mu_cols or not sigma_cols:
        return None
    # Choose the first row as representative
    row = mu_sigma_df.iloc[0]
    mu = [row[c] for c in sorted(mu_cols)]
    sig = [row[c] for c in sorted(sigma_cols)]
    x = list(range(len(mu)))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, mu, label="μ")
    # Shaded uncertainty (no explicit color to comply with neutral style rules)
    ax.fill_between(x, [m - s for m, s in zip(mu, sig)], [m + s for m, s in zip(mu, sig)], alpha=0.25, label="σ band")
    ax.set_title("Predicted Spectrum (μ ± σ)")
    ax.set_xlabel("bin index")
    ax.set_ylabel("depth")
    ax.legend()
    uri = _fig_to_data_uri(fig)
    plt.close(fig)
    return uri


# -----------------------
# Report composition
# -----------------------

def collect_report_data(
    *,
    run_id: str,
    output_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    metrics_json: Optional[Union[str, Path]] = None,
    history_csv: Optional[Union[str, Path]] = None,
    predictions_csv: Optional[Union[str, Path]] = None,
    extra_digest_paths: Optional[List[Union[str, Path]]] = None,
    notes: Optional[str] = None,
) -> ReportData:
    """
    Collects and prepares all data needed for the diagnostics report.

    Parameters
    ----------
    run_id : str
        Unique run identifier (hash, timestamp, UUID).
    output_dir : Path-like
        Artifacts directory where report files will be written.
    config_path : Path-like
        Hydra-composed config path (.yaml).
    metrics_json : Path-like
        JSON with aggregate metrics (e.g., loss, gll, val scores).
    history_csv : Path-like
        Training log (CSV or parquet/json) for learning curves.
    predictions_csv : Path-like
        Predictions (CSV/JSON/parquet) with columns mu_***, sigma_*** if present.
    extra_digest_paths : List[Path-like]
        Additional files/dirs to hash and include in the audit digest.
    notes : str
        Freeform notes to attach to the run block.

    Returns
    -------
    ReportData
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run info
    run = RunInfo(
        run_id=run_id,
        timestamp_utc=_iso_now(),
        repo_version=_read_version_file(Path("VERSION")),
        git_commit=os.environ.get("GIT_COMMIT") or _maybe_git_commit(),
        dvc_lock_hash=_maybe_dvc_lock_hash(),
        env=_minimal_env_snapshot(),
        notes=notes,
    )

    # Config snapshot
    cfg_path = Path(config_path) if config_path else None
    cfg = ConfigSnapshot(
        config_path=str(cfg_path) if cfg_path else None,
        data=_read_yaml(cfg_path),
    )

    # Metrics
    m = _read_json(Path(metrics_json) if metrics_json else None)
    history = _read_table_any(Path(history_csv) if history_csv else None)
    preds = _read_table_any(Path(predictions_csv) if predictions_csv else None)

    metrics = MetricsBlock(
        scalars={k: float(v) for k, v in m.items() if isinstance(v, (int, float))},
        tables={},  # only embed small summaries to avoid heavy HTML
        figures={}
    )

    # Training curves figure
    curve_uri = plot_training_curves(history)
    if curve_uri:
        metrics.figures["training_curves"] = curve_uri

    # Spectrum figure (μ ± σ) if available
    spec_uri = plot_spectrum(preds)
    if spec_uri:
        metrics.figures["predicted_spectrum"] = spec_uri

    # Table previews (cap rows/cols to keep HTML small)
    if pd is not None:
        if history is not None and not history.empty:
            metrics.tables["history_head"] = history.head(20)
        if preds is not None and not preds.empty:
            want = [c for c in preds.columns if c.startswith("mu_") or c.startswith("sigma_")]
            metrics.tables["preds_head"] = preds[["sample_id"] + want[:10]] if "sample_id" in preds.columns else preds[want[:10]].head(10)

    # File digests
    digest_targets = []
    if cfg_path:
        digest_targets.append(cfg_path)
    if metrics_json:
        digest_targets.append(Path(metrics_json))
    if history_csv:
        digest_targets.append(Path(history_csv))
    if predictions_csv:
        digest_targets.append(Path(predictions_csv))
    if extra_digest_paths:
        digest_targets.extend([Path(p) for p in extra_digest_paths])

    digests = _gather_digests([p for p in digest_targets if p])

    return ReportData(
        title="SpectraMind V50 — Diagnostics Report",
        run=run,
        config=cfg,
        metrics=metrics,
        digests=digests,
    )


def _minimal_env_snapshot() -> Dict[str, Any]:
    keys = [
        "PYTHONHASHSEED", "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK",
        "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
    ]
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}


def _read_version_file(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return None


def _maybe_git_commit() -> Optional[str]:
    head = Path(".git/HEAD")
    if not head.exists():
        return None
    try:
        txt = head.read_text(encoding="utf-8").strip()
        if txt.startswith("ref:"):
            ref = txt.split(" ", 1)[1].strip()
            ref_path = Path(".git") / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
        # detached
        return txt[:12]
    except Exception:
        return None


def _maybe_dvc_lock_hash() -> Optional[str]:
    lock = Path("dvc.lock")
    if not lock.exists():
        return None
    try:
        return _sha256_file(lock)
    except Exception:
        return None


# -----------------------
# Rendering
# -----------------------

def render_report(
    data: ReportData,
    out_dir: Union[str, Path],
    filename: str = "report.html",
    templates_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Render the report to HTML (Jinja2) or Markdown fallback if Jinja2 unavailable.

    Returns
    -------
    Path to the generated report file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if Environment is None:
        # Markdown fallback
        md_path = out_path.with_suffix(".md")
        md_path.write_text(_render_markdown(data), encoding="utf-8")
        return md_path

    # Jinja2 HTML
    env = Environment(
        loader=FileSystemLoader(str(templates_dir) if templates_dir else str(_default_templates_dir())),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=False,
    )
    tpl = env.get_template("diagnostics.html.j2")
    html = tpl.render(
        title=data.title,
        run=asdict(data.run),
        config_path=data.config.config_path,
        config_yaml=_yaml_dump_pretty(data.config.data),
        metrics_scalars=data.metrics.scalars,
        metrics_tables=_tables_to_html(data.metrics.tables),
        figures=data.metrics.figures,
        digests=[asdict(d) for d in data.digests],
        generated_at=_iso_now(),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _default_templates_dir() -> Path:
    """Provide an embedded minimal template directory (created in-memory on first use)."""
    # Create a temp template folder under artifacts if not provided
    d = Path(".reports_templates")
    d.mkdir(exist_ok=True)
    tpl = d / "diagnostics.html.j2"
    if not tpl.exists():
        tpl.write_text(_DEFAULT_TEMPLATE_HTML, encoding="utf-8")
    return d


def _yaml_dump_pretty(data: Dict[str, Any]) -> str:
    if not data:
        return "(no config)"
    if yaml is None:
        # JSON as fallback
        return json.dumps(data, indent=2, sort_keys=True)
    try:
        return yaml.safe_dump(data, sort_keys=True, indent=2)  # type: ignore
    except Exception:
        return json.dumps(data, indent=2, sort_keys=True)


def _tables_to_html(tables: Dict[str, "pd.DataFrame"]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if pd is None:
        return out
    for k, df in tables.items():
        try:
            out[k] = df.to_html(index=False, border=0, justify="center")  # type: ignore
        except Exception:
            continue
    return out


def _render_markdown(data: ReportData) -> str:
    lines = []
    lines.append(f"# {data.title}")
    lines.append("")
    lines.append("## Run")
    lines.append("```json")
    lines.append(json.dumps(asdict(data.run), indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Config")
    lines.append("```yaml")
    lines.append(_yaml_dump_pretty(data.config.data))
    lines.append("```")
    lines.append("")
    if data.metrics.scalars:
        lines.append("## Metrics (Scalars)")
        lines.append("```json")
        lines.append(json.dumps(data.metrics.scalars, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    for name, uri in data.metrics.figures.items():
        lines.append(f"### Figure: {name}")
        lines.append(f"![{name}]({uri})")
        lines.append("")
    if data.digests:
        lines.append("## File Digests")
        lines.append("```json")
        lines.append(json.dumps([asdict(d) for d in data.digests], indent=2, sort_keys=True))
        lines.append("```")
    lines.append("")
    lines.append(f"_Generated at: {_iso_now()}_")
    return "\n".join(lines)


_DEFAULT_TEMPLATE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { --fg: #1b1f23; --bg: #ffffff; --muted: #6a737d; --accent: #0366d6; }
    html, body { margin:0; padding:0; background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji", "Segoe UI Symbol", sans-serif; }
    .container { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
    h1, h2, h3 { margin: 1.2rem 0 .8rem; }
    pre, code { background: #f6f8fa; padding: .5rem; border-radius: 6px; overflow-x: auto; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
    .card { border: 1px solid #e1e4e8; border-radius: 8px; padding: 1rem; background: #fff; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: .4rem .6rem; border-bottom: 1px solid #eaecef; font-size: 90%; }
    .muted { color: var(--muted); }
    img { max-width: 100%; height: auto; }
  </style>
</head>
<body>
<div class="container">
  <h1>{{ title }}</h1>

  <h2>Run</h2>
  <div class="card">
    <pre><code>{{ run | tojson(indent=2) }}</code></pre>
  </div>

  <h2>Config</h2>
  <div class="card">
    <div class="muted">Path: {{ config_path or "(none)" }}</div>
    <pre><code>{{ config_yaml }}</code></pre>
  </div>

  {% if metrics_scalars %}
  <h2>Metrics (Scalars)</h2>
  <div class="card">
    <pre><code>{{ metrics_scalars | tojson(indent=2) }}</code></pre>
  </div>
  {% endif %}

  {% if figures %}
  <h2>Figures</h2>
  <div class="grid">
    {% for name, uri in figures.items() %}
    <div class="card">
      <h3>{{ name }}</h3>
      <img alt="{{ name }}" src="{{ uri }}" />
    </div>
    {% endfor %}
  </div>
  {% endif %}

  {% if metrics_tables %}
  <h2>Tables</h2>
  <div class="grid">
    {% for name, html in metrics_tables.items() %}
    <div class="card">
      <h3>{{ name }}</h3>
      {{ html | safe }}
    </div>
    {% endfor %}
  </div>
  {% endif %}

  {% if digests %}
  <h2>Audit — File Digests</h2>
  <div class="card">
    <table>
      <thead><tr><th>Path</th><th>Size (bytes)</th><th>sha256</th></tr></thead>
      <tbody>
      {% for d in digests %}
        <tr>
          <td>{{ d.path }}</td>
          <td>{{ '{:,}'.format(d.size_bytes) }}</td>
          <td><code>{{ d.sha256 }}</code></td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  <p class="muted">Generated at {{ generated_at }}</p>
</div>
</body>
</html>
"""


# -----------------------
# High-level convenience
# -----------------------

def generate_report(
    *,
    run_id: str,
    artifacts_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    metrics_json: Optional[Union[str, Path]] = None,
    history_csv: Optional[Union[str, Path]] = None,
    predictions_csv: Optional[Union[str, Path]] = None,
    extra_digest_paths: Optional[List[Union[str, Path]]] = None,
    notes: Optional[str] = None,
    templates_dir: Optional[Union[str, Path]] = None,
    filename: str = "report.html",
) -> Path:
    """
    One-shot: collect data and render the HTML (or Markdown) report.

    Returns
    -------
    Path to the generated report file.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = collect_report_data(
        run_id=run_id,
        output_dir=artifacts_dir,
        config_path=config_path,
        metrics_json=metrics_json,
        history_csv=history_csv,
        predictions_csv=predictions_csv,
        extra_digest_paths=extra_digest_paths,
        notes=notes,
    )
    report_path = render_report(data, out_dir=artifacts_dir, filename=filename, templates_dir=templates_dir)
    return report_path
