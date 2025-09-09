# src/spectramind/diagnostics/reports.py
# =============================================================================
# SpectraMind V50 — Diagnostics Report Generator (Upgraded)
# -----------------------------------------------------------------------------
# - Optional deps (yaml, pandas, jinja2, matplotlib) are handled gracefully
# - Deterministic, headless-safe plotting (no custom colors)
# - Caps on table sizes and digest traversal for CI/Kaggle speed
# - Root-scoped file digests with ignore patterns
# - HTML (Jinja2) or Markdown fallback; sidecar JSON manifest
# - Minimal environment + versions snapshot for reproducibility
# - No global state leaks (figs closed; buffers freed)
# =============================================================================

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    # py311+ stdlib
    from importlib.metadata import distributions, PackageNotFoundError  # type: ignore
except Exception:  # pragma: no cover
    distributions = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore

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

# Force a non-interactive backend in headless environments *without*
# importing pyplot globally if import fails.
plt = None
try:
    import matplotlib

    # Avoid user/system rc surprises in CI
    if "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg", force=True)  # pragma: no cover
    import matplotlib.pyplot as _plt  # type: ignore

    plt = _plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore

# -----------------------
# Constants / Limits
# -----------------------
MAX_TABLE_ROWS = int(os.environ.get("SM_REPORT_MAX_TABLE_ROWS", "20"))
MAX_TABLE_COLS = int(os.environ.get("SM_REPORT_MAX_TABLE_COLS", "40"))
MAX_DIGEST_FILES = int(os.environ.get("SM_REPORT_MAX_DIGEST_FILES", "5000"))
DIGEST_CHUNK_SIZE = 65536
DEFAULT_REPORT_NAME = "report.html"

# Scope digests to repo by default (safety on Kaggle/CI)
DEFAULT_DIGEST_ROOT = Path(os.environ.get("SM_REPORT_DIGEST_ROOT", ".")).resolve()
DEFAULT_IGNORE_GLOBS = {
    ".git/**",
    ".dvc/cache/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
    "site/**",
    "build/**",
    "dist/**",
    "artifacts/**",  # avoid hashing very large model dirs unless explicitly added
}

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
    versions: Dict[str, str] = field(default_factory=dict)
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
        for chunk in iter(lambda: f.read(DIGEST_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _matches_ignored(p: Path, root: Path, patterns: Iterable[str]) -> bool:
    try:
        rel = p.relative_to(root)
    except Exception:
        return True  # out-of-root: ignore
    s = str(rel).replace("\\", "/")
    # naive glob matching (Path.match matches from end segments too)
    return any(rel.match(glob) or Path(s).match(glob) for glob in patterns)


def _gather_digests(
    paths: List[Path],
    *,
    root: Path = DEFAULT_DIGEST_ROOT,
    ignore_globs: Iterable[str] = DEFAULT_IGNORE_GLOBS,
    max_files: int = MAX_DIGEST_FILES,
) -> List[FileDigest]:
    out: List[FileDigest] = []
    count = 0
    for p in paths:
        p = p.resolve()
        # Skip anything outside root unless explicitly requested and within root
        if not str(p).startswith(str(root)):
            continue
        if p.is_file():
            if not _matches_ignored(p, root, ignore_globs):
                out.append(FileDigest(path=str(p), size_bytes=p.stat().st_size, sha256=_sha256_file(p)))
                count += 1
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if count >= max_files:
                    return out
                if child.is_file() and not _matches_ignored(child, root, ignore_globs):
                    out.append(FileDigest(path=str(child), size_bytes=child.stat().st_size, sha256=_sha256_file(child)))
                    count += 1
    return out


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path or yaml is None or not path.exists():
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
        low = path.suffix.lower()
        if low == ".csv":
            df = pd.read_csv(path)  # type: ignore
        elif low == ".parquet":
            df = pd.read_parquet(path)  # type: ignore
        elif low == ".json":
            df = pd.read_json(path)  # type: ignore
        else:
            return None
        # Light trimming for speed and HTML size
        if df.shape[1] > MAX_TABLE_COLS:
            df = df.iloc[:, :MAX_TABLE_COLS]
        if df.shape[0] > 10_000:
            df = df.head(10_000)  # hard cap for memory safety
        return df
    except Exception:
        return None


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=144)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _safe_close(fig) -> None:
    try:
        if plt is not None:
            plt.close(fig)
    except Exception:  # pragma: no cover
        pass


# -----------------------
# Plot helpers
# -----------------------

def plot_training_curves(history: Optional["pd.DataFrame"]) -> Optional[str]:
    if plt is None or pd is None or history is None or history.empty:
        return None
    cols = [c for c in history.columns if c.lower() not in {"epoch", "step", "time"}]
    if not cols:
        return None
    try:
        x = history["epoch"] if "epoch" in history.columns else history.index
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in cols:
            try:
                ax.plot(x, history[c], label=str(c))
            except Exception:
                continue
        ax.set_title("Training Curves")
        ax.set_xlabel("epoch" if "epoch" in history.columns else "index")
        ax.set_ylabel("value")
        ax.legend(loc="best", frameon=False)
        uri = _fig_to_data_uri(fig)
    finally:
        _safe_close(locals().get("fig"))
    return uri


def plot_spectrum(mu_sigma_df: Optional["pd.DataFrame"]) -> Optional[str]:
    """Plot a representative predicted spectrum (mu ± sigma) if available."""
    if plt is None or pd is None or mu_sigma_df is None or mu_sigma_df.empty:
        return None
    mu_cols = sorted([c for c in mu_sigma_df.columns if c.startswith("mu_")])
    sigma_cols = sorted([c for c in mu_sigma_df.columns if c.startswith("sigma_")])
    if not mu_cols or not sigma_cols:
        return None
    try:
        row = mu_sigma_df.iloc[0]
        mu = [float(row[c]) for c in mu_cols]
        sig = [float(row[c]) for c in sigma_cols]
        x = list(range(len(mu)))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x, mu, label="μ")
        lower = [m - s for m, s in zip(mu, sig)]
        upper = [m + s for m, s in zip(mu, sig)]
        ax.fill_between(x, lower, upper, alpha=0.25, label="σ band")
        ax.set_title("Predicted Spectrum (μ ± σ)")
        ax.set_xlabel("bin index")
        ax.set_ylabel("depth")
        ax.legend(loc="best", frameon=False)
        uri = _fig_to_data_uri(fig)
    finally:
        _safe_close(locals().get("fig"))
    return uri


# -----------------------
# Environment / versions
# -----------------------

def _minimal_env_snapshot() -> Dict[str, Any]:
    keys = [
        "PYTHONHASHSEED", "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK",
        "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
    ]
    snap = {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}
    snap["python"] = sys.version.split()[0]
    return snap


def _package_versions_snapshot(max_pkgs: int = 200) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if distributions is None:
        return out
    try:
        # Keep deterministic order by name
        pkgs = sorted(distributions(), key=lambda d: d.metadata.get("Name", "").lower())
        for d in pkgs[:max_pkgs]:
            name = d.metadata.get("Name")
            version = d.version
            if name and version:
                out[name] = version
    except Exception:
        return out
    return out


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
        return txt[:12]  # detached
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
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = RunInfo(
        run_id=run_id,
        timestamp_utc=_iso_now(),
        repo_version=_read_version_file(Path("VERSION")),
        git_commit=os.environ.get("GIT_COMMIT") or _maybe_git_commit(),
        dvc_lock_hash=_maybe_dvc_lock_hash(),
        env=_minimal_env_snapshot(),
        versions=_package_versions_snapshot(),
        notes=notes,
    )

    cfg_path = Path(config_path).resolve() if config_path else None
    cfg = ConfigSnapshot(
        config_path=str(cfg_path) if cfg_path else None,
        data=_read_yaml(cfg_path) if cfg_path else {},
    )

    m = _read_json(Path(metrics_json) if metrics_json else None)
    history = _read_table_any(Path(history_csv) if history_csv else None)
    preds = _read_table_any(Path(predictions_csv) if predictions_csv else None)

    metrics = MetricsBlock(
        scalars={k: float(v) for k, v in m.items() if isinstance(v, (int, float))},
        tables={},
        figures={},
    )

    # Figures
    curve_uri = plot_training_curves(history)
    if curve_uri:
        metrics.figures["training_curves"] = curve_uri

    spec_uri = plot_spectrum(preds)
    if spec_uri:
        metrics.figures["predicted_spectrum"] = spec_uri

    # Table previews (kept small)
    if pd is not None:
        if history is not None and not history.empty:
            metrics.tables["history_head"] = history.head(MAX_TABLE_ROWS)
        if preds is not None and not preds.empty:
            # prioritize ID + first 10 mu/sigma columns to keep light
            mu = [c for c in preds.columns if c.startswith("mu_")]
            sg = [c for c in preds.columns if c.startswith("sigma_")]
            show_cols: List[str] = []
            if "sample_id" in preds.columns:
                show_cols.append("sample_id")
            show_cols += sorted(mu)[:10] + sorted(sg)[:10]
            metrics.tables["preds_head"] = preds.loc[:, [c for c in show_cols if c in preds.columns]].head(
                min(MAX_TABLE_ROWS, 10)
            )

    # File digests: only include paths within DEFAULT_DIGEST_ROOT
    digest_targets: List[Path] = []
    for maybe in (cfg_path, Path(metrics_json) if metrics_json else None,
                  Path(history_csv) if history_csv else None,
                  Path(predictions_csv) if predictions_csv else None):
        if maybe:
            digest_targets.append(maybe)

    if extra_digest_paths:
        digest_targets.extend(Path(p) for p in extra_digest_paths)

    digests = _gather_digests([p for p in digest_targets if p])

    return ReportData(
        title="SpectraMind V50 — Diagnostics Report",
        run=run,
        config=cfg,
        metrics=metrics,
        digests=digests,
    )


# -----------------------
# Rendering
# -----------------------

def render_report(
    data: ReportData,
    out_dir: Union[str, Path],
    filename: str = DEFAULT_REPORT_NAME,
    templates_dir: Optional[Union[str, Path]] = None,
    write_manifest_json: bool = True,
) -> Path:
    """
    Render the report to HTML (Jinja2) or Markdown fallback if Jinja2 unavailable.

    Returns path to the generated report file (HTML or .md).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Sidecar manifest (.json) for programmatic inspection
    if write_manifest_json:
        manifest = {
            "title": data.title,
            "generated_at": _iso_now(),
            "run": asdict(data.run),
            "config_path": data.config.config_path,
            "metrics_scalars": data.metrics.scalars,
            "digests": [asdict(d) for d in data.digests],
        }
        (out_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if Environment is None:
        md_path = out_path.with_suffix(".md")
        md_path.write_text(_render_markdown(data), encoding="utf-8")
        return md_path

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
    """Provide an embedded minimal template directory (created on first use)."""
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
            trimmed = df.head(MAX_TABLE_ROWS)
            out[k] = trimmed.to_html(index=False, border=0, justify="center")  # type: ignore
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
    @media (prefers-color-scheme: dark) {
      :root { --fg: #e6edf3; --bg: #0d1117; --muted: #8b949e; --accent: #4493f8; }
      pre, code { background: #161b22 !important; }
      .card { background: #0d1117 !important; border-color: #30363d !important; }
      table td, table th { border-bottom-color: #21262d !important; }
    }
    html, body { margin:0; padding:0; background: var(--bg); color: var(--fg);
                 font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial,
                              "Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol", sans-serif; }
    .container { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
    h1, h2, h3 { margin: 1.2rem 0 .8rem; }
    pre, code { background: #f6f8fa; padding: .5rem; border-radius: 6px; overflow-x: auto; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
    .card { border: 1px solid #e1e4e8; border-radius: 8px; padding: 1rem; background: #fff; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: .4rem .6rem; border-bottom: 1px solid #eaecef; font-size: 90%; text-align: left; }
    .muted { color: var(--muted); }
    img { max-width: 100%; height: auto; }
    .kbd { font: 11px/1.4 ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;
           background: #0b0b0c11; border: 1px solid #0001; border-radius: 4px; padding: 1px 4px; }
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
    filename: str = DEFAULT_REPORT_NAME,
) -> Path:
    """
    One-shot: collect data and render the HTML (or Markdown) report.
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
    report_path = render_report(
        data,
        out_dir=artifacts_dir,
        filename=filename,
        templates_dir=templates_dir,
    )
    return report_path
