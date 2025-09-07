Below is a production-grade, Typer-based **unified CLI** for SpectraMind V50.
Save as `src/spectramind/cli.py`.

It implements the core subcommands you outlined:

* `calibrate` → raw → calibrated cubes (optionally via DVC stage)
* `train` → dual encoders + decoder training (delegates to internal API)
* `predict` → checkpoint → predictions CSV
* `diagnose report` → full HTML dashboard (uses `reports.generate_report`)
* `submit` → package & validate submission (zip/csv/schema)

It’s Kaggle/CI-safe (no internet use), strongly-typed, with **loud, typed errors**, optional **Hydra/OmegaConf** overrides if available, optional **DVC** integration, and nice UX (colors, autocompletion, optional rich tracebacks).

> Wherever you see `TODO:` delegates (e.g., `spectramind.pipeline.calibrate.run`), wire them to your real modules.
> The CLI stays thin; your business logic remains in `src/spectramind/{pipeline,models,inference,...}.py`.

```python
# src/spectramind/cli.py
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer

# Optional niceties
try:
    from rich import print as rprint
    from rich.traceback import install as rich_install
    _HAS_RICH = True
    rich_install(show_locals=True, suppress=["typer", "click"])
except Exception:
    _HAS_RICH = False
    rprint = print  # type: ignore[assignment]

# Optional Hydra / OmegaConf support
try:
    from omegaconf import OmegaConf  # type: ignore
    _HAS_OMEGACONF = True
except Exception:
    _HAS_OMEGACONF = False

app = typer.Typer(
    name="spectramind",
    help="SpectraMind V50 — Mission-grade CLI for the NeurIPS 2025 Ariel Data Challenge.",
    no_args_is_help=True,
    add_completion=True,
)

# =============================================================================
# Typed error
# =============================================================================

class SpectraMindError(RuntimeError):
    """Typed error for SpectraMind CLI failures."""

def _fail(msg: str, code: int = 1) -> None:
    if _HAS_RICH:
        rprint(f"[bold red]error:[/bold red] {msg}")
    else:
        typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)

def _ok(msg: str) -> None:
    if _HAS_RICH:
        rprint(f"[bold green]✓[/bold green] {msg}")
    else:
        typer.secho(msg, fg=typer.colors.GREEN)

def _warn(msg: str) -> None:
    if _HAS_RICH:
        rprint(f"[yellow]warn:[/yellow] {msg}")
    else:
        typer.secho(f"warn: {msg}", fg=typer.colors.YELLOW)

# =============================================================================
# Utilities
# =============================================================================

def _run_dvc_stage(stage: str, cwd: Optional[Path] = None, extra: Optional[List[str]] = None) -> None:
    """
    Reproduce a DVC stage with guardrails. Requires 'dvc' on PATH.
    """
    cmd = ["dvc", "repro", "-f", "-s", stage]
    if extra:
        cmd += extra
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
    except FileNotFoundError:
        raise SpectraMindError("DVC is not installed or not on PATH.")
    except subprocess.CalledProcessError as e:
        raise SpectraMindError(f"DVC stage '{stage}' failed: {e}")

def _maybe_load_cfg(config: Optional[Path], overrides: List[str]) -> Optional[dict]:
    """
    If OmegaConf available, merge config and CLI overrides into a dict.
    Otherwise returns None. Non-fatal.
    """
    if not _HAS_OMEGACONF:
        return None
    cfg: dict = {}
    base = {}
    if config and config.exists():
        try:
            base = OmegaConf.to_container(OmegaConf.load(str(config)), resolve=True)  # type: ignore
        except Exception as e:
            raise SpectraMindError(f"Failed to load config: {config} :: {e}")
    try:
        ov = OmegaConf.from_dotlist(overrides) if overrides else OmegaConf.create({})  # type: ignore
        merged = OmegaConf.merge(base, ov)  # type: ignore
        cfg = OmegaConf.to_container(merged, resolve=True)  # type: ignore
        if not isinstance(cfg, dict):
            cfg = {}
        return cfg
    except Exception as e:
        raise SpectraMindError(f"Failed to apply overrides: {overrides} :: {e}")

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _ensure_exists(p: Optional[Path], label: str) -> None:
    if p is not None and not p.exists():
        raise SpectraMindError(f"{label} not found: {p}")

# =============================================================================
# Sub-apps
# =============================================================================

calib_app = typer.Typer(help="Calibration (raw → calibrated cubes)")
train_app = typer.Typer(help="Model training")
predict_app = typer.Typer(help="Prediction / inference")
diagnose_app = typer.Typer(help="Diagnostics & reporting")
submit_app = typer.Typer(help="Submission packaging / validation")

app.add_typer(calib_app, name="calibrate")
app.add_typer(train_app, name="train")
app.add_typer(predict_app, name="predict")
app.add_typer(diagnose_app, name="diagnose")
app.add_typer(submit_app, name="submit")

# =============================================================================
# calibrate
# =============================================================================

@calib_app.command("run")
def calibrate_run(
    raw_dir: Path = typer.Argument(..., exists=True, help="Directory with raw telescope inputs"),
    out_dir: Path = typer.Option(Path("data/interim/calibrated"), help="Output directory for calibrated cubes"),
    config: Optional[Path] = typer.Option(None, exists=True, help="Optional Hydra/OmegaConf config file"),
    override: List[str] = typer.Option([], help="Optional OmegaConf dotlist overrides, e.g. calib.exposure=2.0"),
    use_dvc: bool = typer.Option(False, help="Reproduce via DVC stage 'calibrate' if available"),
    max_runtime_min: int = typer.Option(540, help="Runtime fence (minutes) for Kaggle (default 9h)"),
):
    """
    Run calibration (ADC → dark → flat → CDS → photometry → trace → phase).

    If --use-dvc is set, reproduces the 'calibrate' stage via DVC. Otherwise, calls the python API.
    """
    t0 = time.time()
    try:
        cfg = _maybe_load_cfg(config, override) or {}
        _ensure_exists(raw_dir, "raw_dir")
        _ensure_dir(out_dir)

        if use_dvc:
            _run_dvc_stage("calibrate")
        else:
            # TODO: wire your real calibrator:
            # from spectramind.pipeline.calibrate import run as calib_impl
            # calib_impl(raw_dir=raw_dir, out_dir=out_dir, cfg=cfg)
            _warn("Using placeholder calibrator — wire spectramind.pipeline.calibrate.run(...)")
            # Simulate a basic I/O check
            if not any(raw_dir.iterdir()):
                raise SpectraMindError("raw_dir is empty — no files to calibrate.")
            # Touch a sentinel
            (out_dir / "_calibration_done.txt").write_text("ok\n", encoding="utf-8")

        elapsed = (time.time() - t0) / 60.0
        if elapsed > max_runtime_min:
            raise SpectraMindError(
                f"Calibration exceeded runtime fence: {elapsed:.1f} min > {max_runtime_min} min"
            )
        _ok(f"Calibration completed → {out_dir} (elapsed {elapsed:.1f} min)")
    except SpectraMindError as e:
        _fail(str(e))

# =============================================================================
# train
# =============================================================================

@train_app.command("run")
def train_run(
    data_dir: Path = typer.Option(Path("data/processed/tensors"), exists=True, help="Prepared tensors directory"),
    out_dir: Path = typer.Option(Path("artifacts/checkpoints"), help="Output directory for checkpoints"),
    config: Optional[Path] = typer.Option(None, exists=True, help="Optional training config (.yaml)"),
    override: List[str] = typer.Option([], help="Optional OmegaConf dotlist overrides"),
    seed: int = typer.Option(42, help="Random seed"),
    device: str = typer.Option("auto", help="Device (cpu|cuda|auto)"),
    use_dvc: bool = typer.Option(False, help="Reproduce via DVC stage 'train' if available"),
):
    """
    Train the dual-encoder model (FGS1 encoder + AIRS encoder + heteroscedastic decoder).
    """
    try:
        cfg = _maybe_load_cfg(config, override) or {}
        _ensure_exists(data_dir, "data_dir")
        _ensure_dir(out_dir)

        if use_dvc:
            _run_dvc_stage("train")
        else:
            # TODO: wire your real trainer:
            # from spectramind.training.runner import train as train_impl
            # ckpt = train_impl(data_dir=data_dir, out_dir=out_dir, cfg=cfg, seed=seed, device=device)
            _warn("Using placeholder trainer — wire spectramind.training.runner.train(...)")
            ckpt = out_dir / "model.ckpt"
            ckpt.write_text("placeholder checkpoint", encoding="utf-8")

        _ok(f"Training completed → {out_dir}")
    except SpectraMindError as e:
        _fail(str(e))

# =============================================================================
# predict
# =============================================================================

@predict_app.command("run")
def predict_run(
    ckpt: Path = typer.Argument(..., exists=True, help="Checkpoint to load"),
    data_dir: Path = typer.Option(Path("data/processed/tensors_eval"), exists=True, help="Eval tensors directory"),
    out_csv: Path = typer.Option(Path("artifacts/predictions/preds.csv"), help="Output predictions CSV"),
    config: Optional[Path] = typer.Option(None, exists=True, help="Optional inference config (.yaml)"),
    override: List[str] = typer.Option([], help="Optional OmegaConf dotlist overrides"),
    device: str = typer.Option("auto", help="Device (cpu|cuda|auto)"),
    use_dvc: bool = typer.Option(False, help="Reproduce via DVC stage 'predict' if available"),
):
    """
    Predict spectral μ/σ per bin (283 bins per id) for submission.
    """
    try:
        cfg = _maybe_load_cfg(config, override) or {}
        _ensure_exists(ckpt, "ckpt")
        _ensure_exists(data_dir, "data_dir")
        _ensure_dir(out_csv)

        if use_dvc:
            _run_dvc_stage("predict")
        else:
            # TODO: wire your real inference:
            # from spectramind.inference.predict import run as predict_impl
            # predict_impl(ckpt=ckpt, data_dir=data_dir, out_csv=out_csv, cfg=cfg, device=device)
            _warn("Using placeholder predictor — wire spectramind.inference.predict.run(...)")
            # Create a tiny placeholder CSV with correct columns but no rows
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_csv.write_text("id,bin,mu,sigma\n", encoding="utf-8")

        _ok(f"Predictions written → {out_csv}")
    except SpectraMindError as e:
        _fail(str(e))

# =============================================================================
# diagnose
# =============================================================================

@diagnose_app.command("report")
def diagnose_report(
    pred: Path = typer.Argument(..., exists=True, help="Predictions CSV: id,bin,mu,sigma"),
    out: Path = typer.Option(Path("artifacts/reports/diagnostics_dashboard.html"), help="Output HTML"),
    targets: Optional[Path] = typer.Option(None, exists=True, help="Optional targets CSV: id,bin,target"),
    events: Optional[Path] = typer.Option(None, exists=True, help="Optional JSONL events"),
    schema: Optional[Path] = typer.Option(None, exists=True, help="Optional submission schema (.json)"),
    base: Optional[Path] = typer.Option(None, exists=True, help="Optional baseline preds CSV for Inject-&-Recover"),
    title: str = typer.Option("SpectraMind V50 — Diagnostics Dashboard", help="Report title"),
):
    """
    Build an offline HTML diagnostics dashboard with calibration & residuals plots.

    Uses src/spectramind/reports.py to generate a self-contained report.
    """
    try:
        from spectramind.reports import generate_report, ReportError  # local import to keep CLI slim
    except Exception as e:
        _fail(f"reports module not available: {e}")

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
        _ok(f"Report written → {out_path}")
    except Exception as e:
        # Prefer typed report errors if available
        if e.__class__.__name__ == "ReportError":
            _fail(str(e))
        _fail(f"diagnose report failed: {e}")

# =============================================================================
# submit
# =============================================================================

@submit_app.command("package")
def submit_package(
    preds: Path = typer.Argument(..., exists=True, help="Predictions CSV to package"),
    out_zip: Path = typer.Option(Path("dist/submission.zip"), help="Output ZIP path"),
    schema: Optional[Path] = typer.Option(None, exists=True, help="Optional submission schema to validate"),
    name: str = typer.Option("SpectraMind V50 Submission", help="Package name / label"),
    extra_file: List[Path] = typer.Option([], help="Additional files to include in the archive"),
):
    """
    Package predictions (and extras) into a submission ZIP, with optional JSON schema validation.
    """
    try:
        _ensure_exists(preds, "preds")
        if schema:
            _ensure_exists(schema, "schema")
            # Lightweight validation: re-use schema sampler from reports
            try:
                from spectramind.reports import Predictions, _validate_submission_schema, _read_csv
                df = _read_csv(preds)
                pred = Predictions(df)
                pred.validate()
                msg = _validate_submission_schema(pred, schema)
                if msg:
                    if "failed" in msg.lower():
                        _fail(msg)
                    else:
                        _warn(msg)
            except Exception as e:
                _warn(f"Schema validation skipped: {e}")

        # Build zip
        import zipfile
        _ensure_dir(out_zip)
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(preds, arcname=preds.name)
            manifest = {
                "name": name,
                "generated_by": "spectramind submit package",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
            for p in extra_file:
                if p.exists():
                    zf.write(p, arcname=p.name)
                else:
                    _warn(f"Extra file not found: {p}")

        _ok(f"Submission packaged → {out_zip}")
    except SpectraMindError as e:
        _fail(str(e))


# =============================================================================
# main entrypoint
# =============================================================================

def main() -> None:
    app()

if __name__ == "__main__":
    main()
```

### What this gives you out of the box

* **CLI UX**: autocompletion, colored errors, loud failures, consistent help.
* **Hydra/OmegaConf** (optional): pass `--config configs/train.yaml --override training.epochs=20`.
* **DVC integration**: run any stage with `--use-dvc`; otherwise it calls your Python APIs.
* **Runtime fence**: `calibrate --max-runtime-min 540` to guard Kaggle 9h.
* **Diagnostics**: `diagnose report` produces the self-contained HTML dashboard (from `reports.py`).
* **Submission**: builds a clean ZIP with optional row-wise schema sampling.

### Suggested next steps

* Wire the TODO delegates:

  * `spectramind.pipeline.calibrate.run`
  * `spectramind.training.runner.train`
  * `spectramind.inference.predict.run`
* Add `dvc.yaml` stages named `calibrate`, `train`, `predict`, `submit` to map to these commands.
* In CI, call:

  * `spectramind calibrate run --use-dvc`
  * `spectramind train run --use-dvc`
  * `spectramind predict run --use-dvc`
  * `spectramind diagnose report --pred artifacts/predictions/preds.csv`
  * `spectramind submit package artifacts/predictions/preds.csv`
