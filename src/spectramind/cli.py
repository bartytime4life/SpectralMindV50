# src/spectramind/cli.py
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import typer

# Optional niceties
try:
    from rich import print as rprint
    from rich.traceback import install as rich_install
    from rich.panel import Panel
    from rich.table import Table
    _HAS_RICH = True
    rich_install(show_locals=False, suppress=["typer", "click"])
except Exception:
    _HAS_RICH = False
    rprint = print  # type: ignore[assignment]

# Optional OmegaConf (Hydra-like) support
try:
    from omegaconf import OmegaConf  # type: ignore
    _HAS_OMEGA = True
except Exception:
    _HAS_OMEGA = False

# Optional scientific libs for deterministic seeds (never required at import)
_HAS_TORCH = False
_HAS_NUMPY = False
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    pass

try:
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except Exception:
    pass

from spectramind.utils.logging import get_logger
from spectramind.utils.io import p, read_yaml, read_json, ensure_dir  # YAML optional; JSON always works
from spectramind.train.trainer import train_from_config

__all__ = ["app", "main"]

# ======================================================================================
# App declaration
# ======================================================================================

app = typer.Typer(
    name="spectramind",
    help="SpectraMind V50 — Mission-grade CLI for the NeurIPS 2025 Ariel Data Challenge.",
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

# Subapps
calib_app = typer.Typer(help="Calibration (raw → calibrated cubes)")
train_app = typer.Typer(help="Model training")
predict_app = typer.Typer(help="Prediction / inference")
diagnose_app = typer.Typer(help="Diagnostics & reporting")
submit_app = typer.Typer(help="Submission packaging / validation")
sys_app = typer.Typer(help="System utilities (doctor, version, env, cfg-tools)")

app.add_typer(calib_app, name="calibrate")
app.add_typer(train_app, name="train")
app.add_typer(predict_app, name="predict")
app.add_typer(diagnose_app, name="diagnose")
app.add_typer(submit_app, name="submit")
app.add_typer(sys_app, name="sys")

logger = get_logger(__name__)

# ======================================================================================
# Typed errors / feedback
# ======================================================================================

class SpectraMindError(RuntimeError):
    """Typed error for SpectraMind CLI failures."""

def _is_ci_or_kaggle() -> bool:
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return True
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists():
        return True
    return False

def _ok(msg: str) -> None:
    if _HAS_RICH:
        rprint(f"[bold green]✓[/bold green] {msg}")
    else:
        typer.echo(msg)

def _warn(msg: str) -> None:
    if _HAS_RICH:
        rprint(f"[yellow]warn:[/yellow] {msg}")
    else:
        typer.secho(f"warn: {msg}", fg=typer.colors.YELLOW)

def _fail(msg: str, code: int = 1) -> None:
    if _HAS_RICH:
        rprint(f"[bold red]error:[/bold red] {msg}")
    else:
        typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)

def _ensure_exists(pth: Optional[Path], label: str) -> None:
    if pth is not None and not Path(pth).exists():
        raise SpectraMindError(f"{label} not found: {pth}")

# ======================================================================================
# Determinism / seeding
# ======================================================================================

def _set_seeds(seed: Optional[int], *, deterministic_torch: bool = True) -> None:
    """
    Set global RNG seeds for Python, NumPy, and (optionally) PyTorch.

    Parameters
    ----------
    seed : int | None
        Seed value to set. If None, nothing is changed.
    deterministic_torch : bool, default=True
        If True, configures PyTorch for deterministic ops
        (cudnn.deterministic=True, cudnn.benchmark=False).
    """
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if _HAS_NUMPY:
        _np.random.seed(seed)  # type: ignore[attr-defined]

    if _HAS_TORCH:
        torch.manual_seed(seed)  # type: ignore[name-defined]
        if torch.cuda.is_available():  # type: ignore[name-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]

    _ok(f"Deterministic seeds set → {seed}"
        + ("" if not _HAS_TORCH else f" (torch deterministic={deterministic_torch})"))

# ======================================================================================
# Config loading / overrides
# ======================================================================================

def _coerce_value(val: str) -> Any:
    v = val.strip()
    lv = v.lower()
    if lv in {"true", "yes", "y", "on"}:
        return True
    if lv in {"false", "no", "n", "off"}:
        return False
    if lv in {"null", "none"}:
        return None
    # JSON object/array?
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        try:
            return json.loads(v)
        except Exception:
            pass
    # numeric
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v

def _set_in(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = d
    parts = dotted_key.split(".")
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value

def _load_config_any(config: Optional[Path]) -> Dict[str, Any]:
    """Load YAML/JSON config; returns {} if config is None."""
    if config is None:
        return {}
    cfg_path = p(config)
    if not cfg_path.exists():
        raise SpectraMindError(f"Config not found: {cfg_path}")
    ext = cfg_path.suffix.lower()
    if ext in {".yml", ".yaml"}:
        return read_yaml(cfg_path)  # type: ignore[return-value]
    if ext == ".json":
        return read_json(cfg_path)
    # last resort: try YAML then JSON
    try:
        return read_yaml(cfg_path)  # type: ignore[return-value]
    except Exception:
        return read_json(cfg_path)

def _merge_overrides(base: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply CLI overrides. Uses OmegaConf if available; otherwise manual dot-key set."""
    if not overrides:
        return base
    if _HAS_OMEGA:
        try:
            ov = OmegaConf.from_dotlist(overrides)  # type: ignore
            merged = OmegaConf.merge(OmegaConf.create(base), ov)  # type: ignore
            result = OmegaConf.to_container(merged, resolve=True)  # type: ignore
            if not isinstance(result, dict):
                return base
            return result
        except Exception as e:
            raise SpectraMindError(f"Failed to apply overrides: {overrides} :: {e}")
    # manual fallback
    result = dict(base)
    for item in overrides:
        if "=" not in item:
            _warn(f"Ignoring malformed override (expected key=value): {item}")
            continue
        key, raw = item.split("=", 1)
        _set_in(result, key.strip(), _coerce_value(raw))
    return result

def _hash_config_dict(d: Dict[str, Any]) -> str:
    """Stable sha256 over a canonicalized JSON representation of a dict."""
    enc = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(enc).hexdigest()

# ======================================================================================
# JSONL event logging (offline telemetry)
# ======================================================================================

@dataclass
class Event:
    t: float
    kind: str
    msg: str
    extra: Dict[str, Any]

def _write_event(path: Path, kind: str, msg: str, **extra: Any) -> None:
    ensure_dir(path)
    rec = Event(t=time.time(), kind=kind, msg=msg, extra=extra)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), sort_keys=False) + "\n")

# ======================================================================================
# DVC
# ======================================================================================

def _run_dvc(
    stage: str,
    cwd: Optional[Path] = None,
    extra: Optional[List[str]] = None,
    dry_run: bool = False,
    print_cmd: bool = False,
) -> None:
    cmd = ["dvc", "repro", "-f", "-s", stage]
    if extra:
        cmd += extra
    if print_cmd or dry_run:
        rprint(cmd) if _HAS_RICH else typer.echo(" ".join(cmd))
    if dry_run:
        return
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
    except FileNotFoundError:
        raise SpectraMindError("DVC is not installed or not on PATH.")
    except subprocess.CalledProcessError as e:
        raise SpectraMindError(f"DVC stage '{stage}' failed: {e}")

# ======================================================================================
# Global flags / callback
# ======================================================================================

@app.callback()
def _global(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging (INFO)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode (errors only)"),
    log_file: Optional[Path] = typer.Option(None, help="Optional log file to tee messages"),
    seed: Optional[int] = typer.Option(None, help="Deterministic seed for all stages"),
    events_path: Path = typer.Option(Path("artifacts/logs/events.jsonl"), help="JSONL event stream path"),
) -> None:
    """
    Global flags for logging, determinism, and event stream.
    """
    # Wire seeds early
    _set_seeds(seed, deterministic_torch=True)

    # Set logger level
    if quiet:
        os.environ["SPECTRAMIND_LOGLEVEL"] = "ERROR"
    elif verbose:
        os.environ["SPECTRAMIND_LOGLEVEL"] = "INFO"
    else:
        os.environ.setdefault("SPECTRAMIND_LOGLEVEL", "WARNING")

    # Tee to file if requested
    if log_file:
        try:
            ensure_dir(log_file)
            # simple tee: append startup banner
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] spectramind start pid={os.getpid()}\n")
        except Exception as e:
            _warn(f"Failed to open log_file: {e}")

    # Stash in context
    ctx.obj = {
        "events_path": events_path,
        "log_file": log_file,
        "seed": seed,
    }

# ======================================================================================
# sys: version / doctor / env / cfg-tools
# ======================================================================================

def _read_version() -> str:
    vf = Path("VERSION")
    if vf.exists():
        try:
            return vf.read_text(encoding="utf-8").strip()
        except Exception:
            return "unknown"
    return "unknown"

@sys_app.command("version")
def sys_version() -> None:
    """Print CLI, package, and Python versions."""
    info = {
        "spectramind_cli": "v50",
        "package_version": _read_version(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    rprint(info) if _HAS_RICH else typer.echo(json.dumps(info, indent=2))

@sys_app.command("env")
def sys_env() -> None:
    """Print environment hints (CI/Kaggle)."""
    info = {
        "ci": bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")),
        "kaggle": bool("KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists()),
        "cwd": str(Path.cwd()),
    }
    rprint(info) if _HAS_RICH else typer.echo(json.dumps(info, indent=2))

@sys_app.command("doctor")
def sys_doctor() -> None:
    """Run basic dependency checks (Python, YAML/JSON IO, OmegaConf, DVC)."""
    results = {}
    # IO
    try:
        _ = read_json  # noqa: F401
        results["io"] = True
    except Exception:
        results["io"] = False
    # OmegaConf
    results["omegaconf"] = _HAS_OMEGA
    # DVC
    try:
        subprocess.run(["dvc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        results["dvc"] = True
    except Exception:
        results["dvc"] = False
    rprint(results) if _HAS_RICH else typer.echo(json.dumps(results, indent=2))

@sys_app.command("print-config")
def sys_print_config(
    config: Optional[Path] = typer.Option(None, help="Config (.yaml/.json)"),
    set: List[str] = typer.Option([], "--set", "-s", help="Overrides: key=value (repeatable)"),
) -> None:
    """Print merged config after applying overrides (OmegaConf if available)."""
    cfg = _merge_overrides(_load_config_any(config), set)
    if _HAS_RICH:
        rprint(Panel.fit(json.dumps(cfg, indent=2)))
    else:
        typer.echo(json.dumps(cfg, indent=2))

@sys_app.command("hash-config")
def sys_hash_config(
    config: Optional[Path] = typer.Option(None, help="Config (.yaml/.json)"),
    set: List[str] = typer.Option([], "--set", "-s", help="Overrides: key=value (repeatable)"),
) -> None:
    """Emit a stable sha256 for the merged config (reproducibility aid)."""
    cfg = _merge_overrides(_load_config_any(config), set)
    h = _hash_config_dict(cfg)
    typer.echo(h)

@sys_app.command("seed")
def sys_seed(
    value: int = typer.Option(..., "--value", "-v", help="Seed value to set globally"),
    deterministic_torch: bool = typer.Option(
        True,
        "--deterministic-torch/--no-deterministic-torch",
        help="Configure PyTorch for deterministic ops (cudnn.deterministic=True, benchmark=False).",
    ),
) -> None:
    """
    Set a global random seed across libraries used in this process.

    Examples
    --------
    spectramind sys seed -v 42
    spectramind sys seed --value 2025 --no-deterministic-torch
    """
    try:
        _set_seeds(value, deterministic_torch=deterministic_torch)
        _ok(f"Global seed set to {value} "
            f"({'deterministic torch' if deterministic_torch else 'non-deterministic torch'})")
        # NOTE: environment changes here affect only this process and its children.
        # For persistence across shells, export PYTHONHASHSEED in your shell profile if desired.
    except Exception as e:
        _fail(f"Failed to set seed: {e}")

# ======================================================================================
# calibrate
# ======================================================================================

@calib_app.command("run")
def calibrate_run(
    ctx: typer.Context,
    raw_dir: Path = typer.Argument(..., exists=True, help="Directory with raw telescope inputs"),
    out_dir: Path = typer.Option(Path("data/interim/calibrated"), help="Output directory for calibrated cubes"),
    config: Optional[Path] = typer.Option(None, help="Optional config (.yaml/.json)"),
    set: List[str] = typer.Option([], "--set", "-s", help="Config overrides: key=value (repeatable)"),
    use_dvc: bool = typer.Option(False, help="Reproduce via DVC stage 'calibrate' if available"),
    dvc_print_cmd: bool = typer.Option(False, help="Print DVC command"),
    dry_run: bool = typer.Option(False, help="Don't execute, just show actions"),
    max_runtime_min: int = typer.Option(540, help="Runtime fence (minutes) for Kaggle (default 9h)"),
) -> None:
    """
    Run calibration (ADC → dark → flat → CDS → photometry → trace → phase).
    """
    t0 = time.time()
    events_path: Path = ctx.obj["events_path"]
    try:
        cfg = _merge_overrides(_load_config_any(config), set)
        cfg_hash = _hash_config_dict(cfg)
        _ensure_exists(raw_dir, "raw_dir")
        ensure_dir(out_dir)

        _write_event(events_path, "calibrate:start", "begin", cfg_hash=cfg_hash, raw=str(raw_dir), out=str(out_dir))

        if use_dvc:
            _run_dvc("calibrate", dry_run=dry_run, print_cmd=dvc_print_cmd)
        else:
            if dry_run:
                _ok("DRY-RUN: would call spectramind.pipeline.calibrate.run(...)")
            else:
                # Delegate to your real calibrator
                # from spectramind.pipeline.calibrate import run as calib_impl
                # calib_impl(raw_dir=raw_dir, out_dir=out_dir, cfg=cfg)
                _warn("Using placeholder calibrator — wire spectramind.pipeline.calibrate.run(...)")
                if not any(Path(raw_dir).iterdir()):
                    raise SpectraMindError("raw_dir is empty — no files to calibrate.")
                (Path(out_dir) / "_calibration_done.txt").write_text("ok\n", encoding="utf-8")

        elapsed = (time.time() - t0) / 60.0
        if elapsed > max_runtime_min and not dry_run:
            raise SpectraMindError(
                f"Calibration exceeded runtime fence: {elapsed:.1f} min > {max_runtime_min} min"
            )
        _write_event(events_path, "calibrate:end", "done", elapsed_min=elapsed)
        _ok(f"Calibration completed → {out_dir} (elapsed {elapsed:.1f} min)")
    except SpectraMindError as e:
        _write_event(events_path, "calibrate:error", str(e))
        _fail(str(e))

# ======================================================================================
# train
# ======================================================================================

@train_app.command("run")
def train_run(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, help="Training config (.yaml/.json)"),
    set: List[str] = typer.Option([], "--set", "-s", help="Config overrides: key=value (repeatable)"),
    use_dvc: bool = typer.Option(False, help="Reproduce via DVC stage 'train' if available"),
    dvc_print_cmd: bool = typer.Option(False, help="Print DVC command"),
    dry_run: bool = typer.Option(False, help="Don't execute, just show actions"),
) -> None:
    """
    Train the dual-encoder model (FGS1 encoder + AIRS encoder + heteroscedastic decoder).
    """
    events_path: Path = ctx.obj["events_path"]
    try:
        if use_dvc:
            _run_dvc("train", dry_run=dry_run, print_cmd=dvc_print_cmd)
            if not dry_run:
                _ok("Training (DVC) completed")
            return

        cfg = _merge_overrides(_load_config_any(config), set)
        cfg_hash = _hash_config_dict(cfg)
        _write_event(events_path, "train:start", "begin", cfg_hash=cfg_hash)

        if dry_run:
            _ok("DRY-RUN: would call train_from_config(cfg)")
            return

        ckpt_path, metrics = train_from_config(cfg)
        # Pretty summary
        logger.info("===== Training Summary =====")
        if ckpt_path:
            logger.info("Best/Last checkpoint: %s", ckpt_path)
        else:
            logger.info("Checkpoint: (not available)")
        if metrics:
            pretty = json.dumps({k: float(v) for k, v in metrics.items()}, indent=2, sort_keys=True)
            logger.info("Validation metrics:\n%s", pretty)
        else:
            logger.info("Validation metrics: (none reported)")
        logger.info("============================")

        _write_event(events_path, "train:end", "done", ckpt=str(ckpt_path) if ckpt_path else "", metrics=metrics or {})
        _ok("Training completed")
    except SpectraMindError as e:
        _write_event(events_path, "train:error", str(e))
        _fail(str(e))
    except Exception as e:
        _write_event(events_path, "train:error", f"unhandled: {e}")
        _fail(f"Training failed: {e}")

# ======================================================================================
# predict
# ======================================================================================

@predict_app.command("run")
def predict_run(
    ctx: typer.Context,
    ckpt: Path = typer.Argument(..., exists=True, help="Checkpoint to load"),
    data_dir: Path = typer.Option(Path("data/processed/tensors_eval"), help="Eval tensors directory"),
    out_csv: Path = typer.Option(Path("artifacts/predictions/preds.csv"), help="Output predictions CSV"),
    config: Optional[Path] = typer.Option(None, help="Optional inference config"),
    set: List[str] = typer.Option([], "--set", "-s", help="Config overrides: key=value (repeatable)"),
    use_dvc: bool = typer.Option(False, help="Reproduce via DVC stage 'predict' if available"),
    dvc_print_cmd: bool = typer.Option(False, help="Print DVC command"),
    dry_run: bool = typer.Option(False, help="Don't execute, just show actions"),
) -> None:
    """
    Predict spectral μ/σ per bin (283 bins per id) for submission.
    """
    events_path: Path = ctx.obj["events_path"]
    try:
        _ensure_exists(ckpt, "ckpt")
        ensure_dir(out_csv)

        if use_dvc:
            _run_dvc("predict", dry_run=dry_run, print_cmd=dvc_print_cmd)
            if not dry_run:
                _ok("Prediction (DVC) completed")
            return

        cfg = _merge_overrides(_load_config_any(config), set)
        cfg_hash = _hash_config_dict(cfg)
        _write_event(events_path, "predict:start", "begin", cfg_hash=cfg_hash, ckpt=str(ckpt))

        if dry_run:
            _ok(f"DRY-RUN: would run inference and write → {out_csv}")
            return

        # Delegate to your inference
        # from spectramind.inference.predict import run as impl
        # impl(ckpt=ckpt, data_dir=data_dir, out_csv=out_csv, cfg=cfg)
        _warn("Using placeholder predictor — wire spectramind.inference.predict.run(...)")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("id,bin,mu,sigma\n", encoding="utf-8")

        _write_event(events_path, "predict:end", "done", out_csv=str(out_csv))
        _ok(f"Predictions written → {out_csv}")
    except SpectraMindError as e:
        _write_event(events_path, "predict:error", str(e))
        _fail(str(e))
    except Exception as e:
        _write_event(events_path, "predict:error", f"unhandled: {e}")
        _fail(f"Prediction failed: {e}")

# ======================================================================================
# diagnose
# ======================================================================================

@diagnose_app.command("report")
def diagnose_report(
    pred: Path = typer.Argument(..., exists=True, help="Predictions CSV: id,bin,mu,sigma"),
    out: Path = typer.Option(Path("artifacts/reports/diagnostics_dashboard.html"), help="Output HTML"),
    targets: Optional[Path] = typer.Option(None, help="Optional targets CSV: id,bin,target"),
    events: Optional[Path] = typer.Option(None, help="Optional JSONL events"),
    schema: Optional[Path] = typer.Option(None, help="Optional submission schema (.json)"),
    base: Optional[Path] = typer.Option(None, help="Optional baseline preds CSV for Inject-&-Recover"),
    title: str = typer.Option("SpectraMind V50 — Diagnostics Dashboard", help="Report title"),
) -> None:
    """
    Build an offline HTML diagnostics dashboard with calibration & residuals plots.
    """
    try:
        from spectramind.reports import generate_report  # local import to keep CLI slim
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
        _fail(f"diagnose report failed: {e}")

# ======================================================================================
# submit
# ======================================================================================

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

@submit_app.command("package")
def submit_package(
    preds: Path = typer.Argument(..., exists=True, help="Predictions CSV to package"),
    out_zip: Path = typer.Option(Path("dist/submission.zip"), help="Output ZIP path"),
    schema: Optional[Path] = typer.Option(None, help="Optional submission schema to validate"),
    name: str = typer.Option("SpectraMind V50 Submission", help="Package name / label"),
    extra_file: List[Path] = typer.Option([], help="Additional files to include in the archive"),
) -> None:
    """
    Package predictions (and extras) into a submission ZIP, with optional JSON schema validation.
    """
    try:
        _ensure_exists(preds, "preds")
        if schema:
            _ensure_exists(schema, "schema")
            # Optional validation via reports helpers (best-effort)
            try:
                from spectramind.reports import Predictions, _validate_submission_schema, _read_csv
                df = _read_csv(preds)
                pred_obj = Predictions(df)
                pred_obj.validate()
                msg = _validate_submission_schema(pred_obj, schema)
                if msg:
                    if "failed" in msg.lower():
                        _fail(msg)
                    else:
                        _warn(msg)
            except Exception as e:
                _warn(f"Schema validation skipped: {e}")

        ensure_dir(out_zip)
        import zipfile
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Required predictions
            zf.write(preds, arcname=preds.name)

            # MANIFEST
            manifest = {
                "name": name,
                "generated_by": "spectramind submit package",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "package_version": _read_version(),
                "predictions_sha256": _sha256_file(preds),
            }
            zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

            # Extras
            for ef in extra_file:
                if Path(ef).exists():
                    zf.write(ef, arcname=Path(ef).name)
                else:
                    _warn(f"Extra file not found: {ef}")

        _ok(f"Submission packaged → {out_zip}")
    except SpectraMindError as e:
        _fail(str(e))
    except Exception as e:
        _fail(f"Packaging failed: {e}")

# ======================================================================================
# entrypoint
# ======================================================================================

def main() -> None:
    app()

if __name__ == "__main__":
    main()
