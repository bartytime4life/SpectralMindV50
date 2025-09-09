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
from typing import Any, Dict, List, Optional

import typer

# Optional niceties
try:
    from rich import print as rprint
    from rich.traceback import install as rich_install
    from rich.panel import Panel
    _HAS_RICH = True
    rich_install(show_locals=False, suppress=["typer", "click"])
except Exception:
    _HAS_RICH = False

    def rprint(*args, **kwargs):  # fallback
        print(*args, **kwargs)

# Optional OmegaConf (Hydra-like) support
try:
    from omegaconf import OmegaConf  # type: ignore
    _HAS_OMEGA = True
except Exception:
    _HAS_OMEGA = False

# Optional scientific libs for deterministic seeds
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

# Public logger from package (matches upgraded __init__.py)
from spectramind import get_logger
from spectramind.utils.io import p, read_yaml, read_json, ensure_dir
from spectramind.train.trainer import train_from_config
from spectramind.submit.validate import validate_csv as _validate_submission_csv
from spectramind.utils.manifest import write_run_manifest

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
preproc_app = typer.Typer(help="Preprocess (calibrated → model-ready tensors)")
train_app = typer.Typer(help="Model training")
predict_app = typer.Typer(help="Prediction / inference")
diagnose_app = typer.Typer(help="Diagnostics & reporting")
submit_app = typer.Typer(help="Submission packaging / validation")
sys_app = typer.Typer(help="System utilities (doctor, version, env, cfg-tools)")

app.add_typer(calib_app, name="calibrate")
app.add_typer(preproc_app, name="preprocess")
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
    return bool(
        os.environ.get("CI")
        or os.environ.get("GITHUB_ACTIONS")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or Path("/kaggle").exists()
    )


def _ok(msg: str) -> None:
    rprint(f"[bold green]✓[/bold green] {msg}") if _HAS_RICH else typer.echo(msg)


def _warn(msg: str) -> None:
    rprint(f"[yellow]warn:[/yellow] {msg}") if _HAS_RICH else typer.secho(f"warn: {msg}", fg=typer.colors.YELLOW)


def _fail(msg: str, code: int = 1) -> None:
    rprint(f"[bold red]error:[/bold red] {msg}") if _HAS_RICH else typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _ensure_exists(pth: Optional[Path], label: str) -> None:
    if pth is not None and not Path(pth).exists():
        raise SpectraMindError(f"{label} not found: {pth}")


def _ensure_file_parent(path: Path) -> None:
    """Make sure parent directory exists for a file path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

# ======================================================================================
# Determinism / seeding
# ======================================================================================

def _set_seeds(seed: Optional[int], *, deterministic_torch: bool = True) -> None:
    """
    Set global RNG seeds for Python, NumPy, and (optionally) PyTorch.
    """
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

    # keep BLAS noise down in CI unless user overrides
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    random.seed(seed)

    if _HAS_NUMPY:
        _np.random.seed(seed)  # type: ignore[attr-defined]

    if _HAS_TORCH:
        torch.manual_seed(seed)  # type: ignore[name-defined]
        if torch.cuda.is_available():  # type: ignore[name-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")  # type: ignore[attr-defined]

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
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        try:
            return json.loads(v)
        except Exception:
            pass
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
            return result if isinstance(result, dict) else base
        except Exception as e:
            raise SpectraMindError(f"Failed to apply overrides: {overrides} :: {e}")
    result = dict(base)
    for item in overrides:
        if "=" not in item:
            _warn(f"Ignoring malformed override (expected key=value): {item}")
            continue
        key, raw = item.split("=", 1)
        _set_in(result, key.strip(), _coerce_value(raw))
    return result


def _hash_config_dict(d: Dict[str, Any]) -> str:
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
    """Append a JSONL event record; creates parent dirs if needed."""
    _ensure_file_parent(path)
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
    cmd = ["dvc", "repro", "--single-item", stage]
    if _is_ci_or_kaggle():
        cmd.insert(2, "--no-tty")
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
    manifest_path: Path = typer.Option(Path("artifacts/run_manifest.jsonl"), help="Append run manifests here"),
) -> None:
    """
    Global flags for logging, determinism, event stream, and run manifest.
    """
    # Wire seeds early
    _set_seeds(seed, deterministic_torch=True)

    # Logging level
    if quiet:
        os.environ["SPECTRAMIND_LOGLEVEL"] = "ERROR"
    elif verbose:
        os.environ["SPECTRAMIND_LOGLEVEL"] = "INFO"
    else:
        os.environ.setdefault("SPECTRAMIND_LOGLEVEL", "WARNING")

    # Tee to file if requested
    if log_file:
        try:
            _ensure_file_parent(log_file)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] spectramind start pid={os.getpid()}\n")
        except Exception as e:
            _warn(f"Failed to open log_file: {e}")

    # Initial manifest line (append JSONL)
    try:
        write_run_manifest(manifest_path, extra={"stage": "cli-entry", "pid": os.getpid(), "seeds": seed})
    except Exception as e:
        _warn(f"manifest emit failed: {e}")

    # Stash in context
    ctx.obj = {
        "events_path": events_path,
        "log_file": log_file,
        "seed": seed,
        "manifest_path": manifest_path,
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
        help="Configure PyTorch for deterministic ops.",
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
        _ok(
            f"Global seed set to {value} "
            f"({'deterministic torch' if deterministic_torch else 'non-deterministic torch'})"
        )
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
    manifest_path: Path = ctx.obj["manifest_path"]
    try:
        cfg = _merge_overrides(_load_config_any(config), set)
        cfg_hash = _hash_config_dict(cfg)
        _ensure_exists(raw_dir, "raw_dir")
        ensure_dir(out_dir)
        write_run_manifest(manifest_path, extra={"stage": "calibrate", "cfg_hash": cfg_hash, "raw": str(raw_dir)})

        _write_event(events_path, "calibrate:start", "begin", cfg_hash=cfg_hash, raw=str(raw_dir), out=str(out_dir))

        if use_dvc:
            _run_dvc("calibrate", dry_run=dry_run, print_cmd=dvc_print_cmd)
        else:
            if dry_run:
                _ok("DRY-RUN: would call spectramind.pipeline.calibrate.run(...)")
            else:
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
# preprocess
# ======================================================================================

@preproc_app.command("run")
def preprocess_run(
    ctx: typer.Context,
    calib_dir: Path = typer.Argument(..., exists=True, help="Directory with calibrated data (from 'calibrate')"),
    out_dir: Path = typer.Option(Path("data/processed/tensors_train"), help="Output directory for model tensors"),
    config: Optional[Path] = typer.Option(None, help="Optional preprocessing config (.yaml/.json)"),
    set: List[str] = typer.Option([], "--set", "-s", help="Config overrides: key=value (repeatable)"),
    use_dvc: bool = typer.Option(False, help="Reproduce via DVC stage 'preprocess' if available"),
    dvc_print_cmd: bool = typer.Option(False, help="Print DVC command"),
    dry_run: bool = typer.Option(False, help="Don't execute, just show actions"),
) -> None:
    """
    Run preprocessing / feature extraction (calibrated → model-ready tensors).
    """
    events_path: Path = ctx.obj["events_path"]
    manifest_path: Path = ctx.obj["manifest_path"]
    try:
        cfg = _merge_overrides(_load_config_any(config), set)
        cfg_hash = _hash_config_dict(cfg)
        _ensure_exists(calib_dir, "calib_dir")
        ensure_dir(out_dir)
        write_run_manifest(manifest_path, extra={"stage": "preprocess", "cfg_hash": cfg_hash, "calib": str(calib_dir)})

        _write_event(events_path, "preprocess:start", "begin", cfg_hash=cfg_hash, calib=str(calib_dir), out=str(out_dir))

        if use_dvc:
            _run_dvc("preprocess", dry_run=dry_run, print_cmd=dvc_print_cmd)
        else:
            if dry_run:
                _ok("DRY-RUN: would call spectramind.pipeline.preprocess.run(...)")
            else:
                # from spectramind.pipeline.preprocess import run as preproc_impl
                # preproc_impl(calib_dir=calib_dir, out_dir=out_dir, cfg=cfg)
                _warn("Using placeholder preprocessor — wire spectramind.pipeline.preprocess.run(...)")
                (Path(out_dir) / "_preprocess_done.txt").write_text("ok\n", encoding="utf-8")

        _write_event(events_path, "preprocess:end", "done", out=str(out_dir))
        _ok(f"Preprocess completed → {out_dir}")
    except SpectraMindError as e:
        _write_event(events_path, "preprocess:error", str(e))
        _fail(str(e))
    except Exception as e:
        _write_event(events_path, "preprocess:error", f"unhandled: {e}")
        _fail(f"Preprocess failed: {e}")

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
    manifest_path: Path = ctx.obj["manifest_path"]
    try:
        if use_dvc:
            _run_dvc("train", dry_run=dry_run, print_cmd=dvc_print_cmd)
            if not dry_run:
                _ok("Training (DVC) completed")
            return

        cfg = _merge_overrides(_load_config_any(config), set)
        cfg_hash = _hash_config_dict(cfg)
        write_run_manifest(manifest_path, extra={"stage": "train", "cfg_hash": cfg_hash})

        _write_event(events_path, "train:start", "begin", cfg_hash=cfg_hash)

        if dry_run:
            _ok("DRY-RUN: would call train_from_config(cfg)")
            return

        ckpt_path, metrics = train_from_config(cfg)

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
    manifest_path: Path = ctx.obj["manifest_path"]
    try:
        _ensure_exists(ckpt, "ckpt")
        _ensure_file_parent(out_csv)

        if use_dvc:
            _run_dvc("predict", dry_run=dry_run, print_cmd=dvc_print_cmd)
            if not dry_run:
                _ok("Prediction (DVC) completed")
            return

        cfg = _merge_overrides(_load_config_any(config), set)
        cfg_hash = _hash_config_dict(cfg)
        write_run_manifest(manifest_path, extra={"stage": "predict", "cfg_hash": cfg_hash, "ckpt": str(ckpt)})

        _write_event(events_path, "predict:start", "begin", cfg_hash=cfg_hash, ckpt=str(ckpt))

        if dry_run:
            _ok(f"DRY-RUN: would run inference and write → {out_csv}")
            return

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

@diagnose_app.command("run")
def diagnose_run(
    ctx: typer.Context,
    preds: Path = typer.Argument(..., exists=True, help="Predictions CSV: id,bin,mu,sigma"),
    out_dir: Path = typer.Option(Path("artifacts/diagnostics"), help="Diagnostics output directory"),
    truth: Optional[Path] = typer.Option(None, help="Optional ground-truth CSV: id,bin,target"),
    use_dvc: bool = typer.Option(False, help="Reproduce via DVC stage 'diagnose' if available"),
    dvc_print_cmd: bool = typer.Option(False, help="Print DVC command"),
    dry_run: bool = typer.Option(False, help="Don't execute, just show actions"),
) -> None:
    """
    Run metrics & sanity checks (GLL, residuals, coverage, smoothness) and emit JSON summary.
    """
    events_path: Path = ctx.obj["events_path"]
    manifest_path: Path = ctx.obj["manifest_path"]
    try:
        _ensure_exists(preds, "preds")
        ensure_dir(out_dir)
        write_run_manifest(manifest_path, extra={"stage": "diagnose", "preds": str(preds)})

        if use_dvc:
            _run_dvc("diagnose", dry_run=dry_run, print_cmd=dvc_print_cmd)
            if not dry_run:
                _ok("Diagnostics (DVC) completed")
            return

        if dry_run:
            _ok(f"DRY-RUN: would call spectramind.diagnostics.run_diagnostics(...) → {out_dir}")
            return

        try:
            from spectramind.diagnostics import run_diagnostics
        except Exception as e:
            _fail(f"diagnostics module not available: {e}")

        summary = run_diagnostics(preds_path=preds, truth_path=truth, out_dir=out_dir, report_name="report.html")
        _write_event(events_path, "diagnose:end", "done", out=str(out_dir), summary=summary)
        _ok(f"Diagnostics summary written → {out_dir / 'summary.json'}")
    except SpectraMindError as e:
        _write_event(events_path, "diagnose:error", str(e))
        _fail(str(e))
    except Exception as e:
        _write_event(events_path, "diagnose:error", f"unhandled: {e}")
        _fail(f"Diagnostics failed: {e}")


@diagnose_app.command("report")
def diagnose_report(
    preds: Path = typer.Argument(..., exists=True, help="Predictions table (CSV/Parquet/JSON) with mu_*** and sigma_***"),
    out: Path = typer.Option(Path("artifacts/reports/diagnostics_report.html"), help="Output HTML/MD report path"),
    cfg: Optional[Path] = typer.Option(None, "--config", help="Optional Hydra-composed config YAML to embed"),
    metrics: Optional[Path] = typer.Option(None, "--metrics", help="Optional JSON metrics file to embed"),
    history: Optional[Path] = typer.Option(None, "--history", help="Optional training history table (CSV/Parquet/JSON)"),
    digests: List[Path] = typer.Option([], "--digest", help="Extra files/dirs to hash & include in audit (repeatable)"),
    title: str = typer.Option("SpectraMind V50 — Diagnostics Report", help="Report title override"),
) -> None:
    """
    Build an offline HTML diagnostics report (figures, tables, audit digests) using
    spectramind.diagnostics.reports.generate_report (Jinja2 with Markdown fallback).
    """
    try:
        from spectramind.diagnostics.reports import generate_report  # matches upgraded reports.py
    except Exception as e:
        _fail(f"diagnostics.reports module not available: {e}")

    try:
        _ensure_file_parent(out)

        # Use dirname of output as artifacts_dir; the generator writes report + manifest there.
        artifacts_dir = out.parent
        run_id = f"diag-{int(time.time())}"

        # generate_report writes either HTML (preferred) or Markdown fallback based on deps
        report_path = generate_report(
            run_id=run_id,
            artifacts_dir=artifacts_dir,
            config_path=cfg,
            metrics_json=metrics,
            history_csv=history,
            predictions_csv=preds,
            extra_digest_paths=list(digests),
            notes=title,
            templates_dir=None,
            filename=out.name,  # keep the passed filename (html or .md)
        )
        _ok(f"Report written → {report_path}")
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
    Package predictions (and extras) into a submission ZIP, with optional schema validation.
    """
    try:
        _ensure_exists(preds, "preds")
        _ensure_file_parent(out_zip)

        # Always run our validator first
        res = _validate_submission_csv(csv_path=preds, n_bins=283, strict_ids=True, chunksize=None)
        if not res.ok:
            _warn(f"Validation failed: {len(res.errors)} errors (showing first 10):")
            for e in res.errors[:10]:
                typer.echo(f"- {e}")
            _fail("Submission CSV is invalid.")

        # If a JSON schema is provided and there's a schema validator available, run it too.
        if schema:
            try:
                from spectramind.submit.validate import validate_against_schema as _schema_check  # type: ignore
                sch_ok, sch_msg = _schema_check(preds, schema)
                if not sch_ok:
                    _fail(f"Schema validation failed: {sch_msg}")
                elif sch_msg:
                    _warn(sch_msg)
            except Exception as e:
                _warn(f"Schema validation skipped: {e}")

        import zipfile
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(preds, arcname=preds.name)
            manifest = {
                "name": name,
                "generated_by": "spectramind submit package",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "package_version": _read_version(),
                "predictions_sha256": _sha256_file(preds),
            }
            zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
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


@submit_app.command("validate")
def submit_validate(
    csv: Path = typer.Argument(..., exists=True, help="Predictions CSV to validate"),
    n_bins: int = typer.Option(283, "--n-bins", help="Expected number of spectral bins per id"),
    strict_ids: bool = typer.Option(True, "--strict-ids/--no-strict-ids", help="Enforce non-empty and unique ids"),
    chunksize: Optional[int] = typer.Option(None, help="Validate in chunks (rows per chunk) to reduce memory"),
    show: int = typer.Option(20, help="Show first N errors"),
    json_out: Optional[Path] = typer.Option(None, help="Write full validation report (JSON) to this path"),
) -> None:
    """
    Validate a submission CSV against schema + SpectraMind semantic checks.

    Supports:
      1) Narrow: columns ['id', 'mu', 'sigma'] where mu/sigma are JSON arrays.
      2) Wide:   columns ['id'] + mu_000..mu_{n_bins-1} + sigma_000..sigma_{n_bins-1}
    """
    try:
        res = _validate_submission_csv(
            csv_path=csv,
            n_bins=n_bins,
            strict_ids=strict_ids,
            chunksize=chunksize,
        )
        report = {
            "ok": res.ok,
            "n_rows": res.n_rows,
            "n_valid": res.n_valid,
            "n_errors": len(res.errors),
        }

        if json_out is not None:
            _ensure_file_parent(json_out)
            json_out.write_text(json.dumps({**report, "errors": res.errors}, indent=2), encoding="utf-8")

        if res.ok:
            _ok(f"Validation OK: {res.n_valid}/{res.n_rows} rows valid.")
        else:
            _warn(
                f"Validation failed: {len(res.errors)} errors "
                f"(showing first {min(show, len(res.errors))}; use --show to adjust)"
            )
            for e in res.errors[:show]:
                typer.echo(f"- {e}")
            _fail("Submission CSV is invalid.")
    except FileNotFoundError as e:
        _fail(str(e))
    except Exception as e:
        _fail(f"Validation crashed: {e}")

# ======================================================================================
# entrypoint
# ======================================================================================

def main() -> None:
    app()


if __name__ == "__main__":
    main()
