# src/spectramind/predict/cli.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer

from .core import PredictConfig, predict_to_submission

# -----------------------------------------------------------------------------
# CLI setup
# -----------------------------------------------------------------------------
HELP = """SpectraMind V50 — Prediction CLI

Run inference on calibrated inputs and produce a Kaggle-ready submission bundle
(CSV + manifest, optional ZIP), with reproducible, deterministic settings.

Examples:
  spectramind-predict run --inputs ./inputs --out ./outputs/predict
  spectramind-predict run --inputs ./inputs --ckpt ckpt1.pth --ckpt ckpt2.pth
  spectramind-predict run --inputs ./inputs --jit model.pt --device cpu --precision fp32
  spectramind-predict run --inputs ./inputs --meta extras.json --no-zip
"""

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=HELP,
    rich_markup_mode="markdown",
)


# -----------------------------------------------------------------------------
# Utility helpers (stdlib only; no heavy imports here)
# -----------------------------------------------------------------------------
def _fail(msg: str, code: int = 1) -> None:
    """Print a friendly error and exit."""
    typer.secho(f"✖ {msg}", fg=typer.colors.RED)
    raise typer.Exit(code)


def _ok(msg: str) -> None:
    typer.secho(f"✔ {msg}", fg=typer.colors.GREEN)


def _warn(msg: str) -> None:
    typer.secho(f"⚠ {msg}", fg=typer.colors.YELLOW)


def _require_file(path: Path, desc: str) -> None:
    if not path.exists() or not path.is_file():
        _fail(f"Missing {desc}: {path}")


def _require_dir(path: Path, desc: str, create: bool = False) -> None:
    if path.exists() and not path.is_dir():
        _fail(f"Expected directory for {desc}, found file: {path}")
    if not path.exists():
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                _fail(f"Unable to create {desc} at {path}: {e}")
        else:
            _fail(f"Missing {desc} directory: {path}")


def _preflight_inputs(
    inputs_dir: Path,
    ids_csv: str,
    fgs1_name: str,
    airs_name: str,
) -> None:
    """Validate required inputs exist inside inputs_dir."""
    _require_dir(inputs_dir, "inputs directory", create=False)
    ids = inputs_dir / ids_csv
    fgs = inputs_dir / fgs1_name
    ars = inputs_dir / airs_name

    _require_file(ids, "ids.csv")
    _require_file(fgs, "FGS1 .npy")
    _require_file(ars, "AIRS .npy")

    # quick sanity on ids.csv presence of header (do not parse CSV fully)
    try:
        head = ids.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        if head and "sample_id" not in head[0]:
            _warn(f"ids file {ids} first line does not contain 'sample_id' (continuing)")
    except Exception as e:
        _warn(f"Could not read ids file {ids}: {e}")


def _load_meta(meta_path: Optional[Path]) -> dict:
    if not meta_path:
        return {}
    _require_file(meta_path, "metadata JSON")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        _fail(f"Failed to parse --meta JSON at {meta_path}: {e}")
    return {}


def _normalize_ckpts(ckpt: Optional[List[Path]]) -> list[str]:
    if not ckpt:
        return []
    bad = [p for p in ckpt if not p.exists()]
    if bad:
        _fail(f"Checkpoint(s) not found: {', '.join(str(b) for b in bad)}")
    return [str(p) for p in ckpt]


def _resolve_out(out_dir: Path) -> Path:
    _require_dir(out_dir, "output directory", create=True)
    return out_dir


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
@app.command("run")
def run(
    inputs: Path = typer.Option(..., exists=True, file_okay=False, help="Directory with ids.csv, fgs1.npy, airs.npy"),
    out: Path = typer.Option("outputs/predict", help="Destination directory for submission artifacts"),
    # files
    ids_csv: str = typer.Option("ids.csv", help="CSV with 'sample_id' column"),
    fgs1: str = typer.Option("fgs1.npy", help="FGS1 NPY (shape: N, ...)"),
    airs: str = typer.Option("airs.npy", help="AIRS NPY (shape: N, ...)"),
    # model
    model_class: Optional[str] = typer.Option(None, help="Import path to model class, e.g. spectramind.models.v50.Model"),
    ckpt: Optional[List[Path]] = typer.Option(None, help="One or more .ckpt/.pth files (ensembled by averaging)"),
    jit: Optional[Path] = typer.Option(None, help="TorchScript model (.pt) alternative to model_class+ckpt"),
    model_init_json: Optional[Path] = typer.Option(None, help="JSON with kwargs to instantiate model_class"),
    # inference
    device: str = typer.Option("cuda", help="Device: cuda|cpu|mps"),
    precision: str = typer.Option("fp16", help="Precision: fp32|fp16|bf16"),
    batch_size: int = typer.Option(16, min=1),
    workers: int = typer.Option(0, min=0, help="DataLoader num_workers"),
    pin_memory: bool = typer.Option(True, help="Pin host memory for faster GPU transfer (ignored on CPU)"),
    n_bins: int = typer.Option(283, min=1, help="Number of spectral bins"),
    seed: int = typer.Option(42),
    cudnn_benchmark: bool = typer.Option(False),
    # package
    csv_name: str = typer.Option("submission.csv"),
    zip_name: str = typer.Option("submission.zip"),
    zip: bool = typer.Option(True, "--zip/--no-zip", help="Create submission.zip alongside CSV"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Run schema/physics validation before packaging"),
    meta: Optional[Path] = typer.Option(None, help="Extra JSON metadata to embed in manifest"),
    # diagnostics
    dry_run: bool = typer.Option(False, help="Print resolved config and exit"),
) -> None:
    """Run inference → CSV (+ manifest) → optional ZIP, ready for Kaggle upload."""
    # Preflight checks
    _preflight_inputs(inputs, ids_csv=ids_csv, fgs1_name=fgs1, airs_name=airs)
    _resolve_out(out)
    report_meta = _load_meta(meta)
    ckpt_paths = _normalize_ckpts(ckpt)

    if (jit and (ckpt_paths or model_class)) or (ckpt_paths and jit):
        _warn("Both JIT and ckpt/model_class provided; JIT will take precedence.")
    if not jit and not ckpt_paths and not model_class:
        _warn("No model specified. Expecting either --jit or --ckpt/--model-class; attempting to proceed if defaults exist.")

    cfg = PredictConfig(
        inputs_dir=inputs,
        ids_csv=ids_csv,
        fgs1_path=fgs1,
        airs_path=airs,
        out_dir=out,
        out_csv_name=csv_name,
        out_zip_name=zip_name,
        model_class=model_class,
        ckpt_paths=ckpt_paths,
        jit_path=str(jit) if jit else None,
        model_init_json=str(model_init_json) if model_init_json else None,
        device=device,
        precision=precision,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        n_bins=n_bins,
        seed=seed,
        cudnn_benchmark=cudnn_benchmark,
        validate=validate,
        make_zip=zip,
        report_meta=report_meta or None,
    )

    if dry_run:
        typer.echo(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}, indent=2))
        _ok("Dry run completed (config only).")
        raise typer.Exit(code=0)

    try:
        out_path = predict_to_submission(cfg)
    except Exception as e:
        _fail(f"Prediction failed: {e}")

    _ok(f"submission written → {out_path}")


@app.command("examples")
def examples() -> None:
    """Show useful invocation examples."""
    typer.echo(
        "\n".join(
            [
                "Examples:",
                "  spectramind-predict run --inputs ./inputs --out ./outputs/predict",
                "  spectramind-predict run --inputs ./inputs --ckpt ckpt1.pth --ckpt ckpt2.pth",
                "  spectramind-predict run --inputs ./inputs --jit model.pt --device cpu --precision fp32",
                "  spectramind-predict run --inputs ./inputs --meta extras.json --no-zip",
                "  spectramind-predict run --inputs ./inputs --dry-run   # print resolved config only",
            ]
        )
    )


@app.command("version")
def version() -> None:
    """Print CLI version (if packaged) or fallback."""
    try:
        # Prefer package metadata if installed
        from importlib.metadata import version as _pkg_version  # type: ignore

        typer.echo(_pkg_version("spectramind"))  # package name if published
    except Exception:
        # Fallback: read repo VERSION file if present
        repo_root = Path(__file__).resolve().parents[3]  # .../src/spectramind/predict/cli.py → repo root
        ver_file = repo_root / "VERSION"
        if ver_file.exists():
            typer.echo(ver_file.read_text(encoding="utf-8").strip())
        else:
            typer.echo("0.0.0-dev")


if __name__ == "__main__":  # pragma: no cover
    app()
