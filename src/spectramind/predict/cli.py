# src/spectramind/predict/cli.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .core import PredictConfig, predict_to_submission

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SpectraMind V50 — Prediction CLI")


@app.command("run")
def run(
    inputs: Path = typer.Option(..., exists=True, file_okay=False, help="Directory with ids.csv, fgs1.npy, airs.npy"),
    out: Path = typer.Option("outputs/predict", help="Destination directory for submission artifacts"),
    # files
    ids_csv: str = typer.Option("ids.csv", help="CSV with 'sample_id' column"),
    fgs1: str = typer.Option("fgs1.npy", help="FGS1 NPY (N, ...)"),
    airs: str = typer.Option("airs.npy", help="AIRS NPY (N, ...)"),
    # model
    model_class: Optional[str] = typer.Option(None, help="Import path to model class, e.g. spectramind.models.v50.Model"),
    ckpt: Optional[List[Path]] = typer.Option(None, help="One or more .ckpt/.pth files (ensembled by averaging)"),
    jit: Optional[Path] = typer.Option(None, help="TorchScript model (.pt) alternative to model_class+ckpt"),
    model_init_json: Optional[Path] = typer.Option(None, help="JSON with kwargs to instantiate model_class"),
    # inference
    device: str = typer.Option("cuda", help="cuda|cpu|mps"),
    precision: str = typer.Option("fp16", help="fp32|fp16|bf16"),
    batch_size: int = typer.Option(16, min=1),
    workers: int = typer.Option(0, min=0, help="DataLoader num_workers"),
    pin_memory: bool = typer.Option(True),
    n_bins: int = typer.Option(283, min=1),
    seed: int = typer.Option(42),
    cudnn_benchmark: bool = typer.Option(False),
    # package
    csv_name: str = typer.Option("submission.csv"),
    zip_name: str = typer.Option("submission.zip"),
    no_zip: bool = typer.Option(False, help="If set, do not create submission.zip"),
    no_validate: bool = typer.Option(False, help="If set, skip schema/physics validation before packaging"),
    meta: Optional[Path] = typer.Option(None, help="Extra JSON metadata to embed in manifest"),
):
    """Run inference → CSV (+ manifest) → optional ZIP, ready for Kaggle upload."""
    report_meta = {}
    if meta:
        report_meta = json.loads(Path(meta).read_text(encoding="utf-8"))

    cfg = PredictConfig(
        inputs_dir=inputs,
        ids_csv=ids_csv,
        fgs1_path=fgs1,
        airs_path=airs,
        out_dir=out,
        out_csv_name=csv_name,
        out_zip_name=zip_name,
        model_class=model_class,
        ckpt_paths=[str(p) for p in (ckpt or [])],
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
        validate=(not no_validate),
        make_zip=(not no_zip),
        report_meta=report_meta or None,
    )

    out_path = predict_to_submission(cfg)
    typer.secho(f"submission written → {out_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    app()
