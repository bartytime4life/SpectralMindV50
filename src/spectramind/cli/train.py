# src/spectramind/cli/train.py
from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import typer

from . import cli_app


# ──────────────────────────────────────────────────────────────────────────────
# CLI Options Dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TrainOptions:
    # Hydra composition
    config_name: str = "train"
    overrides: List[str] = field(default_factory=list)
    override_files: List[Path] = field(default_factory=list)

    # Common trainer knobs (the runner decides how to apply these)
    seed: Optional[int] = None
    devices: Optional[str] = None      # e.g. "cpu", "auto", "0", "0,1"
    precision: Optional[str] = None    # e.g. "32", "16-mixed", "bf16-mixed"
    epochs: Optional[int] = None
    resume_from: Optional[str] = None
    log_dir: Optional[str] = None

    # UX / safety
    dry_run: bool = False
    strict: bool = True                # validate unknown Hydra keys
    quiet: bool = False                # CI/Kaggle-friendly output

    # Introspection helpers
    print_config: bool = False         # let runner print composed config
    save_config: Optional[str] = None  # directory to dump composed config YAML


# ──────────────────────────────────────────────────────────────────────────────
# Internal Call
# ──────────────────────────────────────────────────────────────────────────────


def _dispatch_train(opts: TrainOptions) -> int:
    """
    Thin wrapper that calls the internal training entrypoint.

    All business logic MUST live outside the CLI (keeps this import cheap & testable).
    """
    try:
        # Canonical training entrypoint (keep stable for users & CI)
        from spectramind.pipeline.train import run as run_training  # type: ignore
    except Exception as e:
        typer.secho(
            f"[spectramind] Cannot import training entrypoint: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        return 2

    # Compose effective overrides:
    effective_overrides = list(opts.overrides)
    for of in opts.override_files:
        effective_overrides.extend(_load_override_file(of))

    payload = {
        "config_name": opts.config_name,
        "overrides": effective_overrides,
        "seed": opts.seed,
        "devices": opts.devices,
        "precision": opts.precision,
        "epochs": opts.epochs,
        "resume_from": opts.resume_from,
        "log_dir": opts.log_dir,
        "dry_run": opts.dry_run,
        "strict": opts.strict,
        "quiet": opts.quiet,
        # Introspection hints (runner is free to ignore if unsupported)
        "print_config": opts.print_config,
        "save_config": opts.save_config,
        # Environment hints the runner may honor
        "env": {
            "is_kaggle": _is_kaggle(),
            "is_ci": _is_ci(),
            "tty": sys.stderr.isatty(),
        },
    }

    try:
        run_training(**payload)
    except KeyboardInterrupt:
        typer.secho("[spectramind] Training interrupted by user.", fg=typer.colors.YELLOW)
        return 130
    except Exception as e:
        _print_train_error(e, payload, verbose=not opts.quiet)
        return 1

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _is_kaggle() -> bool:
    # Kaggle kernels define env vars & mount /kaggle/input
    return any(
        k in os.environ
        for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DOCKER_IMAGE")
    ) or os.path.exists("/kaggle/input")


def _is_ci() -> bool:
    # Common CI indicators
    return os.environ.get("CI", "").lower() == "true" or "GITHUB_ACTIONS" in os.environ


def _load_override_file(path: Path) -> list[str]:
    """
    Load Hydra override strings from a file.
    - Ignores blank lines and comments starting with '#'
    - Supports 'key=value' lines and group overrides like 'model=v50' or 'loss/smoothness@loss.s1=default'
    """
    out: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read override file '{path}': {e}") from e

    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # allow trailing comments after space + '#'
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        out.append(line)
    return out


def _print_train_error(exc: Exception, payload: dict, *, verbose: bool) -> None:
    typer.secho("\n[spectramind] Training failed.", fg=typer.colors.RED, err=True)
    typer.echo("---- Context ---------------------------------------------------", err=True)
    for k, v in payload.items():
        if k == "overrides":
            typer.echo("overrides:", err=True)
            for ov in v:
                typer.echo(f"  - {ov}", err=True)
        elif k == "env":
            typer.echo("env:", err=True)
            for ek, ev in v.items():
                typer.echo(f"  {ek}: {ev}", err=True)
        else:
            typer.echo(f"{k}: {v}", err=True)
    typer.echo("----------------------------------------------------------------", err=True)
    typer.secho(f"Reason: {type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True)
    if verbose:
        typer.echo("\nTraceback:", err=True)
        typer.echo("".join(traceback.format_exception(exc)), err=True)
    typer.echo(
        "\nTroubleshooting:\n"
        "  • Check Hydra keys/paths (toggle --strict/--no-strict)\n"
        "  • Verify dataset paths exist & are DVC-pulled\n"
        "  • Ensure Kaggle/CI deps match requirements (see requirements-kaggle.txt)\n"
        "  • Try a smoke run: spectramind train --dry-run --epochs 1\n"
        "  • Print config:    spectramind train --print-config -o trainer.max_epochs=1\n",
        err=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Typer Command
# ──────────────────────────────────────────────────────────────────────────────


@cli_app.command("train")
def train(
    # Hydra composition
    config_name: str = typer.Option(
        "train",
        "--config-name",
        "-c",
        help="Hydra config name to compose (e.g., train, debug).",
        show_default=True,
    ),
    override: List[str] = typer.Option(
        [],
        "--override",
        "-o",
        help="Hydra override strings (repeatable). "
             "Examples: data=kaggle trainer.max_epochs=20 model=v50",
    ),
    override_file: List[Path] = typer.Option(
        [],
        "--override-file",
        "-O",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a text file with Hydra overrides (one per line, '#' comments allowed). "
             "Repeatable.",
    ),
    # Trainer knobs
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Global PRNG seed for determinism (where supported).",
    ),
    devices: Optional[str] = typer.Option(
        None,
        "--devices",
        help="Device spec for Lightning/PyTorch (e.g., 'cpu', 'auto', '0', '0,1').",
    ),
    precision: Optional[str] = typer.Option(
        None,
        "--precision",
        help="Numerical precision (e.g., '32', '16-mixed', 'bf16-mixed').",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Override trainer max epochs.",
    ),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume-from",
        help="Path to checkpoint to resume training.",
    ),
    log_dir: Optional[str] = typer.Option(
        None,
        "--log-dir",
        help="Force log/artifact directory (otherwise from config).",
    ),
    # UX / safety
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run a fast smoke to validate wiring (no/very short training).",
    ),
    no_strict: bool = typer.Option(
        False,
        "--no-strict",
        help="Disable strict config key validation (Hydra/OmegaConf).",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Reduce console verbosity (CI/Kaggle-friendly).",
    ),
    # Introspection
    print_config: bool = typer.Option(
        False,
        "--print-config",
        help="Ask the runner to print the composed config before training.",
    ),
    save_config: Optional[str] = typer.Option(
        None,
        "--save-config",
        help="Ask the runner to save the composed config YAMLs to this directory.",
    ),
) -> None:
    """
    Train SpectraMind V50 dual-channel model (FGS1 + AIRS) with Hydra-composed config.

    Examples:
      • spectramind train
      • spectramind train -c train -o data=kaggle -o trainer.max_epochs=5 --dry-run
      • spectramind train -o model=v50 -o loss.composite.weights.fgs1=2.0
      • spectramind train --devices auto --precision 16-mixed --epochs 20
      • spectramind train -O overrides/base.txt -O overrides/ablations.txt --print-config
    """
    opts = TrainOptions(
        config_name=config_name,
        overrides=list(override),
        override_files=list(override_file),
        seed=seed,
        devices=devices,
        precision=precision,
        epochs=epochs,
        resume_from=resume_from,
        log_dir=log_dir,
        dry_run=dry_run,
        strict=not no_strict,
        quiet=quiet or _is_ci() or _is_kaggle(),
        print_config=print_config,
        save_config=save_config,
    )

    code = _dispatch_train(opts)
    if code != 0:
        # Exit non-zero for CI/kernels to catch failures
        raise typer.Exit(code)
