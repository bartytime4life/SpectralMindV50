# src/spectramind/cli/train.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import typer

from . import cli_app

# ---- CLI Options Dataclass ----------------------------------------------------


@dataclass
class TrainOptions:
    config_name: str = "train"
    overrides: List[str] = None
    seed: Optional[int] = None
    devices: Optional[str] = None
    precision: Optional[str] = None
    epochs: Optional[int] = None
    resume_from: Optional[str] = None
    log_dir: Optional[str] = None
    dry_run: bool = False
    strict: bool = True  # fail loud if config keys are wrong (Hydra/OmegaConf)
    quiet: bool = False  # reduce console noise for CI/Kaggle runs


# ---- Internal Call ------------------------------------------------------------


def _dispatch_train(opts: TrainOptions) -> int:
    """
    Thin wrapper that calls the internal training entrypoint.

    Business logic MUST live outside the CLI (keeps this import cheap & testable).
    """
    try:
        # Preferred: a single, stable entrypoint the CLI can call.
        # Adjust import path to your repo’s actual module:
        # e.g., spectramind.training.runner.run, or pipeline.train.run
        from spectramind.pipeline.train import run as run_training  # type: ignore
    except Exception as e:
        typer.secho(
            f"[spectramind] Cannot import training entrypoint: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        return 2

    # Compose a plain dict for the runner; the runner is responsible for Hydra
    # composition and validation so the CLI stays agnostic to config shape.
    payload = {
        "config_name": opts.config_name,
        "overrides": opts.overrides or [],
        "seed": opts.seed,
        "devices": opts.devices,
        "precision": opts.precision,
        "epochs": opts.epochs,
        "resume_from": opts.resume_from,
        "log_dir": opts.log_dir,
        "dry_run": opts.dry_run,
        "strict": opts.strict,
        "quiet": opts.quiet,
        # Environment hints that the runner may choose to honor:
        "env": {
            "is_kaggle": _is_kaggle(),
            "is_ci": _is_ci(),
        },
    }

    try:
        run_training(**payload)
    except KeyboardInterrupt:
        typer.secho("[spectramind] Training interrupted by user.", fg=typer.colors.YELLOW)
        return 130
    except Exception as e:
        # Loud, typed error with helpful remediation
        _print_train_error(e, payload)
        return 1

    return 0


# ---- Helpers -----------------------------------------------------------------


def _is_kaggle() -> bool:
    # Kaggle kernels typically define these env vars
    return any(
        k in os.environ
        for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DOCKER_IMAGE")
    ) or os.path.exists("/kaggle/input")


def _is_ci() -> bool:
    # Common CI indicators
    return os.environ.get("CI", "").lower() == "true" or "GITHUB_ACTIONS" in os.environ


def _print_train_error(exc: Exception, payload: dict) -> None:
    typer.secho("\n[spectramind] Training failed.", fg=typer.colors.RED, err=True)
    typer.echo("---- Context ---------------------------------------------------", err=True)
    for k, v in payload.items():
        if k == "overrides":
            # Print overrides multi-line for readability
            typer.echo(f"{k}:", err=True)
            for ov in v:
                typer.echo(f"  - {ov}", err=True)
        else:
            typer.echo(f"{k}: {v}", err=True)
    typer.echo("----------------------------------------------------------------", err=True)
    typer.secho(f"Reason: {type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True)
    typer.echo(
        "\nTroubleshooting:\n"
        "  • Check your Hydra config/overrides keys (use --strict/--no-strict)\n"
        "  • Verify dataset paths exist and are DVC-pulled locally\n"
        "  • Ensure Kaggle/CI has the required dependencies (see requirements-kaggle.txt)\n"
        "  • Try a minimal dry run: spectramind train --dry-run --epochs 1\n",
        err=True,
    )


# ---- Typer Command -----------------------------------------------------------


@cli_app.command("train")
def train(
    config_name: str = typer.Option(
        "train",
        "--config-name",
        "-c",
        help="Hydra config name to compose (e.g., train, debug).",
        show_default=True,
    ),
    override: List[str] = typer.Option(  # multiple: -o key=val -o group/choice@node
        [],
        "--override",
        "-o",
        help="Hydra override strings (repeat to add multiple). "
             "Examples: data=kaggle trainer.max_epochs=20 model=v50",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Global PRNG seed for full determinism (where supported).",
    ),
    devices: Optional[str] = typer.Option(
        None,
        "--devices",
        help="Device spec for Lightning/PyTorch (e.g., 'cpu', '0', '0,1', 'auto').",
    ),
    precision: Optional[str] = typer.Option(
        None,
        "--precision",
        help="Numerical precision (e.g., '32', '16-mixed', 'bf16-mixed').",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Override max epochs for trainer.",
    ),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume-from",
        help="Path to checkpoint to resume training.",
    ),
    log_dir: Optional[str] = typer.Option(
        None,
        "--log-dir",
        help="Force log/artifact directory (otherwise taken from config).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run fast smoke (no/very short training) to validate wiring.",
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
) -> None:
    """
    Train SpectraMind V50 dual-channel model (FGS1 + AIRS) with Hydra-composed config.

    Examples:
      • spectramind train
      • spectramind train -c train -o data=kaggle -o trainer.max_epochs=5 --dry-run
      • spectramind train -o model=v50 -o loss.composite.weights.fgs1=2.0
      • spectramind train --devices auto --precision 16-mixed --epochs 20
    """
    opts = TrainOptions(
        config_name=config_name,
        overrides=list(override),
        seed=seed,
        devices=devices,
        precision=precision,
        epochs=epochs,
        resume_from=resume_from,
        log_dir=log_dir,
        dry_run=dry_run,
        strict=not no_strict,
        quiet=quiet or _is_ci() or _is_kaggle(),
    )

    code = _dispatch_train(opts)
    # Exit non-zero for CI to catch failures
    if code != 0:
        raise typer.Exit(code)


# Notes:
# • This module only parses CLI args and delegates to the runner. Keep it thin.
# • The internal runner should:
#     - Compose Hydra config (config_name + overrides)
#     - Set seeds, devices, precision, epochs appropriately
#     - Enforce Kaggle/CI guardrails (no internet, time limits)
#     - Produce artifacts & metrics per repo schema
# • See repo blueprint and CLI/UX guidelines for consistency across commands.
