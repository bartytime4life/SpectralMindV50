# src/spectramind/submit/cli.py
from __future__ import annotations

from pathlib import Path
import typer
import pandas as pd

from .validate import validate_csv
from .package import package_submission
from .format import format_predictions

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Submission utilities")


@app.command("validate")
def cli_validate(
    csv: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    n_bins: int = typer.Option(283, help="Number of spectral bins (default 283)"),
) -> None:
    """Validate a submission CSV."""
    report = validate_csv(csv, n_bins=n_bins, strict_order=True, check_unique_ids=True)
    if report.ok:
        typer.secho("OK: submission is valid ✅", fg=typer.colors.GREEN)
    else:
        typer.secho("Submission invalid ❌", fg=typer.colors.RED)
        for e in report.errors:
            typer.echo(f"- {e}")
        raise typer.Exit(code=1)


@app.command("package")
def cli_package(
    csv_or_df: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    out_dir: Path = typer.Argument(...),
    zip: bool = typer.Option(True, "--zip/--no-zip", help="Make submission.zip (default: yes)"),
    n_bins: int = typer.Option(283, help="Number of spectral bins"),
) -> None:
    """Package an already formatted CSV (or predictions CSV) into final artifact(s)."""
    out_path = package_submission(csv_or_df, out_dir, make_zip=zip, n_bins=n_bins)
    typer.secho(f"Packaged → {out_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()