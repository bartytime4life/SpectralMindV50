from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from spectramind.validators.submission import SubmissionValidator, validate_submission

app = typer.Typer(name="submission", help="Submission utilities")


@app.command("validate")
def cli_validate(
    csv_path: Path = typer.Argument(..., help="Path to submission CSV"),
) -> None:
    """Validate a submission CSV. Exit code 0 means valid."""
    ok, errors = validate_submission(csv_path)
    if ok:
        typer.echo("✅ submission valid")
        raise typer.Exit(code=0)
    for e in errors:
        typer.echo(f"❌ {e}", err=True)
    raise typer.Exit(code=1)


@app.command("make-golden")
def cli_make_golden(
    out: Path = typer.Option(
        Path("tests/golden/submission_valid.csv"),
        "--out",
        "-o",
        help="Output CSV (or .csv.gz/.parquet)",
    ),
    rows: int = typer.Option(2, "--rows", "-r", help="Number of rows"),
    bins: Optional[int] = typer.Option(None, "--bins", help="Override bin count"),
    seed: int = typer.Option(7, "--seed", help="RNG seed"),
    gzip: bool = typer.Option(False, "--gzip", help="Write gzip CSV (.csv.gz)"),
    parquet: bool = typer.Option(False, "--parquet", help="Write Parquet instead of CSV"),
    float_format: Optional[str] = typer.Option(None, "--float-format", help="CSV float format"),
):
    """
    Generate the golden CSV used by unit tests.
    Delegates to the tools module to keep single source of truth.
    """
    # Import lazily so tools/ stays optional at runtime
    from tools.make_golden_submission import main as make_main  # type: ignore

    argv = [
        f"--out={str(out)}",
        f"--rows={rows}",
        f"--seed={seed}",
    ]
    if bins is not None:
        argv.append(f"--bins={bins}")
    if gzip:
        argv.append("--gzip")
    if parquet:
        argv.append("--parquet")
    if float_format:
        argv.append(f"--float-format={float_format}")

    rc = make_main(argv)
    raise typer.Exit(code=rc)
