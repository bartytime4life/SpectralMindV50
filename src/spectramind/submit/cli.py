# src/spectramind/submit/cli.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .validate import validate_csv, N_BINS_DEFAULT
from .package import package_submission

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SpectraMind V50 — submission utilities")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@app.command("validate")
def cli_validate(
    csv: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to submission CSV"),
    n_bins: int = typer.Option(N_BINS_DEFAULT, "--bins", help=f"Number of spectral bins (default {N_BINS_DEFAULT})"),
    schema: Optional[Path] = typer.Option(None, "--schema", help="Path to submission.schema.json (optional)"),
    chunksize: Optional[int] = typer.Option(None, "--chunksize", min=1, help="Chunked CSV validation (memory friendly)"),
    id_field: str = typer.Option("id", "--id-field", help="ID column name to expect (default: 'id')"),
    allow_alt_id: bool = typer.Option(True, "--allow-alt-id/--no-allow-alt-id", help="Also accept 'sample_id'"),
    strict_ids: bool = typer.Option(True, "--strict-ids/--no-strict-ids", help="Require non-empty unique IDs"),
    strict_wide_order: bool = typer.Option(
        False, "--strict-wide-order/--no-strict-wide-order", help="Enforce exact wide column order if detected"
    ),
    report: Optional[Path] = typer.Option(None, "--report", help="Write JSON report to this path"),
    max_errors: int = typer.Option(200, "--max-errors", help="Max errors to include in the JSON report"),
) -> None:
    """
    Validate a submission CSV using schema+physics checks.
    """
    res = validate_csv(
        csv,
        n_bins=n_bins,
        strict_ids=strict_ids,
        chunksize=chunksize,
        id_field=id_field,
        allow_alt_id=allow_alt_id,
        schema_path=schema,
        write_report=report,
        max_errors_in_report=max_errors,
        strict_wide_order=strict_wide_order,
    )

    if res.ok:
        typer.secho(
            f"OK ✅  rows={res.n_rows}  valid={res.n_valid}  errors=0",
            fg=typer.colors.GREEN,
        )
        return

    # print a concise summary; if --report was specified, full details are there
    typer.secho(
        f"❌ Validation failed  rows={res.n_rows}  valid={res.n_valid}  errors={len(res.errors)}",
        fg=typer.colors.RED,
        err=True,
    )
    preview = res.errors[: min(10, len(res.errors))]
    for e in preview:
        typer.echo(f"- {e}", err=True)
    if report:
        typer.echo(f"(full report written to {report})", err=True)
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# package
# ---------------------------------------------------------------------------

@app.command("package")
def cli_package(
    csv_or_df: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="CSV to package"),
    out_dir: Path = typer.Argument(..., help="Output directory for artifacts"),
    zip_on: bool = typer.Option(True, "--zip/--no-zip", help="Create submission.zip (default: yes)"),
    zip_name: str = typer.Option("submission.zip", "--zip-name", help="ZIP filename"),
    filename: str = typer.Option("submission.csv", "--filename", help="CSV filename to write"),
    n_bins: int = typer.Option(N_BINS_DEFAULT, "--bins", help=f"Number of spectral bins (default {N_BINS_DEFAULT})"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Validate before packaging"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Fix timestamps for deterministic artifacts"),
    manifest_schema: Optional[Path] = typer.Option(None, "--manifest-schema", help="(future) manifest schema path"),
) -> None:
    """
    Package a validated submission CSV into final artifact(s): CSV + manifest.json (+ ZIP).
    """
    out_path = package_submission(
        csv_or_df,
        out_dir,
        filename=filename,
        make_zip=zip_on,
        zip_name=zip_name,
        n_bins=n_bins,
        seed=seed,
        strict_validate=strict,
        manifest_schema=manifest_schema,
    )
    typer.secho(f"Packaged → {out_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
