from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .run import run_diagnostics

app = typer.Typer(no_args_is_help=True, add_completion=False, help="SpectraMind — diagnostics")

@app.command("run")
def cli_run(
    preds: Path = typer.Argument(..., exists=True, dir_okay=False, help="Predictions file (wide or narrow)"),
    out_dir: Path = typer.Option(Path("artifacts/diagnostics"), help="Output directory"),
    truth: Optional[Path] = typer.Option(None, "--truth", help="Optional truth table (narrow or wide)"),
    report_name: str = typer.Option("report.html", help="Report filename (HTML or .md)"),
) -> None:
    summary = run_diagnostics(preds_path=preds, truth_path=truth, out_dir=out_dir, report_name=report_name)
    typer.echo(f"Wrote {out_dir/'summary.json'}")
    typer.echo(f"Summary: {summary}")

if __name__ == "__main__":
    app()
