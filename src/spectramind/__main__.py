# src/spectramind/__main__.py
# =============================================================================
# 🛰️ SpectraMind V50 — CLI Entrypoint
# -----------------------------------------------------------------------------
# This module enables running the CLI via:
#   python -m spectramind ...
# instead of requiring an installed console_script.
#
# It simply forwards execution to the Typer app defined in `cli.py`.
# =============================================================================

from spectramind.cli import app


def main() -> None:
    """Run the SpectraMind V50 CLI (delegates to Typer `app`)."""
    app()


if __name__ == "__main__":
    main()
