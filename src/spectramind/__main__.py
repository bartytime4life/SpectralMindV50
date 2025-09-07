# src/spectramind/__main__.py
# =============================================================================
# 🛰️ SpectraMind V50 — CLI Entrypoint (python -m spectramind …)
# -----------------------------------------------------------------------------
# Delegates to the Typer app defined in spectramind.cli.
# Adds robust error handling and sensible defaults for CI/Kaggle.
# =============================================================================
from __future__ import annotations

import os
import sys

# Optional pretty tracebacks; never required.
try:
    from rich.traceback import install as _rich_install  # type: ignore
    _rich_install(show_locals=False, suppress=["typer", "click"])
except Exception:
    pass

# Light env hygiene: deterministic hashing & non-interactive matplotlib defaults
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MPLBACKEND", "Agg")
# Mark entrypoint flavor for logs/telemetry (purely informational)
os.environ.setdefault("SPECTRAMIND_ENTRYPOINT", "module")

def _run_cli() -> int:
    """
    Run the SpectraMind CLI.
    Prefers spectramind.cli.main() if available, else calls app() directly.
    Returns an exit code.
    """
    try:
        # Import inside function to avoid import side-effects on module load.
        from spectramind import cli  # type: ignore
    except Exception as e:  # Import failure: print a crisp message and bail.
        sys.stderr.write(f"error: failed to import spectramind.cli: {e}\n")
        return 1

    # Preferred: cli.main() wrapper (prints less on import-time errors).
    try:
        if hasattr(cli, "main") and callable(cli.main):
            cli.main()  # type: ignore[attr-defined]
            return 0
    except SystemExit as se:
        # Typer/Click may raise SystemExit for normal flow; propagate code.
        return int(getattr(se, "code", 0) or 0)
    except KeyboardInterrupt:
        sys.stderr.write("interrupted: keyboard interrupt\n")
        return 130
    except Exception as e:
        sys.stderr.write(f"error: unhandled exception in spectramind.cli.main(): {e}\n")
        return 1

    # Fallback: call the Typer app object directly.
    try:
        if hasattr(cli, "app"):
            # Typer's app() will handle argv and raise SystemExit internally.
            cli.app()  # type: ignore[attr-defined]
            return 0
        else:
            sys.stderr.write("error: spectramind.cli has no 'main' or 'app'\n")
            return 1
    except SystemExit as se:
        return int(getattr(se, "code", 0) or 0)
    except KeyboardInterrupt:
        sys.stderr.write("interrupted: keyboard interrupt\n")
        return 130
    except Exception as e:
        sys.stderr.write(f"error: unhandled exception while running Typer app: {e}\n")
        return 1


def main() -> None:
    """Module entrypoint for `python -m spectramind`."""
    code = _run_cli()
    # Ensure process exits with the right code when run as a module.
    raise SystemExit(code)


if __name__ == "__main__":
    main()
