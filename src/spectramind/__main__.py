# src/spectramind/__main__.py
# =============================================================================
# 🛰️ SpectraMind V50 — CLI Entrypoint (python -m spectramind …)
# -----------------------------------------------------------------------------
# Delegates to the Typer app defined in spectramind.cli.
# • Robust error handling (clean exit codes for CI/Kaggle)
# • Optional rich tracebacks (TTY-aware; never required)
# • Deterministic env & headless plotting defaults
# • Unicode/stdio hygiene on Windows and Kaggle
# • Safe fallbacks: prefer cli.main(), else cli.app()
# =============================================================================
from __future__ import annotations

import os
import sys
from typing import Callable, Optional

# ---- Minimal env hygiene (before importing anything heavy) -------------------
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MPLBACKEND", "Agg")              # headless-safe plotting
os.environ.setdefault("SPECTRAMIND_ENTRYPOINT", "module")
# Keep threaded BLAS predictable on CI/Kaggle unless user overrides
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---- Optional pretty tracebacks (safe if unavailable) -----------------------
def _enable_rich_tracebacks() -> None:
    try:
        # Only turn on in interactive TTYs to avoid noisy CI logs
        if sys.stderr.isatty():
            from rich.traceback import install as _rich_install  # type: ignore
            _rich_install(show_locals=False, suppress=["typer", "click"])
    except Exception:
        pass

_enable_rich_tracebacks()

# ---- Unicode/stdio hygiene (Windows cmd / Kaggle consoles) ------------------
def _force_utf8_stdio() -> None:
    try:
        # Python 3.7+ has .reconfigure() on TextIOBase
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        # Best-effort only
        pass

_force_utf8_stdio()

# ---- SIGPIPE behavior (Unix) ------------------------------------------------
def _set_sigpipe_quiet() -> None:
    # Avoid BrokenPipeErrors when piping to head -n1, etc.
    if os.name != "nt":
        try:
            import signal  # noqa: WPS433
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        except Exception:
            pass

_set_sigpipe_quiet()


def _run_cli() -> int:
    """
    Run the SpectraMind CLI.
    Prefers spectramind.cli.main() if available, else calls Typer app() directly.
    Returns an exit code suitable for CI.
    """
    try:
        # Import inside function to avoid import-time side effects at module import.
        from spectramind import cli  # type: ignore
    except Exception as e:
        # Keep the message crisp and single-line for CI logs
        sys.stderr.write(
            "error: failed to import spectramind.cli: "
            f"{e}. Is SpectraMind installed? e.g. `pip install spectramind` or your local editable install.\n"
        )
        return 1

    def _invoke(fn: Callable[[], object]) -> int:
        """Call a (possibly Typer) callable and normalize exit codes."""
        try:
            rv = fn()
            # Typer/Click usually raises SystemExit; if not, coerce to 0
            return 0 if rv is None else 0
        except SystemExit as se:
            # Respect explicit SystemExit codes
            code = getattr(se, "code", 0)
            try:
                return int(code) if code is not None else 0
            except Exception:
                return 1
        except KeyboardInterrupt:
            sys.stderr.write("interrupted: keyboard interrupt\n")
            return 130  # standard SIGINT exit
        except BrokenPipeError:
            # Quietly succeed when downstream pipe closes
            try:
                sys.stderr.flush()
            except Exception:
                pass
            return 0
        except Exception as e:
            sys.stderr.write(f"error: unhandled exception: {e}\n")
            return 1

    # Preferred: cli.main()
    main_fn: Optional[Callable[[], object]] = getattr(cli, "main", None)  # type: ignore[attr-defined]
    if callable(main_fn):
        return _invoke(main_fn)  # type: ignore[call-arg]

    # Fallback: Typer app() (callable)
    app_obj = getattr(cli, "app", None)  # type: ignore[attr-defined]
    if callable(app_obj):
        return _invoke(app_obj)  # type: ignore[call-arg]

    sys.stderr.write("error: spectramind.cli has neither 'main' nor 'app'\n")
    return 1


def main() -> None:
    """Module entrypoint for `python -m spectramind`."""
    code = _run_cli()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
