# src/spectramind/cli/__init__.py
"""
SpectraMind V50 — CLI Package
=============================

Typer-based, CLI-first orchestration for the NeurIPS 2025 Ariel Data Challenge.
This package wires subcommands (calibrate/train/predict/diagnostics/submit)
into a single `spectramind` entrypoint while remaining:

- **Thin**: subcommands delegate to `spectramind.pipeline/*` and friends.
- **Hydra-safe**: configs are composed by the caller; no hard-coded params here.
- **Deterministic**: no internet calls or hidden side effects on import.
- **Extensible**: supports third-party subcommand plugins via entry points.

Conventions
-----------
Each built-in subcommand module should expose either:
  * `app: typer.Typer`  → mounted as `spectramind <name>`
  * `register(app: typer.Typer) -> None`  → custom mounting
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Dict, Optional, Tuple

import typer

# --------------------------------------------------------------------------------------
# Public app
# --------------------------------------------------------------------------------------

cli_app = typer.Typer(
    name="spectramind",
    help="SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge CLI",
    no_args_is_help=True,
    add_completion=False,  # Explicitly disable shell completion by default
)

__all__ = ["cli_app", "init_cli"]


# --------------------------------------------------------------------------------------
# Internal wiring
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SubcommandSpec:
    name: str
    module: str
    attr_app: str = "app"        # Typer sub-app attribute to mount if present
    attr_register: str = "register"  # Fallback registration function


# Built-in subcommands shipped with SpectraMind V50
_BUILTINS: Tuple[SubcommandSpec, ...] = (
    SubcommandSpec("calibrate", "spectramind.cli.calibrate"),
    SubcommandSpec("train", "spectramind.cli.train"),
    SubcommandSpec("predict", "spectramind.cli.predict"),
    SubcommandSpec("diagnostics", "spectramind.cli.diagnostics"),
    SubcommandSpec("submit", "spectramind.cli.submit"),
)


def _missing_command_factory(name: str, import_err: Exception) -> typer.Typer:
    """
    Create a placeholder command that surfaces a friendly error *when invoked*,
    instead of hard-failing at import time.
    """
    app = typer.Typer(name=name, help=f"{name} (unavailable)")

    @app.callback(no_args_is_help=True)
    def _entrypoint() -> None:  # noqa: D401
        """
        Placeholder for a missing subcommand. Prints diagnostics and exits.
        """
        typer.secho(f"[spectramind] Subcommand '{name}' is not available.", fg=typer.colors.RED, err=True)
        typer.secho(f"Reason: {type(import_err).__name__}: {import_err}", fg=typer.colors.BRIGHT_RED, err=True)
        typer.secho(
            "Tips:\n"
            f"  • Ensure the module providing '{name}' is installed and importable.\n"
            f"  • If this is an optional extra, install extras: `pip install spectramind-v50[{name}]`.\n"
            "  • Run `spectramind diagnostics` to verify your environment.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)

    return app


def _mount_spec(app: typer.Typer, spec: SubcommandSpec) -> None:
    """
    Try to mount a subcommand by importing its module and attaching either:
      * module.app (Typer), or
      * module.register(app) (callable)
    If import fails, mount a graceful placeholder that errors on invocation.
    """
    try:
        mod = import_module(spec.module)
    except Exception as e:  # broad on purpose: surface at call time
        app.add_typer(_missing_command_factory(spec.name, e), name=spec.name)
        return

    # Preferred: a Typer sub-app named `app`
    sub_app = getattr(mod, spec.attr_app, None)
    if isinstance(sub_app, typer.Typer):
        app.add_typer(sub_app, name=spec.name)
        return

    # Fallback: a register() function that mounts into `app`
    registrar = getattr(mod, spec.attr_register, None)
    if callable(registrar):
        registrar(app)
        return

    # If neither pattern is found, add a diagnostic stub
    err = RuntimeError(
        f"Module '{spec.module}' defines neither '{spec.attr_app}: Typer' "
        f"nor '{spec.attr_register}(app: Typer)'."
    )
    app.add_typer(_missing_command_factory(spec.name, err), name=spec.name)


def _discover_plugins() -> Dict[str, Callable[[typer.Typer], None]]:
    """
    Discover plugin registrars via entry points.

    Third-parties may provide additional subcommands with:
        [project.entry-points]
        spectramind.cli_plugins = 
            mytool = mypkg.mytool:register

    Returns a mapping: plugin_name -> registrar(app)
    """
    registrars: Dict[str, Callable[[typer.Typer], None]] = {}
    try:
        from importlib.metadata import entry_points  # py>=3.10
        eps = entry_points().select(group="spectramind.cli_plugins")  # type: ignore[attr-defined]
    except Exception:
        eps = ()

    for ep in eps:
        try:
            func = ep.load()
            if callable(func):
                registrars[ep.name] = func  # type: ignore[assignment]
        except Exception:
            # Ignore broken plugins silently to avoid crashing the CLI.
            # They can be diagnosed via `diagnostics`.
            continue
    return registrars


def _mount_plugins(app: typer.Typer) -> None:
    for name, registrar in _discover_plugins().items():
        try:
            registrar(app)
        except Exception as e:
            # Mount a placeholder that explains the plugin failed to load
            app.add_typer(_missing_command_factory(name, e), name=name)


# --------------------------------------------------------------------------------------
# Top-level callback: global options like --version
# --------------------------------------------------------------------------------------

def _version_string() -> str:
    try:
        pkg_ver = version("spectramind-v50")
    except PackageNotFoundError:
        # Fallback if package metadata is not available (editable install / source run)
        pkg_ver = "0.0.0+local"
    return f"SpectraMind V50 {pkg_ver}"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(_version_string())
        raise typer.Exit()


@cli_app.callback()
def _root(
    version_: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print SpectraMind V50 version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """
    Root callback reserved for global flags. Subcommands live under this node.
    """
    _ = version_  # placate linters; handled in callback


# --------------------------------------------------------------------------------------
# Initialization hook
# --------------------------------------------------------------------------------------

def init_cli(app: Optional[typer.Typer] = None) -> typer.Typer:
    """
    Initialize the CLI by mounting built-in subcommands and discovering plugins.

    This is idempotent and safe to call multiple times.
    """
    target = app or cli_app

    # Attach built-in subcommands
    for spec in _BUILTINS:
        _mount_spec(target, spec)

    # Attach third-party plugin subcommands (optional)
    _mount_plugins(target)

    return target


# Initialize when this module is imported so `spectramind` is ready-to-run.
init_cli()


# --------------------------------------------------------------------------------------
# Optional: compatibility imports (side-effect registration)
# --------------------------------------------------------------------------------------
# These are *best-effort* imports to maintain backward compatibility with code
# that previously relied on side-effect registration during import. They are
# intentionally wrapped in try/except and do not fail the CLI if unavailable.

for _legacy_mod in (
    "spectramind.cli.calibrate",
    "spectramind.cli.train",
    "spectramind.cli.predict",
    "spectramind.cli.diagnostics",
    "spectramind.cli.submit",
):
    try:
        import_module(_legacy_mod)
    except Exception:
        pass
