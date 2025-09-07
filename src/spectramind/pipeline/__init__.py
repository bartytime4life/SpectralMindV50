# src/spectramind/pipeline/__init__.py
from __future__ import annotations

"""
SpectraMind V50 — Pipeline Entrypoints
======================================

Python API facade for the main pipeline stages.

Usage
-----
    >>> from spectramind.pipeline import train, calibrate, predict, diagnostics, submit
    >>> train(config_name="train", overrides=["+env=kaggle"])

Design
------
We *lazy-load* stage modules the first time you call them, so importing this
package is fast and side-effect free (important in CLI, Kaggle, and tests).
Each stage module is expected to expose a `run(**kwargs)` callable.

Stages:
- train         → model training (dual encoders + decoder)
- calibrate     → sensor calibration (FGS1 + AIRS)
- predict       → inference & spectral μ/σ outputs
- diagnostics   → reporting & evaluation
- submit        → Kaggle packaging & validation

If a stage is not implemented yet, calling it raises a clear RuntimeError with
guidance on what to add (module path + expected symbol).
"""

from importlib import import_module
from typing import Any, Callable, Dict, TYPE_CHECKING

__all__: list[str] = [
    "train",
    "calibrate",
    "predict",
    "diagnostics",
    "submit",
]


def _lazy_stage_runner(module_path: str, symbol: str = "run") -> Callable[..., Any]:
    """
    Return a callable that imports `module_path` and forwards to `symbol` at call time.

    Parameters
    ----------
    module_path
        Dotted module path, e.g. 'spectramind.pipeline.train'.
    symbol
        Callable attribute in the module (default: 'run').

    Returns
    -------
    Callable[..., Any]
        A function that imports and dispatches to the underlying stage runner.
    """
    def _runner(*args: Any, **kwargs: Any) -> Any:
        try:
            mod = import_module(module_path)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Pipeline stage module not found: '{module_path}'.\n"
                f"Implement the module or adjust the import path.\n"
                f"Cause: {type(e).__name__}: {e}"
            ) from e
        if not hasattr(mod, symbol):
            raise RuntimeError(
                f"Pipeline stage '{module_path}' is missing the expected callable "
                f"'{symbol}(**kwargs)'. Please expose a '{symbol}' entrypoint."
            )
        fn = getattr(mod, symbol)
        if not callable(fn):
            raise RuntimeError(
                f"Attribute '{symbol}' in module '{module_path}' is not callable."
            )
        return fn(*args, **kwargs)
    return _runner


# Public, lazy stage entrypoints
train: Callable[..., Any] = _lazy_stage_runner("spectramind.pipeline.train", "run")
calibrate: Callable[..., Any] = _lazy_stage_runner("spectramind.pipeline.calibrate", "run")
predict: Callable[..., Any] = _lazy_stage_runner("spectramind.pipeline.predict", "run")
diagnostics: Callable[..., Any] = _lazy_stage_runner("spectramind.pipeline.diagnostics", "run")
submit: Callable[..., Any] = _lazy_stage_runner("spectramind.pipeline.submit", "run")


# --- Optional type hints for better editor support ---------------------------
# If you keep stable signatures in the stage modules, expose them here for static typing.
if TYPE_CHECKING:
    # Import only for type checkers (no runtime import/cost)
    from typing import Iterable, Optional

    def train(
        *,
        config_name: str = "train",
        overrides: Iterable[str] | None = None,
        out_dir: str | None = None,
        strict: bool = True,
        quiet: bool = False,
        env: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def calibrate(
        *,
        config_name: str = "calibrate",
        overrides: Iterable[str] | None = None,
        out_dir: str | None = None,
        strict: bool = True,
        quiet: bool = False,
        dry_run: bool = False,
        env: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def predict(
        *,
        config_name: str = "predict",
        overrides: Iterable[str] | None = None,
        out_dir: str | None = None,
        strict: bool = True,
        quiet: bool = False,
        env: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def diagnostics(
        *,
        config_name: str = "diagnose",
        overrides: Iterable[str] | None = None,
        out_dir: str | None = None,
        strict: bool = True,
        quiet: bool = False,
        env: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def submit(
        *,
        config_name: str = "submit",
        overrides: Iterable[str] | None = None,
        out_dir: str | None = None,
        strict: bool = True,
        quiet: bool = False,
        env: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...
