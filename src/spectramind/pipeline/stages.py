# src/spectramind/pipeline/stages.py
from __future__ import annotations

"""
Stage registry & lazy wrappers.

Each stage must accept a single `cfg` mapping/dict-like (Hydra DictConfig supported
by caller) and return a dictionary serializable to JSON (StageResult.data).
"""

from importlib import import_module
from typing import Any, Callable, Dict, Mapping, Protocol, TypedDict

StageName = str

# Canonical stage names (plural for discoverability in CLIs and DVC)
STAGE_NAMES: tuple[StageName, ...] = (
    "calibrate",
    "preprocess",
    "train",
    "predict",
    "diagnose",
    "submit",
)


class StageCallable(Protocol):
    def __call__(self, cfg: Mapping[str, Any]) -> Mapping[str, Any]: ...


class _ModuleFn(TypedDict):
    mod: str
    fn: str
    # Optional: help/doc for stage
    help: str


# Lazy resolution map — override if your repo uses different entrypoints
_STAGE_MAP: dict[StageName, _ModuleFn] = {
    # Calibrate raw sensor data → calibrated cubes/files under DVC
    "calibrate": {
        "mod": "spectramind.calib.cli",
        "fn": "main",  # expect `main()` to parse config from cfg if provided or env/argv
        "help": "Raw → calibrated (FGS1/AIRS); writes DVC artifacts",
    },
    # Feature extraction / detrending / tensors for model
    "preprocess": {
        "mod": "spectramind.preprocess.cli",
        "fn": "main",
        "help": "Calibrated → model-ready tensors; writes DVC artifacts",
    },
    # Train model (Hydra recommended)
    "train": {
        "mod": "spectramind.train.cli",
        "fn": "run",  # Typer command run()
        "help": "Train model; writes checkpoints and logs",
    },
    # Predict / package submission
    "predict": {
        "mod": "spectramind.predict.cli",
        "fn": "run",  # Typer command run()
        "help": "Predict → CSV(+ZIP)+manifest; Kaggle-ready",
    },
    # Diagnostics (GLL eval, dashboard, spectral checks)
    "diagnose": {
        "mod": "spectramind.diagnose.cli",
        "fn": "run",
        "help": "Generate diagnostics & HTML report",
    },
    # Zip & finalize submission only (if you want a pure submit step)
    "submit": {
        "mod": "spectramind.submit.cli",
        "fn": "package",  # or `main` if you have dedicated CLI
        "help": "Package/validate submission; zip/manifest",
    },
}


def _import_or_raise(mod_name: str, attr_name: str) -> StageCallable:
    try:
        m = import_module(mod_name)
    except Exception as e:  # pragma: no cover
        raise ImportError(f"Failed to import stage module '{mod_name}': {e}") from e
    try:
        fn = getattr(m, attr_name)
    except Exception as e:  # pragma: no cover
        raise AttributeError(f"Stage function '{attr_name}' not found in '{mod_name}'") from e

    if not callable(fn):
        raise TypeError(f"Stage target '{mod_name}.{attr_name}' is not callable")
    return fn  # type: ignore[return-value]


def make_stage_callable(stage: StageName) -> StageCallable:
    if stage not in _STAGE_MAP:
        raise KeyError(f"Unknown stage '{stage}'. Known: {', '.join(STAGE_NAMES)}")
    entry = _STAGE_MAP[stage]
    return _import_or_raise(entry["mod"], entry["fn"])
