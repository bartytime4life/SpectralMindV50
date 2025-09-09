from __future__ import annotations

"""
Stage registry & lazy wrappers.

Each stage must accept a single `cfg` mapping/dict-like (Hydra DictConfig supported
by the caller) and return a dictionary serializable to JSON (StageResult.data).

Design:
- Stages resolve lazily with import_module to keep startup snappy.
- Universal adapter tries fn(cfg) first, then fn(**cfg) with signature filtering.
- Env override: SPECTRAMIND_STAGE_MAP='{"stage":"package.mod:fn", ...}'
- Programmatic override/ reset cache helpers.
"""

from importlib import import_module
import inspect
import json
import os
from typing import Any, Callable, Dict, Mapping, MutableMapping, Protocol, Tuple, TypedDict

StageName = str

# Canonical stage names (plural for discoverability in CLIs and DVC)
# We also accept the alias "diagnose" → "diagnostics"
STAGE_NAMES: Tuple[StageName, ...] = (
    "calibrate",
    "preprocess",
    "train",
    "predict",
    "diagnostics",
    "submit",
)

_ALIAS: Dict[str, str] = {
    "diagnose": "diagnostics",
}

class StageCallable(Protocol):
    def __call__(self, cfg: Mapping[str, Any]) -> Mapping[str, Any]: ...


class _ModuleFn(TypedDict, total=False):
    mod: str
    fn: str
    help: str


# -----------------------------------------------------------------------------#
# Default map (lazy resolution) — adjust to your repo structure
# -----------------------------------------------------------------------------#
# NOTE:
#  • For orchestrators we expose pipeline modules that accept `cfg` mapping.
#  • For legacy CLIs (e.g., train/submit) we keep their modules and rely on the adapter
#    to support `fn(**cfg)` if necessary.

_STAGE_MAP: Dict[StageName, _ModuleFn] = {
    # Calibrate raw sensor data → calibrated cubes/files under DVC (pipeline orchestrator)
    "calibrate": {
        "mod": "spectramind.pipeline.calibrate",
        "fn": "run",
        "help": "Raw → calibrated (FGS1/AIRS); writes DVC artifacts",
    },
    # Feature extraction / detrending / tensors for model
    "preprocess": {
        "mod": "spectramind.pipeline.preprocess",  # optional module; adapter will error if missing
        "fn": "run",
        "help": "Calibrated → model-ready tensors; writes DVC artifacts",
    },
    # Train model (Hydra recommended) — often a Typer CLI under spectramind.train.cli:run
    "train": {
        "mod": "spectramind.train.cli",
        "fn": "run",
        "help": "Train model; writes checkpoints and logs",
    },
    # Predict / package submission (pipeline orchestrator)
    "predict": {
        "mod": "spectramind.pipeline.predict",
        "fn": "run",
        "help": "Predict → CSV(+ZIP)+manifest; Kaggle-ready",
    },
    # Diagnostics / report generation (pipeline orchestrator)
    "diagnostics": {
        "mod": "spectramind.pipeline.diagnostics",
        "fn": "run",
        "help": "Generate diagnostics & HTML report",
    },
    # Zip & finalize submission only (if you want a pure submit step)
    "submit": {
        "mod": "spectramind.submit.cli",
        "fn": "package",  # accept dict cfg via adapter
        "help": "Package/validate submission; zip/manifest",
    },
}

# Accept env overrides: {"stage": "package.mod[:function]"} or {"stage":"package.mod","fn":"function"}
_env_override = os.getenv("SPECTRAMIND_STAGE_MAP")
if _env_override:
    try:
        mapping = json.loads(_env_override)
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                key = _ALIAS.get(k, k)
                if key in _STAGE_MAP:
                    if isinstance(v, str):
                        # allow "pkg.mod:fn"
                        if ":" in v:
                            mod, fn = v.split(":", 1)
                            if mod and fn:
                                _STAGE_MAP[key]["mod"] = mod
                                _STAGE_MAP[key]["fn"] = fn
                        else:
                            _STAGE_MAP[key]["mod"] = v
                    elif isinstance(v, dict):
                        if "mod" in v and isinstance(v["mod"], str):
                            _STAGE_MAP[key]["mod"] = v["mod"]
                        if "fn" in v and isinstance(v["fn"], str):
                            _STAGE_MAP[key]["fn"] = v["fn"]
    except Exception:
        # ignore invalid JSON
        pass

# Cache of resolved & wrapped callables
_RESOLVED: MutableMapping[str, StageCallable] = {}


# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#

def _import_or_raise(mod_name: str, attr_name: str) -> Callable[..., Any]:
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


def _adapt_callable(target: Callable[..., Any]) -> StageCallable:
    """
    Universal adapter:
      1) try fn(cfg)
      2) try fn(**filtered_cfg) using signature introspection
    """
    sig = None
    try:
        sig = inspect.signature(target)
    except Exception:
        sig = None

    def _call(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
        # Path 1: fn(cfg)
        try:
            out = target(cfg)
            if isinstance(out, Mapping):
                return dict(out)
        except TypeError:
            pass  # try kwargs path next

        # Path 2: fn(**filtered_cfg) — map only accepted parameters by name
        if sig is not None:
            kwargs: Dict[str, Any] = {}
            params = sig.parameters
            # if function allows **kwargs we can pass everything
            allow_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if allow_var_kw:
                kwargs = dict(cfg)
            else:
                # only pass keys in function parameters
                for k in cfg.keys():
                    if k in params:
                        kwargs[k] = cfg[k]
            out = target(**kwargs)
            if isinstance(out, Mapping):
                return dict(out)

        # If the target doesn't return a mapping, coerce to empty dict (side-effect stage)
        return {}

    return _call


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#

def make_stage_callable(stage: StageName) -> StageCallable:
    stage_key = _ALIAS.get(stage, stage)
    if stage_key not in _STAGE_MAP:
        raise KeyError(f"Unknown stage '{stage}'. Known: {', '.join(STAGE_NAMES)}")

    cache_key = stage_key
    if cache_key in _RESOLVED:
        return _RESOLVED[cache_key]

    entry = _STAGE_MAP[stage_key]
    target = _import_or_raise(entry["mod"], entry["fn"])
    wrapped = _adapt_callable(target)
    _RESOLVED[cache_key] = wrapped
    return wrapped


def override_stage_map(mapping: Mapping[str, str]) -> None:
    """
    Programmatically override stage → module[:fn] resolution at runtime.

    mapping example:
      {"train": "my.pkg.train_app:entry", "submit": "tools.submitter:run"}
    """
    global _STAGE_MAP
    for k, v in mapping.items():
        key = _ALIAS.get(k, k)
        if key not in _STAGE_MAP or not isinstance(v, str):
            continue
        if ":" in v:
            mod, fn = v.split(":", 1)
            if mod and fn:
                _STAGE_MAP[key]["mod"] = mod
                _STAGE_MAP[key]["fn"] = fn
        else:
            _STAGE_MAP[key]["mod"] = v
    _RESOLVED.clear()


def reset_stage_cache() -> None:
    """Clear resolved wrapper cache (useful for tests/REPL)."""
    _RESOLVED.clear()


def get_stage_help(stage: str) -> str | None:
    """
    Return a short help string for a stage if its module defines `HELP` or a module docstring.
    """
    stage_key = _ALIAS.get(stage, stage)
    entry = _STAGE_MAP.get(stage_key)
    if not entry:
        return None
    try:
        m = import_module(entry["mod"])
    except Exception:
        return None
    help_text = getattr(m, "HELP", None)
    if isinstance(help_text, str) and help_text.strip():
        return help_text.strip()
    if isinstance(m.__doc__, str) and m.__doc__.strip():
        return m.__doc__.strip().splitlines()[0]
    return None
