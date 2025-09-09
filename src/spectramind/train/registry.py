# src/spectramind/train/registry.py
# =============================================================================
# SpectraMind V50 — Lightweight Registry for Builders
# -----------------------------------------------------------------------------
# Exposes decorators & getters for:
#   • Models:     (cfg, *, criterion) -> LightningModule
#   • Losses:     (cfg)               -> callable / nn.Module / loss_fn
#   • Optimizers: (cfg, *, params)    -> torch.optim.Optimizer
#   • Schedulers: (cfg, optimizer, **kw) -> torch.optim.lr_scheduler or Lightning dict
#
# Notes:
# - Used by train/train.py: get_model_builder, get_loss_builder,
#   get_optimizer_builder, get_scheduler_builder.
# - Includes default optimizer/scheduler builders backed by train.optim.
# - Model & Loss registries start empty; register yours via decorators or call `.register()`.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import difflib


# -----------------------------------------------------------------------------
# Generic registry
# -----------------------------------------------------------------------------

@dataclass
class _Entry:
    name: str
    fn: Callable[..., Any]
    help: Optional[str] = None


class Registry:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._store: Dict[str, _Entry] = {}

    # Decorator-based registration
    def register(self, name: Optional[str] = None, *, help: Optional[str] = None):
        """
        Usage:
            @REGISTRY.register("my_key")
            def builder(...): ...
        """
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            key = (name or fn.__name__).lower()
            if key in self._store:
                raise KeyError(f"{self._kind} '{key}' already registered.")
            self._store[key] = _Entry(name=key, fn=fn, help=help)
            return fn
        return _decorator

    # Programmatic registration
    def add(self, name: str, fn: Callable[..., Any], *, help: Optional[str] = None) -> None:
        key = name.lower()
        if key in self._store:
            raise KeyError(f"{self._kind} '{key}' already registered.")
        self._store[key] = _Entry(name=key, fn=fn, help=help)

    def get(self, name: str) -> Callable[..., Any]:
        key = (name or "").lower()
        if key not in self._store:
            self._raise_not_found(key)
        return self._store[key].fn

    def has(self, name: str) -> bool:
        return name.lower() in self._store

    def names(self) -> Iterable[str]:
        return list(self._store.keys())

    def help_map(self) -> Dict[str, Optional[str]]:
        return {k: v.help for k, v in self._store.items()}

    def _raise_not_found(self, key: str) -> None:
        choices = list(self._store.keys())
        msg = [f"Unknown {self._kind} '{key}'."]
        if choices:
            msg.append(f"Available {self._kind}s: {', '.join(sorted(choices))}")
            similar = difflib.get_close_matches(key, choices, n=3, cutoff=0.5)
            if similar:
                msg.append(f"Did you mean: {', '.join(similar)}?")
        raise KeyError(" ".join(msg))


# -----------------------------------------------------------------------------
# Global registries (public)
# -----------------------------------------------------------------------------

_MODEL_BUILDERS = Registry(kind="model builder")
_LOSS_BUILDERS = Registry(kind="loss builder")
_OPTIMIZER_BUILDERS = Registry(kind="optimizer builder")
_SCHEDULER_BUILDERS = Registry(kind="scheduler builder")


# -----------------------------------------------------------------------------
# Public decorators (for user code to register builders)
# -----------------------------------------------------------------------------

def register_model(name: Optional[str] = None, *, help: Optional[str] = None):
    """
    Register a model builder:
        signature: builder(cfg, *, criterion=None) -> LightningModule
    """
    return _MODEL_BUILDERS.register(name, help=help)


def register_loss(name: Optional[str] = None, *, help: Optional[str] = None):
    """
    Register a loss builder:
        signature: builder(cfg) -> criterion (callable/nn.Module)
    """
    return _LOSS_BUILDERS.register(name, help=help)


def register_optimizer(name: Optional[str] = None, *, help: Optional[str] = None):
    """
    Register an optimizer builder:
        signature: builder(cfg, *, params) -> torch.optim.Optimizer
    """
    return _OPTIMIZER_BUILDERS.register(name, help=help)


def register_scheduler(name: Optional[str] = None, *, help: Optional[str] = None):
    """
    Register a scheduler builder:
        signature: builder(cfg, optimizer, **kw) -> scheduler or Lightning dict
    """
    return _SCHEDULER_BUILDERS.register(name, help=help)


# -----------------------------------------------------------------------------
# Public getters (used by train/train.py)
# -----------------------------------------------------------------------------

def get_model_builder(name: Optional[str]) -> Callable[..., Any]:
    if not name:
        raise KeyError("Model builder name not provided.")
    return _MODEL_BUILDERS.get(name)


def get_loss_builder(name: Optional[str]) -> Callable[..., Any]:
    if not name:
        raise KeyError("Loss builder name not provided.")
    return _LOSS_BUILDERS.get(name)


def get_optimizer_builder(name: Optional[str]) -> Callable[..., Any]:
    if not name:
        raise KeyError("Optimizer builder name not provided.")
    return _OPTIMIZER_BUILDERS.get(name)


def get_scheduler_builder(name: Optional[str]) -> Callable[..., Any]:
    if not name:
        raise KeyError("Scheduler builder name not provided.")
    return _SCHEDULER_BUILDERS.get(name)


# -----------------------------------------------------------------------------
# Defaults: optimizer & scheduler builders (wire to spectramind.train.optim)
# -----------------------------------------------------------------------------
# These make the system usable out-of-the-box without custom registry entries.
# You can override by registering a builder with the same name before use.
# -----------------------------------------------------------------------------

def _auto_register_default_optimizers_and_schedulers() -> None:
    try:
        from .optim import OptimizerConfig, SchedulerConfig, build_optimizer, build_scheduler
    except Exception:
        # If optim module is unavailable for some reason, skip auto defaults
        return

    @_OPTIMIZER_BUILDERS.register("adamw", help="AdamW with decoupled weight decay groups")
    def _adamw_builder(cfg: Any, *, params) -> Any:
        """
        cfg can be:
          • Dict-like with keys accepted by OptimizerConfig
          • An OptimizerConfig instance
        """
        ocfg = cfg if isinstance(cfg, OptimizerConfig) else OptimizerConfig(**dict(cfg))
        return build_optimizer(ocfg, params_or_module=params if not hasattr(params, "parameters") else params)

    @_OPTIMIZER_BUILDERS.register("adam", help="Adam (decoupled decay groups)")
    def _adam_builder(cfg: Any, *, params) -> Any:
        ocfg = cfg if isinstance(cfg, OptimizerConfig) else OptimizerConfig(**dict(cfg))
        ocfg.name = "adam"
        return build_optimizer(ocfg, params_or_module=params if not hasattr(params, "parameters") else params)

    @_OPTIMIZER_BUILDERS.register("sgd", help="SGD with momentum/Nesterov")
    def _sgd_builder(cfg: Any, *, params) -> Any:
        ocfg = cfg if isinstance(cfg, OptimizerConfig) else OptimizerConfig(**dict(cfg))
        ocfg.name = "sgd"
        return build_optimizer(ocfg, params_or_module=params if not hasattr(params, "parameters") else params)

    @_SCHEDULER_BUILDERS.register("cosine", help="CosineAnnealingLR")
    def _cosine_sched_builder(cfg: Any, optimizer, **kw) -> Any:
        scfg = cfg if isinstance(cfg, SchedulerConfig) else SchedulerConfig(**dict(cfg))
        scfg.name = "cosine"
        return build_scheduler(scfg, optimizer, **kw)

    @_SCHEDULER_BUILDERS.register("cosine_warm_restarts", help="CosineAnnealingWarmRestarts")
    def _cwr_sched_builder(cfg: Any, optimizer, **kw) -> Any:
        scfg = cfg if isinstance(cfg, SchedulerConfig) else SchedulerConfig(**dict(cfg))
        scfg.name = "cosine_warm_restarts"
        return build_scheduler(scfg, optimizer, **kw)

    @_SCHEDULER_BUILDERS.register("linear_warmup_cosine", help="Linear warmup then cosine (SequentialLR)")
    def _lw_cosine_sched_builder(cfg: Any, optimizer, **kw) -> Any:
        scfg = cfg if isinstance(cfg, SchedulerConfig) else SchedulerConfig(**dict(cfg))
        scfg.name = "linear_warmup_cosine"
        return build_scheduler(scfg, optimizer, **kw)

    @_SCHEDULER_BUILDERS.register("onecycle", help="OneCycleLR (step-wise)")
    def _onecycle_sched_builder(cfg: Any, optimizer, **kw) -> Any:
        scfg = cfg if isinstance(cfg, SchedulerConfig) else SchedulerConfig(**dict(cfg))
        scfg.name = "onecycle"
        return build_scheduler(scfg, optimizer, **kw)

    @_SCHEDULER_BUILDERS.register("exponential", help="ExponentialLR")
    def _exp_sched_builder(cfg: Any, optimizer, **kw) -> Any:
        scfg = cfg if isinstance(cfg, SchedulerConfig) else SchedulerConfig(**dict(cfg))
        scfg.name = "exponential"
        return build_scheduler(scfg, optimizer, **kw)

    @_SCHEDULER_BUILDERS.register("reduce_on_plateau", help="ReduceLROnPlateau (monitored)")
    def _rop_sched_builder(cfg: Any, optimizer, **kw) -> Any:
        scfg = cfg if isinstance(cfg, SchedulerConfig) else SchedulerConfig(**dict(cfg))
        scfg.name = "reduce_on_plateau"
        return build_scheduler(scfg, optimizer, **kw)


# Populate defaults on import
_auto_register_default_optimizers_and_schedulers()


# -----------------------------------------------------------------------------
# Convenience: dump registry contents (optional)
# -----------------------------------------------------------------------------

def debug_dump_registries() -> Dict[str, Dict[str, Optional[str]]]:
    """
    Returns a nested dict of registry names -> {key: help}.
    Useful for debugging or CLI commands like `spectramind registry --list`.
    """
    return {
        "models": _MODEL_BUILDERS.help_map(),
        "losses": _LOSS_BUILDERS.help_map(),
        "optimizers": _OPTIMIZER_BUILDERS.help_map(),
        "schedulers": _SCHEDULER_BUILDERS.help_map(),
    }
