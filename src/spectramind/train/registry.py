# src/spectramind/train/registry.py
# =============================================================================
# SpectraMind V50 — Lightweight Registry for Builders
# -----------------------------------------------------------------------------
# Exposes decorators & getters for:
#   • Models:     (cfg, *, criterion=None) -> LightningModule
#   • Losses:     (cfg)                    -> callable / nn.Module / loss_fn
#   • Optimizers: (cfg, *, params)         -> torch.optim.Optimizer
#   • Schedulers: (cfg, optimizer, **kw)   -> torch.optim.lr_scheduler | Lightning dict
#
# Notes:
# - Used by train/train.py: get_model_builder, get_loss_builder,
#   get_optimizer_builder, get_scheduler_builder.
# - Includes default optimizer/scheduler builders backed by train.optim.
# - Model & Loss registries start empty; register yours via decorators or `.add()`.
# - Thread-safe, alias-friendly, Hydra-friendly (builders accept dict-like cfg).
# =============================================================================

from __future__ import annotations

import difflib
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Generic registry
# -----------------------------------------------------------------------------

@dataclass
class _Entry:
    name: str
    fn: Callable[..., Any]
    help: Optional[str] = None
    aliases: Tuple[str, ...] = ()


class Registry:
    """
    Small, thread-safe registry with:
      • decorator-based and programmatic registration
      • alias support (multiple names -> same builder)
      • 'did you mean' suggestions
      • simple introspection helpers
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._store: Dict[str, _Entry] = {}
        self._lock = RLock()

    # Decorator-based registration
    def register(
        self,
        name: Optional[str] = None,
        *,
        help: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
    ):
        """
        Usage:
            @REGISTRY.register("my_key", help="...", aliases=["my_alias"])
            def builder(...): ...
        """
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            key = (name or fn.__name__).lower()
            with self._lock:
                self._ensure_unique(key)
                ent = _Entry(name=key, fn=fn, help=help, aliases=tuple(a.lower() for a in (aliases or ())))
                self._store[key] = ent
                for al in ent.aliases:
                    self._ensure_unique(al)
                    self._store[al] = ent
            return fn
        return _decorator

    # Programmatic registration
    def add(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        help: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
    ) -> None:
        key = name.lower()
        with self._lock:
            self._ensure_unique(key)
            ent = _Entry(name=key, fn=fn, help=help, aliases=tuple(a.lower() for a in (aliases or ())))
            self._store[key] = ent
            for al in ent.aliases:
                self._ensure_unique(al)
                self._store[al] = ent

    def _ensure_unique(self, key: str) -> None:
        if key in self._store:
            raise KeyError(f"{self._kind} '{key}' already registered.")

    def get(self, name: str) -> Callable[..., Any]:
        key = (name or "").lower()
        with self._lock:
            if key not in self._store:
                self._raise_not_found(key)
            return self._store[key].fn

    def has(self, name: str) -> bool:
        return name.lower() in self._store

    def remove(self, name: str) -> None:
        """Remove a key (and keep alias entries pointing to the same fn intact)."""
        key = name.lower()
        with self._lock:
            if key not in self._store:
                self._raise_not_found(key)
            del self._store[key]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def names(self) -> Iterable[str]:
        return list(self._store.keys())

    def help_map(self) -> Dict[str, Optional[str]]:
        # Only emit canonical entries once (avoid listing aliases twice)
        seen: set[str] = set()
        out: Dict[str, Optional[str]] = {}
        for k, ent in self._store.items():
            if ent.fn in seen:
                continue
            out[ent.name] = ent.help
            seen.add(ent.fn)  # type: ignore[arg-type]
        return out

    def describe(self, name: str) -> str:
        """
        Return a human-readable description for a builder name (includes aliases).
        """
        key = name.lower()
        if key not in self._store:
            self._raise_not_found(key)
        ent = self._store[key]
        alias_str = f" (aliases: {', '.join(ent.aliases)})" if ent.aliases else ""
        help_str = f": {ent.help}" if ent.help else ""
        return f"{self._kind} '{ent.name}'{alias_str}{help_str}"

    def _raise_not_found(self, key: str) -> None:
        choices = sorted(set(self._store.keys()))
        msg = [f"Unknown {self._kind} '{key}'."]
        if choices:
            msg.append(f"Available {self._kind}s: {', '.join(choices)}")
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

def register_model(name: Optional[str] = None, *, help: Optional[str] = None, aliases: Optional[Sequence[str]] = None):
    """
    Register a model builder:
        signature: builder(cfg, *, criterion=None) -> LightningModule
    """
    return _MODEL_BUILDERS.register(name, help=help, aliases=aliases)


def register_loss(name: Optional[str] = None, *, help: Optional[str] = None, aliases: Optional[Sequence[str]] = None):
    """
    Register a loss builder:
        signature: builder(cfg) -> criterion (callable/nn.Module)
    """
    return _LOSS_BUILDERS.register(name, help=help, aliases=aliases)


def register_optimizer(name: Optional[str] = None, *, help: Optional[str] = None, aliases: Optional[Sequence[str]] = None):
    """
    Register an optimizer builder:
        signature: builder(cfg, *, params) -> torch.optim.Optimizer
    """
    return _OPTIMIZER_BUILDERS.register(name, help=help, aliases=aliases)


def register_scheduler(name: Optional[str] = None, *, help: Optional[str] = None, aliases: Optional[Sequence[str]] = None):
    """
    Register a scheduler builder:
        signature: builder(cfg, optimizer, **kw) -> scheduler or Lightning dict
    """
    return _SCHEDULER_BUILDERS.register(name, help=help, aliases=aliases)


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

    # Optimizers
    @_OPTIMIZER_BUILDERS.register("adamw", help="AdamW with decoupled weight decay groups", aliases=("adamw_torch",))
    def _adamw_builder(cfg: Any, *, params) -> Any:
        ocfg = cfg if isinstance(cfg, OptimizerConfig) else OptimizerConfig(**dict(cfg))
        return build_optimizer(ocfg, params_or_module=params if not hasattr(params, "parameters") else params)

    @_OPTIMIZER_BUILDERS.register("adam", help="Adam (decoupled decay groups)", aliases=("adam_torch",))
    def _adam_builder(cfg: Any, *, params) -> Any:
        ocfg = cfg if isinstance(cfg, OptimizerConfig) else OptimizerConfig(**dict(cfg))
        ocfg.name = "adam"
        return build_optimizer(ocfg, params_or_module=params if not hasattr(params, "parameters") else params)

    @_OPTIMIZER_BUILDERS.register("sgd", help="SGD with momentum/Nesterov", aliases=("sgd_torch",))
    def _sgd_builder(cfg: Any, *, params) -> Any:
        ocfg = cfg if isinstance(cfg, OptimizerConfig) else OptimizerConfig(**dict(cfg))
        ocfg.name = "sgd"
        return build_optimizer(ocfg, params_or_module=params if not hasattr(params, "parameters") else params)

    # Optional extras present in optim.py; if user selects them and deps exist, they're used.
    @_OPTIMIZER_BUILDERS.register("adamw8bit", help="AdamW 8-bit (bitsandbytes)")
    def _adamw8_builder(cfg: Any, *, params) -> Any:
        ocfg = cfg if isinstance(cfg, OptimizerConfig) else OptimizerConfig(**dict(cfg))
        ocfg.name = "adamw8bit"
        return build_optimizer(ocfg, params_or_module=params if not hasattr(params, "parameters") else params)

    @_OPTIMIZER_BUILDERS.register("lion", help="Lion optimizer (lion-pytorch/flash-attn)")
    def _lion_builder(cfg: Any, *, params) -> Any:
        ocfg = cfg if isinstance(cfg, OptimizerConfig) else OptimizerConfig(**dict(cfg))
        ocfg.name = "lion"
        return build_optimizer(ocfg, params_or_module=params if not hasattr(params, "parameters") else params)

    @_OPTIMIZER_BUILDERS.register("adafactor", help="Adafactor (transformers)")
    def _adafactor_builder(cfg: Any, *, params) -> Any:
        ocfg = cfg if isinstance(cfg, OptimizerConfig) else OptimizerConfig(**dict(cfg))
        ocfg.name = "adafactor"
        return build_optimizer(ocfg, params_or_module=params if not hasattr(params, "parameters") else params)

    # Schedulers
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