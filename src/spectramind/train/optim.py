# src/spectramind/train/optim.py
# =============================================================================
# SpectraMind V50 — Optimizer & Scheduler Builders
# -----------------------------------------------------------------------------
# - Hydra-friendly creation of optimizers/schedulers
# - Lightning-compatible return structures
# - Robust param grouping for decoupled weight decay (bias/norm exclusions)
# - Optional name-based regex/substring include/exclude filters
# - Optional per-group LR multipliers (e.g., layer-wise decay or head boost)
# - Supports common schedulers: cosine, warm restarts, linear-warmup+cosine,
#   one-cycle, exponential, reduce-on-plateau
# - Optional optimizers if available (lazy imports):
#     * AdamW 8-bit  (bitsandbytes)     name: "adamw8bit"
#     * Lion        (lion-pytorch/flash-attn)  name: "lion"
#     * Adafactor   (transformers)       name: "adafactor"
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import math
import re
import warnings

import torch
from torch import nn
from torch.optim import Optimizer


# -----------------------------------------------------------------------------
# Dataclasses (optional typed configs; Hydra can instantiate these)
# -----------------------------------------------------------------------------

@dataclass
class ParamGroupsConfig:
    # Any param whose name contains one of these substrings will be excluded from weight decay
    decay_exclude_substrings: Tuple[str, ...] = (
        "bias",
        "bn",        # BatchNorm
        "norm",      # LayerNorm/GroupNorm
        "ln",        # LayerNorm
        "embedding", # nn.Embedding
        "emb",       # alias
    )
    # Optional regex filters (applied after substrings)
    decay_exclude_regex: Tuple[str, ...] = tuple()
    decay_include_regex: Tuple[str, ...] = tuple()  # if set, only these receive weight decay

    # Optional LR multipliers per name pattern (apply to param-wise 'lr' in group)
    # Ordered application: first match wins. Patterns can be substrings or regex (prefix 're:')
    lr_multipliers: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class OptimizerConfig:
    name: str = "adamw"          # 'adamw' | 'adam' | 'sgd' | 'adamw8bit' | 'lion' | 'adafactor'
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)     # Adam/AdamW
    eps: float = 1e-8
    momentum: float = 0.9                         # SGD
    nesterov: bool = True                         # SGD
    fused: bool = False                           # use fused Adam/AdamW if available

    # param group handling
    param_groups: ParamGroupsConfig = field(default_factory=ParamGroupsConfig)

    # Adafactor (transformers) options (if chosen)
    adafactor_relative_step: bool = True
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = True


@dataclass
class SchedulerConfig:
    name: Optional[str] = None   # None | 'cosine' | 'cosine_warm_restarts' | 'linear_warmup_cosine'
                                 #      | 'onecycle' | 'exponential' | 'reduce_on_plateau'
    # Generic Lightning metadata
    interval: str = "epoch"      # 'epoch' | 'step'
    frequency: int = 1
    monitor: str = "val_loss"    # for plateaus / monitored schedulers
    mode: str = "min"            # for plateaus

    # Cosine
    T_max: Optional[int] = None  # epochs or steps depending on interval
    eta_min: float = 0.0

    # Warm restarts
    T_0: int = 10
    T_mult: int = 2

    # Linear warmup + cosine
    warmup_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None  # alternative to warmup_steps
    total_steps: Optional[int] = None     # steps for interval='step' (else computed from E x S/E)

    # OneCycle
    max_lr: Optional[float] = None
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4

    # Exponential
    gamma: float = 0.99

    # ReduceLROnPlateau
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    cooldown: int = 0
    min_lr: float = 0.0


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def maybe_scale_lr_with_batch_size(lr: float, actual_batch_size: int,
                                   reference_batch_size: int = 256) -> float:
    """
    Optional helper: scale LR linearly w.r.t. batch size.
    """
    if actual_batch_size <= 0 or reference_batch_size <= 0:
        return lr
    return lr * (actual_batch_size / float(reference_batch_size))


def _is_fused_supported() -> bool:
    # Adam/AdamW fused are available in newer PyTorch builds.
    # We detect via optimizer signature 'fused' in constructor.
    try:
        from inspect import signature
        return "fused" in signature(torch.optim.AdamW).parameters
    except Exception:
        return False


def _param_name(m: nn.Module, p: nn.Parameter) -> str:
    # Attempt to find the parameter name for exclusion matching; fallback to ""
    for n, p0 in m.named_parameters():
        if p is p0:
            return n
    return ""


def _match_any_regex(name: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if not pat:
            continue
        if re.search(pat, name):
            return True
    return False


def _lr_multiplier_for(name: str, multipliers: List[Tuple[str, float]]) -> Optional[float]:
    """
    Return first matching multiplier for param name.
    Pattern semantics:
      • "re:<pattern>" → regex match
      • otherwise      → substring match
    """
    lname = name.lower()
    for pat, mult in multipliers:
        if pat.startswith("re:"):
            if re.search(pat[3:], lname):
                return float(mult)
        else:
            if pat.lower() in lname:
                return float(mult)
    return None


def split_param_groups_for_weight_decay(
    module: nn.Module,
    weight_decay: float,
    exclude_substrings: Iterable[str],
    exclude_regex: Iterable[str] = (),
    include_regex: Iterable[str] = (),
    lr_multipliers: Optional[List[Tuple[str, float]]] = None,
    base_lr: float = 1e-3,
) -> List[Dict[str, Any]]:
    """
    Builds param groups with decoupled weight decay and optional per-group LR multipliers.

    Priority:
      1) include_regex (if provided): only these get decay (others -> no_decay)
      2) exclude_substrings / exclude_regex: go to no_decay
      3) default: go to decay
    """
    decay, no_decay = [], []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        name = _param_name(module, p)

        # include_regex wins: only names matching include_regex get decay
        if include_regex:
            if _match_any_regex(name, include_regex):
                decay.append((name, p))
            else:
                no_decay.append((name, p))
            continue

        # explicit excludes
        lname = name.lower()
        if any(sub in lname for sub in exclude_substrings) or _match_any_regex(name, exclude_regex):
            no_decay.append((name, p))
        else:
            decay.append((name, p))

    groups: List[Dict[str, Any]] = []

    def _make_group(params_with_names: List[Tuple[str, nn.Parameter]], wd: float) -> Dict[str, Any]:
        # Apply optional LR multipliers
        if lr_multipliers:
            # build sub-groups by multiplier
            bucket: Dict[float, List[nn.Parameter]] = {}
            for name, p in params_with_names:
                mult = _lr_multiplier_for(name, lr_multipliers)
                if mult is None:
                    mult = 1.0
                bucket.setdefault(mult, []).append(p)
            # emit one group per multiplier
            out: List[Dict[str, Any]] = []
            for mult, plist in bucket.items():
                out.append({"params": plist, "weight_decay": wd, "lr": base_lr * mult})
            # merging these groups into caller list is easier at call site
            return {"_subgroups": out}  # sentinel
        else:
            return {"params": [p for _, p in params_with_names], "weight_decay": wd}

    if decay:
        g = _make_group(decay, weight_decay)
        if "_subgroups" in g:
            groups.extend(g["_subgroups"])  # type: ignore[index]
        else:
            groups.append(g)
    if no_decay:
        g = _make_group(no_decay, 0.0)
        if "_subgroups" in g:
            groups.extend(g["_subgroups"])  # type: ignore[index]
        else:
            groups.append(g)

    return groups if groups else [{"params": list(module.parameters()), "weight_decay": weight_decay, "lr": base_lr}]


# -----------------------------------------------------------------------------
# Optimizer builders
# -----------------------------------------------------------------------------

def build_optimizer(
    cfg: OptimizerConfig,
    params_or_module: Union[Iterable[nn.Parameter], nn.Module],
) -> Optimizer:
    """
    Create an optimizer from config, supporting AdamW/Adam/SGD and optional extras.
    If a Module is passed, decoupled weight-decay param groups are created automatically.
    """
    # param groups
    if isinstance(params_or_module, nn.Module):
        param_groups = split_param_groups_for_weight_decay(
            params_or_module,
            cfg.weight_decay,
            cfg.param_groups.decay_exclude_substrings,
            exclude_regex=cfg.param_groups.decay_exclude_regex,
            include_regex=cfg.param_groups.decay_include_regex,
            lr_multipliers=cfg.param_groups.lr_multipliers or None,
            base_lr=cfg.lr,
        )
    else:
        # hold weight_decay at cfg.weight_decay; no name-level logic available
        param_groups = [{"params": list(params_or_module), "weight_decay": cfg.weight_decay, "lr": cfg.lr}]

    name = cfg.name.lower()
    fused = bool(cfg.fused and _is_fused_supported())

    # --- Torch stock optimizers
    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=0.0,  # weight decay handled in groups
            fused=fused if _is_fused_supported() else False,
        )

    if name == "adam":
        kw = dict(lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
        if fused and _is_fused_supported():
            kw["fused"] = True
        return torch.optim.Adam(param_groups, **kw)  # type: ignore[arg-type]

    if name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )

    # --- Optional: AdamW 8-bit (bitsandbytes)
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore
            return bnb.optim.AdamW8bit(  # type: ignore[attr-defined]
                param_groups,
                lr=cfg.lr,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=0.0,
            )
        except Exception as e:
            warnings.warn(f"[optim] AdamW8bit requested but bitsandbytes not available: {e}. Falling back to AdamW.")
            return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)

    # --- Optional: Lion
    if name == "lion":
        opt_cls = None
        try:
            from lion_pytorch import Lion as LionOpt  # type: ignore
            opt_cls = LionOpt
        except Exception:
            try:
                # some packages (flash-attn) also export Lion
                from flash_attn.optim import Lion as LionOpt  # type: ignore
            except Exception:
                pass
        if opt_cls is None:
            warnings.warn("[optim] Lion requested but not installed; falling back to AdamW.")
            return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)
        # Lion typically accepts weight_decay (decoupled) and betas= (β1, β2)
        return opt_cls(param_groups, lr=cfg.lr, betas=cfg.betas, weight_decay=0.0)

    # --- Optional: Adafactor (transformers)
    if name == "adafactor":
        try:
            from transformers import Adafactor  # type: ignore
            return Adafactor(
                param_groups,
                lr=cfg.lr,
                scale_parameter=cfg.adafactor_scale_parameter,
                relative_step=cfg.adafactor_relative_step,
                warmup_init=cfg.adafactor_warmup_init,
                weight_decay=0.0,
            )
        except Exception as e:
            warnings.warn(f"[optim] Adafactor requested but transformers not available: {e}. Falling back to AdamW.")
            return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)

    raise ValueError(f"Unsupported optimizer: {cfg.name}")


# -----------------------------------------------------------------------------
# Scheduler builders
# -----------------------------------------------------------------------------

def _linear_warmup_then_cosine(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min: float = 0.0,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Compose LinearLR(warmup) -> CosineAnnealingLR using SequentialLR.
    """
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warm = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    T_max = max(1, total_steps - warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup_steps])


def build_total_steps(
    *,
    steps_per_epoch: Optional[int] = None,
    max_epochs: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> Optional[int]:
    """
    Compute total_steps if possible. Used by step-wise schedulers (OneCycle, linear_warmup_cosine).
    """
    if total_steps is not None:
        return total_steps
    if steps_per_epoch is not None and max_epochs is not None:
        return steps_per_epoch * max_epochs
    return None


def build_scheduler(
    cfg: SchedulerConfig,
    optimizer: Optimizer,
    *,
    steps_per_epoch: Optional[int] = None,
    max_epochs: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Create a scheduler config dict that Lightning can consume directly.
    Returns None if cfg.name is None.
    """
    if not cfg.name:
        return None

    name = cfg.name.lower()
    interval = cfg.interval
    assert interval in ("epoch", "step"), f"Invalid scheduler interval: {interval}"

    # infer steps if needed
    total_steps = build_total_steps(steps_per_epoch=steps_per_epoch, max_epochs=max_epochs, total_steps=total_steps)

    if name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        # T_max in epochs or steps depending on interval
        T_max = cfg.T_max
        if T_max is None:
            if interval == "epoch":
                T_max = max(1, max_epochs or 1)
            else:
                T_max = max(1, total_steps or 1)
        sched = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.eta_min)

    elif name == "cosine_warm_restarts":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        sched = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min
        )

    elif name == "linear_warmup_cosine":
        # compute warmup/total steps
        if interval != "step":
            raise ValueError("linear_warmup_cosine requires interval='step'")
        if total_steps is None:
            raise ValueError("linear_warmup_cosine requires total_steps (steps_per_epoch*max_epochs)")
        if cfg.warmup_steps is None:
            if cfg.warmup_ratio is not None:
                warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
            else:
                raise ValueError("Provide warmup_steps or warmup_ratio for linear_warmup_cosine")
        else:
            warmup_steps = cfg.warmup_steps
        sched = _linear_warmup_then_cosine(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=cfg.eta_min,
        )

    elif name == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR
        if interval != "step":
            raise ValueError("OneCycleLR requires interval='step'")
        if total_steps is None:
            if steps_per_epoch is None or max_epochs is None:
                raise ValueError("OneCycleLR needs total_steps or (steps_per_epoch and max_epochs)")
            total_steps = steps_per_epoch * max_epochs
        max_lr = cfg.max_lr if cfg.max_lr is not None else max(g.get("lr", 0.0) for g in optimizer.param_groups)
        sched = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=cfg.pct_start,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor,
            anneal_strategy="cos",
            cycle_momentum=any("momentum" in g for g in optimizer.param_groups),
        )

    elif name == "exponential":
        from torch.optim.lr_scheduler import ExponentialLR
        sched = ExponentialLR(optimizer, gamma=cfg.gamma)

    elif name == "reduce_on_plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        sched = ReduceLROnPlateau(
            optimizer,
            mode=cfg.mode,
            factor=cfg.factor,
            patience=cfg.patience,
            threshold=cfg.threshold,
            cooldown=cfg.cooldown,
            min_lr=cfg.min_lr,
            verbose=False,
        )
        return {
            "scheduler": sched,
            "monitor": cfg.monitor,
            "interval": "epoch",
            "frequency": cfg.frequency,
        }

    else:
        raise ValueError(f"Unsupported scheduler: {cfg.name}")

    return {
        "scheduler": sched,
        "interval": interval,
        "frequency": cfg.frequency,
        # 'monitor' only needed if ReduceLROnPlateau or other monitored schedulers
    }


# -----------------------------------------------------------------------------
# Lightning convenience: single-call configure_optimizers
# -----------------------------------------------------------------------------

def configure_optimizers_for(
    module: nn.Module,
    opt_cfg: OptimizerConfig,
    sch_cfg: Optional[SchedulerConfig] = None,
    *,
    steps_per_epoch: Optional[int] = None,
    max_epochs: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> Any:
    """
    Build optimizer (and optional scheduler) in a Lightning-friendly way.
    Returns optimizer or (opt, sched_dict) as expected by Lightning.
    """
    optimizer = build_optimizer(opt_cfg, module)

    if sch_cfg is None or sch_cfg.name is None:
        return optimizer

    sched_dict = build_scheduler(
        sch_cfg, optimizer,
        steps_per_epoch=steps_per_epoch,
        max_epochs=max_epochs,
        total_steps=total_steps,
    )
    if sched_dict is None:
        return optimizer
    return [optimizer], [sched_dict]