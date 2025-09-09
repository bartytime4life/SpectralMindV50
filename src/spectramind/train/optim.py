# src/spectramind/train/optim.py
# =============================================================================
# SpectraMind V50 — Optimizer & Scheduler Builders
# -----------------------------------------------------------------------------
# - Hydra-friendly creation of optimizers/schedulers
# - Lightning-compatible return structures
# - Robust param grouping for decoupled weight decay (bias/norm exclusions)
# - Supports common schedulers: cosine, warm restarts, linear-warmup+cosine,
#   one-cycle, exponential, reduce-on-plateau
# - Optional fused optimizers if available (PyTorch >= 2.0)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import math
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
        "emb",       # common alias
    )


@dataclass
class OptimizerConfig:
    name: str = "adamw"          # 'adamw' | 'adam' | 'sgd'
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)     # Adam/AdamW
    eps: float = 1e-8
    momentum: float = 0.9                         # SGD
    nesterov: bool = True                         # SGD
    fused: bool = False                           # use fused Adam/AdamW if available
    # param group handling
    param_groups: ParamGroupsConfig = field(default_factory=ParamGroupsConfig)


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
    total_steps: Optional[int] = None     # steps for interval='step', else computed from E x S/E
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
        if "fused" in signature(torch.optim.AdamW).parameters:
            return True
    except Exception:
        pass
    return False


def _param_name(m: nn.Module, p: nn.Parameter) -> str:
    # Attempt to find the parameter name for exclusion matching; fallback to repr index
    for n, p0 in m.named_parameters():
        if p is p0:
            return n
    return ""


def split_param_groups_for_weight_decay(
    module: nn.Module,
    weight_decay: float,
    exclude_substrings: Iterable[str],
) -> List[Dict[str, Any]]:
    """
    Builds 2 param groups: one with weight decay, one without, using substring-based exclusion.
    """
    decay, no_decay = [], []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        name = _param_name(module, p)
        if any(sub in name.lower() for sub in exclude_substrings):
            no_decay.append(p)
        else:
            decay.append(p)

    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups if groups else [{"params": module.parameters(), "weight_decay": weight_decay}]


# -----------------------------------------------------------------------------
# Optimizer builders
# -----------------------------------------------------------------------------

def build_optimizer(
    cfg: OptimizerConfig,
    params_or_module: Union[Iterable[nn.Parameter], nn.Module],
) -> Optimizer:
    """
    Create an optimizer from config, supporting AdamW/Adam/SGD and fused variants.
    If a Module is passed, decoupled weight-decay param groups are created automatically.
    """
    # param groups
    if isinstance(params_or_module, nn.Module):
        param_groups = split_param_groups_for_weight_decay(
            params_or_module,
            cfg.weight_decay,
            cfg.param_groups.decay_exclude_substrings,
        )
    else:
        param_groups = [{"params": list(params_or_module), "weight_decay": cfg.weight_decay}]

    name = cfg.name.lower()
    fused = bool(cfg.fused and _is_fused_supported())

    if name == "adamw":
        opt = torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=0.0,  # weight decay handled in groups
            fused=fused if _is_fused_supported() else False,
        )
    elif name == "adam":
        # Adam ignores weight_decay (decoupled already)
        # Some builds accept 'fused' too
        kw = dict(lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
        if fused and _is_fused_supported():
            kw["fused"] = True
        opt = torch.optim.Adam(param_groups, **kw)  # type: ignore
    elif name == "sgd":
        opt = torch.optim.SGD(
            param_groups,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.name}")

    return opt


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
    if total_steps is None and steps_per_epoch is not None and max_epochs is not None:
        total_steps = steps_per_epoch * max_epochs

    sched: Any
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
        max_lr = cfg.max_lr if cfg.max_lr is not None else max(g["lr"] for g in optimizer.param_groups)
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
