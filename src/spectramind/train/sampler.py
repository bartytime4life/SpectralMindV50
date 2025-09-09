# src/spectramind/train/sampler.py
# =============================================================================
# SpectraMind V50 — Sampler Builders
# -----------------------------------------------------------------------------
# A composable set of PyTorch Samplers for:
#   • sequential / random (deterministic-per-epoch, optional epoch_size)
#   • weighted sampling (importance or class-balance; with/without replacement)
#   • stratified sampling by class labels (balanced per epoch)
#   • distributed (DDP) variants with proper set_epoch semantics:
#       - DistributedSampler (random/sequential)
#       - DistributedWeightedSampler
#       - DistributedStratifiedSampler (balanced across replicas)
#
# All builders are Hydra-friendly via Config dataclasses and a single
# `build_sampler(...)` entrypoint used by DataModule/DataLoader creation.
#
# Extras in this upgrade:
#   • epoch_size support for random/sequential to trim/extend each epoch
#   • per-replica samples for distributed samplers computed robustly
#   • weighted w/ replacement or w/o replacement
#   • optional subset_indices to restrict the dataset universe
#   • deterministic worker seeding helper (worker_init_fn_deterministic)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import os
import torch
from torch.utils.data import Sampler, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class SamplerConfig:
    """
    kind:
      - 'auto'             → DistributedSampler (if distributed) else random
      - 'sequential'       → deterministic sequential
      - 'random'           → deterministic random (no replacement)
      - 'weighted'         → WeightedRandomSampler; requires weights or labels
      - 'stratified'       → class-balanced epoch sampler; requires labels
      - 'distributed_*'    → explicit distributed variants:
                             'distributed_sequential', 'distributed_random',
                             'distributed_weighted', 'distributed_stratified'
    """
    kind: str = "auto"
    seed: int = 42

    # Weighted / stratified options
    replacement: bool = False
    num_samples: Optional[int] = None   # total samples per (global) epoch or per-replica if distributed
    # Optional: provide weights/labels name keys if not discoverable
    weights_attr: Optional[str] = None  # dataset.<attr> or dataset.<attr>() returning tensor/list
    labels_attr: Optional[str] = None   # dataset.<attr> or dataset.<attr>() returning tensor/list
    # For stratified: floor epochs or oversample to match max class count
    stratified_equalize: bool = True

    # Random / sequential extras
    epoch_size: Optional[int] = None  # if set, limits/extends epoch length (local or per-replica for DDP)

    # Distributed toggles
    distributed_shuffle: bool = True
    distributed_drop_last: bool = False


# -----------------------------------------------------------------------------
# Helpers to discover distributed world/rank and dataset labels/weights
# -----------------------------------------------------------------------------

def _discover_distributed() -> Tuple[int, int, bool]:
    """Return (world_size, rank, is_distributed)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size()
        rk = torch.distributed.get_rank()
        return ws, rk, True
    # Fallback to env
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rk = int(os.environ.get("RANK", "0"))
    return ws, rk, ws > 1


def _to_tensor(x: Any, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _maybe_call_attr(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        return None
    attr = getattr(obj, name)
    return attr() if callable(attr) else attr


def _get_labels(dataset: Dataset, cfg: SamplerConfig) -> Optional[torch.Tensor]:
    """Try common attributes or user-specified labels_attr."""
    candidates = []
    if cfg.labels_attr:
        candidates.append(cfg.labels_attr)
    # common dataset conventions
    candidates += ["labels", "targets", "y", "classes"]
    for nm in candidates:
        v = _maybe_call_attr(dataset, nm)
        if v is not None:
            t = _to_tensor(v).long().view(-1).cpu()
            if t.numel() == len(dataset):
                return t
    return None


def _get_weights(dataset: Dataset, cfg: SamplerConfig) -> Optional[torch.Tensor]:
    """Try explicit weights_attr or common 'weights'/'sample_weights' attributes; else derive from labels."""
    if cfg.weights_attr:
        v = _maybe_call_attr(dataset, cfg.weights_attr)
        if v is not None:
            return _to_tensor(v, dtype=torch.float32).view(-1).cpu()
    for nm in ["weights", "sample_weights"]:
        v = _maybe_call_attr(dataset, nm)
        if v is not None:
            return _to_tensor(v, dtype=torch.float32).view(-1).cpu()
    # derive from labels (inverse frequency) if available
    labels = _get_labels(dataset, cfg)
    if labels is not None:
        counts = torch.bincount(labels).clamp_min(1)
        inv = 1.0 / counts.float()
        return inv[labels]
    return None


def _subset_len(dataset: Dataset) -> int:
    if isinstance(dataset, Subset):
        return len(dataset.indices)  # type: ignore[attr-defined]
    return len(dataset)


# -----------------------------------------------------------------------------
# Core Samplers
# -----------------------------------------------------------------------------

class DeterministicRandomSampler(Sampler[int]):
    """
    Random permutation of indices per epoch with deterministic seed = base + epoch.
    Optional epoch_size allows truncation/extension of each epoch length locally.
    """
    def __init__(self, data_source: Dataset, base_seed: int = 42, epoch_size: Optional[int] = None):
        super().__init__(data_source)
        self.data_source = data_source
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.epoch_size = int(epoch_size) if epoch_size is not None else None

    def __len__(self) -> int:
        return self.epoch_size if self.epoch_size is not None else len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.data_source)
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        perm = torch.randperm(n, generator=g).tolist()
        if self.epoch_size is None:
            return iter(perm)
        es = self.epoch_size
        if es <= n:
            return iter(perm[:es])
        # extend by cycling
        reps = (es + n - 1) // n
        ext = (perm * reps)[:es]
        return iter(ext)


class DeterministicSequentialSampler(Sampler[int]):
    """Sequential indices; supports set_epoch for interface compatibility. Optional epoch_size."""
    def __init__(self, data_source: Dataset, epoch_size: Optional[int] = None):
        super().__init__(data_source)
        self.data_source = data_source
        self.epoch = 0
        self.epoch_size = int(epoch_size) if epoch_size is not None else None

    def __len__(self) -> int:
        return self.epoch_size if self.epoch_size is not None else len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.data_source)
        if self.epoch_size is None or self.epoch_size >= n:
            return iter(range(n))
        return iter(range(self.epoch_size))


class StratifiedBalancedSampler(Sampler[int]):
    """
    Epoch-wise class-balanced sampler. For each epoch, produces equal (or near-equal)
    number of samples from each class by oversampling minority classes if needed.
    """
    def __init__(self, labels: torch.Tensor, base_seed: int = 42, equalize: bool = True,
                 num_samples: Optional[int] = None):
        assert labels.ndim == 1, "labels must be 1-D tensor"
        self.labels = labels.long().cpu()
        self.base_seed = int(base_seed)
        self.equalize = bool(equalize)
        self.epoch = 0

        self.class_to_indices: Dict[int, torch.Tensor] = {}
        for c in torch.unique(self.labels, sorted=True).tolist():
            idx = (self.labels == c).nonzero(as_tuple=True)[0]
            self.class_to_indices[int(c)] = idx
        self.classes = sorted(self.class_to_indices.keys())

        # default epoch size: sum of max class count across classes (balanced)
        if num_samples is None:
            max_c = max(len(v) for v in self.class_to_indices.values())
            self.epoch_len = max_c * len(self.classes) if self.equalize else len(self.labels)
        else:
            self.epoch_len = int(num_samples)

    def __len__(self) -> int:
        return self.epoch_len

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)

        per_class_indices: List[int] = []
        if self.equalize:
            max_c = max(len(v) for v in self.class_to_indices.values())
            for c in self.classes:
                idxs = self.class_to_indices[c]
                choice = torch.randint(low=0, high=len(idxs), size=(max_c,), generator=g)
                per_class_indices.extend(idxs[choice].tolist())
        else:
            for c in self.classes:
                idxs = self.class_to_indices[c]
                per_class_indices.extend(idxs.tolist())

        perm = torch.randperm(len(per_class_indices), generator=g).tolist()
        out = [per_class_indices[i] for i in perm]
        if len(out) >= self.epoch_len:
            return iter(out[: self.epoch_len])
        reps = (self.epoch_len + len(out) - 1) // len(out)
        tiled = (out * reps)[: self.epoch_len]
        return iter(tiled)


class WeightedNoReplacementSampler(Sampler[int]):
    """
    Weighted sampling without replacement (approximation using scores + top-k).
    When replacement=False in torch.multinomial is not feasible for huge N, this provides a
    simple, deterministic alternative per epoch.
    """
    def __init__(self, weights: torch.Tensor, base_seed: int = 42, num_samples: Optional[int] = None):
        super().__init__(None)
        w = torch.as_tensor(weights, dtype=torch.float32).clamp_min(1e-12).cpu()
        self.prob = w / w.sum()
        self.N = w.numel()
        self.epoch = 0
        self.base_seed = int(base_seed)
        self.num_samples = int(num_samples) if num_samples is not None else self.N

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        # Gumbel trick for weighted without replacement: score = log(p) + gumbel
        # then take top-k
        gumbel = -torch.log(-torch.log(torch.rand(self.N, generator=g)))
        scores = torch.log(self.prob) + gumbel
        topk = torch.topk(scores, k=min(self.num_samples, self.N)).indices.tolist()
        return iter(topk)


class DistributedWeightedSampler(Sampler[int]):
    """
    Distributed variant of WeightedRandomSampler.
    - Partitions the global weighted stream across replicas deterministically.
    - Replacement=True behavior (stable for large datasets).
    """
    def __init__(
        self,
        weights: torch.Tensor,     # [N]
        num_replicas: int,
        rank: int,
        base_seed: int = 42,
        num_samples: Optional[int] = None,
    ):
        assert weights.ndim == 1, "weights must be 1-D"
        self.weights = weights.float().cpu()
        self.N = int(self.weights.numel())
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.base_seed = int(base_seed)
        self.epoch = 0

        # per-replica samples per epoch
        self.num_samples = int(num_samples) if num_samples is not None else int(math.ceil(self.N / self.num_replicas))
        s = self.weights.sum().clamp_min(1e-12)
        self.prob = (self.weights / s).view(-1)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        # make replica-specific seed for determinism
        g.manual_seed(self.base_seed + self.epoch * 911 + self.rank)
        idx = torch.multinomial(self.prob, num_samples=self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())


class DistributedStratifiedSampler(Sampler[int]):
    """
    Distributed class-balanced sampler.
    Each replica draws ~num_samples_per_replica examples per epoch, balanced across classes.
    """
    def __init__(
        self,
        labels: torch.Tensor,         # [N]
        num_replicas: int,
        rank: int,
        base_seed: int = 42,
        num_samples: Optional[int] = None,
    ):
        super().__init__(None)
        self.labels = labels.long().cpu()
        self.N = int(self.labels.numel())
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.base_seed = int(base_seed)
        self.epoch = 0

        # per-replica samples per epoch (default ~ N/replicas)
        self.num_samples = int(num_samples) if num_samples is not None else int(math.ceil(self.N / self.num_replicas))

        self.class_to_indices: Dict[int, torch.Tensor] = {}
        for c in torch.unique(self.labels, sorted=True).tolist():
            idxs = (self.labels == c).nonzero(as_tuple=True)[0]
            self.class_to_indices[int(c)] = idxs

        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch * 977 + self.rank)

        per_class = max(1, self.num_samples // self.num_classes)
        picks: List[int] = []
        for c in self.classes:
            idxs = self.class_to_indices[c]
            # sample per_class with replacement (stable across small classes)
            choice = torch.randint(low=0, high=len(idxs), size=(per_class,), generator=g)
            picks.extend(idxs[choice].tolist())

        # If we still need more to hit num_samples, fill from all classes at random
        remaining = self.num_samples - len(picks)
        if remaining > 0:
            all_idx = torch.arange(self.N)
            ext = torch.randint(low=0, high=self.N, size=(remaining,), generator=g)
            picks.extend(all_idx[ext].tolist())

        # shuffle final picks (local)
        perm = torch.randperm(len(picks), generator=g).tolist()
        out = [picks[i] for i in perm]
        return iter(out[: self.num_samples])


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

def build_sampler(
    dataset: Dataset,
    cfg: SamplerConfig,
    *,
    distributed_override: Optional[bool] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    subset_indices: Optional[Sequence[int]] = None,
) -> Sampler[int]:
    """
    Build a sampler for the dataset given config.
    If distributed_override is None, we auto-detect distributed.

    Args:
      subset_indices: if provided, restrict sampling universe to this index subset.
                      Useful when DataLoader uses Subset directly or when you want to share
                      a common sampler across multiple Subsets.
    """
    ws, rk, is_dist = _discover_distributed()
    if world_size is not None:
        ws = world_size
    if rank is not None:
        rk = rank
    if distributed_override is not None:
        is_dist = bool(distributed_override)

    kind = cfg.kind.lower()

    # optional subset universe
    N = _subset_len(dataset)
    universe = torch.arange(N) if subset_indices is None else _to_tensor(subset_indices, dtype=torch.long).view(-1)

    # --- Auto mode
    if kind == "auto":
        if is_dist:
            return DistributedSampler(
                dataset,
                num_replicas=ws,
                rank=rk,
                shuffle=cfg.distributed_shuffle,
                seed=cfg.seed,
                drop_last=cfg.distributed_drop_last,
            )
        # local random
        return DeterministicRandomSampler(dataset, base_seed=cfg.seed, epoch_size=cfg.epoch_size)

    # --- Non-distributed basics
    if kind == "sequential":
        return DeterministicSequentialSampler(dataset, epoch_size=cfg.epoch_size)

    if kind == "random":
        return DeterministicRandomSampler(dataset, base_seed=cfg.seed, epoch_size=cfg.epoch_size)

    if kind == "weighted":
        weights = _get_weights(dataset, cfg)
        if weights is None:
            raise ValueError("Weighted sampler requires `weights` or discoverable labels to derive weights.")
        weights = weights[universe] if subset_indices is not None else weights
        num_samples = cfg.num_samples or (cfg.epoch_size or len(universe))
        if cfg.replacement:
            return torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=num_samples,
                replacement=True,
            )
        # without replacement (approximation)
        return WeightedNoReplacementSampler(weights=weights, base_seed=cfg.seed, num_samples=num_samples)

    if kind == "stratified":
        labels = _get_labels(dataset, cfg)
        if labels is None:
            raise ValueError("Stratified sampler requires labels.")
        labels = labels[universe] if subset_indices is not None else labels
        num_samples = cfg.num_samples or (cfg.epoch_size or len(universe))
        return StratifiedBalancedSampler(
            labels=labels,
            base_seed=cfg.seed,
            equalize=cfg.stratified_equalize,
            num_samples=num_samples,
        )

    # --- Distributed variants
    if kind in ("distributed_sequential", "distributed_random"):
        shuffle = cfg.distributed_shuffle if kind.endswith("random") else False
        return DistributedSampler(
            dataset,
            num_replicas=ws,
            rank=rk,
            shuffle=shuffle,
            seed=cfg.seed,
            drop_last=cfg.distributed_drop_last,
        )

    if kind == "distributed_weighted":
        weights = _get_weights(dataset, cfg)
        if weights is None:
            raise ValueError("Distributed weighted sampler requires `weights` or discoverable labels to derive weights.")
        weights = weights[universe] if subset_indices is not None else weights
        # compute per-replica epoch length
        if cfg.num_samples is not None:
            samples_per_replica = max(1, int(math.ceil(cfg.num_samples / ws)))
        elif cfg.epoch_size is not None:
            samples_per_replica = max(1, int(math.ceil(cfg.epoch_size / ws)))
        else:
            samples_per_replica = int(math.ceil(len(universe) / ws))
        return DistributedWeightedSampler(
            weights=weights,
            num_replicas=ws,
            rank=rk,
            base_seed=cfg.seed,
            num_samples=samples_per_replica,
        )

    if kind == "distributed_stratified":
        labels = _get_labels(dataset, cfg)
        if labels is None:
            raise ValueError("Distributed stratified sampler requires labels.")
        labels = labels[universe] if subset_indices is not None else labels
        if cfg.num_samples is not None:
            samples_per_replica = max(1, int(math.ceil(cfg.num_samples / ws)))
        elif cfg.epoch_size is not None:
            samples_per_replica = max(1, int(math.ceil(cfg.epoch_size / ws)))
        else:
            samples_per_replica = int(math.ceil(len(universe) / ws))
        return DistributedStratifiedSampler(
            labels=labels,
            num_replicas=ws,
            rank=rk,
            base_seed=cfg.seed,
            num_samples=samples_per_replica,
        )

    raise ValueError(f"Unsupported sampler kind: {cfg.kind!r}")


# -----------------------------------------------------------------------------
# Lightning/PL compatibility helper
# -----------------------------------------------------------------------------

def set_epoch_if_possible(sampler: Sampler[int], epoch: int) -> None:
    """
    Lightning calls set_epoch on dataloaders' samplers each epoch.
    Provide a safe no-op if sampler doesn't expose it.
    """
    if hasattr(sampler, "set_epoch") and callable(getattr(sampler, "set_epoch")):
        sampler.set_epoch(epoch)


# -----------------------------------------------------------------------------
# Deterministic worker seeding (optional)
# -----------------------------------------------------------------------------

def worker_init_fn_deterministic(worker_id: int) -> None:
    """
    Deterministic seeding for DataLoader workers. Use as:
        DataLoader(..., worker_init_fn=worker_init_fn_deterministic)
    """
    # Same recipe as PL's seed_everything but localized to worker
    base_seed = torch.initial_seed() % 2**32  # each worker gets unique base from DataLoader
    g = torch.Generator()
    g.manual_seed(base_seed + worker_id)
    # Seed python, numpy if needed
    try:
        import random
        random.seed(base_seed + worker_id)
    except Exception:
        pass
    try:
        import numpy as np
        np.random.seed((base_seed + worker_id) % (2**32 - 1))
    except Exception:
        pass