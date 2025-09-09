# src/spectramind/train/sampler.py
# =============================================================================
# SpectraMind V50 — Sampler Builders
# -----------------------------------------------------------------------------
# A composable set of PyTorch Samplers for:
#   • sequential / random (deterministic-per-epoch)
#   • weighted sampling (importance or class-balance)
#   • stratified sampling by class labels
#   • distributed training (DDP) with proper set_epoch semantics
#
# All builders are Hydra-friendly via Config dataclasses and a single
# `build_sampler(...)` entrypoint used by DataModule/DataLoader creation.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import os
import torch
from torch.utils.data import Sampler, Dataset
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
                             'distributed_sequential', 'distributed_random', 'distributed_weighted'
    """
    kind: str = "auto"
    seed: int = 42
    # Weighted / stratified options
    replacement: bool = False
    num_samples: Optional[int] = None   # if None, defaults to len(dataset) or per-replica len
    # Optional: provide weights/labels name keys if not discoverable
    weights_attr: Optional[str] = None  # dataset.<attr> or dataset.<attr>() returning tensor/list
    labels_attr: Optional[str] = None   # dataset.<attr> or dataset.<attr>() returning tensor/list
    # For stratified: floor epochs or oversample to match max class count
    stratified_equalize: bool = True


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


def _to_tensor(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device) if device else x
    return torch.as_tensor(x, device=device)


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
            t = _to_tensor(v).long().view(-1)
            if t.numel() == len(dataset):
                return t
    return None


def _get_weights(dataset: Dataset, cfg: SamplerConfig) -> Optional[torch.Tensor]:
    """Try explicit weights_attr or common 'weights'/'sample_weights' attributes."""
    if cfg.weights_attr:
        v = _maybe_call_attr(dataset, cfg.weights_attr)
        if v is not None:
            return _to_tensor(v).float().view(-1)
    for nm in ["weights", "sample_weights"]:
        v = _maybe_call_attr(dataset, nm)
        if v is not None:
            return _to_tensor(v).float().view(-1)
    # derive from labels (inverse frequency) if available
    labels = _get_labels(dataset, cfg)
    if labels is not None:
        counts = torch.bincount(labels)
        counts = torch.clamp(counts, min=1)
        inv = 1.0 / counts.float()
        w = inv[labels]
        return w
    return None


# -----------------------------------------------------------------------------
# Core Samplers
# -----------------------------------------------------------------------------

class DeterministicRandomSampler(Sampler[int]):
    """
    Random permutation of indices per epoch with deterministic seed = base + epoch.
    """
    def __init__(self, data_source: Dataset, base_seed: int = 42):
        super().__init__(data_source)
        self.data_source = data_source
        self.base_seed = int(base_seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        perm = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(perm)


class DeterministicSequentialSampler(Sampler[int]):
    """Sequential indices; supports set_epoch for interface compatibility."""
    def __init__(self, data_source: Dataset):
        super().__init__(data_source)
        self.data_source = data_source
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(range(len(self.data_source)))


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
                # sample with replacement to match max count
                choice = torch.randint(low=0, high=len(idxs), size=(max_c,), generator=g)
                per_class_indices.extend(idxs[choice].tolist())
        else:
            # sample proportional to class counts
            for c in self.classes:
                idxs = self.class_to_indices[c]
                per_class_indices.extend(idxs.tolist())

        # Shuffle and then truncate/expand to required epoch length
        perm = torch.randperm(len(per_class_indices), generator=g).tolist()
        out = [per_class_indices[i] for i in perm]
        if len(out) >= self.epoch_len:
            return iter(out[: self.epoch_len])
        # expand if needed
        reps = (self.epoch_len + len(out) - 1) // len(out)
        tiled = (out * reps)[: self.epoch_len]
        return iter(tiled)


class DistributedWeightedSampler(Sampler[int]):
    """
    Distributed variant of WeightedRandomSampler.
    - Partitions the global weighted stream across replicas deterministically.
    - Each epoch draws (approximately) num_samples_per_replica with replacement.
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

        # normalize weights
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
        # sample with replacement according to prob
        idx = torch.multinomial(self.prob, num_samples=self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())


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
) -> Sampler[int]:
    """
    Build a sampler for the dataset given config.
    If distributed_override is None, we auto-detect distributed.
    """
    ws, rk, is_dist = _discover_distributed()
    if world_size is not None:
        ws = world_size
    if rank is not None:
        rk = rank
    if distributed_override is not None:
        is_dist = bool(distributed_override)

    kind = cfg.kind.lower()

    # --- Auto mode
    if kind == "auto":
        if is_dist:
            # default to shuffle in distributed
            return DistributedSampler(
                dataset,
                num_replicas=ws,
                rank=rk,
                shuffle=True,
                seed=cfg.seed,
                drop_last=False,
            )
        # local random
        return DeterministicRandomSampler(dataset, base_seed=cfg.seed)

    # --- Non-distributed basics
    if kind == "sequential":
        return DeterministicSequentialSampler(dataset)

    if kind == "random":
        return DeterministicRandomSampler(dataset, base_seed=cfg.seed)

    if kind == "weighted":
        weights = _get_weights(dataset, cfg)
        if weights is None:
            raise ValueError("Weighted sampler requires `weights` or discoverable labels to derive weights.")
        num_samples = cfg.num_samples or len(dataset)
        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=bool(cfg.replacement),
        )

    if kind == "stratified":
        labels = _get_labels(dataset, cfg)
        if labels is None:
            raise ValueError("Stratified sampler requires labels.")
        num_samples = cfg.num_samples
        return StratifiedBalancedSampler(
            labels=labels,
            base_seed=cfg.seed,
            equalize=cfg.stratified_equalize,
            num_samples=num_samples,
        )

    # --- Distributed variants
    if kind in ("distributed_sequential", "distributed_random"):
        shuffle = kind.endswith("random")
        return DistributedSampler(
            dataset,
            num_replicas=ws,
            rank=rk,
            shuffle=shuffle,
            seed=cfg.seed,
            drop_last=False,
        )

    if kind == "distributed_weighted":
        weights = _get_weights(dataset, cfg)
        if weights is None:
            raise ValueError("Distributed weighted sampler requires `weights` or discoverable labels to derive weights.")
        # if user provided total num_samples, we split per replica
        if cfg.num_samples is not None:
            samples_per_replica = max(1, int(math.ceil(cfg.num_samples / ws)))
        else:
            samples_per_replica = None
        return DistributedWeightedSampler(
            weights=weights,
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
