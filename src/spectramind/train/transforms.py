# src/spectramind/train/transforms.py
# =============================================================================
# SpectraMind V50 — Transform Utilities (Hydra-friendly)
# -----------------------------------------------------------------------------
# - Stateless transforms: clip, log1p, normalize [min/max], standardize [μ/σ]
# - Lightweight augmentations: gaussian noise, jitter along axis, dropout
# - Compose pipeline + apply_to_batch(dict) utilities
# - OnlinePerBinStandardizer: fit/update across batches, then call in pipeline
#
# All transforms accept/return torch.Tensor and are mask-aware where relevant.
# By default, ops apply along axis=-1 (e.g., spectral bins). You can change axis.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Mapping

import math
import torch
import torch.nn.functional as F


Tensor = torch.Tensor
MaybeTensor = Optional[Tensor]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_tensor(x: Union[float, int, Tensor], device: Optional[torch.device] = None) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device) if device is not None else x
    return torch.as_tensor(x, device=device)


def _take_along_axis(x: Tensor, indices: Tensor, axis: int) -> Tensor:
    """
    Equivalent to np.take_along_axis for PyTorch.
    x: [..., D, ...], indices: same rank as x except size D on 'axis' replaced by K
    """
    # Move axis to last
    axis = axis if axis >= 0 else x.ndim + axis
    x_perm = list(range(x.ndim))
    x_perm[axis], x_perm[-1] = x_perm[-1], x_perm[axis]
    x_t = x.permute(*x_perm)
    idx_t = indices.permute(*x_perm)
    out = x_t.gather(dim=-1, index=idx_t)
    # Undo perm
    x_perm[axis], x_perm[-1] = x_perm[-1], x_perm[axis]
    return out.permute(*x_perm)


def _axis_size(x: Tensor, axis: int) -> int:
    axis = axis if axis >= 0 else x.ndim + axis
    return x.shape[axis]


def _broadcast_mask(mask: Optional[Tensor], x: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    # try to broadcast to x shape
    return mask.to(device=x.device, dtype=x.dtype).expand_as(x) if mask.ndim == 0 else mask.to(x.device, dtype=x.dtype)


def _safe_div(num: Tensor, den: Tensor, eps: float = 1e-12) -> Tensor:
    return num / den.clamp_min(eps)


# -----------------------------------------------------------------------------
# Stateless transforms (dataclass configs + callables)
# -----------------------------------------------------------------------------

@dataclass
class ClipConfig:
    min_val: float = 0.0
    max_val: float = 1.0


class Clip:
    def __init__(self, cfg: ClipConfig):
        self.min = float(cfg.min_val)
        self.max = float(cfg.max_val)

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        y = x.clamp(min=self.min, max=self.max)
        if mask is not None:
            y = mask * y + (1 - mask) * x
        return y


@dataclass
class Log1pConfig:
    enable: bool = True
    clamp_min: float = 0.0  # clamp x >= clamp_min before log1p


class Log1p:
    def __init__(self, cfg: Log1pConfig):
        self.enable = bool(cfg.enable)
        self.clamp_min = float(cfg.clamp_min)

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        if not self.enable:
            return x
        x_clamped = x.clamp_min(self.clamp_min)
        y = torch.log1p(x_clamped)
        if mask is not None:
            y = mask * y + (1 - mask) * x
        return y


@dataclass
class StandardizeConfig:
    mean: Optional[float] = None         # scalar mean or None (estimate per-batch)
    std: Optional[float] = None          # scalar std or None (estimate per-batch)
    per_bin: bool = False                # if True: compute per-bin mean/std along batch axis
    batch_axis: Optional[int] = 0        # axis treated as batch axis for per_bin stats
    eps: float = 1e-6
    axis: int = -1                       # transform along this axis (just for symmetry)


class Standardize:
    """
    μ/σ standardization. If mean/std provided: use as constants; else compute on the fly.
    If per_bin=True, compute mean/std across batch axis for each bin along 'axis'.
    """
    def __init__(self, cfg: StandardizeConfig):
        self.mean = cfg.mean
        self.std = cfg.std
        self.per_bin = bool(cfg.per_bin)
        self.batch_axis = cfg.batch_axis
        self.eps = float(cfg.eps)
        self.axis = cfg.axis

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        # dispenses along axis but stats are computed either scalar or per_bin across batch
        if self.mean is not None and self.std is not None:
            m = _as_tensor(self.mean, device=x.device)
            s = _as_tensor(self.std, device=x.device)
            y = (x - m) / s.clamp_min(self.eps)
            if mask is not None:
                y = mask * y + (1 - mask) * x
            return y

        # compute on-the-fly stats
        if self.per_bin:
            if self.batch_axis is None:
                raise ValueError("Standardize(per_bin=True) requires a batch_axis to compute stats.")
            # compute μ,σ over batch axis for each bin (keep dims for broadcasting)
            dims = list(range(x.ndim))
            reduce_dims = [self.batch_axis]
            m = x.mean(dim=reduce_dims, keepdim=True)
            v = x.var(dim=reduce_dims, unbiased=False, keepdim=True)
            s = torch.sqrt(v + self.eps)
            y = (x - m) / s
            if mask is not None:
                y = mask * y + (1 - mask) * x
            return y

        # scalar stats from entire tensor
        m = x.mean()
        v = x.var(unbiased=False)
        s = torch.sqrt(v + self.eps)
        y = (x - m) / s
        if mask is not None:
            y = mask * y + (1 - mask) * x
        return y


@dataclass
class NormalizeConfig:
    min_val: Optional[float] = None      # if provided, use given range; else compute per-batch
    max_val: Optional[float] = None
    per_bin: bool = False
    batch_axis: Optional[int] = 0
    eps: float = 1e-12
    axis: int = -1


class Normalize:
    """
    Min-max normalization into [0,1]. If min/max provided: use constants; else compute on the fly.
    If per_bin=True, compute per-bin min/max across batch_axis.
    """
    def __init__(self, cfg: NormalizeConfig):
        self.min_val = cfg.min_val
        self.max_val = cfg.max_val
        self.per_bin = bool(cfg.per_bin)
        self.batch_axis = cfg.batch_axis
        self.eps = float(cfg.eps)
        self.axis = cfg.axis

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        if self.min_val is not None and self.max_val is not None:
            lo = _as_tensor(self.min_val, device=x.device)
            hi = _as_tensor(self.max_val, device=x.device)
            y = _safe_div(x - lo, (hi - lo), eps=self.eps)
            if mask is not None:
                y = mask * y + (1 - mask) * x
            return y

        if self.per_bin:
            if self.batch_axis is None:
                raise ValueError("Normalize(per_bin=True) requires a batch_axis to compute stats.")
            dims = [self.batch_axis]
            lo = x.amin(dim=dims, keepdim=True)
            hi = x.amax(dim=dims, keepdim=True)
            y = _safe_div(x - lo, hi - lo, eps=self.eps)
            if mask is not None:
                y = mask * y + (1 - mask) * x
            return y

        lo = x.amin()
        hi = x.amax()
        y = _safe_div(x - lo, hi - lo, eps=self.eps)
        if mask is not None:
            y = mask * y + (1 - mask) * x
        return y


# -----------------------------------------------------------------------------
# Augmentations (stateless)
# -----------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    std: float = 0.0
    axis: int = -1
    per_bin: bool = False           # if True, draw per-bin noise; else scalar per item along axis
    seed: Optional[int] = None


class AdditiveGaussianNoise:
    def __init__(self, cfg: NoiseConfig):
        self.std = float(cfg.std)
        self.axis = int(cfg.axis)
        self.per_bin = bool(cfg.per_bin)
        self.seed = cfg.seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        if self.std <= 0:
            return x
        g = torch.Generator(device=x.device)
        seed = (self.seed or 0) + self._epoch * 911
        g.manual_seed(seed)
        if self.per_bin:
            shape = list(x.shape)
            # noise independent per bin; same shape as x
            noise = torch.randn(shape, generator=g, device=x.device, dtype=x.dtype) * self.std
        else:
            # one noise scalar per item along 'axis'
            n = _axis_size(x, self.axis)
            shape = [1] * x.ndim
            shape[self.axis] = n
            noise = torch.randn(shape, generator=g, device=x.device, dtype=x.dtype) * self.std
        y = x + noise
        if mask is not None:
            y = mask * y + (1 - mask) * x
        return y


@dataclass
class JitterConfig:
    magnitude: int = 0     # shift steps along axis
    axis: int = -1
    seed: Optional[int] = None


class JitterAlongAxis:
    """
    Circularly shifts values along 'axis' per item. If magnitude>0, random shift in [-mag, +mag].
    """
    def __init__(self, cfg: JitterConfig):
        self.mag = int(cfg.magnitude)
        self.axis = int(cfg.axis)
        self.seed = cfg.seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        if self.mag <= 0:
            return x
        g = torch.Generator(device=x.device)
        seed = (self.seed or 0) + self._epoch * 977
        g.manual_seed(seed)
        # draw integer shifts uniformly for each item in batch-subspace (everything except 'axis')
        axis_len = _axis_size(x, self.axis)
        # flatten all dims except axis to one
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        perm = list(range(x.ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        xt = x.permute(*perm)  # [..., axis_len]
        flat = xt.reshape(-1, axis_len)  # [M, axis_len]
        M = flat.shape[0]
        shifts = torch.randint(low=-self.mag, high=self.mag + 1, size=(M,), generator=g, device=x.device)
        out = torch.empty_like(flat)
        for i in range(M):
            out[i] = torch.roll(flat[i], shifts=int(shifts[i].item()), dims=-1)
        out = out.view(*xt.shape).permute(*perm)  # back
        if mask is not None:
            out = mask * out + (1 - mask) * x
        return out


@dataclass
class DropoutConfig:
    p: float = 0.0
    axis: int = -1
    seed: Optional[int] = None


class DropoutAlongAxis:
    """
    Drops entries independently along 'axis' with probability p.
    """
    def __init__(self, cfg: DropoutConfig):
        self.p = float(cfg.p)
        self.axis = int(cfg.axis)
        self.seed = cfg.seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        if self.p <= 0:
            return x
        axis_len = _axis_size(x, self.axis)
        shape = list(x.shape)
        # mask with same shape as x for IID per element along axis
        g = torch.Generator(device=x.device)
        seed = (self.seed or 0) + self._epoch * 997
        g.manual_seed(seed)
        drop_mask = (torch.rand(shape, generator=g, device=x.device, dtype=x.dtype) > self.p).to(x.dtype)
        y = x * drop_mask
        if mask is not None:
            y = mask * y + (1 - mask) * x
        return y


# -----------------------------------------------------------------------------
# Compose & builder
# -----------------------------------------------------------------------------

class Compose:
    """Compose list of transforms. Each transform takes (x, mask=None) -> x."""
    def __init__(self, transforms: Sequence[Callable[[Tensor, Optional[Tensor]], Tensor]]):
        self.transforms = list(transforms)

    def set_epoch(self, epoch: int) -> None:
        for t in self.transforms:
            if hasattr(t, "set_epoch") and callable(getattr(t, "set_epoch")):
                t.set_epoch(epoch)

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        y = x
        for t in self.transforms:
            y = t(y, mask=mask)
        return y


@dataclass
class TransformsConfig:
    # Optional sub-transforms (None -> skip)
    standardize: Optional[StandardizeConfig] = None
    normalize: Optional[NormalizeConfig] = None
    clip: Optional[ClipConfig] = None
    log1p: Optional[Log1pConfig] = None
    noise: Optional[NoiseConfig] = None
    jitter: Optional[JitterConfig] = None
    dropout: Optional[DropoutConfig] = None


def build_transforms(cfg: TransformsConfig) -> Compose:
    """
    Build a Compose pipeline from config (order below is typical):
    1) log1p
    2) clip
    3) normalize / standardize (choose one or both as desired)
    4) augmentations: noise, jitter, dropout
    """
    chain: List[Callable[[Tensor, Optional[Tensor]], Tensor]] = []
    if cfg.log1p is not None:
        chain.append(Log1p(cfg.log1p))
    if cfg.clip is not None:
        chain.append(Clip(cfg.clip))
    if cfg.normalize is not None:
        chain.append(Normalize(cfg.normalize))
    if cfg.standardize is not None:
        chain.append(Standardize(cfg.standardize))
    if cfg.noise is not None and cfg.noise.std > 0:
        chain.append(AdditiveGaussianNoise(cfg.noise))
    if cfg.jitter is not None and cfg.jitter.magnitude > 0:
        chain.append(JitterAlongAxis(cfg.jitter))
    if cfg.dropout is not None and cfg.dropout.p > 0:
        chain.append(DropoutAlongAxis(cfg.dropout))
    return Compose(chain)


# -----------------------------------------------------------------------------
# Online per-bin standardizer (fit/update then call)
# -----------------------------------------------------------------------------

class OnlinePerBinStandardizer:
    """
    Maintains running mean/var independently for each bin along `axis` (default last).
    Usage:
        s = OnlinePerBinStandardizer(axis=-1, eps=1e-6)
        s.update(batch_x, mask=batch_mask)   # repeatedly
        y = s(batch_x, mask=batch_mask)      # standardize using running stats
    """
    def __init__(self, axis: int = -1, eps: float = 1e-6):
        self.axis = axis
        self.eps = float(eps)
        self._count: Optional[Tensor] = None   # [bins]
        self._mean: Optional[Tensor] = None    # [bins]
        self._M2: Optional[Tensor] = None      # [bins] (sum of squared diffs)

    def reset(self) -> None:
        self._count = None
        self._mean = None
        self._M2 = None

    @torch.no_grad()
    def update(self, x: Tensor, mask: MaybeTensor = None) -> None:
        # reshape to [N, BINS] where N = product of dims except 'axis'
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        perm = list(range(x.ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        xt = x.permute(*perm)  # [..., BINS]
        flat = xt.reshape(-1, xt.shape[-1])      # [N, BINS]
        if mask is not None:
            mt = mask.permute(*perm).reshape(-1, xt.shape[-1]).to(dtype=flat.dtype, device=flat.device)
        else:
            mt = torch.ones_like(flat)

        # masked batch stats: per bin
        # count per bin:
        count = mt.sum(dim=0)                   # [BINS]
        # mean:
        sum_x = (flat * mt).sum(dim=0)          # [BINS]
        mean = _safe_div(sum_x, count.clamp_min(1.0))
        # sum of squared diffs:
        sum_sq = (mt * (flat - mean)**2).sum(dim=0)  # [BINS]

        # Welford merge
        if self._count is None:
            self._count = count.clone()
            self._mean = mean.clone()
            self._M2 = sum_sq.clone()
        else:
            # merge current (mean, M2, count) with new batch
            c0, m0, M0 = self._count, self._mean, self._M2
            c1, m1, M1 = count, mean, sum_sq
            c = c0 + c1
            delta = m1 - m0
            m = m0 + delta * _safe_div(c1, c.clamp_min(1.0))
            M = M0 + M1 + delta * delta * _safe_div(c0 * c1, c.clamp_min(1.0))
            self._count, self._mean, self._M2 = c, m, M

    def state_dict(self) -> Dict[str, Tensor]:
        return {
            "count": self._count if self._count is not None else torch.tensor([]),
            "mean": self._mean if self._mean is not None else torch.tensor([]),
            "M2": self._M2 if self._M2 is not None else torch.tensor([]),
            "axis": torch.tensor(self.axis),
        }

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self._count = state.get("count", None)
        self._mean = state.get("mean", None)
        self._M2 = state.get("M2", None)
        # axis not strictly required to match; kept for debugging

    def _stats(self, device: torch.device) -> Tuple[Tensor, Tensor]:
        if self._count is None or self._mean is None or self._M2 is None:
            raise RuntimeError("OnlinePerBinStandardizer: stats are empty; call update(...) before using.")
        mean = self._mean.to(device=device)
        var = _safe_div(self._M2, self._count.clamp_min(1.0)).to(device=device)
        std = torch.sqrt(var + self.eps)
        return mean, std

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        mean, std = self._stats(x.device)
        # expand mean/std to broadcast across all dims except axis
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        shape = [1] * x.ndim
        shape[axis] = mean.shape[0]
        m = mean.view(*shape)
        s = std.view(*shape)
        y = (x - m) / s.clamp_min(self.eps)
        if mask is not None:
            y = mask * y + (1 - mask) * x
        return y


# -----------------------------------------------------------------------------
# Batch dictionary utilities
# -----------------------------------------------------------------------------

def apply_to_batch(
    batch: Dict[str, Tensor],
    transforms: Dict[str, Callable[[Tensor, Optional[Tensor]], Tensor]],
    *,
    masks: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """
    Apply per-key transforms to a dict batch. Common keys: 'fgs1', 'airs', 'target', 'mask', etc.
    Example:
        t = build_transforms(cfg.transforms)   # returns Compose
        out = apply_to_batch(
            batch,
            transforms={'fgs1': t, 'airs': t},
            masks={'fgs1': batch.get('mask_fgs1'), 'airs': batch.get('mask_airs')}
        )
    """
    out = dict(batch)
    for key, t in transforms.items():
        if key not in batch:
            continue
        m = masks.get(key) if masks is not None else None
        out[key] = t(batch[key], mask=m)
    return out
