# src/spectramind/train/transforms.py
# =============================================================================
# SpectraMind V50 — Transform Utilities (Hydra-friendly)
# -----------------------------------------------------------------------------
# - Stateless transforms: clip, log1p, normalize [min/max], standardize [μ/σ]
# - Robust standardize: median/MAD option (robust to outliers)
# - Lightweight augmentations: gaussian noise, jitter along axis, dropout
# - SpecAugment-inspired: RandomTime/Bin mask, RandomSpectralCutout
# - Smoothing: 1D conv (moving-average/gaussian-like)
# - Compose pipeline + apply_to_batch(dict) utilities
# - OnlinePerBinStandardizer: fit/update across batches, then call in pipeline
#
# All transforms accept/return torch.Tensor and are mask-aware where relevant.
# By default, ops apply along axis=-1 (e.g., spectral bins). You can change axis.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Mapping

import torch
import torch.nn.functional as F


Tensor = torch.Tensor
MaybeTensor = Optional[Tensor]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_tensor(x: Union[float, int, Tensor], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _axis_size(x: Tensor, axis: int) -> int:
    axis = axis if axis >= 0 else x.ndim + axis
    return x.shape[axis]


def _broadcast_mask(mask: Optional[Tensor], x: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    if mask.dtype != x.dtype:
        mask = mask.to(dtype=x.dtype)
    if mask.device != x.device:
        mask = mask.to(device=x.device)
    # allow broadcasting e.g. [B,1] over [B,D]
    if mask.ndim < x.ndim:
        view = [1] * x.ndim
        view[-1] = mask.shape[-1] if mask.ndim == 1 else mask.shape[-1]
        mask = mask.view(*view)
    return mask


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
    per_bin: bool = False                # if True: compute per-bin stats across batch_axis
    robust: bool = False                 # if True: use median/MAD (robust) instead of mean/std
    batch_axis: Optional[int] = 0        # axis treated as batch axis for per_bin stats
    eps: float = 1e-6
    axis: int = -1                       # reserved for future; ops already along last dim


class Standardize:
    """
    μ/σ standardization. If mean/std provided: use constants; else compute on the fly.
    If per_bin=True, compute stats across batch axis per-bin.
    If robust=True, use median/MAD instead of mean/std.
    """
    def __init__(self, cfg: StandardizeConfig):
        self.mean = cfg.mean
        self.std = cfg.std
        self.per_bin = bool(cfg.per_bin)
        self.robust = bool(cfg.robust)
        self.batch_axis = cfg.batch_axis
        self.eps = float(cfg.eps)

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        if self.mean is not None and self.std is not None:
            m = _as_tensor(self.mean, device=x.device, dtype=x.dtype)
            s = _as_tensor(self.std, device=x.device, dtype=x.dtype)
            y = (x - m) / s.clamp_min(self.eps)
            if mask is not None:
                y = mask * y + (1 - mask) * x
            return y

        if self.per_bin:
            if self.batch_axis is None:
                raise ValueError("Standardize(per_bin=True) requires batch_axis.")
            dims = [self.batch_axis]
            if not self.robust:
                m = x.mean(dim=dims, keepdim=True)
                v = x.var(dim=dims, unbiased=False, keepdim=True)
                s = torch.sqrt(v + self.eps)
            else:
                med = x.median(dim=self.batch_axis, keepdim=True).values
                mad = (x - med).abs().median(dim=self.batch_axis, keepdim=True).values
                # consistent with normal scale: 1.4826 * MAD
                s = (mad * 1.4826).clamp_min(self.eps)
                m = med
            y = (x - m) / s
            if mask is not None:
                y = mask * y + (1 - mask) * x
            return y

        if not self.robust:
            m = x.mean()
            v = x.var(unbiased=False)
            s = torch.sqrt(v + self.eps)
        else:
            med = x.median().values if hasattr(x.median(), "values") else x.median()
            mad = (x - med).abs().median().values if hasattr((x - med).abs().median(), "values") else (x - med).abs().median()
            s = (mad * 1.4826).clamp_min(self.eps)
            m = med
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

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        if self.min_val is not None and self.max_val is not None:
            lo = _as_tensor(self.min_val, device=x.device, dtype=x.dtype)
            hi = _as_tensor(self.max_val, device=x.device, dtype=x.dtype)
            y = _safe_div(x - lo, (hi - lo), eps=self.eps)
            if mask is not None:
                y = mask * y + (1 - mask) * x
            return y

        if self.per_bin:
            if self.batch_axis is None:
                raise ValueError("Normalize(per_bin=True) requires batch_axis.")
            lo = x.amin(dim=[self.batch_axis], keepdim=True)
            hi = x.amax(dim=[self.batch_axis], keepdim=True)
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
# Augmentations (stateless or epoch-aware)
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
            noise = torch.randn_like(x, generator=g) * self.std
        else:
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
        g.manual_seed((self.seed or 0) + self._epoch * 977)
        axis_len = _axis_size(x, self.axis)
        # Move target axis to last; apply per item
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        perm = list(range(x.ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        xt = x.permute(*perm).contiguous()  # [..., axis_len]
        flat = xt.reshape(-1, axis_len)      # [M, axis_len]
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
        g = torch.Generator(device=x.device)
        g.manual_seed((self.seed or 0) + self._epoch * 997)
        drop_mask = (torch.rand_like(x, generator=g) > self.p).to(x.dtype)
        y = x * drop_mask
        if mask is not None:
            y = mask * y + (1 - mask) * x
        return y


# --- SpecAugment-style masks --------------------------------------------------

@dataclass
class TimeMaskConfig:
    max_width: int = 0     # number of consecutive elements to mask along axis
    axis: int = -1
    p: float = 0.0         # probability of applying per sample
    seed: Optional[int] = None
    value: float = 0.0     # fill value when masked


class RandomTimeMask:
    """Masks a random contiguous segment (width<=max_width) along axis."""
    def __init__(self, cfg: TimeMaskConfig):
        self.max_width = int(cfg.max_width)
        self.axis = int(cfg.axis)
        self.p = float(cfg.p)
        self.value = float(cfg.value)
        self.seed = cfg.seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        if self.max_width <= 0 or self.p <= 0:
            return x
        g = torch.Generator(device=x.device)
        g.manual_seed((self.seed or 0) + self._epoch * 1013)
        axis_len = _axis_size(x, self.axis)
        if axis_len <= 1:
            return x
        # Decide per item if we apply
        # Flatten non-axis dims
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        perm = list(range(x.ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        xt = x.permute(*perm).contiguous()
        flat = xt.reshape(-1, axis_len)  # [M, D]
        M = flat.shape[0]
        apply_mask = torch.rand(M, generator=g, device=x.device) < self.p
        out = flat.clone()
        for i in range(M):
            if not apply_mask[i]:
                continue
            width = int(torch.randint(1, self.max_width + 1, (1,), generator=g, device=x.device).item())
            start = int(torch.randint(0, max(1, axis_len - width + 1), (1,), generator=g, device=x.device).item())
            out[i, start:start + width] = self.value
        out = out.view(*xt.shape).permute(*perm)
        if mask is not None:
            out = mask * out + (1 - mask) * x
        return out


# --- Smoothing ---------------------------------------------------------------

@dataclass
class Smooth1DConfig:
    kernel_size: int = 3     # odd
    axis: int = -1


class Smooth1D:
    """Simple moving-average smoothing along axis via 1D conv."""
    def __init__(self, cfg: Smooth1DConfig):
        ks = int(cfg.kernel_size) if cfg.kernel_size % 2 == 1 else int(cfg.kernel_size) + 1
        self.kernel_size = max(1, ks)
        self.axis = int(cfg.axis)

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        if self.kernel_size <= 1:
            return x
        # move axis to last, apply depthwise conv1d on flattened batch
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        perm = list(range(x.ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        xt = x.permute(*perm).contiguous()  # [..., D]
        Bflat, D = int(torch.prod(torch.tensor(xt.shape[:-1]))), xt.shape[-1]
        y = xt.view(Bflat, 1, D)
        kernel = torch.ones(1, 1, self.kernel_size, device=x.device, dtype=x.dtype) / float(self.kernel_size)
        pad = (self.kernel_size // 2, self.kernel_size // 2)
        y = F.pad(y, (pad[0], pad[1]), mode="replicate")
        y = F.conv1d(y, kernel, groups=1)
        out = y.view(*xt.shape).permute(*perm)
        if mask is not None:
            out = mask * out + (1 - mask) * x
        return out


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
    # Preprocess
    standardize: Optional[StandardizeConfig] = None
    normalize: Optional[NormalizeConfig] = None
    clip: Optional[ClipConfig] = None
    log1p: Optional[Log1pConfig] = None
    smooth1d: Optional[Smooth1DConfig] = None
    # Augment
    noise: Optional[NoiseConfig] = None
    jitter: Optional[JitterConfig] = None
    dropout: Optional[DropoutConfig] = None
    timemask: Optional[TimeMaskConfig] = None


def build_transforms(cfg: TransformsConfig) -> Compose:
    """
    Build a Compose pipeline from config (typical order):
    1) log1p
    2) clip
    3) normalize / standardize (choose one or both as desired)
    4) smoothing
    5) augmentations: noise, jitter, dropout, timemask
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
    if cfg.smooth1d is not None and cfg.smooth1d.kernel_size > 1:
        chain.append(Smooth1D(cfg.smooth1d))
    if cfg.noise is not None and cfg.noise.std > 0:
        chain.append(AdditiveGaussianNoise(cfg.noise))
    if cfg.jitter is not None and cfg.jitter.magnitude > 0:
        chain.append(JitterAlongAxis(cfg.jitter))
    if cfg.dropout is not None and cfg.dropout.p > 0:
        chain.append(DropoutAlongAxis(cfg.dropout))
    if cfg.timemask is not None and cfg.timemask.max_width > 0 and cfg.timemask.p > 0:
        chain.append(RandomTimeMask(cfg.timemask))
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
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        perm = list(range(x.ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        xt = x.permute(*perm)  # [..., BINS]
        flat = xt.reshape(-1, xt.shape[-1])      # [N, BINS]
        if mask is not None:
            mt = mask.permute(*perm).reshape(-1, xt.shape[-1]).to(dtype=flat.dtype, device=flat.device)
        else:
            mt = torch.ones_like(flat)

        count = mt.sum(dim=0)
        sum_x = (flat * mt).sum(dim=0)
        mean = _safe_div(sum_x, count.clamp_min(1.0))
        sum_sq = (mt * (flat - mean)**2).sum(dim=0)

        if self._count is None:
            self._count = count.clone()
            self._mean = mean.clone()
            self._M2 = sum_sq.clone()
        else:
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

    def _stats(self, device: torch.device) -> Tuple[Tensor, Tensor]:
        if self._count is None or self._mean is None or self._M2 is None:
            raise RuntimeError("OnlinePerBinStandardizer: stats are empty; call update(...) first.")
        mean = self._mean.to(device=device)
        var = _safe_div(self._M2, self._count.clamp_min(1.0)).to(device=device)
        std = torch.sqrt(var + self.eps)
        return mean, std

    def __call__(self, x: Tensor, mask: MaybeTensor = None) -> Tensor:
        mean, std = self._stats(x.device)
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
    Apply per-key transforms to a dict batch. Common keys: 'fgs1', 'airs', 'mu', 'sigma', 'target', etc.
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