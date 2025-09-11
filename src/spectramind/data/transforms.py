# src/spectramind/dat/transforms.py
from __future__ import annotations

from typing import Optional, Tuple, Union

try:  # optional torch support
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # numpy is typically available for data transforms
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

ArrayLike = Union["np.ndarray", "torch.Tensor"]


def _eps_for_dtype(x: ArrayLike, eps: float | None) -> float:
    """Pick a numerically sensible epsilon for x's dtype if eps is None."""
    if eps is not None:
        return float(eps)
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dtype in (torch.float64, torch.complex128):
            return 1e-12
        return 1e-8
    if _HAS_NUMPY and isinstance(x, np.ndarray):
        if x.dtype == np.float64:
            return 1e-12
        return 1e-8
    return 1e-8


def _is_torch(x: ArrayLike) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _nan_to_mask(x: ArrayLike) -> Optional[ArrayLike]:
    """Return a boolean mask where values are finite; None if backend lacks isfinite."""
    if _is_torch(x):
        # torch.isfinite available for float/complex tensors
        try:
            return torch.isfinite(x)  # type: ignore[no-any-return]
        except Exception:
            return None
    else:
        try:
            return np.isfinite(x)  # type: ignore[no-any-return]
        except Exception:
            return None


def _mean_std(
    x: ArrayLike,
    axis: Optional[Union[int, tuple[int, ...]]],
    keepdims: bool,
    ddof: int,
    mask: Optional[ArrayLike],
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute mean and std along axis, optionally ignoring non-finite via mask.
    For torch: uses dtype/device of x; for numpy: returns numpy arrays.
    """
    if _is_torch(x):
        # torch branch
        if mask is not None:
            # replace masked entries with 0 for sum; count via mask
            m = mask
            cnt = m.sum(dim=axis, keepdim=keepdims).clamp_min(1)  # avoid div by zero
            x0 = torch.where(m, x, torch.zeros((), dtype=x.dtype, device=x.device))
            s = x0.sum(dim=axis, keepdim=keepdims)
            mean = s / cnt
            # variance: sum((x - mean)^2 * mask) / (cnt - ddof)
            xm = torch.where(m, x - mean, torch.zeros((), dtype=x.dtype, device=x.device))
            sq = (xm * xm).sum(dim=axis, keepdim=keepdims)
            denom = (cnt - ddof).clamp_min(1)
            var = sq / denom
            std = torch.sqrt(var)
        else:
            mean = x.mean(dim=axis, keepdim=keepdims)
            # unbiased=False for population variance if ddof==0; otherwise compute manually
            if ddof == 0:
                std = x.std(dim=axis, keepdim=keepdims, unbiased=False)
            else:
                # Manual variance with ddof
                n = torch.tensor(x.numel() if axis is None else x.shape[axis] if isinstance(axis, int)
                                 else torch.prod(torch.tensor([x.shape[a] for a in axis], device=x.device)).item(),
                                 dtype=x.dtype, device=x.device)
                xm = x - mean
                denom = torch.clamp(n - ddof, min=1)
                var = (xm * xm).sum(dim=axis, keepdim=keepdims) / denom
                std = torch.sqrt(var)
        return mean, std

    # numpy branch
    if mask is not None:
        m = mask.astype(bool, copy=False)  # type: ignore[union-attr]
        cnt = m.sum(axis=axis, keepdims=keepdims).clip(min=1)
        x0 = np.where(m, x, 0.0)  # type: ignore[arg-type]
        s = x0.sum(axis=axis, keepdims=keepdims)
        mean = s / cnt
        xm = np.where(m, x - mean, 0.0)  # type: ignore[arg-type]
        sq = (xm * xm).sum(axis=axis, keepdims=keepdims)
        denom = np.clip(cnt - ddof, 1, None)
        var = sq / denom
        std = np.sqrt(var)
    else:
        if ddof == 0:
            mean = np.mean(x, axis=axis, keepdims=keepdims)  # type: ignore[no-untyped-call]
            std = np.std(x, axis=axis, keepdims=keepdims)    # type: ignore[no-untyped-call]
        else:
            mean = np.mean(x, axis=axis, keepdims=keepdims)  # type: ignore[no-untyped-call]
            n = np.prod([x.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,) if axis is not None else range(x.ndim))])  # type: ignore[index]
            xm = x - mean
            denom = max(int(n) - ddof, 1)
            var = np.sum(xm * xm, axis=axis, keepdims=keepdims) / denom  # type: ignore[no-untyped-call]
            std = np.sqrt(var)
    return mean, std


def zscore(
    x: ArrayLike,
    *,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    keepdims: bool = False,
    ddof: int = 0,
    eps: Optional[float] = None,
    mask: Optional[ArrayLike] = None,
    return_stats: bool = False,
) -> ArrayLike | Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Z-score normalize `x` along `axis`: (x - mean) / (std + eps).

    Features
    --------
    - Supports NumPy arrays and PyTorch tensors, including dtype/device preservation.
    - NaN/Inf-safe via auto-mask (or pass an explicit `mask=True` where values are valid).
    - `axis` can be None, int, or tuple of ints (e.g., per-feature or per-batch).
    - `ddof` controls sample vs population std (ddof=0 → population).
    - `eps` defaults based on dtype (1e-8 for float32, 1e-12 for float64).
    - `return_stats=True` returns (z, mean, std) for inverse transform.

    Parameters
    ----------
    x : np.ndarray | torch.Tensor
        Input array or tensor.
    axis : None | int | tuple[int, ...]
        Axes along which to compute mean/std. None → over all elements.
    keepdims : bool
        Keep reduced dims (broadcasting-friendly).
    ddof : int
        Delta degrees of freedom in std/var (0 for population; 1 for sample-like).
    eps : float | None
        Small value added to std for numerical stability. If None, picked by dtype.
    mask : array-like of bool | None
        Boolean mask of valid entries (same shape as x, or broadcastable).
        If None, non-finite values are masked automatically when possible.
    return_stats : bool
        If True, returns (z, mean, std).

    Returns
    -------
    z : np.ndarray | torch.Tensor
        Normalized array/tensor. If `return_stats=True`, also returns (mean, std).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],[4., 5., 6.]])
    >>> z = zscore(x, axis=0)
    >>> z.mean(axis=0).round(7).tolist()
    [0.0, 0.0, 0.0]
    """
    eps_v = _eps_for_dtype(x, eps)

    # Build/validate mask for non-finite safety
    m = mask
    if m is None:
        m = _nan_to_mask(x)

    mean, std = _mean_std(x, axis=axis, keepdims=keepdims, ddof=ddof, mask=m)

    if _is_torch(x):
        std_safe = std + eps_v
        # Avoid "-0.0"; torch.where preserves dtype/device
        z = (x - mean) / std_safe
        if return_stats:
            return z, mean, std
        return z

    # numpy
    std_safe = std + eps_v  # type: ignore[operator]
    z = (x - mean) / std_safe  # type: ignore[operator]
    if _HAS_NUMPY and isinstance(z, np.ndarray):
        # collapse tiny negative zeros introduced by roundoff
        z[np.isclose(z, 0.0)] = 0.0
    if return_stats:
        return z, mean, std
    return z
