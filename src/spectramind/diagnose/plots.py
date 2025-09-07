# src/spectramind/diagnose/plots.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# Headless rendering — safe for CI/Kaggle/servers
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

try:  # optional aesthetics
    import seaborn as sns  # type: ignore

    _HAS_SEABORN = True
except Exception:  # pragma: no cover
    _HAS_SEABORN = False


# --------------------------------------------------------------------------
# Style & IO
# --------------------------------------------------------------------------


def set_style(*, use_seaborn: Optional[bool] = None, font_scale: float = 1.0) -> None:
    """
    Apply a clean, publication-ready style.

    Args:
        use_seaborn: Force seaborn or pure Matplotlib. If None, autodetect.
        font_scale: Global font scaling factor.
    """
    if use_seaborn is None:
        use_seaborn = _HAS_SEABORN

    if use_seaborn:
        sns.set_theme(
            context="notebook",
            style="whitegrid",
            font_scale=font_scale,
            rc={
                "figure.dpi": 120,
                "savefig.dpi": 144,
                "axes.titlesize": "medium",
                "axes.labelsize": "medium",
                "axes.grid": True,
                "grid.alpha": 0.2,
                "legend.frameon": False,
            },
        )
    else:
        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "savefig.dpi": 144,
                "axes.titlesize": "medium",
                "axes.labelsize": "medium",
                "axes.grid": True,
                "grid.alpha": 0.2,
                "legend.frameon": False,
                "font.size": 10 * font_scale,
            }
        )


def savefig(fig: plt.Figure, path: Path | str, *, tight: bool = True, transparent: bool = False) -> None:
    """
    Safe figure save with parent creation and sane defaults.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight" if tight else None, transparent=transparent)


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------


def _as_1d(a: Any) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim == 0:
        x = x[None]
    return x.reshape(-1)


def _nanmask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(arrays[0], dtype=bool)
    for a in arrays:
        mask |= ~np.isfinite(a)
    return ~mask


# --------------------------------------------------------------------------
# Core diagnostic plots
# --------------------------------------------------------------------------


def plot_training_curves(
    history: Mapping[str, Sequence[float]] | Mapping[str, np.ndarray],
    *,
    title: str | None = "Training Curves",
    max_cols: int = 2,
) -> plt.Figure:
    """
    Plot training curves from a Keras/Lightning-style history dict, e.g.:
        {"loss": [...], "val_loss": [...], "metric": [...], "val_metric": [...]}
    """
    keys = sorted(history.keys())
    # Group into paired (train,val) metrics
    groups: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for k in keys:
        if k.startswith("val_"):
            base = k[4:]
            groups.setdefault(base, (None, None))
            train, _ = groups[base]
            groups[base] = (train, k)
        else:
            base = k
            groups.setdefault(base, (None, None))
            _, val = groups[base]
            groups[base] = (k, val)

    n = len(groups)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)

    for ax, (name, (train_k, val_k)) in zip(axes.flat, groups.items()):
        if train_k is not None:
            y = np.asarray(history[train_k], dtype=float)
            ax.plot(np.arange(len(y)), y, label=train_k, lw=2)
        if val_k is not None:
            yv = np.asarray(history[val_k], dtype=float)
            ax.plot(np.arange(len(yv)), yv, label=val_k, lw=2)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.legend()

    # hide spare axes
    for j in range(len(groups), rows * cols):
        axes.flat[j].axis("off")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_parity(y_true: Sequence[float], y_pred: Sequence[float], *, title: str | None = "Parity Plot") -> plt.Figure:
    """
    Parity (y_true vs y_pred) with identity reference.
    """
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    mask = _nanmask(yt, yp)
    yt, yp = yt[mask], yp[mask]

    fig, ax = plt.subplots(figsize=(5, 5))
    lims = [np.nanmin([yt.min(), yp.min()]), np.nanmax([yt.max(), yp.max()])]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6, label="identity")
    ax.scatter(yt, yp, s=18, alpha=0.6)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title or "")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    bins: int = 50,
    title: str | None = "Residuals",
) -> plt.Figure:
    """
    Histogram of residuals and a residuals-vs-predicted panel.
    """
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    mask = _nanmask(yt, yp)
    r = (yp - yt)[mask]
    yp = yp[mask]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(r, bins=bins, alpha=0.8, color="#4C72B0")
    axes[0].axvline(0.0, color="k", lw=1, ls="--")
    axes[0].set_title(f"{title} (hist)")
    axes[0].set_xlabel("residual = pred - true")
    axes[0].set_ylabel("count")

    axes[1].scatter(yp, r, s=12, alpha=0.6)
    axes[1].axhline(0.0, color="k", lw=1, ls="--")
    axes[1].set_title(f"{title} vs predicted")
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("residual")

    fig.tight_layout()
    return fig


def plot_qq(residuals: Sequence[float], *, title: str | None = "Q-Q Plot (Normal)") -> plt.Figure:
    """
    Q-Q plot against standard normal.
    """
    r = _as_1d(residuals)
    r = r[np.isfinite(r)]
    r = np.sort(r)
    n = r.size
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No finite residuals", ha="center", va="center")
        ax.axis("off")
        return fig

    # Theoretical quantiles from standard normal
    p = (np.arange(1, n + 1) - 0.5) / n
    q = np.sqrt(2) * erfinv(2 * p - 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(q, r, s=16, alpha=0.7)
    lims = [min(q.min(), r.min()), max(q.max(), r.max())]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.7)
    ax.set_xlabel("Theoretical quantiles (N(0,1))")
    ax.set_ylabel("Empirical quantiles (residuals)")
    ax.set_title(title or "")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def erfinv(x: np.ndarray | float) -> np.ndarray:
    """
    Vectorized inverse error function via numpy.polynomial approximation if SciPy is absent.
    """
    # try SciPy if available
    try:  # pragma: no cover
        from scipy.special import erfinv as _erfinv  # type: ignore

        return _erfinv(x)
    except Exception:
        pass
    # Winitzki approximation (accurate for |x| < 1)
    x = np.asarray(x, dtype=float)
    a = 0.147  # magic from the approximation
    sgn = np.sign(x)
    ln = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    second = ln / a
    inside = np.sqrt(first * first - second)
    return sgn * np.sqrt(inside - first)


# --------------------------------------------------------------------------
# Spectroscopy-focused plots
# --------------------------------------------------------------------------


def plot_spectrum(
    mu: Sequence[float],
    *,
    sigma: Optional[Sequence[float]] = None,
    truth: Optional[Sequence[float]] = None,
    wavelength: Optional[Sequence[float]] = None,
    title: str | None = "Predicted Spectrum",
    band_labels: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """
    Plot a single 283-bin (or arbitrary length) spectrum with optional uncertainty and truth.

    Args:
        mu: Predicted mean spectrum (length B).
        sigma: Optional predicted std per bin (length B).
        truth: Optional ground-truth spectrum (length B).
        wavelength: Optional x-axis (length B). If None, uses 0..B-1 indices.
        band_labels: Optional labels for bands (tick labels length B).
    """
    mu = _as_1d(mu)
    B = mu.size
    x = np.arange(B) if wavelength is None else _as_1d(wavelength)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(x, mu, lw=2, label="μ (pred)")
    if sigma is not None:
        s = _as_1d(sigma)
        s = np.clip(s, 0.0, np.inf)
        ax.fill_between(x, mu - s, mu + s, color="#4C72B0", alpha=0.20, label="±σ")
    if truth is not None:
        t = _as_1d(truth)
        ax.plot(x, t, lw=1.5, color="#DD8452", alpha=0.9, label="truth")

    ax.set_xlabel("Bin" if wavelength is None else "Wavelength")
    ax.set_ylabel("Transit depth (a.u.)")
    ax.set_title(title or "")
    if band_labels is not None and len(band_labels) == B:
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, rotation=90, fontsize=7)
    ax.legend(loc="best", ncols=3)
    fig.tight_layout()
    return fig


def plot_spectra_grid(
    mu: Sequence[Sequence[float]],
    *,
    sigma: Optional[Sequence[Sequence[float]]] = None,
    truth: Optional[Sequence[Sequence[float]]] = None,
    wavelength: Optional[Sequence[float]] = None,
    titles: Optional[Sequence[str]] = None,
    ncols: int = 3,
) -> plt.Figure:
    """
    Grid of spectra (e.g., first 6 predictions).
    """
    mu = np.asarray(mu, dtype=float)
    N = mu.shape[0]
    ncols = max(1, ncols)
    nrows = math.ceil(N / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * (ncols / 3), 3.0 * nrows), squeeze=False)

    for i in range(N):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        _plot_spectrum_ax(
            ax,
            mu[i],
            sigma=None if sigma is None else np.asarray(sigma)[i],
            truth=None if truth is None else np.asarray(truth)[i],
            wavelength=wavelength,
            title=titles[i] if titles and i < len(titles) else None,
        )

    # hide any unused axes
    for j in range(N, nrows * ncols):
        rr, cc = j // ncols, j % ncols
        axes[rr][cc].axis("off")

    fig.tight_layout()
    return fig


def _plot_spectrum_ax(
    ax: plt.Axes,
    mu: Sequence[float],
    *,
    sigma: Optional[Sequence[float]] = None,
    truth: Optional[Sequence[float]] = None,
    wavelength: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
) -> None:
    mu = _as_1d(mu)
    B = mu.size
    x = np.arange(B) if wavelength is None else _as_1d(wavelength)
    ax.plot(x, mu, lw=1.8, label="μ")
    if sigma is not None:
        s = _as_1d(sigma)
        s = np.clip(s, 0.0, np.inf)
        ax.fill_between(x, mu - s, mu + s, alpha=0.2, label="±σ")
    if truth is not None:
        t = _as_1d(truth)
        ax.plot(x, t, lw=1.4, alpha=0.9, label="truth", color="#DD8452")
    ax.set_xlabel("Bin" if wavelength is None else "Wavelength")
    ax.set_ylabel("Transit depth (a.u.)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


# --------------------------------------------------------------------------
# Uncertainty diagnostics
# --------------------------------------------------------------------------


def plot_uncertainty_calibration(
    mu: Sequence[float],
    sigma: Sequence[float],
    y_true: Sequence[float],
    *,
    bins: int = 40,
    title: str | None = "Uncertainty Calibration",
) -> plt.Figure:
    """
    Assess heteroscedastic Gaussian calibration via:
      - z = (y_true - mu) / sigma  histogram vs N(0,1)
      - PIT (Φ(z)) histogram (should be U[0,1])
      - Coverage curve: nominal vs empirical for central intervals
    """
    m = _as_1d(mu)
    s = np.clip(_as_1d(sigma), 1e-12, np.inf)
    y = _as_1d(y_true)
    mask = _nanmask(m, s, y)
    z = (y[mask] - m[mask]) / s[mask]
    pit = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

    fig = plt.figure(figsize=(12, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.25)

    # z histogram
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(z, bins=bins, density=True, alpha=0.8, color="#4C72B0", label="empirical z")
    # standard normal PDF overlay
    zz = np.linspace(-4, 4, 400)
    ax0.plot(zz, np.exp(-0.5 * zz**2) / np.sqrt(2 * np.pi), "k--", lw=1.5, label="N(0,1)")
    ax0.set_title("z = (y-μ)/σ")
    ax0.set_xlabel("z")
    ax0.set_ylabel("density")
    ax0.legend()

    # PIT histogram
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(pit, bins=bins, range=(0, 1), density=True, alpha=0.8, color="#55A868")
    ax1.axhline(1.0, color="k", lw=1, ls="--")
    ax1.set_title("PIT = Φ(z)")
    ax1.set_xlabel("u")
    ax1.set_ylabel("density")

    # Coverage curve
    ax2 = fig.add_subplot(gs[0, 2])
    alphas = np.linspace(0.01, 0.99, 50)
    nom = alphas
    # empirical: fraction with |z| <= Φ^{-1}((1+a)/2)
    thresh = np.sqrt(2.0) * erfinv(alphas)  # since erfinv(α) -> z s.t. erf(z)=α
    emp = [np.mean(np.abs(z) <= t) for t in thresh]
    ax2.plot(nom, emp, lw=2, label="empirical")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Nominal coverage")
    ax2.set_ylabel("Empirical coverage")
    ax2.set_title("Coverage")
    ax2.legend()

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def erf(x: np.ndarray | float) -> np.ndarray:
    """Vectorized error function with SciPy fallback."""
    try:  # pragma: no cover
        from scipy.special import erf as _erf  # type: ignore

        return _erf(x)
    except Exception:
        # Abramowitz and Stegun approximation (sufficient for plotting)
        x = np.asarray(x, dtype=float)
        a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
        p = 0.3275911
        sgn = np.sign(x)
        x = np.abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-(x * x))
        return sgn * y


# --------------------------------------------------------------------------
# Frequency-domain helper (optional)
# --------------------------------------------------------------------------


def plot_fft_power(signal: Sequence[float], *, fs: float = 1.0, title: str | None = "FFT Power") -> plt.Figure:
    """
    Simple one-sided FFT power spectrum for a real-valued signal.
    """
    x = _as_1d(signal)
    x = x[np.isfinite(x)]
    N = x.size
    if N == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Empty signal", ha="center", va="center")
        ax.axis("off")
        return fig

    # Detrend (remove mean)
    x = x - x.mean()

    # FFT
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(N, d=1.0 / fs)  # Hz
    Pxx = (np.abs(X) ** 2) / N

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(f, Pxx, lw=1.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title or "")
    ax.set_xlim(left=0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Convenience API for batch saving
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FigureSpec:
    fig: plt.Figure
    path: Path
    tight: bool = True
    transparent: bool = False


def save_all(figs: Iterable[FigureSpec]) -> None:
    """
    Save a collection of figures with consistent behavior.
    """
    for spec in figs:
        savefig(spec.fig, spec.path, tight=spec.tight, transparent=spec.transparent)


# --------------------------------------------------------------------------
# Minimal self-test (manual)
# --------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    set_style()

    # Synthetic demo
    B = 283
    x = np.arange(B)
    mu = 0.01 + 0.002 * np.sin(2 * np.pi * x / 50.0)
    sigma = 0.0005 + 0.0002 * np.cos(2 * np.pi * x / 30.0) ** 2
    truth = mu + np.random.normal(0, sigma)

    f1 = plot_spectrum(mu, sigma=sigma, truth=truth, title="Demo Spectrum")
    savefig(f1, Path("artifacts/demo_spectrum.png"))

    y_true = np.random.normal(0, 1, size=1000)
    y_pred = y_true + np.random.normal(0, 0.2, size=1000)
    f2 = plot_parity(y_true, y_pred)
    savefig(f2, Path("artifacts/demo_parity.png"))

    f3 = plot_residuals(y_true, y_pred)
    savefig(f3, Path("artifacts/demo_residuals.png"))

    f4 = plot_uncertainty_calibration(mu=truth, sigma=sigma[: truth.size], y_true=truth)
    savefig(f4, Path("artifacts/demo_uncal.png"))

    s = np.sin(2 * np.pi * 0.05 * np.arange(512)) + 0.3 * np.random.randn(512)
    f5 = plot_fft_power(s, fs=1.0)
    savefig(f5, Path("artifacts/demo_fft.png"))
