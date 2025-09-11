from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# Headless rendering — safe for CI/Kaggle/servers
import matplotlib

# Force non-interactive backend deterministically
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402

try:  # optional aesthetics
    import seaborn as sns  # type: ignore

    _HAS_SEABORN = True
except Exception:  # pragma: no cover
    _HAS_SEABORN = False

__all__ = [
    "set_style",
    "style",
    "savefig",
    "plot_training_curves",
    "plot_parity",
    "plot_residuals",
    "plot_qq",
    "plot_spectrum",
    "plot_spectra_grid",
    "plot_uncertainty_calibration",
    "plot_fft_power",
    "FigureSpec",
    "save_all",
    "erf",
    "erfinv",
]

# --------------------------------------------------------------------------
# Palettes & Style
# --------------------------------------------------------------------------

# Color-blind safe palette (Okabe–Ito)
_PALETTE = dict(
    blue="#0072B2",
    orange="#E69F00",
    green="#009E73",
    red="#D55E00",
    purple="#CC79A7",
    brown="#8B4513",
    yellow="#F0E442",
    gray="#999999",
)


def _rc_defaults(font_scale: float = 1.0) -> dict[str, Any]:
    # Conservative RCs for reproducible rendering in CI/Kaggle
    return {
        "figure.dpi": 120,
        "savefig.dpi": 144,
        "axes.titlesize": "medium",
        "axes.labelsize": "medium",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "legend.frameon": False,
        "font.size": max(6, round(10 * font_scale)),
        "axes.prop_cycle": plt.cycler(
            color=[
                _PALETTE["blue"],
                _PALETTE["orange"],
                _PALETTE["green"],
                _PALETTE["red"],
                _PALETTE["purple"],
                _PALETTE["gray"],
                _PALETTE["brown"],
                _PALETTE["yellow"],
            ]
        ),
    }


def set_style(*, use_seaborn: Optional[bool] = None, font_scale: float = 1.0) -> None:
    """
    Apply a clean, publication-ready style.

    Args:
        use_seaborn: Force seaborn or pure Matplotlib. If None, autodetect.
        font_scale: Global font scaling factor.
    """
    if use_seaborn is None:
        use_seaborn = _HAS_SEABORN

    rc = _rc_defaults(font_scale=font_scale)

    if use_seaborn:
        try:
            sns.set_theme(context="notebook", style="whitegrid", font_scale=font_scale, rc=rc)  # type: ignore
        except Exception:
            plt.rcParams.update(rc)
    else:
        plt.rcParams.update(rc)


@contextmanager
def style(*, use_seaborn: Optional[bool] = None, font_scale: float = 1.0):
    """Context manager to apply style temporarily (restores previous rc)."""
    with plt.rc_context():
        set_style(use_seaborn=use_seaborn, font_scale=font_scale)
        yield


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def savefig(
    fig: plt.Figure,
    path: Path | str,
    *,
    tight: bool = True,
    transparent: bool = False,
    also_svg: bool = False,
    dpi: Optional[int] = None,
) -> Path:
    """
    Safe figure save with parent creation and sane defaults.

    Returns:
        The primary saved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if tight:
            fig.tight_layout()
        fig.savefig(
            path,
            bbox_inches="tight" if tight else None,
            transparent=transparent,
            dpi=dpi,
        )
        if also_svg:
            svg_path = path.with_suffix(".svg")
            fig.savefig(
                svg_path,
                bbox_inches="tight" if tight else None,
                transparent=transparent,
                dpi=dpi,
            )
    finally:
        # Always close the figure to avoid memory/file descriptor leaks in loops
        plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------

def _as_1d(a: Any) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim == 0:
        x = x[None]
    return x.reshape(-1)


def _finite_mask(first: np.ndarray, *rest: np.ndarray) -> np.ndarray:
    """Return mask where all arrays are finite; aligns by shortest length."""
    xs = [np.asarray(first, dtype=float)] + [np.asarray(r, dtype=float) for r in rest]
    n = min(x.size for x in xs) if xs else 0
    if n == 0:
        return np.zeros(0, dtype=bool)
    xs = [x[:n].reshape(-1) for x in xs]
    mask = np.ones(n, dtype=bool)
    for x in xs:
        mask &= np.isfinite(x)
    return mask


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average (alpha in (0,1], higher = less smoothing)."""
    if not (0 < alpha <= 1):
        return values.astype(float, copy=True)
    out = np.empty_like(values, dtype=float)
    m = 0.0
    for i, v in enumerate(values.astype(float)):
        m = alpha * v if i == 0 else (1 - alpha) * m + alpha * v
        out[i] = m
    return out


def _auto_bins(x: np.ndarray, max_bins: int = 100) -> int:
    """Freedman–Diaconis bin count with an upper cap. Falls back to 10."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 10
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(max(q75 - q25, np.finfo(float).eps))
    # Avoid division by zero; bound h away from 0
    h = max(2 * iqr * n ** (-1 / 3), 1e-12)
    bins = int(np.clip(np.ceil((x.max() - x.min()) / h), 10, max_bins))
    return bins


# --------------------------------------------------------------------------
# Core diagnostic plots
# --------------------------------------------------------------------------

def plot_training_curves(
    history: Mapping[str, Sequence[float]] | Mapping[str, np.ndarray],
    *,
    title: str | None = "Training Curves",
    max_cols: int = 2,
    ema_alpha: Optional[float] = None,
) -> plt.Figure:
    """
    Plot training curves from a Keras/Lightning-style history dict, e.g.:
        {"loss": [...], "val_loss": [...], "metric": [...], "val_metric": [...]}

    Args:
        ema_alpha: if provided, smooth each series by EMA with this alpha (0,1].
    """
    # normalize arrays
    hist = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in history.items() if v is not None}

    # Group into paired (train,val) metrics
    keys = sorted(hist.keys())
    groups: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for k in keys:
        if k.startswith("val_"):
            base = k[4:]
            tr, vl = groups.get(base, (None, None))
            groups[base] = (tr, k)
        else:
            base = k
            tr, vl = groups.get(base, (None, None))
            groups[base] = (k, vl)

    n = len(groups)
    cols = min(max_cols, max(1, n))
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), squeeze=False)

    for ax, (name, (train_k, val_k)) in zip(axes.flat, groups.items()):
        if train_k is not None:
            y = hist[train_k]
            if ema_alpha is not None:
                y = _ema(y, ema_alpha)
            ax.plot(np.arange(len(y)), y, label=train_k, lw=2)
        if val_k is not None:
            yv = hist[val_k]
            if ema_alpha is not None:
                yv = _ema(yv, ema_alpha)
            ax.plot(np.arange(len(yv)), yv, label=val_k, lw=2)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.legend(loc="best")

    # hide spare axes
    for j in range(len(groups), rows * cols):
        axes.flat[j].axis("off")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_parity(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    title: str | None = "Parity Plot",
    hexbin_threshold: int = 5000,
) -> plt.Figure:
    """
    Parity (y_true vs y_pred) with identity reference.
    Uses hexbin when there are many points for clarity.
    """
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    n = min(yt.size, yp.size)
    yt, yp = yt[:n], yp[:n]
    mask = _finite_mask(yt, yp)
    yt, yp = yt[mask], yp[mask]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    if yt.size == 0:
        ax.text(0.5, 0.5, "No finite points", ha="center", va="center")
        ax.axis("off")
        return fig

    lims = [float(np.nanmin([yt.min(), yp.min()])), float(np.nanmax([yt.max(), yp.max()]))]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6, label="identity")

    if yt.size >= hexbin_threshold:
        hb = ax.hexbin(yt, yp, gridsize=40, cmap="viridis", mincnt=1)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("count")
    else:
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
    bins: Optional[int] = None,
    title: str | None = "Residuals",
) -> plt.Figure:
    """
    Histogram of residuals and a residuals-vs-predicted panel.
    """
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    n = min(yt.size, yp.size)
    yt, yp = yt[:n], yp[:n]
    mask = _finite_mask(yt, yp)
    r = (yp - yt)[mask]
    yp = yp[mask]

    if r.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No finite residuals", ha="center", va="center")
        ax.axis("off")
        return fig

    if bins is None:
        bins = _auto_bins(r)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].hist(r, bins=bins, alpha=0.85, color=_PALETTE["blue"])
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
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    if n == 0:
        ax.text(0.5, 0.5, "No finite residuals", ha="center", va="center")
        ax.axis("off")
        return fig

    # Theoretical quantiles from standard normal
    p = (np.arange(1, n + 1) - 0.5) / n
    q = np.sqrt(2.0) * erfinv(2 * p - 1)

    lims = [min(q.min(), r.min()), max(q.max(), r.max())]
    ax.scatter(q, r, s=16, alpha=0.7)
    ax.plot(lims, lims, "k--", lw=1, alpha=0.7)
    ax.set_xlabel("Theoretical quantiles (N(0,1))")
    ax.set_ylabel("Empirical quantiles (residuals)")
    ax.set_title(title or "")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def erfinv(x: np.ndarray | float) -> np.ndarray:
    """
    Vectorized inverse error function via SciPy if available, otherwise
    Winitzki approximation (accurate for |x| < 1).
    """
    try:  # pragma: no cover
        from scipy.special import erfinv as _erfinv  # type: ignore

        return _erfinv(x)
    except Exception:
        pass
    x = np.asarray(x, dtype=float)
    # Clip to open interval to avoid log blow-ups at ±1
    eps = 1e-12
    x = np.clip(x, -1 + eps, 1 - eps)
    a = 0.147
    sgn = np.sign(x)
    ln = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    second = ln / a
    inside = np.sqrt(np.maximum(first * first - second, 0.0))
    return sgn * np.sqrt(np.maximum(inside - first, 0.0))


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
    Plot a single spectrum with optional uncertainty and truth.
    """
    mu = _as_1d(mu)
    B = mu.size
    x = np.arange(B) if wavelength is None else _as_1d(wavelength)[:B]

    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.plot(x, mu, lw=2, label="μ (pred)", color=_PALETTE["blue"])
    if sigma is not None:
        s = np.clip(_as_1d(sigma)[:B], 0.0, np.inf)
        ax.fill_between(x, mu - s, mu + s, color=_PALETTE["blue"], alpha=0.20, label="±σ")
    if truth is not None:
        t = _as_1d(truth)[:B]
        ax.plot(x, t, lw=1.5, color=_PALETTE["orange"], alpha=0.9, label="truth")

    ax.set_xlabel("Bin" if wavelength is None else "Wavelength")
    ax.set_ylabel("Transit depth (a.u.)")
    ax.set_title(title or "")
    if band_labels is not None and len(band_labels) >= B:
        ax.set_xticks(x)
        ax.set_xticklabels(list(band_labels)[:B], rotation=90, fontsize=7)
    ax.legend(loc="best", ncols=3)
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
    x = np.arange(B) if wavelength is None else _as_1d(wavelength)[:B]
    ax.plot(x, mu, lw=1.8, label="μ", color=_PALETTE["blue"])
    if sigma is not None:
        s = np.clip(_as_1d(sigma)[:B], 0.0, np.inf)
        ax.fill_between(x, mu - s, mu + s, alpha=0.2, label="±σ", color=_PALETTE["blue"])
    if truth is not None:
        t = _as_1d(truth)[:B]
        ax.plot(x, t, lw=1.4, alpha=0.9, label="truth", color=_PALETTE["orange"])
    ax.set_xlabel("Bin" if wavelength is None else "Wavelength")
    ax.set_ylabel("Transit depth (a.u.)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


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
    Grid of spectra (e.g., first N predictions).
    """
    mu = np.asarray(mu, dtype=float)
    N = int(mu.shape[0])
    ncols = max(1, ncols)
    nrows = max(1, math.ceil(N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * (ncols / 3), 3.1 * nrows), squeeze=False)

    s_arr = None if sigma is None else np.asarray(sigma, dtype=float)
    t_arr = None if truth is None else np.asarray(truth, dtype=float)

    for i in range(N):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        _plot_spectrum_ax(
            ax,
            mu[i],
            sigma=None if s_arr is None else s_arr[i],
            truth=None if t_arr is None else t_arr[i],
            wavelength=wavelength,
            title=titles[i] if titles and i < len(titles) else None,
        )

    # hide any unused axes
    for j in range(N, nrows * ncols):
        rr, cc = j // ncols, j % ncols
        axes[rr][cc].axis("off")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Uncertainty diagnostics
# --------------------------------------------------------------------------

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
        xa = np.abs(x)
        t = 1.0 / (1.0 + p * xa)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-(xa * xa))
        return sgn * y


def plot_uncertainty_calibration(
    mu: Sequence[float],
    sigma: Sequence[float],
    y_true: Sequence[float],
    *,
    bins: Optional[int] = None,
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
    n = min(m.size, s.size, y.size)
    m, s, y = m[:n], s[:n], y[:n]
    mask = _finite_mask(m, s, y)
    if not np.any(mask):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
        ax.axis("off")
        return fig

    z = (y[mask] - m[mask]) / s[mask]
    pit = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

    if bins is None:
        bins = _auto_bins(z)

    fig = plt.figure(figsize=(12.2, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    # z histogram
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(z, bins=bins, density=True, alpha=0.85, color=_PALETTE["blue"], label="empirical z")
    # standard normal PDF overlay
    zz = np.linspace(-4, 4, 400)
    ax0.plot(zz, np.exp(-0.5 * zz**2) / np.sqrt(2 * np.pi), "k--", lw=1.5, label="N(0,1)")
    ax0.set_title("z = (y-μ)/σ")
    ax0.set_xlabel("z")
    ax0.set_ylabel("density")
    ax0.legend()

    # PIT histogram
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(pit, bins=max(10, bins // 2), range=(0, 1), density=True, alpha=0.85, color=_PALETTE["green"])
    ax1.axhline(1.0, color="k", lw=1, ls="--")
    ax1.set_title("PIT = Φ(z)")
    ax1.set_xlabel("u")
    ax1.set_ylabel("density")

    # Coverage curve
    ax2 = fig.add_subplot(gs[0, 2])
    alphas = np.linspace(0.01, 0.99, 50)
    # central (1-α) intervals → threshold in z-space is Φ^{-1}(1-α/2)
    thresh = np.sqrt(2.0) * erfinv(2 * (1 - alphas / 2) - 1)
    emp = np.array([np.mean(np.abs(z) <= t) for t in thresh])
    nom = 1 - alphas
    ax2.plot(nom, emp, lw=2, label="empirical", color=_PALETTE["blue"])
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


def erfinv(x: np.ndarray | float) -> np.ndarray:
    """
    Vectorized inverse error function via SciPy if available, otherwise
    Winitzki approximation (accurate for |x| < 1).
    """
    try:  # pragma: no cover
        from scipy.special import erfinv as _erfinv  # type: ignore

        return _erfinv(x)
    except Exception:
        pass
    x = np.asarray(x, dtype=float)
    # Clip to open interval to avoid log blow-ups at ±1
    eps = 1e-12
    x = np.clip(x, -1 + eps, 1 - eps)
    a = 0.147
    sgn = np.sign(x)
    ln = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    second = ln / a
    inside = np.sqrt(np.maximum(first * first - second, 0.0))
    return sgn * np.sqrt(np.maximum(inside - first, 0.0))


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
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    if N == 0:
        ax.text(0.5, 0.5, "Empty signal", ha="center", va="center")
        ax.axis("off")
        return fig

    # Detrend (remove mean)
    x = x - x.mean()

    # FFT
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(N, d=1.0 / fs)  # Hz
    Pxx = (np.abs(X) ** 2) / max(N, 1)

    ax.plot(f, Pxx, lw=1.5, color=_PALETTE["blue"])
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
    also_svg: bool = False
    dpi: Optional[int] = None


def save_all(figs: Iterable[FigureSpec]) -> None:
    """Save a collection of figures with consistent behavior."""
    for spec in figs:
        savefig(
            spec.fig,
            spec.path,
            tight=spec.tight,
            transparent=spec.transparent,
            also_svg=spec.also_svg,
            dpi=spec.dpi,
        )


# --------------------------------------------------------------------------
# Minimal self-test (manual)
# --------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    with style(font_scale=1.0):
        # Synthetic demo
        B = 283
        x = np.arange(B)
        mu = 0.01 + 0.002 * np.sin(2 * np.pi * x / 50.0)
        sigma = 0.0005 + 0.0002 * np.cos(2 * np.pi * x / 30.0) ** 2
        truth = mu + np.random.normal(0, sigma)

        f1 = plot_spectrum(mu, sigma=sigma, truth=truth, title="Demo Spectrum")
        savefig(f1, Path("artifacts/demo_spectrum.png"), also_svg=True)

        y_true = np.random.normal(0, 1, size=10000)
        y_pred = y_true + np.random.normal(0, 0.2, size=10000)
        f2 = plot_parity(y_true, y_pred)
        savefig(f2, Path("artifacts/demo_parity.png"))

        f3 = plot_residuals(y_true, y_pred)
        savefig(f3, Path("artifacts/demo_residuals.png"))

        f4 = plot_uncertainty_calibration(mu=truth, sigma=sigma[: truth.size], y_true=truth)
        savefig(f4, Path("artifacts/demo_uncal.png"))

        s = np.sin(2 * np.pi * 0.05 * np.arange(512)) + 0.3 * np.random.randn(512)
        f5 = plot_fft_power(s, fs=1.0)
        savefig(f5, Path("artifacts/demo_fft.png"))
