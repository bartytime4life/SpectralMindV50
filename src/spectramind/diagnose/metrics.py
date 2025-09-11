from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
    pd = None  # type: ignore


def _require_np_pd() -> None:
    if np is None or pd is None:
        raise RuntimeError("numpy and pandas are required for diagnostics metrics")


def compute_sanity_checks(preds_wide: "pd.DataFrame") -> Dict[str, float]:
    """
    Fast plausibility checks on μ and σ:
      - fraction of non-finite values
      - fraction of negative sigma
      - basic stats for FGS1 (mu_000)
    """
    _require_np_pd()
    mu_cols = [c for c in preds_wide.columns if c.startswith("mu_")]
    sg_cols = [c for c in preds_wide.columns if c.startswith("sigma_")]
    out: Dict[str, float] = {}

    mu = preds_wide[mu_cols].to_numpy(dtype=float)
    sg = preds_wide[sg_cols].to_numpy(dtype=float)

    nonfinite_mu = ~np.isfinite(mu)
    nonfinite_sg = ~np.isfinite(sg)
    out["frac_nonfinite_mu"] = float(nonfinite_mu.sum() / mu.size) if mu.size else 0.0
    out["frac_nonfinite_sigma"] = float(nonfinite_sg.sum() / sg.size) if sg.size else 0.0
    out["frac_neg_sigma"] = float((sg < 0).sum() / sg.size) if sg.size else 0.0

    if "mu_000" in preds_wide.columns:
        x = preds_wide["mu_000"].astype(float).to_numpy()
        out["mu_000_mean"] = float(np.nanmean(x)) if x.size else 0.0
        out["mu_000_std"] = float(np.nanstd(x)) if x.size else 0.0
    return out


def compute_smoothness_score(preds_wide: "pd.DataFrame") -> float:
    """
    A small smoothness proxy: average L2 norm of second-differences of μ rows.
    Lower is smoother. Scale is data-dependent; used for relative comparisons.
    """
    _require_np_pd()
    mu_cols = sorted([c for c in preds_wide.columns if c.startswith("mu_")])
    if not mu_cols:
        return 0.0
    mu = preds_wide[mu_cols].to_numpy(dtype=float)  # [N, B]
    if mu.shape[1] < 3:
        return 0.0
    d2 = mu[:, 2:] - 2.0 * mu[:, 1:-1] + mu[:, :-2]  # second difference along bins
    # row-wise L2, then mean
    row_l2 = np.sqrt((d2 ** 2).sum(axis=1))
    return float(np.nanmean(row_l2)) if row_l2.size else 0.0


def compute_coverage(
    preds_wide: "pd.DataFrame",
    truth_narrow: Optional["pd.DataFrame"],
    *,
    k: float = 1.0,
) -> float:
    """
    Empirical coverage: fraction of ground-truth points falling within μ ± kσ.
    If no truth is provided, returns NaN.
    """
    _require_np_pd()
    if truth_narrow is None or truth_narrow.empty:
        return float("nan")

    # Prepare lookup for preds: id → row (mu,sigma arrays)
    mu_cols = sorted([c for c in preds_wide.columns if c.startswith("mu_")])
    sg_cols = sorted([c for c in preds_wide.columns if c.startswith("sigma_")])
    if not mu_cols or not sg_cols:
        return float("nan")

    preds_idx = preds_wide.set_index("id")
    total = 0
    hit = 0
    for rid, group in truth_narrow.groupby("id"):
        if rid not in preds_idx.index:
            continue
        prow = preds_idx.loc[rid]
        mu = prow[mu_cols].to_numpy(dtype=float)
        sg = prow[sg_cols].to_numpy(dtype=float)

        bins = group["bin"].astype(int).to_numpy()
        y = group["target"].astype(float).to_numpy()

        # guard indices
        mask = (bins >= 0) & (bins < len(mu))
        if not mask.any():
            continue
        bins = bins[mask]
        y = y[mask]

        lower = mu[bins] - k * sg[bins]
        upper = mu[bins] + k * sg[bins]
        total += len(y)
        hit += int(((y >= lower) & (y <= upper)).sum())

    return float(hit / total) if total else float("nan")


def compute_gll_simple(
    preds_wide: "pd.DataFrame",
    truth_narrow: Optional["pd.DataFrame"],
    *,
    weight_fgs1: float = 58.0,
) -> float:
    """
    A simple Gaussian log-likelihood proxy:
      sum over (id,bin) of [ -0.5*log(2πσ^2) - (y-μ)^2 / (2σ^2) ],
    with FGS1 (bin 0) up-weighted per challenge spec (~58×). Returns mean over all points.
    If no truth is available, returns NaN.
    """
    _require_np_pd()
    if truth_narrow is None or truth_narrow.empty:
        return float("nan")

    mu_cols = sorted([c for c in preds_wide.columns if c.startswith("mu_")])
    sg_cols = sorted([c for c in preds_wide.columns if c.startswith("sigma_")])
    if not mu_cols or not sg_cols:
        return float("nan")

    preds_idx = preds_wide.set_index("id")
    total_weight = 0.0
    agg = 0.0
    ln2pi = np.log(2.0 * np.pi)

    for rid, group in truth_narrow.groupby("id"):
        if rid not in preds_idx.index:
            continue
        prow = preds_idx.loc[rid]
        mu = prow[mu_cols].to_numpy(dtype=float)
        sg = prow[sg_cols].to_numpy(dtype=float)

        bins = group["bin"].astype(int).to_numpy()
        y = group["target"].astype(float).to_numpy()

        # valid subset
        mask = (bins >= 0) & (bins < len(mu))
        if not mask.any():
            continue
        bins = bins[mask]
        y = y[mask]
        m = mu[bins]
        s = sg[bins]
        # avoid zero/negative sigma
        s = np.clip(s, 1e-12, None)

        w = np.ones_like(y, dtype=float)
        w[bins == 0] = weight_fgs1  # FGS1 emphasis

        nll = 0.5 * (ln2pi + np.log(s ** 2) + ((y - m) ** 2) / (s ** 2))
        agg += float((w * (-nll)).sum())
        total_weight += float(w.sum())

    return float(agg / total_weight) if total_weight > 0 else float("nan")
