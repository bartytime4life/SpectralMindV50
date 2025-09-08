# tests/train/test_metrics.py
from __future__ import annotations

import math
import numpy as np
import pytest

from spectramind.train.metrics import (
    gaussian_nll,
    challenge_gll,
    mae,
    rmse,
    coverage,
    sharpness,
)

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------- #
# gaussian_nll
# ----------------------------------------------------------------------------- #
def test_gaussian_nll_scalar_numpy():
    y = np.array([1.0])
    mu = np.array([1.5])
    sigma = np.array([0.5])
    # manual NLL: 0.5 * [ ln(2πσ²) + (y-μ)^2/σ² ]
    var = sigma ** 2
    manual = 0.5 * (np.log(2.0 * math.pi * var) + (y - mu) ** 2 / var)
    got = gaussian_nll(y, mu, sigma, reduction="mean")
    assert np.isclose(got, manual, rtol=1e-7, atol=0.0)


def test_gaussian_nll_batch_numpy_reductions():
    rng = _rng(7)
    N, D = 4, 6
    y = rng.normal(size=(N, D))
    mu = y + rng.normal(scale=0.1, size=(N, D))
    sigma = np.full((N, D), 0.2)

    nll_mean = gaussian_nll(y, mu, sigma, reduction="mean")
    nll_sum = gaussian_nll(y, mu, sigma, reduction="sum")
    nll_none = gaussian_nll(y, mu, sigma, reduction="none")

    assert np.isfinite(nll_mean)
    assert np.isfinite(nll_sum)
    assert nll_none.shape == (N, D)
    # mean ≈ sum / count
    assert np.isclose(nll_mean, np.nansum(nll_none) / (N * D), rtol=1e-6)
    assert np.isclose(nll_sum, np.nansum(nll_none), rtol=1e-6)


def test_gaussian_nll_broadcasting_numpy():
    # y, mu broadcast across last dim; sigma is scalar broadcast
    y = np.array([[0.0, 1.0, 2.0]])
    mu = np.array([[0.5, 1.5, 1.0]])
    sigma = np.array(0.2)  # scalar broadcast
    # Should run and return appropriate shape with reduction="none"
    out = gaussian_nll(y, mu, sigma, reduction="none")
    assert out.shape == y.shape
    # other reductions should be scalar
    assert np.isscalar(gaussian_nll(y, mu, sigma, reduction="mean"))
    assert np.isscalar(gaussian_nll(y, mu, sigma, reduction="sum"))


# ----------------------------------------------------------------------------- #
# challenge_gll
# ----------------------------------------------------------------------------- #
def test_challenge_gll_fgs1_weighting_effect():
    # Two-bin spectrum: bin0 is FGS1
    # Make an error only in bin0 so weighting dominates
    y = np.array([[0.0, 0.0]])
    mu = np.array([[1.0, 0.0]])  # error only at bin 0
    sigma = np.array([[1.0, 1.0]])

    base = gaussian_nll(y, mu, sigma, reduction="mean")
    w58 = challenge_gll(y, mu, sigma, fgs1_weight=58.0, reduction="mean")
    w10 = challenge_gll(y, mu, sigma, fgs1_weight=10.0, reduction="mean")

    # Increasing FGS1 weight should increase weighted mean (since only bin0 has loss)
    assert w58 > w10 > base

    # With two bins, weighted mean = (w*nll0 + 1*nll1)/(w+1) = (w*nll0)/(w+1)
    # nll1=0 here; compare to exact formula using nll0 extracted from 'none'
    nll_elem = gaussian_nll(y, mu, sigma, reduction="none")
    nll0 = float(nll_elem[0, 0])
    expected = (58.0 * nll0) / (58.0 + 1.0)
    assert np.isclose(w58, expected, rtol=1e-6)


# ----------------------------------------------------------------------------- #
# point metrics
# ----------------------------------------------------------------------------- #
def test_mae_rmse_basic_and_masks():
    y = np.array([[0.0, 2.0, 4.0]])
    mu = np.array([[0.0, 1.0, 1.0]])
    mask = np.array([[True, False, True]])  # ignore middle element
    # abs diffs: [0, 1, 3] -> masked -> [0, -, 3] mean -> (0+3)/2 = 1.5
    assert np.isclose(mae(y, mu, mask=mask), 1.5)
    # rmse on masked -> sqrt((0^2 + 3^2)/2) = sqrt(9/2)
    assert np.isclose(rmse(y, mu, mask=mask), math.sqrt(9.0 / 2.0))


# ----------------------------------------------------------------------------- #
# uncertainty diagnostics
# ----------------------------------------------------------------------------- #
def test_uncertainty_diagnostics_deterministic_inside_interval():
    # Make y guaranteed to be inside the 95% interval to avoid stochastic flakiness.
    N, D = 8, 5
    mu = np.zeros((N, D))
    sigma = np.full((N, D), 0.1)
    # 95% interval ≈ μ ± 1.96σ; choose y within ±0.5σ
    y = mu + 0.5 * sigma
    cov = coverage(y, mu, sigma, alpha=0.95)
    shp = sharpness(sigma)
    assert np.isclose(cov, 1.0, rtol=0, atol=0)  # all points inside interval
    assert np.isclose(shp, 0.1)  # mean sigma


# ----------------------------------------------------------------------------- #
# torch parity
# ----------------------------------------------------------------------------- #
@pytest.mark.skipif(not _has_torch(), reason="Torch not installed")
def test_torch_equivalence_to_numpy():
    import torch  # type: ignore

    rng = _rng(123)
    N, D = 3, 7
    y_np = rng.normal(size=(N, D))
    mu_np = y_np + rng.normal(scale=0.1, size=(N, D))
    sigma_np = np.full((N, D), 0.2)

    y_t = torch.tensor(y_np, dtype=torch.float64)
    mu_t = torch.tensor(mu_np, dtype=torch.float64)
    sigma_t = torch.tensor(sigma_np, dtype=torch.float64)

    # gaussian_nll
    g_np = gaussian_nll(y_np, mu_np, sigma_np, reduction="mean")
    g_t = gaussian_nll(y_t, mu_t, sigma_t, reduction="mean")
    assert np.isclose(float(g_t), float(g_np), rtol=1e-6)

    # challenge_gll
    cg_np = challenge_gll(y_np, mu_np, sigma_np, reduction="mean", fgs1_weight=58.0)
    cg_t = challenge_gll(y_t, mu_t, sigma_t, reduction="mean", fgs1_weight=58.0)
    assert np.isclose(float(cg_t), float(cg_np), rtol=1e-6)

    # mae/rmse
    assert np.isclose(float(mae(y_t, mu_t)), float(mae(y_np, mu_np)), rtol=1e-6)
    assert np.isclose(float(rmse(y_t, mu_t)), float(rmse(y_np, mu_np)), rtol=1e-6)

    # coverage/sharpness
    cov_np = coverage(y_np, mu_np, sigma_np, alpha=0.95)
    cov_t = coverage(y_t, mu_t, sigma_t, alpha=0.95)
    assert np.isclose(float(cov_t), float(cov_np), rtol=1e-6)

    shp_np = sharpness(sigma_np)
    shp_t = sharpness(sigma_t)
    assert np.isclose(float(shp_t), float(shp_np), rtol=1e-6)
