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


def test_gaussian_nll_scalar_numpy():
    y = np.array([1.0])
    mu = np.array([1.5])
    sigma = np.array([0.5])
    # manual NLL: 0.5 * [ ln(2πσ²) + (y-μ)^2/σ² ]
    var = sigma ** 2
    manual = 0.5 * (np.log(2.0 * math.pi * var) + (y - mu) ** 2 / var)
    got = gaussian_nll(y, mu, sigma, reduction="mean")
    assert np.isclose(got, manual, rtol=1e-7, atol=0)


def test_gaussian_nll_batch_numpy_reductions():
    rng = np.random.default_rng(0)
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
    # mean ~ sum / count
    assert np.isclose(nll_mean, np.nansum(nll_none) / (N * D), rtol=1e-6)


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


def test_mae_rmse_basic_and_masks():
    y = np.array([[0.0, 2.0, 4.0]])
    mu = np.array([[0.0, 1.0, 1.0]])
    mask = np.array([[True, False, True]])  # ignore middle element
    # abs diffs: [0, 1, 3] -> masked -> [0, -, 3] mean -> (0+3)/2 = 1.5
    assert np.isclose(mae(y, mu, mask=mask), 1.5)
    # rmse on masked -> sqrt((0^2 + 3^2)/2) = sqrt(9/2)
    assert np.isclose(rmse(y, mu, mask=mask), math.sqrt(9.0 / 2.0))


def test_uncertainty_diagnostics():
    rng = np.random.default_rng(1)
    N, D = 8, 5
    mu = np.zeros((N, D))
    sigma = np.full((N, D), 0.1)
    # Sample y inside the 95% interval around mu with small noise
    y = mu + rng.normal(scale=0.05, size=(N, D))
    cov = coverage(y, mu, sigma, alpha=0.95)
    shp = sharpness(sigma)
    assert 0.8 <= cov <= 1.0  # usually high coverage for small noise, not guaranteed exact
    assert np.isclose(shp, 0.1)  # mean sigma


@pytest.mark.skipif("torch" not in globals(), reason="Torch not installed")
def test_torch_equivalence_to_numpy():
    try:
        import torch
    except Exception:
        pytest.skip("Torch not available")

    rng = np.random.default_rng(123)
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