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


def _manual_gaussian_nll(y, mu, sigma):
    var = sigma ** 2
    return 0.5 * (np.log(2.0 * math.pi * var) + (y - mu) ** 2 / var)


# ----------------------------------------------------------------------------- #
# gaussian_nll — core correctness & reductions
# ----------------------------------------------------------------------------- #
def test_gaussian_nll_scalar_numpy():
    y = np.array([1.0])
    mu = np.array([1.5])
    sigma = np.array([0.5])

    manual = _manual_gaussian_nll(y, mu, sigma)
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
    out = gaussian_nll(y, mu, sigma, reduction="none")
    assert out.shape == y.shape
    assert np.isscalar(gaussian_nll(y, mu, sigma, reduction="mean"))
    assert np.isscalar(gaussian_nll(y, mu, sigma, reduction="sum"))


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_gaussian_nll_1d_2d_interop(reduction: str):
    # 1-D vectors should work and be consistent with 2-D (N=1) use
    y1 = np.array([0.0, 1.0, 2.0])
    mu1 = np.array([0.1, 1.1, 2.1])
    sigma1 = np.array([0.2, 0.2, 0.2])

    y2 = y1[None, :]
    mu2 = mu1[None, :]
    sigma2 = sigma1[None, :]

    g1 = gaussian_nll(y1, mu1, sigma1, reduction=reduction)
    g2 = gaussian_nll(y2, mu2, sigma2, reduction=reduction)
    if reduction == "none":
        assert g1.shape == (3,)
        assert g2.shape == (1, 3)
        np.testing.assert_allclose(g1, g2[0], rtol=1e-7)
    else:
        assert np.isscalar(g1)
        assert np.isscalar(g2)
        assert np.isclose(float(g1), float(g2), rtol=1e-7)


def test_gaussian_nll_invalid_reduction_raises():
    y = np.array([0.0])
    mu = np.array([0.0])
    sigma = np.array([0.1])
    with pytest.raises((ValueError, AssertionError, KeyError, TypeError)):
        _ = gaussian_nll(y, mu, sigma, reduction="avg")  # not supported


def test_gaussian_nll_zero_or_negative_sigma_handling():
    y = np.array([0.0, 1.0])
    mu = np.array([0.0, 1.0])
    # sigma includes a zero and a negative to test guarding
    sigma = np.array([0.0, -0.1])
    # Accept either a clear exception or finite guarded result (eps-clamping)
    try:
        out = gaussian_nll(y, mu, sigma, reduction="none")
        assert np.all(np.isfinite(out)), "Expected finite results if eps-clamped"
    except Exception as e:  # noqa: BLE001
        assert isinstance(e, (ValueError, AssertionError, FloatingPointError, ZeroDivisionError))


# ----------------------------------------------------------------------------- #
# challenge_gll — weighting behavior and reductions
# ----------------------------------------------------------------------------- #
def test_challenge_gll_fgs1_weighting_effect_two_bin():
    # Two-bin spectrum: bin0 is FGS1; make error only in bin0 so weighting dominates
    y = np.array([[0.0, 0.0]])
    mu = np.array([[1.0, 0.0]])  # error only at bin 0
    sigma = np.array([[1.0, 1.0]])

    base = gaussian_nll(y, mu, sigma, reduction="mean")
    w58 = challenge_gll(y, mu, sigma, fgs1_weight=58.0, reduction="mean")
    w10 = challenge_gll(y, mu, sigma, fgs1_weight=10.0, reduction="mean")

    assert w58 > w10 > base

    nll_elem = gaussian_nll(y, mu, sigma, reduction="none")
    nll0 = float(nll_elem[0, 0])
    expected = (58.0 * nll0) / (58.0 + 1.0)
    assert np.isclose(w58, expected, rtol=1e-6)


def test_challenge_gll_multi_bin_exact_formula():
    """
    Validate exact formula on a 1xK example where only specific bins have error.
    Weighting should be equivalent to replacing the first-bin weight.
    """
    K = 5
    y = np.zeros((1, K))
    mu = np.zeros((1, K))
    sigma = np.ones((1, K))
    # Introduce error at bins 0 (FGS1) and 3
    mu[0, 0] = 1.0
    mu[0, 3] = 2.0
    nll = gaussian_nll(y, mu, sigma, reduction="none")[0]
    # Weighted mean = (w*nll0 + sum(nll1..))/(w + (K-1))
    for w in (1.0, 10.0, 58.0, 100.0):
        got = challenge_gll(y, mu, sigma, fgs1_weight=w, reduction="mean")
        expected = (w * nll[0] + np.sum(nll[1:])) / (w + (K - 1))
        assert np.isclose(got, expected, rtol=1e-8)


def test_challenge_gll_reductions_shapes_and_types():
    rng = _rng(41)
    B, K = 3, 7
    y = rng.normal(size=(B, K))
    mu = rng.normal(size=(B, K))
    sigma = np.full((B, K), 0.5)

    none = challenge_gll(y, mu, sigma, reduction="none", fgs1_weight=58.0)
    mean = challenge_gll(y, mu, sigma, reduction="mean", fgs1_weight=58.0)
    sm = challenge_gll(y, mu, sigma, reduction="sum", fgs1_weight=58.0)
    assert none.shape == (B, K)
    assert np.isscalar(mean)
    assert np.isscalar(sm)


# ----------------------------------------------------------------------------- #
# point metrics — mae/rmse and masks
# ----------------------------------------------------------------------------- #
def test_mae_rmse_basic_and_masks():
    y = np.array([[0.0, 2.0, 4.0]])
    mu = np.array([[0.0, 1.0, 1.0]])
    mask = np.array([[True, False, True]])  # ignore middle element
    # abs diffs: [0, 1, 3] -> masked -> [0, -, 3] mean -> (0+3)/2 = 1.5
    assert np.isclose(mae(y, mu, mask=mask), 1.5)
    # rmse on masked -> sqrt((0^2 + 3^2)/2) = sqrt(9/2)
    assert np.isclose(rmse(y, mu, mask=mask), math.sqrt(9.0 / 2.0))


def test_mae_rmse_broadcast_and_1d():
    y = np.array([0.0, 1.0, 2.0])
    mu = np.array([[0.0, 2.0, 1.0]])  # will broadcast across batch if supported
    m = mae(y, mu)
    r = rmse(y, mu)
    assert np.isscalar(m)
    assert np.isscalar(r)
    assert m >= 0 and r >= 0


# ----------------------------------------------------------------------------- #
# uncertainty diagnostics — coverage & sharpness
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


def test_coverage_exact_fraction_constructed():
    """
    Build a toy array where exactly half of the elements are inside the interval and
    half are outside (deterministically), then assert coverage=0.5.
    """
    alpha = 0.90  # z ≈ 1.64485
    z = 1.6448536269514722
    # y: two bins, inside/outside toggled
    N, D = 10, 2
    mu = np.zeros((N, D))
    sigma = np.ones((N, D)) * 0.2
    y = np.zeros((N, D))
    # half inside: +0.5σ; half outside: +2.5σ (beyond z=1.64)
    y[: N // 2] = mu[: N // 2] + 0.5 * sigma[: N // 2]
    y[N // 2 :] = mu[N // 2 :] + 2.5 * sigma[N // 2 :]
    cov = coverage(y, mu, sigma, alpha=alpha)
    assert np.isclose(cov, 0.5, atol=1e-12)


def test_sharpness_reductions_and_shapes():
    # Depending on implementation, sharpness may support reduction; if not, basic mean.
    sigma = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]])
    try:
        s_none = sharpness(sigma, reduction="none")  # type: ignore[call-arg]
        assert s_none.shape == sigma.shape
        s_mean = sharpness(sigma, reduction="mean")  # type: ignore[call-arg]
        assert np.isclose(float(s_mean), float(np.mean(sigma)))
        s_sum = sharpness(sigma, reduction="sum")  # type: ignore[call-arg]
        assert np.isclose(float(s_sum), float(np.sum(sigma)))
    except TypeError:
        # If reduction not supported, default should equal mean
        s = sharpness(sigma)
        assert np.isclose(float(s), float(np.mean(sigma)))


# ----------------------------------------------------------------------------- #
# torch parity (optional)
# ----------------------------------------------------------------------------- #
@pytest.mark.skipif(not _has_torch(), reason="Torch not installed")
def test_torch_equivalence_to_numpy():
    import torch  # type: ignore

    rng = _rng(123)
    N, D = 3, 7
    y_np = rng.normal(size=(N, D))
    mu_np = y_np + rng.normal(scale=0.1, size=(N, D))
    sigma_np = np.full((N, D), 0.2)

    for dtype in (torch.float32, torch.float64):
        y_t = torch.tensor(y_np, dtype=dtype)
        mu_t = torch.tensor(mu_np, dtype=dtype)
        sigma_t = torch.tensor(sigma_np, dtype=dtype)

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