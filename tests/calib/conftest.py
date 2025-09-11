# tests/fixtures/imaging.py
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pytest

try:  # optional torch tensors in tests
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Randomness: one session RNG, then spawn sub-generators per test/fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng_session() -> np.random.Generator:
    """Session RNG (stable). Override with SPECTRAMIND_TEST_SEED if set upstream."""
    seed = int(np.random.SeedSequence(42).entropy)  # deterministic default
    return np.random.default_rng(42)

@pytest.fixture
def rng(rng_session: np.random.Generator) -> np.random.Generator:
    """Fresh substream for each test (reproducible & isolated)."""
    # Spawn a child generator so fixture order can’t perturb other tests
    return np.random.Generator(np.random.PCG64(rng_session.integers(0, 2**63 - 1)))

# ---------------------------------------------------------------------------
# Utility builders
# ---------------------------------------------------------------------------

def _make_mesh(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:h, :w]
    return y.astype(np.float32), x.astype(np.float32)

def _gaussian_2d(y: np.ndarray, x: np.ndarray, cy: float, cx: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-(((x - cx) ** 2) + ((y - cy) ** 2)) / (2.0 * sigma**2))

# ---------------------------------------------------------------------------
# Core imaging fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image(rng: np.random.Generator) -> np.ndarray:
    """
    64×64 float32 image with gentle background plane + centered PSF + Gaussian noise.
    """
    h, w = 64, 64
    y, x = _make_mesh(h, w)
    bg = 100.0 + 0.05 * x + 0.03 * y
    psf = _gaussian_2d(y, x, cy=32.0, cx=32.0, sigma=3.0, amp=1000.0)
    noise = rng.normal(0.0, 2.0, size=(h, w)).astype(np.float32)
    return (bg + psf + noise).astype(np.float32)

@pytest.fixture
def master_dark(rng: np.random.Generator) -> np.ndarray:
    """
    64×64 float32 dark frame with low-freq stripe + small Gaussian noise.
    """
    h, w = 64, 64
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    pattern = 5.0 + 0.01 * np.sin(8.0 * np.pi * x)
    return (pattern + rng.normal(0.0, 0.2, size=(h, w))).astype(np.float32)

@pytest.fixture
def master_flat(rng: np.random.Generator) -> np.ndarray:
    """
    64×64 float32 flat field with mild vignetting, normalized to mean=1.
    """
    h, w = 64, 64
    y, x = _make_mesh(h, w)
    flat = 0.9 + 0.0005 * ((x - 32.0) ** 2 + (y - 32.0) ** 2)
    flat = flat / float(flat.mean())
    return flat.astype(np.float32)

@pytest.fixture
def frame_stack(rng: np.random.Generator) -> np.ndarray:
    """
    4×32×32 float32 stack with baseline + linear time trend + final-frame pulse + noise.
    Shape: [T, H, W]
    """
    t, h, w = 4, 32, 32
    y, x = _make_mesh(h, w)
    base = 50.0 + 0.02 * x  # gentle column gradient
    trend = np.linspace(0.0, 5.0, t, dtype=np.float32)[:, None, None]
    noise = rng.normal(0.0, 1.0, size=(t, h, w)).astype(np.float32)
    signal = np.zeros((t, h, w), dtype=np.float32)
    signal[-1] += 20.0
    return (base[None, :, :] + trend + signal + noise).astype(np.float32)

# ---------------------------------------------------------------------------
# Extra handy fixtures for calibration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def psf_kernel() -> np.ndarray:
    """
    9×9 normalized Gaussian PSF kernel (σ=1.5) for convolution tests.
    """
    k = 9
    y, x = _make_mesh(k, k)
    cy = cx = (k - 1) / 2.0
    kern = _gaussian_2d(y, x, cy, cx, sigma=1.5, amp=1.0).astype(np.float32)
    s = float(kern.sum())
    return kern / (s if s > 0 else 1.0)

@pytest.fixture
def bad_pixel_mask() -> np.ndarray:
    """
    64×64 boolean mask with a few hot/dead pixels toggled True.
    """
    m = np.zeros((64, 64), dtype=bool)
    m[5, 5] = True
    m[17, 43] = True
    m[60 - 1, 2] = True
    return m

@pytest.fixture
def cosmic_ray_mask(rng: np.random.Generator) -> np.ndarray:
    """
    64×64 sparse boolean mask simulating random CR hits (~5 pixels).
    """
    m = np.zeros((64, 64), dtype=bool)
    ys = rng.integers(0, 64, size=5)
    xs = rng.integers(0, 64, size=5)
    m[ys, xs] = True
    return m

@pytest.fixture
def gain_readnoise() -> dict:
    """
    Simple detector params for testing variance propagation.
    """
    return {"gain_e_per_adu": 2.0, "read_noise_e": 5.0}

# ---------------------------------------------------------------------------
# Torch compatibility (optional)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image_torch(synthetic_image: np.ndarray):
    """
    Torch tensor variant (cpu float32) if torch is available.
    """
    if not _HAS_TORCH:
        pytest.skip("torch not installed")
    return torch.from_numpy(synthetic_image.copy())

@pytest.fixture
def frame_stack_torch(frame_stack: np.ndarray):
    if not _HAS_TORCH:
        pytest.skip("torch not installed")
    return torch.from_numpy(frame_stack.copy())

# ---------------------------------------------------------------------------
# Parametric variants (sizes/dtypes) — use with @pytest.mark.parametrize
# ---------------------------------------------------------------------------

def make_synthetic(h: int = 64, w: int = 64, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Helper to create a sized synthetic image on demand inside tests."""
    if rng is None:
        rng = np.random.default_rng(0)
    y, x = _make_mesh(h, w)
    bg = 100.0 + 0.05 * x + 0.03 * y
    psf = _gaussian_2d(y, x, cy=h / 2.0, cx=w / 2.0, sigma=min(h, w) / 20.0, amp=800.0)
    noise = rng.normal(0.0, 2.0, size=(h, w)).astype(np.float32)
    return (bg + psf + noise).astype(np.float32)