from __future__ import annotations
import numpy as np
import pytest

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)

@pytest.fixture
def synthetic_image(rng: np.random.Generator) -> np.ndarray:
    h, w = 64, 64
    y, x = np.mgrid[:h, :w]
    bg = 100 + 0.05 * x + 0.03 * y
    psf = 1000 * np.exp(-(((x - 32)**2 + (y - 32)**2) / (2 * 3.0**2)))
    noise = rng.normal(0, 2.0, size=(h, w))
    return (bg + psf + noise).astype(np.float32)

@pytest.fixture
def master_dark(rng: np.random.Generator) -> np.ndarray:
    h, w = 64, 64
    pattern = 5.0 + 0.01 * (np.sin(np.linspace(0, 8*np.pi, w))[None, :])
    return (pattern + rng.normal(0, 0.2, size=(h, w))).astype(np.float32)

@pytest.fixture
def master_flat(rng: np.random.Generator) -> np.ndarray:
    h, w = 64, 64
    y, x = np.mgrid[:h, :w]
    flat = 0.9 + 0.0005 * ((x - 32)**2 + (y - 32)**2)
    flat = flat / flat.mean()
    return flat.astype(np.float32)

@pytest.fixture
def frame_stack(rng: np.random.Generator) -> np.ndarray:
    t, h, w = 4, 32, 32
    base = 50 + 0.02 * (np.mgrid[:h, :w][1])
    trend = np.linspace(0, 5, t)[:, None, None]
    noise = rng.normal(0, 1.0, size=(t, h, w))
    signal = np.zeros((t, h, w), dtype=np.float32)
    signal[-1] += 20.0
    return (base + trend + signal + noise).astype(np.float32)

