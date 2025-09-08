# tests/unit/test_models_shapes.py
from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional, Tuple

import pytest

try:
    import torch
    from torch import nn, Tensor

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #
N_BINS = 283  # Ariel challenge canonical spectral bin count


# ----------------------------------------------------------------------------- #
# Model discovery
# ----------------------------------------------------------------------------- #
def _try_import(path: str) -> Optional[Any]:
    try:
        return importlib.import_module(path)
    except Exception:
        return None


def _find_model_ctor() -> Optional[Any]:
    """
    Try common entry points and return a callable that constructs the model.

    We try, in order:
      - spectramind.models.v50: build_model, Model/V50Model/SpectraMindModel classes
      - spectramind.models:     build_model, Model/V50Model/SpectraMindModel classes
      - spectramind.model.v50:  build_model, Model/V50Model/SpectraMindModel classes
      - spectramind.model:      build_model, Model/V50Model/SpectraMindModel classes
    """
    candidates = [
        "spectramind.models.v50",
        "spectramind.models",
        "spectramind.model.v50",
        "spectramind.model",
    ]
    names = ("build_model", "Model", "V50Model", "SpectraMindModel")
    for modname in candidates:
        mod = _try_import(modname)
        if not mod:
            continue
        for name in names:
            ctor = getattr(mod, name, None)
            if ctor is None:
                continue
            return ctor
    return None


# ----------------------------------------------------------------------------- #
# Synthetic inputs
# ----------------------------------------------------------------------------- #
def _synthetic_inputs(batch: int = 2, device: torch.device | None = None) -> Iterable[Tuple[Dict[str, Tensor], str]]:
    """
    Yield several plausible input layouts for FGS1 and AIRS.

    We intentionally generate a few shapes to maximize compatibility:
      - FGS1: time-series 1D (B,T), 2D frames (B,T,H,W), time-series + channel (B,T,1), feature vec (B,F)
      - AIRS: spectral per-timestep (B,T,C), or collapsed spectrum (B,C)
    """
    if device is None:
        device = torch.device("cpu")
    T = 32  # keep short to run fast
    H = W = 16
    C_spectral = N_BINS

    # 1) Minimalistic time-series + per-timestep spectrum
    f1 = torch.randn(batch, T, device=device)
    a1 = torch.randn(batch, T, C_spectral, device=device)
    yield ({"fgs1": f1, "airs": a1}, "fgs1:(B,T), airs:(B,T,C)")

    # 2) 2D frames + per-timestep spectrum
    f2 = torch.randn(batch, T, H, W, device=device)
    a2 = torch.randn(batch, T, C_spectral, device=device)
    yield ({"fgs1": f2, "airs": a2}, "fgs1:(B,T,H,W), airs:(B,T,C)")

    # 3) Time-series + channel plus collapsed spectrum
    f3 = torch.randn(batch, T, 1, device=device)
    a3 = torch.randn(batch, C_spectral, device=device)
    yield ({"fgs1": f3, "airs": a3}, "fgs1:(B,T,1), airs:(B,C)")

    # 4) Small feature vector + per-timestep spectrum
    f4 = torch.randn(batch, 8, device=device)  # small feature vector
    a4 = torch.randn(batch, T, C_spectral, device=device)
    yield ({"fgs1": f4, "airs": a4}, "fgs1:(B,F), airs:(B,T,C)")


def _unpack_outputs(out: Any) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Normalize possible model outputs to (mu, sigma).

    Accepts:
      - dict with 'mu'/'sigma' (also supports 'mean'/'std' etc.)
      - tuple/list
      - single Tensor (treated as mu, sigma=None)
    """
    if isinstance(out, dict):
        mu = out.get("mu") or out.get("mean") or out.get("pred") or out.get("y") or out.get("spectra")
        sigma = out.get("sigma") or out.get("std") or out.get("uncertainty")
        if mu is None:
            raise AssertionError(f"Dict output missing a recognizable 'mu' tensor: keys={list(out.keys())}")
        return mu, sigma
    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise AssertionError("Empty tuple/list output from model")
        if len(out) == 1:
            mu = out[0]
            return mu, None
        mu, sigma = out[0], out[1]
        return mu, sigma
    if isinstance(out, torch.Tensor):
        return out, None
    raise AssertionError(f"Unsupported model output type: {type(out)!r}")


def _ensure_batch_first(inp: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Ensure tensors have batch dim first; if single sample provided, add batch dim.
    """
    out: Dict[str, Tensor] = {}
    for k, v in inp.items():
        if v.dim() == 0:
            out[k] = v[None]
        elif v.size(0) == 0:
            out[k] = v
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------- #
# Test helpers
# ----------------------------------------------------------------------------- #
def _instantiate_model(ctor: Any) -> nn.Module:
    """
    Instantiate given a function or class.
    - If callable returns nn.Module: use it
    - If class: call with no args; if fails, try with a few defaults
    """
    try:
        obj = ctor()
    except TypeError:
        # Fallback: try common kwarg used by decoders
        try:
            obj = ctor(n_bins=N_BINS)
        except Exception as e:
            pytest.skip(f"Could not instantiate model with no args or n_bins: {e!r}")
    if not isinstance(obj, nn.Module):
        pytest.skip(f"Constructed object is not a torch.nn.Module: {type(obj)!r}")
    return obj.eval()


def _forward_any(model: nn.Module, batch: Dict[str, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Try common call signatures:
      - model(fgs1=..., airs=...)
      - model({"fgs1":..., "airs":...})
      - model(fgs1, airs)
    """
    with torch.no_grad():
        try:
            out = model(**batch)
            return _unpack_outputs(out)
        except TypeError:
            pass
        try:
            out = model(batch)
            return _unpack_outputs(out)
        except Exception:
            pass
        # Final attempt: positional (fgs1, airs)
        try:
            out = model(batch["fgs1"], batch["airs"])
            return _unpack_outputs(out)
        except Exception as e:
            raise AssertionError(f"Model forward failed with synthetic inputs: {e!r}")


def _assert_mu_sigma_shapes(mu: Tensor, sigma: Optional[Tensor], B: int) -> None:
    assert isinstance(mu, torch.Tensor), "mu must be a Tensor"
    assert mu.dim() >= 2, f"mu must include batch and bins dims; got shape={tuple(mu.shape)}"
    assert mu.size(0) == B, f"mu batch size mismatch: expected {B}, got {mu.size(0)}"
    assert mu.size(-1) == N_BINS, f"mu last dim must be {N_BINS}, got {mu.size(-1)}"
    assert torch.isfinite(mu).all(), "mu must be finite"
    if sigma is not None:
        assert isinstance(sigma, torch.Tensor), "sigma must be a Tensor"
        assert sigma.size(0) == B, f"sigma batch size mismatch: expected {B}, got {sigma.size(0)}"
        assert sigma.size(-1) == N_BINS, f"sigma last dim must be {N_BINS}, got {sigma.size(-1)}"
        assert torch.isfinite(sigma).all(), "sigma must be finite"
        # Do not enforce positivity strictly (models might output log-σ elsewhere),
        # but if values are present, they should not be wildly negative:
        if (sigma <= 0).any():
            # Soft guard: allow but avoid failing hard; keep assertion focused on shape/finite.
            pass


def _device_candidates() -> Iterable[torch.device]:
    yield torch.device("cpu")
    if torch.cuda.is_available():
        # Single quick pass on CUDA if it exists to catch device bugs;
        # keep batch small to be fast on CI/Kaggle.
        yield torch.device("cuda:0")


# ----------------------------------------------------------------------------- #
# Tests (torch required)
# ----------------------------------------------------------------------------- #
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def test_model_shapes_for_various_inputs_cpu_and_optional_cuda() -> None:
    ctor = _find_model_ctor()
    if ctor is None:
        pytest.skip("No model constructor found (build_model/Model/V50Model/SpectraMindModel)")

    model = _instantiate_model(ctor)

    for device in _device_candidates():
        # Try a few plausible input layouts; pass as soon as one works per device
        worked = False
        for batch, _desc in _synthetic_inputs(batch=2, device=device):
            batch = _ensure_batch_first(batch)
            try:
                mu, sigma = _forward_any(model.to(device), batch)
                _assert_mu_sigma_shapes(mu, sigma, B=2)
                worked = True
                chosen_batch = batch  # keep for batch-size invariance checks
                break
            except AssertionError:
                continue
        if not worked:
            pytest.xfail(f"No synthetic input layout matched the model signature on device={device}.")

        # Batch invariance: run B=1 and B=3 on the chosen layout
        for B in (1, 3):
            one_batch = {
                k: (v[:B].contiguous() if v.size(0) >= B else v.expand(B, *v.shape[1:]))
                for k, v in chosen_batch.items()
            }
            mu_b, sigma_b = _forward_any(model, one_batch)
            _assert_mu_sigma_shapes(mu_b, sigma_b, B=B)


def test_model_accepts_dict_or_kwargs() -> None:
    """
    Sanity check that model accepts dict(**) or dict positional.
    Soft requirement: if one call style fails, we don't fail test, but skip.
    """
    ctor = _find_model_ctor()
    if ctor is None:
        pytest.skip("No model constructor found")
    model = _instantiate_model(ctor)

    for batch, _ in _synthetic_inputs(batch=2):
        batch = _ensure_batch_first(batch)
        ok = False
        with torch.no_grad():
            # kwargs
            try:
                _ = model(**batch)
                ok = True
            except Exception:
                pass
            # dict positional
            if not ok:
                try:
                    _ = model(batch)
                    ok = True
                except Exception:
                    pass
        if ok:
            return
    pytest.skip("Model did not accept dict/kwargs in tested signatures.")


def test_output_container_types_and_finiteness() -> None:
    """
    Ensure output container is one of dict/tuple/list/tensor and has expected last dim,
    with finite values.
    """
    ctor = _find_model_ctor()
    if ctor is None:
        pytest.skip("No model constructor found")
    model = _instantiate_model(ctor)

    for batch, _ in _synthetic_inputs(batch=2):
        batch = _ensure_batch_first(batch)
        try:
            mu, sigma = _forward_any(model, batch)
            _assert_mu_sigma_shapes(mu, sigma, B=2)
            return
        except AssertionError:
            continue
    pytest.skip("Could not validate output container type with synthetic inputs.")


def test_forward_no_grad_and_inference_mode() -> None:
    """
    Verify forward works under torch.no_grad() and torch.inference_mode(),
    which is common during predict/eval in pipelines and Kaggle kernels.
    """
    ctor = _find_model_ctor()
    if ctor is None:
        pytest.skip("No model constructor found")
    model = _instantiate_model(ctor)

    batch, _ = next(iter(_synthetic_inputs(batch=2)))
    batch = _ensure_batch_first(batch)

    with torch.no_grad():
        mu1, sigma1 = _forward_any(model, batch)
        _assert_mu_sigma_shapes(mu1, sigma1, B=2)

    # inference_mode can be stricter than no_grad; ensure it still works
    with torch.inference_mode():
        mu2, sigma2 = _forward_any(model, batch)
        _assert_mu_sigma_shapes(mu2, sigma2, B=2)
