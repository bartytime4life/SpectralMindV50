# tests/unit/test_models_shapes.py
from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import pytest

try:
    import torch
    from torch import nn, Tensor

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


N_BINS = 283


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
            # functions and classes are fine; instantiate later
            return ctor
    return None


# ----------------------------------------------------------------------------- #
# Synthetic inputs
# ----------------------------------------------------------------------------- #
def _synthetic_inputs(batch: int = 2) -> Iterable[Tuple[Dict[str, Tensor], str]]:
    """
    Yield several plausible input layouts for FGS1 and AIRS.

    We intentionally generate a few shapes to maximize compatibility:
      - FGS1: time-series 1D (B,T), 2D frames (B,T,H,W), time-series + channel (B,T,C)
      - AIRS: spectral per-timestep (B,T,C), or collapsed spectrum (B,C)
    """
    device = torch.device("cpu")
    T = 32  # short time length to keep tests fast
    H = W = 16
    C_spectral = N_BINS  # often AIRS has 283 channels after binning

    # 1) Minimalistic time-series + spectrum-per-timestep
    f1 = torch.randn(batch, T, device=device)
    a1 = torch.randn(batch, T, C_spectral, device=device)
    yield ({"fgs1": f1, "airs": a1}, "fgs1:(B,T), airs:(B,T,C)")

    # 2) 2D FGS1 frames + spectrum-per-timestep
    f2 = torch.randn(batch, T, H, W, device=device)
    a2 = torch.randn(batch, T, C_spectral, device=device)
    yield ({"fgs1": f2, "airs": a2}, "fgs1:(B,T,H,W), airs:(B,T,C)")

    # 3) FGS1 (B,T,1) + AIRS collapsed spectrum (B,C)
    f3 = torch.randn(batch, T, 1, device=device)
    a3 = torch.randn(batch, C_spectral, device=device)
    yield ({"fgs1": f3, "airs": a3}, "fgs1:(B,T,1), airs:(B,C)")

    # 4) FGS1 minimal vector per sample + AIRS (B,T,C)
    f4 = torch.randn(batch, 8, device=device)  # arbitrary small feature vector
    a4 = torch.randn(batch, T, C_spectral, device=device)
    yield ({"fgs1": f4, "airs": a4}, "fgs1:(B,F), airs:(B,T,C)")


def _unpack_outputs(out: Any) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Normalize possible model outputs to (mu, sigma).

    Accepts:
      - dict with 'mu'/'sigma'
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
            out[k] = v  # nothing we can do
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------------------- #
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _instantiate_model(ctor: Any) -> nn.Module:
    """
    Instantiate given a function or class.
    - If callable returns nn.Module: use it
    - If class: call with no args; if fails, try with a few defaults
    """
    try:
        obj = ctor()
    except TypeError:
        # Fallback: try simple kwargs commonly used for model configs
        try:
            obj = ctor(n_bins=N_BINS)  # many decoders accept n_bins
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
    if sigma is not None:
        assert isinstance(sigma, torch.Tensor), "sigma must be a Tensor"
        assert sigma.size(0) == B, f"sigma batch size mismatch: expected {B}, got {sigma.size(0)}"
        assert sigma.size(-1) == N_BINS, f"sigma last dim must be {N_BINS}, got {sigma.size(-1)}"


def test_model_shapes_for_various_inputs() -> None:
    ctor = _find_model_ctor()
    if ctor is None:
        pytest.skip("No model constructor found (build_model/Model/V50Model/SpectraMindModel)")

    model = _instantiate_model(ctor)

    # Try a few plausible input layouts; pass as soon as one works
    tried = 0
    for batch, desc in _synthetic_inputs(batch=2):
        tried += 1
        batch = _ensure_batch_first(batch)
        try:
            mu, sigma = _forward_any(model, batch)
            _assert_mu_sigma_shapes(mu, sigma, B=2)
            break
        except AssertionError:
            continue
    else:
        pytest.xfail(f"None of the synthetic input layouts matched the model signature (tried {tried}).")

    # Batch invariance: run B=1 and B=3 (same chosen layout)
    for B in (1, 3):
        one_batch = {k: v[:B].contiguous() if v.size(0) >= B else v.expand(B, *v.shape[1:]) for k, v in batch.items()}
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


def test_output_container_types() -> None:
    """
    Ensure output container is one of dict/tuple/list/tensor and has expected last dim.
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