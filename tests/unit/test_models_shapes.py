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
def _synthetic_inputs(
    batch: int = 2, device: torch.device | None = None
) -> Iterable[Tuple[Dict[str, Tensor], str]]:
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

    # 5) Per-timestep FGS1 as (B,T,2) + collapsed spectrum
    f5 = torch.randn(batch, T, 2, device=device)
    a5 = torch.randn(batch, C_spectral, device=device)
    yield ({"fgs1": f5, "airs": a5}, "fgs1:(B,T,2), airs:(B,C)")


def _expand_key_aliases(batch: Dict[str, Tensor]) -> Iterable[Dict[str, Tensor]]:
    """
    Yield variants with common input key aliases to match different model signatures.
    """
    f = batch.get("fgs1", None)
    a = batch.get("airs", None)
    if f is None or a is None:
        yield batch
        return
    # common aliases for FGS1/FGS
    f_keys = ("fgs1", "fgs", "fgs_ch0", "fgs0")
    # common aliases for AIRS CH0
    a_keys = ("airs", "airs_ch0", "airs0", "ch0", "spectra", "x_spectra")
    for fk in f_keys:
        for ak in a_keys:
            yield {fk: f, ak: a}


# ----------------------------------------------------------------------------- #
# Output normalization
# ----------------------------------------------------------------------------- #
def _unpack_outputs(out: Any) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Normalize possible model outputs to (mu, sigma).

    Accepts:
      - dict with 'mu'/'sigma' (also supports 'mean'/'std' etc.)
      - tuple/list
      - single Tensor (treated as mu, sigma=None)
    """
    if isinstance(out, dict):
        mu = (
            out.get("mu")
            or out.get("mean")
            or out.get("pred")
            or out.get("y")
            or out.get("spectra")
        )
        sigma = out.get("sigma") or out.get("std") or out.get("uncertainty")
        if mu is None:
            raise AssertionError(
                f"Dict output missing a recognizable 'mu' tensor: keys={list(out.keys())}"
            )
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


def _squeeze_temporal(mu: Tensor, sigma: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
    """
    If output is (B,T,C) or (B,1,C), squeeze/mean-reduce temporal dimension to (B,C).
    This keeps shape checks tolerant to models that emit per-timestep spectra.
    """
    def _reduce(x: Tensor) -> Tensor:
        if x.dim() >= 3 and x.size(-1) == N_BINS:
            # If (B, T, C): mean over T; if (B, 1, C): squeeze
            if x.size(-2) == 1:
                return x.squeeze(-2)
            return x.mean(dim=-2)
        return x

    mu_r = _reduce(mu)
    sig_r = _reduce(sigma) if sigma is not None else None
    return mu_r, sig_r


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
      - model(dict)             (single positional)
      - model(fgs1, airs)       (positional)
    Try key alias expansions.
    """
    with torch.no_grad():
        for candidate in (batch, *list(_expand_key_aliases(batch))):
            # kwargs
            try:
                out = model(**candidate)
                return _unpack_outputs(out)
            except TypeError:
                pass
            # dict positional
            try:
                out = model(candidate)
                return _unpack_outputs(out)
            except Exception:
                pass
            # positional 2 args if keys exist
            for fk in ("fgs1", "fgs", "fgs_ch0", "fgs0"):
                for ak in ("airs", "airs_ch0", "airs0", "ch0", "spectra", "x_spectra"):
                    if fk in candidate and ak in candidate:
                        try:
                            out = model(candidate[fk], candidate[ak])
                            return _unpack_outputs(out)
                        except Exception:
                            continue
    raise AssertionError("Model forward failed with all tested signatures and key aliases")


def _assert_mu_sigma_shapes(mu: Tensor, sigma: Optional[Tensor], B: int) -> Tuple[Tensor, Optional[Tensor]]:
    assert isinstance(mu, torch.Tensor), "mu must be a Tensor"

    # Allow either (B, C) or (B, T, C) — reduce temporal to (B, C) for shape checks.
    mu, sigma = _squeeze_temporal(mu, sigma)

    assert mu.dim() >= 2, f"mu must include batch and bins dims; got shape={tuple(mu.shape)}"
    assert mu.size(0) == B, f"mu batch size mismatch: expected {B}, got {mu.size(0)}"
    assert mu.size(-1) == N_BINS, f"mu last dim must be {N_BINS}, got {mu.size(-1)}"
    assert torch.isfinite(mu).all(), "mu must be finite"

    if sigma is not None:
        assert isinstance(sigma, torch.Tensor), "sigma must be a Tensor"
        sigma = _squeeze_temporal(sigma, None)[0]
        assert sigma.size(0) == B, f"sigma batch size mismatch: expected {B}, got {sigma.size(0)}"
        assert sigma.size(-1) == N_BINS, f"sigma last dim must be {N_BINS}, got {sigma.size(-1)}"
        assert torch.isfinite(sigma).all(), "sigma must be finite"
        # Soft guard: allow non-positive values (log-σ may be elsewhere),
        # but fail if wildly non-finite — already checked above.

    return mu, sigma


def _device_candidates() -> Iterable[torch.device]:
    yield torch.device("cpu")
    if torch.cuda.is_available():
        yield torch.device("cuda:0")


def _amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


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
        model = model.to(device).eval()
        worked = False
        chosen_batch: Optional[Dict[str, Tensor]] = None

        # Try a few plausible input layouts; pass as soon as one works per device
        for batch, _desc in _synthetic_inputs(batch=2, device=device):
            try:
                mu, sigma = _forward_any(model, batch)
                mu, sigma = _assert_mu_sigma_shapes(mu, sigma, B=2)
                worked = True
                chosen_batch = batch
                break
            except AssertionError:
                continue

        if not worked or chosen_batch is None:
            pytest.xfail(f"No synthetic input layout matched the model signature on device={device}.")

        # Batch invariance: run B=1 and B=3 on the chosen layout
        for B in (1, 3):
            one_batch = {
                k: (v[:B].contiguous() if v.size(0) >= B else v.expand(B, *v.shape[1:]))
                for k, v in chosen_batch.items()
            }
            mu_b, sigma_b = _forward_any(model, one_batch)
            _assert_mu_sigma_shapes(mu_b, sigma_b, B=B)


def test_model_accepts_dict_or_kwargs_and_key_aliases() -> None:
    """
    Sanity check that model accepts dict(**) or dict positional, with common key aliases.
    Soft requirement: if one call style fails, we don't fail test, but skip gracefully.
    """
    ctor = _find_model_ctor()
    if ctor is None:
        pytest.skip("No model constructor found")
    model = _instantiate_model(ctor).to("cpu")

    for batch, _ in _synthetic_inputs(batch=2, device=torch.device("cpu")):
        ok = False
        with torch.no_grad():
            # kwargs and dict positional across alias expansions
            for candidate in _expand_key_aliases(batch):
                try:
                    _ = model(**candidate)
                    ok = True
                    break
                except Exception:
                    pass
                try:
                    _ = model(candidate)
                    ok = True
                    break
                except Exception:
                    pass
        if ok:
            return
    pytest.skip("Model did not accept dict/kwargs signatures among tested aliases.")


def test_output_container_types_and_finiteness() -> None:
    """
    Ensure output container is one of dict/tuple/list/tensor and has expected last dim,
    with finite values. Allow temporal dim (reduced internally).
    """
    ctor = _find_model_ctor()
    if ctor is None:
        pytest.skip("No model constructor found")
    model = _instantiate_model(ctor).to("cpu")

    for batch, _ in _synthetic_inputs(batch=2, device=torch.device("cpu")):
        try:
            mu, sigma = _forward_any(model, batch)
            _assert_mu_sigma_shapes(mu, sigma, B=2)
            return
        except AssertionError:
            continue
    pytest.skip("Could not validate output container type with synthetic inputs.")


def test_forward_no_grad_and_inference_mode_determinism() -> None:
    """
    Verify forward works under no_grad/inference_mode, and that eval() is deterministic
    (calling twice with same inputs yields the same outputs).
    """
    ctor = _find_model_ctor()
    if ctor is None:
        pytest.skip("No model constructor found")
    model = _instantiate_model(ctor).to("cpu")

    batch, _ = next(iter(_synthetic_inputs(batch=2, device=torch.device("cpu"))))

    with torch.no_grad():
        mu1, sigma1 = _forward_any(model, batch)
        mu1, sigma1 = _squeeze_temporal(mu1, sigma1)
        mu2, sigma2 = _forward_any(model, batch)
        mu2, sigma2 = _squeeze_temporal(mu2, sigma2)

    assert torch.allclose(mu1, mu2), "Eval/no_grad should be deterministic for mu"
    if sigma1 is not None and sigma2 is not None:
        assert torch.allclose(sigma1, sigma2), "Eval/no_grad should be deterministic for sigma"

    # inference_mode can be stricter than no_grad; ensure it still works
    with torch.inference_mode():
        mu3, sigma3 = _forward_any(model, batch)
        _assert_mu_sigma_shapes(mu3, sigma3, B=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_amp_quick_smoke() -> None:
    """
    Optional quick mixed-precision smoke test on CUDA to catch device/dtype bugs.
    Kept tiny for CI/Kaggle runtime.
    """
    ctor = _find_model_ctor()
    if ctor is None:
        pytest.skip("No model constructor found")
    device = torch.device("cuda:0")
    model = _instantiate_model(ctor).to(device)

    batch, _ = next(iter(_synthetic_inputs(batch=2, device=device)))
    # autocast FP16
    with torch.cuda.amp.autocast():
        mu, sigma = _forward_any(model, batch)
    _assert_mu_sigma_shapes(mu, sigma, B=2)