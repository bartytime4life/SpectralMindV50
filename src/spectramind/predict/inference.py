# src/spectramind/predict/inference.py
# -----------------------------------------------------------------------------
# SpectraMind V50 — Inference / Prediction
# -----------------------------------------------------------------------------
# - Loads a trained heteroscedastic model checkpoint (TorchScript or state_dict)
# - Runs batched inference (FGS1 + AIRS or generic 'inputs')
# - Emits Kaggle-ready submission CSV with columns:
#       sample_id, mu_000..mu_XXX, sigma_000..sigma_XXX
#
# Notes
# -----
# * Hydra-friendly (instantiate dataset/model from config), but includes a safe
#   NPZ-based fallback dataset for Kaggle / CI.
# * Kaggle-safe: torch / numpy / pandas only (no network calls).
# * Numerically robust: clamps σ with configurable floor; replaces non-finite.
# * Evaluates under torch.no_grad() and model.eval() on the chosen device.
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Optional Hydra support (soft import)
try:  # pragma: no cover (optional)
    from omegaconf import DictConfig, OmegaConf  # type: ignore
    from hydra.utils import instantiate  # type: ignore
except Exception:  # pragma: no cover
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore
    instantiate = None  # type: ignore

# Align with submission defaults
from spectramind.submit.validate import N_BINS_DEFAULT

LOGGER = logging.getLogger("spectramind.inference")

MU_PREFIX = "mu_"
SIGMA_PREFIX = "sigma_"
DEFAULT_SIGMA_MIN = 1e-9


# -----------------------------------------------------------------------------
# Basic NPZ dataset fallback (Kaggle/CI-safe)
# -----------------------------------------------------------------------------
class NPZPredictDataset(Dataset):
    """
    Minimal inference dataset reading from an .npz bundle.

    Expected arrays:
      - ids:    shape [N], dtype str or bytes (sample identifiers)
      - fgs1:   shape [N, ...]  (optional)
      - airs:   shape [N, ...]  (optional)
      - inputs: shape [N, ...]  (optional, generic single-branch)

    The model forward should accept one of:
      model(fgs1=..., airs=...)  or  model(inputs=...)  or  model({"fgs1":..., "airs":...}).
    """

    def __init__(self, npz_path: Union[str, Path]):
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"Predict npz not found: {p}")
        bundle = np.load(p, allow_pickle=False)
        files = set(bundle.files)

        if "ids" not in files:
            raise ValueError(f"NPZ must contain an 'ids' array; found files={sorted(files)}")
        self._ids = _to_str_array(bundle["ids"])
        self._has_fgs1 = "fgs1" in files
        self._has_airs = "airs" in files
        self._has_inputs = "inputs" in files

        if not (self._has_inputs or (self._has_fgs1 and self._has_airs)):
            raise ValueError("NPZ must contain either 'inputs' or both 'fgs1' and 'airs' arrays.")

        self._fgs1 = bundle["fgs1"] if self._has_fgs1 else None
        self._airs = bundle["airs"] if self._has_airs else None
        self._inputs = bundle["inputs"] if self._has_inputs else None

        n = len(self._ids)
        for arr, name in [(self._fgs1, "fgs1"), (self._airs, "airs"), (self._inputs, "inputs")]:
            if arr is not None and (arr.ndim < 1 or arr.shape[0] != n):
                raise ValueError(f"Array '{name}' must have first dimension N={n}, got shape {arr.shape if arr is not None else None}")

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: Dict[str, Any] = {"sample_id": self._ids[idx]}
        if self._has_inputs:
            x = self._inputs[idx]
            sample["inputs"] = torch.from_numpy(x).float()
        else:
            f = self._fgs1[idx]
            a = self._airs[idx]
            sample["fgs1"] = torch.from_numpy(f).float()
            sample["airs"] = torch.from_numpy(a).float()
        return sample


def _to_str_array(x: np.ndarray) -> np.ndarray:
    if x.dtype.type is np.bytes_:
        return x.astype(str)
    if np.issubdtype(x.dtype, np.str_):
        return x
    return x.astype(str)


# -----------------------------------------------------------------------------
# Inference configuration
# -----------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    # Required: model checkpoint
    checkpoint: str

    # Output
    output_csv: str = "submission.csv"

    # Data (provide either Hydra dataset or npz_path)
    dataset: Optional["DictConfig"] = None
    npz_path: Optional[str] = None

    # Loader
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True

    # Device / AMP
    device: str = "auto"  # "auto"|"cuda"|"cpu"|"mps"
    precision: str = "fp32"  # "fp32"|"fp16"|"bf16"

    # Bins & numeric safety
    n_bins: int = N_BINS_DEFAULT
    sigma_min: float = DEFAULT_SIGMA_MIN
    replace_nonfinite: bool = True

    # CSV formatting
    id_column: str = "sample_id"


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def select_device(pref: str = "auto") -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if pref == "mps" or (pref == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _amp_dtype(precision: str) -> Optional[torch.dtype]:
    p = (precision or "fp32").lower()
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    return None  # fp32


def load_model_from_checkpoint(ckpt_path: Union[str, Path], device: torch.device) -> torch.nn.Module:
    """
    Loads a torch model checkpoint. Supports:
      - TorchScript JIT models (torch.jit.load)
      - state_dict checkpoints (raw mapping) or Lightning-style {"state_dict": ...}
    The caller is expected to provide an instantiated model if not using JIT; this
    function raises to indicate that state_dict was loaded but no model was present.
    """
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    # Try JIT first (preferred for Kaggle deployment)
    try:
        model = torch.jit.load(str(p), map_location=device)
        LOGGER.info("Loaded TorchScript model from %s", p)
        return model
    except Exception:
        pass

    # Otherwise assume plain/lightning checkpoint
    payload = torch.load(str(p), map_location=device)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        LOGGER.info("Loaded Lightning-style checkpoint (state_dict) from %s", p)
        # surface the state_dict only; user must instantiate model
        raise RuntimeError(
            "Checkpoint contains 'state_dict' only. Instantiate your model (e.g., via Hydra) and call "
            "model.load_state_dict(payload['state_dict'])."
        )
    if isinstance(payload, Mapping):
        LOGGER.info("Loaded raw checkpoint mapping from %s", p)
        raise RuntimeError(
            "Raw checkpoint mapping loaded. Instantiate your model and call model.load_state_dict(payload)."
        )

    raise ValueError(f"Unsupported checkpoint format at {p}")


# -----------------------------------------------------------------------------
# Dataset / loader builders
# -----------------------------------------------------------------------------
def build_dataset(cfg: InferenceConfig) -> Dataset:
    if cfg.dataset is not None and OmegaConf is not None and instantiate is not None:
        ds = instantiate(cfg.dataset)
        if not isinstance(ds, Dataset):
            raise TypeError(f"Hydra dataset _target_ must return a torch.utils.data.Dataset, got {type(ds)}")
        return ds

    if cfg.npz_path:
        return NPZPredictDataset(cfg.npz_path)

    raise ValueError("No dataset specified. Provide 'dataset: {_target_: ...}' or 'npz_path' in config.")


def build_loader(dataset: Dataset, cfg: InferenceConfig) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        shuffle=False,
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Forward helpers
# -----------------------------------------------------------------------------
def _try_forward(model: torch.nn.Module, feed: Dict[str, Any]) -> Any:
    """
    Try calling model with kwargs or a single dict argument for maximum compatibility.
    """
    try:
        return model(**feed)
    except TypeError:
        return model(feed)


def _extract_mu_sigma(out: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accept model outputs:
      - dict with {'mu','sigma'} or {'mu','log_sigma'}
      - tuple/list (mu, sigma)
      - object with attributes .mu and .sigma
    Returns torch tensors [B, ?bins].
    """
    mu: Optional[torch.Tensor] = None
    sigma: Optional[torch.Tensor] = None

    if isinstance(out, Mapping):
        mu = out.get("mu", None)
        sigma = out.get("sigma", None)
        if sigma is None and "log_sigma" in out:
            sigma = torch.exp(out["log_sigma"])
    elif isinstance(out, (tuple, list)) and len(out) >= 2:
        mu, sigma = out[0], out[1]
    else:
        mu = getattr(out, "mu", None)
        sigma = getattr(out, "sigma", None)

    if mu is None or sigma is None:
        raise ValueError("Model output must provide 'mu' and 'sigma' (or 'log_sigma').")
    if not isinstance(mu, torch.Tensor) or not isinstance(sigma, torch.Tensor):
        raise TypeError("Model output 'mu' and 'sigma' must be torch.Tensor.")

    if mu.ndim != 2 or sigma.ndim != 2:
        raise ValueError(f"mu/sigma must be 2D [B, n_bins], got mu={tuple(mu.shape)}, sigma={tuple(sigma.shape)}")

    return mu, sigma


def _clean_nonfinite(x: np.ndarray, fill_value: float) -> np.ndarray:
    if not np.all(np.isfinite(x)):
        x = x.copy()
        x[~np.isfinite(x)] = fill_value
    return x


# -----------------------------------------------------------------------------
# Inference pipeline
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_bins: int,
    precision: str = "fp32",
    sigma_min: float = DEFAULT_SIGMA_MIN,
    replace_nonfinite: bool = True,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    amp_dtype = _amp_dtype(precision)

    ids: List[str] = []
    mus: List[np.ndarray] = []
    sigmas: List[np.ndarray] = []

    for batch in loader:
        # Move tensors to device
        feed: Dict[str, Any] = {}
        for k, v in batch.items():
            if k in ("sample_id", "id"):
                continue
            if torch.is_tensor(v):
                feed[k] = v.to(device, non_blocking=True)
            else:
                feed[k] = v

        # Forward (with AMP where appropriate)
        use_amp = amp_dtype is not None and (device.type in ("cuda", "cpu"))
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):  # type: ignore[arg-type]
                out = _try_forward(model, feed)
        else:
            out = _try_forward(model, feed)

        mu_t, sigma_t = _extract_mu_sigma(out)  # [B, n_bins_actual]

        # Slice or validate bins
        if mu_t.shape[1] < n_bins or sigma_t.shape[1] < n_bins:
            raise ValueError(f"Model produced fewer bins than requested n_bins={n_bins}: mu={mu_t.shape}, sigma={sigma_t.shape}")
        mu_t = mu_t[:, :n_bins].detach().float()
        sigma_t = sigma_t[:, :n_bins].detach().float()

        # Numeric hygiene
        if sigma_min is not None and sigma_min > 0:
            sigma_t = torch.clamp(sigma_t, min=float(sigma_min))

        mu_np = mu_t.cpu().numpy()
        sigma_np = sigma_t.cpu().numpy()

        if replace_nonfinite:
            mu_np = _clean_nonfinite(mu_np, fill_value=0.0)
            sigma_np = _clean_nonfinite(sigma_np, fill_value=float(sigma_min or DEFAULT_SIGMA_MIN))

        # Collect ids
        if "sample_id" in batch:
            batch_ids = _to_str_array(np.array(batch["sample_id"]))
        elif "id" in batch:
            batch_ids = _to_str_array(np.array(batch["id"]))
        else:
            # Fabricate deterministic sequential ids
            start = len(ids)
            batch_ids = np.array([str(start + i) for i in range(mu_np.shape[0])], dtype=str)

        ids.extend(batch_ids.tolist())
        mus.append(mu_np)
        sigmas.append(sigma_np)

    mu_all = np.concatenate(mus, axis=0) if mus else np.zeros((0, n_bins), dtype=np.float32)
    sigma_all = np.concatenate(sigmas, axis=0) if sigmas else np.zeros((0, n_bins), dtype=np.float32)

    # Final sanity
    if mu_all.shape[0] != len(ids) or sigma_all.shape[0] != len(ids):
        raise RuntimeError(f"Row mismatch: ids={len(ids)}, mu={mu_all.shape}, sigma={sigma_all.shape}")
    if not np.all(np.isfinite(mu_all)) or not np.all(np.isfinite(sigma_all)):
        raise ValueError("Non-finite values encountered in predictions.")

    return ids, mu_all, sigma_all


# -----------------------------------------------------------------------------
# Submission writer
# -----------------------------------------------------------------------------
def save_submission_csv(
    ids: Sequence[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    csv_path: Union[str, Path],
    id_column: str = "sample_id",
) -> None:
    """
    Writes a Kaggle-ready submission CSV with columns:
      sample_id, mu_000.., sigma_000..
    """
    n, n_bins_mu = mu.shape
    n2, n_bins_sigma = sigma.shape
    if len(ids) != n or n != n2 or n_bins_mu != n_bins_sigma:
        raise ValueError(
            f"Inconsistent shapes: ids={len(ids)}, mu={mu.shape}, sigma={sigma.shape}"
        )

    cols = [id_column] + [f"{MU_PREFIX}{i:03d}" for i in range(n_bins_mu)] + [
        f"{SIGMA_PREFIX}{i:03d}" for i in range(n_bins_sigma)
    ]
    out_p = Path(csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i, sid in enumerate(ids):
            row = [sid] + mu[i].tolist() + sigma[i].tolist()
            writer.writerow(row)

    LOGGER.info("Wrote submission to %s  (rows=%d)", out_p, len(ids))


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------
def infer(cfg: Union[InferenceConfig, "DictConfig", Mapping[str, Any]]) -> Path:
    """
    Entry point for programmatic use (Hydra-friendly).
    Expect either:
      - InferenceConfig dataclass
      - Hydra DictConfig / plain dict with matching fields
    """
    if not isinstance(cfg, InferenceConfig):
        cfg = _coerce_inference_config(cfg)

    # Dataset + loader
    dataset = build_dataset(cfg)
    loader = build_loader(dataset, cfg)

    # Device
    device = select_device(cfg.device)
    LOGGER.info("Using device: %s, precision=%s, n_bins=%d", device, cfg.precision, cfg.n_bins)

    # Load model (JIT or state_dict flow)
    try:
        model = load_model_from_checkpoint(cfg.checkpoint, device=device)
    except RuntimeError as e:
        # Recommended path: Hydra instantiates the model, then we load the state dict here
        if hasattr(cfg, "model") and getattr(cfg, "model") is not None and OmegaConf is not None and instantiate is not None:
            model = instantiate(cfg.model)
            payload = torch.load(cfg.checkpoint, map_location=device)
            state = payload["state_dict"] if isinstance(payload, Mapping) and "state_dict" in payload else payload
            model.load_state_dict(state, strict=False)
            model.to(device)
            LOGGER.info("Instantiated model via Hydra and loaded state_dict with strict=False.")
        else:
            raise e

    ids, mu, sigma = run_inference(
        model=model,
        loader=loader,
        device=device,
        n_bins=cfg.n_bins,
        precision=cg.precision if (cg := cfg) else "fp32",
        sigma_min=cfg.sigma_min,
        replace_nonfinite=cfg.replace_nonfinite,
    )

    save_submission_csv(
        ids=ids,
        mu=mu,
        sigma=sigma,
        csv_path=cfg.output_csv,
        id_column=cfg.id_column,
    )
    return Path(cfg.output_csv)


def _coerce_inference_config(x: Union["DictConfig", Mapping[str, Any]]) -> InferenceConfig:
    if OmegaConf is not None and isinstance(x, DictConfig):
        x = OmegaConf.to_container(x, resolve=True)
    return InferenceConfig(**dict(x))  # type: ignore


# -----------------------------------------------------------------------------
# Minimal CLI (Kaggle-safe)
# -----------------------------------------------------------------------------
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )


def main() -> None:
    """
    Minimal CLI without Typer/Click to keep Kaggle-safe.
    Usage:
      python -m spectramind.predict.inference \
          --checkpoint artifacts/model.ckpt \
          --npz data/processed/predict_inputs.npz \
          --output_csv artifacts/submission.csv
    """
    import argparse

    _setup_logging()
    parser = argparse.ArgumentParser("SpectraMind V50 — Inference")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_csv", default="submission.csv", type=str)
    parser.add_argument("--npz", dest="npz_path", default=None, type=str)
    parser.add_argument("--device", default="auto", type=str, choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--n_bins", default=N_BINS_DEFAULT, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--sigma_min", default=DEFAULT_SIGMA_MIN, type=float)
    args = parser.parse_args()

    cfg = InferenceConfig(
        checkpoint=args.checkpoint,
        output_csv=args.output_csv,
        npz_path=args.npz_path,
        device=args.device,
        precision=args.precision,
        n_bins=args.n_bins,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        sigma_min=args.sigma_min,
    )
    out = infer(cfg)
    LOGGER.info("Done. Submission: %s", out.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()
