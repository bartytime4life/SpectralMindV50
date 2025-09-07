# src/spectramind/predict/inference.py
# -----------------------------------------------------------------------------
# SpectraMind V50 — Inference / Prediction
# -----------------------------------------------------------------------------
# - Loads a trained heteroscedastic model checkpoint
# - Runs batched inference (FGS1 + AIRS or generic 'inputs')
# - Emits Kaggle-ready submission CSV with columns:
#       id, mu_000..mu_282, sigma_000..sigma_282
#
# Notes
# -----
# * Designed to be Hydra-friendly (instantiate dataset/model from config),
#   but includes safe fallbacks for npz-based datasets.
# * Torch-only, Kaggle-safe, no internet calls.
# * Numerically robust: clamps/cleans σ and replaces non-finite outputs.
# * Evaluates under torch.no_grad() and model.eval() on the chosen device.
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    # Hydra / OmegaConf are optional (but recommended). We soft-import to keep Kaggle-safe.
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
except Exception:  # pragma: no cover - optional
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore

LOGGER = logging.getLogger("spectramind.inference")
BIN_COUNT = 283
MU_PREFIX = "mu_"
SIGMA_PREFIX = "sigma_"
DEFAULT_SIGMA_MIN = 1e-8


# -----------------------------------------------------------------------------
# Basic NPZ dataset fallback (Kaggle/CI-safe)
# -----------------------------------------------------------------------------
class NPZPredictDataset(Dataset):
    """
    Minimal inference dataset reading from an .npz bundle.

    Expected arrays:
      - ids:   shape [N], dtype str or bytes (sample identifiers)
      - fgs1:  shape [N, ...]  (optional)
      - airs:  shape [N, ...]  (optional)
      - inputs: shape [N, ...] (optional, generic single-branch)

    The model forward should accept one of:
      (fgs1=tensor, airs=tensor) or (inputs=tensor)
    """

    def __init__(self, npz_path: Union[str, Path]):
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"Predict npz not found: {p}")
        bundle = np.load(p, allow_pickle=False)
        self._ids = _to_str_array(bundle["ids"])
        self._has_fgs1 = "fgs1" in bundle.files
        self._has_airs = "airs" in bundle.files
        self._has_inputs = "inputs" in bundle.files

        if not (self._has_inputs or (self._has_fgs1 and self._has_airs)):
            raise ValueError(
                "NPZ must contain either 'inputs' or both 'fgs1' and 'airs' arrays."
            )
        self._fgs1 = bundle["fgs1"] if self._has_fgs1 else None
        self._airs = bundle["airs"] if self._has_airs else None
        self._inputs = bundle["inputs"] if self._has_inputs else None

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {"id": self._ids[idx]}
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
    # Anything else: cast robustly
    return x.astype(str)


# -----------------------------------------------------------------------------
# Inference configuration
# -----------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    # Paths
    checkpoint: str
    output_csv: str = "submission.csv"

    # Data
    # Either provide 'dataset' hydra node OR a raw npz path
    dataset: Optional[DictConfig] = None
    npz_path: Optional[str] = None

    # Loader
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True

    # Device / performance
    device: str = "auto"  # ["auto", "cuda", "cpu", "mps"]
    dtype: str = "float32"  # reserved for future use

    # Numeric safety
    sigma_min: float = DEFAULT_SIGMA_MIN
    replace_nonfinite: bool = True

    # CSV formatting
    use_wide_format: bool = True  # wide columns mu_000..mu_282 / sigma_000..sigma_282
    id_column: str = "id"


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def select_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if pref == "mps" or (pref == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def load_model_from_checkpoint(ckpt_path: Union[str, Path], device: torch.device) -> torch.nn.Module:
    """
    Loads a torch model checkpoint. Supports:
      - direct state_dict checkpoints
      - lightning checkpoints with 'state_dict' key
      - scripted/trace models (torch.jit.load)
    The caller is responsible for instantiation if using Hydra (recommended).
    """
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    # Try JIT first (handy for Kaggle deployment)
    try:
        model = torch.jit.load(str(p), map_location=device)
        LOGGER.info("Loaded TorchScript model from %s", p)
        return model
    except Exception:
        pass

    # Otherwise assume plain / lightning checkpoint with a config-managed model
    payload = torch.load(str(p), map_location=device)
    state = None
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
        LOGGER.info("Loaded Lightning-style checkpoint (state_dict) from %s", p)
    elif isinstance(payload, Mapping):
        # best guess
        state = payload
        LOGGER.info("Loaded raw checkpoint (mapping) from %s", p)
    else:
        raise ValueError(f"Unsupported checkpoint format at {p}")

    # The actual model must be instantiated by Hydra outside and provided here,
    # however, to keep this module self-contained we attempt a lazy import path
    # via the checkpoint payload if available (rare). If not, instruct the user.
    raise RuntimeError(
        "Checkpoint loaded, but no model instance to load into.\n"
        "Use Hydra to instantiate your model and call model.load_state_dict(state_dict), "
        "or export a TorchScript model for TorchScript loading."
    )


# -----------------------------------------------------------------------------
# Inference pipeline
# -----------------------------------------------------------------------------
def build_dataset(cfg: InferenceConfig) -> Dataset:
    if cfg.dataset is not None and OmegaConf is not None:
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
        pin_memory=cfg.pin_memory,
        shuffle=False,
        drop_last=False,
    )


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    replace_nonfinite: bool = True,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    ids: List[str] = []
    mus: List[np.ndarray] = []
    sigmas: List[np.ndarray] = []

    for batch in loader:
        # Move tensors to device
        feed: Dict[str, Any] = {}
        for k, v in batch.items():
            if k == "id":
                continue
            if torch.is_tensor(v):
                feed[k] = v.to(device, non_blocking=True)
            else:
                # pass through (e.g., metadata)
                feed[k] = v

        # Forward
        out = model(**feed) if _expects_kwargs(model) else model(feed)
        mu, sigma = _extract_mu_sigma(out)  # both torch tensors [B, BIN_COUNT]

        mu = mu.detach().float()
        sigma = sigma.detach().float()

        # Numeric hygiene
        if sigma_min is not None and sigma_min > 0:
            sigma = torch.clamp(sigma, min=float(sigma_min))

        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()

        if replace_nonfinite:
            mu_np = _clean_nonfinite(mu_np, fill_value=0.0)
            sigma_np = _clean_nonfinite(sigma_np, fill_value=float(sigma_min or DEFAULT_SIGMA_MIN))

        # Collect ids
        if "id" in batch:
            batch_ids = _to_str_array(np.array(batch["id"]))
        else:
            # Fallback: fabricate integer ids if missing
            start = len(ids)
            batch_ids = np.array([str(start + i) for i in range(mu_np.shape[0])], dtype=str)

        ids.extend(batch_ids.tolist())
        mus.append(mu_np)
        sigmas.append(sigma_np)

    mu_all = np.concatenate(mus, axis=0) if mus else np.zeros((0, BIN_COUNT), dtype=np.float32)
    sigma_all = np.concatenate(sigmas, axis=0) if sigmas else np.zeros((0, BIN_COUNT), dtype=np.float32)

    return ids, mu_all, sigma_all


def _expects_kwargs(model: torch.nn.Module) -> bool:
    # Heuristic: if model.forward signature starts with **kwargs or (fgs1, airs)...
    import inspect

    sig = inspect.signature(model.forward)
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    names = list(sig.parameters.keys())
    return set(names) & {"fgs1", "airs", "inputs"} != set()


def _extract_mu_sigma(out: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accept a variety of model outputs:
      - dict with keys {'mu','sigma'} or {'mu','log_sigma'}
      - tuple(mu, sigma)
      - object with attributes .mu and .sigma
    Returns torch tensors of shape [B, BIN_COUNT].
    """
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

    if mu.ndim != 2 or mu.shape[-1] != BIN_COUNT:
        raise ValueError(f"mu must have shape [B, {BIN_COUNT}], got {tuple(mu.shape)}")
    if sigma.ndim != 2 or sigma.shape[-1] != BIN_COUNT:
        raise ValueError(f"sigma must have shape [B, {BIN_COUNT}], got {tuple(sigma.shape)}")

    return mu, sigma


def _clean_nonfinite(x: np.ndarray, fill_value: float) -> np.ndarray:
    if not np.all(np.isfinite(x)):
        bad = ~np.isfinite(x)
        x = x.copy()
        x[bad] = fill_value
    return x


# -----------------------------------------------------------------------------
# Submission writer
# -----------------------------------------------------------------------------
def save_submission_csv(
    ids: Sequence[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    csv_path: Union[str, Path],
    id_column: str = "id",
) -> None:
    """
    Writes a Kaggle-ready submission CSV file with columns:
      id, mu_000..mu_282, sigma_000..sigma_282
    """
    n = len(ids)
    if mu.shape != (n, BIN_COUNT) or sigma.shape != (n, BIN_COUNT):
        raise ValueError(
            f"mu/sigma shapes must be ({n}, {BIN_COUNT}), got {mu.shape}/{sigma.shape}"
        )

    cols = [id_column] + [f"{MU_PREFIX}{i:03d}" for i in range(BIN_COUNT)] + [
        f"{SIGMA_PREFIX}{i:03d}" for i in range(BIN_COUNT)
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
def infer(cfg: Union[InferenceConfig, DictConfig, Mapping[str, Any]]) -> Path:
    """
    Entry point for programmatic use (Hydra-friendly).
    Expect either:
      - InferenceConfig dataclass
      - Hydra DictConfig / plain dict with matching fields
    """
    if not isinstance(cfg, InferenceConfig):
        cfg = _coerce_inference_config(cfg)

    # Build dataset + loader
    dataset = build_dataset(cfg)
    loader = build_loader(dataset, cfg)

    # Device
    device = select_device(cfg.device)
    LOGGER.info("Using device: %s", device)

    # Load model
    try:
        model = load_model_from_checkpoint(cfg.checkpoint, device=device)
    except RuntimeError as e:
        # The recommended path: instantiate model via Hydra and load_state_dict here.
        # We surface a clear error message, but for convenience,
        # attempt to import a factory function if provided in cfg.
        if hasattr(cfg, "model") and cfg.model is not None and OmegaConf is not None:
            model = instantiate(cfg.model)
            payload = torch.load(cfg.checkpoint, map_location=device)
            state = payload["state_dict"] if isinstance(payload, Mapping) and "state_dict" in payload else payload
            model.load_state_dict(state, strict=False)
            model.to(device)
            LOGGER.info("Instantiated model via Hydra and loaded state_dict.")
        else:
            raise e

    ids, mu, sigma = run_inference(
        model=model,
        loader=loader,
        device=device,
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


def _coerce_inference_config(x: Union[DictConfig, Mapping[str, Any]]) -> InferenceConfig:
    if OmegaConf is not None and isinstance(x, DictConfig):
        x = OmegaConf.to_container(x, resolve=True)
    return InferenceConfig(**dict(x))  # type: ignore


# -----------------------------------------------------------------------------
# CLI helper
# -----------------------------------------------------------------------------
def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )


def main():
    """
    Minimal CLI without Typer/Click to keep Kaggle-safe.
    Usage (Hydra users typically call infer(cfg) from their CLI):
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        sigma_min=args.sigma_min,
    )
    out = infer(cfg)
    LOGGER.info("Done. Submission: %s", out.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()
