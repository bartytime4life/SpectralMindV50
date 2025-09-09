# src/spectramind/predict/core.py
from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from spectramind.submit.format import format_predictions
from spectramind.submit.validate import N_BINS_DEFAULT, validate_dataframe
from spectramind.submit.package import package_submission

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

@dataclass
class PredictConfig:
    # IO
    inputs_dir: Union[str, Path]
    ids_csv: Union[str, Path] = "ids.csv"             # must have 'sample_id' column
    fgs1_path: Union[str, Path] = "fgs1.npy"          # shape: [N, ...] (flexible)
    airs_path: Union[str, Path] = "airs.npy"          # shape: [N, ...] (flexible)
    out_dir: Union[str, Path] = "outputs/predict"
    out_csv_name: str = "submission.csv"
    out_zip_name: str = "submission.zip"

    # Model loading
    model_class: Optional[str] = None   # e.g. "spectramind.models.v50.Model"
    ckpt_paths: Optional[List[Union[str, Path]]] = None  # list of checkpoint paths
    jit_path: Optional[Union[str, Path]] = None          # alternative: torchscript .pt
    model_init_json: Optional[Union[str, Path]] = None   # kwargs for model init (__init__)
    device: str = "cuda"                                  # "cuda"|"cpu"|"mps"
    precision: str = "fp16"                               # "fp32"|"fp16"|"bf16"
    strict_load: bool = True                              # strict state_dict load

    # Inference
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    n_bins: int = N_BINS_DEFAULT
    # reproducibility
    seed: int = 42
    cudnn_benchmark: bool = False

    # Post
    validate: bool = True
    make_zip: bool = True
    report_meta: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------------
# Dataset / Loader
# -------------------------------------------------------------------------

class DualChannelNPY(Dataset):
    """
    Minimal dual-channel dataset:
      - ids_csv: must contain 'sample_id' column (strings)
      - fgs1.npy:  arbitrary shape with first dim N
      - airs.npy:  arbitrary shape with first dim N
    """
    def __init__(self, root: Union[str, Path], ids_csv: Union[str, Path],
                 fgs1_path: Union[str, Path], airs_path: Union[str, Path]) -> None:
        self.root = Path(root)
        self.ids = pd.read_csv(self.root / ids_csv, dtype={"sample_id": str})["sample_id"].tolist()

        self.fgs1 = np.load(self.root / fgs1_path, mmap_mode="r")
        self.airs = np.load(self.root / airs_path, mmap_mode="r")
        if self.fgs1.shape[0] != len(self.ids) or self.airs.shape[0] != len(self.ids):
            raise ValueError(
                f"Length mismatch: ids={len(self.ids)}, fgs1[0]={self.fgs1.shape[0]}, airs[0]={self.airs.shape[0]}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "sample_id": self.ids[idx],
            "fgs1": self.fgs1[idx],
            "airs": self.airs[idx],
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Keep sample_id as list[str], collate arrays to np arrays (convert to torch later)
    out = {
        "sample_id": [b["sample_id"] for b in batch],
        "fgs1": np.stack([b["fgs1"] for b in batch], axis=0),
        "airs": np.stack([b["airs"] for b in batch], axis=0),
    }
    return out


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def _ensure_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for prediction but was not found.")

def _seed_everything(seed: int) -> None:
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def _resolve_device(d: str) -> torch.device:
    d = d.lower()
    if d == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "mps" and _HAS_TORCH and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")

def _load_json(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"model_init_json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def _import_obj(path: str) -> Any:
    """Import an object from 'module:obj' or 'module.obj'."""
    if ":" in path:
        mod, obj = path.split(":", 1)
    else:
        mod, obj = path.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, obj)

def _amp_dtype(precision: str) -> Optional[torch.dtype]:
    p = precision.lower()
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    return None  # fp32


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------

def _load_model(cfg: PredictConfig, device: torch.device) -> Any:
    """
    Load either:
      - TorchScript via cfg.jit_path, or
      - Eager model via cfg.model_class + ckpt_paths[0]
    """
    _ensure_torch()
    if cfg.jit_path:
        model = torch.jit.load(str(cfg.jit_path), map_location=device)
        model.eval()
        return model

    if not cfg.model_class:
        raise ValueError("Either 'jit_path' or 'model_class' must be provided.")
    if not cfg.ckpt_paths or len(cfg.ckpt_paths) == 0:
        raise ValueError("'ckpt_paths' must contain at least one checkpoint path when using 'model_class'.")

    init_kwargs = _load_json(cfg.model_init_json)
    cls = _import_obj(cfg.model_class)
    model = cls(**init_kwargs) if init_kwargs else cls()
    model = model.to(device)
    # load first ckpt (for structure); others will be ensembled in forward loop
    state = torch.load(str(cfg.ckpt_paths[0]), map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=cfg.strict_load)
    model.eval()
    return model


# -------------------------------------------------------------------------
# Predict loop (with optional checkpoint ensembling)
# -------------------------------------------------------------------------

@torch.no_grad()
def _forward_one(model: Any, batch: Dict[str, Any], device: torch.device, amp_dtype: Optional[torch.dtype]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expects model to return (mu, sigma) as torch tensors of shape [B, n_bins].
    Converts inputs:
      - batch['fgs1'], batch['airs'] -> torch tensors on device
    """
    f = torch.from_numpy(batch["fgs1"]).to(device)
    a = torch.from_numpy(batch["airs"]).to(device)

    if amp_dtype is None:
        out = model({"fgs1": f, "airs": a})
    else:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            out = model({"fgs1": f, "airs": a})

    if isinstance(out, (tuple, list)) and len(out) == 2:
        mu_t, sigma_t = out
    elif isinstance(out, Mapping) and "mu" in out and "sigma" in out:
        mu_t, sigma_t = out["mu"], out["sigma"]
    else:
        raise RuntimeError("Model output must be (mu, sigma) or dict with keys 'mu' and 'sigma'.")

    mu = mu_t.detach().cpu().float().numpy()
    sigma = sigma_t.detach().cpu().float().numpy()
    return mu, sigma


def predict_to_dataframe(cfg: PredictConfig) -> pd.DataFrame:
    """
    Run inference and return a formatted submission-ready DataFrame
    (sample_id + mu_000.. + sigma_000..).
    """
    _ensure_torch()
    _seed_everything(cfg.seed)

    device = _resolve_device(cfg.device)
    if _HAS_TORCH and device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    ds = DualChannelNPY(cfg.inputs_dir, cfg.ids_csv, cfg.fgs1_path, cfg.airs_path)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and (device.type == "cuda"),
        collate_fn=_collate,
        drop_last=False,
        shuffle=False,
    )

    model = _load_model(cfg, device)
    amp_dtype = _amp_dtype(cfg.precision)

    # optional multi-ckpt ensembling (average in model space)
    ckpts = list(cfg.ckpt_paths or []) if not cfg.jit_path else []
    if cfg.jit_path:
        ckpts = []  # TorchScript path: single model
    else:
        # ensure first one already loaded; others will be loaded iteratively
        ckpts = ckpts

    all_ids: List[str] = []
    all_mu: List[np.ndarray] = []
    all_sigma: List[np.ndarray] = []

    for batch in dl:
        all_ids.extend(batch["sample_id"])
        mu_ens: Optional[np.ndarray] = None
        sg_ens: Optional[np.ndarray] = None

        # First: current model (already has ckpt_paths[0] loaded)
        mu, sg = _forward_one(model, batch, device, amp_dtype)
        mu_ens = mu
        sg_ens = sg

        # Ensembling with remaining ckpts
        for ck in ckpts[1:]:
            state = torch.load(str(ck), map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=cfg.strict_load)
            model.eval()
            mu_i, sg_i = _forward_one(model, batch, device, amp_dtype)
            mu_ens += mu_i
            sg_ens += sg_i

        n_ens = max(1, len(ckpts))
        mu_ens /= n_ens
        sg_ens /= n_ens

        all_mu.append(mu_ens)
        all_sigma.append(sg_ens)

    mu_np = np.concatenate(all_mu, axis=0)
    sigma_np = np.clip(np.concatenate(all_sigma, axis=0), 1e-9, None)  # guard: non-negative sigma

    df = format_predictions(all_ids, mu_np[:, :cfg.n_bins], sigma_np[:, :cfg.n_bins], n_bins=cfg.n_bins)
    return df


def predict_to_submission(cfg: PredictConfig) -> Path:
    """
    Produce a submission directory with CSV + manifest (+ optional ZIP).
    Validates by default.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = predict_to_dataframe(cfg)

    if cfg.validate:
        report = validate_dataframe(df, n_bins=cfg.n_bins, strict_order=True, check_unique_ids=True, id_field="sample_id")
        report.raise_if_failed()

    # package (CSV + manifest + zip)
    extra = {"predict_config": asdict(cfg)}
    out_zip_or_csv = package_submission(
        df,
        out_dir=out_dir,
        filename=cfg.out_csv_name,
        make_zip=cfg.make_zip,
        zip_name=cfg.out_zip_name,
        n_bins=cfg.n_bins,
        extra_meta=(cfg.report_meta or extra),
    )
    return out_zip_or_csv
