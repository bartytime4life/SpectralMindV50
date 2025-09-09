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


# =============================================================================
# Config
# =============================================================================

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

    # Numerical guards
    sigma_floor: float = 1e-9  # minimum σ (avoid zero/neg and NLL blowups)


# =============================================================================
# Dataset / Loader
# =============================================================================

class DualChannelNPY(Dataset):
    """
    Minimal dual-channel dataset:
      - ids_csv: must contain 'sample_id' column (strings)
      - fgs1.npy: arbitrary shape with first dim N
      - airs.npy: arbitrary shape with first dim N
    """
    def __init__(self, root: Union[str, Path], ids_csv: Union[str, Path],
                 fgs1_path: Union[str, Path], airs_path: Union[str, Path]) -> None:
        self.root = Path(root)
        ids_fp = self.root / ids_csv
        fgs_fp = self.root / fgs1_path
        airs_fp = self.root / airs_path
        for fp, desc in [(ids_fp, "ids.csv"), (fgs_fp, "FGS1 .npy"), (airs_fp, "AIRS .npy")]:
            if not fp.exists():
                raise FileNotFoundError(f"Missing {desc}: {fp}")

        # ids
        df_ids = pd.read_csv(ids_fp, dtype={"sample_id": str})
        if "sample_id" not in df_ids.columns:
            raise ValueError(f"ids file {ids_fp} must contain a 'sample_id' column, got columns={list(df_ids.columns)}")
        self.ids: List[str] = df_ids["sample_id"].tolist()
        if len(self.ids) == 0:
            raise ValueError(f"No sample_id rows found in ids file {ids_fp}")

        # memmap npy
        try:
            self.fgs1 = np.load(fgs_fp, mmap_mode="r")
            self.airs = np.load(airs_fp, mmap_mode="r")
        except Exception as e:
            raise RuntimeError(f"Failed to load npy arrays: {e}")

        if self.fgs1.ndim < 1 or self.airs.ndim < 1:
            raise ValueError(f"Expected arrays with first dimension N; got shapes fgs1={self.fgs1.shape}, airs={self.airs.shape}")
        if self.fgs1.shape[0] != len(self.ids) or self.airs.shape[0] != len(self.ids):
            raise ValueError(
                f"Length mismatch: ids={len(self.ids)}, fgs1[0]={self.fgs1.shape[0]}, airs[0]={self.airs.shape[0]}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # NOTE: slices into memmap; do not modify arrays in-place
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


# =============================================================================
# Utilities
# =============================================================================

def _ensure_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for prediction but was not found.")


def _seed_everything(seed: int) -> None:
    # numpy
    np.random.seed(seed)
    # torch
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cuDNN determinism trade-offs; benchmark disabled unless explicitly set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON {p}: {e}")


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


def _safe_numpy(x: np.ndarray, name: str) -> np.ndarray:
    """Ensure output has finite values."""
    if not np.all(np.isfinite(x)):
        raise ValueError(f"Non-finite values detected in {name}")
    return x


def _check_model_output(mu: torch.Tensor, sigma: torch.Tensor, n_bins: int) -> None:
    if mu.ndim != 2 or sigma.ndim != 2:
        raise RuntimeError(f"Model outputs must be 2D [B, n_bins], got mu={mu.shape}, sigma={sigma.shape}")
    if mu.shape != sigma.shape:
        raise RuntimeError(f"mu and sigma shapes must match, got mu={mu.shape}, sigma={sigma.shape}")
    if mu.shape[1] < n_bins or sigma.shape[1] < n_bins:
        raise RuntimeError(f"Model produced fewer bins than requested n_bins={n_bins}: mu={mu.shape}, sigma={sigma.shape}")


# =============================================================================
# Model loading
# =============================================================================

def _load_model(cfg: PredictConfig, device: torch.device) -> Any:
    """
    Load either:
      - TorchScript via cfg.jit_path, or
      - Eager model via cfg.model_class + ckpt_paths[0]
    """
    _ensure_torch()
    if cfg.jit_path:
        p = Path(cfg.jit_path)
        if not p.exists():
            raise FileNotFoundError(f"TorchScript model not found: {p}")
        model = torch.jit.load(str(p), map_location=device)
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
    missing, unexpected = model.load_state_dict(state, strict=cfg.strict_load)
    if not cfg.strict_load and (missing or unexpected):  # type: ignore[truthy-function]
        # provide a soft warning via exception message
        raise RuntimeError(f"Non-strict load had missing={missing} unexpected={unexpected}")
    model.eval()
    return model


# =============================================================================
# Predict loop (with optional checkpoint ensembling)
# =============================================================================

@torch.no_grad()
def _forward_one(model: Any, batch: Dict[str, Any], device: torch.device, amp_dtype: Optional[torch.dtype], n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expects model to return (mu, sigma) as torch tensors of shape [B, n_bins] (or more bins).
    Converts inputs:
      - batch['fgs1'], batch['airs'] -> torch tensors on device
    """
    f = torch.from_numpy(batch["fgs1"]).to(device)
    a = torch.from_numpy(batch["airs"]).to(device)

    # CPU autocast is supported in modern PyTorch; guard if unavailable
    use_amp = amp_dtype is not None and (device.type in ("cuda", "cpu"))

    if use_amp:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):  # type: ignore[arg-type]
            out = model({"fgs1": f, "airs": a})
    else:
        out = model({"fgs1": f, "airs": a})

    if isinstance(out, (tuple, list)) and len(out) == 2:
        mu_t, sigma_t = out
    elif isinstance(out, Mapping) and "mu" in out and "sigma" in out:
        mu_t, sigma_t = out["mu"], out["sigma"]
    else:
        raise RuntimeError("Model output must be (mu, sigma) or dict with keys 'mu' and 'sigma'.")

    if not isinstance(mu_t, torch.Tensor) or not isinstance(sigma_t, torch.Tensor):
        raise RuntimeError("Model outputs must be torch.Tensor")

    _check_model_output(mu_t, sigma_t, n_bins=n_bins)

    # Slice to requested n_bins (in case the model returns more)
    mu = mu_t[:, :n_bins].detach().cpu().float().numpy()
    sigma = sigma_t[:, :n_bins].detach().cpu().float().numpy()
    return _safe_numpy(mu, "mu"), _safe_numpy(sigma, "sigma")


def predict_to_dataframe(cfg: PredictConfig) -> pd.DataFrame:
    """
    Run inference and return a formatted submission-ready DataFrame
    (sample_id + mu_000.. + sigma_000..).
    """
    _ensure_torch()
    _seed_everything(cfg.seed)

    device = _resolve_device(cfg.device)
    if _HAS_TORCH and device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark  # user-controlled

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
    # First model already loaded from ckpts[0] (if eager)

    all_ids: List[str] = []
    all_mu: List[np.ndarray] = []
    all_sigma: List[np.ndarray] = []

    for batch in dl:
        all_ids.extend(batch["sample_id"])

        # Start with current model (with ckpt_paths[0] loaded if eager)
        mu_ens, sg_ens = _forward_one(model, batch, device, amp_dtype, n_bins=cfg.n_bins)

        # Ensembling with remaining ckpts (eager)
        for ck in ckpts[1:]:
            state = torch.load(str(ck), map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=cfg.strict_load)
            model.eval()
            mu_i, sg_i = _forward_one(model, batch, device, amp_dtype, n_bins=cfg.n_bins)
            mu_ens += mu_i
            sg_ens += sg_i

        n_ens = max(1, len(ckpts))
        mu_ens /= n_ens
        sg_ens /= n_ens

        all_mu.append(mu_ens)
        all_sigma.append(sg_ens)

    mu_np = np.concatenate(all_mu, axis=0)
    sigma_np = np.concatenate(all_sigma, axis=0)

    # numeric guards
    sigma_np = np.clip(sigma_np, cfg.sigma_floor, None)  # guard: non-negative, non-zero
    if not np.all(np.isfinite(mu_np)) or not np.all(np.isfinite(sigma_np)):
        raise ValueError("Non-finite values encountered in predictions.")

    df = format_predictions(all_ids, mu_np[:, :cfg.n_bins], sigma_np[:, :cfg.n_bins], n_bins=cfg.n_bins)
    return df


def predict_to_submission(cfg: PredictConfig) -> Path:
    """
    Produce a submission directory with CSV + manifest (+ optional ZIP).
    Validates by default and writes a predict_config.json snapshot for reproducibility.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot (minimal JSON-serializable)
    cfg_snapshot = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in asdict(cfg).items()
    }
    (out_dir / "predict_config.json").write_text(json.dumps(cfg_snapshot, indent=2), encoding="utf-8")

    df = predict_to_dataframe(cfg)

    if cfg.validate:
        report = validate_dataframe(df, n_bins=cfg.n_bins, strict_order=True, check_unique_ids=True)
        report.raise_if_failed()

    # package (CSV + manifest + zip)
    extra_meta = dict(cfg.report_meta or {})
    extra_meta.setdefault("predict_config", cfg_snapshot)

    out_zip_or_csv = package_submission(
        df,
        out_dir=out_dir,
        filename=cfg.out_csv_name,
        make_zip=cfg.make_zip,
        zip_name=cfg.out_zip_name,
        n_bins=cfg.n_bins,
        extra_meta=extra_meta,
    )
    return out_zip_or_csv
