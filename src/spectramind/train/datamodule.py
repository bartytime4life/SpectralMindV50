# src/spectramind/train/datamodule.py
# =============================================================================
# SpectraMind V50 — Ariel DataModule
# -----------------------------------------------------------------------------
# A flexible LightningDataModule for V50 that supports:
#   • Hydra-instantiated datasets (`_target_`)
#   • NPZ-directory datasets (quick path for Kaggle/experiments)
#   • Single-source NPZ with train/val/test splits (reproducible, indices persisted)
#   • Optional samplers & transforms (Hydra-friendly)
#
# Dual-channel (FGS1 + AIRS) compatible: the collate_fn comes from
# src/spectramind/train/collate.py and can shape/pad variable-length series.
#
# Usage examples (Hydra):
#   data:
#     batch_size: 32
#     num_workers: 4
#     collate:
#       key_fgs1: fgs1
#       key_airs: airs
#     datamodule:
#       _target_: spectramind.train.datamodule.ArielDataModule
#       # Option A: hydra datasets
#       train_dataset:
#         _target_: spectramind.data.ArielTrainDataset
#         root: ${paths.raw}/train
#       val_dataset:
#         _target_: spectramind.data.ArielValDataset
#         root: ${paths.raw}/val
#       # Optionally add samplers/transforms:
#       train_sampler:
#         _target_: spectramind.train.sampler.SamplerConfig
#         kind: auto
#         seed: ${seed}
#       train_transforms:
#         _target_: spectramind.train.transforms.TransformsConfig
#         standardize:
#           per_bin: true
#           batch_axis: 0
#
#       # Option B: npz dirs
#       # train_npz_dir: ${paths.processed}/train_npz
#       # val_npz_dir: ${paths.processed}/val_npz
#       # test_npz_dir: ${paths.processed}/test_npz
#
#       # Option C: single npz dir with split
#       # split_npz_dir: ${paths.processed}/all_npz
#       # split:
#       #   train: 0.8
#       #   val: 0.2
#       #   test: 0.0
#       #   seed: 1337
#
# Notes:
#   • This module avoids any remote/network IO (Kaggle/offline safe).
#   • Split indices are persisted under `<split_npz_dir>/.splits/` for reproducibility.
# =============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List, Union

# Guarded torch / lightning imports
try:  # pragma: no cover
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split, Subset
    import pytorch_lightning as pl
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore
    pl = object  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    random_split = None  # type: ignore
    Subset = object  # type: ignore
    _TORCH_IMPORT_ERROR = _e
else:
    _TORCH_IMPORT_ERROR = None

# Hydra utils (optional at runtime)
try:  # pragma: no cover
    from hydra.utils import instantiate
    from omegaconf import DictConfig, OmegaConf
except Exception:
    instantiate = None  # type: ignore
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore

# Local utilities
from .collate import build_collate_fn, CollateConfig
# (Optional) Samplers & transforms are soft-deps; import lazily in builder calls.
# from .sampler import build_sampler, SamplerConfig
# from .transforms import build_transforms, TransformsConfig, apply_to_batch


# ----------------------------------------------------------------------------- #
# Guards / helpers
# ----------------------------------------------------------------------------- #

def _require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "torch / pytorch_lightning are required to use ArielDataModule."
        ) from _TORCH_IMPORT_ERROR


def _maybe_instantiate(node: Any, **overrides: Any) -> Any:
    """Hydra instantiate if `_target_` is present, else return None."""
    if node is None or instantiate is None:
        return None
    try:
        container = OmegaConf.to_container(node, resolve=True) if OmegaConf else dict(node)
        if isinstance(container, dict) and "_target_" in container:
            return instantiate({**container, **overrides})
    except Exception:
        pass
    return None


def _to_tensor(arr: Any) -> "torch.Tensor":
    t = torch.as_tensor(arr)
    if not t.is_floating_point():
        t = t.float()
    return t.contiguous()


class np_load:
    """Tiny context manager for numpy.load with allow_pickle=False."""
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._f = None

    def __enter__(self):
        import numpy as np
        self._f = np.load(self.path, allow_pickle=False)
        return self._f

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._f is not None:
            self._f.close()
        return False


# ----------------------------------------------------------------------------- #
# Simple NPZ dataset helpers (for fast prototyping / Kaggle)
# ----------------------------------------------------------------------------- #

class _NPZPairDataset(Dataset):
    """
    Minimal dataset that loads paired AIRS/FGS1 (and targets if available) from a directory of NPZ files.
    Expected NPZ keys per file:
      - 'airs':  np.ndarray
      - 'fgs1':  np.ndarray
      - optional targets:
         • 'mu'   (float32 [283] or [1+283])
         • 'sigma' (float32 [283] or [1+283])
         • 'y'     (observed spectrum, if using supervised targets)
    File naming is free-form; any .npz under the directory will be indexed.
    """
    def __init__(self, root: str | Path, require_targets: bool = True) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"NPZ dataset root not found: {self.root}")
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found under {self.root}")
        self.require_targets = require_targets

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        with np_load(path) as data:
            sample: Dict[str, Any] = {}
            if "airs" not in data or "fgs1" not in data:
                raise KeyError(f"NPZ missing 'airs'/'fgs1': {path}")
            sample["airs"] = _to_tensor(data["airs"])
            sample["fgs1"] = _to_tensor(data["fgs1"])
            for key in ("mu", "sigma", "y"):
                if key in data:
                    sample[key] = _to_tensor(data[key])
            if self.require_targets and ("mu" not in sample or "sigma" not in sample):
                raise KeyError(f"Targets 'mu'/'sigma' required but not found in {path}")
        return sample


# ----------------------------------------------------------------------------- #
# Wrapper datasets for on-the-fly transforms (sample-level)
# ----------------------------------------------------------------------------- #

class _TransformDataset(Dataset):
    """
    Wrap a dict-sample dataset and apply a transform pipeline to chosen keys.
    `transforms` is a callable that accepts (x, mask=None) -> x; mapping controls per-key application.
    """
    def __init__(
        self,
        ds: Dataset,
        *,
        transforms: Optional[Any] = None,           # e.g., Compose from transforms.build_transforms
        apply_to_keys: Optional[Sequence[str]] = None,
        mask_keys: Optional[Dict[str, str]] = None,
    ) -> None:
        self.ds = ds
        self.transforms = transforms
        self.apply_to_keys = list(apply_to_keys or [])
        self.mask_keys = dict(mask_keys or {})

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.ds[idx]
        if self.transforms is None or not self.apply_to_keys:
            return sample
        out = dict(sample)
        for key in self.apply_to_keys:
            if key not in out:
                continue
            mask = None
            mask_key = self.mask_keys.get(key)
            if mask_key and mask_key in out:
                mask = out[mask_key]
            out[key] = self.transforms(out[key], mask=mask)
        return out


# ----------------------------------------------------------------------------- #
# Public DataModule
# ----------------------------------------------------------------------------- #

@dataclass
class LoaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    drop_last: bool = False


@dataclass
class SplitConfig:
    train: float = 0.8
    val: float = 0.2
    test: float | None = None
    seed: int = 1337


class ArielDataModule(pl.LightningDataModule):  # type: ignore
    """
    LightningDataModule for SpectraMind V50 with strong flexibility:
      • Hydra datasets: supply `train_dataset`, `val_dataset`, `test_dataset` nodes with `_target_`.
      • NPZ directories: `train_npz_dir`, `val_npz_dir`, `test_npz_dir`.
      • Single-source split: `split_npz_dir` + `split` ratios (persisted indices).
      • Optional samplers & transforms per split.

    Collate function (FGS1 + AIRS aware) is injected via `CollateConfig`.
    """

    def __init__(
        self,
        # Loader knobs
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = 2,
        drop_last: bool = False,

        # Collate config
        collate: Optional[Dict[str, Any]] = None,

        # Hydra dataset nodes (each optional)
        train_dataset: Optional[DictConfig | Dict[str, Any]] = None,
        val_dataset: Optional[DictConfig | Dict[str, Any]] = None,
        test_dataset: Optional[DictConfig | Dict[str, Any]] = None,

        # NPZ directories (each optional)
        train_npz_dir: Optional[str | Path] = None,
        val_npz_dir: Optional[str | Path] = None,
        test_npz_dir: Optional[str | Path] = None,

        # Single NPZ source with split
        split_npz_dir: Optional[str | Path] = None,
        split: Optional[Dict[str, Any]] = None,

        # (Optional) Samplers per split (Hydra)
        train_sampler: Optional[Dict[str, Any]] = None,
        val_sampler: Optional[Dict[str, Any]] = None,
        test_sampler: Optional[Dict[str, Any]] = None,

        # (Optional) Transforms per split (Hydra)
        train_transforms: Optional[Dict[str, Any]] = None,
        val_transforms: Optional[Dict[str, Any]] = None,
        test_transforms: Optional[Dict[str, Any]] = None,

        # Optional: which keys to apply transforms to (default: both inputs)
        transform_keys: Optional[Sequence[str]] = ("airs", "fgs1"),
        transform_mask_keys: Optional[Dict[str, str]] = None,

        # Split indices cache directory (for single-source split)
        split_cache_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        _require_torch()

        # Save hyperparameters (visible in Lightning logs)
        self.save_hyperparameters(ignore=[
            "collate",
            "train_dataset", "val_dataset", "test_dataset",
            "train_transforms", "val_transforms", "test_transforms",
            "transform_keys", "transform_mask_keys",
        ])

        # Build collate_fn from CollateConfig
        if collate is None:
            self.collate_cfg = CollateConfig()  # defaults
        else:
            try:
                self.collate_cfg = CollateConfig(**(
                    OmegaConf.to_container(collate, resolve=True) if OmegaConf else dict(collate)
                ))
            except Exception:
                self.collate_cfg = CollateConfig()
        self.collate_fn = build_collate_fn(self.collate_cfg)

        # Store dataset configs / paths
        self._train_ds_cfg = train_dataset
        self._val_ds_cfg = val_dataset
        self._test_ds_cfg = test_dataset

        self._train_npz_dir = Path(train_npz_dir) if train_npz_dir else None
        self._val_npz_dir = Path(val_npz_dir) if val_npz_dir else None
        self._test_npz_dir = Path(test_npz_dir) if test_npz_dir else None

        self._split_npz_dir = Path(split_npz_dir) if split_npz_dir else None
        self._split_cfg = self._normalize_split(split)
        self._split_cache_dir = Path(split_cache_dir) if split_cache_dir else (
            (self._split_npz_dir / ".splits") if self._split_npz_dir else None
        )

        # Sampler/transform configs (store raw; build lazily to avoid imports at init)
        self._train_sampler_cfg = train_sampler
        self._val_sampler_cfg = val_sampler
        self._test_sampler_cfg = test_sampler

        self._train_tf_cfg = train_transforms
        self._val_tf_cfg = val_transforms
        self._test_tf_cfg = test_transforms
        self._transform_keys = list(transform_keys or [])
        self._transform_mask_keys = dict(transform_mask_keys or {})

        # Placeholders
        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None

        # Persisted split indices (for single-source split)
        self._split_indices: Optional[Dict[str, List[int]]] = None

    # ------------------------ Lightning API ------------------------ #

    def prepare_data(self) -> None:
        """Placeholder for network IO / dataset downloads; do nothing for Kaggle/offline."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Build datasets exactly once per stage."""
        if stage in (None, "fit"):
            if self._train_ds is None or self._val_ds is None:
                self._train_ds, self._val_ds = self._build_train_val()
        if stage == "validate":
            if self._val_ds is None:
                _, self._val_ds = self._build_train_val()
        if stage in ("test", "predict"):
            if self._test_ds is None:
                self._test_ds = self._build_test()

    # dataloaders ---------------------------------------------------- #

    def train_dataloader(self) -> DataLoader:
        sampler = self._build_sampler(self._train_sampler_cfg, self._train_ds, split="train")
        # If sampler present, don't shuffle in DataLoader
        shuffle = bool(self.hparams.shuffle) and (sampler is None)
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            drop_last=self.hparams.drop_last,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = self._build_sampler(self._val_sampler_cfg, self._val_ds, split="val")
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = self._build_sampler(self._test_sampler_cfg, self._test_ds, split="test")
        return DataLoader(
            self._test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    predict_dataloader = test_dataloader

    # ------------------------ Builders ------------------------ #

    def _maybe_wrap_with_transforms(self, ds: Dataset, tf_cfg: Optional[Dict[str, Any]]) -> Dataset:
        if tf_cfg is None:
            return ds
        # Lazy import of transforms to keep module import light
        from .transforms import build_transforms
        transforms = build_transforms(
            OmegaConf.to_container(tf_cfg, resolve=True) if OmegaConf else dict(tf_cfg)
        )
        return _TransformDataset(
            ds, transforms=transforms, apply_to_keys=self._transform_keys,
            mask_keys=self._transform_mask_keys,
        )

    def _build_train_val(self) -> Tuple[Dataset, Dataset]:
        # 1) Hydra datasets
        train_ds = _maybe_instantiate(self._train_ds_cfg)
        val_ds = _maybe_instantiate(self._val_ds_cfg)
        if train_ds is not None and val_ds is not None:
            return (
                self._maybe_wrap_with_transforms(train_ds, self._train_tf_cfg),
                self._maybe_wrap_with_transforms(val_ds, self._val_tf_cfg),
            )

        # 2) NPZ directories
        if self._train_npz_dir and self._val_npz_dir:
            return (
                self._maybe_wrap_with_transforms(_NPZPairDataset(self._train_npz_dir, require_targets=True), self._train_tf_cfg),
                self._maybe_wrap_with_transforms(_NPZPairDataset(self._val_npz_dir, require_targets=True), self._val_tf_cfg),
            )

        # 3) Single-source split
        if self._split_npz_dir:
            full = _NPZPairDataset(self._split_npz_dir, require_targets=True)
            train_idx, val_idx, _ = self._get_or_make_split_indices(len(full))
            train_ds = Subset(full, train_idx)
            val_ds = Subset(full, val_idx)
            return (
                self._maybe_wrap_with_transforms(train_ds, self._train_tf_cfg),
                self._maybe_wrap_with_transforms(val_ds, self._val_tf_cfg),
            )

        raise RuntimeError(
            "ArielDataModule: unable to build train/val datasets. Provide either "
            "`train_dataset/val_dataset` (Hydra), `train_npz_dir/val_npz_dir` (NPZ), "
            "or `split_npz_dir` (single source)."
        )

    def _build_test(self) -> Dataset:
        # 1) Hydra dataset
        test_ds = _maybe_instantiate(self._test_ds_cfg)
        if test_ds is not None:
            return self._maybe_wrap_with_transforms(test_ds, self._test_tf_cfg)

        # 2) NPZ directory
        if self._test_npz_dir:
            return self._maybe_wrap_with_transforms(_NPZPairDataset(self._test_npz_dir, require_targets=False), self._test_tf_cfg)

        # 3) From split: carve out test portion
        if self._split_npz_dir and (self._split_cfg.test is not None) and self._split_cfg.test > 0.0:
            full = _NPZPairDataset(self._split_npz_dir, require_targets=False)
            _, _, test_idx = self._get_or_make_split_indices(len(full))
            test_ds = Subset(full, test_idx)
            return self._maybe_wrap_with_transforms(test_ds, self._test_tf_cfg)

        raise RuntimeError(
            "ArielDataModule: unable to build test dataset. Provide `test_dataset` (Hydra), "
            "`test_npz_dir` (NPZ), or specify `split_npz_dir` with a non-zero `split.test`."
        )

    # ------------------------ Sampler builder ------------------------ #

    def _build_sampler(self, cfg: Optional[Dict[str, Any]], dataset: Optional[Dataset], *, split: str) -> Optional[Any]:
        if cfg is None or dataset is None:
            return None
        # Lazy import to keep module import light
        from .sampler import build_sampler
        return build_sampler(dataset, cfg)

    # ------------------------ Split management (persisted) ------------------------ #

    def _split_cache_file(self) -> Optional[Path]:
        if self._split_cache_dir is None:
            return None
        self._split_cache_dir.mkdir(parents=True, exist_ok=True)
        fname = f"split_seed{self._split_cfg.seed}_t{self._split_cfg.train:.4f}_v{self._split_cfg.val:.4f}_s.json"
        return self._split_cache_dir / fname

    def _get_or_make_split_indices(self, total_len: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Return (train_idx, val_idx, test_idx). Test idx may be empty if split.test is 0/None.
        Indices are reproducible (seeded) and persisted to disk.
        """
        train_r = float(self._split_cfg.train)
        val_r = float(self._split_cfg.val)
        test_r = float(self._split_cfg.test or 0.0)
        cache_file = self._split_cache_file()

        if cache_file and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                idx_train = list(map(int, data["train"]))
                idx_val = list(map(int, data["val"]))
                idx_test = list(map(int, data.get("test", [])))
                if len(idx_train) + len(idx_val) + len(idx_test) <= total_len:
                    return idx_train, idx_val, idx_test
            except Exception:
                pass  # regenerate

        # Build fresh split
        g = torch.Generator()
        g.manual_seed(int(self._split_cfg.seed))

        all_idx = torch.randperm(total_len, generator=g).tolist()
        n_test = int(round(test_r * total_len))
        n_train = int(round(train_r * total_len))
        n_val = int(round(val_r * total_len))

        # make sure we don't exceed total_len due to rounding
        taken = n_test + n_train + n_val
        if taken > total_len:
            # shrink val first, then train
            overflow = taken - total_len
            drop_val = min(overflow, n_val)
            n_val -= drop_val
            overflow -= drop_val
            if overflow > 0:
                n_train -= overflow

        # allocate
        idx_test = all_idx[:n_test]
        idx_train = all_idx[n_test:n_test + n_train]
        idx_val = all_idx[n_test + n_train:n_test + n_train + n_val]

        # persist
        if cache_file:
            try:
                cache_file.write_text(json.dumps({"train": idx_train, "val": idx_val, "test": idx_test}), encoding="utf-8")
            except Exception:
                pass

        return idx_train, idx_val, idx_test

    # ------------------------ Utils ------------------------ #

    @staticmethod
    def _normalize_split(split: Optional[Dict[str, Any]]) -> SplitConfig:
        if split is None:
            return SplitConfig()
        if OmegaConf is not None and hasattr(split, "keys"):
            try:
                split = OmegaConf.to_container(split, resolve=True)  # type: ignore
            except Exception:
                split = dict(split)
        sc = SplitConfig(**{**SplitConfig().__dict__, **dict(split or {})})
        # Basic sanity
        parts = [p for p in (sc.train, sc.val, sc.test or 0.0) if p is not None]
        if any(x < 0 for x in parts):
            raise ValueError("Split ratios must be non-negative.")
        if sc.train + sc.val + (sc.test or 0.0) > 1.0 + 1e-6:
            raise ValueError("Split ratios must sum to <= 1.0.")
        return sc

    # Convenient properties
    @property
    def steps_per_epoch(self) -> Optional[int]:
        try:
            n = len(self._train_ds)  # type: ignore[arg-type]
            bs = int(self.hparams.batch_size)
            return max(1, n // bs) if n and bs else None
        except Exception:
            return None