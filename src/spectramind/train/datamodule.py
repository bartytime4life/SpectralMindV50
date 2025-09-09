
# src/spectramind/train/datamodule.py
# =============================================================================
# SpectraMind V50 — Ariel DataModule
# -----------------------------------------------------------------------------
# A flexible LightningDataModule for V50 that supports:
#   • Hydra-instantiated datasets (`_target_`)
#   • NPZ-directory datasets (quick path for Kaggle/experiments)
#   • Single-source NPZ with train/val/test splits
#
# Dual-channel (FGS1 + AIRS) compatible: the collate_fn comes from
# src/spectramind/train/collate.py and will shape/pad variable-length series.
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
#       #   seed: 1337
#
# =============================================================================

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Guarded torch / lightning imports
try:  # pragma: no cover
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
    import pytorch_lightning as pl
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore
    pl = object  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    random_split = None  # type: ignore
    TensorDataset = object  # type: ignore
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

# Local dual-channel collate
from .collate import build_collate_fn, CollateConfig


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
        # DictConfig or dict
        container = OmegaConf.to_container(node, resolve=True) if OmegaConf else dict(node)
        if isinstance(container, dict) and "_target_" in container:
            return instantiate({**container, **overrides})
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------- #
# Simple NPZ dataset helpers (for fast prototyping / Kaggle)
# ----------------------------------------------------------------------------- #

class _NPZPairDataset(Dataset):
    """
    Minimal dataset that loads paired AIRS/FGS1 (and targets if available) from a directory of NPZ files.
    Expected NPZ keys per file:
      - 'airs':  np.ndarray (... time, H, W or channels ...)
      - 'fgs1':  np.ndarray (... time, ...)
      - optional targets:
         • 'mu'   (float32 [283] or [1+283])
         • 'sigma' (float32 [283] or [1+283])
         • 'y'     (observed spectrum, if using supervised targets)
    File naming is free-form; any .npz under the directory will be indexed.

    NOTE: This is a convenience dataset. For full control, prefer Hydra datasets.
    """
    def __init__(self, root: str | Path, require_targets: bool = True) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"NPZ dataset root not found: {self.root}")
        self.files = sorted([p for p in self.root.glob("*.npz")])
        if not self.files:
            raise RuntimeError(f"No .npz files found under {self.root}")
        self.require_targets = require_targets

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        with np_load(path) as data:
            sample: Dict[str, Any] = {}
            # Required inputs
            if "airs" not in data or "fgs1" not in data:
                raise KeyError(f"NPZ missing 'airs'/'fgs1': {path}")
            sample["airs"] = _to_tensor(data["airs"])
            sample["fgs1"] = _to_tensor(data["fgs1"])

            # Optional targets
            for key in ("mu", "sigma", "y"):
                if key in data:
                    sample[key] = _to_tensor(data[key])

            if self.require_targets and ("mu" not in sample or "sigma" not in sample):
                raise KeyError(f"Targets 'mu'/'sigma' required but not found in {path}")

        return sample


def _to_tensor(arr: Any) -> "torch.Tensor":
    t = torch.as_tensor(arr)
    # use float32 by default
    if not t.is_floating_point():
        t = t.float()
    return t.contiguous()


class np_load:
    """
    Tiny context manager for numpy.load with allow_pickle=False to avoid accidental pickles.
    """
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
      • Single-source split: `split_npz_dir` + `split` ratios.

    Collate function (FGS1 + AIRS aware) is injected via `CollateConfig`.

    Example (NPZ split):
      data:
        batch_size: 32
        num_workers: 4
        collate:
          key_fgs1: fgs1
          key_airs: airs
        datamodule:
          _target_: spectramind.train.datamodule.ArielDataModule
          split_npz_dir: ${paths.processed}/all_npz
          split:
            train: 0.8
            val: 0.2
            seed: 1337
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
    ) -> None:
        super().__init__()
        _require_torch()

        # Save hyperparameters (visible in Lightning logs)
        self.save_hyperparameters(ignore=["collate", "train_dataset", "val_dataset", "test_dataset"])

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

        # Placeholders
        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None

    # ------------------------ Lightning API ------------------------ #

    def prepare_data(self) -> None:
        """
        Placeholder for downloading / data checks. For Kaggle/offline usage, no remote IO here.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Build datasets exactly once per stage.
        """
        # Already built
        if stage == "fit" or stage is None:
            if self._train_ds is None or self._val_ds is None:
                self._train_ds, self._val_ds = self._build_train_val()

        if stage == "validate":
            if self._val_ds is None:
                _, self._val_ds = self._build_train_val()

        if stage == "test" or stage == "predict":
            if self._test_ds is None:
                self._test_ds = self._build_test()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            drop_last=self.hparams.drop_last,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    predict_dataloader = test_dataloader

    # ------------------------ Builders ------------------------ #

    def _build_train_val(self) -> Tuple[Dataset, Dataset]:
        # 1) Hydra datasets
        train_ds = _maybe_instantiate(self._train_ds_cfg)
        val_ds = _maybe_instantiate(self._val_ds_cfg)
        if train_ds is not None and val_ds is not None:
            return train_ds, val_ds

        # 2) NPZ directories
        if self._train_npz_dir and self._val_npz_dir:
            return (
                _NPZPairDataset(self._train_npz_dir, require_targets=True),
                _NPZPairDataset(self._val_npz_dir, require_targets=True),
            )

        # 3) Single-source split
        if self._split_npz_dir:
            full = _NPZPairDataset(self._split_npz_dir, require_targets=True)
            train_len = int(round(self._split_cfg.train * len(full)))
            val_len = int(round(self._split_cfg.val * len(full)))
            remain = len(full) - train_len - val_len
            test_len = int(round((self._split_cfg.test or 0.0) * len(full)))
            if test_len > 0:
                # keep train/val; test is built in _build_test to avoid double split
                remain = len(full) - train_len - val_len - test_len

            g = torch.Generator()
            g.manual_seed(int(self._split_cfg.seed))
            parts = [train_len, val_len]
            if remain > 0:
                parts.append(remain)
            subsets = random_split(full, parts, generator=g)

            train_ds = subsets[0]
            val_ds = subsets[1]
            return train_ds, val_ds

        raise RuntimeError(
            "ArielDataModule: unable to build train/val datasets. Provide either "
            "`train_dataset/val_dataset` (Hydra), `train_npz_dir/val_npz_dir` (NPZ), "
            "or `split_npz_dir` (single source)."
        )

    def _build_test(self) -> Dataset:
        # 1) Hydra dataset
        test_ds = _maybe_instantiate(self._test_ds_cfg)
        if test_ds is not None:
            return test_ds

        # 2) NPZ directory
        if self._test_npz_dir:
            return _NPZPairDataset(self._test_npz_dir, require_targets=False)

        # 3) From split: carve out test portion deterministically
        if self._split_npz_dir and (self._split_cfg.test is not None) and self._split_cfg.test > 0.0:
            full = _NPZPairDataset(self._split_npz_dir, require_targets=False)
            total = len(full)
            n_test = int(round(self._split_cfg.test * total))
            if n_test <= 0:
                # fallback: use remaining tail
                n_test = max(1, total - int(round((self._split_cfg.train + self._split_cfg.val) * total)))
            g = torch.Generator()
            g.manual_seed(int(self._split_cfg.seed) + 101)  # different stream
            # We split off exactly `n_test` items; remaining is ignored for test split
            test_subset, _ = random_split(full, [n_test, total - n_test], generator=g)
            return test_subset

        raise RuntimeError(
            "ArielDataModule: unable to build test dataset. Provide `test_dataset` (Hydra), "
            "`test_npz_dir` (NPZ), or specify `split_npz_dir` with a non-zero `split.test`."
        )

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