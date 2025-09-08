# src/spectramind/data/datamodule.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

try:
    import lightning as L  # Lightning 2.x
except Exception as _e:
    # Delay the import error until class construction to keep import-time light.
    L = None  # type: ignore


__all__ = [
    "SpectraDataModule",
    "DualChannelDataset",
    "FGS1Dataset",
    "AIRSDataset",
]


# ==============================================================================
# Utilities
# ==============================================================================


def _is_kaggle() -> bool:
    return any(
        k in os.environ
        for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DOCKER_IMAGE")
    ) or os.path.exists("/kaggle/input")


def _maybe_float32(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(x)
    return t.to(dtype=torch.float32) if t.dtype.is_floating_point else t


def _load_blob(path: Path) -> Dict[str, np.ndarray]:
    """
    Load a single-sample blob. Supports:
      - .npy → array
      - .npz → dict-like with arrays
    For .npy we assume the caller knows what it represents (fgs1/airs/target)
    and passes the right filename. For .npz, we expect keys like 'fgs1','airs','y'.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing blob: {path}")
    if path.suffix == ".npy":
        arr = np.load(path)
        return {"data": arr}
    if path.suffix == ".npz":
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    raise ValueError(f"Unsupported blob extension: {path.suffix} ({path})")


def _read_manifest(csv_path: Path) -> List[Dict[str, str]]:
    """
    Read a CSV manifest with at least 'sample_id' column.
    Optional columns:
      - 'split' (values: train/val/test or custom)
      - 'y' or 'target' (regression; floats) or multiple target columns
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest missing 'sample_id' column: {csv_path}")
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def _deterministic_split(
    dataset: Dataset[Any],
    lengths: Sequence[int],
    seed: int,
) -> List[Subset[Any]]:
    """
    Deterministic torch random_split wrapper. Works without global seed pollution.
    """
    g = torch.Generator()
    g.manual_seed(int(seed) & 0xFFFFFFFF)
    return list(random_split(dataset, lengths, generator=g))


# ==============================================================================
# Datasets
# ==============================================================================


class FGS1Dataset(Dataset[Dict[str, Any]]):
    """
    Loads FGS1 time-series arrays per sample_id from a directory of files.
    Supports either:
      - <fgs1_dir>/<sample_id>.npy  (single array)
      - <fgs1_dir>/<sample_id>.npz  (expects key 'fgs1' or 'data')
    """

    def __init__(self, fgs1_dir: Path, sample_ids: Sequence[str]) -> None:
        self.fgs1_dir = Path(fgs1_dir)
        self.sample_ids = list(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.sample_ids[idx]
        npy = self.fgs1_dir / f"{sid}.npy"
        npz = self.fgs1_dir / f"{sid}.npz"

        blob: Dict[str, np.ndarray]
        if npy.exists():
            blob = _load_blob(npy)
            arr = blob["data"]
        elif npz.exists():
            blob = _load_blob(npz)
            arr = blob.get("fgs1", blob.get("data"))
            if arr is None:
                raise KeyError(f"NPZ missing 'fgs1' or 'data' key: {npz}")
        else:
            raise FileNotFoundError(f"No FGS1 blob for sample_id={sid} in {self.fgs1_dir}")

        x_fgs1 = _maybe_float32(arr)
        return {"sample_id": sid, "fgs1": x_fgs1}


class AIRSDataset(Dataset[Dict[str, Any]]):
    """
    Loads AIRS spectral arrays per sample_id from a directory of files.
    Supports:
      - <airs_dir>/<sample_id>.npy  (single array)
      - <airs_dir>/<sample_id>.npz  (expects key 'airs' or 'data')
    """

    def __init__(self, airs_dir: Path, sample_ids: Sequence[str]) -> None:
        self.airs_dir = Path(airs_dir)
        self.sample_ids = list(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.sample_ids[idx]
        npy = self.airs_dir / f"{sid}.npy"
        npz = self.airs_dir / f"{sid}.npz"

        blob: Dict[str, np.ndarray]
        if npy.exists():
            blob = _load_blob(npy)
            arr = blob["data"]
        elif npz.exists():
            blob = _load_blob(npz)
            arr = blob.get("airs", blob.get("data"))
            if arr is None:
                raise KeyError(f"NPZ missing 'airs' or 'data' key: {npz}")
        else:
            raise FileNotFoundError(f"No AIRS blob for sample_id={sid} in {self.airs_dir}")

        x_airs = _maybe_float32(arr)
        return {"sample_id": sid, "airs": x_airs}


class DualChannelDataset(Dataset[Dict[str, Any]]):
    """
    Combines FGS1 and AIRS channels per sample_id using two source directories.
    If labels are present in the manifest rows, they are attached as 'y'.
    """

    def __init__(
        self,
        fgs1_dir: Path,
        airs_dir: Path,
        manifest_rows: Sequence[Dict[str, str]],
        target_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.fgs1_dir = Path(fgs1_dir)
        self.airs_dir = Path(airs_dir)
        self.rows = list(manifest_rows)
        self.sample_ids = [r["sample_id"] for r in self.rows]

        # detect targets (regression) — robust to 'y' or 'target' or multi-head columns
        if target_columns is None:
            tc: List[str] = []
            if any("y" in r for r in self.rows):
                tc.append("y")
            if any("target" in r for r in self.rows):
                tc.append("target")
            self.target_columns = list(dict.fromkeys(tc)) or []
        else:
            self.target_columns = list(target_columns)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        sid = row["sample_id"]

        # Load FGS1
        npy = self.fgs1_dir / f"{sid}.npy"
        npz = self.fgs1_dir / f"{sid}.npz"
        if npy.exists():
            fa = _load_blob(npy)["data"]
        elif npz.exists():
            fb = _load_blob(npz)
            fa = fb.get("fgs1", fb.get("data"))
            if fa is None:
                raise KeyError(f"FGS1 NPZ missing 'fgs1' or 'data' for {sid}")
        else:
            raise FileNotFoundError(f"FGS1 blob not found for {sid}")

        # Load AIRS
        npy = self.airs_dir / f"{sid}.npy"
        npz = self.airs_dir / f"{sid}.npz"
        if npy.exists():
            aa = _load_blob(npy)["data"]
        elif npz.exists():
            ab = _load_blob(npz)
            aa = ab.get("airs", ab.get("data"))
            if aa is None:
                raise KeyError(f"AIRS NPZ missing 'airs' or 'data' for {sid}")
        else:
            raise FileNotFoundError(f"AIRS blob not found for {sid}")

        sample: Dict[str, Any] = {
            "sample_id": sid,
            "fgs1": _maybe_float32(fa),
            "airs": _maybe_float32(aa),
        }

        # Optional target(s)
        if self.target_columns:
            ys: List[float] = []
            has_any = False
            for col in self.target_columns:
                v = row.get(col, "")
                if v != "":
                    try:
                        ys.append(float(v))
                        has_any = True
                    except Exception:
                        ys.append(float("nan"))
                else:
                    ys.append(float("nan"))
            if has_any:
                y_tensor = torch.tensor(ys, dtype=torch.float32)
                sample["y"] = y_tensor if len(ys) > 1 else y_tensor.squeeze(-1)

        return sample


# ==============================================================================
# Lightning DataModule
# ==============================================================================


@dataclass(slots=True)
class _DMConfig:
    # paths
    root: Path
    fgs1_dir: Path
    airs_dir: Path
    # manifests (one of: a single CSV with split column, or explicit train/val/test CSVs)
    manifest_csv: Optional[Path] = None
    train_csv: Optional[Path] = None
    val_csv: Optional[Path] = None
    test_csv: Optional[Path] = None

    # split behavior if only a single CSV (ratio-based)
    train_ratio: float = 0.9
    val_ratio: float = 0.1  # test may be absent; often created externally
    seed: int = 42

    # loader knobs
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last: bool = False

    # targets
    target_columns: Optional[List[str]] = None


class SpectraDataModule:
    """
    PyTorch Lightning DataModule for dual-channel SpectraMind V50.
    - Hydra-friendly: pass Hydra DictConfig via **kwargs or construct explicitly.
    - Kaggle-safe defaults (low worker count; pin_memory on GPU).
    - Deterministic splits.
    """

    def __init__(self, **cfg: Any) -> None:
        if L is None:  # pragma: no cover
            raise RuntimeError(
                "lightning is not installed. Please install `lightning` (v2.x) "
                "or adapt the DataModule to raw PyTorch loaders."
            )

        # Normalize paths and defaults
        root = Path(cfg.get("paths", {}).get("root", cfg.get("root", ".")))
        fgs1_dir = Path(cfg.get("paths", {}).get("fgs1_dir", cfg.get("fgs1_dir", root / "data/fgs1")))
        airs_dir = Path(cfg.get("paths", {}).get("airs_dir", cfg.get("airs_dir", root / "data/airs")))

        manifest_csv = _opt_path(cfg.get("manifest_csv"))
        train_csv = _opt_path(cfg.get("train_csv"))
        val_csv = _opt_path(cfg.get("val_csv"))
        test_csv = _opt_path(cfg.get("test_csv"))

        self.conf = _DMConfig(
            root=root,
            fgs1_dir=fgs1_dir,
            airs_dir=airs_dir,
            manifest_csv=manifest_csv,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            train_ratio=float(cfg.get("train_ratio", 0.9)),
            val_ratio=float(cfg.get("val_ratio", 0.1)),
            seed=int(cfg.get("seed", 42)),
            batch_size=int(cfg.get("batch_size", 32)),
            num_workers=int(cfg.get("num_workers", 2 if _is_kaggle() else os.cpu_count() or 2)),
            pin_memory=bool(cfg.get("pin_memory", True)),
            persistent_workers=bool(cfg.get("persistent_workers", False)),
            drop_last=bool(cfg.get("drop_last", False)),
            target_columns=list(cfg.get("target_columns", [])) or None,
        )

        # Runtime handles
        self._ds_train: Optional[Dataset[Dict[str, Any]]] = None
        self._ds_val: Optional[Dataset[Dict[str, Any]]] = None
        self._ds_test: Optional[Dataset[Dict[str, Any]]] = None

        # Lightning parent
        class _PLDM(L.LightningDataModule):  # type: ignore[attr-defined]
            pass

        # Mix-in Lightning’s base to this instance for pl hooks without inheritance gymnastics
        self.__class__ = type(self.__class__.__name__, (self.__class__, _PLDM), {})

    # ------------------------- Lightning Hooks ---------------------------------

    def prepare_data(self) -> None:  # noqa: D401
        """
        No downloads here (Kaggle/CI must mount data). Optionally, verify directories.
        """
        for p in (self.conf.fgs1_dir, self.conf.airs_dir):
            if not p.exists():
                raise FileNotFoundError(f"Required data directory not found: {p}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._setup_fit()
        if stage in (None, "test", "validate", "predict"):
            self._setup_eval(stage)

    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        assert self._ds_train is not None, "call setup('fit') first"
        return DataLoader(
            self._ds_train,
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
            persistent_workers=self.conf.persistent_workers and self.conf.num_workers > 0,
            drop_last=self.conf.drop_last,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        assert self._ds_val is not None, "call setup('fit') first"
        return DataLoader(
            self._ds_val,
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
            persistent_workers=self.conf.persistent_workers and self.conf.num_workers > 0,
            drop_last=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Dict[str, Any]]:
        assert self._ds_test is not None, "call setup('test') first"
        return DataLoader(
            self._ds_test,
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
            persistent_workers=self.conf.persistent_workers and self.conf.num_workers > 0,
            drop_last=False,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self) -> DataLoader[Dict[str, Any]]:
        # Reuse test loader
        return self.test_dataloader()

    # ---------------------------- Private --------------------------------------

    def _setup_fit(self) -> None:
        rows = self._load_rows_for_fit()
        ds_full = DualChannelDataset(
            fgs1_dir=self.conf.fgs1_dir,
            airs_dir=self.conf.airs_dir,
            manifest_rows=rows,
            target_columns=self.conf.target_columns,
        )

        # deterministic split
        n = len(ds_full)
        n_train = int(round(n * self.conf.train_ratio))
        n_val = max(1, int(round(n * self.conf.val_ratio)))
        n_train = min(n - n_val, n_train)
        lengths = [n_train, n - n_train] if self.conf.val_ratio <= 0 else [n_train, n_val, n - n_train - n_val]
        subsets = _deterministic_split(ds_full, lengths, seed=self.conf.seed)

        self._ds_train = subsets[0]
        self._ds_val = subsets[1] if len(subsets) > 1 else subsets[0]

    def _setup_eval(self, stage: Optional[str]) -> None:
        # test/predict use explicit test_csv if provided; else fall back to val subset
        if self.conf.test_csv and self.conf.test_csv.exists():
            rows_test = _read_manifest(self.conf.test_csv)
            self._ds_test = DualChannelDataset(
                fgs1_dir=self.conf.fgs1_dir,
                airs_dir=self.conf.airs_dir,
                manifest_rows=rows_test,
                target_columns=self.conf.target_columns,
            )
        else:
            # If no explicit test set, reuse val
            self._ds_test = self._ds_val

    def _load_rows_for_fit(self) -> List[Dict[str, str]]:
        # If separate train/val manifests were provided, merge them with a split label
        if self.conf.train_csv:
            rows_train = _read_manifest(self.conf.train_csv)
            for r in rows_train:
                r.setdefault("split", "train")
            if self.conf.val_csv and self.conf.val_csv.exists():
                rows_val = _read_manifest(self.conf.val_csv)
                for r in rows_val:
                    r.setdefault("split", "val")
                return rows_train + rows_val
            return rows_train

        # Else use the single manifest and (optionally) filter by split for fit
        if self.conf.manifest_csv:
            rows = _read_manifest(self.conf.manifest_csv)
            # If there is an explicit split column, prefer rows labeled for fit
            has_split = any("split" in r for r in rows)
            if has_split:
                fit_rows = [r for r in rows if r.get("split", "").lower() in ("train", "val", "fit", "")]
                return fit_rows or rows
            return rows

        # Nothing found → instruct the user
        raise FileNotFoundError(
            "No training manifest found. Provide one of: "
            "`manifest_csv`, or `train_csv` (optionally with `val_csv`)."
        )

    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Minimal, safe collate:
          - Pads 1D/2D FGS1 sequences to max length in batch (left-aligned, zero pad)
          - Stacks AIRS spectra as tensors
          - Aggregates optional targets 'y'
        """
        sample_ids = [b["sample_id"] for b in batch]

        # FGS1 variable-length padding (assume [T] or [T, D])
        fgs1_list = [b["fgs1"] for b in batch]
        fgs1_shapes = [tuple(x.shape) for x in fgs1_list]
        max_len = max(s[0] for s in fgs1_shapes)
        fgs1_dims = 1 if all(len(s) == 1 for s in fgs1_shapes) else 2
        if fgs1_dims == 1:
            out_fgs1 = torch.zeros(len(batch), max_len, dtype=torch.float32)
            for i, x in enumerate(fgs1_list):
                t = x.shape[0]
                out_fgs1[i, :t] = x.to(torch.float32)
        else:
            feat_dim = max(s[1] if len(s) > 1 else 1 for s in fgs1_shapes)
            out_fgs1 = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
            for i, x in enumerate(fgs1_list):
                t = x.shape[0]
                d = x.shape[1] if x.ndim > 1 else 1
                out_fgs1[i, :t, :d] = x.to(torch.float32 if x.dtype.is_floating_point else torch.float32)

        # AIRS spectra stack (assume fixed length C, 1D or 2D)
        airs_list = [b["airs"] for b in batch]
        try:
            out_airs = torch.stack([_maybe_float32(x) for x in airs_list], dim=0)
        except Exception:
            # If shapes differ slightly, pad to max length
            lengths = [x.shape[0] for x in airs_list]
            C = max(lengths)
            out_airs = torch.zeros(len(batch), C, dtype=torch.float32)
            for i, x in enumerate(airs_list):
                c = x.shape[0]
                out_airs[i, :c] = _maybe_float32(x)

        result: Dict[str, Any] = {
            "sample_id": sample_ids,
            "fgs1": out_fgs1,
            "airs": out_airs,
        }

        # Optional targets
        if "y" in batch[0]:
            ys = [b.get("y") for b in batch]
            if all(y is not None and isinstance(y, torch.Tensor) and y.ndim == 0 for y in ys):
                result["y"] = torch.stack([y for y in ys if y is not None]).to(torch.float32)
            else:
                # ragged/multi-head → best effort stack with padding
                maxd = max(int(y.numel()) if isinstance(y, torch.Tensor) else 0 for y in ys)
                Y = torch.full((len(batch), maxd), float("nan"), dtype=torch.float32)
                for i, y in enumerate(ys):
                    if isinstance(y, torch.Tensor):
                        n = int(y.numel())
                        Y[i, :n] = y.reshape(-1).to(torch.float32)
                result["y"] = Y

        return result


# ==============================================================================
# Helpers
# ==============================================================================


def _opt_path(p: Any) -> Optional[Path]:
    if p in (None, "", False):
        return None
    return Path(str(p))
