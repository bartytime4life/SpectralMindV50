
# src/spectramind/train/datasets.py
# =============================================================================
# SpectraMind V50 — Dataset Implementations (FGS1 + AIRS)
# -----------------------------------------------------------------------------
# A small family of PyTorch Dataset helpers that feed the dual-channel pipeline:
#   • ArielPairDataset          — abstract conveniences shared by all datasets
#   • NPZDirectoryDataset       — scans a directory of .npz files (fast path / Kaggle)
#   • IndexedNPZDataset         — drives samples from a CSV/TSV index (paths + labels)
#   • H5Dataset (optional)      — loads from a single HDF5 file with groups per sample
#
# All datasets emit a dict with (at minimum):
#   {
#     "airs": Tensor[...],     # spectrometer time-series / stacks
#     "fgs1": Tensor[...],     # photometric time-series
#     # optional targets (if supervised):
#     "mu": Tensor[283 or 284],
#     "sigma": Tensor[283 or 284],
#     "y": Tensor[283 or 284],
#     # optional metadata:
#     "id": str, "index": int, ...
#   }
#
# Normalization / slicing / channel selection are supported via kwargs:
#   normalize={"airs": {"mean": ..., "std": ...}, "fgs1": {...}}
#   select={"airs_channels": [int, ...], "mu_bins": [int, ...], "sigma_bins": [int, ...]}
#
# NOTE: Prefer Hydra `_target_` instantiation in configs; these classes are designed
#       to be passed to the ArielDataModule (or instantiated directly in configs).
# =============================================================================

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

# Guarded Torch import (dataset API)
try:  # pragma: no cover
    import torch
    from torch.utils.data import Dataset
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    _TORCH_IMPORT_ERROR = _e
else:
    _TORCH_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "torch is required to use spectramind.train.datasets.*"
        ) from _TORCH_IMPORT_ERROR


def _to_tensor(x: Any, dtype: Optional[torch.dtype] = torch.float32) -> "torch.Tensor":
    t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.contiguous()


class np_load:
    """
    Safe context manager for numpy.load (allow_pickle=False).
    """
    def __init__(self, path: Union[str, Path]) -> None:
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


def _maybe_select_1d(arr: "torch.Tensor", idxs: Optional[Sequence[int]]) -> "torch.Tensor":
    if idxs is None:
        return arr
    return arr.index_select(dim=-1, index=torch.as_tensor(list(idxs), dtype=torch.long, device=arr.device))


def _apply_norm(x: "torch.Tensor", mean: Optional[Union[float, Sequence[float]]] = None,
                std: Optional[Union[float, Sequence[float]]] = None) -> "torch.Tensor":
    if mean is None and std is None:
        return x
    if isinstance(mean, Sequence):
        mean = _to_tensor(mean, dtype=x.dtype).to(x.device)
    if isinstance(std, Sequence):
        std = _to_tensor(std, dtype=x.dtype).to(x.device)
    if mean is not None:
        x = x - mean
    if std is not None:
        x = x / (std + 1e-8)
    return x


# -----------------------------------------------------------------------------
# Base: ArielPairDataset
# -----------------------------------------------------------------------------

@dataclass
class NormalizeSpec:
    mean: Optional[Union[float, Sequence[float]]] = None
    std: Optional[Union[float, Sequence[float]]] = None


@dataclass
class SelectSpec:
    airs_channels: Optional[Sequence[int]] = None   # subset AIRS spectral axis (if last dim is channels)
    mu_bins: Optional[Sequence[int]] = None         # subset mu
    sigma_bins: Optional[Sequence[int]] = None      # subset sigma


class ArielPairDataset(Dataset):  # type: ignore
    """
    Abstract base providing convenience hooks for normalization, selection, and key validation.
    Concrete subclasses must implement:
      - __len__(self) -> int
      - _load_raw(self, idx) -> Dict[str, Any]  # returns raw numpy-like arrays
    """
    REQUIRED_INPUT_KEYS: Tuple[str, str] = ("airs", "fgs1")

    def __init__(
        self,
        normalize: Optional[Mapping[str, NormalizeSpec]] = None,
        select: Optional[SelectSpec] = None,
        require_targets: bool = True,
        to_float_dtype: "torch.dtype" = torch.float32,
    ) -> None:
        _require_torch()
        super().__init__()
        self.normalize = normalize or {}
        self.select = select or SelectSpec()
        self.require_targets = require_targets
        self.to_float_dtype = to_float_dtype

    # Subclasses implement
    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def _load_raw(self, idx: int) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self._load_raw(idx)

        # Validate required keys
        for k in self.REQUIRED_INPUT_KEYS:
            if k not in raw:
                raise KeyError(f"Sample #{idx} missing required key '{k}'")

        # Convert to tensors
        sample: Dict[str, Any] = {}
        for k, v in raw.items():
            if k in ("id", "path", "meta"):  # leave metadata as-is
                sample[k] = v
                continue
            # auto dtype float for tensors; allow ints to stay ints for indices in meta
            sample[k] = _to_tensor(v, dtype=self.to_float_dtype if k not in ("index",) else None)

        # Apply feature selection (channels / bins)
        if "airs" in sample and self.select.airs_channels is not None:
            sample["airs"] = _maybe_select_1d(sample["airs"], self.select.airs_channels)

        if "mu" in sample and self.select.mu_bins is not None:
            sample["mu"] = _maybe_select_1d(sample["mu"], self.select.mu_bins)

        if "sigma" in sample and self.select.sigma_bins is not None:
            sample["sigma"] = _maybe_select_1d(sample["sigma"], self.select.sigma_bins)

        # Apply normalization
        if "airs" in sample and "airs" in self.normalize:
            ns = self.normalize["airs"]
            sample["airs"] = _apply_norm(sample["airs"], ns.mean, ns.std)
        if "fgs1" in sample and "fgs1" in self.normalize:
            ns = self.normalize["fgs1"]
            sample["fgs1"] = _apply_norm(sample["fgs1"], ns.mean, ns.std)

        # Enforce target presence if required
        if self.require_targets and ("mu" not in sample or "sigma" not in sample):
            raise KeyError(f"Targets 'mu'/'sigma' required but missing for sample #{idx}")

        # Best-effort ensure contiguous memory (collate friendliness)
        for k, v in list(sample.items()):
            if isinstance(v, torch.Tensor):
                sample[k] = v.contiguous()

        # Attach index (useful for diagnostics)
        sample.setdefault("index", idx)
        return sample


# -----------------------------------------------------------------------------
# NPZDirectoryDataset
# -----------------------------------------------------------------------------

class NPZDirectoryDataset(ArielPairDataset):
    """
    Scans a directory for .npz files. Expected NPZ keys per file:
      - 'airs'  (required)
      - 'fgs1'  (required)
      - optional targets: 'mu', 'sigma', 'y'
      - optional 'id'     (string identifier)
    """
    def __init__(
        self,
        root: Union[str, Path],
        normalize: Optional[Mapping[str, NormalizeSpec]] = None,
        select: Optional[SelectSpec] = None,
        require_targets: bool = True,
        to_float_dtype: "torch.dtype" = torch.float32,
        sort: bool = True,
        glob_pattern: str = "*.npz",
    ) -> None:
        super().__init__(normalize=normalize, select=select, require_targets=require_targets, to_float_dtype=to_float_dtype)
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"NPZ root not found: {self.root}")
        files = list(self.root.glob(glob_pattern))
        if sort:
            files = sorted(files)
        if not files:
            raise RuntimeError(f"No NPZ files found under {self.root} (pattern={glob_pattern})")
        self.files: List[Path] = files

    def __len__(self) -> int:
        return len(self.files)

    def _load_raw(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        with np_load(path) as npz:
            raw: Dict[str, Any] = {}
            # required
            if "airs" not in npz or "fgs1" not in npz:
                raise KeyError(f"File missing 'airs'/'fgs1': {path}")
            raw["airs"] = npz["airs"]
            raw["fgs1"] = npz["fgs1"]
            # optional targets
            for key in ("mu", "sigma", "y"):
                if key in npz:
                    raw[key] = npz[key]
            # metadata
            raw["path"] = str(path)
            raw["id"] = str(path.stem)
        return raw


# -----------------------------------------------------------------------------
# IndexedNPZDataset
# -----------------------------------------------------------------------------

class IndexedNPZDataset(ArielPairDataset):
    """
    CSV/TSV-driven dataset for .npz samples. The index file contains rows with at least:
       path[, id, mu_path, sigma_path, y_path]
    The `path` entries are resolved relative to `base_dir` (or absolute if so provided).

    This allows mixing different folders or file names and overriding target locations.
    """
    def __init__(
        self,
        index_file: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        delimiter: Optional[str] = None,            # auto by extension if None
        has_header: bool = True,
        normalize: Optional[Mapping[str, NormalizeSpec]] = None,
        select: Optional[SelectSpec] = None,
        require_targets: bool = True,
        to_float_dtype: "torch.dtype" = torch.float32,
    ) -> None:
        super().__init__(normalize=normalize, select=select, require_targets=require_targets, to_float_dtype=to_float_dtype)
        self.index_file = Path(index_file)
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file}")
        self.base_dir = Path(base_dir) if base_dir is not None else None

        if delimiter is None:
            if self.index_file.suffix.lower() in (".tsv",):
                delimiter = "\t"
            else:
                delimiter = ","

        # parse index
        self.rows: List[Dict[str, str]] = []
        with self.index_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter) if has_header else csv.reader(f, delimiter=delimiter)
            if has_header:
                for row in reader:  # type: ignore
                    self.rows.append({k: (v or "").strip() for k, v in row.items()})
            else:
                # manual field names
                for r in reader:  # type: ignore
                    # Expect: path[, id, mu_path, sigma_path, y_path]
                    row = {
                        "path": (r[0] if len(r) > 0 else "").strip(),
                        "id": (r[1] if len(r) > 1 else "").strip(),
                        "mu_path": (r[2] if len(r) > 2 else "").strip(),
                        "sigma_path": (r[3] if len(r) > 3 else "").strip(),
                        "y_path": (r[4] if len(r) > 4 else "").strip(),
                    }
                    self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"No rows parsed from index: {self.index_file}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute() and self.base_dir is not None:
            path = self.base_dir / path
        return path

    def _load_raw(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        npz_path = self._resolve(row["path"])
        if not npz_path.exists():
            raise FileNotFoundError(f"Sample npz missing: {npz_path}")

        out: Dict[str, Any] = {}
        with np_load(npz_path) as npz:
            if "airs" not in npz or "fgs1" not in npz:
                raise KeyError(f"File missing 'airs'/'fgs1': {npz_path}")
            out["airs"] = npz["airs"]
            out["fgs1"] = npz["fgs1"]
            # Optionally, take targets from NPZ itself
            for k in ("mu", "sigma", "y"):
                if k in npz:
                    out[k] = npz[k]

        # Allow overriding targets by separate paths in the index
        for key, col in (("mu", "mu_path"), ("sigma", "sigma_path"), ("y", "y_path")):
            p = row.get(col, "") if row else ""
            if p:
                tpath = self._resolve(p)
                with np_load(tpath) as tnpz:
                    if key not in tnpz:
                        # If the separate file contains the array directly, look for "arr_0"
                        if "arr_0" in tnpz:
                            out[key] = tnpz["arr_0"]
                        else:
                            raise KeyError(f"{tpath} does not contain '{key}' or 'arr_0'")
                    else:
                        out[key] = tnpz[key]

        out["id"] = (row.get("id") or npz_path.stem)
        out["path"] = str(npz_path)
        return out


# -----------------------------------------------------------------------------
# H5Dataset (optional)
# -----------------------------------------------------------------------------

class H5Dataset(ArielPairDataset):
    """
    Loads all samples from a single HDF5 file. Expects a layout such as:

      /samples/<sid>/airs   -> np.ndarray
      /samples/<sid>/fgs1   -> np.ndarray
      /samples/<sid>/mu     -> (optional) np.ndarray
      /samples/<sid>/sigma  -> (optional) np.ndarray
      /samples/<sid>/y      -> (optional) np.ndarray

    If `sample_ids` is None, attempts to enumerate groups under /samples.

    NOTE: Requires `h5py`.
    """
    def __init__(
        self,
        h5_path: Union[str, Path],
        sample_ids: Optional[Sequence[str]] = None,
        normalize: Optional[Mapping[str, NormalizeSpec]] = None,
        select: Optional[SelectSpec] = None,
        require_targets: bool = True,
        to_float_dtype: "torch.dtype" = torch.float32,
        group_prefix: str = "/samples",
    ) -> None:
        super().__init__(normalize=normalize, select=select, require_targets=require_targets, to_float_dtype=to_float_dtype)
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        self.group_prefix = group_prefix.rstrip("/")
        self._ids = list(sample_ids) if sample_ids is not None else self._discover_ids()

    def _discover_ids(self) -> List[str]:
        try:
            import h5py  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("H5Dataset requires `h5py` installed.") from e
        sid: List[str] = []
        with h5py.File(self.h5_path, "r") as f:
            grp = f.get(self.group_prefix, None)
            if grp is None:
                raise KeyError(f"Missing group '{self.group_prefix}' in {self.h5_path}")
            for k in grp.keys():
                sid.append(str(k))
        if not sid:
            raise RuntimeError(f"No sample groups under {self.group_prefix} in {self.h5_path}")
        return sid

    def __len__(self) -> int:
        return len(self._ids)

    def _load_raw(self, idx: int) -> Dict[str, Any]:
        try:
            import h5py  # type: ignore
            import numpy as np  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError("H5Dataset requires `h5py` installed.") from e

        sid = self._ids[idx]
        gpath = f"{self.group_prefix}/{sid}"
        out: Dict[str, Any] = {"id": sid, "path": f"{self.h5_path}:{gpath}"}

        with h5py.File(self.h5_path, "r") as f:
            grp = f.get(gpath, None)
            if grp is None:
                raise KeyError(f"Missing group '{gpath}' in {self.h5_path}")

            def _req(name: str):
                if name not in grp:
                    raise KeyError(f"'{name}' missing in {gpath}")
                return grp[name][()]

            out["airs"] = _req("airs")
            out["fgs1"] = _req("fgs1")

            for k in ("mu", "sigma", "y"):
                if k in grp:
                    out[k] = grp[k][()]

        return out


__all__ = [
    "ArielPairDataset",
    "NPZDirectoryDataset",
    "IndexedNPZDataset",
    "H5Dataset",
    "NormalizeSpec",
    "SelectSpec",
]