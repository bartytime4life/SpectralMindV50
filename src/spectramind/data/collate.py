# src/spectramind/data/collate.py
from __future__ import annotations

"""
SpectraMind V50 — Data Collation Utilities
==========================================

This module provides functions to collate batches of FGS1 + AIRS telescope
observations into model-ready tensors. It is designed for use in PyTorch
DataLoaders and adheres to NASA-grade reproducibility and Kaggle-safe
constraints.

Key Principles
--------------
- FGS1: single-channel photometric time series (1D).
- AIRS: multi-channel spectroscopy time series (2D: time × channels).
- Output: standardized dictionary with torch tensors and optional metadata.
- Safe defaults: deterministic padding, float32 precision, Hydra/DVC-friendly.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Tuple


__all__ = ["collate_batch"]


def _pad_and_stack(
    seqs: List[torch.Tensor], pad_value: float = 0.0, batch_first: bool = True
) -> torch.Tensor:
    """
    Pad variable-length sequences to the same length and stack into a batch.

    Args:
        seqs: List of [T, ...] tensors (FGS1 or AIRS time-series).
        pad_value: Fill value for padding.
        batch_first: If True, output shape = [B, T, ...]; else [T, B, ...].

    Returns:
        Padded tensor with uniform time dimension.
    """
    if not seqs:
        raise ValueError("No sequences provided to pad_and_stack().")
    return pad_sequence(seqs, batch_first=batch_first, padding_value=pad_value)


def collate_batch(
    batch: List[Dict[str, Any]],
    pad_value: float = 0.0,
    device: str | torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader — merges a list of samples into a batch.

    Each sample is expected to be a dict with keys:
        - "fgs1": torch.Tensor [T]  (photometric time series)
        - "airs": torch.Tensor [T, C] (spectrometer time × channels)
        - "target": torch.Tensor [C]  (optional, ground truth spectrum)
        - "meta": dict (optional metadata, not collated into tensors)

    Args:
        batch: List of sample dicts.
        pad_value: Fill value for time padding.
        device: Optional torch device to move tensors onto.

    Returns:
        Dict with batched tensors:
            {
              "fgs1": [B, T_max],
              "airs": [B, T_max, C],
              "target": [B, C] (if present),
              "mask": [B, T_max] (1=valid, 0=padded)
            }
    """
    fgs1_seqs, airs_seqs, targets = [], [], []
    for sample in batch:
        fgs1 = sample.get("fgs1")
        airs = sample.get("airs")
        tgt = sample.get("target")

        if fgs1 is None or airs is None:
            raise KeyError("Each sample must contain 'fgs1' and 'airs' tensors.")

        fgs1_seqs.append(fgs1.float())
        airs_seqs.append(airs.float())
        if tgt is not None:
            targets.append(tgt.float())

    # Pad + stack
    fgs1_batch = _pad_and_stack(fgs1_seqs, pad_value=pad_value)  # [B, T]
    airs_batch = _pad_and_stack(airs_seqs, pad_value=pad_value)  # [B, T, C]

    # Build mask (1 for valid timesteps, 0 for padded)
    lengths = torch.tensor([seq.shape[0] for seq in fgs1_seqs], dtype=torch.long)
    max_len = fgs1_batch.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.to(dtype=torch.float32)

    out: Dict[str, torch.Tensor] = {
        "fgs1": fgs1_batch,
        "airs": airs_batch,
        "mask": mask,
    }

    if targets:
        out["target"] = torch.stack(targets, dim=0)

    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}

    return out
