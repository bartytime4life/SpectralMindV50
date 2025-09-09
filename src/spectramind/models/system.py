# src/spectramind/models/system.py
from __future__ import annotations

"""
SpectraMind V50 — Lightning System
==================================

Dual-encoder fusion (FGS1 photometry + AIRS spectroscopy) with a
heteroscedastic head producing μ and σ for n_bins spectral channels.

Batch contract (training / validation):
  batch = {
    "fgs1":   FloatTensor [B, T] or [B, T, 1],
    "airs":   FloatTensor [B, C, T],
    "target": FloatTensor [B, n_bins]  # optional at inference
  }

Forward output:
  {"mu": FloatTensor [B, n_bins], "sigma": FloatTensor [B, n_bins]}
"""

from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .encoders.fgs1 import FGS1Encoder
from .encoders.airs import AIRSEncoder
from .fusion.xattn import CrossAttentionFuse
from .heads.hetero import HeteroHead


class SpectraSystem(pl.LightningModule):
    """
    Wiring:
      FGS1Encoder → proj_fgs1 ┐
                               ├─ CrossAttentionFuse → pool → HeteroHead → μ, σ
      AIRSEncoder → proj_airs ┘

    Notes
    -----
    • σ is floored in the head for numeric stability
    • First-bin GLL boost (`fgs1_boost`) reflects challenge weighting of white-light channel
    • Optional smoothness & nonnegativity penalties for physics-consistency
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        loss_cfg: Optional[Dict[str, Any]] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        metrics_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # Save minimal hparams; avoid dumping whole cfgs (can be large)
        self.save_hyperparameters(
            {
                "n_bins": int(model_cfg.get("n_bins", 283)),
                "d_model": int(model_cfg.get("d_model", 256)),
                "d_fgs1": int(model_cfg.get("d_fgs1", 128)),
                "d_airs": int(model_cfg.get("d_airs", 128)),
                "n_heads": int(model_cfg.get("n_heads", 4)),
                "dropout": float(model_cfg.get("dropout", 0.1)),
            }
        )

        n_bins = self.hparams["n_bins"]
        d_model = self.hparams["d_model"]
        d_fgs1 = self.hparams["d_fgs1"]
        d_airs = self.hparams["d_airs"]
        n_heads = self.hparams["n_heads"]
        dropout = self.hparams["dropout"]

        # Encoders
        self.enc_fgs1 = FGS1Encoder(
            d_in=int(model_cfg.get("fgs1_d_in", 1)),
            d_model=d_fgs1,
            depth=int(model_cfg.get("fgs1_depth", 3)),
            dropout=dropout,
        )
        self.enc_airs = AIRSEncoder(
            chans_in=int(model_cfg.get("airs_chans_in", 1)),
            d_model=d_airs,
            time_kernel=int(model_cfg.get("airs_time_kernel", 7)),
            depth=int(model_cfg.get("airs_depth", 3)),
            dropout=dropout,
        )

        # Project to common latent dimension
        self.proj_fgs1 = nn.Linear(d_fgs1, d_model)
        self.proj_airs = nn.Linear(d_airs, d_model)

        # Cross-attention fusion
        self.fuse = CrossAttentionFuse(d_model=d_model, n_heads=n_heads, dropout=dropout)

        # Heteroscedastic decoder
        self.head = HeteroHead(
            d_model=d_model,
            n_bins=n_bins,
            dropout=dropout,
            sigma_floor=float(model_cfg.get("sigma_floor", 1e-9)),
        )

        # Loss / optimization configs
        self.loss_cfg = dict(loss_cfg or {})
        self.optimizer_cfg = dict(optimizer_cfg or {})
        self.scheduler_cfg = dict(scheduler_cfg or {})
        self.metrics_cfg = dict(metrics_cfg or {})

        # Composite loss weights
        self.gll_weight = float(self.loss_cfg.get("gll_weight", 1.0))
        self.fgs1_boost = float(self.loss_cfg.get("fgs1_boost", 58.0))  # upweight μ_000 (white-light)
        self.smooth_weight = float(self.loss_cfg.get("smooth_weight", 0.0))
        self.nonneg_weight = float(self.loss_cfg.get("nonneg_weight", 0.0))

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward path for inference/training. Expects "fgs1" and "airs" in batch.
        Returns dict with "mu" and "sigma" (both [B, n_bins]).
        """
        fgs1 = batch["fgs1"].float()  # [B, T] | [B, T, 1]
        airs = batch["airs"].float()  # [B, C, T]

        # normalize shapes
        if fgs1.ndim == 2:  # [B, T] → [B, T, 1]
            fgs1 = fgs1.unsqueeze(-1)
        if airs.ndim == 2:  # [B, T] → [B, 1, T]
            airs = airs.unsqueeze(1)

        # Encode → [B, Lx, d_x]
        ef = self.enc_fgs1(fgs1)     # [B, Lf, d_fgs1]
        ea = self.enc_airs(airs)     # [B, La, d_airs]

        # Project to common latent dimension
        ef = self.proj_fgs1(ef)      # [B, Lf, d_model]
        ea = self.proj_airs(ea)      # [B, La, d_model]

        # Cross-attention (AIRs queries over FGS1)
        z = self.fuse(queries=ea, keys=ef, values=ef)  # [B, La, d_model]

        # Pool temporal tokens
        z = z.mean(dim=1)  # [B, d_model]

        # Decode
        mu, sigma = self.head(z)     # [B, n_bins], [B, n_bins]
        return {"mu": mu, "sigma": sigma}

    # ---------------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------------
    def _gll(self, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Gaussian negative log-likelihood (per-bin), averaged across batch, with optional
        boost of the white-light channel (bin 0).
        """
        eps = 1e-12
        var = sigma.square().clamp_min(eps)  # numeric safety
        nll = 0.5 * (var.log() + (target - mu).square() / var)  # [B, n_bins]
        if nll.shape[1] > 0 and self.fgs1_boost > 1.0:
            nll[:, 0] = nll[:, 0] * self.fgs1_boost
        return nll.mean()

    def _phys_penalties(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Optional physics-aware penalties:
          • Smoothness: L2 of second difference across bins
          • Soft non-negativity: L2 penalty on negative parts
        """
        total = torch.zeros((), device=mu.device, dtype=mu.dtype)
        # Smoothness
        if self.smooth_weight > 0 and mu.shape[1] > 2:
            d2 = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]
            total = total + self.smooth_weight * d2.square().mean()
        # Non-negativity
        if self.nonneg_weight > 0:
            neg = (-mu).clamp_min(0)
            total = total + self.nonneg_weight * neg.square().mean()
        return total

    def _compute_loss(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        mu, sigma = out["mu"], out["sigma"]
        target = batch.get("target", None)
        if target is None:  # inference path
            return mu.sum() * 0.0
        target = target.float()
        gll = self._gll(mu, sigma, target)
        phys = self._phys_penalties(mu)
        return self.gll_weight * gll + phys

    # ---------------------------------------------------------------------
    # Lightning hooks
    # ---------------------------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], _: int) -> torch.Tensor:
        out = self.forward(batch)
        loss = self._compute_loss(out, batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], _: int) -> None:
        out = self.forward(batch)
        loss = self._compute_loss(out, batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def test_step(self, batch: Dict[str, torch.Tensor], _: int) -> None:
        out = self.forward(batch)
        loss = self._compute_loss(out, batch)
        self.log("test/loss", loss, on_epoch=True)

    # ---------------------------------------------------------------------
    # Optimizer setup
    # ---------------------------------------------------------------------
    def configure_optimizers(self):
        lr = float(self.optimizer_cfg.get("lr", 2e-4))
        wd = float(self.optimizer_cfg.get("weight_decay", 0.0))
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        if not self.scheduler_cfg:
            return opt

        name = str(self.scheduler_cfg.get("name", "none")).lower()
        if name == "cosine":
            tmax = int(self.scheduler_cfg.get("tmax", 100))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax)
            return {"optimizer": opt, "lr_scheduler": sched}

        if name == "step":
            step_size = int(self.scheduler_cfg.get("step_size", 10))
            gamma = float(self.scheduler_cfg.get("gamma", 0.1))
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
            return {"optimizer": opt, "lr_scheduler": sched}

        return opt
