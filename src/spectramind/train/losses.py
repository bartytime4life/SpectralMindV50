from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch, torch.nn.functional as F

@dataclass
class LossConfig:
    w_gll: float = 1.0
    w_tv: float = 0.0
    w_curv: float = 0.0
    w_nonneg: float = 0.0
    w_calib: float = 0.0
    tv_eps: float = 1e-6

def gaussian_log_likelihood(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Stable heteroscedastic Gaussian NLL per sample (sum over bins)
    sigma = torch.clamp(sigma, min=1e-8)
    z = (target - mu) / sigma
    return 0.5 * (z * z + 2.0 * torch.log(sigma) + torch.log(torch.tensor(2.0 * 3.141592653589793, device=sigma.device))).sum(dim=-1)

def tv_penalty(mu: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # total variation along spectral bin axis (assume last dim is bins)
    d = torch.diff(mu, dim=-1)
    return torch.sqrt(d * d + eps).sum(dim=-1)

def curvature_penalty(mu: torch.Tensor) -> torch.Tensor:
    d2 = torch.diff(mu, n=2, dim=-1)
    return (d2 * d2).sum(dim=-1)

def nonneg_penalty(mu: torch.Tensor) -> torch.Tensor:
    return F.relu(-mu).sum(dim=-1)

def build_composite_loss(cfg: LossConfig):
    def loss_fn(pred: Tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        mu, sigma = pred  # [B, BINS], [B, BINS]
        terms = {}
        gll = gaussian_log_likelihood(mu, sigma, target).mean()
        terms["gll"] = float(gll.detach().cpu())
        total = cfg.w_gll * gll

        if cfg.w_tv:
            tv = tv_penalty(mu, eps=cfg.tv_eps).mean()
            total = total + cfg.w_tv * tv; terms["tv"] = float(tv.detach().cpu())
        if cfg.w_curv:
            curv = curvature_penalty(mu).mean()
            total = total + cfg.w_curv * curv; terms["curv"] = float(curv.detach().cpu())
        if cfg.w_nonneg:
            nn = nonneg_penalty(mu).mean()
            total = total + cfg.w_nonneg * nn; terms["nonneg"] = float(nn.detach().cpu())
        # cfg.w_calib reserved for coverage/ECE-style penalties if targets are available

        return total, terms
    return loss_fn
