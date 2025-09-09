# src/spectramind/models/__init__.py
# =============================================================================
# SpectraMind V50 — Model Package
# -----------------------------------------------------------------------------
# This module collects all model components (encoders, decoders, fusion,
# losses, and system builders) into a clean import surface. It aligns with
# SpectraMind’s dual-channel, physics-informed design:
#
#   • FGS1 Encoder  — photometric branch (Mamba SSM / CNN-lite time series)
#   • AIRS Encoder — spectroscopic branch (CNN/GNN spectral modeling)
#   • Fusion       — cross-attention dual encoder fusion (ADR-0004) [oai_citation:0‡ADR 0004 — Dual Encoder Fusion (FGS1 + AIRS).pdf](file-service://file-4CBsvxoriyyazqtkG3ekUJ)
#   • Decoder      — heteroscedastic μ/σ heads (Gaussian log-likelihood aligned)
#   • Losses       — composite physics-informed losses (ADR-0002) [oai_citation:1‡Composite Physics-Informed Loss (SpectraMind V50 ADR 0002).pdf](file-service://file-J6yjmT4fX3kDFt12DGCxrv)
#   • System       — wrapper to assemble encoders+fusion+decoder into a
#                    PyTorch LightningModule for training/export
#
# References:
#   - Repo Design: modular src/models/ for architectures & losses [oai_citation:2‡SpectraMind V50 – Ariel Challenge Solution Repository Design.pdf](file-service://file-K3MdDx2jYVx6rs8sZ6Wjd4) [oai_citation:3‡NeurIPS 2025 Ariel Data Challenge – SpectraMind V50 Repository Design.pdf](file-service://file-MBSfd7TrKDCPvsodX5J7TF)
#   - Kaggle Integration: models defined here are wrapped by Hydra configs
#     in configs/model/*.yaml [oai_citation:4‡Spectra Mind V50 – Full Repository Scaffold (winning Repo).pdf](file-service://file-RW8armTgHo1ZaifnQsckVp)
# =============================================================================

from .fgs1_encoder import FGS1Encoder
from .airs_encoder import AIRSEncoder
from .fusion_xattn import CrossAttentionFusion
from .decoder import SpectrumDecoder
from .losses import (
    gaussian_log_likelihood,
    smoothness_penalty,
    nonnegativity_penalty,
    band_coherence_penalty,
    calibration_penalty,
)
from .system import SpectraMindSystem

__all__ = [
    # Encoders
    "FGS1Encoder",
    "AIRSEncoder",

    # Fusion
    "CrossAttentionFusion",

    # Decoder
    "SpectrumDecoder",

    # Losses
    "gaussian_log_likelihood",
    "smoothness_penalty",
    "nonnegativity_penalty",
    "band_coherence_penalty",
    "calibration_penalty",

    # System wrapper
    "SpectraMindSystem",
]
