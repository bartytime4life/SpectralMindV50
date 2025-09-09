# src/spectramind/models/heads/__init__.py
# =============================================================================
# SpectraMind V50 — Model Heads
# -----------------------------------------------------------------------------
# Output modules (a.k.a. "heads") that map fused latent representations into
# physically-informed predictions. All heads follow a PyTorch nn.Module API.
#
# Current components:
#   • HeteroscedasticHead — joint μ/σ prediction for 283-bin spectra [oai_citation:0‡SpectraMind V50: NeurIPS 2025 Ariel Data Challenge Solution Documentation.pdf](file-service://file-V6bB4YZsySBoh5PXGeDi7S)
#   • (future) CalibHead   — output re-scaling or bias correction
#   • (future) AuxHeads    — e.g., transit timing or diagnostic regressors
#
# References:
#   - Repository Design docs [oai_citation:1‡SpectraMind V50 – Ariel Challenge Solution Repository Design.pdf](file-service://file-K3MdDx2jYVx6rs8sZ6Wjd4) [oai_citation:2‡NeurIPS 2025 Ariel Data Challenge – SpectraMind V50 Repository Design.pdf](file-service://file-MBSfd7TrKDCPvsodX5J7TF)
#   - ADR-0002: Physics-Informed Losses [oai_citation:3‡Composite Physics-Informed Loss (SpectraMind V50 ADR 0002).pdf](file-service://file-J6yjmT4fX3kDFt12DGCxrv)
#   - ADR-0004: Dual Encoder Fusion [oai_citation:4‡ADR 0004 — Dual Encoder Fusion (FGS1 + AIRS).pdf](file-service://file-4CBsvxoriyyazqtkG3ekUJ)
# =============================================================================

from .hetero import HeteroscedasticHead

__all__ = [
    "HeteroscedasticHead",
]
