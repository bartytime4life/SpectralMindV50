# src/spectramind/models/fusion/__init__.py
# =============================================================================
# SpectraMind V50 — Fusion Modules
# -----------------------------------------------------------------------------
# Fusion blocks combine modality-specific encoders (FGS1 photometry and AIRS
# spectroscopy) into a unified latent representation. ADR-0004 establishes
# cross-attention fusion as the baseline method, with slots for alternative
# strategies (concat, late fusion, gating) if needed for ablations.
#
# Key references:
#   • ADR-0004 — Dual Encoder Fusion (FGS1 + AIRS) [oai_citation:0‡ADR 0004 — Dual Encoder Fusion (FGS1 + AIRS).pdf](file-service://file-4CBsvxoriyyazqtkG3ekUJ)
#   • Repo design & model docs [oai_citation:1‡SpectraMind V50 – Ariel Challenge Solution Repository Design.pdf](file-service://file-K3MdDx2jYVx6rs8sZ6Wjd4) [oai_citation:2‡NeurIPS 2025 Ariel Data Challenge – SpectraMind V50 Repository Design.pdf](file-service://file-MBSfd7TrKDCPvsodX5J7TF)
#   • Physics-aware dual-channel integration [oai_citation:3‡SpectraMind V50: NeurIPS 2025 Ariel Data Challenge Solution Documentation.pdf](file-service://file-V6bB4YZsySBoh5PXGeDi7S)
# =============================================================================

from .xattn import CrossAttentionFusion

__all__ = [
    # Baseline fusion (ADR-0004)
    "CrossAttentionFusion",
]
