# src/spectramind/models/encoders/__init__.py
# =============================================================================
# SpectraMind V50 — Encoders
# -----------------------------------------------------------------------------
# Modality-specific encoders that transform raw/tokenized inputs into latent
# representations:
#   • FGS1Encoder — photometric (white-light) time-series branch
#   • AIRSEncoder — spectroscopic (283-channel) branch
#
# Notes
# -----
# • We support both filename styles to keep imports resilient during refactors:
#     - fgs1_encoder.py / airs_encoder.py
#     - fgs1.py        / airs.py
# • A tiny registry is provided for Hydra/CLI factory-style instantiation.
# =============================================================================

from __future__ import annotations

from typing import Dict, Type

# --- Robust imports for encoder classes --------------------------------------
FGS1Encoder = None  # type: ignore[assignment]
AIRSEncoder = None  # type: ignore[assignment]

# Try common filenames first (fgs1_encoder.py / airs_encoder.py)
try:  # pragma: no cover - import plumbing
    from .fgs1_encoder import FGS1Encoder as _FGS1Encoder  # type: ignore
    FGS1Encoder = _FGS1Encoder
except Exception:  # pragma: no cover
    try:
        from .fgs1 import FGS1Encoder as _FGS1Encoder  # type: ignore
        FGS1Encoder = _FGS1Encoder
    except Exception:
        pass

try:  # pragma: no cover
    from .airs_encoder import AIRSEncoder as _AIRSEncoder  # type: ignore
    AIRSEncoder = _AIRSEncoder
except Exception:  # pragma: no cover
    try:
        from .airs import AIRSEncoder as _AIRSEncoder  # type: ignore
        AIRSEncoder = _AIRSEncoder
    except Exception:
        pass

# --- Public exports -----------------------------------------------------------
__all__ = [
    name
    for name, obj in (
        ("FGS1Encoder", FGS1Encoder),
        ("AIRSEncoder", AIRSEncoder),
    )
    if obj is not None
]

# --- Minimal registry for factories/Hydra ------------------------------------
ENCODER_REGISTRY: Dict[str, Type] = {}
if FGS1Encoder is not None:
    ENCODER_REGISTRY["fgs1"] = FGS1Encoder
if AIRSEncoder is not None:
    ENCODER_REGISTRY["airs"] = AIRSEncoder
