# src/spectramind/train/__init__.py
# =============================================================================
# SpectraMind V50 — Training Package Init
# -----------------------------------------------------------------------------
# Exposes the primary training entrypoint (`train_from_config`) and registry
# utilities for model, loss, optimizer, and scheduler builders. This ensures a
# clean, discoverable API surface for external modules, CLI commands, and
# Kaggle notebooks.
#
# Design notes:
#   • Registry-based builders keep experiment configs declarative (Hydra-safe).
#   • Only the stable public API is exported via __all__.
#   • Internal helpers remain private to this package.
#   • Compatible with NASA-grade reproducibility standards (MCP).
# =============================================================================

from .trainer import train_from_config
from .registry import (
    get_model_builder,
    get_loss_builder,
    get_optimizer_builder,
    get_scheduler_builder,
)

__all__: list[str] = [
    "train_from_config",
    "get_model_builder",
    "get_loss_builder",
    "get_optimizer_builder",
    "get_scheduler_builder",
]