# src/spectramind/pipeline/__init__.py
from __future__ import annotations

# Re-export entrypoints for convenience: `from spectramind.pipeline import train`
from .train import run as train      # noqa: F401
from .calibrate import run as calibrate  # noqa: F401
from .predict import run as predict  # noqa: F401
from .diagnostics import run as diagnostics  # noqa: F401
from .submit import run as submit    # noqa: F401

__all__ = ["train", "calibrate", "predict", "diagnostics", "submit"]
