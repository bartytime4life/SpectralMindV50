from __future__ import annotations
from typing import Any, Dict, Optional
try:
    import pytorch_lightning as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # soft optional

from .metrics import MetricsAggregator, CSVWriter
from .jsonl import JSONLLogger

class LightningLoggerAdapter:
    """
    Minimal adapter to capture Lightning metrics into local CSV/JSONL.
    Use when you don't want tensorboard/W&B. Works with .log_dict / manual calls.
    """
    def __init__(self, jsonl: JSONLLogger, csv_path: Optional[str] = None, ema_alpha: float = 0.1):
        self.jsonl = jsonl
        self.agg = MetricsAggregator(ema_alpha=ema_alpha)
        self.csv = CSVWriter(csv_path) if csv_path else None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, phase: str = "train"):
        merged = self.agg.update(**metrics)
        payload = {"kind": "metric", "phase": phase, **merged}
        if step is not None:
            payload["step"] = int(step)
        self.jsonl.log(**payload)
        if self.csv:
            row = {"phase": phase, **merged}
            if step is not None:
                row["step"] = int(step)
            self.csv.write(row)

    def close(self):
        if self.csv:
            self.csv.close()
