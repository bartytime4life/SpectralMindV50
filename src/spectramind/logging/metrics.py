from __future__ import annotations
import csv, io, os
from typing import Dict, Optional
from .utils import iso_now, safe_mkdir

class MetricsAggregator:
    """
    Stores latest values and exponential moving averages (EMA) per key.
    """
    def __init__(self, ema_alpha: float = 0.1) -> None:
        self.alpha = float(ema_alpha)
        self.latest: Dict[str, float] = {}
        self.ema: Dict[str, float] = {}

    def update(self, **metrics: float) -> Dict[str, float]:
        out = {}
        for k, v in metrics.items():
            v = float(v)
            self.latest[k] = v
            if k in self.ema:
                self.ema[k] = self.alpha * v + (1.0 - self.alpha) * self.ema[k]
            else:
                self.ema[k] = v
            out[k] = v
            out[f"{k}_ema"] = self.ema[k]
        return out

class CSVWriter:
    """
    Simple metrics CSV writer: writes header on first write, then rows w/ ISO ts.
    """
    def __init__(self, path: str) -> None:
        self.path = os.fspath(path)
        safe_mkdir(os.path.dirname(self.path) or ".")
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._header_written = os.path.getsize(self.path) > 0

    def write(self, row: Dict[str, float]) -> None:
        row = {"ts": iso_now(), **row}
        if self._writer is None:
            fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
            if not self._header_written:
                self._writer.writeheader()
                self._header_written = True
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
