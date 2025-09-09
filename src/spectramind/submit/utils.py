# src/spectramind/submit/utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json_pretty(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")