from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def generate_diagnostics_report(results: Dict[str, Any], out_path: Path) -> Path:
    """
    Generate a JSONL or HTML report from diagnostics results.

    Args:
        results: Dictionary from diagnostics modules.
        out_path: Output path (.json or .html).

    Returns:
        Path to saved report.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".json":
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    elif out_path.suffix == ".html":
        with open(out_path, "w") as f:
            f.write("<html><body><pre>")
            f.write(json.dumps(results, indent=2))
            f.write("</pre></body></html>")
    else:
        raise ValueError("Unsupported extension: must be .json or .html")

    return out_path
