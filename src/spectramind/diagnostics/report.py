# src/spectramind/diagnostics/report.py
# =============================================================================
# SpectraMind V50 — Diagnostics Report Generator (Upgraded)
# -----------------------------------------------------------------------------
# Produces JSON and/or HTML reports from diagnostic results, with:
#   • Deterministic JSON: sorted keys, UTF-8, canonical NumPy-safe encoding
#   • Content hash (sha256) for reproducibility tracking
#   • Styled, collapsible HTML; auto figure gallery if paths are provided
#   • Zero external deps (NumPy optional), CI/Kaggle-friendly
# =============================================================================

from __future__ import annotations

import json
import math
import os
import socket
import hashlib
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

try:
    import numpy as _np  # optional
except Exception:  # pragma: no cover
    _np = None  # type: ignore


# ----------------------------------------------------------------------------- #
# Public API
# ----------------------------------------------------------------------------- #

def generate_diagnostics_report(
    results: Dict[str, Any],
    out_path: Path,
    title: str = "SpectraMind V50 — Diagnostics Report",
    *,
    metadata: Optional[Dict[str, Any]] = None,
    figures: Optional[Sequence[Union[str, Path]]] = None,
    write_hash_sidecar: bool = True,
) -> Path:
    """
    Generate a JSON or HTML diagnostics report.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary of results from diagnostics modules.
        Tip: If you provide a top-level key "figures" with a list of file paths,
        the HTML renderer will pick those up automatically (or pass `figures=`).
    out_path : Path
        Output path (.json or .html). Parent dirs will be created.
    title : str
        Title for HTML report.
    metadata : Optional[Dict[str, Any]]
        Extra metadata to merge under a `_meta` key (timestamp, version, etc.).
    figures : Optional[Sequence[Union[str, Path]]]
        Paths to figure images to include in the HTML gallery (png/svg/jpg/webp).
    write_hash_sidecar : bool
        If True and `out_path.suffix == ".json"`, also emits `<name>.json.sha256`.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If the output extension is unsupported.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge metadata in a deterministic way under a reserved key `_meta`
    enriched = _with_meta(results, metadata)

    # Auto-detect figures from results if not supplied
    if figures is None:
        figures = _extract_figures(enriched)

    if out_path.suffix.lower() == ".json":
        txt, sha = _to_canonical_json_with_hash(enriched)
        out_path.write_text(txt, encoding="utf-8")
        if write_hash_sidecar:
            (out_path.with_suffix(out_path.suffix + ".sha256")).write_text(sha + "\n", encoding="utf-8")
        return out_path

    if out_path.suffix.lower() == ".html":
        html = _render_html(enriched, title=title, figures=figures)
        out_path.write_text(html, encoding="utf-8")
        return out_path

    raise ValueError(f"Unsupported extension for report: {out_path.suffix}")


def generate_json_and_html(
    results: Dict[str, Any],
    out_base: Path,
    *,
    title: str = "SpectraMind V50 — Diagnostics Report",
    metadata: Optional[Dict[str, Any]] = None,
    figures: Optional[Sequence[Union[str, Path]]] = None,
) -> Tuple[Path, Path]:
    """
    Convenience: write both `<out_base>.json` and `<out_base>.html`.

    Returns
    -------
    (json_path, html_path)
    """
    json_path = Path(str(out_base) + ".json")
    html_path = Path(str(out_base) + ".html")
    enriched = _with_meta(results, metadata)
    if figures is None:
        figures = _extract_figures(enriched)

    # JSON
    txt, sha = _to_canonical_json_with_hash(enriched)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(txt, encoding="utf-8")
    (json_path.with_suffix(".json.sha256")).write_text(sha + "\n", encoding="utf-8")

    # HTML
    html = _render_html(enriched, title=title, figures=figures)
    html_path.write_text(html, encoding="utf-8")
    return json_path, html_path


# ----------------------------------------------------------------------------- #
# Helpers — canonical JSON, meta, figures, and HTML rendering
# ----------------------------------------------------------------------------- #

def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _with_meta(results: Mapping[str, Any], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # Shallow copy; reserve _meta; avoid overwriting user-provided keys unexpectedly.
    base: Dict[str, Any] = dict(results)
    meta: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "tool": "spectramind.diagnostics.report",
        "version": "v50",
    }
    if metadata:
        # user metadata wins on conflict to be explicit
        meta.update(dict(metadata))
    # Merge under a reserved key
    if "_meta" in base and isinstance(base["_meta"], Mapping):
        merged = dict(base["_meta"])
        merged.update(meta)
        base["_meta"] = merged
    else:
        base["_meta"] = meta
    return base


def _extract_figures(results: Mapping[str, Any]) -> List[Path]:
    figs: List[Path] = []
    candidates: Iterable = ()
    # Accept top-level "figures": [pathlike, ...]
    if isinstance(results.get("figures"), (list, tuple)):
        candidates = results["figures"]  # type: ignore
    # Also accept nested "artifacts" → "figures"
    elif isinstance(results.get("artifacts"), Mapping):
        maybe = results["artifacts"].get("figures")  # type: ignore
        if isinstance(maybe, (list, tuple)):
            candidates = maybe

    for c in candidates or []:
        p = Path(str(c))
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".webp"} and p.exists():
            figs.append(p)
    return figs


# ------------------- Canonical JSON encoding (NumPy-safe) -------------------- #

def _default_json(obj: Any) -> Any:
    """JSON fallback encoder for NumPy, Path, set/tuple, dataclass, bytes, complex."""
    # dataclasses
    if is_dataclass(obj):
        return asdict(obj)

    # Path-like
    if isinstance(obj, (Path, )):
        return str(obj)

    # sets & tuples
    if isinstance(obj, (set, tuple)):
        return list(obj)

    # complex numbers → {"real": x, "imag": y}
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}

    # bytes → base64-ish string (hex to keep deps zero)
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes_hex__": bytes(obj).hex()}

    # NumPy dtypes/scalars/arrays
    if _np is not None:
        if isinstance(obj, (_np.generic, )):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            # Safe conversion: nested lists; beware huge arrays (user responsibility)
            return obj.tolist()

    # Fallback: best-effort string
    return str(obj)


def _to_canonical_json_with_hash(payload: Mapping[str, Any]) -> Tuple[str, str]:
    """
    Serialize mapping to canonical, deterministic JSON and return (text, sha256).
    - Sorted keys
    - Indent 2, trailing newline omitted (hash is over exact bytes)
    """
    txt = json.dumps(payload, default=_default_json, ensure_ascii=False, sort_keys=True, indent=2)
    sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    return txt, sha


# ------------------------------ HTML Rendering ------------------------------ #

_HTML_STYLE = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial,sans-serif;max-width:980px;margin:40px auto;padding:0 16px;line-height:1.5}
h1{color:#0b66c3;font-size:1.6em;margin:.2em 0 .6em}
h2{font-size:1.2em;margin-top:1.2em}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,Liberation Mono,monospace}
pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow:auto}
.small{color:#666;font-size:.9em}
.kv{display:grid;grid-template-columns:220px 1fr;gap:.4rem .8rem;align-items:baseline}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-top:8px}
.card{border:1px solid #e3e7ee;border-radius:8px;padding:8px;background:#fff}
summary{cursor:pointer;color:#0b66c3}
hr{border:none;border-top:1px solid #e3e7ee;margin:18px 0}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef6ff;color:#0b66c3;border:1px solid #cfe3ff;font-size:.85em}
"""

def _render_html(results: Mapping[str, Any], *, title: str, figures: Optional[Sequence[Union[str, Path]]]) -> str:
    meta = results.get("_meta", {})
    # Canonical JSON for the <pre> block (not hashed here; purely visual)
    json_txt = json.dumps(results, default=_default_json, ensure_ascii=False, sort_keys=True, indent=2)

    # Simple KV header (selected meta fields)
    header_rows = []
    def add_row(k: str, v: Any) -> None:
        if v is None:
            return
        header_rows.append(f"<div><strong>{k}</strong></div><div>{_esc(str(v))}</div>")

    add_row("Generated (UTC)", meta.get("generated_at_utc"))
    add_row("Host", meta.get("host"))
    add_row("PID", meta.get("pid"))
    add_row("Tool", meta.get("tool", "spectramind.diagnostics.report"))
    add_row("Version", meta.get("version", "v50"))

    # Figures gallery
    fig_html = ""
    figs = [Path(str(p)) for p in (figures or []) if Path(str(p)).exists()]
    if figs:
        cards = []
        for p in figs:
            caption = _esc(p.name)
            # Use relative path text; image src is the filesystem path. For CI artifacts, this is fine.
            cards.append(
                f"<div class='card'><img src='{_esc(str(p))}' alt='{caption}' "
                f"style='max-width:100%;height:auto;border-radius:6px;'/><div class='small'>{caption}</div></div>"
            )
        fig_html = f"<h2>Figures <span class='badge'>{len(cards)}</span></h2><div class='gallery'>{''.join(cards)}</div>"

    # Optional array previews: summarize any obvious large arrays in top-level
    previews = _array_previews(results)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{_esc(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>{_HTML_STYLE}</style>
</head>
<body>
  <h1>{_esc(title)}</h1>

  <div class="kv">{''.join(header_rows)}</div>

  {fig_html}

  {previews}

  <h2>Results (JSON)</h2>
  <details open><summary>Show/Hide JSON</summary>
    <pre>{_esc(json_txt)}</pre>
  </details>

  <hr/>
  <div class="small">SpectraMind V50 • Diagnostics Report • {_esc(meta.get('generated_at_utc', ''))}</div>
</body>
</html>"""
    return html


def _array_previews(results: Mapping[str, Any]) -> str:
    """
    Produce small previews for obvious arrays under top-level keys:
    - If value is a NumPy array or list of numbers, show shape/len/dtype and first 8 values.
    - Keeps HTML tiny & safe; does not walk deeply nested structures.
    """
    blocks: List[str] = []
    for k, v in results.items():
        if k == "_meta":
            continue
        # numpy array
        if _np is not None and isinstance(v, _np.ndarray):
            shape = "×".join(map(str, v.shape))
            dtype = str(v.dtype)
            flat = v.ravel()
            head = flat[:8].tolist()
            block = (
                f"<details><summary>Preview: <code>{_esc(k)}</code> — array shape {shape}, dtype {dtype}</summary>"
                f"<pre>head = { _esc(json.dumps(head, ensure_ascii=False)) }</pre></details>"
            )
            blocks.append(block)
            continue
        # python list of numbers (shallow)
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) or _is_np_number(x) for x in v[:8]):
            head = v[:8]
            block = (
                f"<details><summary>Preview: <code>{_esc(k)}</code> — list len {len(v)}</summary>"
                f"<pre>head = { _esc(json.dumps(head, ensure_ascii=False)) }</pre></details>"
            )
            blocks.append(block)
    if not blocks:
        return ""
    return "<h2>Quick Previews</h2>" + "".join(blocks)


def _is_np_number(x: Any) -> bool:
    if _np is None:
        return False
    return isinstance(x, _np.generic)


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )