## 1) Deterministic manifest + report hashing

Add a SHA-256 fingerprint of the generated `report.html`/`.md` to the sidecar JSON manifest. This helps trace exact report content across runs and is cheap to compute.

```python
# === in render_report(), just before `return out_path` ===
    # Sidecar manifest (.json) for programmatic inspection
    if write_manifest_json:
        manifest = {
            "title": data.title,
            "generated_at": _iso_now(),
            "run": asdict(data.run),
            "config_path": data.config.config_path,
            "metrics_scalars": data.metrics.scalars,
            "digests": [asdict(d) for d in data.digests],
        }
        (out_dir / "report_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

+   # Attach a content hash of the produced report
+   try:
+       rep_bytes = out_path.read_bytes()
+       rep_sha256 = hashlib.sha256(rep_bytes).hexdigest()
+       man_path = out_dir / "report_manifest.json"
+       if man_path.exists():
+           m = json.loads(man_path.read_text(encoding="utf-8"))
+           m["report_file"] = out_path.name
+           m["report_sha256"] = rep_sha256
+           man_path.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
+   except Exception:
+       pass
```

---

## 2) Safer ignore matching (glob semantics)

Use `fnmatch` semantics against POSIX forward-slash paths. This is both predictable and portable.

```python
+import fnmatch

def _matches_ignored(p: Path, root: Path, patterns: Iterable[str]) -> bool:
    try:
        rel = p.relative_to(root)
    except Exception:
        return True  # out-of-root: ignore
-    s = str(rel).replace("\\", "/")
-    # naive glob matching (Path.match matches from end segments too)
-    return any(rel.match(glob) or Path(s).match(glob) for glob in patterns)
+    s = str(rel).replace("\\", "/")
+    return any(fnmatch.fnmatch(s, pat) for pat in patterns)
```

---

## 3) Robust `importlib.metadata` handling

Ensure the optional import never crashes and is used consistently. Your current code works, but this keeps it explicit and avoids shadowing.

```python
# at top
-try:
-    # py311+ stdlib
-    from importlib.metadata import distributions, PackageNotFoundError  # type: ignore
-except Exception:  # pragma: no cover
-    distributions = None  # type: ignore
-    PackageNotFoundError = Exception  # type: ignore
+try:
+    from importlib.metadata import distributions as _distributions  # type: ignore
+except Exception:  # pragma: no cover
+    _distributions = None  # type: ignore

# ...

def _package_versions_snapshot(max_pkgs: int = 200) -> Dict[str, str]:
    out: Dict[str, str] = {}
-    if distributions is None:
+    if _distributions is None:
        return out
    try:
        # Keep deterministic order by name
-        pkgs = sorted(distributions(), key=lambda d: d.metadata.get("Name", "").lower())
+        pkgs = sorted(_distributions(), key=lambda d: d.metadata.get("Name", "").lower())
        for d in pkgs[:max_pkgs]:
            name = d.metadata.get("Name")
            version = d.version
            if name and version:
                out[name] = version
    except Exception:
        return out
    return out
```

---

## 4) Header-safe HTML tables

Pandas’ `.to_html()` can inject borders or vary CSS across versions. We keep your minimalist output but normalize HTML a bit to avoid layout surprises:

```python
def _tables_to_html(tables: Dict[str, "pd.DataFrame"]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if pd is None:
        return out
    for k, df in tables.items():
        try:
            trimmed = df.head(MAX_TABLE_ROWS)
-            out[k] = trimmed.to_html(index=False, border=0, justify="center")  # type: ignore
+            html = trimmed.to_html(index=False, border=0, justify="center", escape=True)  # type: ignore
+            # normalize whitespace for deterministic diffs
+            out[k] = "\n".join(line.rstrip() for line in html.splitlines())
        except Exception:
            continue
    return out
```

---

## 5) Defensive plotting & resource cleanup

On some CI stacks `matplotlib` will happily create figures even when disk is full and then fail. Wrap fig save with guardrails and always close.

```python
def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
-    fig.savefig(buf, format="png", bbox_inches="tight", dpi=144)
+    try:
+        fig.savefig(buf, format="png", bbox_inches="tight", dpi=144)
+    except Exception:
+        return ""
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"
```

And skip attaching empty data URIs:

```python
def plot_training_curves(history: Optional["pd.DataFrame"]) -> Optional[str]:
    # ...
-    return uri
+    return uri or None

def plot_spectrum(mu_sigma_df: Optional["pd.DataFrame"]) -> Optional[str]:
    # ...
-    return uri
+    return uri or None
```

---

## 6) Deterministic template bootstrap

Your on-the-fly template generator is great. Add a comment so future maintainers don’t nuke it by accident:

```python
def _default_templates_dir() -> Path:
    """Provide an embedded minimal template directory (created on first use).
    Note: repo-local cache; safe to clean, will regenerate automatically."""
    d = Path(".reports_templates")
    d.mkdir(exist_ok=True)
    # ...
```

---

## 7) Tiny nit fixes

* **Avoid shadowing**: `matplotlib.pyplot as _plt` → `plt = _plt` is fine, but move `plt=None` above the import block (you already did).
* **YAML dump fallback**: add `allow_unicode=True` for non-ASCII configs (matches your UTF-8 discipline).

```python
def _yaml_dump_pretty(data: Dict[str, Any]) -> str:
    if not data:
        return "(no config)"
    if yaml is None:
        return json.dumps(data, indent=2, sort_keys=True)
    try:
-        return yaml.safe_dump(data, sort_keys=True, indent=2)  # type: ignore
+        return yaml.safe_dump(data, sort_keys=True, indent=2, allow_unicode=True)  # type: ignore
    except Exception:
        return json.dumps(data, indent=2, sort_keys=True)
```

---

## 8) (Optional) CLI shim

If you want a tiny convenience entry point:

```python
# src/spectramind/diagnostics/cli.py
from __future__ import annotations
from pathlib import Path
import typer
from .reports import generate_report

app = typer.Typer(no_args_is_help=True)

@app.command()
def render(
    run_id: str,
    artifacts_dir: Path,
    config: Path = typer.Option(None, "--config"),
    metrics: Path = typer.Option(None, "--metrics"),
    history: Path = typer.Option(None, "--history"),
    preds: Path = typer.Option(None, "--preds"),
    notes: str = typer.Option(None, "--notes"),
):
    path = generate_report(
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        config_path=config,
        metrics_json=metrics,
        history_csv=history,
        predictions_csv=preds,
        notes=notes,
    )
    typer.echo(f"Report → {path}")

if __name__ == "__main__":
    app()
```
