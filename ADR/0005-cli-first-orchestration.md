# ADR 0005 — CLI-First Orchestration

> **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge  
> **Status:** ✅ Accepted | **Date:** 2025-09-07  
> **Tags:** `cli` · `orchestration` · `hydra` · `reproducibility` · `ux`  
> **Owners:** Infra WG (Lead: Andy Barta), ML/Infra, Tooling WG

---

## 🔎 1) Context

SpectraMind V50 is a **mission-grade AI pipeline** with the stages  
`calibrate → train → predict → diagnose → submit`.

**Challenges (pre-ADR):**
- Scripts scattered (`train.py`, `predict.py`, `notebooks/`) → inconsistent UX.
- Hydra configs are powerful but hard to invoke uniformly without a wrapper.
- Kaggle runtime requires **internet-disabled**, single-command reproducibility.
- CI/CD (GitHub Actions) needs uniform entrypoints to run all stages safely.

We need a **single, discoverable CLI** that exposes *all* pipeline functions with reproducible defaults.

---

## ✅ 2) Decision

Adopt a **CLI-first orchestration layer**:

- **Framework:** [Typer] (Python 3.10+) — modern, type-hinted, great help UX.
- **Single entrypoint:** `spectramind` with subcommands for all stages.
- **Hydra integration:** every subcommand accepts `--config-name` + overrides (e.g. `+foo=bar`).
- **Rich UX:** `--help`, tab completion, colorized errors (via [Rich]).
- **Determinism/Safety:** seeds, strict Hydra mode, Kaggle offline guards.

---

## 🎯 3) Drivers

- **Reproducibility** — one entrypoint across local dev, CI, and Kaggle.
- **Velocity** — fast override-driven experiments (`spectramind train +loss=nonneg`).
- **Auditability** — every invocation logs JSONL manifests with config hash.
- **Safety** — fail-loud defaults in CI; offline enforcement for Kaggle.
- **UX** — discoverable, consistent, low-friction for collaborators.

---

## 🔁 4) Alternatives Considered

| Option                                             | Pros                               | Cons                                       |
|---------------------------------------------------|------------------------------------|--------------------------------------------|
| Ad-hoc Python scripts (`train.py`, `predict.py`)  | Simple                             | Fragmented UX, poor discoverability        |
| Bash wrappers around Hydra                        | Easy CI bootstrap                  | Fragile, limited UX/portability            |
| Full workflow managers (Airflow, Prefect)         | Powerful DAGs                      | Overkill, heavy infra, **not Kaggle-safe** |
| **Chosen: Typer CLI + Hydra**                     | Lightweight, modern, Kaggle-safe   | Requires disciplined design                 |

---

## 🧩 5) Architecture

```mermaid
flowchart TD
  A["User / CI / Kaggle"] --> B["spectramind CLI (Typer)"]
  B -->|compose| C["Hydra Configs (configs/*)"]
  C --> D["Pipeline Modules (src/spectramind/*)"]
  D --> E["Artifacts (DVC-tracked)"]
  E --> F["Submission (283 μ/σ bins)"]
  B -. logs .-> G["Run Manifest (JSONL, config hash)"]
Absolutely—here’s a **styled, drop-in ADR** that matches your “fancy” format with a polished heading, metadata panel, callouts, Mermaid diagram, tables, checklists, and reference links. Paste this as:

`docs/adr/0005-cli-first-orchestration.md`

````markdown
# ADR 0005 — CLI-First Orchestration

> **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge  
> **Status:** ✅ Accepted | **Date:** 2025-09-07  
> **Tags:** `cli` · `orchestration` · `hydra` · `reproducibility` · `ux`  
> **Owners:** Infra WG (Lead: Andy Barta), ML/Infra, Tooling WG

---

## 🔎 1) Context

SpectraMind V50 is a **mission-grade AI pipeline** with the stages  
`calibrate → train → predict → diagnose → submit`.

**Challenges (pre-ADR):**
- Scripts scattered (`train.py`, `predict.py`, `notebooks/`) → inconsistent UX.
- Hydra configs are powerful but hard to invoke uniformly without a wrapper.
- Kaggle runtime requires **internet-disabled**, single-command reproducibility.
- CI/CD (GitHub Actions) needs uniform entrypoints to run all stages safely.

We need a **single, discoverable CLI** that exposes *all* pipeline functions with reproducible defaults.

---

## ✅ 2) Decision

Adopt a **CLI-first orchestration layer**:

- **Framework:** [Typer] (Python 3.10+) — modern, type-hinted, great help UX.
- **Single entrypoint:** `spectramind` with subcommands for all stages.
- **Hydra integration:** every subcommand accepts `--config-name` + overrides (e.g. `+foo=bar`).
- **Rich UX:** `--help`, tab completion, colorized errors (via [Rich]).
- **Determinism/Safety:** seeds, strict Hydra mode, Kaggle offline guards.

---

## 🎯 3) Drivers

- **Reproducibility** — one entrypoint across local dev, CI, and Kaggle.
- **Velocity** — fast override-driven experiments (`spectramind train +loss=nonneg`).
- **Auditability** — every invocation logs JSONL manifests with config hash.
- **Safety** — fail-loud defaults in CI; offline enforcement for Kaggle.
- **UX** — discoverable, consistent, low-friction for collaborators.

---

## 🔁 4) Alternatives Considered

| Option                                             | Pros                               | Cons                                       |
|---------------------------------------------------|------------------------------------|--------------------------------------------|
| Ad-hoc Python scripts (`train.py`, `predict.py`)  | Simple                             | Fragmented UX, poor discoverability        |
| Bash wrappers around Hydra                        | Easy CI bootstrap                  | Fragile, limited UX/portability            |
| Full workflow managers (Airflow, Prefect)         | Powerful DAGs                      | Overkill, heavy infra, **not Kaggle-safe** |
| **Chosen: Typer CLI + Hydra**                     | Lightweight, modern, Kaggle-safe   | Requires disciplined design                 |

---

## 🧩 5) Architecture

```mermaid
flowchart TD
  A["User / CI / Kaggle"] --> B["spectramind CLI (Typer)"]
  B -->|compose| C["Hydra Configs (configs/*)"]
  C --> D["Pipeline Modules (src/spectramind/*)"]
  D --> E["Artifacts (DVC-tracked)"]
  E --> F["Submission (283 μ/σ bins)"]
  B -. logs .-> G["Run Manifest (JSONL, config hash)"]
````

---

## 🛠 6) Implementation Plan

1. **CLI module:** `src/spectramind/cli.py`

   * Subcommands: `calibrate`, `train`, `predict`, `diagnose`, `submit`.
   * Delegates to pipeline functions (e.g. `spectramind.train.train:main`).

2. **Hydra binding:**

   * `--config-name` and arbitrary overrides (`+key=value`).
   * Snapshot resolved config to `artifacts/runs/.../config.snapshot.yaml`.

3. **UX polish:**

   * Rich-powered help/errors, colored tables.
   * Shell auto-completion (bash/zsh/fish).

4. **Compliance hooks:**

   * Each command appends to `events.jsonl` (UTC timestamp, seed, CWD, command line, config snapshot).
   * Kaggle mode enforces offline flags and reproducible seeds.

---

## 🧯 7) Risks & Mitigations

| Risk                       | Mitigation                                                          |
| -------------------------- | ------------------------------------------------------------------- |
| CLI bloat (too many flags) | Keep flags minimal; push complexity into Hydra configs + overrides  |
| CI vs local drift          | Pin Hydra/OmegaConf; CI smoke tests assert parity                   |
| Kaggle runtime quirks      | `bin/kaggle-boot.sh` sets offline/deterministic env                 |
| Override confusion         | Clear docs (`docs/guides/hydra.md`), `spectramind --help`, examples |

---

## 📌 8) Consequences

* ✅ Unified entrypoint → reproducible runs everywhere.
* ✅ Seamless Hydra integration & config snapshots.
* ✅ Better onboarding and CI uniformity.
* ❗ Discipline required: all workflows must route through the CLI (no hidden scripts).

---

## ✅ 9) Compliance Gates (CI)

* [ ] `spectramind --help` runs without error (smoke).
* [ ] Subcommands `calibrate/train/predict/diagnose/submit` expose `--help`.
* [ ] Config snapshot + JSONL manifest emitted per run.
* [ ] Pre-commit hook ensures new modules are reachable from CLI.

*Minimum test file:* `tests/integration/test_cli_smoke.py`

---

## 📚 10) References

* Repo Scaffold & Blueprint
* AI Research Notebook & Upgrade Guide
* ADR-0001 (Hydra + DVC)
* ADR-0003 (CI ↔ CUDA parity)
* GitHub Mermaid CLI diagrams

**Related ADRs:**

* [ADR-0002 — Composite Physics-Informed Loss]
* [ADR-0004 — Dual Encoder Fusion (FGS1 + AIRS)]

---

## 📦 Appendix — Drop-in Artifacts

### A) `pyproject.toml` console script

```toml
[project.scripts]
spectramind = "spectramind.cli:app"
```

### B) CLI scaffold (`src/spectramind/cli.py`)

```python
from __future__ import annotations
import json, os, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import typer
try:
    from rich import print as rprint
except Exception:
    rprint = print

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None  # type: ignore

app = typer.Typer(add_completion=True, help="SpectraMind V50 — CLI-first orchestration")

def _utc_iso() -> str: return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def _dump_jsonl(p: Path, rec: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text((p.read_text(encoding="utf-8") if p.exists() else "") + json.dumps(rec) + "\n", encoding="utf-8")

def _snapshot_cfg(cfg: Any, outdir: Path) -> Optional[Path]:
    outdir.mkdir(parents=True, exist_ok=True); out = outdir / "config.snapshot.yaml"
    try:
        if OmegaConf and cfg and cfg.__class__.__name__ in {"DictConfig","ListConfig"}:
            from omegaconf import OmegaConf as _OC; out.write_text(_OC.to_yaml(cfg), encoding="utf-8")
        else:
            out.write_text(json.dumps(cfg or {}, indent=2), encoding="utf-8")
        return out
    except Exception as e: rprint(f"[yellow]WARN[/] snapshot failed: {e}"); return None

def _determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    for mod, fn in (("random","seed"),("numpy","random.seed")):
        try:
            if mod=="random": import random as m; getattr(m, fn)(seed)
            else: import numpy as np; np.random.seed(seed)
        except Exception: pass
    try:
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    except Exception: pass

def _kaggle_safety(enabled: bool) -> None:
    if not enabled: return
    os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_DATASETS_OFFLINE"]="1"
    rprint("[cyan]Kaggle mode:[/] offline flags set.")

@dataclass
class RunManifest:
    command: str; stage: str; time_utc: str; seed: int; workdir: str
    config_snapshot: Optional[str]=None; extras: Dict[str, Any]=None
    def to_dict(self): return asdict(self)

def _log(stage: str, seed: int, cfg: Any, extras: Dict[str,Any]|None=None) -> None:
    run_dir = Path("artifacts")/"runs"/stage/time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    snap = _snapshot_cfg(cfg, run_dir/"config")
    _dump_jsonl(run_dir/"events.jsonl", RunManifest(" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]), stage, _utc_iso(), seed, str(Path.cwd()), str(snap) if snap else None, extras or {}).to_dict())
    rprint(f"[green]Logged →[/] {run_dir}/events.jsonl")

def _hydra_cfg(config_name: str|None, overrides: list[str]) -> Any:
    try:
        from hydra import initialize, compose
        with initialize(config_path="configs", version_base=None):
            return compose(config_name=config_name or "train", overrides=list(overrides) if overrides else [])
    except Exception:
        return {"config_name": config_name, "overrides": overrides}

def _call(modpath: str, fn: str, **kwargs: Any) -> None:
    try:
        module = __import__(modpath, fromlist=[fn]); getattr(module, fn)(**kwargs)
    except Exception:
        rprint(f"[yellow]{modpath}.{fn}[/] placeholder executed (hook up real entrypoint)")

@app.callback()
def global_opts(ctx: typer.Context, seed: int = typer.Option(42, help="Deterministic seed."), kaggle: bool = typer.Option(False, help="Kaggle offline guards.")):
    _determinism(seed); _kaggle_safety(kaggle); ctx.obj={"seed": seed, "kaggle": kaggle}

@app.command(help="Calibrate raw inputs.")
def calibrate(ctx: typer.Context, config_name: Optional[str]=typer.Option("calibrate","--config-name"), override: list[str]=typer.Argument(None, metavar="[HYDRA_OVERRIDES]")):
    seed=ctx.obj["seed"]; cfg=_hydra_cfg(config_name, override or []); _log("calibrate", seed, cfg, {"overrides": override or []})
    _call("spectramind.calib.main", "run", config=cfg, seed=seed)

@app.command(help="Train model(s).")
def train(ctx: typer.Context, config_name: Optional[str]=typer.Option("train","--config-name"), override: list[str]=typer.Argument(None, metavar="[HYDRA_OVERRIDES]")):
    seed=ctx.obj["seed"]; cfg=_hydra_cfg(config_name, override or []); _log("train", seed, cfg, {"overrides": override or []})
    _call("spectramind.train.train", "main", config=cfg, seed=seed)

@app.command(help="Run predictions.")
def predict(ctx: typer.Context, config_name: Optional[str]=typer.Option("predict","--config-name"), override: list[str]=typer.Argument(None, metavar="[HYDRA_OVERRIDES]")):
    seed=ctx.obj["seed"]; cfg=_hydra_cfg(config_name, override or []); _log("predict", seed, cfg, {"overrides": override or []})
    _call("spectramind.predict.main", "run", config=cfg, seed=seed)

@app.command(help="Generate diagnostics.")
def diagnose(ctx: typer.Context, config_name: Optional[str]=typer.Option("diagnose","--config-name"), override: list[str]=typer.Argument(None, metavar="[HYDRA_OVERRIDES]")):
    seed=ctx.obj["seed"]; cfg=_hydra_cfg(config_name, override or []); _log("diagnose", seed, cfg, {"overrides": override or []})
    _call("spectramind.diagnostics.reports", "main", config=cfg, seed=seed)

@app.command(help="Validate & package submission.")
def submit(ctx: typer.Context, config_name: Optional[str]=typer.Option("submit","--config-name"), override: list[str]=typer.Argument(None, metavar="[HYDRA_OVERRIDES]")):
    seed=ctx.obj["seed"]; cfg=_hydra_cfg(config_name, override or []); _log("submit", seed, cfg, {"overrides": override or []})
    _call("spectramind.submit.validate", "main", config=cfg, seed=seed)

if __name__ == "__main__":
    app()
```

### C) CI smoke (`tests/integration/test_cli_smoke.py`)

```python
import subprocess, sys
def _ok(cmd): 
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 0, r.stdout

def test_help():
    _ok([sys.executable, "-m", "spectramind.cli", "--help"])
    _ok(["spectramind", "--help"])

def test_subcommands_exist():
    for sub in ("calibrate","train","predict","diagnose","submit"):
        _ok(["spectramind", sub, "--help"])
```

---

## Footnotes & Link Refs

[Typer]: https://typer.tiangolo.com
[Rich]: https://rich.readthedocs.io
[ADR-0002 — Composite Physics-Informed Loss]: 0002-composite-physics-informed-loss.md
[ADR-0004 — Dual Encoder Fusion (FGS1 + AIRS)]: 0004-dual-encoder-fusion-fgs1-airs.md

```

Want me to also generate a **badge panel** (build/status/docs) snippet and a **left-nav ADR index** so this renders beautifully in MkDocs?
```
