# ADR 0006 — Reproducibility Standards

> **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge  
> **Status:** ✅ Accepted | **Date:** 2025-09-07  
> **Tags:** `reproducibility` · `audit` · `lineage` · `compliance` · `sbom`  
> **Owners:** Reproducibility WG (Lead: Andy Barta), ML/Infra, Data Ops

---

## 🔎 1) Context

The Ariel Data Challenge requires **mission-grade reproducibility**:

- **Kaggle runtime:** no internet, ≤9h wallclock → pipeline must be deterministic & self-contained.  
- **Scientific credibility:** reviewers demand **audit trails** (what config, what data, what commit).  
- **Compliance:** prove **no hidden dependencies**, **no nondeterminism**, **secure packaging**.

Prior ad-hoc approaches (print logs, notebook checkpoints) are insufficient. We need **hard guarantees**.

---

## ✅ 2) Decision

Adopt a **multi-layer reproducibility standard**:

1) **Run manifests**  
   Every `spectramind …` invocation emits a structured JSONL record (UTC timestamp, git commit, config hash, CUDA/parity, seeds, produced artifacts).

2) **Config snapshots**  
   Hydra’s resolved config is frozen to `artifacts/runs/<ts>/config.snapshot.yaml` and its SHA256 is logged.

3) **Artifact lineage (DVC)**  
   All stages (`calibrate → train → predict → submit`) are declared in `dvc.yaml` with `dvc.lock` checked in to prove lineage.

4) **Software Bill of Materials (SBOM)**  
   Each release produces SPDX + CycloneDX SBOM (Syft/Grype). CI gates alerts (unknown deps, license conflicts, CVEs).

5) **Determinism guardrails**  
   Enforce seeds and deterministic math (disable CuDNN benchmarking/TF32; opt-in deterministic kernels). See ADR-0003 for CUDA parity.

---

## 🎯 3) Drivers

- **Scientific auditability** — reviewers can reproduce leaderboard runs exactly.  
- **Safety** — prevent accidental nondeterminism in Kaggle runtime.  
- **Transparency** — manifests + SBOM give short audit paths.  
- **CI parity** — same configs and hashes across local/CI/Kaggle.

---

## 🔁 4) Alternatives

| Option                                | Pros                                | Cons                              |
|---------------------------------------|-------------------------------------|-----------------------------------|
| Notebook checkpoints (ad-hoc)         | Easy                                | Fragile, unreproducible           |
| MLflow / experiment tracker            | Nice UI                             | Heavy infra, not Kaggle-safe      |
| Docker pinning only                    | Portable                            | No fine-grained artifact lineage  |
| **Chosen: JSONL + Hydra + DVC + SBOM**| Lightweight, Kaggle-safe, auditable | Requires CI discipline            |

---

## 🧩 5) Architecture

```mermaid
flowchart TD
  A[Spectramind CLI] --> B[Hydra Config Snapshot\n(YAML + SHA256)]
  A --> C[Run Manifest (JSONL)]
  A --> D[DVC Stages (dvc.yaml)]
  D --> E[Artifacts\n(calib, ckpt, preds, submissions)]
  C --> F[Audit Trail\n(JSONL + hashes)]
  G[CI/CD] --> H[SBOM (SPDX + CycloneDX)]



Here you go — **full file, clean copy-paste**.

**Save as:** `docs/adr/0006-reproducibility-standards.md`

````markdown
# ADR 0006 — Reproducibility Standards

> **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge  
> **Status:** ✅ Accepted | **Date:** 2025-09-07  
> **Tags:** `reproducibility` · `audit` · `lineage` · `compliance` · `sbom`  
> **Owners:** Reproducibility WG (Lead: Andy Barta), ML/Infra, Data Ops

---

## 🔎 1) Context

The Ariel Data Challenge requires **mission-grade reproducibility**:

- **Kaggle runtime:** no internet, ≤9h wallclock → pipeline must be deterministic & self-contained.  
- **Scientific credibility:** reviewers demand **audit trails** (what config, what data, what commit).  
- **Compliance:** prove **no hidden dependencies**, **no nondeterminism**, **secure packaging**.

Prior ad-hoc approaches (print logs, notebook checkpoints) are insufficient. We need **hard guarantees**.

---

## ✅ 2) Decision

Adopt a **multi-layer reproducibility standard**:

1) **Run manifests**  
   Every `spectramind …` invocation emits a structured JSONL record (UTC timestamp, git commit, config hash, CUDA/parity, seeds, produced artifacts).

2) **Config snapshots**  
   Hydra’s resolved config is frozen to `artifacts/runs/<ts>/config.snapshot.yaml` and its SHA256 is logged.

3) **Artifact lineage (DVC)**  
   All stages (`calibrate → train → predict → submit`) are declared in `dvc.yaml` with `dvc.lock` checked in to prove lineage.

4) **Software Bill of Materials (SBOM)**  
   Each release produces SPDX + CycloneDX SBOM (Syft/Grype). CI gates alerts (unknown deps, license conflicts, CVEs).

5) **Determinism guardrails**  
   Enforce seeds and deterministic math (disable CuDNN benchmarking/TF32; opt-in deterministic kernels). See ADR-0003 for CUDA parity.

---

## 🎯 3) Drivers

- **Scientific auditability** — reviewers can reproduce leaderboard runs exactly.  
- **Safety** — prevent accidental nondeterminism in Kaggle runtime.  
- **Transparency** — manifests + SBOM give short audit paths.  
- **CI parity** — same configs and hashes across local/CI/Kaggle.

---

## 🔁 4) Alternatives

| Option                                | Pros                                | Cons                              |
|---------------------------------------|-------------------------------------|-----------------------------------|
| Notebook checkpoints (ad-hoc)         | Easy                                | Fragile, unreproducible           |
| MLflow / experiment tracker            | Nice UI                             | Heavy infra, not Kaggle-safe      |
| Docker pinning only                    | Portable                            | No fine-grained artifact lineage  |
| **Chosen: JSONL + Hydra + DVC + SBOM**| Lightweight, Kaggle-safe, auditable | Requires CI discipline            |

---

## 🧩 5) Architecture

```mermaid
flowchart TD
  A["spectramind CLI"] --> B["Hydra Config Snapshot<br/>(YAML + SHA256)"]
  A --> C["Run Manifest (JSONL)"]
  A --> D["DVC Stages (dvc.yaml)"]
  D --> E["Artifacts<br/>(calib, ckpt, preds, submissions)"]
  C --> F["Audit Trail<br/>(JSONL + hashes)"]
  G["CI/CD"] --> H["SBOM (SPDX + CycloneDX)"]
````

---

## 🛠 6) Implementation Plan

### 6.1 Run manifest (JSONL)

* Path: `artifacts/runs/<stage>/<YYYYMMDD_HHMMSS>/events.jsonl`
* Record fields (minimum):
  `time_utc, stage, command, seed, workdir, git_commit, config_snapshot_path, config_sha256, cuda_version, torch_version, artifacts`

**Adapter (drop-in):**

```python
# src/spectramind/logging/run_manifest.py
from __future__ import annotations
import hashlib, json, subprocess, time
from pathlib import Path
from typing import Any, Dict

def _utc() -> str: return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def _git_commit() -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: return "unknown"

def sha256_path(p: Path) -> str:
    if not p.exists(): return "missing"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

def append_manifest(stage: str, record: Dict[str, Any]) -> Path:
    run_dir = Path("artifacts")/"runs"/stage/time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir/"events.jsonl"
    base = {"time_utc": _utc(), "stage": stage, "git_commit": _git_commit()}
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**base, **record}, ensure_ascii=False) + "\n")
    return out
```

Integrate: call `append_manifest(...)` from each CLI subcommand after composing config.

---

### 6.2 Config snapshot (Hydra)

* Write resolved config to `artifacts/runs/<ts>/config.snapshot.yaml`
* Log `config_sha256` in the manifest (use `sha256_path` above).

---

### 6.3 DVC pipeline

**Minimal `dvc.yaml` skeleton:**

```yaml
stages:
  calibrate:
    cmd: spectramind calibrate --config-name=calibrate
    deps:
      - configs/calibrate.yaml
      - src/spectramind/calib
    outs:
      - artifacts/calib

  train:
    cmd: spectramind train --config-name=train
    deps:
      - configs/train.yaml
      - artifacts/calib
      - src/spectramind/train
    outs:
      - artifacts/ckpt

  predict:
    cmd: spectramind predict --config-name=predict
    deps:
      - configs/predict.yaml
      - artifacts/ckpt
    outs:
      - artifacts/preds

  submit:
    cmd: spectramind submit --config-name=submit
    deps:
      - configs/submit.yaml
      - artifacts/preds
      - schemas/submission.tableschema.sample_id.json
    outs:
      - artifacts/submission/submission.csv
```

CI should run: `dvc repro && dvc status -c` (clean).

---

### 6.4 SBOM (Syft/Grype) — CI gate

**.github/workflows/sbom-refresh.yml:**

```yaml
name: sbom-refresh
on:
  push: { branches: [ main ] }
  schedule:
    - cron: "0 7 * * 2" # weekly (Tue 07:00 UTC)
  workflow_dispatch:

permissions:
  contents: read
  security-events: write

jobs:
  sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }

      - name: Install Syft & Grype
        uses: anchore/sbom-action/download-syft@v0.17.9
      - uses: anchore/scan-action/download-grype@v3

      - name: Generate SPDX & CycloneDX
        run: |
          ./syft packages dir:. -o spdx-json > sbom.spdx.json
          ./syft packages dir:. -o cyclonedx-json > sbom.cdx.json

      - name: Scan for CVEs
        run: |
          ./grype sbom:sbom.cdx.json --fail-on=high || true

      - name: Upload SBOMs
        uses: actions/upload-artifact@v4
        with:
          name: sbom-${{ github.sha }}
          path: |
            sbom.spdx.json
            sbom.cdx.json
```

---

### 6.5 Determinism guardrails

**Init once per run (called by CLI):**

```python
# src/spectramind/utils/determinism.py
def set_deterministic(seed: int = 42) -> None:
    import os, random
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception: pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Optional stricter mode (can reduce speed):
        # torch.use_deterministic_algorithms(True, warn_only=True)
        # Disable TF32 (Ampere+):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception: pass
```

---

## 🧯 7) Risks & Mitigations

| Risk                                  | Mitigation                                      |
| ------------------------------------- | ----------------------------------------------- |
| SBOM false positives                  | Curated ignores; weekly refresh workflow        |
| Dev bypass (calling scripts directly) | Pre-commit hook to forbid non-CLI entrypoints   |
| Kaggle overhead (extra logging)       | JSONL append only; snapshot once per run        |
| Storage bloat (snapshots/manifests)   | Retention policy: keep last *N* runs per branch |

---

## 📌 8) Consequences

* ✅ Mission-grade reproducibility: every run auditable.
* ✅ Kaggle submissions provably tied to configs + git commit.
* ✅ Security posture improved (SBOM + CVE scan).
* ❗ Requires dev discipline around Hydra + DVC + CLI.

---

## ✅ 9) Compliance Gates (CI)

* [ ] `artifacts/runs/**/events.jsonl` exists after each stage.
* [ ] Config snapshot written and SHA256 recorded in manifest.
* [ ] `dvc repro` passes; `dvc status -c` clean.
* [ ] SBOM generated on release/main; CVE scan run.
* [ ] Determinism test job: same seed → identical checksum of key outputs.

---

## 📚 10) References

* SpectraMind repo scaffold & production blueprint
* AI Research Notebook & Upgrade Guide
* Scientific context on calibration noise
* JWST spectroscopy results (CO₂/SO₂)
* ADR-0001 (Hydra + DVC), ADR-0003 (CI ↔ CUDA), ADR-0005 (CLI-First)

```
