# ADR 0001 — Choose Hydra + DVC for Configuration & Data/Experiment Lineage

* **Status:** ✅ Accepted  
* **Date:** 2025-09-06  
* **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge (FGS1 + AIRS)  
* **Tags:** configuration, reproducibility, pipeline, hydra, dvc  
* **Owners:** Architecture WG (Lead: Andy Barta), ML/Infra, Data Ops  

---

## 1. Context

SpectraMind V50 targets a **mission-grade, fully reproducible pipeline** for multi-sensor fusion (FGS1 photometry + AIRS spectroscopy). The system must:

- Compose complex experiment configs (env/data/calib/model/training/loss/logger) safely and repeatably across local, CI, and Kaggle runtimes.  
- Track dataset lineage, calibration parameters, and artifacts across iterative experiments and submissions.  
- Provide deterministic re-runs, short audit trails, and human-readable manifests for scientific review.  

We require a **configuration framework** and a **data/experiment lineage framework** that:

- Interoperate cleanly with CLI workflows.  
- Scale from laptops → GPU clusters → Kaggle (no internet, 9h runtime).  
- Enforce determinism and auditability without heavy infra.  

---

## 2. Decision

Adopt **Hydra** for hierarchical configuration and **DVC** for data & pipeline lineage.

### Hydra

- Unified configs for env/data/calib/model/training/loss/logger.  
- Safe composition via `defaults` and CLI overrides (`+env=local`, `+data=kaggle`, `+model=v50`).  
- Frozen OmegaConf dumps + hashes per run → reproducible snapshots.  

### DVC

- Encodes calibration → training → prediction → submission as `dvc.yaml` stages.  
- Tracks artifacts (raw → calibrated → tensors → ckpts → preds).  
- Remote storage (local/S3/SSH) keeps Git lean.  

**Division of labor:** Hydra owns **experiment shape & parameters**; DVC owns **artifacts & DAG**.  
CLI (`spectramind calibrate/train/predict/submit`) binds them.  

---

## 3. Drivers

- **Reproducibility** — exact reconstruction from config + DVC cache.  
- **Safety** — no hidden defaults; CI/Kaggle fail loud.  
- **Iteration speed** — tweak configs, not code.  
- **Auditability** — JSONL manifests + DVC graph.  
- **Portability** — identical runs across local/Linux/Kaggle.  

---

## 4. Alternatives

1. **Pydantic + Makefiles + Git LFS** — too brittle, no DAG lineage.  
2. **MLFlow + YAML + scripts** — infra heavy, Kaggle-unfriendly.  
3. **Poetry + Docker only** — simple but poor reproducibility and audit trails.  

Hydra + DVC balance **safety, determinism, and offline constraints**.  

---

## 5. Scope

- **In-scope:** configs, pipeline DAG, calibration lineage, artifacts, manifests, CI gates.  
- **Out-of-scope:** dashboards/experiment UI (future W&B/MLFlow), large sweeps.  

---

## 6. Architecture Overview

```mermaid
flowchart LR
  A["Hydra Configs\n(env, data, calib, model, training, loss, logger)"]
    -->|OmegaConf snapshot| B["Run Manifest"]

  A --> C["CLI (Typer) spectramind"]
  C --> D["DVC Pipeline\n(dvc.yaml stages)"]
  D --> E["Artifacts\n(raw, calib, tensors, ckpts, preds)"]
  E --> F["Submission\n(283-bin μ/σ)"]
  D --> G["Remote Cache\n(S3 / SSH / Local)"]
````

---

## 7. Implementation Plan

### Hydra

* Config tree: `configs/{env,data,calib,model,training,loss,logger}`.
* Snapshot + hash each run into `artifacts/runs/<run_id>/`.
* Guards: forbid missing keys, pin seeds, enforce deterministic dataloaders.

### DVC

* Pipeline stages: `calibrate`, `train`, `predict`, `submit`.
* Lock file (`dvc.lock`) enforced in CI.
* Default remote = `localcache`; optional S3/SSH.

---

## 8. Snippets

**Hydra (configs/train.yaml)**

```yaml
defaults:
  - env: local
  - data: kaggle
  - calib: nominal
  - model: v50
  - training: trainer
  - loss: composite
  - logger: jsonl

seed: 2025
```

**DVC (dvc.yaml)**

```yaml
stages:
  calibrate:
    cmd: spectramind calibrate --config-name train +phase=calib
    deps: [configs/calib, data/raw]
    outs: [data/calib]
  train:
    cmd: spectramind train --config-name train
    deps: [configs/model, configs/training, data/tensors]
    outs: [artifacts/ckpts]
  predict:
    cmd: spectramind predict --config-name predict
    deps: [artifacts/ckpts/best.ckpt, data/tensors_eval]
    outs: [artifacts/preds]
  submit:
    cmd: spectramind submit artifacts/preds --out artifacts/submissions
    deps: [artifacts/preds]
    outs: [artifacts/submissions]
```

**Run Manifest (JSON)**

```json
{
  "project": "spectramind-v50",
  "run_id": "2025-09-06T21-55-00Z_local_v50_nominal",
  "config_hash": "a5b1…",
  "hydra_config": "artifacts/runs/.../config.yaml",
  "git_rev": "<sha>",
  "stages": ["calibrate","train","predict"],
  "artifacts": {
    "ckpt": "artifacts/ckpts/best.ckpt",
    "preds": "artifacts/preds/val.csv"
  }
}
```

---

## 9. Risks

* **Config sprawl** → add lint, group READMEs, ADR templates.
* **Nondeterminism** → CI determinism job (`torch.use_deterministic_algorithms(True)`).
* **DVC contention** → localcache + quotas.
* **Kaggle runtime** → smoke configs, artifact packing.

---

## 10. Consequences

* ✅ Repeatable experiments, CI-verified lineage.
* ❌ YAML discipline + Hydra/DVC learning curve required.

---

## 11. CI Compliance Gates

* [ ] `spectramind doctor` passes (env, CUDA, seeds).
* [ ] Pre-commit hooks clean.
* [ ] `dvc status` clean; `dvc repro` succeeds.
* [ ] Run manifest + config snapshot archived.
* [ ] Submission schema valid (283-bin μ/σ).

---

## 12. Revisit Triggers

Revisit if Kaggle rules change, new sensors added, calibration diverges, or we require experiment UIs.

---

## 13. References

* Repo: `configs/*/ARCHITECTURE.md`, `dvc.yaml`, `Makefile`.
* Docs: Hydra, DVC, reproducible ML papers.
* Related: ADR 0002 — Physics-Informed Losses.

---

## 14. FAQ

**Q:** Why not Git LFS?
**A:** No DAG semantics; DVC encodes lineage + cache policy.

**Q:** Hydra without DVC?
**A:** Possible, but artifact lineage lost.

**Q:** Quick smoke test?
**A:** `+data=smoke +training=trainer_smoke +env=ci` with `dvc repro -s train`.

**Q:** How does this help reviews?
**A:** PR diffs show config deltas; CI proves pipeline; manifests ensure reproducibility.

```
