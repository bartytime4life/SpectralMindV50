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
- Enforce determinism and auditability without adding heavy infra.  

---

## 2. Decision

Adopt **Hydra** for hierarchical configuration and **DVC** for data & pipeline lineage.

### Hydra (Structured Configs + Config Groups)

- Single source of truth for environment, data, calibration, model, training, loss, logger, runners.  
- Safe composition via `defaults` and overrides (`+env=local`, `+data=kaggle`, `+calib=nominal`, `+model=v50`).  
- Repro snapshots: frozen OmegaConf dump + content hash per run.  

### DVC (with Git as control plane)

- Tracks raw → calibrated → tensorized data artifacts and model checkpoints.  
- Encodes pipeline stages (`dvc.yaml`) for **calibrate → train → predict → submit** with explicit deps/outs.  
- Supports remotes (local/SSH/S3/GDrive) for heavy artifacts, keeping Git lean.  

**Division of labor:** Hydra owns **experiment shape & parameters**; DVC owns **artifacts & pipeline DAG**.  
CLI (`spectramind calibrate/train/predict/submit`) binds them.  

---

## 3. Decision Drivers

- **Reproducibility** — exact reconstruction of any leaderboard submission from config snapshot + DVC cache.  
- **Safety** — no hidden defaults; fail loud in CI/Kaggle.  
- **Iteration speed** — swap configs (`loss.smoothness.weight=0.1`) without code edits.  
- **Auditability** — JSONL manifests + DVC DAG explainability.  
- **Portability** — identical behavior across local/Linux/Kaggle (no internet, 9h cap).  

---

## 4. Alternatives Considered

1. **Pydantic configs + Makefiles + Git LFS**  
   - ✅ Simple; Python type safety.  
   - ❌ No first-class composition, weak lineage, brittle artifact orchestration.  

2. **MLFlow + YAML + ad-hoc scripts**  
   - ✅ Experiment UI, metrics registry.  
   - ❌ Determinism depends on discipline; heavy infra; weak dataset versioning under Kaggle constraints.  

3. **Poetry scripts + custom cache + Docker only**  
   - ✅ Simple mental model.  
   - ❌ No graph lineage, opaque cache invalidation, poor audit trails.  

**Why Hydra + DVC?** Balanced safety, determinism, lean Git, and proven patterns under offline constraints.  

---

## 5. Scope & Non-Goals

- **In-scope:** configs, calibration lineage, pipeline DAG, artifact stores, run manifests, CLI integration, CI gates.  
- **Out-of-scope:** experiment UI/dashboarding (may add W&B/MLFlow later); hyper-parameter sweeps beyond simple grid/random.  

---

## 6. Architecture Overview

```mermaid
flowchart LR
  A[Hydra Configs<br/>(env, data, calib, model, train, loss, logger)]
    -->|OmegaConf snapshot| B[Run Manifest]

  A --> C[CLI (Typer)<br/>spectramind *]

  C --> D[DVC Pipeline<br/>(dvc.yaml stages)]

  D --> E[Artifacts<br/>(raw, calib, tensors, ckpts, preds)]

  E --> F[Submission<br/>(283-bin μ/σ)]

  D --> G[Remote Cache<br/>(S3 / SSH / Local)]
