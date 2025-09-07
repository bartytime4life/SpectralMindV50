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

flowchart LR
  A["Hydra Configs (env, data, calib, model, train, loss, logger)"] -->|OmegaConf snapshot| B["Run Manifest"]

  A --> C["CLI (Typer) spectramind *"]

  C --> D["DVC Pipeline (dvc.yaml stages)"]

  D --> E["Artifacts (raw, calib, tensors, ckpts, preds)"]

  E --> F["Submission (283-bin mu sigma)"]

  D --> G["Remote Cache (S3 / SSH / Local)"]


````

---

## 7. Implementation Plan

### Hydra

* **Config layout (`configs/`):**

  * `env/` — `local.yaml`, `kaggle.yaml`, `ci.yaml`
  * `data/` — `kaggle.yaml`, `smoke.yaml`, `science.yaml`
  * `calib/` — `nominal.yaml`, `dev.yaml` (+ ADC/CDS/dark/flat/trace/phase)
  * `model/` — `v50.yaml`, ablations
  * `training/` — `trainer.yaml`, `precision.yaml`, `num_workers/`, `accumulate_grad_batches/`
  * `loss/` — `smoothness.yaml`, `nonneg.yaml`, `molecular.yaml`, `composite.yaml`
  * `logger/` — `jsonl.yaml`, `wandb_off.yaml`
  * Entry points: `train.yaml`, `predict.yaml`

* **Snapshot & hashing:** persist merged config + hash under `artifacts/runs/<run_id>/`.

* **Guards:** forbid missing keys, require explicit defaults, pin seeds, deterministic data loaders.

### DVC

* **Pipeline (`dvc.yaml`):**

  * `calibrate`: raw → calib data.
  * `train`: calib tensors → ckpts.
  * `predict`: ckpts → predictions.
  * `submit`: predictions → submission CSV.

* **Remotes:** default `localcache`; optional S3/SSH team remotes.

* **Locks:** commit `dvc.lock`; enforce `dvc repro` in CI.

---

## 8. Example Snippets

### Hydra (`configs/train.yaml`)

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

### DVC (`dvc.yaml`)

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

### Run Manifest

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

## 9. Risks & Mitigations

* **Config sprawl** → add config lint, group READMEs, ADR patterns.
* **Hidden nondeterminism** → pinned seeds, CI “determinism” job, `torch.use_deterministic_algorithms(True)`.
* **DVC remote contention** → default localcache + GC, optional team remote quotas.
* **Kaggle constraints (no internet, 9h cap)** → smoke configs, artifact packing.

---

## 10. Consequences

* ✅ Repeatable experiments, fast ablations, audit-friendly submissions.
* ❌ Dev learning curve (Hydra/DVC); YAML/metadata discipline needed.

---

## 11. Compliance Checklist (CI Gates)

* [ ] `spectramind doctor` passes (env, CUDA, seeds).
* [ ] `pre-commit run -a` clean.
* [ ] `dvc status` clean; `dvc repro` succeeds.
* [ ] Run manifest + config snapshot archived.
* [ ] Submission CSV schema-valid (283-bin μ/σ).

---

## 12. How to Revisit

Revisit if: Kaggle rules change, new sensors added, calibration diverges, or richer experiment UI is required.

---

## 13. References

* Internal: `configs/*/ARCHITECTURE.md`, `docs/architecture/`, `dvc.yaml`, `Makefile`.
* External: Hydra docs, DVC docs, reproducible ML literature.
* Related: [ADR 0002 — Physics-Informed Losses](0002-physics-informed-losses.md)

---

## 14. FAQ

**Q:** Why not Git LFS for models?
**A:** No pipeline semantics; DVC encodes DAG + cache policy.

**Q:** Hydra without DVC?
**A:** Possible, but artifact lineage is lost.

**Q:** Quick smoke test?
**A:** `+data=smoke +training=trainer_smoke +env=ci` with `dvc repro -s train`.

**Q:** How does this aid reviews?
**A:** PRs show config deltas, CI proves pipeline, manifests ensure local reproducibility.

```

Would you like me to also regenerate an **`adr/README.md` index file** that lists ADR 0001 and ADR 0002 for navigation, like a mini RFC index?
```
