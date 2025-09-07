# ADR 0001 — Choose Hydra + DVC for Configuration & Data/Experiment Lineage

* **Status:** Accepted
* **Date:** 2025-09-06
* **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge (FGS1 + AIRS)
* **Owners:** Architecture WG (Lead: Andy Barta), ML/Infra, Data Ops

---

## 1. Context

SpectraMind V50 targets a mission‑grade, fully reproducible pipeline for multi‑sensor fusion (FGS1 photometry + AIRS spectroscopy). The system must:

* Compose complex experiment configs (env/data/calib/model/training/loss/logger) safely and repeatably across local, CI, and Kaggle runtimes.
* Track dataset lineage, calibration parameters, and artifacts across iterative experiments and submissions.
* Provide deterministic re‑runs, short audit trails, and human‑readable manifests for scientific review.

We need a **configuration framework** and a **data/experiment lineage framework** that interoperate cleanly, are CLI‑first, and scale from laptops to GPU runners without code changes.

---

## 2. Decision

Adopt **Hydra** for hierarchical configuration and **DVC** for data & pipeline lineage.

* **Hydra (with Structured Configs & Config Groups)**

  * Single source of truth for environment, data, calibration, model, training, loss, logger, runners.
  * Safe composition via `defaults` and group overrides (`+env=local`, `+data=kaggle`, `+calib=nominal`, `+model=v50`).
  * Repro snapshots (frozen OmegaConf dump + content hash) stored per run.

* **DVC (with Git as control plane)**

  * Tracks raw → calibrated → tensorized data artifacts and model checkpoints.
  * Defines pipeline stages (`dvc.yaml`) for **calibrate → train → predict → submit** with explicit inputs/outputs.
  * Remote storage (local/SSH/S3/GDrive) for heavy artifacts, enabling lean Git.

Hydra owns **experiment shape & parameters**; DVC owns **data, artifacts, and pipeline execution graph**. CLI tools bind them: `spectramind calibrate/train/predict/submit`.

---

## 3. Decision Drivers

* **Reproducibility:** exact re‑construction of any leaderboard submission from config snapshot + DVC cache.
* **Safety:** avoid config drift; prevent hidden defaults; fail “loud” in CI/Kaggle.
* **Speed of iteration:** swap configs (e.g., `loss.smoothness.weight=0.1`) without code edits.
* **Auditability:** readable manifests (JSONL, YAML) + DVC DAG explainability.
* **Portability:** consistent behavior across local/Linux/Kaggle images (no internet, 9h time budget).

---

## 4. Alternatives Considered

1. **Pydantic‑only configs + Makefiles + plain Git LFS**

   * Pros: minimal stack, Python type safety.
   * Cons: no first‑class composition, weak experiment lineage, manual artifact orchestration, brittle in CI/Kaggle.

2. **MLFlow for tracking + YAML configs + ad‑hoc scripts**

   * Pros: experiment UI, metrics registry.
   * Cons: artifact determinism depends on discipline; heavier infra; weaker dataset versioning under no‑internet constraints.

3. **Poetry scripts + custom data cache + Docker‑only**

   * Pros: simple mental model.
   * Cons: no graph lineage, opaque cache invalidation, poor audit trails; heavier images for Kaggle.

**Why Hydra + DVC?** Best balance of composition safety, artifact determinism, lean Git repo, and proven patterns under offline or restricted environments.

---

## 5. Scope & Non‑Goals

* **In‑scope:** configs, data/calibration lineage, pipeline DAG, artifact stores, run manifests, CLI integration, CI enforcement.
* **Out‑of‑scope:** experiment UI/dashboarding (can be added later via Weights & Biases/MLflow); hyper‑parameter sweeps beyond simple grid/random.

---

## 6. Architecture Overview

```mermaid
flowchart LR
  A[Hydra Configs\n(env,data,calib,model,train,loss,logger)] -->|OmegaConf snapshot| B[Run Manifest]
  A --> C[CLI (Typer)\n spectramind *]
  C --> D[DVC Pipeline\n(dvc.yaml stages)]
  D --> E[Artifacts\n(raw, calib, tensors, ckpts, predictions)]
  E --> F[Submission\n(283‑bin μ/σ)]
  D --> G[Remote Cache\n(S3/SSH/Local)]
```

---

## 7. Implementation Plan

### 7.1 Hydra

* **Config layout** under `configs/`:

  * `env/` (`local.yaml`, `kaggle.yaml`, `ci.yaml`)
  * `data/` (`kaggle.yaml`, `smoke.yaml`, `science.yaml`)
  * `calib/` (`nominal.yaml`, `dev.yaml`) — ADC, CDS, dark, flat, trace, phase.
  * `model/` (`v50.yaml`, ablations)
  * `training/` (`trainer.yaml`, `precision.yaml`, `num_workers/`, `accumulate_grad_batches/`)
  * `loss/` (`smoothness.yaml`, `nonneg.yaml`, `molecular.yaml`, `composite.yaml`)
  * `logger/` (`jsonl.yaml`, `wandb_off.yaml`)
  * `train.yaml`, `predict.yaml` (top‑level entry points)

* **Snapshot & hashing**: dump full merged config + compute content hash; persist to `artifacts/runs/<run_id>/config.yaml` and `run_hash.json`.

* **Guards**: forbid missing keys; require explicit defaults; pin seeds; assert deterministic data loaders for CI/Kaggle.

### 7.2 DVC

* **Pipeline file** `dvc.yaml`:

  * `calibrate`: inputs = raw + `configs/calib/*`; outs = `data/calib/`.
  * `train`: deps = tensors + `configs/model/*` + `configs/training/*`; outs = `artifacts/ckpts/`.
  * `predict`: deps = ckpt + eval tensors; outs = `artifacts/preds/`.
  * `submit`: deps = preds; outs = `artifacts/submissions/*.csv`.

* **Remotes**: default `localcache`; optional S3/SSH for team.

* **Locks & reproducibility**: commit `dvc.lock`; enforce `dvc repro` in CI before release.

---

## 8. Example Snippets

### 8.1 Hydra top‑level (`configs/train.yaml`)

```yaml
# Compose defaults from groups
defaults:
  - env: local
  - data: kaggle
  - calib: nominal
  - model: v50
  - training: trainer
  - loss: composite
  - logger: jsonl

# Inline tweaks allowed via CLI overrides
seed: 2025
```

### 8.2 DVC pipeline (`dvc.yaml`)

```yaml
stages:
  calibrate:
    cmd: spectramind calibrate --config-name train +phase=calib
    deps:
      - configs/calib
      - data/raw
    outs:
      - data/calib

  train:
    cmd: spectramind train --config-name train
    deps:
      - configs/model
      - configs/training
      - data/tensors
    outs:
      - artifacts/ckpts

  predict:
    cmd: spectramind predict --config-name predict
    deps:
      - artifacts/ckpts/best.ckpt
      - data/tensors_eval
    outs:
      - artifacts/preds

  submit:
    cmd: spectramind submit artifacts/preds --out artifacts/submissions
    deps:
      - artifacts/preds
    outs:
      - artifacts/submissions
```

### 8.3 Run manifest (written by CLI)

```json
{
  "project": "spectramind-v50",
  "run_id": "2025-09-06T21-55-00Z_local_v50_nominal",
  "config_hash": "a5b1…",
  "hydra_config_path": "artifacts/runs/…/config.yaml",
  "dvc_rev": "<git-commit-sha>",
  "stages": ["calibrate","train","predict"],
  "artifacts": {
    "ckpt": "artifacts/ckpts/best.ckpt",
    "preds": "artifacts/preds/val.csv"
  }
}
```

---

## 9. Risks & Mitigations

* **Risk:** Config sprawl → confusion.
  **Mitigation:** Config lint (pre‑commit), ADR‑driven patterns, group READMEs, examples.

* **Risk:** Hidden nondeterminism (randomness, worker races).
  **Mitigation:** pinned seeds; CI “determinism” job; enforce `torch.use_deterministic_algorithms(True)` where feasible.

* **Risk:** DVC remote contention/quotas.
  **Mitigation:** default local cache; scheduled cache GC; optional team remote with quotas.

* **Risk:** Kaggle no‑internet & 9h timeout.
  **Mitigation:** pack artifacts into dataset; small `smoke` configs; stage time budgets.

---

## 10. Consequences

* **Positive:** repeatable experiments; quick ablations; smaller PRs; audit‑friendly submissions; easy rollback.
* **Negative:** developer learning curve (Hydra/DVC); additional YAML/metadata discipline.

---

## 11. Compliance Checklist (CI gates)

* [ ] `spectramind doctor` passes (env, CUDA, seeds, file perms).
* [ ] `pre-commit run -a` clean (ruff/black/yamllint/detect‑secrets).
* [ ] `dvc status` clean; `dvc repro` succeeds on CI runner.
* [ ] Run manifest + config snapshot written and archived.
* [ ] `artifacts/submissions/*` includes 283‑bin μ/σ csv with schema.

---

## 12. How to Revisit

Revisit if any of the following occurs:

* Kaggle rules change (runtime, storage, dependency policies).
* Data modality changes (additional sensors) or calibration chain diverges.
* Team needs richer experiment UI → consider MLflow/W\&B *in addition* to DVC.

---

## 13. References & Further Reading

* Internal: `configs/*/ARCHITECTURE.md`, `docs/architecture/`, `Makefile`, `dvc.yaml`, `CONTRIBUTING.md`.
* External: Hydra docs, DVC docs, papers on reproducible ML in restricted environments.

---

## 14. FAQ

**Q:** Why not store models with Git LFS only?
**A:** LFS lacks pipeline semantics & reproducible dependency graphs; DVC encodes both DAG & cache policies.

**Q:** Can we use Hydra without DVC?
**A:** Yes, but you lose artifact lineage and deterministic stage recomputation; pairing is the win.

**Q:** How do we run quick smoke tests?
**A:** `+data=smoke +training=trainer_smoke +env=ci` with `dvc repro -s train`.

**Q:** How does this help reviews?
**A:** PRs show exact config deltas; CI proves the pipeline; manifests let reviewers reproduce locally.
