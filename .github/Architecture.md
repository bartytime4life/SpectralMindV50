# SpectraMind V50 — Architecture

**Mission-grade, CLI-first, Hydra-driven, DVC-tracked, Kaggle-ready** repository for the NeurIPS 2025 **Ariel Data Challenge**.
Physics-informed, neuro-symbolic pipeline for **dual-sensor fusion** (FGS1 photometry + AIRS spectroscopy) producing calibrated **μ/σ** over **283** spectral bins.

---

## Design Principles

* **CLI-first UX**: Single entrypoint (`spectramind`) with subcommands for `calibrate`, `preprocess`, `train`, `predict`, `diagnose`, `submit`.
* **Config-as-code**: All runtime parameters live in **Hydra** YAML; no magic constants.
* **Reproducibility**: Pipeline stages defined in **DVC** with explicit inputs/outputs and cached artifacts.
* **Physics-informed ML**: Composite losses (GLL + smoothness + nonnegativity + band coherence), dual encoders for FGS1/AIRS.
* **Explainability & Lineage**: Deterministic diagnostics (FFT, UMAP, SHAP) + per-run manifests, config snapshots, hashes.
* **Governance**: Artifact retention, archival, and provenance policy (see ADR-0009).
* **Kaggle-safe**: Offline execution mode, slim deps, time/space-aware stages.

---

## High-Level Pipeline

```mermaid
flowchart TD
  A[Raw Inputs<br/>FGS1 & AIRS] --> B[Calibrate<br/>(ADC, dark, flat, trace, phase, photometry)]
  B --> C[Preprocess<br/>tensors, masks, splits]
  C --> D[Train<br/>dual encoders + physics-informed loss]
  D --> E[Predict<br/>μ, σ for 283 bins]
  E --> F[Diagnose<br/>FFT, UMAP, SHAP, lineage]
  E --> G[Submit<br/>CSV/ZIP bundle]
  subgraph Tracking
    H[DVC cache & remotes]
    I[Hydra config snapshots]
    J[Run manifests (JSONL)]
  end
  B -. outputs/.. .-> H
  C -. features/.. .-> H
  D -. ckpt/metrics .-> H
  E -. preds/.. .-> H
  F -. reports/.. .-> H
  D -. snapshot .-> I
  E -. events .-> J
```

**Stages** (DVC): `calibrate` → `preprocess` → `train` → `predict` → `diagnose` → `submit`.
Each stage has clear input/outputs and is independently invocable via CLI + Hydra overrides.

---

## Repository Topography (authoritative)

```
spectramind-v50/
├─ src/spectramind/           # library & CLI entry
│  ├─ cli.py                  # Typer app (subcommands)
│  ├─ data/                   # datamodule, datasets, collate
│  ├─ calib/                  # calibration ops
│  ├─ preprocess/             # tensorization, masks, splits
│  ├─ models/                 # fgs1_encoder, airs_encoder, fusion, heads
│  ├─ losses/                 # gll, smoothness, nonneg, coherence
│  ├─ diagnose/               # fft, umap, shap, report builder
│  └─ utils/                  # io, hashing, schema, timing, seed
├─ configs/                   # Hydra configs (composable)
│  ├─ train.yaml, predict.yaml, diagnose.yaml, submit.yaml
│  ├─ env/ (local|kaggle)     # env-specific overrides
│  ├─ data/, calib/, model/, training/, loss/, logger/
├─ dvc.yaml                   # stages & artifacts
├─ .github/workflows/         # CI/CD (see below)
├─ ADR/                       # architecture decision records
└─ outputs/                   # DVC-tracked artifacts (ckpt, preds, reports)
```

---

## Configuration & Hydra

* **Defaults**: Composed from `configs/**`.
* **Overrides**: On CLI via `+key=value` or `group=name` (e.g. `python -m spectramind train env=kaggle +training.epochs=8`).
* **Snapshots**: Each run materializes a **config snapshot** (YAML) in the run folder + logged to lineage.

---

## Data & DVC

* **DVC** declares **each stage** with **deps** (code + data) and **outs** (artifacts).
* **Caching**: No re-run if inputs/params unchanged; speeding iteration.
* **Remotes**: Use configured DVC remotes for long-term cache; Kaggle runs use pre-packaged datasets.
* **Artifacts**:

  * `calibrate/` → calibrated cubes (FGS1 & AIRS)
  * `preprocess/` → model-ready tensors + splits
  * `train/` → checkpoints, metrics (CSV/JSON)
  * `predict/` → predictions (μ, σ per sample)
  * `diagnose/` → `report.html`, plots, lineage manifests
  * `submit/` → validated CSV/ZIP

---

## Model Architecture (summary)

* **Dual Encoders**:

  * *FGS1 branch*: temporal encoder (e.g., SSM/Mamba or light-weight transformer) focusing on broadband transit & baseline.
  * *AIRS branch*: spectral encoder (CNN/GNN hybrid) capturing correlated bands across 283 channels.
* **Fusion**: Cross-attention (or equivalent) to align photometric events with spectral dynamics.
* **Heads**: Heteroscedastic regression → **μ** & **σ** per wavelength.
* **Loss**:

  * Gaussian Log-Likelihood (FGS1 up-weighted)
  * Smoothness (second-difference)
  * Non-negativity (soft hinge)
  * Band-coherence (local correlation prior)
  * Optional calibration regularizers

---

## Diagnostics & Explainability

* **FFT**: Residual periodicities (jitter/thermal drift fingerprinting).
* **UMAP**: Latent embeddings to inspect regime clusters & OOD.
* **SHAP**: Feature attributions (summary/butterfly plots) for sanity checking.
* **Lineage Report**: One-click `report.html` bundling: config snapshot, DVC hashes, commit SHA, metrics table, plots.
* **Determinism**: Headless, seed-controlled, CI-invokable (`spectramind diagnose`).

---

## Artifact Governance

* **Ephemeral**: Scratch tensors & debug plots auto-swept after CI job.
* **Retained (30d)**: Active experiments’ checkpoints, metrics, diagnostics (DVC cache).
* **Archived (long-term)**: Accepted submissions, release ckpts, milestone diagnostics — tagged & pushed under DVC with provenance.
* **Provenance**: Each retained/archived artifact must include config snapshot, DVC lock/hash, git SHA, JSONL events manifest.

See **ADR-0009 — Artifact Retention & Governance** for exact policy and compliance gates.

---

## CI/CD Workflows

Located in `.github/workflows/`:

* **`ci.yml`**: lint, type-check, unit tests, minimal pipeline smoke (CPU), build diagnostics on a small slice.
* **`kaggle_notebook_ci.yml`**: build & validate Kaggle submission bundle (offline-safe check, size/time guards).
* **`release.yml`**: tag release, push DVC artifacts (archival tier), attach CHANGELOG, produce SBOM.
* **`sbom-refresh.yml`**: refresh CycloneDX/SPDX, dependency audits.
* **`docs.yml`**: build/validate docs, render Mermaid diagrams, link ADRs.

```mermaid
flowchart LR
  dev[PR / branch] --> ci[CI: lint/test/smoke]
  ci -->|pass| main[(main)]
  main --> release[Release CI]
  release --> archive[Archive artifacts (DVC push + tag)]
  main --> kaggle[Kaggle CI]
  kaggle --> ok[Submission bundle OK]
```

**Branching**: PRs to `main` must pass CI + include ADR updates when architectural changes occur.

---

## Environments

* **Local**: Full pipeline; GPU optional.
* **CI**: Minimal deterministic slices; artifact sweeper enforced.
* **Kaggle**: Offline; `env=kaggle` config redirects paths, disables internet, uses slim deps & pre-packaged data.

---

## Naming & Paths (conventions)

* Runs: `runs/{stage}/{YYYYmmdd_HHMMSS}_{gitsha[:7]}_{hydra_job_id}/`
* Artifacts: under `outputs/{stage}/…` (DVC tracked)
* Manifests: `outputs/diagnostics/run_manifest.jsonl`, `config_snapshot.yaml`
* Submissions: `outputs/submit/submission.csv` (+ checksum)

---

## ADRs (key)

* **ADR-0002**: Composite Physics-Informed Loss
* **ADR-0004**: Dual Encoder Fusion (FGS1 + AIRS)
* **ADR-0008**: Explainability & Diagnostics (FFT, UMAP, SHAP, lineage)
* **ADR-0009**: Artifact Retention & Governance

> Keep ADR index up-to-date and reflect dependencies in the Mermaid graph in `ADR/README.md`.

---

## Reproduce & Verify (quick checks)

* **End-to-end (local)**:

  ```bash
  python -m spectramind calibrate
  python -m spectramind preprocess
  python -m spectramind train
  python -m spectramind predict
  python -m spectramind diagnose
  python -m spectramind submit
  ```
* **Determinism**: Set `seed` in Hydra; CI checks hash stability on minimal slices.
* **Lineage**: Every prediction row traceable via run folder → config snapshot → DVC lock → git SHA.

---

## Security / SBOM

* Build CycloneDX/SPDX SBOMs in CI, pin core deps, forbid internet in Kaggle mode, scan wheels & licenses.
* Optional: `pip-audit`, `trivy` integration as gates in `ci.yml`.

---

## Contact / Contributing

* Open PRs with focused commits and updated ADRs/diagrams.
* Use `+key=value` Hydra overrides instead of code edits for experiments.
* Keep DVC outs small & atomic; prefer multi-stage granularity for cache hits.

---

*Last updated: 2025-09-13*
