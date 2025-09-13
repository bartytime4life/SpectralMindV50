# 🛰️ SpectraMind V50 — Architecture

Mission-grade, **CLI-first, Hydra-driven, DVC-tracked, Kaggle-ready** repository for the NeurIPS 2025 **Ariel Data Challenge**.
Physics-informed, neuro-symbolic pipeline for **dual-sensor fusion** (FGS1 photometry + AIRS spectroscopy) producing calibrated **μ/σ** over **283** spectral bins.

---

## 📐 Design Principles

* **CLI-first UX**: Single entrypoint (`spectramind`) with subcommands `calibrate`, `preprocess`, `train`, `predict`, `diagnose`, `submit`.
* **Config-as-code**: All runtime parameters live in Hydra YAML.
* **Reproducibility**: Stages defined in DVC with explicit inputs/outputs and cached artifacts.
* **Physics-informed ML**: Composite loss (GLL + smoothness + nonnegativity + band coherence). Dual encoders for FGS1/AIRS.
* **Explainability & lineage**: Deterministic diagnostics (FFT, UMAP, SHAP) + manifests, config snapshots, hashes.
* **Governance**: Artifact retention & provenance policies (ADR-0009).
* **Kaggle-safe**: Offline mode, slim deps, resource-aware stages.

---

## 🔄 High-Level Pipeline

```mermaid
flowchart TD
  A[Raw Inputs<br/>FGS1 + AIRS] --> B[Calibrate<br/>ADC, dark, flat, trace, phase]
  B --> C[Preprocess<br/>tensor packs, binning, splits]
  C --> D[Train<br/>dual encoders + physics loss]
  D --> E[Predict<br/>μ, σ (283 bins)]
  E --> F[Diagnose<br/>FFT, UMAP, SHAP]
  E --> G[Submit<br/>CSV + ZIP]

  subgraph Tracking
    H[DVC cache & remotes]
    I[Hydra config snapshots]
    J[Run manifests<br/>(JSONL)]
  end

  B --> H
  C --> H
  D --> H
  E --> H
  F --> H
  D --> I
  E --> J

```

**Stages (DVC):** `calibrate → preprocess → train → predict → diagnose → submit`.

---

## 📂 Repository Topography

```text
spectramind-v50/
├─ src/spectramind/        # library & CLI entry
│  ├─ cli.py               # Typer app
│  ├─ data/                # datamodules, loaders
│  ├─ calib/               # calibration ops
│  ├─ preprocess/          # tensorization, masks
│  ├─ models/              # fgs1_encoder, airs_encoder, fusion, decoder
│  ├─ losses/              # gll, smoothness, nonneg, coherence
│  ├─ diagnose/            # fft, umap, shap, reports
│  └─ utils/               # io, hashing, schema, seed
├─ configs/                # Hydra configs
│  ├─ train.yaml, predict.yaml, diagnose.yaml, submit.yaml
│  ├─ env/{local,kaggle}.yaml
│  ├─ data/, calib/, model/, training/, loss/, logger/
├─ dvc.yaml                # pipeline stages
├─ ADR/                    # architecture decision records
├─ .github/workflows/      # CI/CD
├─ assets/diagrams/        # Mermaid pipeline & ADR graphs
└─ outputs/                # DVC-tracked artifacts
```

---

## 🧠 Model Architecture

* **FGS1 encoder** → temporal SSM (Mamba or transformer).
* **AIRS encoder** → CNN/GNN hybrid across 283 channels.
* **Fusion** → cross-attention fusion.
* **Heads** → heteroscedastic regression → μ & σ.
* **Loss** → GLL (FGS1 up-weighted) + smoothness + nonnegativity + coherence.

---

## 📊 Diagnostics & Explainability

* **FFT**: residual periodicities.
* **UMAP**: latent clustering.
* **SHAP**: feature attribution.
* **Lineage reports**: config snapshot + DVC hashes + metrics table.

---

## 📦 Artifact Governance

* **Ephemeral**: CI scratch, debug plots swept.
* **Retained (30d)**: active ckpts + diagnostics in DVC cache.
* **Archived**: accepted submissions, tagged ckpts.
* **Provenance**: every artifact carries config snapshot + git SHA + JSONL manifest.

---

## ⚙️ CI/CD Workflows

* `ci.yml` → lint, type-check, unit tests, smoke pipeline.
* `kaggle_notebook_ci.yml` → offline submission bundle check.
* `release.yml` → DVC push + CHANGELOG + SBOM.
* `sbom-refresh.yml` → SPDX/CycloneDX refresh.
* `docs.yml` → build MkDocs + Mermaid diagrams.

```mermaid
flowchart LR
  dev[PR / branch] --> ci[CI: lint/test/smoke]
  ci -->|pass| main[(main)]
  main --> release[Release CI]
  release --> archive[Archive (DVC push + tag)]
  main --> kaggle[Kaggle CI]
  kaggle --> ok[Submission OK]
```

---

## ✅ ADR References

* **ADR-0002** — Composite Physics-Informed Loss
* **ADR-0004** — Dual Encoder Fusion
* **ADR-0008** — Explainability & Diagnostics (planned)
* **ADR-0009** — Artifact Retention & Governance (planned)

---

