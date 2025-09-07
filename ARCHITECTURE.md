# 🛰️ SpectraMind V50 — Architecture Overview

Mission-grade, physics-informed, neuro-symbolic pipeline for the **NeurIPS 2025 Ariel Data Challenge**.
Implements multi-sensor fusion of **FGS1 time-series** + **AIRS spectroscopy**, producing calibrated per-bin **μ/σ predictions over 283 spectral channels**.

---

## 1. CLI-First Orchestration

* Single entrypoint: `spectramind` (Typer-based CLI).
* Subcommands:

  * `calibrate` → raw telescope → calibrated cubes
  * `train` → encoders + decoder optimization
  * `predict` → model checkpoint → submission outputs
  * `diagnose` → reproducibility/debug tools
  * `submit` → Kaggle/CI-ready artifacts

✅ **Design principle:** all pipeline logic exposed via CLI, Hydra config injection only (no hardcoded params).

---

## 2. Hydra Configuration System

* **Config groups:**

  * `env/` → runtime (local, Kaggle, HPC, CI)
  * `data/` → ingestion, calibration, preprocessing
  * `calib/` → ADC → dark → flat → CDS → photometry → trace → phase
  * `model/` → encoders, fusion, decoder
  * `training/` → optimizers, schedulers, precision, workers
  * `loss/` → smoothness, non-negativity, coherence, uncertainty terms
  * `logger/` → JSONL event logs, tensorboard, wandb

✅ **Principle:** *one config = one experiment snapshot*. Hashes + JSON schema enforce reproducibility.

---

## 3. Data Flow & Processing Stages

```
raw (FGS1 + AIRS)  
   ↓ calib (ADC → dark → flat → CDS → photometry → trace → phase)  
   ↓ tensors (DVC-tracked, physics-aligned)  
   ↓ encoders  
      • FGS1: state-space / sequence encoder (e.g. Mamba SSM)  
      • AIRS: CNN/GNN spectral encoder  
   ↓ fusion (cross-modal alignment)  
   ↓ decoder → spectral μ/σ (283 bins)  
   ↓ submission (CSV, schema-validated)  
```

✅ **Physics-aware guarantees:** temporal alignment, smoothness priors, spectral band coherence.

---

## 4. Model Architecture

* **Dual encoders:**

  * `fgs1_encoder.py` → denoise, bin, downsample, transit-aware sequence model.
  * `airs_encoder.py` → wavelength-structured CNN/GNN.
* **Fusion:** cross-attention or concatenation into latent joint space.
* **Decoder:** heteroscedastic regression head → mean & variance per bin.
* **Constraints:** enforced via symbolic loss engine (non-negativity, smoothness, coherence).

✅ **Symbolic/physics overlays** keep outputs interpretable & leaderboard-safe.

---

## 5. Reproducibility & Lineage

* **DVC pipelines:** each stage (calib → features → train → predict → submit) hashed & cached.
* **Event logging:** structured JSONL (`events.schema.json`).
* **Snapshot manifests:** config hash + git commit + artifact digest.
* **CI/CD:** Kaggle notebook smoke tests, SBOM refresh, auto-linting, reproducibility checks.

✅ *Nothing is untracked — every artifact, config, and run is lineage-linked.*

---

## 6. Error Handling & UX

* Rich Typer CLI (with autocompletion, colored errors).
* All runtime errors (missing inputs, mis-keyed configs) fail **loud, typed, and user-friendly**.
* DVC guardrails: prevent 9h Kaggle runtime overruns.
* Pre-commit hooks enforce formatting, typing, SBOM compliance.

---

## 7. Alignment with Challenge Constraints

* **≤ 9h runtime on Kaggle GPUs.**
* **No internet dependencies** (offline-safe bootstrap via `bin/kaggle-boot.sh`).
* **283 spectral bins** enforced via schema validation.
* **μ/σ outputs** with calibrated uncertainty (heteroscedastic regression).

---

⚡ **In summary:** SpectraMind V50 architecture is a *clean, reproducible, symbolic+neural system*. CLI-first orchestration, Hydra configs, DVC lineage, and dual-channel physics-informed modeling ensure **scientific fidelity + competition performance**.
