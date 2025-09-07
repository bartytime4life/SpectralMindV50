# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

Mission-grade, **CLI-first**, **Hydra-driven**, **DVC-tracked**, **Kaggle-ready** repo.  
Physics-informed, neuro-symbolic pipeline for **multi-sensor fusion** (FGS1 photometry + AIRS spectroscopy) producing calibrated **μ/σ** over **283 spectral bins**.

---

## ✨ Key Features

- **Dual-channel encoders**:  
  - FGS1 time-series → SSM/Mamba  
  - AIRS spectra → CNN/GNN  
  - Fused heteroscedastic decoder (μ, σ)  

- **Symbolic & physics constraints**: smoothness, non-negativity, molecular band priors (rules DSL).  
- **Uncertainty calibration**: post-train temperature scaling on σ (val split).  
- **Diagnostics**: inject-and-recover tests (e.g. CO₂ bands, photometric drift) with HTML dashboard.  
- **Reproducibility**: Hydra config snapshots, DVC lineage, deterministic kernels.  
- **Kaggle-safe**: slim deps, no internet, ≤ 9h guardrails.  

---

## 🚀 Quickstart

### 0) Environment
```bash
# Python 3.10+
make dev            # dev deps + pre-commit + pytest
pre-commit install  # optional, hooks on commit
````

### 1) End-to-end (DVC)

```bash
dvc repro
# Runs: calibrate → preprocess → train → predict → diagnose
```

### 2) Stage-by-stage (CLI)

```bash
spectramind calibrate --config-name calibrate +env=local +calib=nominal
spectramind train     --config-name train     +model=v50 +data=kaggle
spectramind predict   --config-name predict   ckpt=artifacts/checkpoints/model.ckpt
spectramind diagnose report --out artifacts/reports/diagnostics_dashboard.html
spectramind submit    --config-name submit    inputs.pred_path=artifacts/predictions/mu.csv
```

### 3) Hydra overrides

```bash
spectramind train --config-name train \
  +model=v50 \
  +search=encoder_depth,bins \
  loss.smoothness.lam=5e-4 loss.symbolic.enabled=true

# Grid sweep
spectramind train --multirun +search=encoder_depth,bins
```

### 4) Uncertainty calibration

```bash
spectramind diagnose calibration \
  --dataset val \
  --ckpt artifacts/checkpoints/model.ckpt
```

### 5) Validate submission (CI gate)

```bash
bash scripts/validate_submission.sh dist/submission.json
```

---

## 🔄 Pipeline Stages

| Stage      | CLI                     | DVC outs                                       |
| ---------- | ----------------------- | ---------------------------------------------- |
| calibrate  | `spectramind calibrate` | `data/interim/calibrated/`                     |
| preprocess | *(internal)*            | `data/processed/tensors/`                      |
| train      | `spectramind train`     | `artifacts/checkpoints/model.ckpt`             |
| predict    | `spectramind predict`   | `artifacts/predictions/{mu.csv,sigma.csv}`     |
| diagnose   | `spectramind diagnose`  | `artifacts/reports/diagnostics_dashboard.html` |
| submit     | `spectramind submit`    | `dist/submission.{zip,json,csv}`               |

---

## ⚙️ Configuration (Hydra)

* All runtime parameters live in `configs/` (env, data, model, training, loss, logger, search).
* Compose with defaults + CLI overrides:

```bash
spectramind train --config-name train +env=kaggle +data=kaggle +model=v50
```

* Snapshot saved under `artifacts/configs/run.yaml`.

---

## 📐 Losses & Physics Constraints

Configurable via Hydra (`configs/loss/constraints.yaml`):

```yaml
loss:
  smoothness: {enabled: true, lam: 1e-3}
  band_priors:
    enabled: true
    bands: [[130,145],[190,205]]
    weight: 1e-3
  symbolic:
    enabled: true
    rules:
      - {name: nonneg_mu, target: mu, expr: 'x >= 0', weight: 1.0}
      - {name: bounded_sigma, target: sigma, expr: 'abs(x) < 5.0', weight: 0.1}
```

---

## 🔬 Diagnostics

* **Inject**: synthetic bands (e.g. CO₂), photometric drifts.
* **Recover**: verify detection/neutralization in predictions.
* **Report**: HTML dashboard → `artifacts/reports/diagnostics_dashboard.html`.

```bash
spectramind diagnose report --out artifacts/reports/diagnostics_dashboard.html
pytest -q tests/diagnostics
```

---

## 📦 Submission

* **Schema validation**: JSON schema enforced pre-package.
* **Packaging**: config snapshot + checksums.

```bash
spectramind submit --config-name submit inputs.pred_path=artifacts/predictions/mu.csv
bash scripts/validate_submission.sh dist/submission.json
```

---

## 🧭 Principles

* **Reproducible**: seeds, Hydra snapshots, DVC lineage, artifact checksums.
* **Physics-aware**: smoothness, non-negativity, band coherence, symbolic rules.
* **Kaggle-ready**: slim deps, ≤ 9h runtime, no internet.
* **Auditable**: JSONL logs, CI SBOM, schema-valid submissions.

---

## 🛠️ Make Targets

```bash
make dev     # install dev deps
make test    # pytest suite
make bench   # quick ablation / nightly CI
make docs    # build MkDocs
make clean   # clear caches/artifacts
```

---

## 📝 Kaggle Notes

* Always set `+env=kaggle +data=kaggle`; outputs → `/kaggle/working`.
* Use `requirements-kaggle.txt`.
* Disable heavy diagnostics.
* Respect 9h GPU wallclock + 30 GB RAM.

---

```
