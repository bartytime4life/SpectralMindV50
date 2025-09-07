# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

Mission-grade, **CLI-first**, **Hydra-driven**, **DVC-tracked**, **Kaggle-ready** repo.
Physics-informed, neuro-symbolic pipeline for **multi-sensor fusion** (FGS1 photometry + AIRS spectroscopy) producing calibrated **μ/σ** over **283 spectral bins** (GLL-scored).

---

## ✨ Key Features

* **Dual-channel encoders**

  * FGS1 time-series → SSM/Mamba
  * AIRS spectra → CNN/GNN
  * Late fusion → **heteroscedastic** decoder (per-bin μ, σ)

* **Symbolic + physics constraints**
  Smoothness, non-negativity, molecular band priors via a rules DSL.

* **Uncertainty calibration**
  Post-train temperature scaling on σ (val split).

* **Diagnostics**
  Inject-and-recover tests (e.g., CO₂ bands, white-light drift) with an HTML dashboard.

* **Reproducibility**
  Hydra snapshots, DVC lineage, deterministic kernels, artifact checksums.

* **Kaggle-safe**
  Slim deps, **no internet**, ≤ **9h** wallclock guardrails.

---

## 📦 Requirements

* Python **3.10+**
* Git, DVC (with a remote configured), Make
* GPU recommended (CUDA 11.x+)

```bash
# Dev environment (recommended)
make dev            # installs dev deps, pre-commit, pytest
pre-commit install  # optional, enable hooks on commit
```

> Tip: For Kaggle notebooks, use `requirements-kaggle.txt` only.

---

## 🗺️ Repository (essentials)

```
.
├─ configs/              # Hydra configs (env, data, model, loss, train, search, submit)
├─ data/                 # DVC-tracked (raw/interim/processed)
├─ artifacts/            # checkpoints, reports, predictions, config snapshots
├─ dist/                 # packaged submissions
├─ scripts/              # helper scripts (validate, render_diagrams, submit, etc.)
├─ src/                  # spectramind package (cli/, models/, pipeline/, diagnostics/)
└─ dvc.yaml              # calibrate → preprocess → train → predict → diagnose → submit
```

---

## 🚀 Quickstart

### 0) DVC remote

```bash
# Example: local (adjust to your infra)
dvc remote add -d localcache path/to/cache
dvc push
```

### 1) End-to-end (DVC)

```bash
dvc repro
# Runs: calibrate → preprocess → train → predict → diagnose → (optional) submit
```

### 2) Stage-by-stage (CLI)

```bash
spectramind calibrate --config-name calibrate +env=local +calib=nominal
spectramind train     --config-name train     +model=v50 +data=kaggle
spectramind predict   --config-name predict   ckpt=artifacts/checkpoints/model.ckpt
spectramind diagnose report --out artifacts/reports/diagnostics_dashboard.html
spectramind submit    --config-name submit    inputs.pred_path=artifacts/predictions/mu.csv
```

### 3) Hydra overrides & sweeps

```bash
# One run with overrides
spectramind train --config-name train \
  +env=local +data=kaggle +model=v50 \
  loss.smoothness.lam=5e-4 loss.symbolic.enabled=true

# Grid sweep (search defs in configs/search/)
spectramind train --multirun +search=encoder_depth,bins
```

### 4) Uncertainty calibration (σ)

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

| Stage      | CLI                     | DVC out                                        |
| ---------- | ----------------------- | ---------------------------------------------- |
| calibrate  | `spectramind calibrate` | `data/interim/calibrated/`                     |
| preprocess | *(internal)*            | `data/processed/tensors/`                      |
| train      | `spectramind train`     | `artifacts/checkpoints/model.ckpt`             |
| predict    | `spectramind predict`   | `artifacts/predictions/{mu.csv,sigma.csv}`     |
| diagnose   | `spectramind diagnose`  | `artifacts/reports/diagnostics_dashboard.html` |
| submit     | `spectramind submit`    | `dist/submission.{zip,json,csv}`               |

---

## ⚙️ Configuration (Hydra)

All runtime parameters live in `configs/` (env, data, model, training, loss, logger, search).

```bash
# Compose defaults + overrides
spectramind train --config-name train +env=kaggle +data=kaggle +model=v50
# Snapshot saved at: artifacts/configs/run.yaml
```

**Determinism:** seeds, cudnn flags, and rank-safe samplers are set by default in `configs/env/*`.

---

## 📐 Losses & Physics Constraints

Enable/scale via Hydra (`configs/loss/constraints.yaml`):

```yaml
loss:
  smoothness: {enabled: true, lam: 1e-3}
  band_priors:
    enabled: true
    bands: [[130,145],[190,205]]   # index ranges
    weight: 1e-3
  symbolic:
    enabled: true
    rules:
      - {name: nonneg_mu,     target: mu,    expr: 'x >= 0',   weight: 1.0}
      - {name: bounded_sigma, target: sigma, expr: 'abs(x) < 5.0', weight: 0.1}
```

---

## 🔬 Diagnostics

* **Inject** synthetic signals (e.g. CO₂ bands, white-light drift).
* **Recover** : verify detection/neutralization in predictions.
* **Report** : HTML dashboard → `artifacts/reports/diagnostics_dashboard.html`.

```bash
spectramind diagnose report --out artifacts/reports/diagnostics_dashboard.html
pytest -q tests/diagnostics
```

---

## 🧪 Metric & Notes

* Competition metric: **Gaussian Log-Likelihood (GLL)** on 283 bins (FGS1 “white-light” bin carries high weight).
* Penalizes over-confident σ → **calibrate** predicted uncertainties (`diagnose calibration`).

---

## 📦 Submission

Schema-checked and reproducible packaging:

```bash
spectramind submit --config-name submit \
  inputs.pred_path=artifacts/predictions/mu.csv \
  inputs.sigma_path=artifacts/predictions/sigma.csv
bash scripts/validate_submission.sh dist/submission.json
```

**Bundle includes** config snapshot, checksums, and manifest for audit.

---

## 🧭 Principles

* **Reproducible** : seeds, Hydra snapshots, DVC lineage, checksums.
* **Physics-aware** : smoothness, non-negativity, band coherence, symbolic rules.
* **Kaggle-ready** : slim deps, ≤ 9h runtime, no internet.
* **Auditable** : JSONL logs, SBOM in CI, schema-valid submissions.

---

## 🛠️ Make Targets

```bash
make dev     # install dev deps
make test    # pytest
make bench   # quick ablation / nightly CI
make docs    # MkDocs build
make clean   # clear caches/artifacts
```

---

## 📝 Kaggle Notes

* Always set `+env=kaggle +data=kaggle`; outputs write to `/kaggle/working`.
* Use `requirements-kaggle.txt` (no extras).
* Prefer `spectramind predict` + light diagnostics; disable heavy FFT/UMAP in competition runs.
* Respect **9h GPU** wallclock / **30 GB** RAM.

---

## 🧯 Troubleshooting

* **OOM during train** → reduce `data.loader.num_workers`, batch size, or enable gradient checkpointing in `configs/model/*`.
* **DVC missing remote** → run `dvc remote list` and `dvc remote add -d <name> <url>`; then `dvc push`.
* **Non-determinism** → set `+env=<profile>` that forces cudnn deterministic kernels; ensure cudatoolkit matches driver.
* **Slow GLL** → verify σ calibration and disable over-tight priors; check FGS1 handling.

---

## 🤝 Contributing

* PRs welcome. Run `make test` locally and keep docs/code in sync.
* Style: `ruff + black`, type hints where public.
* Add/update Hydra schemas + example configs for new features.

---

## 🔐 Security

* No dynamic code exec in configs.
* All submissions validated via JSON schema + checksum manifest.
* SBOM generated in CI; pinned deps in `requirements*.txt`.

---

## 📄 License

MIT (see `LICENSE`).

---

## 🗺️ Acknowledgements

Thanks to the Ariel community and the broader OSS ecosystem (Hydra, DVC, Typer, PyTorch) that makes mission-grade research pipelines possible.

---
