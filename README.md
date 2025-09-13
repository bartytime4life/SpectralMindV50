# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

Mission-grade, **CLI-first**, **Hydra-driven**, **DVC-tracked**, **Kaggle-ready** repo.  
Physics-informed, neuro-symbolic pipeline for **multi-sensor fusion** (FGS1 photometry + AIRS spectroscopy) producing calibrated **μ/σ** over **283 spectral bins** (scored by Gaussian Log-Likelihood, GLL).

---

## ✨ Key Features

- **Dual-channel encoders**
  - FGS1 time-series → SSM/Mamba
  - AIRS spectra → CNN/GNN
  - Late fusion → **heteroscedastic** decoder (per-bin μ, σ)

- **Symbolic + physics constraints**  
  Smoothness, non-negativity, and molecular band priors via a rules DSL.

- **Uncertainty calibration**  
  Post-train temperature scaling for σ (validated on `val`).

- **Diagnostics**  
  Inject-and-recover tests (e.g., CO₂ bands, white-light drift) + HTML dashboard.

- **Reproducibility**  
  Hydra snapshots, DVC lineage, deterministic seeds, artifact checksums.

- **Kaggle-safe**  
  Slim deps, **no internet**, **≤9h** guardrails.

---

## 🧩 Pipeline (high level)

```mermaid
flowchart LR
  R[Raw inputs] --> C[Calibrate]
  C --> P[Preprocess]
  P --> T[Train]
  T --> I[Predict]
  I --> D[Diagnose]
  I --> S[Submit]
  classDef n fill:#0f766e,color:#fff,stroke:#0f766e,stroke-width:1px
  class R,C,P,T,I,D,S n
````

* **Calibrate**: ADC/dark/flat/trace/phase/photometry
* **Preprocess**: mask → detrend → normalize → bin/window → pack → tokenize → export
* **Train**: dual encoders + fusion decoder (μ, σ)
* **Predict**: submission-ready μ/σ
* **Diagnose**: metrics, plots, HTML report; σ calibration
* **Submit**: schema-checked ZIP + manifest

---

## 📦 Requirements

* Python **3.11+**
* Git, DVC (remote configured), Make
* NVIDIA GPU recommended (CUDA 11.x+)

```bash
# Dev environment
make dev            # venv + deps + hooks
pre-commit install  # optional (enable hooks on commit)
```

> Kaggle: use only `requirements-kaggle.txt` (no extras). Outputs write to `/kaggle/working`.

---

## 🗺️ Repository (essentials)

```
.
├─ configs/              # Hydra configs: env, data, preprocess, model, training, loss, logger
├─ data/                 # DVC-tracked: raw/ calibrated/ processed/
├─ artifacts/            # ckpt, predictions, diagnostics, scaler stats, manifests
├─ dist/                 # packaged submissions
├─ scripts/              # helpers (render_diagrams, validate_submission, etc.)
├─ src/                  # spectramind/ (cli, models, pipeline, diagnostics, validators)
├─ dvc.yaml              # calibrate → preprocess → train → predict → diagnose → submit
└─ Makefile              # mission-grade targets (incl. preset shortcuts)
```

---

## 🚀 Quickstart

### 0) Configure DVC remote

```bash
# Example local cache (adjust for S3/GDrive/…)
dvc remote add -d localcache ./dvc-remote
dvc push
```

### 1) End-to-end (DVC)

```bash
dvc repro
# Runs: calibrate → preprocess → train → predict → diagnose (and you can run submit)
```

### 2) Preprocess presets (Hydra)

```bash
# CI/Kaggle budget
make preprocess.fast    SPLIT=train
# Balanced default
make preprocess.nominal SPLIT=val
# Research-grade (Parquet+Zstd, overlap, stricter checks)
make preprocess.strict  SPLIT=test
```

### 3) Stage-by-stage (CLI)

```bash
spectramind calibrate  --config-name calibrate
spectramind preprocess --config-name preprocess +defaults='[/preprocess/presets/nominal]' split=train
spectramind train      --config-name train     +model=v50
spectramind predict    --config-name predict   ckpt=artifacts/ckpt.pt
spectramind diagnose   --config-name diagnose  inputs.pred_path=artifacts/predictions/preds.csv
spectramind submit     --config-name submit    inputs.pred_path=artifacts/predictions/preds.csv
```

### 4) Hydra overrides & sweeps

```bash
# One run with overrides
spectramind train --config-name train \
  +env=local +data=default +model=v50 \
  loss.smoothness.lam=5e-4 loss.symbolic.enabled=true

# Grid sweep (see configs/search/)
spectramind train --multirun +search=encoder_depth,bins
```

### 5) Uncertainty calibration (σ)

```bash
spectramind diagnose calibration --dataset val --ckpt artifacts/ckpt.pt
```

---

## 🔄 DVC Stages & Outputs

| Stage      | CLI                      | DVC output                                     |
| ---------- | ------------------------ | ---------------------------------------------- |
| calibrate  | `spectramind calibrate`  | `data/calibrated/` (persist)                   |
| preprocess | `spectramind preprocess` | `data/processed/` (persist)                    |
| train      | `spectramind train`      | `artifacts/ckpt.pt` + `artifacts/metrics.json` |
| predict    | `spectramind predict`    | `artifacts/predictions/preds.csv`              |
| diagnose   | `spectramind diagnose`   | `artifacts/diagnostics/report.html` (+ plots)  |
| submit     | `spectramind submit`     | `dist/submission.zip`                          |

---

## ⚙️ Configuration (Hydra)

* Root config: `configs/config.yaml` (groups at root, Hydra settings in `hydra:`).
* Presets: `configs/preprocess/presets/{fast,nominal,strict}.yaml`
* Methods: `configs/preprocess/method/*.yaml` (load, mask, detrend, normalize, binning, window, pack, tokenize, export)

```bash
spectramind train --config-name train +env=local +data=default +model=v50
# Snapshot at: artifacts/configs/run.yaml
```

**Determinism:** seeds, cudnn flags, and rank-safe samplers are set in `configs/env/*`.

---

## 📐 Losses & Physics Constraints (example)

```yaml
loss:
  smoothness: { enabled: true, lam: 1e-3 }
  band_priors:
    enabled: true
    bands: [[130,145],[190,205]]   # index ranges
    weight: 1e-3
  symbolic:
    enabled: true
    rules:
      - { name: nonneg_mu,     target: mu,    expr: 'x >= 0',        weight: 1.0 }
      - { name: bounded_sigma, target: sigma, expr: 'abs(x) < 5.0',  weight: 0.1 }
```

---

## 🔬 Diagnostics

* **Inject** synthetic signals; **recover** expected effects.
* **Report**: `artifacts/diagnostics/report.html`.

```bash
spectramind diagnose report --out artifacts/diagnostics/report.html
pytest -q -m "not slow"
```

---

## 🧪 Metric & Notes

* Competition metric: **GLL** over 283 bins. Over-confident σ is penalized → tune/calibrate σ.
* FGS1 “white-light” handling is pivotal; verify pipeline settings in preprocess configs.

---

## 📦 Submission

Schema-checked, reproducible packaging with manifest:

```bash
spectramind submit --config-name submit \
  inputs.pred_path=artifacts/predictions/preds.csv
bash scripts/validate_submission.sh dist/submission.json
```

---

## 🛠️ Make Targets

```bash
make dev            # venv + dev deps + pre-commit
make check          # precommit + lint + type + tests
make preprocess.*   # fast | nominal | strict (SPLIT, OVERRIDES supported)
make docs           # MkDocs (renders Mermaid first if scripts present)
make scan           # SBOM + pip-audit + linters (local, non-failing)
make clean          # caches/builds
```

> Preset shortcuts accept `OVERRIDES="k=v k=v"` for Hydra keys:
> `make preprocess.nominal SPLIT=train OVERRIDES="io.format=npz preprocess/window.center_transit=false"`

---

## 📝 Kaggle Notes

* Use `+env=kaggle +data=kaggle` profiles; outputs write to `/kaggle/working`.
* Prefer `spectramind predict` + light diagnostics; turn off heavy FFT/UMAP in comp runs.
* Respect **≤9h GPU** wallclock and **≤30 GB** RAM.

---

## 🧯 Troubleshooting

* **OOM during train** → reduce workers, batch size; enable grad-checkpointing in model config.
* **DVC remote missing** → `dvc remote list`, then `dvc remote add -d <name> <url>` and `dvc push`.
* **Drift/instability** → use `preprocess/strict.yaml` and check `normalize` stats + `window` phase alignment.
* **σ too sharp** → run `diagnose calibration` or relax symbolic bounds.

---

## 🤝 Contributing

* PRs welcome. Run `make check` locally; keep docs and configs in sync.
* Style: Ruff + Black, NumPy docstrings, type hints on public APIs.
* Add/update Hydra schemas + example configs for new features.

---

## 🔐 Security

* No dynamic code exec in configs.
* Submissions validated by JSON schema + checksum manifest.
* SBOM generated in CI; pinned deps in `requirements*.txt`.

---

## 📄 License

MIT (see `LICENSE`).

---

## 🙏 Acknowledgements

Thanks to the Ariel community and the OSS stack (Hydra, DVC, Typer, PyTorch).

```
