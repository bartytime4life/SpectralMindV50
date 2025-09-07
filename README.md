# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

Mission-grade, **CLI-first**, **Hydra-driven**, **DVC-tracked**, **Kaggle-ready** repo. Physics-informed, neuro-symbolic pipeline for **multi-sensor fusion** (FGS1 + AIRS) producing calibrated **μ/σ** over **283 spectral bins**.

**Key features**

* **Dual-channel encoders**: FGS1 time-series (SSM/Mamba) + AIRS spectrum (CNN/GNN) → fused heteroscedastic decoder (μ, σ)
* **Symbolic/physics constraints**: smoothness, non-negativity, molecular band priors via a tiny rules DSL
* **Uncertainty calibration**: temperature scaling for σ (post-train, val-set)
* **Diagnostics**: inject-&-recover tests (CO₂ band, photometric trends) + HTML dashboard
* **Reproducibility**: Hydra config snapshots, DVC lineage, deterministic flags
* **Kaggle-safe**: slim deps, no internet assumptions, ≤ 9h runtime guardrails

---

## Quickstart

### 0) Environment

```bash
# Python 3.10+
make dev            # installs dev deps, pre-commit, test tools
pre-commit install  # optional, enables linters/formatters
```

### 1) End-to-end locally (DVC)

```bash
# Runs: calibrate → preprocess → train → predict → diagnose
dvc repro
```

### 2) Or stage-by-stage (CLI)

```bash
# Calibrate + preprocess
spectramind calibrate --config-name calibrate +env=local +calib=nominal
# Train
spectramind train     --config-name train     +model=v50 +data=kaggle
# Predict
spectramind predict   --config-name predict   ckpt=artifacts/checkpoints/model.ckpt
# Diagnose (render HTML dashboard)
spectramind diagnose report --out artifacts/reports/diagnostics_dashboard.html
# Submit (package+validate)
spectramind submit    --config-name submit    inputs.pred_path=artifacts/predictions/mu.csv
```

### 3) Hydra overrides (examples)

```bash
# Switch encoders / bins / loss weights
spectramind train --config-name train \
  +model=v50 \
  +search=encoder_depth,bins \
  loss.smoothness.lam=5e-4 loss.symbolic.enabled=true

# Multirun grid sweeps (Hydra)
spectramind train --multirun +search=encoder_depth,bins
```

### 4) Uncertainty calibration

```bash
# Fit scalar temperature T on validation split and rescale σ
spectramind diagnose calibration --dataset val --ckpt artifacts/checkpoints/model.ckpt
```

### 5) Validate a submission (CI uses this)

```bash
bash scripts/validate_submission.sh dist/submission.json
```

---

## Pipeline Stages

| Stage          | CLI                            | DVC outs                                       |
| -------------- | ------------------------------ | ---------------------------------------------- |
| **calibrate**  | `spectramind calibrate`        | `data/interim/calibrated/`                     |
| **preprocess** | *(internal; prepares tensors)* | `data/processed/tensors/`                      |
| **train**      | `spectramind train`            | `artifacts/checkpoints/model.ckpt`             |
| **predict**    | `spectramind predict`          | `artifacts/predictions/{mu.csv,sigma.csv}`     |
| **diagnose**   | `spectramind diagnose report`  | `artifacts/reports/diagnostics_dashboard.html` |
| **submit**     | `spectramind submit`           | `dist/submission.zip / submission.json/csv`    |

End-to-end:

```bash
dvc repro
```

---

## Configuration (Hydra)

* All runtime parameters live under `configs/` (env, data, model, training, loss, logger, search).
* Compose defaults and override at the CLI:

```bash
spectramind train --config-name train +env=kaggle +data=kaggle +model=v50
```

* A run snapshot is saved to `artifacts/configs/run.yaml` (use for repro & lineage).

---

## Losses & Physics Constraints

Enable/disable via Hydra (`configs/loss/constraints.yaml`):

```yaml
loss:
  smoothness: {enabled: true, lam: 1e-3}           # quadratic spectral smoothness
  band_priors:
    enabled: true
    bands: [[130,145],[190,205]]                   # example AIRS CO2/H2O ranges (bins)
    weight: 1e-3
  symbolic:
    enabled: true
    rules:
      - {name: nonneg_mu, type: constraint, target: mu, expr: 'x >= 0', weight: 1.0}
      - {name: bounded_sigma, type: constraint, target: sigma, expr: 'abs(x) < 5.0', weight: 0.1}
```

---

## Diagnostics

* **Inject:** Gaussian absorption bands; photometric trends (FGS1)
* **Recover:** Train/predict → ensure signals are detected/neutralized appropriately
* **Report:** HTML summary in `artifacts/reports/diagnostics_dashboard.html`

```bash
spectramind diagnose report --out artifacts/reports/diagnostics_dashboard.html
pytest -q tests/diagnostics
```

---

## Submission

* **Validate** against JSON schema; CI runs `scripts/validate_submission.sh` before packaging.
* **Package** reproducible artifact with config snapshot + checksum.

```bash
spectramind submit --config-name submit inputs.pred_path=artifacts/predictions/mu.csv
bash scripts/validate_submission.sh dist/submission.json
```

---

## Principles

* **Reproducible**: fixed seeds, Hydra config snapshot, artifact checksums, DVC lineage
* **Physics-aware**: smoothness, non-negativity, band coherence, symbolic rules
* **Kaggle-ready**: slim deps, deterministic seeds, ≤ 9h guardrails; no internet assumptions
* **Auditable**: JSONL event logs; CI SBOM; schema validation pre-submit

---

## Make Targets

```bash
make dev     # install dev deps, pre-commit, linters
make test    # run pytest suite
make bench   # quick ablation (CI also runs nightly)
make docs    # build MkDocs
make clean   # clean caches/dist
```

---

## Notes for Kaggle

* Set `+env=kaggle +data=kaggle` and write all outputs to `/kaggle/working`.
* Use `requirements-kaggle.txt` only.
* Disable non-essential diagnostics; respect time/memory caps.

---
