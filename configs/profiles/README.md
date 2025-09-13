# Profiles — SpectraMind V50

Runtime overlays you can stack on top of the base training configs. Profiles **do not** replace your stack — they import `/training/*` configs and tweak only what’s necessary (precision, cadence, logging, workers, etc.).

---

## Quick Start

```bash
# Kaggle GPU (T4/L4 ~16GB), AMP on, offline logging
spectramind train +defaults='[/profiles/kaggle]'

# Local development (fast feedback, chatty UI; partial epochs)
spectramind train +defaults='[/profiles/local_dev]'

# GitHub Actions / Kaggle CI smoke (2 epochs, FP32)
spectramind train +defaults='[/profiles/ci_fast]'

# Ampere (A100/L4/4090): bf16 + TF32 (verify numerics first)
spectramind train +defaults='[/profiles/ampere_bf16]'

# Full production run (longer schedule, warmup + cosine)
spectramind train +defaults='[/profiles/full]'

# Priors OFF (GLL-only) ablation overlay
spectramind train +defaults='[/profiles/ablation_priors_off]'
````

---

## What each profile does (at a glance)

| Profile               | Precision          | Epochs / Cadence                      | Workers / I/O                    | Logging            | Notes                            |
| --------------------- | ------------------ | ------------------------------------- | -------------------------------- | ------------------ | -------------------------------- |
| `kaggle`              | `16-mixed`         | `val_check_interval=0.5`              | `num_workers=2`, pin, prefetch=2 | offline true       | Kaggle-safe defaults + warmup    |
| `local_dev`           | `32-true`          | partial epochs (`0.25/0.5` train/val) | `num_workers=4`                  | default            | Fast feedback + profiler toggle  |
| `ci_fast`             | `32-true`          | 2 epochs, tiny slices                 | deterministic workers=0          | offline            | Uses `/training/fast_ci`         |
| `ampere_bf16`         | `bf16-mixed`+TF32  | normal cadence                        | `num_workers=4`, pin cuda        | default            | For A100/L4/4090; validate first |
| `full`                | `16-mixed`         | long run, cosine + warmup             | env-driven                       | default            | Production-style schedule        |
| `ablation_priors_off` | inherit            | inherit                               | inherit                          | inherit            | Zeroes aux priors; GLL on        |
| `debug`               | `32-true`          | 1 batch/1 epoch, anomaly on           | workers=0                        | verbose            | Fail-fast loud sanity            |
| `overfit_small`       | `32-true`          | `overfit_batches=0.01`                | inherit                          | default            | Validate learning signal         |
| `low_vram`            | `16-mixed`         | warmup on                             | `batch=16`, `workers=1`          | default            | Adds `accumulate_grad_batches=4` |
| `cpu`                 | `32-true`          | short                                 | workers=0, no pin                | default            | CPU-only bring-up                |
| `ddp`                 | inherit            | multi-GPU (single node)               | `workers=4`, distributed sampler | default            | Launch with `torchrun`           |
| `wandb_online`        | inherit            | inherit                               | inherit                          | **W\&B online**    | Disable on Kaggle                |
| `mlflow`              | inherit            | inherit                               | inherit                          | **MLflow online**  | Requires `MLFLOW_TRACKING_URI`   |
| `profiling`           | `bf16-mixed`       | short profile run                     | `workers=4`, pin, prefetch=4     | default            | `trainer.profiler=advanced`      |
| `amp_off`             | `32-true`          | inherit                               | inherit                          | default            | Disable AMP to de-risk numerics  |
| `bf16_strict`         | `bf16-mixed`       | inherit                               | inherit                          | default            | TF32 disabled, stricter kernels  |
| `resume`              | inherit            | inherit                               | inherit                          | inherit            | Resume via `SM_RESUME_CKPT`      |
| `notebook`            | inherit            | short, friendly cadence               | `workers=2`, no persistent       | offline by default | Kaggle/Jupyter ergonomics        |
| `predict_only`        | inherit            | `max_epochs=0`, no ckpt/ES            | inherit                          | inherit            | Inference-only                   |
| `diagnose`            | inherit            | short train + frequent val            | inherit                          | inherit            | Pairs with `diagnose` stage      |
| `hyper_sweep`         | inherit            | moderate                              | inherit                          | offline            | Hydra multirun friendly          |
| `kaggle_submit`       | inherit (`kaggle`) | longer                                | inherit                          | offline            | Produce submission artifacts     |

> All profiles compose `/training/*` configs; only minimal keys change.

---

## Common environment overrides (`SM_*`)

Flip knobs without editing files:

```bash
# Precision & TF32
SM_PRECISION=bf16-mixed SM_TF32_ALLOW=true SM_TF32_CUDNN=true \

# Run length & cadence
SM_MAX_EPOCHS=80 SM_VAL_INTERVAL=0.25 SM_LOG_EVERY=25 \

# DataLoader
SM_WORKERS=4 SM_PREFETCH=4 SM_PIN_MEMORY=true \

# Accumulation & LR scaling
SM_ACCUM=4 SM_LR_SCALE=true SM_REF_BS=256 \

# Checkpointing / EarlyStop
SM_CKPT_TOPK=1 SM_ES_PATIENCE=6 \

# Logging
SM_LOG_OFFLINE=true SM_RUN_SUFFIX=-kA
```

Hydra CLI edits are also supported, e.g.:

```bash
spectramind train +data_loader.batch_size=24 trainer.accumulate_grad_batches=4
```

---

## Composition tips (Hydra best-practices)

* **Profiles are overlays** — they should not redefine `_target_` blocks from scratch; keep those in `/training/*` factories and only adjust parameters.
* **Mutually exclusive cadence** — if you set `ModelCheckpoint.every_n_train_steps`, keep `every_n_epochs: null` to avoid double writes.
* **Determinism** — for investigations, prefer `profiles/amp_off` or `bf16_strict`; combine with determinism toggles in your `precision.yaml`.
* **DDP** — use `profiles/ddp` and launch with `torchrun --nproc_per_node=N ...`. The DataModule should use a `DistributedSampler` when `strategy=ddp`.
* **Kaggle** — prefer `profiles/kaggle` (offline logging, low workers, warmup on). Flip to `kaggle_submit` when producing artifacts.

---

## Handy recipes

**Low-VRAM A/B** (simulate big batch via accumulation):

```bash
spectramind train +defaults='[/profiles/low_vram]' \
  +optimizer.lr_scaling.enabled=true \
  +optimizer.lr_scaling.reference_batch_size=256 \
  data_loader.batch_size=16 trainer.accumulate_grad_batches=8
```

**Overfit sanity**:

```bash
spectramind train +defaults='[/profiles/overfit_small]' \
  +loss.composite.weights.smoothness=0.1
```

**DDP 2× GPUs (single node)**:

```bash
torchrun --nproc_per_node=2 -m spectramind train +defaults='[/profiles/ddp]'
```

**Resume**:

```bash
SM_RESUME_CKPT=artifacts/models/last.ckpt \
spectramind train +defaults='[/profiles/resume]'
```

**Ablation — priors OFF**:

```bash
spectramind train +defaults='[/profiles/ablation_priors_off]' \
  +trainer.max_epochs=25
```

---

## Troubleshooting

* **“Missing monitored metric” on checkpoint/ES** — ensure your validation step logs `${callbacks.model_checkpoint.monitor}` (default `val/loss`).
* **W\&B / MLflow network errors on Kaggle** — use `profiles/kaggle` (offline) or set `training.logger.offline=true`.
* **Deadlocks with workers** — use `num_workers: 0..2` and `persistent_workers: false` on Kaggle/CI; consider `SM_MP_CTX=spawn`.
* **Unstable AMP** — switch to `profiles/amp_off` (pure FP32) or `bf16_strict` (no TF32). Check `precision.grad_scaler` settings for fp16.

---

## Files in this folder

* `kaggle.yaml`, `local_dev.yaml`, `ci_fast.yaml`, `ampere_bf16.yaml`, `full.yaml`,
  `ablation_priors_off.yaml`, `debug.yaml`, `overfit_small.yaml`, `low_vram.yaml`,
  `cpu.yaml`, `ddp.yaml`, `wandb_online.yaml`, `mlflow.yaml`, `profiling.yaml`,
  `amp_off.yaml`, `bf16_strict.yaml`, `resume.yaml`, `notebook.yaml`,
  `predict_only.yaml`, `diagnose.yaml`, `hyper_sweep.yaml`, `kaggle_submit.yaml`, and this `README.md`.

> Add or tweak profiles freely—each should remain a thin overlay on `/training/*` without duplicating targets or factory wiring.

