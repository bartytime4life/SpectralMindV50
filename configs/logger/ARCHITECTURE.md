# 📒 SpectraMind V50 — Logging Architecture

This directory provides modular Hydra configs for logging backends:

- **default.yaml** – composite logger (console + JSONL + CSV by default; TB/W&B opt-in)
- **console.yaml** – Rich console logging
- **jsonl.yaml** – JSONL sink for metrics/events
- **csv.yaml** – CSV sink for metrics (nice for DVC `metrics show`)
- **tensorboard.yaml** – TensorBoard logging (optional)
- **wandb.yaml** – Weights & Biases logging (disabled by default; Kaggle-safe)

## Why this design

- **Kaggle-safe defaults**: no network calls; local sinks only.
- **Hydra composable**: any sink can be toggled or overridden from CLI.
- **Loss-aware**: supports scalar/histogram toggles and intervals to mirror `loss/composite.yaml`.
- **Reproducible**: JSONL/CSV are easy to diff, parse, and track via DVC.

## Typical usage

### Default (console + jsonl + csv)
```bash
python -m spectramind.train logger=default
