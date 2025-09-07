# Architecture Overview

- CLI-first via Typer (`spectramind`).
- Hydra config groups: env, data, calib, model, training, loss, logger.
- Data flow: raw → calib (ADC→dark→flat→CDS→photometry→trace→phase) → tensors → encoders (FGS1/AIRS) → fusion → decoder → 283-bin μ/σ.
- Reproducibility: config snapshot hash, JSONL event logs, DVC pipeline stages.
