# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

Mission‑grade, CLI‑first, Hydra‑driven, DVC‑tracked, Kaggle‑ready repo. Physics‑informed, neuro‑symbolic pipeline for multi‑sensor fusion (FGS1 + AIRS) producing calibrated μ/σ over 283 spectral bins.

## Quickstart
```bash
# 1) Local dev
make dev

# 2) Calibrate → Train → Predict → Submit
spectramind calibrate --config-name train +calib=nominal +env=local
spectramind train     --config-name train +model=v50 +data=kaggle
spectramind predict   --config-name predict ckpt=artifacts/ckpt.pt
spectramind submit    --config-name submit inputs.pred_path=outputs/preds.csv
```

## Principles
- Reproducible (seeds, config hash, manifests)
- Physics‑aware (smoothness/non‑negativity/band coherence)
- DVC lineage for data + models
- Kaggle runtime guardrails (≤ 9h)
