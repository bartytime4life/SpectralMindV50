# Model Profiles — SpectraMind V50

**Goal:** dual-channel encoders (FGS1 photometry, AIRS spectroscopy) → cross-attention fusion → heteroscedastic decoder (μ, σ over 283 bins) with physics-informed loss.

**Components**
- `fgs1_encoder`: temporal encoder specialized for white-light (FGS1). Default: SSM/Mamba-style or light Transformer.
- `airs_encoder`: spectral encoder (temporal×spectral tokens). Default: Conv/Graph hybrid with spectral pos-enc.
- `fusion`: cross-attention block(s) with residual MLP; optional gating of FGS1 features.
- `decoder`: dual-head MLP for μ/σ; σ via softplus+eps; optional spectral norm and dropout.

**Presets**
- `v50_small` — light depth/width for CI/Kaggle time budget.
- `v50` — recommended default (balanced).
- `v50_large` — extended depth/width for ablations.

**Contracts**
- Outputs: dict with `mu: (B, BINS)`, `sigma: (B, BINS)`. Bin 0 is FGS1 (“white-light”); bins 1..BINS-1 are AIRS.
- BINS = `${data.channels.airs.bins}` (defaults to 283).
- All modules are configured via `_target_` strings in code; the YAMLs remain the single source of hyperparams.
