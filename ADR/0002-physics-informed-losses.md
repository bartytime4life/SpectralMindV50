# ADR 0002 — Physics-Informed Losses

**Status:** Proposed  
**Date:** 2025-09-06  
**Context:** NeurIPS 2025 Ariel Data Challenge — SpectraMind V50 repository

---

## 🎯 Decision

We introduce **physics-informed composite loss functions** for the SpectraMind V50 pipeline.  
These losses combine the baseline **Gaussian Log-Likelihood (GLL)** (competition metric) with **domain-specific priors** enforcing physical plausibility of exoplanet spectra.

---

## 🔍 Context

- The Ariel challenge requires predicting **283-bin transmission spectra (μ/σ)** with well-calibrated uncertainties:contentReference[oaicite:0]{index=0}.  
- The metric is GLL, heavily weighted on the **FGS1 channel (~58×)**:contentReference[oaicite:1]{index=1}.  
- Prior challenges (2024) showed that **overconfident predictions** are penalized and that **physically valid constraints** (smoothness, positivity, coherence) improve generalization:contentReference[oaicite:2]{index=2}.  
- Our repository design explicitly supports **modular loss configs** under `configs/loss/`:contentReference[oaicite:3]{index=3}.

To outperform leaderboard baselines, we must **inject astrophysical priors** into training.

---

## 📐 Decision Details

We define a **composite physics-informed loss**:

\[
\mathcal{L} = \mathcal{L}_{GLL} + \lambda_{smooth} \mathcal{L}_{smooth} + \lambda_{nonneg} \mathcal{L}_{nonneg} + \lambda_{band} \mathcal{L}_{band} + \lambda_{calib} \mathcal{L}_{calib}
\]

- **\(\mathcal{L}_{GLL}\)** — Gaussian Log-Likelihood baseline:contentReference[oaicite:4]{index=4}.
- **\(\mathcal{L}_{smooth}\)** — Penalizes high curvature in spectra (encourages broad molecular features:contentReference[oaicite:5]{index=5}).
- **\(\mathcal{L}_{nonneg}\)** — Enforces non-negative transit depths (physical constraint: no negative absorption).
- **\(\mathcal{L}_{band}\)** — Promotes **band coherence** across contiguous wavelength ranges (e.g. H₂O, CO₂ features:contentReference[oaicite:6]{index=6}).
- **\(\mathcal{L}_{calib}\)** — Optional calibration loss aligning AIRS with FGS1 baseline (prevents cross-channel inconsistency:contentReference[oaicite:7]{index=7}).

All weights \(\lambda\) are configurable in `configs/loss/composite.yaml`.

---

## ✅ Consequences

**Pros:**
- Improves leaderboard robustness under **OOD regimes**:contentReference[oaicite:8]{index=8}.  
- Encourages **scientifically credible spectra** (smooth, positive, molecularly coherent).  
- Fully modular — losses can be toggled or tuned via Hydra configs:contentReference[oaicite:9]{index=9}.  
- Keeps alignment with Kaggle reproducibility and DVC pipeline:contentReference[oaicite:10]{index=10}.

**Cons:**
- Risk of underfitting if priors are too strong.  
- Requires careful λ-tuning (sweeps increase compute).  
- Must document scientific justification for each prior to avoid leaderboard over-engineering.

---

## 📂 Implementation

- Add to `src/spectramind/models/losses.py`: physics-informed loss terms.  
- YAML configs under `configs/loss/`:  
  - `smoothness.yaml`  
  - `nonneg.yaml`  
  - `band_coherence.yaml`  
  - `calibration.yaml`  
  - `composite.yaml` (master composition).  
- Log each loss component separately in JSONL and W&B:contentReference[oaicite:11]{index=11}.  
- Add validation plots in diagnostics (FFT/UMAP + smoothness checks:contentReference[oaicite:12]{index=12}).

---

## 📖 References

- NeurIPS 2025 Ariel Challenge overview:contentReference[oaicite:13]{index=13}  
- Lessons from 2024 challenge:contentReference[oaicite:14]{index=14}  
- SpectraMind repository design:contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}  
- Recent Nature spectroscopy papers (JWST, WASP-39b, SO₂ detection):contentReference[oaicite:17]{index=17}

---
