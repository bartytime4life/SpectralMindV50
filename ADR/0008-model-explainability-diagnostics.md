# ADR-0008 — Model Explainability & Diagnostics (FFT, UMAP, SHAP, Lineage Reports)

**Status:** 🚧 Draft  
**Date:** 2025-09-13  
**Author:** SpectraMind V50 Team  

---

## Context

SpectraMind V50 solves a dual-sensor exoplanet spectrum extraction problem under extreme noise.  
While prior ADRs established calibration, physics-informed loss, and dual-encoder fusion:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3},  
the pipeline currently lacks **transparent diagnostics** and **explainability tooling**.  

Scientific and competition drivers demand that models are not only performant (high GLL score) but also **interpretable**:  
- Astronomers require physical plausibility (e.g., smoothness, non-negativity, molecular coherence:contentReference[oaicite:4]{index=4}).  
- Kaggle submissions must be reproducible and debuggable.  
- Reviewers and collaborators expect lineage reporting for each submission (inputs, configs, hashes).  

Without built-in explainability, models risk being **black boxes** with untrustworthy outputs.

---

## Decision

We will integrate a **diagnostics & explainability suite** into SpectraMind V50 with the following components:

1. **FFT Analysis** — Fast Fourier Transform of lightcurves to detect residual systematics (jitter, thermal drift):contentReference[oaicite:5]{index=5}.  
2. **UMAP Embeddings** — Nonlinear dimensionality reduction for latent features to visualize clustering of planetary regimes:contentReference[oaicite:6]{index=6}.  
3. **SHAP Values** — Model-agnostic feature attribution to explain spectral predictions and identify dominant input regions.  
4. **Lineage Reports** — Auto-generated HTML/JSON reports combining Hydra config snapshots, DVC hashes, metrics, and diagnostics plots:contentReference[oaicite:7]{index=7}.

All diagnostics will be:
- Invoked via `spectramind diagnose` (Typer CLI).  
- Configurable via Hydra (`configs/diagnostics/*.yaml`).  
- Stored as DVC-tracked artifacts under `outputs/diagnostics/`.  

---

## Drivers

- **Scientific credibility**: Exoplanet spectra must be interpretable by domain experts:contentReference[oaicite:8]{index=8}.  
- **Reproducibility**: Every run must produce a consistent diagnostics bundle for audit:contentReference[oaicite:9]{index=9}.  
- **Competition edge**: Transparent diagnostics accelerate debugging and leaderboard iteration.  
- **Community alignment**: Mirrors best practices in Kaggle research notebooks:contentReference[oaicite:10]{index=10}.  

---

## Alternatives Considered

- **Minimal diagnostics (metrics only):** Too shallow; no physical validation.  
- **Ad hoc notebook analysis:** Non-reproducible; diverges from CLI-first repo design.  
- **External post-hoc explainability tools only:** Increases friction; breaks lineage tracking.  

---

## Risks

- **Runtime overhead** — FFT/UMAP/SHAP can be computationally expensive; mitigated with sampling configs.  
- **Interpretability gap** — SHAP values may mislead if not carefully contextualized; mitigated by combining multiple diagnostics.  
- **Maintenance burden** — Requires upkeep of plotting libraries (matplotlib, seaborn, UMAP, shap).  

---

## Compliance Gates

- `spectramind diagnose` must run in CI and Kaggle with deterministic outputs.  
- Diagnostics YAML configs must be DVC-tracked.  
- Reports must include:
  - FFT spectra plots  
  - UMAP embeddings of latent space  
  - SHAP summary plots  
  - Lineage manifest (configs + hashes + metrics)  
- Submission CI will fail if `diagnostics/report.html` is missing.  

---

## References

- FFT & UMAP Technical Reference:contentReference[oaicite:11]{index=11}  
- Kaggle Notebooks & Explainability Guidelines:contentReference[oaicite:12]{index=12}  
- SpectraMind V50 Repository Design:contentReference[oaicite:13]{index=13}:contentReference[oaicite:14]{index=14}  
- Scientific Context: Exoplanet Spectroscopy Challenges:contentReference[oaicite:15]{index=15}  
- Physics-Informed Losses (ADR-0002):contentReference[oaicite:16]{index=16}  
- Dual Encoder Fusion (ADR-0004):contentReference[oaicite:17]{index=17}

---
