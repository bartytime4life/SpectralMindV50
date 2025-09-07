# ADR 0002 — Physics-Informed Losses

* **Status:** ✅ Accepted
* **Date:** 2025-09-06
* **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge
* **Tags:** loss, physics-informed, reproducibility, exoplanet, astrophysics
* **Owners:** ML/Physics WG (Lead: Andy Barta), Spectroscopy Science Council

---

## 1. Context

The NeurIPS 2025 Ariel Data Challenge requires predicting **283-bin transmission spectra (μ/σ)** with calibrated uncertainties.

* The official metric is **Gaussian Log-Likelihood (GLL)**, with the **FGS1 bin weighted \~58×** relative to others.
* Prior challenges (2024) showed leaderboard failures when models were **overconfident** or produced **physically invalid spectra** (negative depths, jagged noise).
* Our repository design explicitly supports modular Hydra configs under `configs/loss/`.
* Physics-informed priors (smoothness, positivity, coherence) are expected to improve generalization and credibility of outputs.

Without these constraints, models may optimize for Kaggle score but yield **scientifically implausible spectra**.

---

## 2. Decision

We introduce a **composite physics-informed loss**:

$$
\mathcal{L} = \mathcal{L}_{GLL}
+ \lambda_{smooth} \mathcal{L}_{smooth}
+ \lambda_{nonneg} \mathcal{L}_{nonneg}
+ \lambda_{band} \mathcal{L}_{band}
+ \lambda_{calib} \mathcal{L}_{calib}
$$

### Components

* **$\mathcal{L}_{GLL}$** — baseline Gaussian log-likelihood (competition metric).
* **$\mathcal{L}_{smooth}$** — penalizes excessive curvature in spectra (encourages broad molecular features).
* **$\mathcal{L}_{nonneg}$** — enforces non-negative transit depths (no negative absorption).
* **$\mathcal{L}_{band}$** — encourages **band coherence** across contiguous bins for molecular signatures (H₂O, CO₂, CH₄).
* **$\mathcal{L}_{calib}$** — cross-channel calibration loss aligning AIRS with FGS1 to avoid inconsistency.

All λ hyperparameters are tunable in `configs/loss/composite.yaml`.

---

## 3. Architecture


```mermaid
flowchart TD
  A["GLL baseline (competition metric)"] --> L["Composite Loss"]
  B["Smoothness prior"] --> L
  C["Non-negativity prior"] --> L
  D["Band coherence prior"] --> L
  E["Calibration prior (FGS1<->AIRS)"] --> L
  L --> F["Backpropagation -> Model update"]



* Each loss lives in `src/spectramind/losses/*.py`.
* Configured via Hydra (`configs/loss/*.yaml`).
* Logged separately in JSONL, W\&B, and diagnostics.

---

## 4. Consequences

### ✅ Pros

* Improves leaderboard robustness under **OOD regimes**.
* Produces **scientifically credible spectra** (smooth, positive, coherent).
* Modular: priors can be toggled on/off, tuned, or ablated via Hydra configs.
* Aligned with reproducibility (DVC, Hydra, CI/CD).

### ⚠️ Cons

* Over-regularization risk → underfit spectra.
* Requires λ sweeps (adds compute).
* Must justify priors scientifically (avoid leaderboard over-engineering).

---

## 5. Implementation Plan

1. **Loss terms** → `src/spectramind/losses/{smoothness,nonneg,band_coherence,calibration,composite}.py`.
2. **Hydra configs** → `configs/loss/{smoothness.yaml,nonneg.yaml,band_coherence.yaml,calibration.yaml,composite.yaml}`.
3. **Logging** → per-component loss values in JSONL + W\&B.
4. **Diagnostics** → FFT/UMAP checks for smoothness & coherence.
5. **Validation** → ablation runs to verify incremental contribution of each prior.

---

## 6. Risks & Mitigations

| Risk                                        | Mitigation                                                    |
| ------------------------------------------- | ------------------------------------------------------------- |
| Over-constraining leads to underfit         | Start with small λ, increase gradually; perform ablations.    |
| Scientific overfitting to challenge dataset | Validate against synthetic OOD datasets (random atmospheres). |
| Compute cost of sweeps                      | Use early stopping + Bayesian search for λ tuning.            |
| Calibration mismatch (FGS1/AIRS)            | Regularize with weight annealing schedule.                    |

---

## 7. Compliance Gates (CI)

* [ ] Composite loss unit tests (numerical stability, nonneg enforcement).
* [ ] Each prior independently toggleable via Hydra.
* [ ] JSONL logs include per-loss breakdown.
* [ ] Diagnostic plots generated in `artifacts/reports/`.
* [ ] Kaggle runtime guardrails ensure no internet calls during training.

---

## 8. References

* NeurIPS 2025 Ariel Challenge docs
* Lessons from 2024 challenge (smoothness + nonnegativity helped generalization)
* SpectraMind repo design: ADR 0001 (Hydra+DVC), ADR 0003 (CI↔CUDA parity)
* Recent Nature spectroscopy papers (JWST WASP-39b, SO₂ detection, CO₂ absorption)

---
