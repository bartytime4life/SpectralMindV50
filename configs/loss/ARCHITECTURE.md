# Loss Profiles — SpectraMind V50

The challenge metric is a **Gaussian Log-Likelihood (GLL)** over 283 bins plus the FGS1
white-light channel. Our training objective is a **composite**:

    L_total = L_GLL
            + λ_smooth · L_smooth
            + λ_nonneg · L_nonneg
            + λ_band   · L_band
            + λ_calib  · L_calib

- **L_GLL**   : heteroscedastic NLL with **FGS1** up-weight matching leaderboard rules.
- **L_smooth**: second-difference (curvature) penalty to suppress unphysical jaggies.
- **L_nonneg**: soft hinge on μ to discourage negative transit depths.
- **L_band**  : bandwise coherence (e.g., local corr/TV) to keep features broad & plausible.
- **L_calib** : optional term to couple AIRS to FGS1 baseline (guarded to never alter FGS1).

Each term is **batched, differentiable**, and can be toggled or re-weighted via Hydra.
