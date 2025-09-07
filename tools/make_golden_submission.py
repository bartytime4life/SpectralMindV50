# tools/make_golden_submission.py
"""
SpectraMind V50 — Golden Submission Generator
=============================================

Creates a deterministic "golden" submission CSV for unit tests.
Matches the schema in `schemas/submission.schema.json`:
  - 1 column: sample_id
  - 283 mu_* columns
  - 283 sigma_* columns

Usage:
    python tools/make_golden_submission.py
"""

import numpy as np
import pandas as pd
import pathlib
import json

N_BINS = 283
SAMPLE_IDS = ["row_0", "row_1"]
RNG_SEED = 7


def build_columns(n_bins: int) -> list[str]:
    """Return ordered submission columns: sample_id, mu_*, sigma_*."""
    mu_cols = [f"mu_{i:03d}" for i in range(n_bins)]
    sigma_cols = [f"sigma_{i:03d}" for i in range(n_bins)]
    return ["sample_id"] + mu_cols + sigma_cols


def generate_data(n_rows: int, n_bins: int, seed: int) -> pd.DataFrame:
    """Generate reproducible Gaussian μ and uniform σ."""
    rng = np.random.default_rng(seed)
    mus = rng.normal(0, 0.1, (n_rows, n_bins))
    sigmas = rng.uniform(1e-3, 0.2, (n_rows, n_bins))

    data = {"sample_id": SAMPLE_IDS[:n_rows]}
    for i in range(n_bins):
        data[f"mu_{i:03d}"] = mus[:, i]
    for i in range(n_bins):
        data[f"sigma_{i:03d}"] = sigmas[:, i]

    return pd.DataFrame(data, columns=build_columns(n_bins))


def main():
    df = generate_data(n_rows=2, n_bins=N_BINS, seed=RNG_SEED)

    out_path = pathlib.Path("tests/golden/submission_valid.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Wrote golden submission to {out_path.resolve()}")
    print(df.head())


if __name__ == "__main__":
    main()