# Preprocess — SpectraMind V50

**Purpose:** Consume calibrated outputs (FGS1 + AIRS) and produce **model-ready tensors**
for train/val/test with consistent contracts and schemas.

**Stage order (logical):**
1) **load**: resolve calibrated artifacts for split.
2) **mask**: apply NaN/saturation/cosmic masks; build boolean masks.
3) **detrend**: optional polynomial/Savitzky–Golay detrending (per time-series).
4) **normalize/scale**: per-channel z-score or robust scaling; store stats.
5) **binning/window**: uniform time windows (T) with stride; optional AIRS bin ops.
6) **pack**: assemble tensors `{fgs1: (B,T), airs: (B,T,BINS)}` + masks/meta.
7) **tokenize/pe**: build positional encodings (temporal/spectral).
8) **augment (train only)**: time-shift/noise/dropout-time; never applied to val/test.
9) **export**: write features as `.npz` or Parquet + small manifest JSON.

**I/O conventions:**
- **Input (calibrated)**: `${data.exports.calibrated_root}/`
- **Output (features)** : `${data.exports.features_root}/`
- Each split gets its subdir: `features/{train,val,test}/`.
- All paths reference `paths.*` and `data.*` roots to be portable across envs.

**Schema (typical `.npz`):**
- `fgs1`: float32 `(N, T)`           — white-light series (masked values are 0; see `mask_fgs1`)
- `airs`:  float32 `(N, T, BINS)`     — spectrogram (masked values are 0; see `mask_airs`)
- `mask_fgs1`: bool `(N, T)`          — valid=1 / masked=0
- `mask_airs`:  bool `(N, T, BINS)`
- `pe_time`: float32 `(N, T, D_t)`    — temporal positional encodings
- `pe_spec`: float32 `(1, BINS, D_s)` — spectral PE (broadcast on batch/time)
- `y`:       float32 `(N, BINS)`      — (train/val) labels μ; `None` for test
- `meta`:    JSON (per-sample id, indices, scaling stats keys, etc.)

Always persist **scaler stats** (μ/σ) per channel/split for deterministic inference.
