# 🧪 Preprocess — SpectraMind V50

**Purpose:** Consume calibrated outputs (FGS1 + AIRS) and produce **model-ready tensors** for train/val/test with consistent contracts and schemas.

---

## 🧩 Stage Order (Logical)

```mermaid
flowchart TB
  A["Calibrated Inputs<br/>${data.exports.calibrated_root}"]:::raw
  B["load<br/><code>method/load.yaml</code>"]:::stage
  C["mask<br/><code>method/mask.yaml</code>"]:::stage
  D["detrend<br/><code>method/detrend.yaml</code>"]:::stage
  E["normalize<br/><code>method/normalize.yaml</code>"]:::stage
  F["binning + window<br/><code>method/binning.yaml</code><br/><code>method/window.yaml</code>"]:::stage
  G["pack<br/><code>method/pack.yaml</code>"]:::stage
  H["tokenize / positional encodings<br/><code>method/tokenize.yaml</code>"]:::stage
  I["augment (train only)<br/><code>method/augment.yaml</code>"]:::stage
  J["export<br/><code>method/export.yaml</code> → npz/parquet + manifest"]:::stage
  K["Features<br/>${data.exports.features_root}/{train,val,test}"]:::raw

  A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K

  classDef raw fill:#0b3d5c,color:#fff,stroke:#0b3d5c,stroke-width:1px
  classDef stage fill:#0f766e,color:#fff,stroke:#0f766e,stroke-width:1px
````

**Notes**

* **Augment** is enforced train-only in code; never applied to val/test.
* **Normalize** fits stats on **train** split and reuses them everywhere (mask-aware).
* **AIRS binning** prefers calibration-strict in `strict`, calibration-or-fixed in `nominal`, and usually off in `fast`.

---

## 🔧 Entry Points (Hydra)

Run via the CLI (examples):

```bash
# Fast preset (CI/Kaggle time budget)
python -m spectramind preprocess +defaults='[/preprocess/presets/fast]' split=train

# Balanced default
python -m spectramind preprocess +defaults='[/preprocess/presets/nominal]' split=val

# Research-grade rigor + Parquet + overlap
python -m spectramind preprocess +defaults='[/preprocess/presets/strict]' split=test
```

Handy env overrides (no YAML edits):

```bash
export SM_NUM_WORKERS=6
export SM_PIN_MEMORY=true
export SM_LOAD_MAX_SAMPLES=2048
```

---

## 📐 Shapes & Contracts

### Canonical shapes after **pack**

* `fgs1` : **(N, T)** float32 — white-light series
* `airs` : **(N, T, B)** float32 — spectrogram (B = spectral bins)
* `mask_fgs1` : **(N, T)** bool
* `mask_airs` : **(N, T, B)** bool
* `pe.time` : **(N, T, D\_t)** float32 *(if emitted)*
* `pe.spec` : **(1, B, D\_s)** float32 *(broadcast across batch/time; if emitted)*
* `y` : **(N, B)** float32 (train/val only)
* `meta` : JSON per-sample metadata (IDs, centers, provenance)

> Masked numeric values in tensors are **zero-filled**; actual visibility is carried by boolean masks.

### Window geometry

* `T` (time length) = `${preprocess.shapes.time_len}`
* `stride` = `${preprocess.shapes.stride}` (may be `< T` for overlap)

### Split safety

* Normalization **fits on train**; **apply** on {train,val,test}.
* Token/PE generation is **pure** (no data leakage).

---

## 🗂️ Presets Overview

| Preset      | File                   | Intended Use        | Key Traits                                                                                                                      |
| ----------- | ---------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **fast**    | `presets/fast.yaml`    | CI/Kaggle budget    | `detrend:false`, robust normalize, no binning/augment, NPZ export                                                               |
| **nominal** | `presets/nominal.yaml` | Day-to-day training | poly detrend (FGS1), robust normalize, calib-or-fixed binning, Hann windows, light augment                                      |
| **strict**  | `presets/strict.yaml`  | Research/ablations  | Savitzky–Golay detrend (FGS1 + AIRS), robust normalize, calib-strict binning, Hann + 50% overlap, Parquet+Zstd, deep assertions |

---

## 📦 Schema (NPZ / Parquet)

### Typical NPZ keys

```
fgs1, airs, mask_fgs1, mask_airs, pe.time, pe.spec, y, meta
```

### Parquet layout (strict)

* **Table**: rows = windows/samples; columns grouped by namespace
* Example columns:

  * `fgs1[0..T-1]` (float32)
  * `airs[0..T-1,0..B-1]` (flattened or struct list depending on writer)
  * `mask_fgs1[0..T-1]` (bool)
  * `mask_airs[0..T-1,0..B-1]` (bool)
  * `pe.time[0..T-1,0..D_t-1]` (optional)
  * `pe.spec[0..B-1,0..D_s-1]` (optional, deduplicated via side table)
  * `y[0..B-1]` (float32; train/val only)
  * `meta` (JSON string or struct columns)

> Writer details are handled by `method/export.yaml`; Parquet prefers Arrow-compatible structs for analysis (Polars/Arrow/Spark).

---

## 🧱 Stage Contracts (What each method MUST guarantee)

1. **load**

   * Accept `.parquet` or `.npz` for `fgs1`, `airs`, `y` (train/val).
   * Deterministic shuffling via `${preprocess.seed}`.
   * Paired channels aligned by `sample_id`.
   * Validates required columns; supports `SM_LOAD_MAX_SAMPLES`.

2. **mask**

   * Build `mask_fgs1`, `mask_airs` via NaN/Inf, saturation, spike/cosmic logic.
   * Optional dilation + min-run filtering; gap-fill **only inside masked regions**.
   * Coverage assertions (fail if > thresholds).

3. **detrend**

   * Poly or Savitzky–Golay over time; channel-aware enable flags.
   * Never unmask; operates on unmasked values only.

4. **normalize**

   * Strategy = `zscore|robust|minmax` (mask-aware).
   * **Fit on train** (per-sensor scope), persist stats at `${io.features_root}/scaler`.
   * Require stats when applying; clamp post-norm tails (e.g., Z in ±6).

5. **binning + window**

   * AIRS binning strategy per preset (`calib_strict|calib_or_fixed|off`).
   * Window `[length=T, stride]`; phase alignment optional with tolerance.
   * Label slicing aligned to window; mask propagation maintained.

6. **pack**

   * Emit fused `x` (if configured) and per-sensor tensors + masks.
   * Enforce shapes (T, B); validate time/bin alignment; ensure finiteness.
   * Keep selected `meta` fields; optional computed coverage metrics.

7. **tokenize / PE**

   * Generate `pe.time (N,T,D_t)` and `pe.spec (1,B,D_s)`; **never** unmask.
   * Optionally concatenate to inputs (feature dimension) with guardrails.

8. **augment (train only)**

   * Physics-aware noise/jitter/dropout/smoothing; reproducible via run seed.
   * Off for val/test regardless of preset.

9. **export**

   * NPZ (zip) or Parquet (snappy/zstd).
   * Writes small `manifest.json` with key knobs (preset, version, shapes, strategies, worker count, etc.).

---

## 🧾 Manifest (example)

```json
{
  "preset": "nominal",
  "version": 1,
  "seed": 42,
  "shapes": { "time_len": 512, "stride": 512, "bins": 283 },
  "detrend": { "mode": "poly", "poly.order": 1 },
  "normalize": { "strategy": "robust", "scope": "per-channel" },
  "binning": { "strategy": "calib_or_fixed" },
  "window": { "kind": "hann" },
  "runtime": { "num_workers": 4, "pin_memory": true },
  "scaler_stats": {
    "fgs1": "scaler/fgs1.per_channel.robust.npz",
    "airs": "scaler/airs.per_channel.robust.npz"
  }
}
```

---

## 🧨 Safety & Assertions (Fail Fast)

* **Path sanity**: calibrated\_root & features\_root must exist (or be creatable).
* **Mask coverage**: `fgs1/airs` masked fraction ≤ preset thresholds.
* **Normalization**: stats available before apply; finite output required.
* **Alignment**: time/bin alignment between sensors post-windowing.
* **Calibration** (strict): binning must resolve from calib tables or **fail**.
* **Concat guardrail**: `(D_t + D_s)` ≤ `tokenize.validate.max_concat_dim_increase`.

---

## 🔬 Debugging & Diagnostics

* Set `report.enable: true` in methods to emit counts, histograms, and coverage.
* Use `preview.enable: true` in `pack` to snapshot a few packed samples under `${io.features_root}/_preview/pack`.
* Toggle env:

  * `SM_MASK_REPORT_SAMPLES`, `SM_WIN_REPORT_SAMPLES`, `SM_TOK_REPORT_SAMPLES`, `SM_PACK_REPORT_SAMPLES`.

---

## 🧭 Typical Workflows

### One-shot for all splits (nominal)

```bash
for SPLIT in train val test; do
  python -m spectramind preprocess +defaults='[/preprocess/presets/nominal]' split=${SPLIT}
done
```

### Train on fast, evaluate on strict

```bash
python -m spectramind preprocess +defaults='[/preprocess/presets/fast]'   split=train
python -m spectramind preprocess +defaults='[/preprocess/presets/strict]' split=val
python -m spectramind preprocess +defaults='[/preprocess/presets/strict]' split=test
```

---

## 📎 I/O Conventions

* **Input (calibrated)**: `${data.exports.calibrated_root}/`
* **Output (features)**: `${data.exports.features_root}/`
* Split subdirs: `features/{train,val,test}/`
* All paths ingress through Hydra `paths.*` / `data.*`—portable across local/CI/Kaggle.

---

## ✅ Quick Checklist

* [ ] `load` patterns accept **parquet/npz** and filter temp files
* [ ] `mask` union logic is **OR**, gap fill restricted to **masked** spans
* [ ] `detrend` leaves masks intact; AIRS detrend enabled in **strict**
* [ ] `normalize` fits on **train**; exports stats to `scaler/`
* [ ] `window` respects **phase alignment** when available; min unmasked fraction per channel
* [ ] `pack` validates shapes & emits fused/per-sensor + masks + meta
* [ ] `tokenize` produces **time/spec** PEs; concat guarded by max increase
* [ ] `augment` is **train-only** and physics-safe
* [ ] `export` writes **npz/parquet** + `manifest.json` (minimal but sufficient)

---
