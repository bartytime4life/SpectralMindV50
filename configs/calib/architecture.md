# 🔬 Calibration Config Architecture — SpectraMind V50

Calibration configs (`configs/calib/*.yaml`) define the **instrumental corrections**
applied to raw Ariel telescope data before it enters the ML encoders.

---

## 📂 Directory Layout

```

configs/calib/
├── nominal.yaml        # Default full calibration chain
├── fast.yaml           # Lightweight, CI/debug version
├── strict.yaml         # Validation-grade version (all checks)
├── ARCHITECTURE.md     # (this document)
└── method/             # Stage-level configs
├── adc.yaml        # ADC bias/gain conversion
├── cds.yaml        # Correlated double sampling
├── dark.yaml       # Dark subtraction
├── flat.yaml       # Flat-field correction
├── trace.yaml      # Wavelength solution
├── photometry.yaml # FGS1 photometric detrend
└── phase.yaml      # Phase folding & alignment

````

---

## ⚙️ Design Principles

1. **Hydra-first modularity**  
   - Each stage lives in its own `method/*.yaml`.  
   - Profiles (`nominal.yaml`, `fast.yaml`, `strict.yaml`) compose subsets of stages.

2. **Physics-informed calibration**  
   - Explicitly encodes the Ariel mission pipeline:  
     ADC → CDS → DARK → FLAT → TRACE → PHOTOMETRY → PHASE:contentReference[oaicite:3]{index=3}.  
   - Keeps FGS1 and AIRS branches explicit.

3. **Reproducibility**  
   - Stage configs pin input reference files (e.g. `adc_info.csv`, `axis_info.parquet`).  
   - Calibration configs are hashed and logged with every run:contentReference[oaicite:4]{index=4}.

4. **Profiles for context**  
   - `nominal`: Full physics-correct pipeline for experiments.  
   - `fast`: Minimal subset (ADC + phase align) for CI/debugging.  
   - `strict`: Adds diagnostic checks (FFT, UMAP, residual stats) for scientific review.

---

## 🧩 Stage Overview

| Stage        | Purpose                                                   | Applies to |
|--------------|-----------------------------------------------------------|------------|
| **ADC**      | Convert raw ADU counts → flux units using gain/bias map.  | FGS1 + AIRS |
| **CDS**      | Correlated double sampling: remove kTC/reset noise.       | FGS1 + AIRS |
| **DARK**     | Subtract dark current reference frames.                   | FGS1 + AIRS |
| **FLAT**     | Flat-field correction using wavelength-dependent maps.    | AIRS only  |
| **TRACE**    | Map detector pixels → physical wavelength grid.           | AIRS only  |
| **PHOTOMETRY** | Extract FGS1 lightcurve, detrend systematics.           | FGS1 only  |
| **PHASE**    | Phase-fold & align lightcurves using ephemerides.         | FGS1 + AIRS |

---

## 🔄 Typical Usage

```bash
# Run nominal calibration chain
python -m spectramind.calibrate calib=nominal

# Run fast CI/debug calibration
python -m spectramind.calibrate calib=fast

# Run strict validation with diagnostics
python -m spectramind.calibrate calib=strict
````

---

## 📌 Notes

* Reference files (bias/gain, axis info, etc.) are sourced from Kaggle dataset mounts.
* All calibration stages are optional toggles; disabling one sets `enable: false`.
* Outputs are DVC-tracked so calibrated cubes can be reused downstream.
* Diagnostic profiles (`strict.yaml`) dump FFT/UMAP visual checks to ensure calibration realism.

---

✅ With this design, **calibration configs are modular, Hydra-safe, Kaggle-compliant, and scientifically aligned** with Ariel mission physics.
