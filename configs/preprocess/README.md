# Preprocess — SpectraMind V50

> CLI-first pipeline that turns **calibrated** FGS1 + AIRS into **model-ready** tensors with masks, PEs, and manifests.

- Architecture: see [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- Presets: [`presets/fast.yaml`](./presets/fast.yaml) · [`presets/nominal.yaml`](./presets/nominal.yaml) · [`presets/strict.yaml`](./presets/strict.yaml)
- Methods: [`method/`](./method)

---

## Quickstart

```bash
# From repo root
make preprocess.nominal SPLIT=train
make preprocess.nominal SPLIT=val
make preprocess.nominal SPLIT=test
Other presets:

bash
Copy code
make preprocess.fast   SPLIT=train     # CI/Kaggle budget
make preprocess.strict SPLIT=val       # research-grade; Parquet+Zstd
Hydra overrides inline:

bash
Copy code
# Example: strict preset, stride override, and NPZ output
make preprocess.strict SPLIT=train OVERRIDES="preprocess/shapes.stride=640 io.format=npz"
Environment knobs (no YAML edits)
Var	Default	Purpose
SM_NUM_WORKERS	preset value	DataLoader workers
SM_PIN_MEMORY	preset value	Torch dataloader pinning
SM_LOAD_MAX_SAMPLES	—	Debug slice during load
SM_MASK_REPORT_SAMPLES	128	Mask report sampling
SM_WIN_REPORT_SAMPLES	128	Window report sampling
SM_TOK_REPORT_SAMPLES	64	Tokenize report sampling
SM_PACK_REPORT_SAMPLES	64	Pack report sampling

Outputs
bash
Copy code
${data.exports.features_root}/
└─ {train,val,test}/
   ├─ *.npz | *.parquet
   └─ manifest.json
Scaler stats (fit on train, applied everywhere):

bash
Copy code
${io.features_root}/scaler/
  ├─ fgs1.per_channel.<strategy>.npz
  └─ airs.per_channel.<strategy>.npz
Common patterns
Fast smoke test (CI/Kaggle):

bash
Copy code
make preprocess.fast SPLIT=train
Train on fast, eval on strict:

bash
Copy code
make preprocess.fast   SPLIT=train
make preprocess.strict SPLIT=val
make preprocess.strict SPLIT=test
Troubleshooting
Missing scaler stats on val/test → ensure you ran SPLIT=train first for the same preset (stats live under features_root/scaler).

Phase alignment skipped → check meta.phase_center presence; see method/window.yaml: phase.on_missing.

Coverage assertion tripped → see method/mask.yaml: assertions.max_mask_coverage.*.
