📜 Changelog — SpectraMind V50

All notable changes to this project will be documented here.
This project adheres to Semantic Versioning and the
Keep a Changelog format.

⸻

[Unreleased]

🚀 Added
	•	JSON Schemas:
	•	schemas/submission.tableschema.sample_id.json — Tableschema with sample_id as canonical key; stricter enums/patterns and full worked examples.
	•	Integration tests:
	•	tests/integration/test_calib_chain.py.
	•	tests/integration/test_end_to_end_cli.py.
	•	Kaggle assets:
	•	kaggle/README.md, pinned requirements-kaggle.txt, and notebook templates wired to CLI (spectramind.py).
	•	Docs: expanded calibration internals in src/spectramind/calib/, pipeline diagrams in assets/diagrams/, and repo conventions in docs/.
	•	Diagnostics:
	•	FFT/UMAP analysis in src/spectramind/diagnostics/spectral_analysis.py and src/spectramind/diagnostics/dimensionality.py — see ADR 0004 Dual Encoder Fusion.
	•	Physics-informed checks in src/spectramind/validators/physics.py — see ADR 0002 Physics-Informed Losses.
	•	Unified HTML/JSONL report generator src/spectramind/diagnostics/report.py — see ADR 0002 + ADR 0004.
	•	ADC quick-look: src/spectramind/diagnostics/adc_diag.py — see ADR 0002.

🔄 Changed
	•	Trace modeling (src/spectramind/calib/trace.py): multi-order ready center/width, NaN-safe math, stable denominators, background models (column_median, row_poly), Torch-first execution.
	•	Photometry (src/spectramind/calib/photometry.py): batch-aware [..., T, H, W], adaptive apertures, PSF-weighted optimal extraction with variance propagation, full NaN safety.
	•	CI workflows: .github/workflows/ hardened with matrix pinning, cache keys, deterministic pytest.
	•	Pre-commit stack: .pre-commit-config.yaml updated (ruff/black/isort/mypy/bandit/secrets).
	•	Schemas: tightened submission.schema.json + events.schema.json; added drift tests.
	•	Diagnostics reporting (diagnostics/report.py): improved JSON/HTML generator (inline CSS, title support, deterministic UTF-8) — see ADR 0002.

🛠️ Fixed
	•	Time-axis normalization and mask/variance alignment across calibration modules (src/spectramind/calib/*).
	•	PSF normalization edge cases in photometry.py.
	•	Docstring mismatches and dtype inconsistencies (Torch vs NumPy).
	•	ADC calibration diagnostics clamped NaN/Inf + guarded matplotlib fallback (adc_diag.py) — see ADR 0002.

🧪 Performance
	•	Vectorized photometry loops & reduced tensor allocations (photometry.py).
	•	NumPy fallback only for polyfit; Torch-first everywhere else (trace.py).
	•	FFT analysis normalized per-sample, NaN-safe, batch-aware (spectral_analysis.py) — see ADR 0004.

🔒 Security
	•	All new diagnostics modules passed Bandit + CodeQL scanning (ci.yml).

⚠️ Deprecated
	•	None.

❌ Removed
	•	None.

⸻