📘 Kaggle Integration — SpectraMind V50

This directory contains Kaggle-specific assets for the
NeurIPS 2025 Ariel Data Challenge.

⸻

📂 Contents
	•	notebook_template.ipynb — lightweight, submission-ready starter
(auto-detects Kaggle vs local, runs with internet disabled).
	•	README.md — this document.

⸻

🚀 Usage on Kaggle
	1.	Attach code & config
	•	Export this repo as a Kaggle Dataset (code only).
	•	Attach it in your Kaggle Notebook (as /kaggle/input/spectramind-v50/).
	2.	Attach competition dataset
	•	ariel-data-challenge-2025 (FGS1 + AIRS) ￼Ariel Data Challenge Dataset.pdf.
	•	Available under Add Data → Competition Data.
	3.	Install pinned requirements

pip install -r /kaggle/input/spectramind-v50/requirements-kaggle.txt

Dependencies are slim & Kaggle-safe (no internet needed at runtime).

	4.	Run the template
Open notebook_template.ipynb, adjust configs if needed, then run all cells:
	•	Generates outputs/config_snapshot.json.
	•	Produces predictions in outputs/submission.csv.
	•	Zips to submission.zip for leaderboard upload.

⸻

🧭 Workflow (End-to-End)

flowchart LR
  A["Calibration: FGS1 + AIRS → calibrated cubes"] --> B["Encoders: FGS1=Mamba SSM; AIRS=GNN/CNN"]
  B --> C["Decoder: μ & σ (heteroscedastic GLL loss)"]
  C --> D["Diagnostics: GLL, FFT, UMAP, symbolic checks"]
  D --> E["Submission bundle: CSV + manifest"]
  E --> F["Kaggle leaderboard"]

This mirrors the SpectraMind V50 architecture ￼ ￼.

⸻

🛡️ Guardrails
	•	No internet (Kaggle rule). All data must come from attached datasets ￼Ariel Data Challenge Dataset.pdf.
	•	≤ 9h runtime per kernel. Configs include “fast” modes for testing ￼.
	•	Pinned deps — never pip install unpinned packages.
	•	Reproducibility — configs + snapshots are stored in outputs/ ￼.

⸻

🔑 References
	•	[SpectraMind V50 repository design & architecture] ￼ ￼
	•	[Kaggle dataset linking & workflow notes] ￼Ariel Data Challenge Dataset.pdf ￼
	•	[Dual-channel solution (FGS1 + AIRS) & competition metric (GLL)] ￼ ￼
	•	[Scientific context: exoplanet spectroscopy challenges & advances] ￼
	•	[Recent Nature papers on JWST/Ariel spectral extraction] ￼

⸻

📌 Tip: Use the two-kernel pattern — one for training, one for inference/submission ￼. Both import and call the src/spectramind library to avoid duplicated code.

⸻
