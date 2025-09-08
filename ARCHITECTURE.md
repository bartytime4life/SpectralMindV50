🛰️ SpectraMind V50 — Architecture Overview (Upgraded)

Mission-grade, physics-informed, neuro-symbolic pipeline for the NeurIPS 2025 Ariel Data Challenge.
Implements multi-sensor fusion of FGS1 time-series + AIRS spectroscopy, producing calibrated per-bin μ/σ predictions across 283 spectral channels ￼ ￼.

⸻

1. CLI-First Orchestration
	•	Unified entrypoint: spectramind (Typer-based CLI).
	•	Subcommands:
	•	calibrate → raw telescope → calibrated cubes
	•	train → dual encoders + heteroscedastic decoder
	•	predict → checkpoint → submission outputs
	•	diagnose → reproducibility & debug tooling
	•	submit → Kaggle/CI-ready packages

✅ Principle: All logic exposed through CLI + Hydra overrides, no hidden params ￼ ￼.

⸻

2. Hydra Configuration System
	•	Config groups:
	•	env/ → runtime (local, Kaggle, HPC, CI)
	•	data/ → ingestion, calibration, preprocessing
	•	calib/ → ADC → dark → flat → CDS → photometry → trace → phase
	•	model/ → encoders, fusion, decoder
	•	training/ → optimizers, schedulers, precision, workers
	•	loss/ → smoothness, non-negativity, coherence, uncertainty terms
	•	logger/ → JSONL, tensorboard, wandb

✅ Guarantee: One config = one reproducible experiment. Snapshotted + hashed vs schema ￼ ￼.

⸻

3. Data Flow

flowchart TD
    A["Raw Inputs (FGS1 + AIRS)"]
      --> B["Calibration (ADC → dark → flat → CDS → photometry → trace → phase)"]
      --> C["Physics-aligned tensors (DVC-tracked)"]
      --> D["Encoders"]
          D1["FGS1 → State-Space (Mamba SSM)"]
          D2["AIRS → CNN/GNN Spectral"]
          D --> D1
          D --> D2
      --> E["Fusion (cross-modal alignment)"]
      --> F["Decoder → μ/σ per bin (283)"]
      --> G["Submission (CSV, schema-validated)"]

✅ Physics-aware: temporal sync, smoothness priors, spectral coherence ￼ ￼.

⸻

4. Model Architecture
	•	Dual encoders:
	•	fgs1_encoder.py → denoise, bin, transit-aware sequence.
	•	airs_encoder.py → wavelength-structured CNN/GNN.
	•	Fusion: cross-attention or concat → latent joint space.
	•	Decoder: heteroscedastic regression head (μ, σ per bin).
	•	Constraints: symbolic loss engine (non-negativity, smoothness, coherence) ￼.

✅ Hybrid overlays: symbolic + neural → interpretable + leaderboard-safe.

⸻

5. Reproducibility & Lineage
	•	DVC pipelines: calibrate → preprocess → train → predict → submit.
	•	Event logs: structured JSONL (schemas/events.schema.json).
	•	Run manifests: config hash + git commit + artifact digests.
	•	CI/CD: Kaggle smoke tests, SBOM refresh, lint/type gates, determinism checks ￼ ￼.

✅ Every artifact, config, and run is lineage-linked.

⸻

6. Error Handling & UX
	•	Rich Typer CLI: autocompletion, colorized errors, typed exceptions.
	•	Runtime guardrails: loud failures on mis-configs or missing inputs.
	•	DVC runtime fences: protect against >9h Kaggle jobs.
	•	Pre-commit hooks: enforce lint, type, SBOM compliance ￼.

⸻

7. Challenge Alignment
	•	≤ 9h runtime on Kaggle GPUs.
	•	Offline-safe: no internet; bootstrap via bin/kaggle-boot.sh.
	•	283 spectral bins enforced via schema.
	•	μ/σ outputs with calibrated uncertainty (heteroscedastic regression) ￼.

⸻

⚡ Summary:
SpectraMind V50 = clean, reproducible, symbolic+neural architecture.
CLI-first orchestration + Hydra configs + DVC lineage + dual-channel encoders deliver scientific fidelity and competition performance under Kaggle constraints.

⸻