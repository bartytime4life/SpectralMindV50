# 🚀 Pull Request

## Summary
<!-- 1–3 sentences: what does this PR change and why? Link to issue(s) if applicable. -->
Fixes #<issue-id> (optional)

---

## Type of change
<!-- Select all that apply -->
- [ ] Feature / enhancement
- [ ] Bug fix
- [ ] Documentation only
- [ ] Refactor (no behavior change)
- [ ] Performance / memory
- [ ] Security / hardening
- [ ] CI/CD or DevEx
- [ ] Other:

---

## Scope of changes
<!-- Check the touched surfaces and list primary entry points / files. -->
- [ ] CLI (`spectramind <subcommand>`)
- [ ] Hydra configs (`configs/**.yaml`)
- [ ] Pipeline stages (`src/spectramind/pipeline/*`)
- [ ] Models / encoders / decoders (`src/spectramind/models/*`)
- [ ] Losses / physics constraints (smoothness, non-negativity, band-coherence, calibration)
- [ ] Diagnostics / reports (`src/spectramind/diagnostics/*`)
- [ ] DVC pipeline / data lineage (`dvc.yaml`, artifacts)
- [ ] Docs / diagrams (Mermaid in Markdown)
- [ ] Tests (unit / integration / CLI)
- [ ] GitHub Actions / CI
- [ ] Other:

Key files:
- …

---

## Reproducibility & Config-as-Code
<!-- Explain how to run and reproduce. Include exact CLI + Hydra overrides used. -->
**Example commands**
```bash
# Calibrate → Train → Predict (example)
spectramind calibrate +env=kaggle
spectramind train    +data.kaggle=true trainer.max_epochs=5
spectramind predict  ckpt=artifacts/model.ckpt

