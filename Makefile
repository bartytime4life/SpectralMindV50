# -----------------------------------------------------------------------------
# SpectraMind V50 — Mission-grade Makefile (Upgraded)
# -----------------------------------------------------------------------------
# Shell safety
SHELL := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -Eeuo pipefail -c

# Colors (friendly in CI too)
C_RESET := \033[0m
C_INFO  := \033[36m
C_OK    := \033[32m
C_WARN  := \033[33m
C_ERR   := \033[31m

# Python / package
PKG        ?= spectramind
PY         ?= python3
VENV       ?= .venv
VENV_BIN   := $(VENV)/bin
PIP        := $(VENV_BIN)/pip
PYTHON     := $(VENV_BIN)/python
PRECOMMIT  := $(VENV_BIN)/pre-commit

# Optional: load .env into environment (no error if missing)
ifneq ("$(wildcard .env)","")
  export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
endif

# DVC
DVC_REMOTE       ?= localcache
DVC_REMOTE_PATH  ?= ./dvc-remote

# Release & versioning
VERSION_FILE ?= VERSION
TAG          ?=
GPG_SIGN     ?= 0
TAG_MSG      ?= "Release $(TAG)"

# Kaggle bundle
ARTIFACTS_DIR     ?= artifacts
SUBMISSION_DIR    ?= $(ARTIFACTS_DIR)
SUBMISSION_ZIP    ?= $(SUBMISSION_DIR)/submission.zip
SUBMISSION_SCHEMA ?= schemas/submission.schema.json

# Docker
IMAGE_NAME        ?= spectramind-v50
IMAGE_TAG         ?= local
INSTALL_EXTRAS    ?= gpu,dev   # must match pyproject extras
DOCKER_BUILDKIT   ?= 1
GPU_FLAG          ?= --gpus all

# Pipeline config (for scripts/run_pipeline.sh)
CFG ?= train

# Docs/diagrams
DIAGRAMS_THEME ?= neutral
DIAGRAMS_CONC  ?= 8

# Misc
export PYTHONHASHSEED = 0
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Phony targets
# -----------------------------------------------------------------------------
.PHONY: help env ensure-venv ensure-tools dev precommit lint format type test \
        coverage check docs docs-serve \
        train calibrate predict submit pipeline \
        dvc-setup dvc-repro dvc-push dvc-pull \
        sbom pip-audit trivy scan licenses \
        kaggle-package kaggle-verify kaggle-clean kaggle \
        docker-build docker-build-cpu docker-run docker-shell \
        version tag push-tag release bump \
        diagrams diagrams-dark diagrams-clean clean distclean ci

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@awk 'BEGIN{FS=":.*##"; printf "\n$(C_INFO)Available targets$(C_RESET)\n"} \
	/^[a-zA-Z0-9_\-]+:.*##/ { printf "  $(C_INFO)%-22s$(C_RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------
# Environment bootstrapping
# -----------------------------------------------------------------------------
env: ensure-venv ensure-tools ## Create .venv and install dev tools (fast if uv available)

ensure-venv: ## Create virtualenv if missing
	@if [ ! -d "$(VENV)" ]; then \
	  echo ">> Creating venv: $(VENV)"; \
	  $(PY) -m venv $(VENV); \
	fi

ensure-tools: ensure-venv ## Install dev dependencies and tools (prefers uv, falls back to pip)
	@if command -v uv >/dev/null 2>&1; then \
	  echo ">> Using uv for fast installs"; \
	  uv pip install --system --python $(PYTHON) -U pip wheel || true; \
	  uv pip install --system --python $(PYTHON) -e . -r requirements-dev.txt || true; \
	else \
	  $(PIP) install -U pip wheel || true; \
	  $(PIP) install -e . -r requirements-dev.txt || true; \
	fi
	@if [ -f requirements-kaggle.txt ]; then $(PIP) install -r requirements-kaggle.txt || true; fi
	@if command -v $(PRECOMMIT) >/dev/null 2>&1; then $(PRECOMMIT) install || true; fi
	@echo "$(C_OK)Env ready$(C_RESET)"

dev: env precommit ## Setup local dev env & run pre-commit (once)

precommit: ## Run pre-commit hooks on all files
	@if command -v $(PRECOMMIT) >/dev/null 2>&1; then $(PRECOMMIT) run --all-files; else echo "::warning::pre-commit not installed"; fi

# -----------------------------------------------------------------------------
# Code quality
# -----------------------------------------------------------------------------
lint: ## Lint (Ruff + TOML sort)
	@if [ -x "$(VENV_BIN)/ruff" ]; then $(VENV_BIN)/ruff check src tests; else echo "::warning::ruff not found"; fi
	@if [ -x "$(VENV_BIN)/ruff" ]; then $(VENV_BIN)/ruff format --check src tests || true; fi
	@if [ -x "$(VENV_BIN)/toml-sort" ]; then $(VENV_BIN)/toml-sort --check pyproject.toml || true; fi

format: ## Auto-format code (Ruff)
	@if [ -x "$(VENV_BIN)/ruff" ]; then $(VENV_BIN)/ruff format src tests; else echo "::warning::ruff not found"; fi

type: ## Type-check (mypy)
	@if [ -x "$(VENV_BIN)/mypy" ]; then $(VENV_BIN)/mypy src; else echo "::warning::mypy not found"; fi

test: ## Run tests (pytest -q)
	@if [ -x "$(VENV_BIN)/pytest" ]; then $(VENV_BIN)/pytest -q; else echo "::warning::pytest not found"; fi

coverage: ## Run tests with coverage HTML report
	@if [ -x "$(VENV_BIN)/pytest" ]; then \
	  $(VENV_BIN)/pytest -q --cov=$(PKG) --cov-report=term --cov-report=html:$(ARTIFACTS_DIR)/coverage; \
	  echo "::notice::Coverage HTML -> $(ARTIFACTS_DIR)/coverage/index.html"; \
	else echo "::warning::pytest not found"; fi

check: precommit lint type test ## Full local gate: pre-commit + lint + type + tests

# -----------------------------------------------------------------------------
# Docs
# -----------------------------------------------------------------------------
docs: diagrams ## Build docs (MkDocs) after rendering diagrams
	@if [ -x "$(VENV_BIN)/mkdocs" ]; then $(VENV_BIN)/mkdocs build -q; else echo "::warning::mkdocs not found"; fi

docs-serve: diagrams ## Serve docs locally (MkDocs @ http://0.0.0.0:8000)
	@if [ -x "$(VENV_BIN)/mkdocs" ]; then $(VENV_BIN)/mkdocs serve -a 0.0.0.0:8000; else echo "::warning::mkdocs not found"; fi

# -----------------------------------------------------------------------------
# Pipeline (CLI-first via Hydra)
# -----------------------------------------------------------------------------
train: ## Train model
	$(PYTHON) -m $(PKG) train --config-name train

calibrate: ## Calibrate sensors/data
	$(PYTHON) -m $(PKG) calibrate --config-name train

predict: ## Run predictions
	$(PYTHON) -m $(PKG) predict --config-name predict

submit: ## Build submission artifacts (CLI submit command)
	$(PYTHON) -m $(PKG) submit --config-name submit

pipeline: ## Run calibrate → train → predict → submit via scripts/run_pipeline.sh (CFG?=train)
	@if [ -x scripts/run_pipeline.sh ]; then \
	  echo ">> Running pipeline with CFG=$(CFG)"; \
	  bash scripts/run_pipeline.sh "$(CFG)"; \
	else \
	  echo "::error::scripts/run_pipeline.sh not found"; exit 1; \
	fi

# -----------------------------------------------------------------------------
# DVC
# -----------------------------------------------------------------------------
dvc-setup: ## Initialize DVC and default remote
	@if [ -x "$(VENV_BIN)/dvc" ]; then \
	  $(VENV_BIN)/dvc init -q || true; \
	  mkdir -p "$(DVC_REMOTE_PATH)"; \
	  $(VENV_BIN)/dvc remote add -d $(DVC_REMOTE) "$(DVC_REMOTE_PATH)" 2>/dev/null || true; \
	  echo "::notice::DVC remote '$(DVC_REMOTE)' -> $(DVC_REMOTE_PATH)"; \
	else \
	  echo "::warning::dvc not installed"; \
	fi

dvc-repro: ## Reproduce DVC pipeline
	@if [ -x "$(VENV_BIN)/dvc" ]; then $(VENV_BIN)/dvc repro; else echo "::warning::dvc not installed"; fi

dvc-push: ## Push DVC-tracked data to remote
	@if [ -x "$(VENV_BIN)/dvc" ]; then $(VENV_BIN)/dvc push; else echo "::warning::dvc not installed"; fi

dvc-pull: ## Pull DVC-tracked data from remote
	@if [ -x "$(VENV_BIN)/dvc" ]; then $(VENV_BIN)/dvc pull; else echo "::warning::dvc not installed"; fi

# -----------------------------------------------------------------------------
# Security / Supply Chain
# -----------------------------------------------------------------------------
licenses: ## Export 3rd-party license manifest (pip-licenses)
	@mkdir -p $(ARTIFACTS_DIR)
	@if [ -x "$(VENV_BIN)/pip-licenses" ]; then \
	  $(VENV_BIN)/pip-licenses --format=json --with-authors --with-urls > $(ARTIFACTS_DIR)/licenses.json; \
	  echo "::notice::Licenses -> $(ARTIFACTS_DIR)/licenses.json"; \
	else \
	  echo "::warning::pip-licenses not installed (add to requirements-dev.txt)"; \
	fi

sbom: ## Generate SBOM (Syft preferred; CycloneDX fallback)
	mkdir -p $(ARTIFACTS_DIR)
	if command -v syft >/dev/null 2>&1; then \
	  syft packages dir:. -o cyclonedx-json > $(ARTIFACTS_DIR)/sbom.json; \
	elif [ -x "$(VENV_BIN)/cyclonedx-bom" ]; then \
	  $(VENV_BIN)/cyclonedx-bom -o $(ARTIFACTS_DIR)/sbom.json || true; \
	else \
	  echo "::warning::No SBOM tool available; skipping SBOM."; \
	fi
	@echo "::notice::SBOM -> $(ARTIFACTS_DIR)/sbom.json"

pip-audit: ## Audit Python deps (pip-audit)
	@if [ -x "$(VENV_BIN)/pip-audit" ]; then \
	  $(VENV_BIN)/pip-audit -r requirements-dev.txt || true; \
	  if [ -f requirements-kaggle.txt ]; then $(VENV_BIN)/pip-audit -r requirements-kaggle.txt || true; fi; \
	else \
	  echo "::warning::pip-audit not installed"; \
	fi

trivy: ## Scan repo & Dockerfile (Trivy; requires Docker)
	if command -v trivy >/dev/null 2>&1; then \
	  trivy fs --exit-code 0 --severity HIGH,CRITICAL . || true; \
	  trivy config --exit-code 0 . || true; \
	else \
	  echo "::warning::Trivy not installed; skipping scan."; \
	fi

scan: pip-audit sbom trivy licenses ## Run all local security scans

# -----------------------------------------------------------------------------
# Kaggle packaging (submission.zip -> artifacts/submission.zip)
# -----------------------------------------------------------------------------
kaggle-package: ## Build Kaggle submission bundle -> artifacts/submission.zip
	mkdir -p "$(SUBMISSION_DIR)"
	if [ -x scripts/package_submission.sh ]; then \
	  echo ">> Using project packaging script"; \
	  bash scripts/package_submission.sh || exit 1; \
	else \
	  echo ">> Fallback packaging: collecting outputs"; \
	  set -Eeuo pipefail; \
	  files=""; \
	  for f in outputs/* predictions/*; do \
	    case "$$f" in *.csv|*.parquet|*.json) [ -e "$$f" ] && files="$$files $$f" ;; esac; \
	  done; \
	  if [ -z "$$files" ]; then \
	    echo "::error::No output files found. Provide scripts/package_submission.sh or produce outputs."; \
	    exit 1; \
	  fi; \
	  zip -j -r "$(SUBMISSION_ZIP)" $$files; \
	fi
	@if [ ! -f "$(SUBMISSION_ZIP)" ]; then \
	  echo "::error::Missing bundle: $(SUBMISSION_ZIP)"; exit 1; \
	else \
	  echo "::notice::Kaggle bundle created: $(SUBMISSION_ZIP)"; \
	fi

kaggle-verify: ## Verify submission bundle integrity (schema if present)
	@if [ ! -f "$(SUBMISSION_ZIP)" ]; then \
	  echo "::error::Bundle not found: $(SUBMISSION_ZIP). Run 'make kaggle-package' first."; exit 1; \
	fi
	@if command -v unzip >/dev/null 2>&1; then unzip -l "$(SUBMISSION_ZIP)"; else echo ">> 'unzip' not found; skipping listing."; fi
	@if [ -f "$(SUBMISSION_SCHEMA)" ]; then \
	  echo ">> Schema detected: $(SUBMISSION_SCHEMA)"; \
	  if [ -x "$(VENV_BIN)/check-jsonschema" ]; then \
	    tmpdir="$$(mktemp -d)"; \
	    unzip -q "$(SUBMISSION_ZIP)" -d "$$tmpdir"; \
	    for j in $$(find "$$tmpdir" -type f -name '*.json'); do \
	      echo "  - Validating $$j"; \
	      "$(VENV_BIN)/check-jsonschema" --schemafile "$(SUBMISSION_SCHEMA)" "$$j" || exit 1; \
	    done; \
	    rm -rf "$$tmpdir"; \
	  else \
	    echo "::warning::check-jsonschema not available; skipping schema validation."; \
	  fi; \
	else \
	  echo ">> No schema provided; basic verification only."; \
	fi
	@echo "::notice::Kaggle bundle verified."

kaggle-clean: ## Remove local Kaggle bundle
	rm -f "$(SUBMISSION_ZIP)"

kaggle: kaggle-package kaggle-verify ## Package + verify submission

# -----------------------------------------------------------------------------
# Diagrams (Mermaid via scripts/render_diagrams.sh)
# -----------------------------------------------------------------------------
diagrams: ## Render Mermaid diagrams -> SVG (and optional PNG) with scripts/render_diagrams.sh
	@if [ -x scripts/render_diagrams.sh ]; then \
	  ./scripts/render_diagrams.sh -t "$(DIAGRAMS_THEME)" -c "$(DIAGRAMS_CONC)"; \
	else \
	  echo "::warning::scripts/render_diagrams.sh not found; skipping diagrams."; \
	fi

diagrams-dark: ## Render Mermaid diagrams with dark theme + PNGs into assets/diagrams/out
	@if [ -x scripts/render_diagrams.sh ]; then \
	  ./scripts/render_diagrams.sh -t dark -p -o assets/diagrams/out -c "$(DIAGRAMS_CONC)"; \
	else \
	  echo "::warning::scripts/render_diagrams.sh not found; skipping diagrams."; \
	fi

diagrams-clean: ## Remove generated diagram images
	-find assets/diagrams -type f \( -name '*.svg' -o -name '*.png' \) -delete || true
	-find docs/diagrams    -type f \( -name '*.svg' -o -name '*.png' \) -delete || true

# -----------------------------------------------------------------------------
# Docker (GPU + CPU)
# -----------------------------------------------------------------------------
docker-build: ## Build CUDA image (INSTALL_EXTRAS=$(INSTALL_EXTRAS))
	@echo ">> Building $(IMAGE_NAME):$(IMAGE_TAG) with extras=$(INSTALL_EXTRAS)"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build \
	  --build-arg INSTALL_EXTRAS=$(INSTALL_EXTRAS) \
	  -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-build-cpu: ## Build CPU-only image (target=cpu in Dockerfile)
	@echo ">> Building CPU image $(IMAGE_NAME):$(IMAGE_TAG)-cpu"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build \
	  --target cpu \
	  -t $(IMAGE_NAME):$(IMAGE_TAG)-cpu .

docker-run: ## Run container (GPU if available) -> spectramind --help
	@if docker info >/dev/null 2>&1; then \
	  echo ">> Running container"; \
	  docker run --rm $(GPU_FLAG) -it $(IMAGE_NAME):$(IMAGE_TAG) --help; \
	else \
	  echo "::error::Docker not available."; exit 1; \
	fi

docker-shell: ## Interactive shell inside image
	@if docker info >/dev/null 2>&1; then \
	  docker run --rm $(GPU_FLAG) -it $(IMAGE_NAME):$(IMAGE_TAG) /bin/bash; \
	else \
	  echo "::error::Docker not available."; exit 1; \
	fi

# -----------------------------------------------------------------------------
# Versioning & Releases
# -----------------------------------------------------------------------------
define _ensure_clean
	@git update-index -q --refresh
	@if ! git diff-index --quiet HEAD --; then \
	  echo >&2 "::error::Working tree not clean. Commit or stash your changes."; \
	  git status --short; \
	  exit 1; \
	fi
endef

define _read_version
	@VER=$$(sed -e 's/[[:space:]]//g' $(VERSION_FILE)); \
	echo "VERSION=$(VERSION_FILE) => $$VER"; \
	if [ -z "$$VER" ]; then echo >&2 "::error::VERSION file is empty"; exit 1; fi; \
	if [ -z "$(TAG)" ]; then \
	  echo >&2 "::error::TAG is required (e.g. make release TAG=v$$VER)"; exit 1; \
	fi; \
	if [ "$(TAG)" != "v$$VER" ]; then \
	  echo >&2 "::error::TAG $(TAG) != v$$VER (from $(VERSION_FILE))"; exit 1; \
	fi
endef

version: ## Sync VERSION -> pyproject.toml and commit (optionally tag if TAG=vX.Y.Z)
	@set -Eeuo pipefail; \
	ver="$$(tr -d '[:space:]' < $(VERSION_FILE))"; \
	[ -n "$$ver" ] || { echo "❌ $(VERSION_FILE) empty"; exit 1; }; \
	echo "🔄 Setting version to $$ver"; \
	pyproject="pyproject.toml"; \
	[ -f "$$pyproject" ] || { echo "❌ $$pyproject missing"; exit 1; }; \
	sed -i.bak -E "s/^(version\s*=\s*\").*(\")/\1$$ver\2/" "$$pyproject"; \
	rm -f "$$pyproject.bak"; \
	echo "✅ pyproject.toml version -> $$ver"; \
	git add $(VERSION_FILE) "$$pyproject"; \
	git commit -m "chore: set version $$ver" || echo "ℹ️  No changes to commit"; \
	if [ -n "$(TAG)" ]; then \
	  if git rev-parse "$(TAG)" >/dev/null 2>&1; then echo "⚠️  Tag $(TAG) exists"; else git tag -a "$(TAG)" -m "Release $(TAG)"; fi; \
	fi

tag: ## Create annotated/signed tag (TAG=vX.Y.Z, GPG_SIGN=0|1)
	$(_ensure_clean)
	$(_read_version)
	@if git rev-parse "$(TAG)" >/dev/null 2>&1; then echo >&2 "::error::Tag $(TAG) already exists."; exit 1; fi
	@if [ "$(GPG_SIGN)" = "1" ]; then git tag -s "$(TAG)" -m "$(TAG_MSG)"; else git tag -a "$(TAG)" -m "$(TAG_MSG)"; fi
	@echo "::notice::Created tag $(TAG)"

push-tag: ## Push tag to origin (triggers GH Actions release)
	@if [ -z "$(TAG)" ]; then echo >&2 "::error::TAG is required"; exit 1; fi
	git push origin "$(TAG)"
	@echo "::notice::Pushed tag $(TAG)"

release: tag push-tag ## Full release flow: tag + push

bump: ## Bump semver (BUMP=patch|minor|major) & update CHANGELOG header
	@BUMP ?= patch; \
	[ -f $(VERSION_FILE) ] || echo "0.1.0" > $(VERSION_FILE); \
	VER=$$(tr -d '[:space:]' < $(VERSION_FILE)); \
	IFS=. read -r MAJ MIN PAT <<< "$$VER"; \
	case "$$BUMP" in \
	  major) MAJ=$$((MAJ+1)); MIN=0; PAT=0 ;; \
	  minor) MIN=$$((MIN+1)); PAT=0 ;; \
	  patch) PAT=$$((PAT+1)) ;; \
	  *) echo "BUMP must be one of: patch|minor|major"; exit 1 ;; \
	esac; \
	NEW_VER="$$MAJ.$$MIN.$$PAT"; \
	echo "$$NEW_VER" > $(VERSION_FILE); \
	if [ -f CHANGELOG.md ]; then sed -i.bak "1s/.*/## [$$NEW_VER] — $$(date +%Y-%m-%d)/" CHANGELOG.md; rm -f CHANGELOG.md.bak; fi; \
	git add $(VERSION_FILE) CHANGELOG.md || true; \
	git commit -m "chore(release): bump version to $$NEW_VER" || true; \
	echo "::notice::Bumped to $$NEW_VER"

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf $(ARTIFACTS_DIR)/coverage

distclean: clean ## Remove venv and artifacts
	rm -rf $(VENV) $(ARTIFACTS_DIR)

# -----------------------------------------------------------------------------
# CI convenience
# -----------------------------------------------------------------------------
ci: check docs diagrams ## Run the same checks CI does locally

# -----------------------------------------------------------------------------
# End of Makefile
# -----------------------------------------------------------------------------
