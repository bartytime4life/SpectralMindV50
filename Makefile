# -----------------------------------------------------------------------------
# SpectraMind V50 — Mission-grade Makefile
# -----------------------------------------------------------------------------
# Shell safety
SHELL := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -eo pipefail -c

# Python / package
PKG        ?= spectramind
PY         ?= python
VENV       ?= .venv
VENV_BIN   := $(VENV)/bin
PIP        := $(VENV_BIN)/pip
PYTHON     := $(VENV_BIN)/python
PRECOMMIT  := $(VENV_BIN)/pre-commit

# DVC
DVC_REMOTE ?= localcache

# Release
VERSION_FILE ?= VERSION
TAG          ?=
GPG_SIGN     ?= 0
TAG_MSG      ?= "Release $(TAG)"

# Misc
export PYTHONHASHSEED = 0
ARTIFACTS_DIR := artifacts

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Phony targets
# -----------------------------------------------------------------------------
.PHONY: help dev precommit lint format type test check docs docs-serve \
        train calibrate predict submit \
        dvc-setup dvc-repro dvc-push dvc-pull \
        sbom pip-audit trivy scan \
        tag push-tag release bump \
        clean distclean

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@awk 'BEGIN{FS=":.*##"; printf "\n\033[1mAvailable targets\033[0m\n"} \
	/^[a-zA-Z0-9_\-]+:.*##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------
# Dev / toolchain
# -----------------------------------------------------------------------------
dev: ## Setup local dev environment (.venv, dev deps, pre-commit)
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip wheel
	$(PIP) install -e . -r requirements-dev.txt
	$(PRECOMMIT) install

precommit: ## Run pre-commit hooks on all files
	$(PRECOMMIT) run --all-files

lint: ## Lint (Ruff, TOML sort check)
	$(VENV_BIN)/ruff check src tests
	$(VENV_BIN)/ruff format --check src tests || true
	$(VENV_BIN)/toml-sort --check pyproject.toml || true

format: ## Auto-format code (Ruff)
	$(VENV_BIN)/ruff format src tests

type: ## Type-check (mypy)
	$(VENV_BIN)/mypy src

test: ## Run tests (pytest -q)
	$(VENV_BIN)/pytest -q

check: precommit lint type test ## Full local gate: pre-commit + lint + type + tests

docs: ## Build docs (MkDocs)
	$(VENV_BIN)/mkdocs build -q

docs-serve: ## Serve docs locally (MkDocs)
	$(VENV_BIN)/mkdocs serve -a 0.0.0.0:8000

# -----------------------------------------------------------------------------
# Pipeline (CLI-first via Hydra)
# -----------------------------------------------------------------------------
train: ## Train model
	$(PYTHON) -m $(PKG) train --config-name train

calibrate: ## Calibrate sensors/data
	$(PYTHON) -m $(PKG) calibrate --config-name train

predict: ## Run predictions
	$(PYTHON) -m $(PKG) predict --config-name predict

submit: ## Build submission artifacts
	$(PYTHON) -m $(PKG) submit --config-name submit

# -----------------------------------------------------------------------------
# DVC
# -----------------------------------------------------------------------------
dvc-setup: ## Initialize DVC and default remote
	$(VENV_BIN)/dvc init -q || true
	$(VENV_BIN)/dvc remote add -d $(DVC_REMOTE) ./dvc-remote 2>/dev/null || true

dvc-repro: ## Reproduce DVC pipeline
	$(VENV_BIN)/dvc repro

dvc-push: ## Push DVC-tracked data to remote
	$(VENV_BIN)/dvc push

dvc-pull: ## Pull DVC-tracked data from remote
	$(VENV_BIN)/dvc pull

# -----------------------------------------------------------------------------
# Security / Supply Chain
# -----------------------------------------------------------------------------
sbom: ## Generate SBOM (CycloneDX via Syft)
	mkdir -p $(ARTIFACTS_DIR)
	if command -v syft >/dev/null 2>&1; then \
	  syft packages dir:. -o cyclonedx-json > $(ARTIFACTS_DIR)/sbom.json; \
	else \
	  echo "Syft not installed; skipping SBOM generation."; \
	fi

pip-audit: ## Audit Python deps (pip-audit)
	$(VENV_BIN)/pip-audit -r requirements-dev.txt || true
	if [ -f requirements-kaggle.txt ]; then $(VENV_BIN)/pip-audit -r requirements-kaggle.txt || true; fi

trivy: ## Scan repo & Dockerfile (Trivy; requires Docker + Trivy)
	if command -v trivy >/dev/null 2>&1; then \
	  trivy fs --exit-code 0 --severity HIGH,CRITICAL . || true; \
	  if [ -f Dockerfile ]; then trivy config --exit-code 0 . || true; fi \
	else \
	  echo "Trivy not installed; skipping scan."; \
	fi

scan: pip-audit sbom trivy ## Run all local security scans

# -----------------------------------------------------------------------------
# Releases (pairs with .github/workflows/release.yml)
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
	if [ -z "$(TAG)" ]; then \
	  echo >&2 "::error::TAG is required (e.g. make release TAG=v$$VER)"; exit 1; \
	fi; \
	if [ "$(TAG)" != "v$$VER" ]; then \
	  echo >&2 "::error::TAG $(TAG) != v$$VER (from $(VERSION_FILE))"; exit 1; \
	fi
endef

tag: ## Create annotated/signed tag (TAG=vX.Y.Z, GPG_SIGN=0|1)
	$(_ensure_clean)
	$(_read_version)
	@if git rev-parse "$(TAG)" >/dev/null 2>&1; then \
	  echo >&2 "::error::Tag $(TAG) already exists."; exit 1; \
	fi
	@if [ "$(GPG_SIGN)" = "1" ]; then \
	  git tag -s "$(TAG)" -m "$(TAG_MSG)"; \
	else \
	  git tag -a "$(TAG)" -m "$(TAG_MSG)"; \
	fi
	@echo "::notice::Created tag $(TAG)"

push-tag: ## Push tag to origin (triggers GH Actions release)
	@if [ -z "$(TAG)" ]; then echo >&2 "::error::TAG is required"; exit 1; fi
	git push origin "$(TAG)"
	@echo "::notice::Pushed tag $(TAG)"

release: tag push-tag ## Full release flow: tag + push

bump: ## Bump semantic version (patch|minor|major) & update CHANGELOG heading
	@BUMP ?= patch; \
	if [ ! -f $(VERSION_FILE) ]; then echo "0.1.0" > $(VERSION_FILE); fi; \
	VER=$$(sed -e 's/[[:space:]]//g' $(VERSION_FILE)); \
	IFS=. read -r MAJ MIN PAT <<< "$$VER"; \
	case "$$BUMP" in \
	  major) MAJ=$$((MAJ+1)); MIN=0; PAT=0 ;; \
	  minor) MIN=$$((MIN+1)); PAT=0 ;; \
	  patch) PAT=$$((PAT+1)) ;; \
	  *) echo "BUMP must be one of: patch|minor|major"; exit 1 ;; \
	esac; \
	NEW_VER="$$MAJ.$$MIN.$$PAT"; \
	echo "$$NEW_VER" > $(VERSION_FILE); \
	sed -i.bak "1s/.*/## [$$NEW_VER] — $$(date +%Y-%m-%d)/" CHANGELOG.md || true; rm -f CHANGELOG.md.bak; \
	git add $(VERSION_FILE) CHANGELOG.md || true; \
	git commit -m "chore(release): bump version to $$NEW_VER" || true; \
	echo "::notice::Bumped to $$NEW_VER"

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

distclean: clean ## Remove venv and artifacts
	rm -rf $(VENV) $(ARTIFACTS_DIR)

# -----------------------------------------------------------------------------
# CI convenience (optional)
# -----------------------------------------------------------------------------
ci: check docs ## Run the same checks CI does locally

# -----------------------------------------------------------------------------
# End of Makefile
# -----------------------------------------------------------------------------
