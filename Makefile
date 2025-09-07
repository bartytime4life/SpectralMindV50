SHELL:=/usr/bin/env bash
.ONESHELL:
.SHELLFLAGS:=-eo pipefail -c
PY:=python
PKG:=spectramind
VENV?=.venv
DVC_REMOTE?=localcache

export PYTHONHASHSEED=0

.PHONY: dev precommit lint type test docs train calibrate predict submit dvc-setup sbom

dev:
	python -m venv $(VENV)
	$(VENV)/bin/pip install -U pip wheel
	$(VENV)/bin/pip install -e . -r requirements-dev.txt
	$(VENV)/bin/pre-commit install

precommit:
	pre-commit run --all-files

lint:
	ruff check src tests
	ruff format --check src tests || true
	toml-sort --check pyproject.toml || true

type:
	mypy src

test:
	pytest -q

train:
	$(PY) -m $(PKG) train --config-name train

calibrate:
	$(PY) -m $(PKG) calibrate --config-name train

predict:
	$(PY) -m $(PKG) predict --config-name predict

submit:
	$(PY) -m $(PKG) submit --config-name submit

dvc-setup:
	dvc init -q || true
	dvc remote add -d $(DVC_REMOTE) ./dvc-remote 2>/dev/null || true

sbom:
	syft packages dir:. -o cyclonedx-json > artifacts/sbom.json || true
