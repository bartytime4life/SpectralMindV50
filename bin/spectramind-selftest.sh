#!/usr/bin/env bash
set -euo pipefail
spectramind --help >/dev/null
pytest -q
