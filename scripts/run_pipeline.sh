#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-train}
spectramind calibrate --config-name "$CFG"
spectramind train     --config-name "$CFG"
spectramind predict   --config-name predict
spectramind submit    --config-name submit
