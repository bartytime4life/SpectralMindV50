#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-data/calibrated}"
DST="${2:-artifacts/kaggle/precalibrated.zip}"

mkdir -p "$(dirname "$DST")"
if [ ! -d "$SRC" ]; then
  echo "Source directory '$SRC' not found." >&2
  exit 1
fi

( cd "$SRC" && zip -r -q "$(realpath -m "$DST")" . )
echo "Wrote $(du -h "$DST" | awk '{print $1}')" "→ $DST"
