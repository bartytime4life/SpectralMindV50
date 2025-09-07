#!/usr/bin/env bash
set -euo pipefail
NEW=${1:?new version required}
echo "$NEW" > VERSION
git add VERSION
git commit -m "chore(version): bump to $NEW"
