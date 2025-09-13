#!/usr/bin/env python
from __future__ import annotations
import re, sys, pathlib, tomllib

root = pathlib.Path(__file__).resolve().parents[1]
py = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
v_pyproj = py["project"]["version"]

v_file = (root / "VERSION").read_text("utf-8").strip()
init = (root / "src/spectramind/__init__.py").read_text("utf-8")
m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
v_init = m.group(1) if m else ""

mismatch = []
if v_pyproj != v_file:
    mismatch.append(f"VERSION != pyproject ({v_file} vs {v_pyproj})")
if v_pyproj != v_init:
    mismatch.append(f"__init__ != pyproject ({v_init} vs {v_pyproj})")

if mismatch:
    print("Version mismatch:\n - " + "\n - ".join(mismatch))
    sys.exit(1)
print(f"Version OK: {v_pyproj}")
