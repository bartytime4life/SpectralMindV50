Kaggle Input Symlinks — Fast, Zero-Copy Mounts to /kaggle/input

Kaggle mounts competition data and attached datasets read-only under /kaggle/input/*.
Your repo usually expects project-relative paths (e.g. data/raw, data/processed).
This guide shows how to symlink Kaggle inputs into your repo layout so everything “just works” without copying gigabytes.

⸻

TL;DR
	•	Never write to /kaggle/input/* — it’s read-only.
	•	Create symlinks inside your working dir (e.g. /kaggle/working/<repo>) that point to /kaggle/input/*.
	•	Prefer relative symlinks; fallback to copy only if the environment disallows symlinks (e.g., Windows).

⸻

Common Paths on Kaggle

What	Path on Kaggle	Notes
Competition input	/kaggle/input/ariel-data-challenge-2025/	Name varies; check the right-hand “Input” panel.
Attached dataset	/kaggle/input/<dataset-slug>/	One folder per dataset you attach in the editor.
Working directory	/kaggle/working/	Read-write; persisted as notebook output.
Repo code (unzipped)	/kaggle/working/<repo>	Unzip here, then make symlinks relative to this folder.


⸻

Minimal Shell Recipe (Notebook cell)

Paste into the first Code cell of your Kaggle notebook.

# 1) Ensure repo root under /kaggle/working
REPO=/kaggle/working/spectramind-v50
mkdir -p "$REPO"

# (Optional) unzip your repo if uploaded as a dataset
# unzip -q /kaggle/input/spectramind-v50-repo/spectramind-v50.zip -d /kaggle/working

# 2) Create expected data layout
mkdir -p "$REPO/data/raw" "$REPO/data/interim" "$REPO/data/processed" "$REPO/data/external"
mkdir -p "$REPO/artifacts" "$REPO/models"

# 3) Map Kaggle inputs -> repo structure via symlinks (adjust dataset slugs as needed)
ln -sfn /kaggle/input/ariel-data-challenge-2025  "$REPO/data/raw/adc2025"
# Example for an additional attached dataset:
# ln -sfn /kaggle/input/my-aux-dataset            "$REPO/data/external/aux"

# 4) (Optional) convenience link at repo root
ln -sfn /kaggle/input/ariel-data-challenge-2025  "$REPO/.kaggle_input"

# 5) Verify
echo "[Mapped inputs]"
ls -lah "$REPO/data/raw" || true

	•	ln -sfn is idempotent — re-running the cell updates the link atomically.
	•	Keep dataset slugs in sync with what you attached in the notebook (Input panel).

⸻

Python Helper (Symlink with Fallback to Copy)

Use this if you want a resilient setup that still works in environments without symlink support.

from pathlib import Path
import os, shutil, errno

def ensure_link(target: Path, link_path: Path):
    target = Path(target)
    link_path = Path(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file/dir/symlink
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            else:
                shutil.rmtree(link_path)
        except Exception:
            pass

    try:
        # Prefer relative symlink (portable within /kaggle/working)
        rel = os.path.relpath(target, start=link_path.parent)
        os.symlink(rel, link_path, target_is_directory=target.is_dir())
        print(f"Symlinked {link_path} -> {rel}")
    except OSError as e:
        # Fallback: copy if symlink permission is denied
        if e.errno in (errno.EPERM, errno.EACCES, getattr(errno, "EWINDOWS", 10000)):
            print(f"Symlink denied; copying {target} -> {link_path}")
            if target.is_dir():
                shutil.copytree(target, link_path)
            else:
                shutil.copy2(target, link_path)
        else:
            raise

REPO = Path("/kaggle/working/spectramind-v50")
ensure_link(Path("/kaggle/input/ariel-data-challenge-2025"), REPO/"data/raw/adc2025")


⸻

Typical Maps for SpectraMind V50

Adjust to match your Hydra configs (e.g., configs/data/kaggle.yaml).

# Raw competition data (read-only via symlink)
ln -sfn /kaggle/input/ariel-data-challenge-2025  /kaggle/working/spectramind-v50/data/raw/adc2025

# External metadata (if attached)
# ln -sfn /kaggle/input/ariel-axis-info  /kaggle/working/spectramind-v50/data/external/axis

# Writable outputs
mkdir -p /kaggle/working/spectramind-v50/artifacts
mkdir -p /kaggle/working/spectramind-v50/models

Config tip (Hydra)

# configs/data/kaggle.yaml
repo: /kaggle/working/spectramind-v50
data:
  raw_dir:       ${repo}/data/raw/adc2025
  interim_dir:   ${repo}/data/interim
  processed_dir: ${repo}/data/processed
external:
  axis_dir:      ${repo}/data/external/axis
artifacts_dir:    ${repo}/artifacts
models_dir:       ${repo}/models


⸻

Verifying Symlinks

# Show symlink targets (short)
find /kaggle/working/spectramind-v50/data -maxdepth 2 -type l -ls

# Quick sanity on expected files
ls -lah /kaggle/working/spectramind-v50/data/raw/adc2025 | head -n 20

If you see “No such file or directory” following a link, the target dataset slug is wrong or not attached in the notebook.

⸻

Notes & Gotchas
	•	Symlinks are allowed in Kaggle and work well under /kaggle/working.
You cannot create links inside /kaggle/input (read-only).
	•	If you export a dataset from notebook outputs, symlinks may be materialized (copied) or omitted — treat them as runtime mounts and recreate them on startup.
	•	On Windows local dev, symlink creation may require admin rights; use the Python helper’s copy fallback.
	•	Some tools (e.g., certain archivers, DVC defaults) may not preserve symlinks — prefer runtime linkage and avoid archiving symlinks.

⸻

One-Liner Bootstrap (Paste in first cell)

REPO=/kaggle/working/spectramind-v50
mkdir -p "$REPO/data/raw" "$REPO/data/interim" "$REPO/data/processed" "$REPO/data/external" "$REPO/artifacts" "$REPO/models"
ln -sfn /kaggle/input/ariel-data-challenge-2025  "$REPO/data/raw/adc2025"
echo "Mounted Kaggle inputs into $REPO"


⸻

FAQ

Q: Why symlink instead of copying?
A: Copying large competition data wastes time & disk. Symlinks are instant and keep a single source of truth.

Q: Can I symlink the other way (from /kaggle/input to /kaggle/working)?
A: No — /kaggle/input is read-only. Create links inside /kaggle/working.

Q: Will links persist across sessions?
A: Only if you save the working dir as an output; when exported, links may be materialized or dropped. Always recreate links at startup (first cell).

Q: How do I make configs portable?
A: Define a single ${repo} root var (Hydra) and reference ${repo}/data/.... Provide a kaggle.yaml profile that assumes these mounts.

⸻

Happy mounting!
