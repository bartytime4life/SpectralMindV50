# Kaggle Input Symlinks — Fast, Zero-Copy Mounts to `/kaggle/input`

Kaggle mounts competition data and attached datasets read-only under `/kaggle/input/*`.
Our code and configs often expect project-relative paths (e.g., `data/raw`, `data/processed`).
This note shows how to **symlink** Kaggle inputs into the repo layout so everything “just works” without copying gigabytes.

---

## TL;DR

* **Never write to** `/kaggle/input/*` — it’s read-only.
* Create **symlinks in your working dir** (e.g., `/kaggle/working/spectramind-v50/data/raw → /kaggle/input/ariel-data-challenge-2025`) so your existing paths resolve.
* Prefer **relative symlinks**, fall back to **copy** only if the environment does not allow symlinks.

---

## Common Paths on Kaggle

| What                  | Where it is on Kaggle                      | Notes                                            |
| --------------------- | ------------------------------------------ | ------------------------------------------------ |
| Competition input     | `/kaggle/input/ariel-data-challenge-2025/` | Name varies; check the sidebar “Input” panel.    |
| Attached dataset      | `/kaggle/input/<dataset-slug>/`            | One folder per dataset you attach in the editor. |
| Working directory     | `/kaggle/working/`                         | Read-write; persisted as notebook output.        |
| Repo code (if zipped) | `/kaggle/working/<repo>` (after you unzip) | Make symlinks relative to here.                  |

---

## Minimal Shell Recipe (Notebook cell)

```bash
# 1) Ensure an expected repo root
REPO=/kaggle/working/spectramind-v50
mkdir -p "$REPO"

# (Optional) unzip your repo bundle if you uploaded one as an input dataset
# unzip -q /kaggle/input/spectramind-v50-repo/spectramind-v50.zip -d /kaggle/working

# 2) Create the internal data layout (empty dirs that will host links)
mkdir -p "$REPO/data/raw" "$REPO/data/interim" "$REPO/data/processed" "$REPO/data/external"

# 3) Map Kaggle inputs -> repo structure via symlinks (adjust dataset names as needed)
ln -sfn /kaggle/input/ariel-data-challenge-2025             "$REPO/data/raw/adc2025"
# Example for an additional attached dataset:
# ln -sfn /kaggle/input/my-aux-dataset                      "$REPO/data/external/my-aux"

# 4) (Optional) Create short convenience links at repo root
ln -sfn /kaggle/input/ariel-data-challenge-2025             "$REPO/.kaggle_input"

# 5) Verify
ls -lah "$REPO/data/raw"
```

* `ln -sfn` is safe to re-run; it **overwrites stale links** atomically.
* Keep **dataset slugs** updated to match what you attached in the notebook (`Input` panel).

---

## Python Helper (Symlink with Fallback Copy)

> Use this if you want a resilient cell that still works in environments without symlink support.

```python
from pathlib import Path
import os, shutil, errno

def ensure_link(target: Path, link_path: Path):
    target = Path(target)
    link_path = Path(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            else:
                shutil.rmtree(link_path)
        except Exception:
            pass
    try:
        # Prefer relative symlinks
        rel = os.path.relpath(target, start=link_path.parent)
        os.symlink(rel, link_path, target_is_directory=target.is_dir())
        print(f"Symlinked {link_path} -> {rel}")
    except OSError as e:
        if e.errno in (errno.EPERM, errno.EACCES, errno.EWINDOWS):  # fallback (e.g., Windows)
            print(f"Symlink denied; copying {target} -> {link_path}")
            if target.is_dir():
                shutil.copytree(target, link_path)
            else:
                shutil.copy2(target, link_path)
        else:
            raise

REPO = Path("/kaggle/working/spectramind-v50")
ensure_link(Path("/kaggle/input/ariel-data-challenge-2025"), REPO/"data/raw/adc2025")
```

---

## Typical Maps for SpectraMind V50

> Adjust to match your Hydra configs (e.g., `configs/data/kaggle.yaml`)

```bash
# Raw competition data
ln -sfn /kaggle/input/ariel-data-challenge-2025  /kaggle/working/spectramind-v50/data/raw/adc2025

# External metadata (if attached)
# ln -sfn /kaggle/input/ariel-axis-info           /kaggle/working/spectramind-v50/data/external/axis

# Outputs (read-write)
mkdir -p /kaggle/working/spectramind-v50/artifacts
mkdir -p /kaggle/working/spectramind-v50/models
```

Then point configs to repo-relative paths (e.g., `data.raw_dir: ${repo}/data/raw/adc2025`).

---

## Verifying Symlinks

```bash
# Show symlink targets
find /kaggle/working/spectramind-v50/data -maxdepth 2 -type l -ls

# Quick sanity on expected files
ls -lah /kaggle/working/spectramind-v50/data/raw/adc2025 | head -n 20
```

If you see “No such file or directory” when following a link, the **target dataset slug is wrong** or not attached.

---

## Notes & Gotchas

* **Symlinks are allowed** in Kaggle runtimes and work well inside `/kaggle/working`.
  They **cannot** be created within `/kaggle/input` (read-only).
* If you **export a dataset** from notebook outputs, symlinks are typically **materialized** (copied) or **omitted** depending on packing rules; treat symlinks as **runtime-only mounts**.
* On **Windows local dev**, symlink creation may require admin rights; use the Python fallback to copy.
* Tools like **DVC** or some archivers may **not preserve symlinks**; prefer absolute/relative paths at runtime and avoid archiving symlinks unless you know the behavior.
* Hydra tips:

  * Keep a single `${repo}` root var and define all data paths as `${repo}/data/...`.
  * Provide a **`kaggle.yaml`** profile that **assumes these mounts** to avoid per-cell path tweaks.

---

## One-Liner Bootstrap (Paste in first cell)

```bash
REPO=/kaggle/working/spectramind-v50
mkdir -p "$REPO/data/raw" "$REPO/data/interim" "$REPO/data/processed" "$REPO/data/external" "$REPO/artifacts" "$REPO/models"
ln -sfn /kaggle/input/ariel-data-challenge-2025  "$REPO/data/raw/adc2025"
echo "Mounted Kaggle inputs into $REPO"
```

---

## FAQ

**Q:** Why symlink instead of copying?
**A:** Copying large competition data wastes time & disk. Symlinks are instant and keep a single source of truth.

**Q:** Can I symlink the other way (from `/kaggle/input` to `/kaggle/working`)?
**A:** No — `/kaggle/input` is read-only. Create links **inside** `/kaggle/working`.

**Q:** Will this persist across sessions?
**A:** The **links** persist only if you save the working dir as output; but they will likely be **materialized** into copies when exported. Recreate links in the first cell of any notebook.

---

Happy mounting!
