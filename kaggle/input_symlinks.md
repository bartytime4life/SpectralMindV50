# Kaggle Input Symlinks — Fast, Zero-Copy Mounts to `/kaggle/input`

Kaggle mounts competition and attached datasets **read-only** in `/kaggle/input/*`.
Your repo likely expects a project tree (e.g., `data/raw`, `data/processed`) under **`/kaggle/working/<repo>`**.

**Use symlinks** from your repo tree to `/kaggle/input/*` so your code “just works” (no gigabyte copies).

---

## TL;DR

* ✅ **Never write** to `/kaggle/input/*` (read-only).
* ✅ Make **symlinks** from `/kaggle/working/<repo>` to `/kaggle/input/<dataset>` (zero-copy).
* ✅ Prefer **relative links** (robust), and **fallback to copy** if symlinks aren’t allowed (e.g., Windows).

---

## Common Kaggle Paths

| What                 | Path on Kaggle                             | Notes                                             |
| -------------------- | ------------------------------------------ | ------------------------------------------------- |
| Competition input    | `/kaggle/input/ariel-data-challenge-2025/` | Name varies — check the right-hand “Input” panel. |
| Attached dataset     | `/kaggle/input/<dataset-slug>/`            | One folder per dataset you attach.                |
| Working directory    | `/kaggle/working/`                         | Read-write; notebook outputs are saved here.      |
| Repo root (unzipped) | `/kaggle/working/<repo>`                   | Unzip/import here, then mount with symlinks.      |

---

## 🌱 Minimal Bootstrap (Bash cell — first notebook cell)

Paste this in **the first code cell** of your Kaggle notebook. It is **idempotent**.

```bash
# 1) Define repo root under /kaggle/working
REPO=/kaggle/working/spectramind-v50
mkdir -p "$REPO"

# (Optional) unzip your repo if it is attached as a dataset (update the slug + zip path)
# unzip -q /kaggle/input/spectramind-v50-repo/spectramind-v50.zip -d /kaggle/working

# 2) Create expected repo data layout (writable)
mkdir -p "$REPO/data/raw" "$REPO/data/interim" "$REPO/data/processed" "$REPO/data/external"
mkdir -p "$REPO/artifacts" "$REPO/models"

# 3) Map Kaggle inputs -> repo structure via symlinks (adjust dataset slugs as needed)
ln -sfn /kaggle/input/ariel-data-challenge-2025  "$REPO/data/raw/adc2025"
# Example for an extra attached dataset:
# ln -sfn /kaggle/input/my-aux-dataset            "$REPO/data/external/aux"

# 4) (Optional) convenience link at repo root
ln -sfn /kaggle/input/ariel-data-challenge-2025  "$REPO/.kaggle_input"

# 5) Verify
echo "[Mapped inputs]"
ls -lah "$REPO/data/raw" || true
```

Notes:

* `ln -sfn` is **atomic & idempotent** — re-running updates the link safely.
* Make sure your dataset slugs match the **Input** panel in the Kaggle editor.

---

## 🐍 Python Helper (Symlink with Copy Fallback)

Use this if you want a robust, cross-platform helper in your notebooks or scripts.

```python
from pathlib import Path
import os, shutil, errno

def ensure_link(target: Path, link_path: Path):
    """
    Create a relative symlink from link_path -> target.
    Falls back to copy if symlink creation is denied.
    """
    target = Path(target)
    link_path = Path(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove any existing node (file/dir/symlink)
    if link_path.exists() or link_path.is_symlink():
        try:
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            else:
                shutil.rmtree(link_path)
        except Exception as e:
            print(f"[warn] Could not remove existing {link_path}: {e}")

    try:
        # Create a relative symlink (portable inside /kaggle/working)
        rel = os.path.relpath(target, start=link_path.parent)
        os.symlink(rel, link_path, target_is_directory=target.is_dir())
        print(f"[ok] Symlinked {link_path} -> {rel}")
    except OSError as e:
        # Fallback: copy if permission denied or symlink unsupported
        if e.errno in (errno.EPERM, errno.EACCES, getattr(errno, "EWINDOWS", 10000)):
            print(f"[warn] Symlink denied; copying {target} -> {link_path}")
            if target.is_dir():
                shutil.copytree(target, link_path)
            else:
                shutil.copy2(target, link_path)
        else:
            raise

# Example usage:
REPO = Path("/kaggle/working/spectramind-v50")
ensure_link(Path("/kaggle/input/ariel-data-challenge-2025"), REPO/"data/raw/adc2025")
```

---

## 🔧 Typical Maps for SpectraMind V50

Mirror these to your Hydra Kaggle profile.

```bash
# Raw competition data (read-only via symlink)
ln -sfn /kaggle/input/ariel-data-challenge-2025  /kaggle/working/spectramind-v50/data/raw/adc2025

# External metadata (if you attached a dataset with e.g., axis info)
# ln -sfn /kaggle/input/ariel-axis-info        /kaggle/working/spectramind-v50/data/external/axis

# Writable outputs
mkdir -p /kaggle/working/spectramind-v50/artifacts
mkdir -p /kaggle/working/spectramind-v50/models
```

**Hydra profile example** (`configs/data/kaggle.yaml`):

```yaml
repo: /kaggle/working/spectramind-v50

data:
  raw_dir:       ${repo}/data/raw/adc2025
  interim_dir:   ${repo}/data/interim
  processed_dir: ${repo}/data/processed

external:
  axis_dir:      ${repo}/data/external/axis

artifacts_dir:    ${repo}/artifacts
models_dir:       ${repo}/models
```

---

## 🔎 Verifying Symlinks

```bash
# Show symlink targets (short)
find /kaggle/working/spectramind-v50/data -maxdepth 2 -type l -ls

# Quick sanity on expected files
ls -lah /kaggle/working/spectramind-v50/data/raw/adc2025 | head -n 20
```

If you get “No such file or directory” following a link, the target dataset **slug is wrong** or **not attached**.

---

## 🧠 Notes & Gotchas

* ✅ Symlinks **are allowed** under `/kaggle/working` (but not in `/kaggle/input`, which is read-only).
* 🧳 If you export the working directory as a dataset, symlinks may be **materialized** or **omitted** — recreate them at startup.
* 🪟 On Windows local dev, symlink creation may require admin rights; the Python helper will **copy** as a fallback.
* 📦 Some tools (e.g., certain archivers, default DVC settings) may **not preserve symlinks** — prefer runtime linkage and avoid archiving symlinks.

---

## ⚡ One-Liner Bootstrap (First Cell)

```bash
REPO=/kaggle/working/spectramind-v50
mkdir -p "$REPO/data/raw" "$REPO/data/interim" "$REPO/data/processed" "$REPO/data/external" "$REPO/artifacts" "$REPO/models"
ln -sfn /kaggle/input/ariel-data-challenge-2025  "$REPO/data/raw/adc2025"
echo "Mounted Kaggle inputs into $REPO"
```

---

## FAQ

**Q: Why symlink instead of copying?**
A: Copying large datasets wastes time and disk; symlinks are instant and keep a **single source of truth**.

**Q: Can I symlink the other way (from `/kaggle/input` to `/kaggle/working`)?**
A: No — `/kaggle/input` is read-only. Create links **inside `/kaggle/working`**.

**Q: Will links persist across sessions?**
A: Only if you save `/kaggle/working` as outputs; when exported, symlinks may be **materialized or dropped**. Always recreate on startup.

**Q: How do I keep configs portable?**
A: Use a single `${repo}` root and put Kaggle specifics in `configs/*/kaggle.yaml`. Assume the symlink mounts described above.

---

**Happy mounting!** 🚀
