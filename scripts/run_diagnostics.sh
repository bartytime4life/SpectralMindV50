#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Diagnostics Runner (Upgraded)
# -----------------------------------------------------------------------------
# Generates diagnostics (FFT/UMAP/plots/HTML report) for a finished run.
# Works in both "pipeline-led" (runs spectramind diagnose) and "post-hoc"
# (given predictions + truth paths) modes.
#
# Usage:
#   ./scripts/run_diagnostics.sh [options] [-- ARGS...]
#
# Options:
#   -c, --config-name NAME   Hydra config for diagnose stage (default: diagnose)
#   -p, --pred PATH          Predictions CSV/Parquet(.csv[.gz]|.parquet) to analyze
#   -t, --truth PATH         Ground-truth CSV/Parquet (optional, enables metrics)
#   -o, --outdir DIR         Output dir (default env-aware: artifacts/diagnostics)
#   -n, --name  BASENAME     Report base name (default: diag_YYYYmmdd_HHMMSS)
#   -N, --no-cli             Skip `spectramind diagnose` (post-hoc only)
#   -F, --fft                Attempt quick FFT sketch (if numpy/matplotlib)
#   -U, --umap               Attempt quick UMAP sketch (if umap-learn/matplotlib)
#   -Q, --quiet              Less verbose output
#   -h, --help               Show help
#
# Passthrough:
#   Everything after `--` is forwarded to `spectramind diagnose` as-is.
#
# Notes:
# - Fails fast (set -Eeuo pipefail). Use repository Python env.
# - Detects Kaggle & picks safe outdir (/kaggle/working/artifacts/diagnostics).
# - Tries to locate predictions/truth automatically if not provided.
# - Generates: HTML report (if deps present) + PNGs + diagnostics_manifest.json
# -----------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# --- Defaults ----------------------------------------------------------------
CFG_NAME="diagnose"
PRED_PATH=""
TRUTH_PATH=""
OUTDIR=""
BASENAME=""
QUIET="0"
SKIP_CLI="0"
DO_FFT="0"
DO_UMAP="0"
EXTRA_ARGS=()

DEFAULT_OUTDIR_LOCAL="artifacts/diagnostics"
DEFAULT_OUTDIR_KAGGLE="/kaggle/working/artifacts/diagnostics"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { [[ "$QUIET" = "1" ]] || printf "[ %s ] [diag] %s\n" "$(timestamp)" "$*"; }
warn(){ printf "[ %s ] [diag][WARN] %s\n"  "$(timestamp)" "$*" >&2; }
die() {  printf "[ %s ] [diag][ERROR] %s\n" "$(timestamp)" "$*" >&2; exit 1; }

detect_env(){ [[ -d "/kaggle/input" ]] && echo "kaggle" || echo "local"; }
ENV_TYPE="$(detect_env)"
has_cmd(){ command -v "$1" >/dev/null 2>&1; }
usage(){ sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'; }
fsize(){ stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

# --- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config-name) CFG_NAME="${2:-}"; shift 2 ;;
    -p|--pred)        PRED_PATH="${2:-}"; shift 2 ;;
    -t|--truth)       TRUTH_PATH="${2:-}"; shift 2 ;;
    -o|--outdir)      OUTDIR="${2:-}"; shift 2 ;;
    -n|--name)        BASENAME="${2:-}"; shift 2 ;;
    -N|--no-cli)      SKIP_CLI="1"; shift ;;
    -F|--fft)         DO_FFT="1"; shift ;;
    -U|--umap)        DO_UMAP="1"; shift ;;
    -Q|--quiet)       QUIET="1"; shift ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; EXTRA_ARGS=("$@"); break ;;
    *)                warn "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# --- Resolve outdir/basename -------------------------------------------------
if [[ -z "$OUTDIR" ]]; then
  if [[ "$ENV_TYPE" = "kaggle" ]]; then OUTDIR="$DEFAULT_OUTDIR_KAGGLE"; else OUTDIR="$DEFAULT_OUTDIR_LOCAL"; fi
fi
mkdir -p "$OUTDIR"
[[ -n "$BASENAME" ]] || BASENAME="diag_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTDIR/$BASENAME"
mkdir -p "$RUN_DIR" "$RUN_DIR/assets"

log "Environment: $ENV_TYPE"
log "Output dir : $RUN_DIR"

# --- Discovery helpers -------------------------------------------------------
autofind_predictions() {
  for f in \
    "outputs/predictions.csv" "outputs/predictions.csv.gz" "outputs/predictions.parquet" \
    "artifacts/predictions.csv" "artifacts/predictions.csv.gz" "artifacts/predictions.parquet" \
    "predictions/predictions.csv" "predictions/predictions.csv.gz" "predictions/predictions.parquet"
  do [[ -f "$f" ]] && { echo "$f"; return; }; done
  find . -maxdepth 4 -type f \( -name '*pred*.csv' -o -name '*pred*.csv.gz' -o -name '*pred*.parquet' \) | head -n1 || true
}
autofind_truth() {
  for f in \
    "data/processed/val_labels.csv" "data/processed/val_labels.csv.gz" "data/processed/val_labels.parquet" \
    "data/processed/train_labels.csv" "data/processed/train_labels.csv.gz" "data/processed/train_labels.parquet"
  do [[ -f "$f" ]] && { echo "$f"; return; }; done
  find data -maxdepth 4 -type f \( -name '*label*.csv' -o -name '*label*.csv.gz' -o -name '*label*.parquet' \) | head -n1 || true
}

# --- Run CLI diagnose (optional) --------------------------------------------
maybe_run_cli() {
  if [[ "$SKIP_CLI" = "1" ]]; then log "Skipping spectramind diagnose (post-hoc only)"; return 0; fi
  if has_cmd spectramind; then
    log "Running: spectramind diagnose --config-name $CFG_NAME ${EXTRA_ARGS[*]}"
    set +e; spectramind diagnose --config-name "$CFG_NAME" "${EXTRA_ARGS[@]}"; rc=$?; set -e
    [[ $rc -eq 0 ]] || warn "spectramind diagnose returned $rc; continuing with post-hoc"
  else
    warn "spectramind CLI not found; skipping CLI diagnose"
  fi
}

# --- Execute pre-stage -------------------------------------------------------
maybe_run_cli

# If user didn’t pass predictions/truth, try to find them
[[ -n "$PRED_PATH"  ]] || PRED_PATH="$(autofind_predictions || true)"
[[ -n "$TRUTH_PATH" ]] || TRUTH_PATH="$(autofind_truth || true)"

[[ -n "$PRED_PATH"  ]] && log "Predictions: $PRED_PATH" || warn "Predictions not provided & not found automatically."
[[ -n "$TRUTH_PATH" ]] && log "Truth      : $TRUTH_PATH"

# --- Python diagnostics engine -----------------------------------------------
export RUN_DIR PRED_PATH TRUTH_PATH DO_FFT DO_UMAP

python - <<'PY'
import os, sys, json, math, gzip, io, hashlib, warnings
from pathlib import Path

RUN_DIR   = Path(os.environ.get("RUN_DIR","."))
PRED_PATH = os.environ.get("PRED_PATH") or ""
TRUTH_PATH= os.environ.get("TRUTH_PATH") or ""
DO_FFT    = os.environ.get("DO_FFT","0") == "1"
DO_UMAP   = os.environ.get("DO_UMAP","0") == "1"

# ---------------- I/O helpers ----------------
def _open_any(path:str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path, "r", encoding="utf-8", errors="ignore")

def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def _have(mod): 
    try: __import__(mod); return True
    except Exception: return False

def _read_table(path: str):
    """Read CSV/Parquet returning (pandas.DataFrame, format_str)."""
    import pandas as pd
    if path.endswith(".parquet"):
        return pd.read_parquet(path), "parquet"
    # CSV (possibly gz)
    return pd.read_csv(path), "csv"

def _safe_savefig(fig, outpath: Path):
    try:
        fig.savefig(outpath, bbox_inches="tight")
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

# ---------------- Load data ----------------
pred_df = None; truth_df = None; fmt_pred=""; fmt_truth=""
if PRED_PATH:
    try:
        pred_df, fmt_pred = _read_table(PRED_PATH)
    except Exception as e:
        print(f"[diag][ERROR] failed to read predictions: {e}", file=sys.stderr); sys.exit(2)

if TRUTH_PATH:
    try:
        truth_df, fmt_truth = _read_table(TRUTH_PATH)
    except Exception as e:
        print(f"[diag][WARN] failed to read truth: {e}", file=sys.stderr)
        truth_df = None

# Expect schema: sample_id + mu_000..mu_282 + sigma_000..sigma_282 (or similar)
def _mu_sigma_cols(df):
    mu = [c for c in df.columns if c.startswith("mu_")]
    sg = [c for c in df.columns if c.startswith("sigma_")]
    return sorted(mu), sorted(sg)

summary = {
    "pred": {"path": PRED_PATH or "", "format": fmt_pred, "exists": bool(pred_df is not None)},
    "truth": {"path": TRUTH_PATH or "", "format": fmt_truth, "exists": bool(truth_df is not None)},
    "metrics": {},
}

RUN_DIR.mkdir(parents=True, exist_ok=True)
assets = RUN_DIR / "assets"
assets.mkdir(exist_ok=True)

# ---------------- Basic sanity/shape ----------------
if pred_df is not None:
    mu_cols, sg_cols = _mu_sigma_cols(pred_df)
    summary["pred"]["rows"] = int(pred_df.shape[0])
    summary["pred"]["cols"] = int(pred_df.shape[1])
    summary["pred"]["mu_cols"] = len(mu_cols)
    summary["pred"]["sigma_cols"] = len(sg_cols)
    summary["pred"]["head_sample_id"] = str(pred_df.iloc[0,0]) if pred_df.shape[0] else ""
    # quick numeric guard on a slice
    try:
        import numpy as np
        sample = pred_df[mu_cols[:4] + sg_cols[:4]].to_numpy(dtype=float)
        if not np.isfinite(sample).all():
            summary["pred"]["finite_sample_8"] = False
        else:
            summary["pred"]["finite_sample_8"] = True
    except Exception:
        pass

# ---------------- Metrics (if truth) ----------------
if pred_df is not None and truth_df is not None:
    try:
        import pandas as pd, numpy as np
        on = [c for c in ("sample_id","id","row_id") if c in pred_df.columns and c in truth_df.columns][:1]
        if not on:
            raise ValueError("No common id column among {sample_id,id,row_id}")
        joined = pred_df.merge(truth_df, on=on[0], suffixes=("", "_truth"), how="inner")
        mu_cols = sorted([c for c in pred_df.columns if c.startswith("mu_") and c + "_truth" in joined.columns])
        if not mu_cols:
            raise ValueError("No overlapping mu_* columns between pred and truth")
        y = joined[[c + "_truth" for c in mu_cols]].to_numpy(float)
        yhat = joined[mu_cols].to_numpy(float)
        resid = yhat - y
        mae = float(np.mean(np.abs(resid)))
        rmse = float(np.sqrt(np.mean(resid**2)))
        per_bin_rmse = dict(zip(mu_cols, np.sqrt(np.mean((yhat - y)**2, axis=0)).tolist()))
        summary["metrics"]["join_rows"] = int(joined.shape[0])
        summary["metrics"]["mae"] = mae
        summary["metrics"]["rmse"] = rmse
        # Save a small JSON of top/worst bins
        import json
        worst = sorted(per_bin_rmse.items(), key=lambda kv: kv[1], reverse=True)[:10]
        best  = sorted(per_bin_rmse.items(), key=lambda kv: kv[1])[:10]
        with open(assets / "per_bin_rmse_top10.json","w") as f: json.dump(worst, f, indent=2)
        with open(assets / "per_bin_rmse_best10.json","w") as f: json.dump(best,  f, indent=2)
        summary["metrics"]["per_bin_rmse_top10_path"] = str((assets / "per_bin_rmse_top10.json").resolve())
        summary["metrics"]["per_bin_rmse_best10_path"] = str((assets / "per_bin_rmse_best10.json").resolve())
        # Quick plot for distribution
        if _have("matplotlib"):
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8,3))
            plt.hist(list(per_bin_rmse.values()), bins=40)
            plt.title("Per-bin RMSE distribution")
            plt.xlabel("RMSE"); plt.ylabel("count")
            _safe_savefig(fig, assets / "per_bin_rmse_hist.png")
            summary["metrics"]["per_bin_rmse_hist"] = str((assets/"per_bin_rmse_hist.png").resolve())
    except Exception as e:
        summary["metrics"]["error"] = str(e)

# ---------------- FFT sketch (optional) ----------------
if DO_FFT and pred_df is not None and _have("numpy") and _have("matplotlib"):
    try:
        import numpy as np, matplotlib.pyplot as plt
        mu_cols, _ = _mu_sigma_cols(pred_df)
        # choose a single row to visualize spectral content
        row0 = pred_df[mu_cols].iloc[0].to_numpy(float)
        spec = np.fft.rfft(row0 - row0.mean())
        f = np.fft.rfftfreq(row0.size, d=1.0)
        fig = plt.figure(figsize=(8,3))
        plt.plot(f, np.abs(spec)); plt.title("FFT magnitude (row 0, mu_*)"); plt.xlabel("freq"); plt.ylabel("|X|")
        _safe_savefig(fig, assets / "fft_row0.png")
        summary["fft_plot"] = str((assets/"fft_row0.png").resolve())
    except Exception as e:
        summary["fft_error"] = str(e)

# ---------------- UMAP sketch (optional) ----------------
if DO_UMAP and pred_df is not None and _have("umap") and _have("matplotlib"):
    try:
        import numpy as np, umap, matplotlib.pyplot as plt
        mu_cols, _ = _mu_sigma_cols(pred_df)
        X = pred_df[mu_cols].to_numpy(float)
        n = min(5000, X.shape[0])
        Xs = X[:n]
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        emb = reducer.fit_transform(Xs)
        fig = plt.figure(figsize=(6,5))
        plt.scatter(emb[:,0], emb[:,1], s=3, alpha=0.6)
        plt.title(f"UMAP(mu_*) — n={n}")
        _safe_savefig(fig, assets / "umap_mu.png")
        summary["umap_plot"] = str((assets/"umap_mu.png").resolve())
    except Exception as e:
        summary["umap_error"] = str(e)

# ---------------- HTML report (best-effort) ----------------
def _write_html(summary:dict):
    out = RUN_DIR / "report.html"
    def esc(s): 
        return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SpectraMind Diagnostics</title>
<style>body{{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px}}
code,pre{{background:#f5f5f5;padding:2px 4px;border-radius:4px}}
h1{{margin-top:0}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{border:1px solid #ddd;border-radius:12px;padding:12px;box-shadow:0 1px 2px #0001}}
img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px}}
.small{{color:#777;font-size:90%}}</style></head>
<body>
<h1>🛰️ SpectraMind V50 — Diagnostics</h1>
<p class="small">Generated at: {esc(str(Path.cwd()))}</p>
<div class="card"><h2>Summary</h2>
<pre>{esc(json.dumps(summary, indent=2)[:5000])}</pre></div>
<div class="grid">
  <div class="card"><h3>Per-bin RMSE Histogram</h3>
    <img src="assets/per_bin_rmse_hist.png" onerror="this.outerHTML='<p class=small>No plot available.</p>'">
  </div>
  <div class="card"><h3>FFT Sketch</h3>
    <img src="assets/fft_row0.png" onerror="this.outerHTML='<p class=small>No FFT available.</p>'">
  </div>
  <div class="card"><h3>UMAP Sketch</h3>
    <img src="assets/umap_mu.png" onerror="this.outerHTML='<p class=small>No UMAP available.</p>'">
  </div>
</div>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    return str(out.resolve())

report_path = None
try:
    report_path = _write_html(summary)
except Exception as e:
    summary["html_error"] = str(e)

# ---------------- Manifest ----------------
manifest = {
    "generated_at": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "pred": summary.get("pred",{}),
    "truth": summary.get("truth",{}),
    "metrics": summary.get("metrics",{}),
    "assets_dir": str((RUN_DIR/"assets").resolve()),
    "report_html": report_path or "",
}
with open(RUN_DIR / "diagnostics_manifest.json","w") as f:
    json.dump(manifest, f, indent=2)

print("[diag] Wrote manifest:", (RUN_DIR / "diagnostics_manifest.json"))
if report_path:
    print("[diag] Report HTML  :", report_path)
PY

# --- Zip result for convenience ----------------------------------------------
ZIP_PATH="$RUN_DIR.zip"
if has_cmd zip; then
  (cd "$(dirname "$RUN_DIR")" && zip -q -r "$(basename "$ZIP_PATH")" "$(basename "$RUN_DIR")")
  log "Diagnostics bundle -> $ZIP_PATH"
else
  warn "zip not found; skipping archive"
fi

# --- Summary -----------------------------------------------------------------
echo
echo "──────────────────────────────────────────────────────────────────────────────"
echo " Diagnostics summary"
echo "  • Run dir   : $RUN_DIR"
[[ -n "$PRED_PATH"  ]] && echo "  • Predictions: $PRED_PATH ($(fsize "$PRED_PATH" 2>/dev/null || echo 0) bytes)"
[[ -n "$TRUTH_PATH" ]] && echo "  • Truth      : $TRUTH_PATH ($(fsize "$TRUTH_PATH" 2>/dev/null || echo 0) bytes)"
echo "  • Bundle    : $ZIP_PATH"
echo "  • FFT       : $([ "$DO_FFT" = "1" ] && echo ENABLED || echo disabled)"
echo "  • UMAP      : $([ "$DO_UMAP" = "1" ] && echo ENABLED || echo disabled)"
echo "──────────────────────────────────────────────────────────────────────────────"
