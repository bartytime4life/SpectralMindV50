from __future__ import annotations
import json, os, platform, subprocess, sys
import typer

app = typer.Typer()

def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as e:
        return f"ERR: {e.output.strip()}"

@app.command("doctor")
def doctor_cuda(
    cuda: bool = typer.Option(False, "--cuda", help="Include CUDA parity checks"),
    fail_on_mismatch: bool = typer.Option(False, "--fail-on-mismatch"),
    emit_json: str = typer.Option("", "--emit-json", help="Write JSON report to path"),
):
    import torch

    report = {
        "host": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
        "nvidia": {},
        "cuda_runtime": {},
        "nvcc": {},
        "determinism": {}
    }

    if cuda:
        report["nvidia"]["smi"] = _sh("nvidia-smi || true")
        report["cuda_runtime"]["version_json"] = _sh("cat /usr/local/cuda/version.json 2>/dev/null || true")
        report["cuda_runtime"]["version_txt"] = _sh("cat /usr/local/cuda/version.txt 2>/dev/null || true")
        report["nvcc"]["version"] = _sh("nvcc --version || true")

        # Determinism policy we enforce
        report["determinism"] = {
            "torch.use_deterministic_algorithms": True,
            "cudnn.deterministic": True,
            "cudnn.benchmark": False,
            "float32_matmul_precision": "highest",
            "tf32": {
                "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE", "unset"),
            }
        }

    # Kaggle target
    target = {
        "torch": { "version": "2.3.1+cu121", "cuda": "12.1" },
        "cudnn": { "major": 8 },  # allows any 8.x
    }

    mismatches = []

    if report["torch"]["version"] != target["torch"]["version"]:
        mismatches.append(f'torch.version {report["torch"]["version"]} != {target["torch"]["version"]}')
    if not (report["torch"]["cuda"] or "").startswith(target["torch"]["cuda"]):
        mismatches.append(f'torch.version.cuda {report["torch"]["cuda"]} != {target["torch"]["cuda"]}.x')

    cudnn_v = report["torch"]["cudnn"]
    if isinstance(cudnn_v, int) and cudnn_v // 10000 != target["cudnn"]["major"]:
        mismatches.append(f'cuDNN major {(cudnn_v // 10000) if cudnn_v else None} != {target["cudnn"]["major"]}')

    ok = not mismatches
    result = { "ok": ok, "mismatches": mismatches }

    if emit_json:
        with open(emit_json, "w") as f:
            json.dump({**report, "result": result}, f, indent=2)

    typer.echo(json.dumps(result, indent=2))
    if fail_on_mismatch and not ok:
        raise typer.Exit(code=1)
