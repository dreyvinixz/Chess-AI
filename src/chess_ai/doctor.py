"""Local hardware and dependency diagnostics."""

import json
import platform
import shutil
import subprocess
import sys

import torch


def system_report() -> dict:
    cuda = torch.cuda.is_available()
    report = {
        "os": platform.platform(),
        "wsl2": "microsoft" in platform.release().lower(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "stockfish": shutil.which("stockfish"),
        "browser_automation": _has_playwright(),
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "vram_mb": round(properties.total_memory / 1048576),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "fp16_tested": _fp16_test(),
            }
        )
    command = shutil.which("nvidia-smi")
    if command:
        result = subprocess.run(
            [
                command,
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        report["nvidia_smi"] = (
            result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        )
    report["environment"] = "READY" if cuda and report["stockfish"] else "WARNING"
    return report


def _has_playwright() -> bool:
    try:
        import playwright  # noqa: F401

        return True
    except ImportError:
        return False


def _fp16_test() -> bool:
    try:
        x = torch.ones((16, 16), device="cuda", dtype=torch.float16)
        return bool(torch.isfinite(x @ x).all().item())
    except RuntimeError:
        return False


def write_report(path: str) -> dict:
    report = system_report()
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
