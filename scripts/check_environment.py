#!/usr/bin/env python3
"""Read-only environment check for PathRel GPU training.

The script deliberately does not install packages, change driver settings, or allocate a large
tensor.  It reports enough information to select a compatible PyTorch wheel and a conservative
first training configuration on a remote workstation.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def _run(command: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi was not found on PATH"}

    query = [
        executable,
        "--query-gpu=name,memory.total,memory.free,driver_version,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    code, stdout, stderr = _run(query)
    if code != 0:
        return {"available": False, "reason": stderr or f"nvidia-smi exited with {code}"}

    devices: list[dict[str, str]] = []
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 5:
            continue
        devices.append(
            {
                "name": fields[0],
                "memory_total_mb": fields[1],
                "memory_free_mb": fields[2],
                "driver_version": fields[3],
                "bus_id": fields[4],
            }
        )
    return {"available": True, "devices": devices, "raw": stdout}


def _torch_report() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on target machine
        return {"imported": False, "reason": f"torch import failed: {exc}"}

    report: dict[str, Any] = {
        "imported": True,
        "version": torch.__version__,
        "built_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if not torch.cuda.is_available():
        report["reason"] = "PyTorch cannot see a CUDA device"
        return report

    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        item: dict[str, Any] = {
            "index": index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_gib": round(properties.total_memory / (1024**3), 2),
            "multi_processor_count": properties.multi_processor_count,
        }
        try:
            # is_bf16_supported() uses the current CUDA device rather than a device index.
            # Select the device explicitly so multi-GPU reports are not all attributed to GPU 0.
            with torch.cuda.device(index):
                item["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except (AttributeError, RuntimeError, TypeError):
            item["bf16_supported"] = "unknown"
        devices.append(item)
    report["devices"] = devices
    return report


def _recommendation(smi: dict[str, Any], torch_report: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    names = [str(item.get("name", "")) for item in smi.get("devices", [])]
    names_lower = " ".join(names).lower()

    if "blackwell" in names_lower or "pro 6000" in names_lower:
        recommendations.append(
            "RTX PRO 6000 Blackwell is normally paired with an official CUDA 12.8 or CUDA 13.0 PyTorch wheel; verify the driver before choosing the index URL."
        )
    elif "p6000" in names_lower and "pro" not in names_lower:
        recommendations.append(
            "This looks like the older Quadro P6000/Pascal family. Do not assume a current PyTorch wheel supports it; use the PyTorch/CUDA matrix for the installed driver and expect to fall back to a legacy stack."
        )
    elif "ada" in names_lower:
        recommendations.append(
            "RTX 6000 Ada can use a current CUDA 12.x PyTorch wheel; select cu126/cu128 according to the driver reported above."
        )
    else:
        recommendations.append(
            "GPU model is not recognized by this helper. Select the PyTorch wheel from the official installer using the reported driver and CUDA support."
        )

    if not torch_report.get("imported", False):
        recommendations.append("Install PyTorch inside a fresh Python 3.10-3.12 environment; do not copy this machine's .venv.")
    elif not torch_report.get("cuda_available", False):
        recommendations.append("Torch imported but CUDA is unavailable. Check the NVIDIA driver, wheel index, and CUDA_VISIBLE_DEVICES before training.")
    else:
        recommendations.append("CUDA is visible to PyTorch. Start with one GPU and the 24x24 synthetic smoke run before scaling map resolution or sample count.")

    recommendations.extend(
        [
            "The current reachability layer expands [B,K,Q,H,W] and performs up to H*W Python iterations; it is compute/memory intensive at large maps even with 96 GiB VRAM.",
            "For the first run keep batch_size=4-8, K=4-8, Q=2, radii=[0,1,2], and max_reachability_steps=H*W. Profile before increasing any of these values.",
            "This prototype does not yet implement AMP, multi-GPU DDP, or resume training. Keep the reachability/oracle path in fp32 until an explicit mixed-precision test is added.",
        ]
    )
    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable report instead of the human-readable summary",
    )
    args = parser.parse_args()

    smi = _nvidia_smi()
    torch_report = _torch_report()
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "nvidia_smi": smi,
        "torch": torch_report,
        "recommendations": _recommendation(smi, torch_report),
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("PathRel environment check (read-only)")
    print(f"Python: {platform.python_version()} | {platform.platform()}")
    if smi.get("available"):
        for index, device in enumerate(smi.get("devices", [])):
            print(
                "GPU[{index}]: {name} | {total} MiB total | {free} MiB free | driver {driver}".format(
                    index=index,
                    name=device.get("name", "unknown"),
                    total=device.get("memory_total_mb", "?"),
                    free=device.get("memory_free_mb", "?"),
                    driver=device.get("driver_version", "?"),
                )
            )
    else:
        print(f"nvidia-smi: unavailable ({smi.get('reason', 'unknown reason')})")

    if torch_report.get("imported"):
        print(
            "Torch: {version} | built CUDA {cuda} | cuda_available={available} | devices={count}".format(
                version=torch_report.get("version"),
                cuda=torch_report.get("built_cuda"),
                available=torch_report.get("cuda_available"),
                count=torch_report.get("device_count"),
            )
        )
        for device in torch_report.get("devices", []):
            print(
                "  torch.cuda:{index} {name} | CC {cc} | {memory} GiB | BF16 {bf16}".format(
                    index=device.get("index"),
                    name=device.get("name"),
                    cc=device.get("compute_capability"),
                    memory=device.get("total_memory_gib"),
                    bf16=device.get("bf16_supported"),
                )
            )
    else:
        print(f"Torch: unavailable ({torch_report.get('reason', 'unknown reason')})")

    print("Recommendations:")
    for recommendation in report["recommendations"]:
        print(f"- {recommendation}")


if __name__ == "__main__":
    main()
