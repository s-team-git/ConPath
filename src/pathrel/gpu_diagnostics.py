"""Small, read-only CUDA diagnostics used by experiment entry points.

``torch.cuda.is_available()`` answers whether *this Python process* can open a CUDA
device.  It does not answer whether the host has a GPU or whether the NVIDIA kernel
driver is loaded.  Keeping the distinction in one formatter prevents experiment logs
from incorrectly recommending a driver reinstall when the real problem is missing
container device passthrough.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any


def cuda_unavailable_message(torch_module: Any) -> str:
    """Return an actionable explanation for a requested but unavailable CUDA device."""

    built_cuda = getattr(getattr(torch_module, "version", None), "cuda", None)
    visible_nodes = sorted(glob.glob("/dev/nvidia*"))
    compute_nodes = [
        path
        for path in visible_nodes
        if Path(path).name in {"nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"}
        or Path(path).name.removeprefix("nvidia").isdigit()
    ]
    kernel_driver = Path("/proc/driver/nvidia/version").is_file()
    masked = os.environ.get("CUDA_VISIBLE_DEVICES")

    lines = [
        "CUDA was requested, but this Python process cannot open a CUDA device.",
        f"PyTorch build: {getattr(torch_module, '__version__', 'unknown')} (CUDA {built_cuda or 'none'}); ",
        f"torch.cuda.is_available()={bool(torch_module.cuda.is_available())}; ",
        f"device_count={int(torch_module.cuda.device_count())}.",
    ]
    if kernel_driver and not compute_nodes:
        lines.append(
            "Host evidence is present (/proc/driver/nvidia/version), but this session has no "
            "/dev/nvidia* compute nodes. Run the venv on the host or relaunch the container/job "
            "with NVIDIA GPU passthrough (for Docker, --gpus all)."
        )
    elif not kernel_driver and not visible_nodes:
        lines.append(
            "No NVIDIA kernel report or device nodes are visible to this session; check the host/job "
            "GPU allocation before changing the PyTorch environment."
        )
    elif masked is not None and masked.strip() in {"", "-1"}:
        lines.append(
            f"CUDA_VISIBLE_DEVICES={masked!r} masks all GPUs for this process; unset it or select a visible index."
        )
    elif masked is not None:
        lines.append(f"CUDA_VISIBLE_DEVICES={masked!r}; verify that the selected index is valid in this job.")
    else:
        lines.append(
            "Device nodes are visible but CUDA initialization still failed; inspect driver permissions, "
            "the PyTorch CUDA wheel, and the kernel log."
        )
    return "\n".join(lines)

