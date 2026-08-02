"""Global determinism and device resolution.

Device policy (see docs/05_local_dev.md):
  - Channel physics stays in NumPy float64 on the CPU, always. MPS has no
    float64, and silently downcasting the CIR would corrupt the Gate-1
    validation against the closed form.
  - Only detector/controller tensors move to the accelerator.
  - One device per experiment family. Never extend a results table across
    devices: MPS and CUDA are not bit-identical.
"""

from __future__ import annotations

import os
import random

__all__ = ["set_all_seeds", "resolve_device", "device_report"]


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' to the best available device, or validate an explicit one.

    Returns one of: 'cuda', 'cuda:N', 'mps', 'cpu'.
    """
    try:
        import torch
    except ImportError:
        if requested not in ("auto", "cpu"):
            raise RuntimeError(f"device {requested!r} requested but torch is not installed") from None
        return "cpu"

    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("mps requested but not available")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Fix every RNG we touch. Call once, at the top of every run."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        if deterministic:
            # MPS ignores this; harmless.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def device_report() -> dict:
    """Environment provenance recorded in every run record."""
    import platform

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": resolve_device("auto"),
        "torch": None,
        "accelerator_name": None,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        if torch.cuda.is_available():
            report["accelerator_name"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            report["accelerator_name"] = "Apple Silicon (MPS)"
    except ImportError:
        pass
    return report
