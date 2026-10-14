"""Read-only acceleration diagnostics for GPU-capable toolbox workflows."""
from __future__ import annotations

import sys


def torch_gpu_status() -> str:
    """Check execution on the device preferred by automatic PyTorch selection."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.zeros((1,), device="cuda")
            return f"CUDA GPU: {torch.cuda.get_device_name(0)}"
        if torch.backends.mps.is_available():
            torch.zeros((1,), device="mps")
            return "Apple MPS GPU"
        if getattr(torch.version, "cuda", None) is None and sys.platform != "darwin":
            return "CPU fallback: install CUDA-enabled PyTorch for a compatible NVIDIA GPU (see GPU setup documentation)."
        return "CPU fallback: no compatible CUDA/MPS GPU is available to PyTorch."
    except Exception as exc:
        return f"GPU runtime check failed: {exc}"


def gpu_runtime_statuses() -> dict[str, str]:
    """Report existing backends, without implying every module supports every GPU."""
    statuses = {"U-Net / MotionScore (PyTorch)": torch_gpu_status()}
    try:
        from bone_microarchitecture import thickness_backend_status

        status = thickness_backend_status()
        statuses["Microarchitecture / Functional Bone"] = (
            f"{status['thickness_backend']}: {status['thickness_device']} (GPU={status['thickness_gpu']}). "
            f"{status['thickness_backend_reason']}"
        )
    except Exception as exc:
        statuses["Microarchitecture / Functional Bone"] = f"GPU runtime check unavailable: {exc}"
    if sys.platform == "darwin":
        try:
            from plate_rod_thinning.metal_backend import status

            metal = status()
            statuses["Plate/Rod"] = f"Metal GPU: {metal.device}" if metal.available else f"CPU fallback: {metal.reason}"
        except Exception as exc:
            statuses["Plate/Rod"] = f"Metal check unavailable: {exc}"
    else:
        statuses["Plate/Rod"] = "CPU: this module's GPU backend is macOS Metal only."
    try:
        from spine_segment.device import resolve_device

        statuses["Spine Segmentation (current runtime)"] = f"Auto selects {resolve_device('auto')}; an external runtime is checked in the module."
    except Exception as exc:
        statuses["Spine Segmentation (current runtime)"] = f"Runtime check unavailable: {exc}"
    return statuses
