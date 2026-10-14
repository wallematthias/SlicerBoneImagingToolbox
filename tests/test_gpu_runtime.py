from types import SimpleNamespace
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_torch_gpu_check_executes_on_selected_device(monkeypatch):
    from SlicerBoneImagingToolboxLib.gpu_runtime import torch_gpu_status

    allocations = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "Windows NVIDIA GPU"),
        zeros=lambda shape, device: allocations.append(device),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    status = torch_gpu_status()
    assert status == "CUDA GPU: Windows NVIDIA GPU"
    assert allocations == ["cuda"]


def test_torch_gpu_check_explains_cpu_only_build(monkeypatch):
    from SlicerBoneImagingToolboxLib.gpu_runtime import torch_gpu_status

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sys, "platform", "win32")
    status = torch_gpu_status()
    assert "CPU" in status and "CUDA-enabled PyTorch" in status


def test_torch_gpu_check_reports_execution_failure(monkeypatch):
    from SlicerBoneImagingToolboxLib.gpu_runtime import torch_gpu_status

    def broken_allocation(shape, device):
        raise RuntimeError("GPU driver failure")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "GPU"),
        zeros=broken_allocation,
    ))
    assert "GPU driver failure" in torch_gpu_status()
