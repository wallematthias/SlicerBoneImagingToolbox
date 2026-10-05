from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "HRpQCTTools" / "VoidspaceHRpQCT" / "VoidspaceHRpQCT.py"


def _install_slicer_import_stubs(monkeypatch) -> None:
    qt = types.ModuleType("qt")
    ctk = types.ModuleType("ctk")
    slicer = types.ModuleType("slicer")
    scripted = types.ModuleType("slicer.ScriptedLoadableModule")

    class _Base:
        def __init__(self, *args, **kwargs):
            pass

        def delayDisplay(self, *args, **kwargs):
            pass

    class _QProcess:
        MergedChannels = 0

    qt.QProcess = _QProcess
    qt.QIcon = lambda *_args, **_kwargs: None
    scripted.ScriptedLoadableModule = _Base
    scripted.ScriptedLoadableModuleWidget = _Base
    scripted.ScriptedLoadableModuleLogic = _Base
    scripted.ScriptedLoadableModuleTest = _Base
    slicer.ScriptedLoadableModule = scripted
    slicer.util = types.SimpleNamespace()
    slicer.app = types.SimpleNamespace(applicationFilePath=lambda: sys.executable)

    monkeypatch.setitem(sys.modules, "qt", qt)
    monkeypatch.setitem(sys.modules, "ctk", ctk)
    monkeypatch.setitem(sys.modules, "slicer", slicer)
    monkeypatch.setitem(sys.modules, "slicer.ScriptedLoadableModule", scripted)


def _import_voidspace_module(monkeypatch):
    _install_slicer_import_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location("voidspace_slicer_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_voidspace_module_is_registered_in_toolbox() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "toolbox_modules.json").read_text(encoding="utf-8"))

    assert "add_subdirectory(HRpQCTTools/VoidspaceHRpQCT)" in cmake
    entry = next(module for module in manifest["modules"] if module["path"] == "HRpQCTTools/VoidspaceHRpQCT")
    assert entry["title"] == "Voidspace"
    assert entry["section"] == "Microstructural Analysis"


def test_voidspace_scene_module_wraps_core_without_registration_ownership(monkeypatch) -> None:
    module = _import_voidspace_module(monkeypatch)
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert module.VoidspaceHRpQCTLogic.scene_cli_command("/tmp/job.json") == [
        "-c",
        module._VOIDSPACE_SCENE_PROCESS_SCRIPT,
        "/tmp/job.json",
    ]
    assert "from voidspace import compare, run_case" in source
    assert "large_mask_path" in source
    assert "all_mask_path" in source
    assert "large_void_path" not in source
    assert "all_void_path" not in source
    assert 'VOIDSPACE_MINIMUM_VERSION = "0.1.5"' in source
    assert "baseline_segmentation_path" in source
    assert "followup_segmentation_path" in source
    assert "baseline_mask_path" in source
    assert "followup_mask_path" in source


def test_longitudinal_scene_script_publishes_real_core_change_outputs(tmp_path, monkeypatch):
    module = _import_voidspace_module(monkeypatch)
    core = pytest.importorskip("voidspace")
    np = pytest.importorskip("numpy")
    sitk = pytest.importorskip("SimpleITK")
    image = sitk.GetImageFromArray(np.ones((9, 9, 9), dtype=np.uint8))
    segmentation = tmp_path / "seg.nrrd"
    sitk.WriteImage(image, str(segmentation))
    job = {
        "mode": "longitudinal",
        "baseline_segmentation_path": str(segmentation),
        "followup_segmentation_path": str(segmentation),
        "output_dir": str(tmp_path / "results"),
        "outputs_json_path": str(tmp_path / "outputs.json"),
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(core.__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", module._VOIDSPACE_SCENE_PROCESS_SCRIPT, str(job_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = json.loads(Path(job["outputs_json_path"]).read_text(encoding="utf-8"))
    assert Path(outputs["expanded"]).name == "voidspace_expanded_mask.nii.gz"
    assert Path(outputs["contracted"]).name == "voidspace_contracted_mask.nii.gz"
    assert all(Path(path).is_file() for path in outputs.values())
