from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types


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
    assert 'VOIDSPACE_MINIMUM_VERSION = "0.1.3"' in source
    assert "baseline_segmentation_path" in source
    assert "followup_segmentation_path" in source
    assert "baseline_mask_path" in source
    assert "followup_mask_path" in source
    assert "register" not in source.lower()
