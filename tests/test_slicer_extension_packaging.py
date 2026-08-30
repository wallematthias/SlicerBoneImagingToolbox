from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_shared_toolbox_library_is_installed_with_scripted_modules() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "TOOLBOX_SHARED_PYTHON_SCRIPTS" in cmake
    assert "ctkMacroCompilePythonScript" in cmake
    assert "SlicerBoneImagingToolboxLib/__init__.py" in cmake
    assert "SlicerBoneImagingToolboxLib/package_status.py" in cmake
    assert "SlicerBoneImagingToolboxLib/slicer_update_ui.py" in cmake
    assert "SlicerBoneImagingToolboxLib/vertebra_labels.py" in cmake
    assert "Slicer_QTSCRIPTEDMODULES_LIB_DIR" in cmake


def test_setup_module_is_registered_with_extension_packaging() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = (ROOT / "toolbox_modules.json").read_text(encoding="utf-8")

    assert "add_subdirectory(Setup/BoneImagingToolboxSetup)" in cmake
    assert '"path": "Setup/BoneImagingToolboxSetup"' in manifest
    assert '"section": "Setup"' in manifest


def test_bone_microarchitecture_module_is_registered_with_extension_packaging() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = (ROOT / "toolbox_modules.json").read_text(encoding="utf-8")

    assert "add_subdirectory(HRpQCTTools/BoneMicroarchitecture)" in cmake
    assert '"path": "HRpQCTTools/BoneMicroarchitecture"' in manifest
    assert '"title": "Bone Microarchitecture"' in manifest
    assert '"section": "HR-pQCT"' in manifest


def test_public_fea_and_mechanoregulation_modules_are_registered_with_extension_packaging() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = (ROOT / "toolbox_modules.json").read_text(encoding="utf-8")

    assert "add_subdirectory(HRpQCTTools/ParOSolFEA)" in cmake
    assert '"path": "HRpQCTTools/ParOSolFEA"' in manifest
    assert '"title": "ParOSol FEA"' in manifest
    assert '"section": "FEA"' in manifest

    assert "add_subdirectory(HRpQCTTools/MechanoregulationHRpQCT)" in cmake
    assert '"path": "HRpQCTTools/MechanoregulationHRpQCT"' in manifest
    assert '"title": "Bone Mechanoregulation"' in manifest
    assert '"section": "Mechanoregulation"' in manifest


def test_python_unittest_scripts_define_matching_test_classes() -> None:
    for cmake_path in [*ROOT.glob("*Tools/*/CMakeLists.txt"), *ROOT.glob("Setup/*/CMakeLists.txt")]:
        cmake = cmake_path.read_text(encoding="utf-8")
        if "slicer_add_python_unittest" not in cmake:
            continue

        module_match = re.search(r"set\s*\(\s*MODULE_NAME\s+([A-Za-z0-9_]+)\s*\)", cmake)
        assert module_match, f"{cmake_path} must declare MODULE_NAME"
        module_name = module_match.group(1)
        module_source = (cmake_path.parent / f"{module_name}.py").read_text(encoding="utf-8")

        assert (
            f"class {module_name}Test(" in module_source
        ), f"{module_name}.py is registered as a Python unittest but has no {module_name}Test class"


def test_timelapsed_release_smoke_tests_skip_optional_runtime_dependencies() -> None:
    module_source = (
        ROOT / "HRpQCTTools" / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py"
    ).read_text(encoding="utf-8")

    test_class_start = module_source.index("class TimelapsedHRpQCTTest(")
    test_class_source = module_source[test_class_start:]

    assert "self.skipTest" in test_class_source
    assert "logic.pipeline_status()" in test_class_source
    assert "timelapsed-hrpqct pipeline is not installed" in test_class_source
    assert "PyYAML is not available" in test_class_source
