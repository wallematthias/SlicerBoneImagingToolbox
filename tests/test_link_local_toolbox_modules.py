from pathlib import Path
import runpy
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "link_local_toolbox_modules.py"


def _load_helper():
    sys.modules.setdefault("slicer", types.SimpleNamespace())
    return runpy.run_path(str(SCRIPT_PATH), run_name="link_local_toolbox_modules_test")


def test_local_link_helper_removes_renamed_microarchitecture_module_path() -> None:
    helper = _load_helper()
    stale_path = ROOT / "HRpQCTTools" / "MicroarchitectureHRpQCT"
    active_paths = {str((ROOT / "HRpQCTTools" / "BoneMicroarchitecture").resolve())}

    assert helper["_is_stale_toolbox_path"](stale_path, ROOT, active_paths)


def test_local_link_helper_removes_deleted_worktree_module_path() -> None:
    helper = _load_helper()
    stale_path = (
        ROOT
        / ".worktrees"
        / "derivatives-overhaul"
        / "HRpQCTTools"
        / "BoneMicroarchitecture"
    )
    active_paths = {str((ROOT / "HRpQCTTools" / "BoneMicroarchitecture").resolve())}

    assert helper["_is_stale_toolbox_path"](stale_path, ROOT, active_paths)


def test_local_link_helper_does_not_reference_removed_registered_common_region_module() -> None:
    helper = _load_helper()

    assert "RegisteredCommonRegion" not in helper["DEFAULT_BUILTIN_MODULE_DIRS"]
    assert "RegisteredCommonRegion" not in helper["LEGACY_MODULE_DIR_NAMES"]


def test_local_link_helper_removes_renamed_batch_module_path() -> None:
    helper = _load_helper()
    stale_path = ROOT / "IOTools" / "HRpQCTBatch"
    active_paths = {str((ROOT / "IOTools" / "BatchProcessor").resolve())}

    assert helper["_is_stale_toolbox_path"](stale_path, ROOT, active_paths)


def test_local_link_helper_removes_standalone_module_extension_paths() -> None:
    helper = _load_helper()
    stale_path = ROOT.parent / "SlicerMotionScoreHRpQCT" / "MotionScoreHRpQCT"
    active_paths = {str((ROOT / "HRpQCTTools" / "MotionScoreHRpQCT").resolve())}

    assert helper["_is_stale_toolbox_path"](stale_path, ROOT, active_paths)
