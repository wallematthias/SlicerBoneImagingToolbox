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
