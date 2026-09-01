from pathlib import Path
import sys

import slicer


DEFAULT_BUILTIN_MODULE_DIRS = (
    "Setup/BoneImagingToolboxSetup",
    "IOTools/ScancoIO",
    "HRpQCTTools/MotionScoreHRpQCT",
    "HRpQCTTools/SegmentationHRpQCT",
    "HRpQCTTools/DeriveLabelsHRpQCT",
    "HRpQCTTools/TimelapsedHRpQCT",
    "HRpQCTTools/MechanoregulationHRpQCT",
    "HRpQCTTools/BoneMicroarchitecture",
    "HRpQCTTools/PlateRodMorphometryHRpQCT",
    "HRpQCTTools/ParOSolFEA",
    "CTTools/SpineSegmentationCT",
)
LEGACY_MODULE_DIR_NAMES = {
    "TimelapsedHRpQCT",
    "MotionScoreHRpQCT",
    "ScancoIO",
    "SegmentationHRpQCT",
    "DeriveLabelsHRpQCT",
    "RegisteredCommonRegion",
    "MicroarchitectureHRpQCT",
    "BoneMicroarchitecture",
    "PlateRodMorphometryHRpQCT",
    "HRpQCTSegmentation",
    "SpineSegmentationCT",
    "BoneImagingToolboxSetup",
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text == "@Invalid()":
        return []
    return [text]


def _repo_root():
    script_path = globals().get("SCRIPT_PATH")
    if script_path:
        return Path(script_path).expanduser().resolve().parents[1]
    return Path(__file__).resolve().parents[1]


def _is_stale_python_relative_path(path, module_names):
    path_obj = Path(path)
    return (
        "Slicer.app" in path_obj.parts
        and "Python" in path_obj.parts
        and path_obj.name in module_names
    )


def _is_stale_toolbox_path(path, repo_root, active_module_paths):
    path_obj = Path(path).expanduser()
    resolved = path_obj.resolve()
    if str(resolved) in active_module_paths:
        return False
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return False
    return resolved.name in LEGACY_MODULE_DIR_NAMES


def _load_registry(repo_root):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from SlicerBoneImagingToolboxLib.registry import (
            TOOLBOX_DISPLAY_NAME,
            builtin_module_dirs,
            toolbox_module_dirs,
        )

        return (
            TOOLBOX_DISPLAY_NAME,
            builtin_module_dirs(repo_root),
            toolbox_module_dirs(repo_root),
        )
    except Exception:
        return (
            "Bone Imaging Toolbox",
            DEFAULT_BUILTIN_MODULE_DIRS,
            tuple((repo_root / name).resolve() for name in DEFAULT_BUILTIN_MODULE_DIRS),
        )


def main():
    repo_root = _repo_root()
    display_name, builtin_module_dirs, module_dirs = _load_registry(repo_root)
    missing_builtin_dirs = [name for name in builtin_module_dirs if not (repo_root / name).is_dir()]
    if missing_builtin_dirs:
        raise RuntimeError(
            f"Could not resolve the {display_name} repository root. "
            f"Missing built-in module folders: {', '.join(missing_builtin_dirs)}. "
            "Run this script with SCRIPT_PATH set to the full script path, as shown in the README."
        )
    module_paths = [str(path) for path in module_dirs if path.is_dir()]
    active_module_paths = set(module_paths)
    module_names = {Path(path).name for path in module_paths}

    settings = slicer.app.revisionUserSettings()
    key = "Modules/AdditionalPaths"
    current = [
        path
        for path in _as_list(settings.value(key))
        if not _is_stale_python_relative_path(path, module_names)
        and not _is_stale_toolbox_path(path, repo_root, active_module_paths)
    ]

    for path in module_paths:
        if path not in current:
            current.append(path)

    settings.setValue(key, current)
    settings.sync()
    print("Updated Modules/AdditionalPaths:")
    for path in current:
        print(f"  - {path}")
    print(f"Restart Slicer to load the {display_name} modules.")


if __name__ == "__main__":
    main()
