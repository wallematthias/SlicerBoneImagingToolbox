from pathlib import Path

import slicer


MODULE_DIRS = (
    "TimelapsedHRpQCT",
    "MotionScoreHRpQCT",
    "ScancoIO",
    "HRpQCTSegmentation",
)


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


def _is_stale_python_relative_path(path):
    path_obj = Path(path)
    return (
        "Slicer.app" in path_obj.parts
        and "Python" in path_obj.parts
        and path_obj.name in MODULE_DIRS
    )


def main():
    repo_root = _repo_root()
    if not all((repo_root / name).is_dir() for name in MODULE_DIRS):
        raise RuntimeError(
            "Could not resolve the TimelapsedHRpQCTSlicer repository root. "
            "Run this script with SCRIPT_PATH set to the full script path, as shown in the README."
        )
    module_paths = [str((repo_root / name).resolve()) for name in MODULE_DIRS]

    settings = slicer.app.revisionUserSettings()
    key = "Modules/AdditionalPaths"
    current = [
        path
        for path in _as_list(settings.value(key))
        if not _is_stale_python_relative_path(path)
    ]

    for path in module_paths:
        if path not in current:
            current.append(path)

    settings.setValue(key, current)
    settings.sync()
    print("Updated Modules/AdditionalPaths:")
    for path in current:
        print(f"  - {path}")
    print("Restart Slicer to load the HR-pQCT toolbox modules.")


if __name__ == "__main__":
    main()
