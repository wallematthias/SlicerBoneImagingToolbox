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
    return [text] if text else []


def main():
    repo_root = Path(__file__).resolve().parents[1]
    module_paths = [str((repo_root / name).resolve()) for name in MODULE_DIRS]

    settings = slicer.app.revisionUserSettings()
    key = "Modules/AdditionalPaths"
    current = _as_list(settings.value(key))

    for path in module_paths:
        if path not in current:
            current.append(path)

    settings.setValue(key, tuple(current))
    print("Updated Modules/AdditionalPaths:")
    for path in current:
        print(f"  - {path}")
    print("Restart Slicer to load the HR-pQCT toolbox modules.")


if __name__ == "__main__":
    main()
