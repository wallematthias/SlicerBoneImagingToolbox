"""Point local Slicer module discovery at this checkout first.

Run from Slicer's Python console or with Slicer --python-script. This is a
developer helper for testing a worktree without stale module paths shadowing it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import slicer


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text == "@Invalid()":
        return []
    return [text]


def _repo_root():
    script_path = globals().get("SCRIPT_PATH") or __file__
    return Path(script_path).expanduser().resolve().parents[1]


def _worktree_module_paths(repo_root):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from SlicerBoneImagingToolboxLib.registry import toolbox_module_dirs

    return [str(path) for path in toolbox_module_dirs(repo_root) if path.is_dir()]


def _is_shadowing_toolbox_module(path, repo_root, active_paths):
    resolved = Path(path).expanduser().resolve()
    if str(resolved) in active_paths:
        return False
    if resolved.name in {Path(active).name for active in active_paths}:
        marker = "SlicerBoneImagingToolbox"
        return marker in resolved.parts or "SlicerBoneImagingToolbox-private" in resolved.parts
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return False
    return True


def main():
    repo_root = _repo_root()
    module_paths = _worktree_module_paths(repo_root)
    active_paths = set(module_paths)
    settings = slicer.app.revisionUserSettings()
    key = "Modules/AdditionalPaths"
    kept = []
    for path in _as_list(settings.value(key)):
        if _is_shadowing_toolbox_module(path, repo_root, active_paths):
            continue
        if path not in kept and path not in active_paths:
            kept.append(path)
    updated = module_paths + kept
    settings.setValue(key, updated)
    settings.sync()
    print("Updated Modules/AdditionalPaths:")
    for path in updated:
        print(f"  - {path}")
    print(f"Restart Slicer to load modules from: {repo_root}")
    if bool(globals().get("EXIT_SLICER_AFTER_SETUP", False)):
        slicer.app.exit(0)


if __name__ == "__main__":
    main()
