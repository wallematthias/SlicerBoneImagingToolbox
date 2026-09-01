"""Verify Slicer imports Bone Imaging Toolbox modules from this checkout."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import slicer


MODULES = (
    "TimelapsedHRpQCT",
    "MotionScoreHRpQCT",
    "SegmentationHRpQCT",
    "BoneMicroarchitecture",
    "PlateRodMorphometryHRpQCT",
    "ParOSolFEA",
    "MechanoregulationHRpQCT",
    "ScancoIO",
    "SpineSegmentationCT",
    "BoneImagingToolboxSetup",
)


def _repo_root():
    script_path = globals().get("SCRIPT_PATH") or __file__
    return Path(script_path).expanduser().resolve().parents[1]


def main():
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    imported = {}
    for name in MODULES:
        module = importlib.import_module(name)
        module_file = Path(getattr(module, "__file__", "")).resolve()
        if not module_file.is_relative_to(repo_root):
            raise RuntimeError(f"{name} imported from {module_file}, expected under {repo_root}")
        imported[name] = str(module_file)

    from SlicerBoneImagingToolboxLib import motionscore_scene, spine_segmentation_batch, timelapsed_scene

    helpers = {
        "motionscore_scene": Path(motionscore_scene.__file__).resolve(),
        "spine_segmentation_batch": Path(spine_segmentation_batch.__file__).resolve(),
        "timelapsed_scene": Path(timelapsed_scene.__file__).resolve(),
    }
    for name, path in helpers.items():
        if not path.is_relative_to(repo_root):
            raise RuntimeError(f"{name} imported from {path}, expected under {repo_root}")

    print(f"Verified Bone Imaging Toolbox checkout: {repo_root}")
    for name, path in imported.items():
        print(f"  - {name}: {path}")
    for name, path in helpers.items():
        print(f"  - {name}: {path}")
    if bool(globals().get("EXIT_SLICER_AFTER_VERIFY", False)):
        slicer.app.exit(0)


if __name__ == "__main__":
    main()
