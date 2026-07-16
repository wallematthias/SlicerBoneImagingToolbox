from __future__ import annotations

from pathlib import Path
import json
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def _cmake_project_name() -> str:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"^\s*project\s*\(\s*([A-Za-z0-9_.+-]+)", cmake, re.MULTILINE)
    assert match, "Top-level CMakeLists.txt must declare project(<extension-name>)."
    return match.group(1)


def _s4ext_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        fields[key] = value.strip()
    return fields


def test_extension_description_name_matches_cmake_project_and_manifest() -> None:
    project_name = _cmake_project_name()
    extension_files = sorted(REPO_ROOT.glob("*.s4ext"))
    manifest = json.loads((REPO_ROOT / "toolbox_modules.json").read_text(encoding="utf-8"))

    assert [path.name for path in extension_files] == [f"{project_name}.s4ext"]
    assert manifest["name"] == project_name


def test_extension_metadata_uses_current_toolbox_repository() -> None:
    project_name = _cmake_project_name()
    fields = _s4ext_fields(REPO_ROOT / f"{project_name}.s4ext")
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert fields["scmurl"] == "https://github.com/wallematthias/SlicerBoneImagingToolbox.git"
    assert fields["homepage"] == "https://github.com/wallematthias/SlicerBoneImagingToolbox"
    assert fields["category"] == "Quantification"
    assert fields["enabled"] == "1"

    assert f'set(EXTENSION_HOMEPAGE "{fields["homepage"]}")' in cmake
    assert f'set(EXTENSION_CATEGORY "{fields["category"]}")' in cmake
    assert f'set(EXTENSION_ICONURL "{fields["iconurl"]}")' in cmake
    assert f'set(EXTENSION_SCREENSHOTURLS "{fields["screenshoturls"]}")' in cmake


def test_extension_metadata_has_no_legacy_packaging_names() -> None:
    packaging_files = [
        REPO_ROOT / "CMakeLists.txt",
        REPO_ROOT / "BoneImagingToolbox.s4ext",
        REPO_ROOT / "toolbox_modules.json",
    ]
    legacy_names = ("HRpQCTToolbox", "TimelapsedHRpQCTSlicer")

    for path in packaging_files:
        text = path.read_text(encoding="utf-8")
        for legacy_name in legacy_names:
            assert legacy_name not in text, f"{path.name} still references {legacy_name}"
