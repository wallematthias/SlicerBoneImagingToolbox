from pathlib import Path
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SlicerBoneImagingToolboxLib.registry import discover_external_module_dirs, toolbox_module_dirs
from SlicerBoneImagingToolboxLib.updater import (
    ModuleUpdateContext,
    detect_update_context,
    find_git_root,
    find_toolbox_root,
)


def _make_toolbox(root: Path) -> Path:
    for relative_path in (
        "HRpQCTTools/TimelapsedHRpQCT",
        "HRpQCTTools/MotionScoreHRpQCT",
        "HRpQCTTools/SegmentationHRpQCT",
        "IOTools/ScancoIO",
        "CTTools/SpineSegmentationCT",
    ):
        module_dir = root / relative_path
        name = module_dir.name
        module_dir.mkdir(parents=True, exist_ok=True)
        (module_dir / f"{name}.py").write_text("# module\n", encoding="utf-8")
    return root


def test_find_toolbox_root_from_module_file(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "SlicerBoneImagingToolbox-main")
    module_file = toolbox / "HRpQCTTools" / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py"

    assert find_toolbox_root(module_file) == toolbox


def test_find_git_root_detects_normal_checkout(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "checkout")
    (toolbox / ".git").mkdir()

    assert find_git_root(toolbox / "IOTools" / "ScancoIO" / "ScancoIO.py") == toolbox


def test_find_git_root_detects_worktree_git_file(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "worktree")
    (toolbox / ".git").write_text("gitdir: ../.git/worktrees/worktree\n", encoding="utf-8")

    assert find_git_root(toolbox / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py") == toolbox


def test_find_git_root_ignores_unrelated_parent_git_checkout(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / ".git").mkdir()
    toolbox = _make_toolbox(parent / "manual-download")

    assert find_git_root(toolbox / "IOTools" / "ScancoIO" / "ScancoIO.py") is None


def test_detect_update_context_prefers_git_strategy_for_git_clone(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "checkout")
    (toolbox / ".git").mkdir()

    context = detect_update_context(toolbox / "HRpQCTTools" / "MotionScoreHRpQCT" / "MotionScoreHRpQCT.py")

    assert context == ModuleUpdateContext(toolbox_root=toolbox, git_root=toolbox, strategy="git")


def test_detect_update_context_uses_zip_strategy_for_manual_download(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "SlicerBoneImagingToolbox-main")

    context = detect_update_context(toolbox / "HRpQCTTools" / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py")

    assert context == ModuleUpdateContext(toolbox_root=toolbox, git_root=None, strategy="zip")


def test_toolbox_module_dirs_include_external_scripted_modules(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "SlicerBoneImagingToolbox-main")
    external_module = toolbox / "ExternalModules" / "SlicerParOSol" / "ParOSolFEA"
    external_module.mkdir(parents=True)
    (external_module / "CMakeLists.txt").write_text("project(ParOSolFEA)\n", encoding="utf-8")
    (external_module / "ParOSolFEA.py").write_text("# module\n", encoding="utf-8")

    assert discover_external_module_dirs(toolbox) == (external_module.resolve(),)
    assert toolbox_module_dirs(toolbox)[-1] == external_module.resolve()


def test_builtin_modules_use_expected_slicer_subcategories() -> None:
    expected = {
        "HRpQCTTools/TimelapsedHRpQCT/TimelapsedHRpQCT.py": 'parent.categories = ["Bone Imaging.Timelapsed Methods"]',
        "HRpQCTTools/MotionScoreHRpQCT/MotionScoreHRpQCT.py": 'parent.categories = ["Bone Imaging.Quality Control"]',
        "HRpQCTTools/SegmentationHRpQCT/SegmentationHRpQCT.py": 'parent.categories = ["Bone Imaging.Segmentation Methods"]',
        "HRpQCTTools/DeriveLabelsHRpQCT/DeriveLabelsHRpQCT.py": 'parent.categories = ["Bone Imaging.Segmentation Methods"]',
        "HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py": 'parent.categories = ["Bone Imaging.Analysis Methods"]',
        "HRpQCTTools/PlateRodMorphometryHRpQCT/PlateRodMorphometryHRpQCT.py": 'parent.categories = ["Bone Imaging.Analysis Methods"]',
        "IOTools/ScancoIO/ScancoIO.py": 'parent.categories = ["Bone Imaging.I/O"]',
        "CTTools/SpineSegmentationCT/SpineSegmentationCT.py": 'parent.categories = ["Bone Imaging.Segmentation Methods"]',
        "Setup/BoneImagingToolboxSetup/BoneImagingToolboxSetup.py": 'parent.categories = ["Bone Imaging.Setup"]',
    }
    for relative_path, category_line in expected.items():
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert category_line in source

    segmentation_source = (
        REPO_ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")
    assert 'parent.title = "Bone Contours"' in segmentation_source

    manifest = json.loads((REPO_ROOT / "toolbox_modules.json").read_text(encoding="utf-8"))
    sections = {module["path"]: module["section"] for module in manifest["modules"]}
    expected_sections = {
        "HRpQCTTools/TimelapsedHRpQCT": "Timelapsed Methods",
        "HRpQCTTools/MotionScoreHRpQCT": "Quality Control",
        "HRpQCTTools/SegmentationHRpQCT": "Segmentation Methods",
        "HRpQCTTools/DeriveLabelsHRpQCT": "Segmentation Methods",
        "HRpQCTTools/BoneMicroarchitecture": "Analysis Methods",
        "HRpQCTTools/PlateRodMorphometryHRpQCT": "Analysis Methods",
        "IOTools/ScancoIO": "I/O",
        "CTTools/SpineSegmentationCT": "Segmentation Methods",
        "Setup/BoneImagingToolboxSetup": "Setup",
    }
    for path, section in expected_sections.items():
        assert sections[path] == section


def test_local_link_helper_fallback_includes_all_builtin_modules() -> None:
    helper = (REPO_ROOT / "scripts" / "link_local_toolbox_modules.py").read_text(encoding="utf-8")

    assert '"CTTools/SpineSegmentationCT"' in helper
    assert '"HRpQCTTools/DeriveLabelsHRpQCT"' in helper
    assert '"HRpQCTTools/PlateRodMorphometryHRpQCT"' in helper
    assert '"Setup/BoneImagingToolboxSetup"' in helper
    assert '"PlateRodMorphometryHRpQCT"' in helper
    assert '"DeriveLabelsHRpQCT"' in helper
    assert '"BoneImagingToolboxSetup"' in helper
