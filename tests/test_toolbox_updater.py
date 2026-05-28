from pathlib import Path

from SlicerTimelapsedHRpQCTLib.updater import (
    ModuleUpdateContext,
    detect_update_context,
    find_git_root,
    find_toolbox_root,
)


def _make_toolbox(root: Path) -> Path:
    for name in ("TimelapsedHRpQCT", "MotionScoreHRpQCT", "HRpQCTSegmentation", "ScancoIO"):
        module_dir = root / name
        module_dir.mkdir(parents=True, exist_ok=True)
        (module_dir / f"{name}.py").write_text("# module\n", encoding="utf-8")
    return root


def test_find_toolbox_root_from_module_file(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "SlicerTimelapsedHRpQCT-main")
    module_file = toolbox / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py"

    assert find_toolbox_root(module_file) == toolbox


def test_find_git_root_detects_normal_checkout(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "checkout")
    (toolbox / ".git").mkdir()

    assert find_git_root(toolbox / "ScancoIO" / "ScancoIO.py") == toolbox


def test_find_git_root_detects_worktree_git_file(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "worktree")
    (toolbox / ".git").write_text("gitdir: ../.git/worktrees/worktree\n", encoding="utf-8")

    assert find_git_root(toolbox / "HRpQCTSegmentation" / "HRpQCTSegmentation.py") == toolbox


def test_find_git_root_ignores_unrelated_parent_git_checkout(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / ".git").mkdir()
    toolbox = _make_toolbox(parent / "manual-download")

    assert find_git_root(toolbox / "ScancoIO" / "ScancoIO.py") is None


def test_detect_update_context_prefers_git_strategy_for_git_clone(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "checkout")
    (toolbox / ".git").mkdir()

    context = detect_update_context(toolbox / "MotionScoreHRpQCT" / "MotionScoreHRpQCT.py")

    assert context == ModuleUpdateContext(toolbox_root=toolbox, git_root=toolbox, strategy="git")


def test_detect_update_context_uses_zip_strategy_for_manual_download(tmp_path: Path) -> None:
    toolbox = _make_toolbox(tmp_path / "SlicerTimelapsedHRpQCT-main")

    context = detect_update_context(toolbox / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py")

    assert context == ModuleUpdateContext(toolbox_root=toolbox, git_root=None, strategy="zip")
