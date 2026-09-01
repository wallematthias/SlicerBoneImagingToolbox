from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "IOTools" / "DatasetNamingHelper" / "DatasetNamingHelper.py"


def test_dataset_naming_helper_is_registered_with_extension() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = (ROOT / "toolbox_modules.json").read_text(encoding="utf-8")

    assert "add_subdirectory(IOTools/DatasetNamingHelper)" in cmake
    assert '"path": "IOTools/DatasetNamingHelper"' in manifest
    assert '"title": "Dataset Naming Helper"' in manifest
    assert '"section": "I/O"' in manifest


def test_dataset_naming_helper_uses_shared_discovery_and_naming_api() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "build_naming_rows" in source
    assert "build_rename_plan" in source
    assert "execute_rename_plan" in source
    assert "undo_rename_manifest" in source
    assert "suggested_filename" in source
    assert "metadata_reader=self._read_metadata" in source
    assert "py_aimio.aim_info" in source
    assert "discover_raw_sessions" not in source
    assert "def _parse_filename" not in source
    assert 'parent.contributors = ["Matthias Walle"]' in source
    assert "Author: Matthias Walle" in source


def test_dataset_naming_helper_ui_supports_review_and_sidecar_export() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "self.dataRootSelector" in source
    assert "self.discoverButton" in source
    assert "self.namingTable" in source
    assert "self.writeSidecarsButton" in source
    assert "self.renameButton" in source
    assert "self.undoRenameButton" in source
    assert "Subject" in source
    assert "Session" in source
    assert "Site category" in source
    assert "Suggested filename" in source
    assert "Problem" in source
    assert "review recommended" in source.lower()


def test_dataset_naming_helper_can_rename_and_undo_with_manifest() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "Rename files" in source
    assert "Undo rename" in source
    assert "dataset_rename_manifest.json" in source
    assert "def _rename_files" in source
    assert "def _undo_rename" in source
