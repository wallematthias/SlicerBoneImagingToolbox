from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


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
    assert "suggested_mids_relative_paths" in source
    assert "metadata_reader=self._read_metadata" in source
    assert "py_aimio.aim_info" in source
    assert "discover_raw_sessions" not in source
    assert "def _parse_filename" not in source
    assert 'parent.contributors = ["Matthias Walle"]' in source
    assert "Author: Matthias Walle" in source


def test_dataset_naming_helper_ui_supports_review_and_reformat_actions() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "self.dataRootSelector" in source
    assert "self.discoverButton" in source
    assert "self.namingTable" in source
    assert "Write sidecars" not in source
    assert "buttons.addWidget(self.exportPlanButton)" in source
    assert "self.anonymizeButton" in source
    assert "buttons.addWidget(self.anonymizeButton)" in source
    assert source.index("buttons.addWidget(self.exportPlanButton)") < source.index("buttons.addWidget(self.renameButton)")
    assert "self.renameButton" in source
    assert "self.undoRenameButton" in source
    assert "Subject" in source
    assert "Session" in source
    assert "Site category" in source
    assert "Suggested path" in source
    assert "Problem" in source
    assert "review recommended" in source.lower()
    assert "self.namingTable.itemChanged.connect(self._on_table_item_changed)" in source
    assert "apply_naming_row_overrides" in source
    assert "def _refresh_derived_table_cells" in source


def test_dataset_naming_helper_can_rename_and_undo_with_manifest() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "Rename files" in source
    assert "sub-*/ses-*/xct" in source
    assert "Undo rename" in source
    assert "dataset_rename_manifest.json" in source
    assert "def _rename_files" in source
    assert "def _undo_rename" in source


def test_dataset_naming_helper_splits_public_and_private_metadata() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "split_identity_metadata" in source
    assert "dataset_private_identity_manifest.json" in source
    assert "def _anonymize_metadata" in source


def test_naming_table_refresh_keeps_theme_defaults_and_readable_review_cells():
    """A cell edit must not turn dark-theme table backgrounds white."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    widget = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "DatasetNamingHelperWidget")
    refresh = next(node for node in widget.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_refresh_derived_table_cells")
    headers = ["Role", "Subject", "Session", "Site", "Site category", "Stack",
               "Confidence", "Problem", "Suggested path"]

    class Item:
        def __init__(self):
            self.background = None
            self.foreground = None

        def setText(self, text):
            self.text = text

        def setToolTip(self, text):
            self.tooltip = text

        def setBackground(self, brush):
            self.background = brush

        def setForeground(self, brush):
            self.foreground = brush

    rows = [SimpleNamespace(path=Path("scan.AIM"), role="image", subject_id="001",
                            session_id="001", site="radius", site_category="radius",
                            stack_index=None, confidence="review", problem=problem)
            for problem in ("", "Check subject")]
    items = {(row, col): Item() for row in range(2) for col in range(len(headers))}
    palette = SimpleNamespace(brush=lambda role: {"highlight": "dark-accent",
                                                  "highlighted-text": "light-text"}[role])
    table = SimpleNamespace(blockSignals=lambda value: None,
                            item=lambda row, col: items[row, col], palette=palette)
    instance = SimpleNamespace(_rows=rows, namingTable=table, statusLabel=SimpleNamespace())
    qt = SimpleNamespace(QColor=lambda *rgb: rgb, QBrush=lambda: None,
                         QPalette=SimpleNamespace(Highlight="highlight",
                                                  HighlightedText="highlighted-text"))
    namespace = {"qt": qt, "HEADERS": headers, "Path": Path,
                 "suggested_mids_relative_paths": lambda rows: {},
                 "suggested_mids_relative_path": lambda row: Path("suggested.AIM")}
    exec(compile(ast.Module(body=[refresh], type_ignores=[]), str(MODULE), "exec"), namespace)
    namespace["_refresh_derived_table_cells"](instance)
    assert all(items[0, col].background is None for col in range(len(headers)))
    assert all(items[0, col].foreground is None for col in range(len(headers)))
    for header in ("Problem", "Confidence"):
        item = items[1, headers.index(header)]
        assert item.background == "dark-accent"
        assert item.foreground == "light-text"
    rows[1].problem = ""
    namespace["_refresh_derived_table_cells"](instance)
    for header in ("Problem", "Confidence"):
        item = items[1, headers.index(header)]
        assert item.background is None
        assert item.foreground is None
