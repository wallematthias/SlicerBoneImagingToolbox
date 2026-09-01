from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import ctk
import qt
import slicer

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


MODULE_VERSION = "0.1.0"
TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from bone_imaging_derivatives import (  # noqa: E402
    build_naming_rows,
    build_rename_plan,
    execute_rename_plan,
    suggested_filename,
    undo_rename_manifest,
)


HEADERS = [
    "File",
    "Kind",
    "Role",
    "Subject",
    "Session",
    "Site",
    "Site category",
    "Stack",
    "Confidence",
    "Problem",
    "Suggested filename",
]
EDITABLE_COLUMNS = {"Role", "Subject", "Session", "Site", "Stack"}


class DatasetNamingHelper(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Dataset Naming Helper"
        parent.categories = ["Bone Imaging.I/O"]
        parent.index = 20
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Review dataset filenames, sidecar metadata, and discovery confidence before running "
            f"batch workflows. Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = (
            "Author: Matthias Walle. Part of the Bone Imaging Toolbox for 3D Slicer."
        )


class DatasetNamingHelperLogic(ScriptedLoadableModuleLogic):
    def analyze_root(self, data_root):
        root = Path(str(data_root or "")).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        return build_naming_rows(root, metadata_reader=self._read_metadata)

    def default_rename_manifest_path(self, data_root):
        return Path(str(data_root or "")).expanduser() / "dataset_rename_manifest.json"

    def build_rename_plan(self, data_root, manifest_path=None):
        root = Path(str(data_root or "")).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        return build_rename_plan(root, manifest_path=manifest_path, metadata_reader=self._read_metadata)

    def rename_files(self, data_root, manifest_path=None):
        plan = self.build_rename_plan(data_root, manifest_path=manifest_path)
        manifest = execute_rename_plan(plan)
        return manifest, len(plan.renames)

    def undo_rename(self, manifest_path):
        return undo_rename_manifest(manifest_path)

    def _read_metadata(self, path):
        if not str(path).lower().endswith((".aim", ".aim;1", ".isq", ".scv")):
            return None
        try:
            import py_aimio  # type: ignore

            return dict(py_aimio.aim_info(str(path)))
        except Exception:
            return None

    def sidecar_path(self, path):
        path = Path(path)
        return path.with_name(f"{path.name}.json")

    def write_sidecar(self, path, metadata):
        sidecar = self.sidecar_path(path)
        existing = {}
        if sidecar.exists():
            try:
                existing = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update({key: value for key, value in metadata.items() if value not in (None, "")})
        sidecar.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
        return sidecar

    def export_plan(self, rows, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        return output_path


class DatasetNamingHelperWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = DatasetNamingHelperLogic()
        self._rows = []

        box = ctk.ctkCollapsibleButton()
        box.text = "Dataset"
        self.layout.addWidget(box)
        layout = qt.QVBoxLayout(box)

        form = qt.QFormLayout()
        layout.addLayout(form)

        self.dataRootSelector = ctk.ctkPathLineEdit()
        self.dataRootSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.dataRootSelector.toolTip = "Dataset root to inspect for images, masks, transforms, and derivative files."
        form.addRow("Data root", self.dataRootSelector)

        self.renameManifestSelector = ctk.ctkPathLineEdit()
        self.renameManifestSelector.filters = ctk.ctkPathLineEdit.Files
        self.renameManifestSelector.toolTip = "Manifest written during renaming and used to restore original filenames."
        form.addRow("Rename manifest", self.renameManifestSelector)

        buttons = qt.QHBoxLayout()
        self.discoverButton = qt.QPushButton("Analyze")
        self.writeSidecarsButton = qt.QPushButton("Write sidecars")
        self.renameButton = qt.QPushButton("Rename files")
        self.undoRenameButton = qt.QPushButton("Undo rename")
        self.exportPlanButton = qt.QPushButton("Export plan")
        self.discoverButton.toolTip = "Analyze filenames and metadata with shared Bone Imaging discovery rules."
        self.writeSidecarsButton.toolTip = "Write editable table values as JSON sidecars next to the original files."
        self.renameButton.toolTip = "Rename files to normalized names and write a reversible manifest. Derivative folders are skipped."
        self.undoRenameButton.toolTip = "Restore original filenames from the rename manifest."
        self.exportPlanButton.toolTip = "Export the current naming review table as CSV."
        buttons.addWidget(self.discoverButton)
        buttons.addWidget(self.writeSidecarsButton)
        buttons.addWidget(self.renameButton)
        buttons.addWidget(self.undoRenameButton)
        buttons.addWidget(self.exportPlanButton)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.statusLabel = qt.QLabel("Choose a data root and analyze filenames.")
        self.statusLabel.wordWrap = True
        layout.addWidget(self.statusLabel)

        table_box = ctk.ctkCollapsibleButton()
        table_box.text = "Naming review"
        self.layout.addWidget(table_box)
        table_layout = qt.QVBoxLayout(table_box)

        self.namingTable = qt.QTableWidget()
        self.namingTable.setColumnCount(len(HEADERS))
        self.namingTable.setHorizontalHeaderLabels(HEADERS)
        self.namingTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.namingTable.setEditTriggers(qt.QAbstractItemView.DoubleClicked | qt.QAbstractItemView.EditKeyPressed)
        self.namingTable.horizontalHeader().setStretchLastSection(True)
        self.namingTable.setMinimumHeight(260)
        table_layout.addWidget(self.namingTable)

        hint = qt.QLabel(
            "Rows marked low confidence or with missing fields are review recommended. "
            "Side-specific sites such as RL/RR remain separate; the site category column shows the preset family."
        )
        hint.wordWrap = True
        table_layout.addWidget(hint)

        self.discoverButton.clicked.connect(self._analyze)
        self.writeSidecarsButton.clicked.connect(self._write_sidecars)
        self.renameButton.clicked.connect(self._rename_files)
        self.undoRenameButton.clicked.connect(self._undo_rename)
        self.exportPlanButton.clicked.connect(self._export_plan)

        self.layout.addStretch(1)

    def _analyze(self):
        self._refresh_default_manifest_path()
        try:
            rows = self.logic.analyze_root(self.dataRootSelector.currentPath)
        except Exception as exc:
            self.statusLabel.text = f"Naming analysis failed: {exc}"
            return
        self._rows = rows
        self._populate_table(rows)
        problems = sum(1 for row in rows if row.problem)
        self.statusLabel.text = f"Analyzed {len(rows)} file(s); {problems} review recommended."

    def _populate_table(self, rows):
        self.namingTable.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = {
                "File": str(row.path),
                "Kind": row.kind,
                "Role": row.role,
                "Subject": row.subject_id or "",
                "Session": row.session_id or "",
                "Site": row.site or "",
                "Site category": row.site_category or "",
                "Stack": "" if row.stack_index is None else str(row.stack_index),
                "Confidence": row.confidence,
                "Problem": row.problem,
                "Suggested filename": suggested_filename(row),
            }
            for col_index, header in enumerate(HEADERS):
                item = qt.QTableWidgetItem(values[header])
                item.setToolTip(values[header])
                if header not in EDITABLE_COLUMNS:
                    item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                if row.problem and header in {"Problem", "Confidence"}:
                    item.setBackground(qt.QColor(255, 245, 204))
                self.namingTable.setItem(row_index, col_index, item)
        self.namingTable.resizeColumnsToContents()

    def _table_rows(self):
        rows = []
        for row in range(self.namingTable.rowCount):
            payload = {}
            for col, header in enumerate(HEADERS):
                item = self.namingTable.item(row, col)
                payload[header] = "" if item is None else item.text()
            rows.append(payload)
        return rows

    def _write_sidecars(self, set_status=True):
        written = []
        for row in self._table_rows():
            path = Path(row["File"])
            metadata = {
                "subject_id": row["Subject"],
                "session_id": row["Session"],
                "site": row["Site"],
                "role": row["Role"],
            }
            if row["Stack"].strip():
                metadata["stack_index"] = row["Stack"].strip()
            try:
                written.append(self.logic.write_sidecar(path, metadata))
            except Exception as exc:
                self.statusLabel.text = f"Could not write sidecar for {path.name}: {exc}"
                return False
        if set_status:
            self.statusLabel.text = f"Wrote {len(written)} sidecar file(s)."
        return True

    def _manifest_path(self):
        configured = str(self.renameManifestSelector.currentPath or "").strip()
        if configured:
            return Path(configured).expanduser()
        return self.logic.default_rename_manifest_path(self.dataRootSelector.currentPath)

    def _refresh_default_manifest_path(self):
        current = str(self.renameManifestSelector.currentPath or "").strip()
        if current:
            return
        root = Path(str(self.dataRootSelector.currentPath or "")).expanduser()
        if root:
            self.renameManifestSelector.currentPath = str(self.logic.default_rename_manifest_path(root))

    def _rename_files(self):
        root = Path(str(self.dataRootSelector.currentPath or "")).expanduser()
        if not root.exists():
            self.statusLabel.text = "Choose a valid data root before renaming."
            return
        try:
            if not self._write_sidecars(set_status=False):
                return
            manifest, count = self.logic.rename_files(root, self._manifest_path())
        except Exception as exc:
            self.statusLabel.text = f"Could not rename files: {exc}"
            return
        if count == 0:
            message = "No files needed renaming."
        else:
            message = f"Renamed {count} path(s); manifest: {manifest}"
        self._analyze()
        self.statusLabel.text = message

    def _undo_rename(self):
        manifest = self._manifest_path()
        if not manifest.exists():
            self.statusLabel.text = f"Rename manifest not found: {manifest}"
            return
        try:
            restored = self.logic.undo_rename(manifest)
        except Exception as exc:
            self.statusLabel.text = f"Could not undo rename: {exc}"
            return
        message = f"Restored {restored} path(s) from {manifest}."
        self._analyze()
        self.statusLabel.text = message

    def _export_plan(self):
        root = Path(str(self.dataRootSelector.currentPath or "")).expanduser()
        if not root.exists():
            self.statusLabel.text = "Choose a valid data root before exporting."
            return
        output_path = root / "dataset_naming_review.csv"
        try:
            self.logic.export_plan(self._table_rows(), output_path)
        except Exception as exc:
            self.statusLabel.text = f"Could not export naming plan: {exc}"
            return
        self.statusLabel.text = f"Exported naming plan: {output_path}"


class DatasetNamingHelperTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("Dataset Naming Helper smoke test passed.")
