from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import re
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
    suggested_mids_relative_path,
    suggested_mids_relative_paths,
    split_identity_metadata,
    undo_rename_manifest,
)
import bone_imaging_derivatives.naming as _naming_api  # noqa: E402

if not hasattr(_naming_api, "apply_naming_row_overrides"):
    _naming_api = importlib.reload(_naming_api)
apply_naming_row_overrides = _naming_api.apply_naming_row_overrides


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
    "Suggested path",
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

    def default_private_identity_manifest_path(self, data_root):
        return Path(str(data_root or "")).expanduser() / "PRIVATE_DO_NOT_SHARE" / "dataset_private_identity_manifest.json"

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

    def anonymize_metadata(self, rows, private_manifest_path):
        private_records = []
        public_count = 0
        for row in rows:
            path = Path(row["File"])
            original_table_metadata = {
                "subject_id": row["Subject"],
                "session_id": row["Session"],
                "site": row["Site"],
                "role": row["Role"],
            }
            if row["Stack"].strip():
                original_table_metadata["stack_index"] = row["Stack"].strip()
            raw_metadata = self._read_metadata(path) or {}
            public_metadata, private_metadata = split_identity_metadata({**raw_metadata, **original_table_metadata})
            public_metadata.update(self._public_metadata_from_suggested_path(row))
            sidecar = self.write_sidecar(path, public_metadata)
            public_count += 1
            private_metadata["original_table_metadata"] = original_table_metadata
            if private_metadata:
                private_records.append(
                    {
                        "source_path": str(path),
                        "public_sidecar_path": str(sidecar),
                        "private_metadata": private_metadata,
                    }
                )
        manifest_path = Path(private_manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 1,
            "warning": "Private metadata may contain identifiers. Do not share this folder.",
            "records": private_records,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return manifest_path, public_count, len(private_records)

    def _public_metadata_from_suggested_path(self, row):
        suggested = str(row.get("Suggested path", "") or "")
        metadata = {}
        subject_match = re.search(r"(?:^|/)sub-([^/]+)", suggested)
        session_match = re.search(r"(?:^|/)ses-([^/]+)", suggested)
        voi_match = re.search(r"_voi-([A-Za-z0-9]+)", suggested)
        if subject_match:
            metadata["subject_id"] = subject_match.group(1)
        if session_match:
            metadata["session_id"] = session_match.group(1)
        if voi_match:
            metadata["site"] = voi_match.group(1)
        return metadata

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

        self.discoverButton = qt.QPushButton("Analyze")
        self.discoverButton.setStyleSheet("QPushButton { background-color: #2563eb; color: white; font-weight: 600; padding: 6px 12px; }")
        self.discoverButton.toolTip = "Analyze filenames and metadata with shared Bone Imaging discovery rules."
        analyze_row = qt.QHBoxLayout()
        analyze_row.addWidget(self.discoverButton)
        analyze_row.addStretch(1)
        layout.addLayout(analyze_row)

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

        buttons = qt.QHBoxLayout()
        self.exportPlanButton = qt.QPushButton("Export plan")
        self.anonymizeButton = qt.QPushButton("Anonymize metadata")
        self.renameButton = qt.QPushButton("Rename files")
        self.undoRenameButton = qt.QPushButton("Undo rename")
        self.exportPlanButton.toolTip = "Export the current naming review table as CSV."
        self.anonymizeButton.toolTip = (
            "Write pseudonymized public discovery sidecars and move raw AIM/header metadata into PRIVATE_DO_NOT_SHARE."
        )
        self.renameButton.toolTip = "Reformat files to the normalized sub-*/ses-*/xct layout and write a reversible manifest."
        self.undoRenameButton.toolTip = "Restore original filenames from the rename manifest."
        buttons.addWidget(self.exportPlanButton)
        buttons.addWidget(self.anonymizeButton)
        buttons.addWidget(self.renameButton)
        buttons.addWidget(self.undoRenameButton)
        buttons.addStretch(1)
        table_layout.addLayout(buttons)

        hint = qt.QLabel(
            "Rows marked low confidence or with missing fields are review recommended. "
            "Side-specific sites such as RL/RR remain separate; the site category column shows the preset family."
        )
        hint.wordWrap = True
        table_layout.addWidget(hint)

        self.discoverButton.clicked.connect(self._analyze)
        self.namingTable.itemChanged.connect(self._on_table_item_changed)
        self.renameButton.clicked.connect(self._rename_files)
        self.undoRenameButton.clicked.connect(self._undo_rename)
        self.exportPlanButton.clicked.connect(self._export_plan)
        self.anonymizeButton.clicked.connect(self._anonymize_metadata)

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
        suggestions = suggested_mids_relative_paths(rows)
        self.namingTable.blockSignals(True)
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
                "Suggested path": str(suggestions.get(Path(row.path), suggested_mids_relative_path(row))),
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
        self.namingTable.blockSignals(False)

    def _on_table_item_changed(self, item):
        if item is None:
            return
        row_index = item.row()
        header = HEADERS[item.column()]
        if header not in EDITABLE_COLUMNS or row_index < 0 or row_index >= len(self._rows):
            return
        overrides = {
            "Role": "role",
            "Subject": "subject_id",
            "Session": "session_id",
            "Site": "site",
            "Stack": "stack_index",
        }
        row_overrides = {overrides[header]: item.text()}
        self._rows[row_index] = apply_naming_row_overrides(self._rows[row_index], row_overrides)
        self._refresh_derived_table_cells()

    def _refresh_derived_table_cells(self):
        suggestions = suggested_mids_relative_paths(self._rows)
        self.namingTable.blockSignals(True)
        try:
            for row_index, row in enumerate(self._rows):
                values = {
                    "Role": row.role,
                    "Subject": row.subject_id or "",
                    "Session": row.session_id or "",
                    "Site": row.site or "",
                    "Site category": row.site_category or "",
                    "Stack": "" if row.stack_index is None else str(row.stack_index),
                    "Confidence": row.confidence,
                    "Problem": row.problem,
                    "Suggested path": str(suggestions.get(Path(row.path), suggested_mids_relative_path(row))),
                }
                for header, value in values.items():
                    col_index = HEADERS.index(header)
                    table_item = self.namingTable.item(row_index, col_index)
                    if table_item is None:
                        table_item = qt.QTableWidgetItem()
                        if header not in EDITABLE_COLUMNS:
                            table_item.setFlags(table_item.flags() & ~qt.Qt.ItemIsEditable)
                        self.namingTable.setItem(row_index, col_index, table_item)
                    table_item.setText(value)
                    table_item.setToolTip(value)
                    if header in {"Problem", "Confidence"} and row.problem:
                        table_item.setBackground(qt.QColor(255, 245, 204))
                    else:
                        table_item.setBackground(qt.QColor(255, 255, 255))
            problems = sum(1 for row in self._rows if row.problem)
            self.statusLabel.text = f"Analyzed {len(self._rows)} file(s); {problems} review recommended."
        finally:
            self.namingTable.blockSignals(False)

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

    def _private_identity_manifest_path(self):
        return self.logic.default_private_identity_manifest_path(self.dataRootSelector.currentPath)

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

    def _anonymize_metadata(self):
        root = Path(str(self.dataRootSelector.currentPath or "")).expanduser()
        if not root.exists():
            self.statusLabel.text = "Choose a valid data root before anonymizing metadata."
            return
        try:
            manifest, public_count, private_count = self.logic.anonymize_metadata(
                self._table_rows(),
                self._private_identity_manifest_path(),
            )
        except Exception as exc:
            self.statusLabel.text = f"Could not anonymize metadata: {exc}"
            return
        self.statusLabel.text = (
            f"Wrote {public_count} public sidecar(s) and {private_count} private identity record(s): {manifest}. "
            "AIM headers and filenames are not rewritten by this action; use Rename files for anonymized paths."
        )


class DatasetNamingHelperTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("Dataset Naming Helper smoke test passed.")
