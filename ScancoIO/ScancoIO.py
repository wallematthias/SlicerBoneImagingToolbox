import json
import tempfile
from pathlib import Path
import importlib
import sys

import qt
import ctk
import slicer
import SimpleITK as sitk

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


MODULE_VERSION = "0.1.0"
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
TOOLBOX_ROOT = MODULE_DIR.parent
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from SlicerTimelapsedHRpQCTLib.slicer_update_ui import run_toolbox_update_dialog

AIM_METADATA_ATTRIBUTE = "HRpQCT.AIMMetadata"
AIM_SOURCE_ATTRIBUTE = "HRpQCT.AIMSourcePath"
AIM_SCALING_ATTRIBUTE = "HRpQCT.AIMScaling"


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _metadata_json(metadata):
    return json.dumps(metadata or {}, indent=2, sort_keys=True, default=_json_default)


def _image_geometry_metadata(image):
    return {
        "dimensions": tuple(int(v) for v in image.GetSize()),
        "spacing": tuple(float(v) for v in image.GetSpacing()),
        "element_size": tuple(float(v) for v in image.GetSpacing()),
        "origin": tuple(float(v) for v in image.GetOrigin()),
        "direction": tuple(float(v) for v in image.GetDirection()),
    }


def _parse_table_value(text):
    text = str(text).strip()
    if not text:
        return ""
    if "," in text:
        return [_parse_table_value(part) for part in text.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _aim_io_module():
    from ScancoIOLib import aim_io

    required = ("is_aimio_available", "read_aim", "write_aim", "log_to_dict")
    if not all(hasattr(aim_io, name) for name in required):
        aim_io = importlib.reload(aim_io)
    return aim_io


def _normalize_processing_log(metadata):
    aim_io = _aim_io_module()

    if not isinstance(metadata, dict):
        return {}
    processing_log = metadata.get("processing_log")
    if isinstance(processing_log, dict):
        return processing_log
    legacy_processing_log = metadata.get("processing_log_dict")
    if isinstance(legacy_processing_log, dict):
        return legacy_processing_log
    raw_log = metadata.get("processing_log_raw")
    if isinstance(raw_log, str) and raw_log.strip():
        return aim_io.log_to_dict(raw_log)
    if isinstance(processing_log, str) and processing_log.strip():
        return aim_io.log_to_dict(processing_log)
    return {}


class ScancoIO(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Scanco I/O"
        parent.categories = ["HR-pQCT"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Import Scanco AIM images into Slicer and export edited grayscale "
            f"or mask volumes back to AIM. Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = "Part of the HR-pQCT Toolbox for 3D Slicer."


class ScancoIOLogic(ScriptedLoadableModuleLogic):
    def is_core_available(self):
        return _aim_io_module().is_aimio_available()

    def install_or_update_core(self):
        slicer.util.pip_install("--upgrade --force-reinstall --no-cache-dir aimio-py")

    def import_aim(
        self,
        aim_path,
        scaling,
        volume_name=None,
        as_segmentation=False,
        reference_volume_node=None,
    ):
        aim_io = _aim_io_module()

        aim_path = Path(aim_path)
        if not aim_path.exists():
            raise FileNotFoundError(f"AIM file does not exist: {aim_path}")
        image, metadata = aim_io.read_aim(aim_path, scaling=scaling)
        name = volume_name.strip() if volume_name else aim_path.stem

        with tempfile.TemporaryDirectory(prefix="hrpqct_aim_import_") as temp_dir:
            nrrd_path = Path(temp_dir) / "imported_aim.nrrd"
            if as_segmentation:
                label_image = sitk.Cast(image != 0, sitk.sitkUInt8)
                if reference_volume_node is not None:
                    self._validate_image_matches_reference(label_image, reference_volume_node)
                sitk.WriteImage(label_image, str(nrrd_path))
                loaded = slicer.util.loadSegmentation(
                    str(nrrd_path),
                    {"name": name},
                    returnNode=True,
                )
            else:
                sitk.WriteImage(image, str(nrrd_path))
                loaded = slicer.util.loadVolume(str(nrrd_path), {"name": name}, returnNode=True)

        if isinstance(loaded, tuple):
            success, volume_node = loaded
        else:
            success, volume_node = bool(loaded), loaded
        if not success or volume_node is None:
            raise RuntimeError(f"Could not load imported AIM volume into Slicer: {aim_path}")

        if as_segmentation:
            segmentation_node = volume_node
            segmentation_node.CreateDefaultDisplayNodes()
            if reference_volume_node is not None:
                segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume_node)
                reference_role = slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole()
                segmentation_node.SetNodeReferenceID(reference_role, reference_volume_node.GetID())
                segmentation_node.SetNodeReferenceID(
                    "HRpQCT.ReferenceVolume",
                    reference_volume_node.GetID(),
                )
                segmentation_node.SetAttribute(
                    "HRpQCT.ReferenceVolumeID",
                    reference_volume_node.GetID(),
                )
                transform_id = reference_volume_node.GetTransformNodeID()
                if transform_id:
                    segmentation_node.SetAndObserveTransformNodeID(transform_id)
            display_node = segmentation_node.GetDisplayNode()
            if display_node is not None:
                display_node.SetVisibility(True)
                display_node.SetVisibility2DFill(True)
                display_node.SetVisibility2DOutline(True)

        metadata_text = json.dumps(metadata, sort_keys=True, default=_json_default)
        volume_node.SetAttribute(AIM_SOURCE_ATTRIBUTE, str(aim_path))
        volume_node.SetAttribute(AIM_SCALING_ATTRIBUTE, scaling)
        volume_node.SetAttribute(
            AIM_METADATA_ATTRIBUTE,
            metadata_text,
        )
        return volume_node

    def _validate_image_matches_reference(self, image, reference_node):
        reference_image_data = reference_node.GetImageData()
        if reference_image_data is not None:
            image_dims = tuple(image.GetSize())
            reference_dims = tuple(reference_image_data.GetDimensions())
            if image_dims != reference_dims:
                raise ValueError(
                    "AIM segmentation dimensions do not match the selected reference volume: "
                    f"segmentation={image_dims}, reference={reference_dims}."
                )

    def export_aim(
        self,
        volume_node,
        output_path,
        *,
        as_mask=False,
        unit="auto",
        metadata_json=None,
        header_metadata=None,
        allow_minimal_metadata=False,
        log="Exported from Slicer HR-pQCT Toolbox",
    ):
        aim_io = _aim_io_module()

        if volume_node is None:
            raise ValueError("Select a scalar volume to export.")
        if not str(output_path).strip():
            raise ValueError("Choose an output AIM path.")
        output_path = Path(output_path)
        if hasattr(volume_node, "IsA") and volume_node.IsA("vtkMRMLLabelMapVolumeNode"):
            as_mask = True

        with tempfile.TemporaryDirectory(prefix="hrpqct_aim_export_") as temp_dir:
            nrrd_path = Path(temp_dir) / "slicer_volume.nrrd"
            if not slicer.util.saveNode(volume_node, str(nrrd_path)):
                raise RuntimeError("Could not save selected Slicer volume for AIM export.")
            image = sitk.ReadImage(str(nrrd_path))

        metadata = None
        metadata_json = Path(metadata_json) if metadata_json else None
        if metadata_json and metadata_json.exists():
            metadata = aim_io.aim_metadata_from_import_json(metadata_json, image, log=log)
        else:
            metadata_text = volume_node.GetAttribute(AIM_METADATA_ATTRIBUTE)
            if metadata_text:
                metadata = json.loads(metadata_text)

        if header_metadata:
            metadata = {**(metadata or {}), **header_metadata}

        if metadata is not None:
            metadata.update(_image_geometry_metadata(image))
            metadata.setdefault("position", (0, 0, 0))
            metadata.setdefault("offset", (0, 0, 0))

        if metadata is None and not allow_minimal_metadata:
            raise ValueError(
                "No AIM metadata is attached to this volume. Import an AIM with this module, "
                "provide an imported-stack metadata JSON, or enable minimal metadata export."
            )

        write_unit = "native" if as_mask else None
        if not as_mask and unit and unit != "auto":
            write_unit = unit
        if metadata is not None and as_mask:
            metadata["unit"] = "native"
        aim_io.write_aim(image, output_path, metadata=metadata, unit=write_unit, mask=as_mask)
        return output_path


class ScancoIOWidget(ScriptedLoadableModuleWidget):
    def _tip(self, widget, text):
        widget.toolTip = str(text)
        return widget

    def setup(self):
        super().setup()
        self.logic = ScancoIOLogic()

        self._build_import_section()
        self._build_export_section()
        self._build_log_section()
        self.layout.addStretch(1)
        self._log("Ready.")

    def _build_import_section(self):
        collapsible = ctk.ctkCollapsibleButton()
        collapsible.text = "Import AIM"
        self.layout.addWidget(collapsible)
        form = qt.QFormLayout(collapsible)

        self.installButton = qt.QPushButton("Install / Update AIM I/O")
        self.updateToolboxButton = qt.QPushButton("Check toolbox updates")
        self._tip(self.installButton, "Install or update the lightweight AIM I/O dependency in Slicer Python.")
        self._tip(self.updateToolboxButton, "Check whether this local Slicer toolbox checkout has upstream updates.")
        self.installButton.clicked.connect(self._install_core)
        self.updateToolboxButton.clicked.connect(self._check_toolbox_updates)
        installRowWidget = qt.QWidget()
        installRow = qt.QHBoxLayout(installRowWidget)
        installRow.setContentsMargins(0, 0, 0, 0)
        installRow.addWidget(self.installButton)
        installRow.addWidget(self.updateToolboxButton)
        form.addRow(installRowWidget)

        self.importPathEdit = qt.QLineEdit()
        self.importPathEdit.textChanged.connect(self._on_import_path_changed)
        self._lastAutoVolumeName = ""
        browse = qt.QPushButton("Browse...")
        browse.clicked.connect(self._browse_import_path)
        self._tip(self.importPathEdit, "Path to the Scanco AIM file to import. The volume name updates when this path changes.")
        self._tip(browse, "Select an AIM file from disk.")
        row = qt.QHBoxLayout()
        row.addWidget(self.importPathEdit)
        row.addWidget(browse)
        form.addRow("AIM file", row)

        self.scalingCombo = qt.QComboBox()
        for label, value in [
            ("Density/BMD", "bmd"),
            ("Native Scanco values", "native"),
            ("Mu", "mu"),
            ("HU", "hu"),
        ]:
            self.scalingCombo.addItem(label, value)
        self._tip(self.scalingCombo, "Numeric scaling for scalar-volume import. Segmentations always load from native nonzero values.")
        form.addRow("Load values as", self.scalingCombo)

        self.importAsCombo = qt.QComboBox()
        self.importAsCombo.addItem("Scalar volume", "volume")
        self.importAsCombo.addItem("Segmentation (nonzero mask)", "segmentation")
        self.importAsCombo.currentIndexChanged.connect(self._on_import_as_changed)
        self._tip(self.importAsCombo, "Load AIM as an editable scalar volume or as a Slicer segmentation from nonzero voxels.")
        form.addRow("Load into Slicer as", self.importAsCombo)

        self.importReferenceSelector = slicer.qMRMLNodeComboBox()
        self.importReferenceSelector.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode"]
        self.importReferenceSelector.selectNodeUponCreation = False
        self.importReferenceSelector.addEnabled = False
        self.importReferenceSelector.removeEnabled = False
        self.importReferenceSelector.noneEnabled = True
        self.importReferenceSelector.showHidden = False
        self.importReferenceSelector.showChildNodeTypes = False
        self.importReferenceSelector.setMRMLScene(slicer.mrmlScene)
        self.importReferenceSelector.enabled = False
        self._tip(
            self.importReferenceSelector,
            "Optional reference volume for segmentation imports. Leave empty to use the AIM file geometry.",
        )
        form.addRow("Reference volume", self.importReferenceSelector)

        self.volumeNameEdit = qt.QLineEdit()
        self._tip(self.volumeNameEdit, "Name assigned to the loaded Slicer volume or segmentation node.")
        form.addRow("Volume name", self.volumeNameEdit)

        self.importButton = qt.QPushButton("Import AIM")
        self.importButton.clicked.connect(self._import_aim)
        self._tip(self.importButton, "Import the selected AIM into the Slicer scene and attach available AIM metadata.")
        form.addRow(self.importButton)

    def _build_export_section(self):
        collapsible = ctk.ctkCollapsibleButton()
        collapsible.text = "Export AIM"
        self.layout.addWidget(collapsible)
        form = qt.QFormLayout(collapsible)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = False
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._on_volume_selected)
        self._tip(self.volumeSelector, "Scalar or labelmap volume to export back to Scanco AIM format.")
        form.addRow("Volume", self.volumeSelector)

        self.exportPathEdit = qt.QLineEdit()
        browse = qt.QPushButton("Browse...")
        browse.clicked.connect(self._browse_export_path)
        self._tip(self.exportPathEdit, "Destination AIM file path for export.")
        self._tip(browse, "Choose where to write the exported AIM file.")
        row = qt.QHBoxLayout()
        row.addWidget(self.exportPathEdit)
        row.addWidget(browse)
        form.addRow("Output AIM", row)

        self.exportModeCombo = qt.QComboBox()
        self.exportModeCombo.addItem("Grayscale image", "grayscale")
        self.exportModeCombo.addItem("Binary mask (0/127)", "mask")
        self._tip(self.exportModeCombo, "Export scalar intensities, or convert nonzero voxels to a binary Scanco mask label.")
        form.addRow("Export as", self.exportModeCombo)

        self.unitCombo = qt.QComboBox()
        self.unitCombo.addItem("Auto from metadata", "auto")
        self.unitCombo.addItem("Native", "native")
        self.unitCombo.addItem("BMD", "BMD")
        self.unitCombo.addItem("HU", "HU")
        self._tip(self.unitCombo, "Unit convention for grayscale export. Auto reuses metadata from imported AIM volumes when available.")
        form.addRow("Grayscale unit", self.unitCombo)

        self.metadataJsonEdit = qt.QLineEdit()
        browse_meta = qt.QPushButton("Browse...")
        browse_meta.clicked.connect(self._browse_metadata_json)
        self._tip(self.metadataJsonEdit, "Optional imported-stack metadata JSON used when the selected volume has no attached AIM metadata.")
        self._tip(browse_meta, "Select metadata JSON from a prior timelapsed import or edited header export.")
        meta_row = qt.QHBoxLayout()
        meta_row.addWidget(self.metadataJsonEdit)
        meta_row.addWidget(browse_meta)
        form.addRow("Metadata JSON", meta_row)

        load_header = qt.QPushButton("Load header from selected volume")
        load_header.clicked.connect(self._load_header_from_selected_volume)
        self._tip(load_header, "Populate editable processing-log and header fields from metadata attached to the selected Slicer volume.")
        form.addRow(load_header)

        self.processingLogTable = qt.QTableWidget()
        self.processingLogTable.setColumnCount(2)
        self.processingLogTable.setHorizontalHeaderLabels(["Field", "Value"])
        self.processingLogTable.horizontalHeader().setStretchLastSection(True)
        self.processingLogTable.setMinimumHeight(220)
        self._tip(self.processingLogTable, "Editable Scanco processing-log fields used for AIM calibration and unit conversion on export.")
        form.addRow("Processing log", self.processingLogTable)

        log_buttons = qt.QHBoxLayout()
        self.addLogRowButton = qt.QPushButton("Add log field")
        self.addLogRowButton.clicked.connect(self._add_processing_log_row)
        self.removeLogRowButton = qt.QPushButton("Remove selected")
        self.removeLogRowButton.clicked.connect(self._remove_selected_processing_log_rows)
        self._tip(self.addLogRowButton, "Add a processing-log field/value row.")
        self._tip(self.removeLogRowButton, "Remove selected processing-log rows from the export metadata.")
        log_buttons.addWidget(self.addLogRowButton)
        log_buttons.addWidget(self.removeLogRowButton)
        form.addRow(log_buttons)

        self.headerEdit = qt.QTextEdit()
        self.headerEdit.setMinimumHeight(110)
        self.headerEdit.setPlaceholderText(
            "Other AIM header metadata JSON. Geometry is refreshed from the selected volume at export."
        )
        self._tip(self.headerEdit, "Editable JSON for non-processing-log AIM header fields. Geometry is refreshed from the selected volume.")
        form.addRow("Other header", self.headerEdit)

        self.allowMinimalCheck = qt.QCheckBox("Allow export with minimal geometry metadata")
        self.allowMinimalCheck.checked = False
        self._tip(self.allowMinimalCheck, "Allow export when no original AIM/header metadata is available, using geometry from the selected volume.")
        form.addRow(self.allowMinimalCheck)

        self.exportButton = qt.QPushButton("Export AIM")
        self.exportButton.clicked.connect(self._export_aim)
        self._tip(self.exportButton, "Write the selected Slicer volume or mask to a Scanco AIM file.")
        form.addRow(self.exportButton)

    def _build_log_section(self):
        self.messageLabel = qt.QLabel()
        self.messageLabel.wordWrap = True
        self.layout.addWidget(self.messageLabel)

    def _browse_import_path(self):
        path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select AIM file",
            "",
            "AIM files (*.AIM *.aim);;All files (*)",
        )
        if isinstance(path, (tuple, list)):
            path = path[0] if path else ""
        if path:
            self.importPathEdit.text = path
            self._update_volume_name_from_import_path(path)

    def _browse_export_path(self):
        path = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save AIM file",
            "",
            "AIM files (*.AIM *.aim);;All files (*)",
        )
        if isinstance(path, (tuple, list)):
            path = path[0] if path else ""
        if path:
            if not str(path).lower().endswith(".aim"):
                path = f"{path}.AIM"
            self.exportPathEdit.text = path

    def _browse_metadata_json(self):
        path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select metadata JSON",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if isinstance(path, (tuple, list)):
            path = path[0] if path else ""
        if path:
            self.metadataJsonEdit.text = path

    def _node_header_metadata(self, node):
        if node is None:
            return None
        metadata_text = node.GetAttribute(AIM_METADATA_ATTRIBUTE)
        if not metadata_text:
            return None
        return json.loads(metadata_text)

    def _set_header_metadata(self, metadata):
        metadata = dict(metadata or {})
        processing_log = _normalize_processing_log(metadata)
        metadata.pop("processing_log_raw", None)
        metadata.pop("processing_log", None)
        metadata.pop("processing_log_dict", None)
        self._set_processing_log_table(processing_log)
        self.headerEdit.setPlainText(_metadata_json(metadata))

    def _set_processing_log_table(self, log_dict):
        self.processingLogTable.setRowCount(0)
        for key, value in (log_dict or {}).items():
            row = self.processingLogTable.rowCount
            self.processingLogTable.insertRow(row)
            self.processingLogTable.setItem(row, 0, qt.QTableWidgetItem(str(key)))
            if isinstance(value, (list, tuple)):
                text = ", ".join(str(v) for v in value)
            else:
                text = str(value)
            self.processingLogTable.setItem(row, 1, qt.QTableWidgetItem(text))

    def _processing_log_from_table(self):
        out = {}
        for row in range(self.processingLogTable.rowCount):
            key_item = self.processingLogTable.item(row, 0)
            value_item = self.processingLogTable.item(row, 1)
            key = key_item.text().strip() if key_item else ""
            if not key:
                continue
            value = value_item.text().strip() if value_item else ""
            out[key] = _parse_table_value(value)
        return out

    def _add_processing_log_row(self):
        row = self.processingLogTable.rowCount
        self.processingLogTable.insertRow(row)
        self.processingLogTable.setItem(row, 0, qt.QTableWidgetItem(""))
        self.processingLogTable.setItem(row, 1, qt.QTableWidgetItem(""))
        self.processingLogTable.setCurrentCell(row, 0)

    def _remove_selected_processing_log_rows(self):
        rows = sorted({index.row() for index in self.processingLogTable.selectedIndexes()}, reverse=True)
        for row in rows:
            self.processingLogTable.removeRow(row)

    def _load_header_from_selected_volume(self):
        metadata = self._node_header_metadata(self.volumeSelector.currentNode())
        if metadata is None:
            self._log("Selected volume has no stored AIM header metadata.")
            return
        self._set_header_metadata(metadata)
        self._log("Loaded AIM header metadata from selected volume.")

    def _on_volume_selected(self, node):
        try:
            if (
                node is not None
                and hasattr(node, "IsA")
                and node.IsA("vtkMRMLLabelMapVolumeNode")
                and hasattr(self, "exportModeCombo")
            ):
                index = self.exportModeCombo.findData("mask")
                if index >= 0:
                    self.exportModeCombo.setCurrentIndex(index)
            metadata = self._node_header_metadata(node)
            if metadata is not None:
                self._set_header_metadata(metadata)
        except Exception as exc:
            self._log(f"Could not load AIM header metadata: {exc}")

    def _edited_header_metadata(self):
        text = self.headerEdit.toPlainText().strip()
        if not text:
            return None
        metadata = json.loads(text)
        if not isinstance(metadata, dict):
            raise ValueError("AIM header JSON must be an object/dictionary.")
        processing_log = self._processing_log_from_table()
        if processing_log:
            metadata["processing_log"] = processing_log
        return metadata

    def _install_core(self):
        try:
            self._log("Installing aimio-py...")
            self.logic.install_or_update_core()
            self._log("AIM I/O dependency is installed.")
        except Exception as exc:
            self._error(exc)

    def _check_toolbox_updates(self):
        run_toolbox_update_dialog(__file__, log=self._log)

    def _on_import_path_changed(self, path):
        self._update_volume_name_from_import_path(path)

    def _on_import_as_changed(self, _index=None):
        as_segmentation = self.importAsCombo.currentData == "segmentation"
        self.scalingCombo.enabled = not as_segmentation
        self.importReferenceSelector.enabled = as_segmentation

    def _update_volume_name_from_import_path(self, path):
        path = str(path or "").strip()
        if not path:
            return
        suggested = Path(path).stem
        current = str(self.volumeNameEdit.text or "").strip()
        if not current or current == self._lastAutoVolumeName:
            self.volumeNameEdit.text = suggested
            self._lastAutoVolumeName = suggested

    def _import_aim(self):
        try:
            as_segmentation = self.importAsCombo.currentData == "segmentation"
            scaling = "native" if as_segmentation else self.scalingCombo.currentData
            node = self.logic.import_aim(
                self.importPathEdit.text,
                scaling=scaling,
                volume_name=self.volumeNameEdit.text,
                as_segmentation=as_segmentation,
                reference_volume_node=self.importReferenceSelector.currentNode() if as_segmentation else None,
            )
            if not as_segmentation:
                self.volumeSelector.setCurrentNode(node)
                self.importReferenceSelector.setCurrentNode(node)
            self._set_header_metadata(self._node_header_metadata(node))
            node_kind = "segmentation" if as_segmentation else "volume"
            self._log(f"Imported {node.GetName()} as {node_kind} from {self.importPathEdit.text}")
        except Exception as exc:
            self._error(exc)

    def _export_aim(self):
        try:
            mode = self.exportModeCombo.currentData
            output = self.logic.export_aim(
                self.volumeSelector.currentNode(),
                self.exportPathEdit.text,
                as_mask=(mode == "mask"),
                unit=self.unitCombo.currentData,
                metadata_json=self.metadataJsonEdit.text.strip() or None,
                header_metadata=self._edited_header_metadata(),
                allow_minimal_metadata=bool(self.allowMinimalCheck.checked),
            )
            self._log(f"Wrote {output}")
        except Exception as exc:
            self._error(exc)

    def _log(self, text):
        self.messageLabel.setText(text)

    def _error(self, exc):
        self.messageLabel.setText(f"<b>Error:</b> {exc}")
        slicer.util.errorDisplay(str(exc))


class ScancoIOTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("ScancoIO smoke test passed.")
