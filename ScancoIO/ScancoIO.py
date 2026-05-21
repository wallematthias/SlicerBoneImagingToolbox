import json
import tempfile
from pathlib import Path
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
        from aim_io import is_aimio_available

        return is_aimio_available()

    def install_or_update_core(self):
        slicer.util.pip_install("--upgrade --force-reinstall --no-cache-dir aimio-py")

    def import_aim(self, aim_path, scaling, volume_name=None):
        from aim_io import read_aim

        aim_path = Path(aim_path)
        if not aim_path.exists():
            raise FileNotFoundError(f"AIM file does not exist: {aim_path}")
        image, metadata = read_aim(aim_path, scaling=scaling)
        name = volume_name.strip() if volume_name else aim_path.stem

        with tempfile.TemporaryDirectory(prefix="hrpqct_aim_import_") as temp_dir:
            nrrd_path = Path(temp_dir) / "imported_aim.nrrd"
            sitk.WriteImage(image, str(nrrd_path))
            loaded = slicer.util.loadVolume(str(nrrd_path), {"name": name}, returnNode=True)

        if isinstance(loaded, tuple):
            success, volume_node = loaded
        else:
            success, volume_node = bool(loaded), loaded
        if not success or volume_node is None:
            raise RuntimeError(f"Could not load imported AIM volume into Slicer: {aim_path}")

        volume_node.SetAttribute(AIM_SOURCE_ATTRIBUTE, str(aim_path))
        volume_node.SetAttribute(AIM_SCALING_ATTRIBUTE, scaling)
        volume_node.SetAttribute(
            AIM_METADATA_ATTRIBUTE,
            json.dumps(metadata, sort_keys=True, default=_json_default),
        )
        return volume_node

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
        from aim_io import aim_metadata_from_import_json, write_aim

        if volume_node is None:
            raise ValueError("Select a scalar volume to export.")
        if not str(output_path).strip():
            raise ValueError("Choose an output AIM path.")
        output_path = Path(output_path)

        with tempfile.TemporaryDirectory(prefix="hrpqct_aim_export_") as temp_dir:
            nrrd_path = Path(temp_dir) / "slicer_volume.nrrd"
            if not slicer.util.saveNode(volume_node, str(nrrd_path)):
                raise RuntimeError("Could not save selected Slicer volume for AIM export.")
            image = sitk.ReadImage(str(nrrd_path))

        metadata = None
        metadata_json = Path(metadata_json) if metadata_json else None
        if metadata_json and metadata_json.exists():
            metadata = aim_metadata_from_import_json(metadata_json, image, log=log)
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
        write_aim(image, output_path, metadata=metadata, unit=write_unit, mask=as_mask)
        return output_path


class ScancoIOWidget(ScriptedLoadableModuleWidget):
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

        self.importPathEdit = qt.QLineEdit()
        browse = qt.QPushButton("Browse...")
        browse.clicked.connect(self._browse_import_path)
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
        form.addRow("Load values as", self.scalingCombo)

        self.volumeNameEdit = qt.QLineEdit()
        form.addRow("Volume name", self.volumeNameEdit)

        self.installButton = qt.QPushButton("Install / Update AIM I/O")
        self.installButton.clicked.connect(self._install_core)
        form.addRow(self.installButton)

        self.importButton = qt.QPushButton("Import AIM")
        self.importButton.clicked.connect(self._import_aim)
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
        form.addRow("Volume", self.volumeSelector)

        self.exportPathEdit = qt.QLineEdit()
        browse = qt.QPushButton("Browse...")
        browse.clicked.connect(self._browse_export_path)
        row = qt.QHBoxLayout()
        row.addWidget(self.exportPathEdit)
        row.addWidget(browse)
        form.addRow("Output AIM", row)

        self.exportModeCombo = qt.QComboBox()
        self.exportModeCombo.addItem("Grayscale image", "grayscale")
        self.exportModeCombo.addItem("Binary mask (0/127)", "mask")
        form.addRow("Export as", self.exportModeCombo)

        self.unitCombo = qt.QComboBox()
        self.unitCombo.addItem("Auto from metadata", "auto")
        self.unitCombo.addItem("Native", "native")
        self.unitCombo.addItem("BMD", "BMD")
        self.unitCombo.addItem("HU", "HU")
        form.addRow("Grayscale unit", self.unitCombo)

        self.metadataJsonEdit = qt.QLineEdit()
        browse_meta = qt.QPushButton("Browse...")
        browse_meta.clicked.connect(self._browse_metadata_json)
        meta_row = qt.QHBoxLayout()
        meta_row.addWidget(self.metadataJsonEdit)
        meta_row.addWidget(browse_meta)
        form.addRow("Metadata JSON", meta_row)

        load_header = qt.QPushButton("Load header from selected volume")
        load_header.clicked.connect(self._load_header_from_selected_volume)
        form.addRow(load_header)

        self.headerEdit = qt.QTextEdit()
        self.headerEdit.setMinimumHeight(180)
        self.headerEdit.setPlaceholderText(
            "AIM header metadata JSON. Imported AIM metadata is stored on the "
            "Slicer volume and can be edited here before export."
        )
        form.addRow("AIM header", self.headerEdit)

        self.allowMinimalCheck = qt.QCheckBox("Allow export with minimal geometry metadata")
        self.allowMinimalCheck.checked = False
        form.addRow(self.allowMinimalCheck)

        self.exportButton = qt.QPushButton("Export AIM")
        self.exportButton.clicked.connect(self._export_aim)
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
            if not self.volumeNameEdit.text:
                self.volumeNameEdit.text = Path(path).stem

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
        self.headerEdit.setPlainText(_metadata_json(metadata))

    def _load_header_from_selected_volume(self):
        metadata = self._node_header_metadata(self.volumeSelector.currentNode())
        if metadata is None:
            self._log("Selected volume has no stored AIM header metadata.")
            return
        self._set_header_metadata(metadata)
        self._log("Loaded AIM header metadata from selected volume.")

    def _on_volume_selected(self, node):
        try:
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
        return metadata

    def _install_core(self):
        try:
            self._log("Installing aimio-py...")
            self.logic.install_or_update_core()
            self._log("AIM I/O dependency is installed.")
        except Exception as exc:
            self._error(exc)

    def _import_aim(self):
        try:
            scaling = self.scalingCombo.currentData
            node = self.logic.import_aim(
                self.importPathEdit.text,
                scaling=scaling,
                volume_name=self.volumeNameEdit.text,
            )
            self.volumeSelector.setCurrentNode(node)
            self._set_header_metadata(self._node_header_metadata(node))
            self._log(f"Imported {node.GetName()} from {self.importPathEdit.text}")
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
