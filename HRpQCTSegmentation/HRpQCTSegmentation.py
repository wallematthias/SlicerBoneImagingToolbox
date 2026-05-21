import numpy as np
import qt
import ctk
import slicer

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


MODULE_VERSION = "0.1.0"


class HRpQCTSegmentation(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Contours and Segmentation"
        parent.categories = ["HR-pQCT"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Create simple HR-pQCT threshold segmentations and continue manual "
            f"cleanup in Slicer's Segment Editor. Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = "Part of the HR-pQCT Toolbox for 3D Slicer."


class HRpQCTSegmentationLogic(ScriptedLoadableModuleLogic):
    def create_threshold_segmentation(
        self,
        volume_node,
        *,
        lower_threshold,
        upper_threshold,
        segmentation_name=None,
        segment_name="Bone",
    ):
        if volume_node is None:
            raise ValueError("Select an input volume.")
        lower = float(lower_threshold)
        upper = float(upper_threshold)
        if upper <= lower:
            raise ValueError("Upper threshold must be greater than lower threshold.")

        image_arr = slicer.util.arrayFromVolume(volume_node)
        mask_arr = ((image_arr >= lower) & (image_arr <= upper)).astype(np.uint8)

        label_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{volume_node.GetName()}_{segment_name}_label",
        )
        slicer.util.updateVolumeFromArray(label_node, mask_arr)
        label_node.CopyOrientation(volume_node)

        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            segmentation_name or f"{volume_node.GetName()} segmentation",
        )
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            label_node,
            segmentation_node,
        )
        segmentation = segmentation_node.GetSegmentation()
        if segmentation.GetNumberOfSegments() > 0:
            segment_id = segmentation.GetNthSegmentID(0)
            segmentation.GetSegment(segment_id).SetName(segment_name)
        slicer.mrmlScene.RemoveNode(label_node)
        return segmentation_node


class HRpQCTSegmentationWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = HRpQCTSegmentationLogic()
        self._build_threshold_section()
        self._build_log_section()
        self.layout.addStretch(1)
        self._log("Ready.")

    def _build_threshold_section(self):
        collapsible = ctk.ctkCollapsibleButton()
        collapsible.text = "Threshold Segmentation"
        self.layout.addWidget(collapsible)
        form = qt.QFormLayout(collapsible)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = False
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Input volume", self.volumeSelector)

        self.lowerSpin = qt.QDoubleSpinBox()
        self.lowerSpin.minimum = -100000.0
        self.lowerSpin.maximum = 100000.0
        self.lowerSpin.decimals = 3
        self.lowerSpin.value = 225.0
        form.addRow("Lower threshold", self.lowerSpin)

        self.upperSpin = qt.QDoubleSpinBox()
        self.upperSpin.minimum = -100000.0
        self.upperSpin.maximum = 100000.0
        self.upperSpin.decimals = 3
        self.upperSpin.value = 100000.0
        form.addRow("Upper threshold", self.upperSpin)

        self.segmentNameEdit = qt.QLineEdit()
        self.segmentNameEdit.text = "Bone"
        form.addRow("Segment name", self.segmentNameEdit)

        self.segmentationNameEdit = qt.QLineEdit()
        form.addRow("Segmentation name", self.segmentationNameEdit)

        self.createButton = qt.QPushButton("Create Segmentation")
        self.createButton.clicked.connect(self._create_segmentation)
        form.addRow(self.createButton)

        self.openEditorButton = qt.QPushButton("Open Segment Editor")
        self.openEditorButton.clicked.connect(self._open_segment_editor)
        form.addRow(self.openEditorButton)

    def _build_log_section(self):
        self.messageLabel = qt.QLabel()
        self.messageLabel.wordWrap = True
        self.layout.addWidget(self.messageLabel)

    def _create_segmentation(self):
        try:
            segmentation_node = self.logic.create_threshold_segmentation(
                self.volumeSelector.currentNode(),
                lower_threshold=self.lowerSpin.value,
                upper_threshold=self.upperSpin.value,
                segmentation_name=self.segmentationNameEdit.text.strip() or None,
                segment_name=self.segmentNameEdit.text.strip() or "Bone",
            )
            self._log(f"Created {segmentation_node.GetName()}.")
            self._open_segment_editor()
        except Exception as exc:
            self._error(exc)

    def _open_segment_editor(self):
        slicer.util.selectModule("SegmentEditor")

    def _log(self, text):
        self.messageLabel.setText(text)

    def _error(self, exc):
        self.messageLabel.setText(f"<b>Error:</b> {exc}")
        slicer.util.errorDisplay(str(exc))


class HRpQCTSegmentationTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("HRpQCTSegmentation smoke test passed.")
