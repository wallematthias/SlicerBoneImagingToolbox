import tempfile
from pathlib import Path

import ctk
import qt
import slicer
import SimpleITK as sitk

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


MODULE_VERSION = "0.2.0"

SITE_PRESETS = {
    "radius": {
        "inner": {
            "site": "radius",
            "endosteal_threshold": 500.0,
            "endosteal_kernelsize": 3,
            "gaussian_sigma": 1.5,
            "peel": 3,
            "trabecular_close_radius": 15,
            "morphology_downsample_factor": 1,
            "use_adaptive_threshold": False,
        },
        "outer": {
            "periosteal_threshold": 300.0,
            "periosteal_kernelsize": 5,
            "periosteal_open_radius": 2,
            "gaussian_sigma": 1.5,
            "morphology_downsample_factor": 1,
            "use_adaptive_threshold": True,
        },
    },
    "tibia": {
        "inner": {
            "site": "tibia",
            "endosteal_threshold": 500.0,
            "endosteal_kernelsize": 3,
            "gaussian_sigma": 1.5,
            "peel": 3,
            "trabecular_close_radius": 25,
            "morphology_downsample_factor": 1,
            "use_adaptive_threshold": False,
        },
        "outer": {
            "periosteal_threshold": 300.0,
            "periosteal_kernelsize": 5,
            "periosteal_open_radius": 2,
            "gaussian_sigma": 1.5,
            "morphology_downsample_factor": 1,
            "use_adaptive_threshold": False,
        },
    },
    "knee": {
        "inner": {
            "site": "knee",
            "endosteal_threshold": 250.0,
            "endosteal_kernelsize": 3,
            "gaussian_sigma": 2.0,
            "peel": 4,
            "trabecular_close_radius": 36,
            "morphology_downsample_factor": 3,
            "use_adaptive_threshold": False,
        },
        "outer": {
            "periosteal_threshold": 150.0,
            "periosteal_kernelsize": 16,
            "periosteal_open_radius": 8,
            "gaussian_sigma": 1.5,
            "morphology_downsample_factor": 3,
            "use_adaptive_threshold": False,
        },
    },
}

METHOD_PRESETS = {
    "seg_gauss": {
        "gaussian_sigma": 0.8,
        "trab_threshold": 320.0,
        "cort_threshold": 450.0,
        "adaptive_low_threshold": 100.0,
        "adaptive_high_threshold": 300.0,
        "adaptive_block_size": 13,
        "min_size_voxels": 64,
        "keep_largest_component": True,
        "laplace_hamming_threshold": 15564.0,
        "laplace_hamming_backend": "cpu",
    },
    "adaptive": {
        "gaussian_sigma": 0.8,
        "trab_threshold": 320.0,
        "cort_threshold": 450.0,
        "adaptive_low_threshold": 100.0,
        "adaptive_high_threshold": 300.0,
        "adaptive_block_size": 13,
        "min_size_voxels": 64,
        "keep_largest_component": True,
        "laplace_hamming_threshold": 15564.0,
        "laplace_hamming_backend": "cpu",
    },
    "laplace_hamming": {
        "gaussian_sigma": 0.8,
        "trab_threshold": 320.0,
        "cort_threshold": 450.0,
        "adaptive_low_threshold": 100.0,
        "adaptive_high_threshold": 300.0,
        "adaptive_block_size": 13,
        "min_size_voxels": 70,
        "keep_largest_component": False,
        "laplace_hamming_threshold": 15564.0,
        "laplace_hamming_backend": "cpu",
    },
}


class HRpQCTSegmentation(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Contours and Segmentation"
        parent.categories = ["HR-pQCT"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Generate HR-pQCT full, trabecular, cortical, and binary segmentation "
            f"masks using site presets and standard segmentation methods. Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = "Part of the HR-pQCT Toolbox for 3D Slicer."


class HRpQCTSegmentationLogic(ScriptedLoadableModuleLogic):
    def is_pipeline_available(self):
        try:
            import timelapsedhrpqct  # noqa: F401

            return True
        except Exception:
            return False

    def install_or_update_pipeline(self):
        slicer.util.pip_install("--upgrade --force-reinstall --no-cache-dir timelapsed-hrpqct")

    def _volume_to_sitk(self, volume_node):
        if volume_node is None:
            raise ValueError("Select an input volume.")
        with tempfile.TemporaryDirectory(prefix="hrpqct_seg_in_") as temp_dir:
            path = Path(temp_dir) / "input.nrrd"
            if not slicer.util.saveNode(volume_node, str(path)):
                raise RuntimeError("Could not save selected Slicer volume for processing.")
            return sitk.ReadImage(str(path))

    def _sitk_to_labelmap(self, image, name, reference_node):
        with tempfile.TemporaryDirectory(prefix="hrpqct_seg_out_") as temp_dir:
            path = Path(temp_dir) / f"{name}.nrrd"
            sitk.WriteImage(sitk.Cast(image > 0, sitk.sitkUInt8), str(path))
            loaded = slicer.util.loadLabelVolume(str(path), {"name": name}, returnNode=True)
        if isinstance(loaded, tuple):
            success, label_node = loaded
        else:
            success, label_node = bool(loaded), loaded
        if not success or label_node is None:
            raise RuntimeError(f"Could not load generated labelmap: {name}")
        label_node.CopyOrientation(reference_node)
        return label_node

    def _add_labelmap_segment(self, label_node, segmentation_node, segment_name):
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            label_node,
            segmentation_node,
        )
        segmentation = segmentation_node.GetSegmentation()
        if segmentation.GetNumberOfSegments() > 0:
            segment_id = segmentation.GetNthSegmentID(segmentation.GetNumberOfSegments() - 1)
            segmentation.GetSegment(segment_id).SetName(segment_name)

    def generate_hrpqct_masks(
        self,
        volume_node,
        *,
        site,
        method,
        output_prefix=None,
        create_labelmaps=True,
        open_segment_editor=False,
        params=None,
    ):
        from timelapsedhrpqct.processing.contour_generation import (
            ContourGenerationParams,
            InnerContourParams,
            OuterContourParams,
            SegmentationParams,
            generate_masks_from_image,
        )

        image = self._volume_to_sitk(volume_node)
        params = dict(params or {})
        site_defaults = SITE_PRESETS[str(site)]
        method_defaults = METHOD_PRESETS[str(method)]

        inner_params = dict(site_defaults["inner"])
        outer_params = dict(site_defaults["outer"])
        segmentation_params = dict(method_defaults)
        segmentation_params.update(params.get("segmentation", {}))
        segmentation_params["method"] = str(method)

        inner_params.update(params.get("inner", {}))
        outer_params.update(params.get("outer", {}))
        inner_params["site"] = str(site)

        generated = generate_masks_from_image(
            image,
            ContourGenerationParams(
                outer=OuterContourParams(**outer_params),
                inner=InnerContourParams(**inner_params),
                segmentation=SegmentationParams(**segmentation_params),
            ),
            verbose=False,
        )

        prefix = output_prefix.strip() if output_prefix else volume_node.GetName()
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            f"{prefix}_HRpQCT_segmentation",
        )
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

        outputs = {}
        for role, image_out, segment_name in [
            ("full", generated.full, "Full mask"),
            ("trab", generated.trab, "Trabecular mask"),
            ("cort", generated.cort, "Cortical mask"),
            ("seg", generated.seg, "Bone segmentation"),
        ]:
            label_node = self._sitk_to_labelmap(image_out, f"{prefix}_{role}", volume_node)
            self._add_labelmap_segment(label_node, segmentation_node, segment_name)
            if create_labelmaps:
                outputs[role] = label_node
            else:
                slicer.mrmlScene.RemoveNode(label_node)

        if open_segment_editor:
            slicer.util.selectModule("SegmentEditor")

        return segmentation_node, outputs, generated.metadata


class HRpQCTSegmentationWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = HRpQCTSegmentationLogic()
        self._build_segmentation_section()
        self._build_log_section()
        self.layout.addStretch(1)
        self._apply_site_preset()
        self._apply_method_preset()
        self._update_dependency_ui()
        self._log("Ready.")

    def _build_segmentation_section(self):
        collapsible = ctk.ctkCollapsibleButton()
        collapsible.text = "HR-pQCT Segmentation"
        self.layout.addWidget(collapsible)
        form = qt.QFormLayout(collapsible)

        self.pipelineStatusLabel = qt.QLabel()
        self.installButton = qt.QPushButton("Install / Update timelapsed-hrpqct")
        self.installButton.clicked.connect(self._install_pipeline)
        form.addRow("Status", self.pipelineStatusLabel)
        form.addRow(self.installButton)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = False
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Input volume", self.volumeSelector)

        self.siteCombo = qt.QComboBox()
        for label, value in [("Radius", "radius"), ("Tibia", "tibia"), ("Knee", "knee")]:
            self.siteCombo.addItem(label, value)
        self.siteCombo.currentIndexChanged.connect(self._apply_site_preset)
        form.addRow("Site preset", self.siteCombo)

        self.methodCombo = qt.QComboBox()
        for label, value in [
            ("Standard Gaussian (trab 320 / cort 450)", "seg_gauss"),
            ("Laplace-Hamming", "laplace_hamming"),
            ("Adaptive threshold", "adaptive"),
        ]:
            self.methodCombo.addItem(label, value)
        self.methodCombo.currentIndexChanged.connect(self._apply_method_preset)
        form.addRow("Segmentation method", self.methodCombo)

        self.outputPrefixEdit = qt.QLineEdit()
        form.addRow("Output prefix", self.outputPrefixEdit)

        self.createLabelmapsCheck = qt.QCheckBox()
        self.createLabelmapsCheck.checked = True
        form.addRow("Create labelmaps", self.createLabelmapsCheck)

        self.openEditorCheck = qt.QCheckBox()
        self.openEditorCheck.checked = False
        form.addRow("Open Segment Editor", self.openEditorCheck)

        expert = ctk.ctkCollapsibleButton()
        expert.text = "Expert Settings"
        expert.collapsed = True
        self.layout.addWidget(expert)
        expert_form = qt.QFormLayout(expert)

        self.trabThresholdSpin = self._double_spin(0, 5000, 1, 320.0)
        self.cortThresholdSpin = self._double_spin(0, 5000, 1, 450.0)
        self.gaussSigmaSpin = self._double_spin(0, 10, 2, 0.8)
        expert_form.addRow("Trab threshold", self.trabThresholdSpin)
        expert_form.addRow("Cort threshold", self.cortThresholdSpin)
        expert_form.addRow("Gaussian sigma", self.gaussSigmaSpin)

        self.adaptiveLowSpin = self._double_spin(-1000, 5000, 1, 100.0)
        self.adaptiveHighSpin = self._double_spin(-1000, 5000, 1, 300.0)
        self.adaptiveBlockSpin = qt.QSpinBox()
        self.adaptiveBlockSpin.minimum = 3
        self.adaptiveBlockSpin.maximum = 101
        self.adaptiveBlockSpin.singleStep = 2
        self.adaptiveBlockSpin.value = 13
        expert_form.addRow("Adaptive low", self.adaptiveLowSpin)
        expert_form.addRow("Adaptive high", self.adaptiveHighSpin)
        expert_form.addRow("Adaptive block size", self.adaptiveBlockSpin)

        self.lhThresholdSpin = self._double_spin(0, 100000, 1, 15564.0)
        self.lhBackendCombo = qt.QComboBox()
        for label, value in [("CPU", "cpu"), ("Auto", "auto"), ("Torch MPS", "torch_mps")]:
            self.lhBackendCombo.addItem(label, value)
        expert_form.addRow("LH threshold", self.lhThresholdSpin)
        expert_form.addRow("LH backend", self.lhBackendCombo)

        self.minSizeSpin = qt.QSpinBox()
        self.minSizeSpin.minimum = 0
        self.minSizeSpin.maximum = 1000000
        self.minSizeSpin.value = 64
        self.keepLargestCheck = qt.QCheckBox()
        self.keepLargestCheck.checked = True
        expert_form.addRow("Min component voxels", self.minSizeSpin)
        expert_form.addRow("Keep largest", self.keepLargestCheck)

        self.outerThresholdSpin = self._double_spin(-1000, 5000, 1, 300.0)
        self.innerThresholdSpin = self._double_spin(-1000, 5000, 1, 500.0)
        self.trabCloseSpin = qt.QSpinBox()
        self.trabCloseSpin.minimum = 0
        self.trabCloseSpin.maximum = 200
        self.trabCloseSpin.value = 25
        self.outerKernelSpin = self._kernel_spin(5)
        self.innerKernelSpin = self._kernel_spin(3)
        self.outerOpenSpin = qt.QSpinBox()
        self.outerOpenSpin.minimum = 0
        self.outerOpenSpin.maximum = 100
        self.outerOpenSpin.value = 2
        self.peelSpin = qt.QSpinBox()
        self.peelSpin.minimum = 0
        self.peelSpin.maximum = 50
        self.peelSpin.value = 3
        expert_form.addRow("Periosteal threshold", self.outerThresholdSpin)
        expert_form.addRow("Endosteal threshold", self.innerThresholdSpin)
        expert_form.addRow("Trab close radius", self.trabCloseSpin)
        expert_form.addRow("Periosteal kernel", self.outerKernelSpin)
        expert_form.addRow("Endosteal kernel", self.innerKernelSpin)
        expert_form.addRow("Periosteal open radius", self.outerOpenSpin)
        expert_form.addRow("Peel", self.peelSpin)

        self.createButton = qt.QPushButton("Generate Masks And Segmentation")
        self.createButton.clicked.connect(self._create_segmentation)
        form.addRow(self.createButton)

        self.openEditorButton = qt.QPushButton("Open Segment Editor")
        self.openEditorButton.clicked.connect(self._open_segment_editor)
        form.addRow(self.openEditorButton)

    def _double_spin(self, minimum, maximum, decimals, value):
        spin = qt.QDoubleSpinBox()
        spin.minimum = float(minimum)
        spin.maximum = float(maximum)
        spin.decimals = int(decimals)
        spin.value = float(value)
        return spin

    def _kernel_spin(self, value):
        spin = qt.QSpinBox()
        spin.minimum = 1
        spin.maximum = 101
        spin.singleStep = 2
        spin.value = int(value)
        return spin

    def _build_log_section(self):
        self.messageLabel = qt.QLabel()
        self.messageLabel.wordWrap = True
        self.layout.addWidget(self.messageLabel)

    def _apply_site_preset(self):
        if not hasattr(self, "siteCombo"):
            return
        preset = SITE_PRESETS[self.siteCombo.currentData]
        inner = preset["inner"]
        outer = preset["outer"]
        self.outerThresholdSpin.value = float(outer["periosteal_threshold"])
        self.innerThresholdSpin.value = float(inner["endosteal_threshold"])
        self.trabCloseSpin.value = int(inner["trabecular_close_radius"])
        self.outerKernelSpin.value = int(outer["periosteal_kernelsize"])
        self.innerKernelSpin.value = int(inner["endosteal_kernelsize"])
        self.outerOpenSpin.value = int(outer["periosteal_open_radius"])
        self.peelSpin.value = int(inner["peel"])

    def _apply_method_preset(self):
        if not hasattr(self, "methodCombo"):
            return
        preset = METHOD_PRESETS[self.methodCombo.currentData]
        self.gaussSigmaSpin.value = float(preset["gaussian_sigma"])
        self.trabThresholdSpin.value = float(preset["trab_threshold"])
        self.cortThresholdSpin.value = float(preset["cort_threshold"])
        self.adaptiveLowSpin.value = float(preset["adaptive_low_threshold"])
        self.adaptiveHighSpin.value = float(preset["adaptive_high_threshold"])
        self.adaptiveBlockSpin.value = int(preset["adaptive_block_size"])
        self.minSizeSpin.value = int(preset["min_size_voxels"])
        self.keepLargestCheck.checked = bool(preset["keep_largest_component"])
        self.lhThresholdSpin.value = float(preset["laplace_hamming_threshold"])
        backend = str(preset["laplace_hamming_backend"])
        count = self.lhBackendCombo.count
        if callable(count):
            count = count()
        for i in range(int(count)):
            if self.lhBackendCombo.itemData(i) == backend:
                self.lhBackendCombo.setCurrentIndex(i)
                break

    def _collect_params(self):
        block_size = int(self.adaptiveBlockSpin.value)
        if block_size % 2 == 0:
            block_size += 1
        return {
            "segmentation": {
                "gaussian_sigma": float(self.gaussSigmaSpin.value),
                "trab_threshold": float(self.trabThresholdSpin.value),
                "cort_threshold": float(self.cortThresholdSpin.value),
                "adaptive_low_threshold": float(self.adaptiveLowSpin.value),
                "adaptive_high_threshold": float(self.adaptiveHighSpin.value),
                "adaptive_block_size": block_size,
                "min_size_voxels": int(self.minSizeSpin.value),
                "keep_largest_component": bool(self.keepLargestCheck.checked),
                "laplace_hamming_threshold": float(self.lhThresholdSpin.value),
                "laplace_hamming_backend": str(self.lhBackendCombo.currentData),
            },
            "outer": {
                "periosteal_threshold": float(self.outerThresholdSpin.value),
                "periosteal_kernelsize": int(self.outerKernelSpin.value),
                "periosteal_open_radius": int(self.outerOpenSpin.value),
            },
            "inner": {
                "endosteal_threshold": float(self.innerThresholdSpin.value),
                "endosteal_kernelsize": int(self.innerKernelSpin.value),
                "peel": int(self.peelSpin.value),
                "trabecular_close_radius": int(self.trabCloseSpin.value),
            },
        }

    def _create_segmentation(self):
        try:
            if not self.logic.is_pipeline_available():
                raise RuntimeError("Install or update timelapsed-hrpqct first.")
            segmentation_node, labelmaps, metadata = self.logic.generate_hrpqct_masks(
                self.volumeSelector.currentNode(),
                site=str(self.siteCombo.currentData),
                method=str(self.methodCombo.currentData),
                output_prefix=self.outputPrefixEdit.text.strip() or None,
                create_labelmaps=bool(self.createLabelmapsCheck.checked),
                open_segment_editor=bool(self.openEditorCheck.checked),
                params=self._collect_params(),
            )
            counts = metadata.get("voxel_counts", {})
            label_text = f" Created {len(labelmaps)} labelmaps." if labelmaps else ""
            self._log(
                f"Created {segmentation_node.GetName()}.{label_text} "
                f"Voxel counts: full={counts.get('full')}, trab={counts.get('trab')}, "
                f"cort={counts.get('cort')}, seg={counts.get('seg')}."
            )
        except Exception as exc:
            self._error(exc)

    def _install_pipeline(self):
        try:
            self._log("Installing timelapsed-hrpqct...")
            self.logic.install_or_update_pipeline()
            self._update_dependency_ui()
            self._log("timelapsed-hrpqct is installed.")
        except Exception as exc:
            self._error(exc)

    def _update_dependency_ui(self):
        if self.logic.is_pipeline_available():
            self.pipelineStatusLabel.text = "Installed"
            self.pipelineStatusLabel.styleSheet = "color: #228b22;"
        else:
            self.pipelineStatusLabel.text = "Not installed"
            self.pipelineStatusLabel.styleSheet = "color: #cc5500;"

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
