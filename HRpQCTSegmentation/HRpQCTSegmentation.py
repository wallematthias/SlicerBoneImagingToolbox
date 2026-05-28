import tempfile
import sys
import importlib
import inspect
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import ctk
import numpy as np
import qt
import slicer
import SimpleITK as sitk

_TOOLBOX_ROOT = Path(__file__).resolve().parent.parent
if str(_TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLBOX_ROOT))

_GEODESIC_CONTOUR_LOCAL_REPO = _TOOLBOX_ROOT.parent / "hrpqct-geodesic-contour"
_GEODESIC_CONTOUR_LOCAL_SRC = _GEODESIC_CONTOUR_LOCAL_REPO / "src"
if _GEODESIC_CONTOUR_LOCAL_SRC.exists() and str(_GEODESIC_CONTOUR_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(_GEODESIC_CONTOUR_LOCAL_SRC))

from SlicerTimelapsedHRpQCTLib.slicer_update_ui import run_toolbox_update_dialog

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


MODULE_VERSION = "0.2.0"
AIM_SOURCE_ATTRIBUTE = "HRpQCT.AIMSourcePath"

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

SEGMENTATION_METHODS = {"seg_gauss", "adaptive", "laplace_hamming", "none"}
PERIOSTEAL_CONTOUR_METHODS = {"standard", "geodesic_fracture", "none"}
ENDOSTEAL_CONTOUR_METHODS = {"standard", "none"}


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

    def is_geodesic_contour_available(self):
        try:
            import hrpqct_geodesic_contour  # noqa: F401

            return True
        except Exception:
            return False

    def install_or_update_pipeline(self):
        slicer.util.pip_install("--upgrade --force-reinstall --no-cache-dir timelapsed-hrpqct")

    def install_or_update_geodesic_contour(self):
        if _GEODESIC_CONTOUR_LOCAL_REPO.exists():
            slicer.util.pip_install("edt>=2.4")
            slicer.util.pip_install(f"--no-deps -e {_GEODESIC_CONTOUR_LOCAL_REPO}")
        else:
            slicer.util.pip_install("--upgrade --force-reinstall --no-cache-dir hrpqct-geodesic-contour")
        self._reload_and_validate_geodesic_contour()

    def install_or_update_contouring_dependencies(self):
        self.install_or_update_pipeline()
        self.install_or_update_geodesic_contour()

    def _reload_and_validate_geodesic_contour(self):
        importlib.invalidate_caches()
        sys.modules.pop("hrpqct_geodesic_contour.core", None)
        sys.modules.pop("hrpqct_geodesic_contour", None)
        geodesic_contour = importlib.import_module("hrpqct_geodesic_contour")
        contour_parameters = inspect.signature(geodesic_contour.contour).parameters
        required_parameters = {"fill_holes", "progress_callback", "cancel_callback"}
        missing_parameters = sorted(required_parameters.difference(contour_parameters))
        if missing_parameters:
            raise RuntimeError(
                "Installed hrpqct-geodesic-contour is missing required API arguments: "
                + ", ".join(missing_parameters)
            )

    def _volume_to_sitk(self, volume_node):
        if volume_node is None:
            raise ValueError("Select an input volume.")
        with tempfile.TemporaryDirectory(prefix="hrpqct_seg_in_") as temp_dir:
            path = Path(temp_dir) / "input.nrrd"
            if not slicer.util.saveNode(volume_node, str(path)):
                raise RuntimeError("Could not save selected Slicer volume for processing.")
            return sitk.ReadImage(str(path))

    def _laplace_hamming_support_image(self, volume_node, reference_image):
        source_path = volume_node.GetAttribute(AIM_SOURCE_ATTRIBUTE) if volume_node is not None else None
        if not source_path:
            raise ValueError(
                "Laplace-Hamming segmentation needs the original AIM source. "
                "Load the image with the Scanco I/O module first so scanner-source metadata is attached."
            )
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Original AIM source for Laplace-Hamming does not exist: {source_path}")

        from timelapsedhrpqct.io.aim import read_aim

        hu_image, _metadata = read_aim(source_path, scaling="hu")
        hu_arr = np.rint(sitk.GetArrayFromImage(hu_image)).astype(np.int16, copy=False)
        image = sitk.GetImageFromArray(hu_arr)
        if image.GetSize() != reference_image.GetSize():
            raise ValueError(
                "Original AIM source size does not match the selected Slicer volume. "
                f"AIM size={image.GetSize()}, selected volume size={reference_image.GetSize()}."
            )
        image.CopyInformation(reference_image)
        return image, source_path

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

    def _geodesic_full_mask_xyz(
        self,
        image,
        *,
        params=None,
        progress_callback=None,
        cancel_callback=None,
    ):
        from hrpqct_geodesic_contour import contour

        params = dict(params or {})
        geodesic_params = dict(params.get("geodesic", {}))
        arr_zyx = sitk.GetArrayFromImage(image)
        arr_xyz = np.transpose(arr_zyx, (2, 1, 0))

        mask_xyz, support_masks = contour(
            arr_xyz,
            voxel_size_mm=tuple(float(value) for value in image.GetSpacing()),
            bone_threshold=float(geodesic_params.get("bone_threshold", 250.0)),
            fill_holes=bool(geodesic_params.get("fill_holes", True)),
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        return mask_xyz.astype(bool, copy=False), len(support_masks)

    def generate_hrpqct_masks(
        self,
        volume_node,
        *,
        site,
        segmentation_method=None,
        periosteal_contour_method="standard",
        endosteal_contour_method="standard",
        method=None,
        output_prefix=None,
        create_labelmaps=True,
        open_segment_editor=False,
        params=None,
        progress_callback=None,
        cancel_callback=None,
    ):
        image = self._volume_to_sitk(volume_node)
        if segmentation_method is None:
            segmentation_method = "seg_gauss" if method is None else str(method)
        segmentation_method = str(segmentation_method)
        periosteal_contour_method = str(periosteal_contour_method)
        endosteal_contour_method = str(endosteal_contour_method)
        if segmentation_method not in SEGMENTATION_METHODS:
            raise ValueError(f"Unsupported bone segmentation method: {segmentation_method}")
        if periosteal_contour_method not in PERIOSTEAL_CONTOUR_METHODS:
            raise ValueError(f"Unsupported periosteal contour method: {periosteal_contour_method}")
        if endosteal_contour_method not in ENDOSTEAL_CONTOUR_METHODS:
            raise ValueError(f"Unsupported endosteal contour method: {endosteal_contour_method}")
        if periosteal_contour_method == "none" and endosteal_contour_method == "standard":
            raise ValueError("Standard endosteal contour requires a periosteal contour.")

        from timelapsedhrpqct.processing.contour_generation import (
            ContourGenerationParams,
            InnerContourParams,
            OuterContourParams,
            SegmentationParams,
            _contour_support_binarization_xyz,
            _ensure_bool,
            _segment_bone_xyz,
            generate_masks_from_image,
            inner_contour,
            numpy_xyz_to_sitk_binary,
            outer_contour,
            sitk_to_numpy_xyz,
        )

        params = dict(params or {})
        site_defaults = SITE_PRESETS[str(site)]
        method_defaults = METHOD_PRESETS.get(segmentation_method, METHOD_PRESETS["seg_gauss"])

        inner_params = dict(site_defaults["inner"])
        outer_params = dict(site_defaults["outer"])
        segmentation_params = dict(method_defaults)
        segmentation_params.update(params.get("segmentation", {}))
        segmentation_params["method"] = "seg_gauss" if segmentation_method == "none" else segmentation_method
        segmentation_params["enabled"] = segmentation_method != "none"

        inner_params.update(params.get("inner", {}))
        outer_params.update(params.get("outer", {}))
        inner_params["site"] = str(site)

        contour_params = ContourGenerationParams(
            outer=OuterContourParams(**outer_params),
            inner=InnerContourParams(**inner_params),
            segmentation=SegmentationParams(**segmentation_params),
        )
        outer_options = asdict(contour_params.outer)
        inner_options = asdict(contour_params.inner)
        segmentation_support_params = contour_params.segmentation

        segmentation_image = None
        source_path = None
        if segmentation_method == "laplace_hamming":
            segmentation_image, source_path = self._laplace_hamming_support_image(volume_node, image)

        if (
            periosteal_contour_method == "standard"
            and endosteal_contour_method == "standard"
            and segmentation_method != "none"
        ):
            generated = generate_masks_from_image(
                image,
                contour_params,
                segmentation_image=segmentation_image,
                verbose=False,
            )
        else:
            image_xyz = sitk_to_numpy_xyz(image)
            segmentation_source = segmentation_image if segmentation_image is not None else image
            segmentation_image_xyz = sitk_to_numpy_xyz(segmentation_source)
            spacing_xyz = tuple(float(value) for value in image.GetSpacing())

            geodesic_support_count = 0
            outer_refine_meta = {}
            if periosteal_contour_method == "geodesic_fracture":
                full_xyz, geodesic_support_count = self._geodesic_full_mask_xyz(
                    image,
                    params=params,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                )
            elif periosteal_contour_method == "standard":
                outer_support_xyz = _contour_support_binarization_xyz(
                    segmentation_image_xyz,
                    params=segmentation_support_params,
                    spacing_xyz=spacing_xyz,
                    role="outer",
                )
                full_xyz, outer_refine_meta = outer_contour(
                    image_xyz,
                    spacing_xyz=spacing_xyz,
                    options=outer_options,
                    support_mask_xyz=outer_support_xyz,
                    verbose=False,
                )
            else:
                full_xyz = np.asarray(image_xyz > 0, dtype=bool)

            inner_support_xyz = _contour_support_binarization_xyz(
                segmentation_image_xyz,
                params=segmentation_support_params,
                spacing_xyz=spacing_xyz,
                full_mask_xyz=full_xyz,
                role="inner",
            )
            if endosteal_contour_method == "standard":
                trab_xyz, cort_xyz = inner_contour(
                    image_xyz,
                    full_xyz,
                    site=str(site),
                    spacing_xyz=spacing_xyz,
                    options=inner_options,
                    support_mask_xyz=inner_support_xyz,
                    verbose=False,
                )
            else:
                trab_xyz = np.zeros_like(full_xyz, dtype=bool)
                cort_xyz = _ensure_bool(full_xyz)

            full_xyz = _ensure_bool(full_xyz)
            trab_xyz = _ensure_bool(trab_xyz) & full_xyz
            cort_xyz = _ensure_bool(cort_xyz) & full_xyz
            if segmentation_method == "none":
                seg_xyz = np.zeros_like(full_xyz, dtype=bool)
            elif segmentation_method in {"adaptive", "laplace_hamming"} and inner_support_xyz is not None:
                seg_xyz = _ensure_bool(inner_support_xyz) & full_xyz
            else:
                seg_xyz = _segment_bone_xyz(
                    image_xyz=segmentation_image_xyz,
                    full_mask_xyz=full_xyz,
                    trab_mask_xyz=trab_xyz,
                    cort_mask_xyz=cort_xyz,
                    params=segmentation_support_params,
                    spacing_xyz=spacing_xyz,
                )
                seg_xyz = _ensure_bool(seg_xyz) & full_xyz

            generated = SimpleNamespace(
                full=numpy_xyz_to_sitk_binary(full_xyz, image),
                trab=numpy_xyz_to_sitk_binary(trab_xyz, image),
                cort=numpy_xyz_to_sitk_binary(cort_xyz, image),
                seg=numpy_xyz_to_sitk_binary(seg_xyz, image),
                metadata={
                    "contour_method": "split_contour_generation",
                    "segmentation_method": segmentation_method,
                    "periosteal_contour_method": periosteal_contour_method,
                    "endosteal_contour_method": endosteal_contour_method,
                    "geodesic_support_mask_count": geodesic_support_count,
                    "outer_edge_refinement": outer_refine_meta,
                    "voxel_counts": {
                        "seg": int(seg_xyz.sum()),
                        "full": int(full_xyz.sum()),
                        "trab": int(trab_xyz.sum()),
                        "cort": int(cort_xyz.sum()),
                    },
                },
            )

        generated.metadata["segmentation_method"] = segmentation_method
        generated.metadata["periosteal_contour_method"] = periosteal_contour_method
        generated.metadata["endosteal_contour_method"] = endosteal_contour_method
        if segmentation_method == "laplace_hamming":
            generated.metadata["segmentation_method"] = "laplace_hamming"
            generated.metadata["segmentation_input_unit"] = "scanco_hu_int16"
            generated.metadata["segmentation_input_path"] = str(source_path)
            generated.metadata["segmentation_input_reader"] = "py_aimio_hu_int16"
            generated.metadata["voxel_counts"]["seg"] = int(
                sitk.GetArrayFromImage(generated.seg).astype(bool, copy=False).sum()
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
        self._geodesic_cancel_requested = False
        self._build_segmentation_section()
        self._build_log_section()
        self.layout.addStretch(1)
        self._apply_site_preset()
        self._apply_segmentation_preset()
        self._update_dependency_ui()
        self._log("Ready.")

    def _build_segmentation_section(self):
        collapsible = ctk.ctkCollapsibleButton()
        collapsible.text = "HR-pQCT Segmentation"
        self.layout.addWidget(collapsible)
        form = qt.QFormLayout(collapsible)

        self.pipelineStatusLabel = qt.QLabel()
        self.installButton = qt.QPushButton("Install / Update contouring dependencies")
        self.updateToolboxButton = qt.QPushButton("Check toolbox updates")
        self.installButton.clicked.connect(self._install_contouring_dependencies)
        self.updateToolboxButton.clicked.connect(self._check_toolbox_updates)
        installRowWidget = qt.QWidget()
        installRow = qt.QHBoxLayout(installRowWidget)
        installRow.setContentsMargins(0, 0, 0, 0)
        installRow.addWidget(self.installButton)
        installRow.addWidget(self.updateToolboxButton)
        form.addRow("Status", self.pipelineStatusLabel)
        form.addRow(installRowWidget)

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

        self.segmentationMethodCombo = qt.QComboBox()
        for label, value in [
            ("Standard Gaussian (trab 320 / cort 450)", "seg_gauss"),
            ("Laplace-Hamming", "laplace_hamming"),
            ("Adaptive threshold", "adaptive"),
            ("None", "none"),
        ]:
            self.segmentationMethodCombo.addItem(label, value)
        self.segmentationMethodCombo.currentIndexChanged.connect(self._apply_segmentation_preset)
        form.addRow("Bone segmentation", self.segmentationMethodCombo)

        self.periostealContourCombo = qt.QComboBox()
        for label, value in [
            ("Standard", "standard"),
            ("Geodesic fracture", "geodesic_fracture"),
            ("None", "none"),
        ]:
            self.periostealContourCombo.addItem(label, value)
        form.addRow("Periosteal (outer) contour", self.periostealContourCombo)

        self.endostealContourCombo = qt.QComboBox()
        for label, value in [
            ("Standard", "standard"),
            ("None", "none"),
        ]:
            self.endostealContourCombo.addItem(label, value)
        form.addRow("Endosteal (inner) contour", self.endostealContourCombo)

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

        self.geodesicBoneThresholdSpin = self._double_spin(0, 5000, 1, 250.0)
        self.geodesicFillHolesCheck = qt.QCheckBox()
        self.geodesicFillHolesCheck.checked = True
        expert_form.addRow("Geodesic bone threshold", self.geodesicBoneThresholdSpin)
        expert_form.addRow("Fill geodesic holes", self.geodesicFillHolesCheck)

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
        self.trabCloseSpin.value = int(inner["trabecular_close_radius"])
        self.outerKernelSpin.value = int(outer["periosteal_kernelsize"])
        self.innerKernelSpin.value = int(inner["endosteal_kernelsize"])
        self.outerOpenSpin.value = int(outer["periosteal_open_radius"])
        self.peelSpin.value = int(inner["peel"])

    def _apply_segmentation_preset(self):
        if not hasattr(self, "segmentationMethodCombo"):
            return
        if str(self.segmentationMethodCombo.currentData) == "none":
            return
        preset = METHOD_PRESETS[self.segmentationMethodCombo.currentData]
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
                "periosteal_kernelsize": int(self.outerKernelSpin.value),
                "periosteal_open_radius": int(self.outerOpenSpin.value),
            },
            "inner": {
                "endosteal_kernelsize": int(self.innerKernelSpin.value),
                "peel": int(self.peelSpin.value),
                "trabecular_close_radius": int(self.trabCloseSpin.value),
            },
            "geodesic": {
                "bone_threshold": float(self.geodesicBoneThresholdSpin.value),
                "fill_holes": bool(self.geodesicFillHolesCheck.checked),
            },
        }

    def _create_segmentation(self):
        try:
            segmentation_method = str(self.segmentationMethodCombo.currentData)
            periosteal_method = str(self.periostealContourCombo.currentData)
            endosteal_method = str(self.endostealContourCombo.currentData)
            progress_callback = None
            cancel_callback = None
            progress_dialog = None
            if periosteal_method == "geodesic_fracture":
                if not self.logic.is_geodesic_contour_available():
                    raise RuntimeError("Install or update contouring dependencies first.")
                progress_dialog, progress_callback, cancel_callback = self._create_geodesic_progress_dialog()
            if (
                segmentation_method != "none"
                or periosteal_method == "standard"
                or endosteal_method == "standard"
            ) and not self.logic.is_pipeline_available():
                raise RuntimeError("Install or update timelapsed-hrpqct first.")
            try:
                segmentation_node, labelmaps, metadata = self.logic.generate_hrpqct_masks(
                    self.volumeSelector.currentNode(),
                    site=str(self.siteCombo.currentData),
                    segmentation_method=segmentation_method,
                    periosteal_contour_method=periosteal_method,
                    endosteal_contour_method=endosteal_method,
                    output_prefix=self.outputPrefixEdit.text.strip() or None,
                    create_labelmaps=bool(self.createLabelmapsCheck.checked),
                    open_segment_editor=bool(self.openEditorCheck.checked),
                    params=self._collect_params(),
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                )
            finally:
                if progress_dialog is not None:
                    progress_dialog.close()
            counts = metadata.get("voxel_counts", {})
            label_text = f" Created {len(labelmaps)} labelmaps." if labelmaps else ""
            self._log(
                f"Created {segmentation_node.GetName()}.{label_text} "
                f"Voxel counts: full={counts.get('full')}, trab={counts.get('trab')}, "
                f"cort={counts.get('cort')}, seg={counts.get('seg')}."
            )
        except Exception as exc:
            self._error(exc)

    def _create_geodesic_progress_dialog(self):
        self._geodesic_cancel_requested = False
        dialog = qt.QProgressDialog(
            "Preparing geodesic fracture contour...",
            "Cancel",
            0,
            0,
            slicer.util.mainWindow(),
        )
        dialog.setWindowTitle("Geodesic Fracture Contour")
        dialog.setWindowModality(qt.Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.canceled.connect(self._request_geodesic_cancel)
        dialog.show()
        slicer.app.processEvents()

        def progress_callback(event):
            message = str(event.get("message") or event.get("stage") or "Working...")
            dialog.setLabelText(message)
            total = event.get("total")
            current = event.get("current")
            if total:
                dialog.setRange(0, int(total))
                dialog.setValue(min(int(current or 0), int(total)))
            else:
                dialog.setRange(0, 0)
            slicer.app.processEvents()
            if self._geodesic_cancel_requested or dialog.wasCanceled:
                raise RuntimeError("Geodesic contour generation was cancelled.")

        def cancel_callback():
            slicer.app.processEvents()
            return bool(self._geodesic_cancel_requested or dialog.wasCanceled)

        return dialog, progress_callback, cancel_callback

    def _request_geodesic_cancel(self):
        self._geodesic_cancel_requested = True
        self._log("Cancelling geodesic contour...")

    def _install_contouring_dependencies(self):
        try:
            self._log("Installing contouring dependencies...")
            self.logic.install_or_update_contouring_dependencies()
            self._update_dependency_ui()
            self._log("Contouring dependencies are installed.")
        except Exception as exc:
            self._error(exc)

    def _check_toolbox_updates(self):
        run_toolbox_update_dialog(__file__, log=self._log)

    def _update_dependency_ui(self):
        pipeline_available = self.logic.is_pipeline_available()
        geodesic_available = self.logic.is_geodesic_contour_available()
        if pipeline_available and geodesic_available:
            self.pipelineStatusLabel.text = "Installed"
            self.pipelineStatusLabel.styleSheet = "color: #228b22;"
        elif pipeline_available or geodesic_available:
            missing = "geodesic contour" if pipeline_available else "timelapsed-hrpqct"
            self.pipelineStatusLabel.text = f"Partly installed; missing {missing}"
            self.pipelineStatusLabel.styleSheet = "color: #cc5500;"
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
