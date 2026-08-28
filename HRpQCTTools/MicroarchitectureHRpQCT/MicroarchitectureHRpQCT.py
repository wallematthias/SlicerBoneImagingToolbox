from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
import sys
import tempfile

import ctk
import numpy as np
import qt
import slicer
import SimpleITK as sitk
import vtk

TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))
MICROARCHITECTURE_LOCAL_REPO = TOOLBOX_ROOT.parent / "bone-microarchitecture"
MICROARCHITECTURE_LOCAL_SRC = MICROARCHITECTURE_LOCAL_REPO / "src"
if MICROARCHITECTURE_LOCAL_SRC.exists() and str(MICROARCHITECTURE_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(MICROARCHITECTURE_LOCAL_SRC))
SCANCO_IO_DIR = TOOLBOX_ROOT / "IOTools" / "ScancoIO"
if str(SCANCO_IO_DIR) not in sys.path:
    sys.path.insert(0, str(SCANCO_IO_DIR))

from SlicerBoneImagingToolboxLib.slicer_pip import slicer_pip_install  # noqa: E402
from SlicerBoneImagingToolboxLib.slicer_update_ui import run_toolbox_update_dialog  # noqa: E402

from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


MODULE_VERSION = "0.1.0"
AIM_SOURCE_ATTRIBUTE = "HRpQCT.AIMSourcePath"
AIM_SCALING_ATTRIBUTE = "HRpQCT.AIMScaling"
SEGMENT_NAME_HINTS = {
    "trabecular segmentation": ("Trabecular mask", "trabecular", "trab"),
    "periosteal mask": ("Full mask", "periosteal", "full"),
    "bone segmentation": ("Bone segmentation", "bone", "seg"),
    "cortical mask": ("Cortical mask", "cortical", "cort"),
}
SEGMENT_ROLE_HINTS = {
    "trabecular segmentation": ("trab", "distal_trab", "proximal_trab"),
    "periosteal mask": ("full", "distal_full", "proximal_full"),
    "bone segmentation": ("seg",),
    "cortical mask": ("cort", "distal_cort", "proximal_cort"),
}


class MicroarchitectureHRpQCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Microarchitecture"
        parent.categories = ["Bone Imaging.HR-pQCT"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Compute HR-pQCT microarchitecture measurements from Slicer masks.\n"
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = (
            "This module wraps a lightweight Python microarchitecture core for Slicer."
        )


class MicroarchitectureHRpQCTLogic(ScriptedLoadableModuleLogic):
    def core_runtime_status(self):
        try:
            version = metadata.version("bone-microarchitecture")
        except metadata.PackageNotFoundError:
            if MICROARCHITECTURE_LOCAL_SRC.exists():
                try:
                    importlib.import_module("bone_microarchitecture")
                except Exception as exc:
                    return False, f"Microarchitecture core source found but not ready: {exc}"
                return True, "Microarchitecture core available from local source."
            return False, "Microarchitecture core is not installed in Slicer Python."
        except Exception as exc:
            return False, f"Microarchitecture core package status could not be checked: {exc}"

        try:
            importlib.import_module("bone_microarchitecture")
        except Exception as exc:
            return False, f"Microarchitecture core installed but not ready ({version}): {exc}"
        return True, f"Microarchitecture core available ({version})."

    def is_core_available(self):
        return self.core_runtime_status()[0]

    def install_or_update_core(self):
        slicer_pip_install("--upgrade --prefer-binary numpy>=1.26,<2.0 scipy>=1.11")
        if sys.platform == "darwin":
            slicer_pip_install("--upgrade --prefer-binary pyobjc-framework-Metal>=10")
        else:
            slicer_pip_install("--upgrade --prefer-binary pyopencl>=2024.1")
        if MICROARCHITECTURE_LOCAL_REPO.exists():
            slicer_pip_install(f"--no-deps -e {MICROARCHITECTURE_LOCAL_REPO}")
        else:
            slicer_pip_install("--upgrade --prefer-binary bone-microarchitecture>=0.1.0")
        importlib.invalidate_caches()
        for name in list(sys.modules):
            if name == "bone_microarchitecture" or name.startswith("bone_microarchitecture."):
                sys.modules.pop(name, None)

    def _volume_to_sitk_uint8(self, volume_node, role, selected_segment_id=None, reference_node=None):
        return self._volume_to_sitk(
            volume_node,
            role,
            sitk.sitkUInt8,
            selected_segment_id=selected_segment_id,
            reference_node=reference_node,
        )

    def _volume_to_sitk(self, volume_node, role, pixel_type=None, *, selected_segment_id=None, reference_node=None):
        if volume_node is None:
            raise ValueError(f"Select a {role}.")
        temporary_node = None
        if volume_node.IsA("vtkMRMLSegmentationNode"):
            volume_node = temporary_node = self._segmentation_node_to_labelmap(
                volume_node,
                role,
                selected_segment_id=selected_segment_id,
                reference_node=reference_node,
            )
        try:
            with tempfile.TemporaryDirectory(prefix="hrpqct_microarch_in_") as temp_dir:
                path = Path(temp_dir) / f"{role.replace(' ', '_')}.nrrd"
                if not slicer.util.saveNode(volume_node, str(path)):
                    raise RuntimeError(f"Could not save selected {role} for microarchitecture processing.")
                if pixel_type is None:
                    return sitk.ReadImage(str(path))
                return sitk.ReadImage(str(path), pixel_type)
        finally:
            if temporary_node is not None:
                slicer.mrmlScene.RemoveNode(temporary_node)

    def _segment_id_for_role(self, segmentation_node, role, selected_segment_id=None):
        segmentation = segmentation_node.GetSegmentation()
        if selected_segment_id:
            segment = segmentation.GetSegment(str(selected_segment_id))
            if segment is None:
                raise ValueError(
                    f"Selected segment ID {selected_segment_id} was not found in {segmentation_node.GetName()}."
                )
            return str(selected_segment_id)
        if segmentation.GetNumberOfSegments() == 1:
            return segmentation.GetNthSegmentID(0)
        hints = SEGMENT_NAME_HINTS.get(str(role), (str(role),))
        normalized_hints = [str(hint).strip().lower() for hint in hints if str(hint).strip()]
        role_hints = {
            str(hint).strip().lower()
            for hint in SEGMENT_ROLE_HINTS.get(str(role), ())
            if str(hint).strip()
        }
        best_fallback = None
        for index in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(index)
            segment = segmentation.GetSegment(segment_id)
            segment_name = str(segment.GetName() if segment is not None else "").strip()
            segment_name_lower = segment_name.lower()
            if segment is not None and role_hints and hasattr(segment, "GetTag"):
                segment_role = self._segment_tag_value(segment, "HRpQCT.Role").strip().lower()
                if segment_role in role_hints:
                    return segment_id
            if segment_name_lower in normalized_hints:
                return segment_id
            if best_fallback is None and any(hint in segment_name_lower for hint in normalized_hints):
                best_fallback = segment_id
        if best_fallback is not None:
            return best_fallback
        raise ValueError(
            f"Could not find a {role} segment in {segmentation_node.GetName()}. "
            f"Expected one of: {', '.join(hints)}."
        )

    def _segment_tag_value(self, segment, tag_name):
        if segment is None or not hasattr(segment, "GetTag"):
            return ""
        try:
            tag_value = vtk.mutable("")
            if segment.GetTag(str(tag_name), tag_value):
                if hasattr(tag_value, "get"):
                    return str(tag_value.get())
                return str(tag_value)
            return ""
        except TypeError:
            return str(segment.GetTag(str(tag_name)) or "")

    def _first_available_reference_node(self, *nodes):
        for node in nodes:
            if node is not None and not node.IsA("vtkMRMLSegmentationNode"):
                return node
        return None

    def _segmentation_reference_node(self, segmentation_node, roles_and_segment_ids):
        segment_ids = vtk.vtkStringArray()
        seen = set()
        for role, selected_segment_id in roles_and_segment_ids:
            segment_id = self._segment_id_for_role(
                segmentation_node,
                role,
                selected_segment_id=selected_segment_id,
            )
            if segment_id not in seen:
                segment_ids.InsertNextValue(segment_id)
                seen.add(segment_id)
        if segment_ids.GetNumberOfValues() == 0:
            return None
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{segmentation_node.GetName()}_microarchitecture_reference_geometry",
        )
        try:
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentation_node,
                segment_ids,
                labelmap_node,
            )
            return labelmap_node
        except Exception:
            slicer.mrmlScene.RemoveNode(labelmap_node)
            raise

    def _segmentation_node_to_labelmap(self, segmentation_node, role, *, selected_segment_id=None, reference_node=None):
        segment_id = self._segment_id_for_role(segmentation_node, role, selected_segment_id=selected_segment_id)
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{segmentation_node.GetName()}_{role.replace(' ', '_')}",
        )
        segment_ids = vtk.vtkStringArray()
        segment_ids.InsertNextValue(segment_id)
        try:
            extent_mode = getattr(slicer.vtkSlicerSegmentationsModuleLogic, "EXTENT_REFERENCE_GEOMETRY", None)
            # Shared reference geometry keeps segmentation segment exports on the same grid.
            export_args = [segmentation_node, segment_ids, labelmap_node]
            if reference_node is not None:
                export_args.append(reference_node)
            if reference_node is not None and extent_mode is not None:
                export_args.append(extent_mode)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                *export_args,
            )
            return labelmap_node
        except Exception:
            slicer.mrmlScene.RemoveNode(labelmap_node)
            raise

    def _sitk_to_scalar_volume(self, image, name, reference_node, map_role):
        with tempfile.TemporaryDirectory(prefix="hrpqct_microarch_out_") as temp_dir:
            path = Path(temp_dir) / f"{name}.nrrd"
            sitk.WriteImage(image, str(path))
            loaded = slicer.util.loadVolume(str(path), {"name": name})
        if isinstance(loaded, tuple):
            success, volume_node = loaded
        else:
            success, volume_node = bool(loaded), loaded
        if not success or volume_node is None:
            raise RuntimeError(f"Could not load generated microarchitecture map: {name}")
        if hasattr(reference_node, "CopyOrientation"):
            volume_node.CopyOrientation(reference_node)
        volume_node.SetAttribute("BoneImaging.Microarchitecture.Engine", "bone_microarchitecture")
        volume_node.SetAttribute("BoneImaging.Microarchitecture.MapRole", map_role)
        return volume_node

    def _array_to_sitk_like(self, array, reference_image):
        image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
        image.CopyInformation(reference_image)
        return image

    def _aimio_bmd_image_from_source(self, grayscale_node):
        source_path = grayscale_node.GetAttribute(AIM_SOURCE_ATTRIBUTE) if grayscale_node is not None else None
        if not source_path:
            return None, {}
        source_path = Path(source_path)
        if not source_path.exists():
            return None, {}
        try:
            from ScancoIOLib import aim_io
        except Exception:
            return None, {}
        image, metadata = aim_io.read_image(source_path, scaling="density")
        metadata = dict(metadata or {})
        metadata["grayscale_reader"] = "aimio-py"
        metadata["grayscale_units"] = "bmd"
        metadata["grayscale_source_path"] = str(source_path)
        return image, metadata

    def _calibrated_grayscale_image(
        self,
        grayscale_node,
        *,
        prefer_aimio=True,
        image_units="bmd",
    ):
        metadata = {
            "grayscale_reader": "selected_slicer_volume",
            "grayscale_units": str(image_units).lower(),
        }
        if grayscale_node is None:
            return None, metadata
        if prefer_aimio:
            image, aimio_metadata = self._aimio_bmd_image_from_source(grayscale_node)
            if image is not None:
                return image, aimio_metadata
        image = self._volume_to_sitk(grayscale_node, "grayscale/BMD volume")
        scaling = grayscale_node.GetAttribute(AIM_SCALING_ATTRIBUTE)
        if scaling:
            metadata["grayscale_slicer_scaling"] = str(scaling)
        return image, metadata

    def _bmd_image(self, image, image_units, mu_scaling, mu_water, rescale_slope, rescale_intercept):
        image = sitk.Cast(image, sitk.sitkFloat32)
        units = str(image_units or "bmd").strip().lower()
        if units == "bmd":
            return image
        if units == "hu":
            attenuation = (image / 1000.0 + 1.0) * float(mu_water)
            return attenuation * float(rescale_slope) + float(rescale_intercept)
        if units == "attenuation":
            return image * float(rescale_slope) + float(rescale_intercept)
        if units == "scanco":
            attenuation = image / float(mu_scaling)
            return attenuation * float(rescale_slope) + float(rescale_intercept)
        raise ValueError("Image units must be one of: BMD, HU, Scanco native, or Attenuation.")

    def _create_measurement_table(self, metrics, maps, name, trab_seg_node, peri_mask_node):
        from bone_microarchitecture.results import SUMMARY_COLUMNS, measurement_rows

        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        table_node.SetAttribute("BoneImaging.Microarchitecture.Engine", "bone_microarchitecture")
        table_node.SetAttribute("BoneImaging.Microarchitecture.TrabecularSegmentationID", trab_seg_node.GetID())
        table_node.SetAttribute("BoneImaging.Microarchitecture.PeriostealMaskID", peri_mask_node.GetID())

        columns = []
        for column_name in SUMMARY_COLUMNS:
            column = vtk.vtkStringArray()
            column.SetName(column_name)
            columns.append(column)

        rows = measurement_rows(metrics, maps)
        for row in rows:
            for column in columns:
                value = row.get(column.GetName(), "")
                if isinstance(value, float):
                    value = f"{value:.8g}"
                column.InsertNextValue(str(value))

        for column in columns:
            table_node.GetTable().AddColumn(column)
        table_node.Modified()
        return table_node

    def compute_trabecular_microarchitecture(
        self,
        trabecular_segmentation_node,
        periosteal_mask_node,
        *,
        grayscale_node=None,
        bone_segmentation_node=None,
        cortical_mask_node=None,
        trabecular_segment_id=None,
        periosteal_segment_id=None,
        bone_segment_id=None,
        cortical_segment_id=None,
        image_units="bmd",
        prefer_aimio_grayscale=True,
        mu_scaling=8192,
        mu_water=0.2409,
        rescale_slope=1603.51904,
        rescale_intercept=-391.209015,
        thickness_method="hildebrand",
        thickness_backend="auto",
        output_prefix="",
        create_maps=True,
        csv_output_path="",
    ):
        from bone_microarchitecture import compute_microarchitecture

        if bone_segmentation_node is None:
            raise ValueError(
                "Select a bone segmentation so trabecular and cortical bone measures can be intersected "
                "with the compartment masks."
            )

        temporary_reference_node = None
        reference_node = self._first_available_reference_node(
            grayscale_node,
            bone_segmentation_node,
            periosteal_mask_node,
            trabecular_segmentation_node,
            cortical_mask_node,
        )
        if (
            reference_node is None
            and trabecular_segmentation_node is not None
            and trabecular_segmentation_node.IsA("vtkMRMLSegmentationNode")
        ):
            roles_and_segment_ids = [
                ("trabecular segmentation", trabecular_segment_id),
                ("periosteal mask", periosteal_segment_id),
            ]
            for optional_node, role, segment_id in (
                (bone_segmentation_node, "bone segmentation", bone_segment_id),
                (cortical_mask_node, "cortical mask", cortical_segment_id),
            ):
                if optional_node is trabecular_segmentation_node:
                    roles_and_segment_ids.append((role, segment_id))
            temporary_reference_node = self._segmentation_reference_node(
                trabecular_segmentation_node,
                roles_and_segment_ids,
            )
            reference_node = temporary_reference_node

        try:
            trab_seg = self._volume_to_sitk_uint8(
                trabecular_segmentation_node,
                "trabecular segmentation",
                selected_segment_id=trabecular_segment_id,
                reference_node=reference_node,
            )
            peri_mask = self._volume_to_sitk_uint8(
                periosteal_mask_node,
                "periosteal mask",
                selected_segment_id=periosteal_segment_id,
                reference_node=reference_node,
            )
            if trab_seg.GetSize() != peri_mask.GetSize():
                raise ValueError(
                    "Trabecular segmentation and periosteal mask sizes must match. "
                    f"Got {trab_seg.GetSize()} and {peri_mask.GetSize()}."
                )

            bone_seg = None
            cort_mask = None
            if bone_segmentation_node is not None:
                bone_seg = self._volume_to_sitk_uint8(
                    bone_segmentation_node,
                    "bone segmentation",
                    selected_segment_id=bone_segment_id,
                    reference_node=reference_node,
                )
            if cortical_mask_node is not None:
                cort_mask = self._volume_to_sitk_uint8(
                    cortical_mask_node,
                    "cortical mask",
                    selected_segment_id=cortical_segment_id,
                    reference_node=reference_node,
                )
            for name, image in (("Bone segmentation", bone_seg), ("Cortical mask", cort_mask)):
                if image is not None and image.GetSize() != trab_seg.GetSize():
                    raise ValueError(f"{name}, trabecular segmentation, and periosteal mask sizes must match.")

            provenance_metadata = {}
            bmd_image = None
            if grayscale_node is not None:
                grayscale, provenance_metadata = self._calibrated_grayscale_image(
                    grayscale_node,
                    prefer_aimio=prefer_aimio_grayscale,
                    image_units=image_units,
                )
                if grayscale.GetSize() != trab_seg.GetSize():
                    raise ValueError("Grayscale/BMD volume size must match the selected masks.")
                bmd_image = self._bmd_image(
                    grayscale,
                    provenance_metadata.get("grayscale_units", image_units),
                    mu_scaling,
                    mu_water,
                    rescale_slope,
                    rescale_intercept,
                )

            core_result = compute_microarchitecture(
                bone_mask=None if bone_seg is None else sitk.GetArrayFromImage(bone_seg),
                periosteal_mask=sitk.GetArrayFromImage(peri_mask),
                trabecular_mask=sitk.GetArrayFromImage(trab_seg),
                cortical_mask=None if cort_mask is None else sitk.GetArrayFromImage(cort_mask),
                grayscale=None if bmd_image is None else sitk.GetArrayFromImage(bmd_image),
                spacing=tuple(reversed(tuple(trab_seg.GetSpacing()))),
                thickness_method=str(thickness_method),
                thickness_backend=str(thickness_backend),
            )

            metrics = dict(core_result.measurements)
            prefix = (str(output_prefix).strip() or trabecular_segmentation_node.GetName() or "HRpQCT").strip()
            table_node = self._create_measurement_table(
                metrics,
                core_result.maps,
                f"{prefix}_microarchitecture",
                trabecular_segmentation_node,
                periosteal_mask_node,
            )
            table_node.SetAttribute("BoneImaging.Microarchitecture.ThicknessMethod", str(core_result.metadata.get("thickness_method", thickness_method)))
            table_node.SetAttribute("BoneImaging.Microarchitecture.ThicknessBackend", str(core_result.metadata.get("thickness_backend", thickness_backend)))
            for key, value in provenance_metadata.items():
                table_node.SetAttribute(f"BoneImaging.Microarchitecture.{key}", str(value))

            output_nodes = {"table": table_node}
            if create_maps:
                for map_role, array in core_result.maps.items():
                    output_nodes[f"{map_role.replace('.', '').lower()}_map"] = self._sitk_to_scalar_volume(
                        self._array_to_sitk_like(array, trab_seg),
                        f"{prefix}_{map_role.replace('.', '')}_map",
                        trabecular_segmentation_node,
                        map_role,
                    )

            csv_path = str(csv_output_path or "").strip()
            if csv_path:
                from bone_microarchitecture.results import write_measurement_csv

                write_measurement_csv(csv_path, metrics, core_result.maps)

            return table_node, output_nodes, metrics, dict(core_result.maps)
        finally:
            if temporary_reference_node is not None:
                slicer.mrmlScene.RemoveNode(temporary_reference_node)


class MicroarchitectureHRpQCTWidget(ScriptedLoadableModuleWidget):
    def _tip(self, widget, text):
        widget.toolTip = str(text)
        return widget

    def setup(self):
        super().setup()
        self.logic = MicroarchitectureHRpQCTLogic()
        self._lastMetrics = None
        self._lastMaps = None
        self._lastTableNode = None

        box = ctk.ctkCollapsibleButton()
        box.text = "Bone Microarchitecture"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)

        self.statusLabel = qt.QLabel()
        self.statusLabel.wordWrap = True
        self._tip(self.statusLabel, "Shows whether the microarchitecture core is available in Slicer Python.")
        form.addRow("Status", self.statusLabel)

        status_buttons = qt.QWidget()
        status_layout = qt.QHBoxLayout(status_buttons)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.installButton = qt.QPushButton("Install / update microarchitecture core")
        self.updateToolboxButton = qt.QPushButton("Check toolbox updates")
        self._tip(self.installButton, "Install or update the lightweight Python core used for microarchitecture measurements.")
        self._tip(self.updateToolboxButton, "Check whether this local Slicer toolbox checkout has upstream updates.")
        self.installButton.clicked.connect(self._install_core)
        self.updateToolboxButton.clicked.connect(self._check_toolbox_updates)
        status_layout.addWidget(self.installButton)
        status_layout.addWidget(self.updateToolboxButton)
        form.addRow(status_buttons)

        self.grayscaleSelector = slicer.qMRMLNodeComboBox()
        self.grayscaleSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.grayscaleSelector.selectNodeUponCreation = False
        self.grayscaleSelector.addEnabled = False
        self.grayscaleSelector.removeEnabled = False
        self.grayscaleSelector.noneEnabled = True
        self.grayscaleSelector.setMRMLScene(slicer.mrmlScene)
        self._tip(
            self.grayscaleSelector,
            "Optional grayscale or BMD-calibrated image used for Tb.BMD and Ct.BMD.",
        )
        form.addRow("Grayscale/BMD volume", self.grayscaleSelector)

        self.boneSegmentationSelector, self.boneSegmentationSegmentCombo = self._mask_selector_row(
            form,
            "Bone segmentation",
            "bone segmentation",
            "Binary mineralized bone segmentation. When selected, it is intersected with the trabecular and cortical compartment masks.",
        )
        self.periostealMaskSelector, self.periostealMaskSegmentCombo = self._mask_selector_row(
            form,
            "Full/periosteal mask",
            "periosteal mask",
            "Full periosteal compartment mask defining the total trabecular analysis region.",
        )
        self.trabecularSegmentationSelector, self.trabecularSegmentationSegmentCombo = self._mask_selector_row(
            form,
            "Trabecular compartment mask",
            "trabecular segmentation",
            "Trabecular compartment mask. Bone measures use this region intersected with the bone segmentation. BMD measures use the full compartment regions.",
        )
        self.corticalMaskSelector, self.corticalMaskSegmentCombo = self._mask_selector_row(
            form,
            "Cortical compartment mask",
            "cortical mask",
            "Optional cortical compartment mask. Cortical bone measures use this region intersected with the bone segmentation. BMD measures use the full compartment regions.",
        )

        self.imageUnitsCombo = qt.QComboBox()
        for label, value in [("BMD", "bmd"), ("HU", "hu"), ("Scanco native", "scanco"), ("Attenuation", "attenuation")]:
            self.imageUnitsCombo.addItem(label, value)
        self._tip(self.imageUnitsCombo, "Units of the optional grayscale/BMD image for BMD conversion.")
        form.addRow("Image units", self.imageUnitsCombo)

        calibration = ctk.ctkCollapsibleButton()
        calibration.text = "BMD Calibration"
        calibration.collapsed = True
        self.layout.addWidget(calibration)
        calibration_form = qt.QFormLayout(calibration)

        self.muScalingSpin = self._double_spin(1, 100000, 1, 8192)
        self.muWaterSpin = self._double_spin(0, 10, 4, 0.2409)
        self.rescaleSlopeSpin = self._double_spin(-100000, 100000, 4, 1603.51904)
        self.rescaleInterceptSpin = self._double_spin(-100000, 100000, 4, -391.209015)
        self._tip(self.muScalingSpin, "mu_scaling value for Scanco native to BMD conversion.")
        self._tip(self.muWaterSpin, "mu_water value for HU to BMD conversion.")
        self._tip(self.rescaleSlopeSpin, "rescale_slope value for BMD conversion.")
        self._tip(self.rescaleInterceptSpin, "rescale_intercept value for BMD conversion.")
        calibration_form.addRow("mu_scaling", self.muScalingSpin)
        calibration_form.addRow("mu_water", self.muWaterSpin)
        calibration_form.addRow("rescale_slope", self.rescaleSlopeSpin)
        calibration_form.addRow("rescale_intercept", self.rescaleInterceptSpin)

        thickness_settings = ctk.ctkCollapsibleButton()
        thickness_settings.text = "Thickness Settings"
        thickness_settings.collapsed = True
        self.layout.addWidget(thickness_settings)
        thickness_form = qt.QFormLayout(thickness_settings)

        self.thicknessMethodCombo = qt.QComboBox()
        for label, value in [("Exact sphere fitting", "hildebrand"), ("Bounded EDT", "edt")]:
            self.thicknessMethodCombo.addItem(label, value)
        self.thicknessBackendCombo = qt.QComboBox()
        for label, value in [("CPU", "cpu"), ("Apple MPS (macOS)", "mps"), ("OpenCL GPU", "opencl")]:
            self.thicknessBackendCombo.addItem(label, value)
        try:
            from bone_microarchitecture import default_thickness_backend

            default_backend = default_thickness_backend()
            index = self.thicknessBackendCombo.findData(default_backend)
            if index >= 0:
                self.thicknessBackendCombo.setCurrentIndex(index)
        except Exception:
            pass
        self._tip(
            self.thicknessMethodCombo,
            "Thickness calculation method. Exact sphere fitting is the measurement default; bounded EDT is a fast preview/fallback.",
        )
        self._tip(
            self.thicknessBackendCombo,
            "Backend for exact sphere fitting. Auto defaults to Apple Metal on macOS, OpenCL on Windows/Linux when available, and CPU otherwise.",
        )
        thickness_form.addRow("Method", self.thicknessMethodCombo)
        thickness_form.addRow("Backend", self.thicknessBackendCombo)

        self.outputPrefixEdit = qt.QLineEdit()
        self._tip(self.outputPrefixEdit, "Optional prefix for the output table and automatically loaded map nodes.")
        form.addRow("Output prefix", self.outputPrefixEdit)

        self.runButton = qt.QPushButton("Run microarchitecture")
        self.runButton.clicked.connect(self._run_microarchitecture)
        self._tip(
            self.runButton,
            "Compute microarchitecture measurements, show the Slicer table, and load available "
            "Tb.Th, Tb.Sp, Tb.N, Ct.Th, Ct.Po.Dm, Tb.BMD, and Ct.BMD map volumes.",
        )
        form.addRow(self.runButton)

        self.exportCsvButton = qt.QPushButton("Export measurements CSV")
        self.exportCsvButton.enabled = False
        self.exportCsvButton.clicked.connect(self._export_measurements_csv)
        self._tip(self.exportCsvButton, "Export the most recent microarchitecture measurement table to CSV.")
        form.addRow(self.exportCsvButton)

        self.logText = qt.QTextEdit()
        self.logText.readOnly = True
        self.logText.minimumHeight = 120
        self.logText.placeholderText = "Microarchitecture log"
        self.layout.addWidget(self.logText)
        self.layout.addStretch(1)
        self._update_dependency_ui()

    def _labelmap_selector(self):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode", "vtkMRMLSegmentationNode"]
        selector.selectNodeUponCreation = False
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = True
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def _segment_combo(self):
        combo = qt.QComboBox()
        combo.addItem("Auto", "")
        combo.enabled = False
        self._tip(combo, "Segment to use when the selected node is a Slicer segmentation.")
        return combo

    def _mask_selector_row(self, form, label, role, tooltip):
        selector = self._labelmap_selector()
        segment_combo = self._segment_combo()
        self._tip(selector, tooltip)

        row_widget = qt.QWidget()
        row_layout = qt.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(selector, 2)
        row_layout.addWidget(qt.QLabel("Segment"))
        row_layout.addWidget(segment_combo, 1)
        form.addRow(label, row_widget)

        selector.currentNodeChanged.connect(
            lambda _node, active_selector=selector, active_combo=segment_combo, active_role=role: self._refresh_segment_combo(
                active_selector,
                active_combo,
                active_role,
            )
        )
        return selector, segment_combo

    def _refresh_segment_combo(self, selector, segment_combo, role):
        segment_combo.blockSignals(True)
        segment_combo.clear()
        segment_combo.addItem("Auto", "")
        node = selector.currentNode()
        is_segmentation = bool(node is not None and node.IsA("vtkMRMLSegmentationNode"))
        segment_combo.enabled = is_segmentation
        if is_segmentation:
            try:
                auto_id = self.logic._segment_id_for_role(node, role)
            except Exception:
                auto_id = None
            segmentation = node.GetSegmentation()
            for index in range(segmentation.GetNumberOfSegments()):
                segment_id = segmentation.GetNthSegmentID(index)
                segment = segmentation.GetSegment(segment_id)
                segment_name = str(segment.GetName() if segment is not None else segment_id)
                label = f"{segment_name}"
                if auto_id and segment_id == auto_id:
                    label = f"{segment_name} (auto)"
                segment_combo.addItem(label, segment_id)
        segment_combo.blockSignals(False)

    def _selected_segment_id(self, segment_combo):
        selected_segment_id = str(segment_combo.currentData or "").strip()
        return selected_segment_id or None

    def _double_spin(self, minimum, maximum, decimals, value):
        spin = qt.QDoubleSpinBox()
        spin.minimum = minimum
        spin.maximum = maximum
        spin.decimals = decimals
        spin.value = value
        return spin

    def _log(self, message):
        self.logText.append(str(message).rstrip())
        self.logText.ensureCursorVisible()

    def _with_wait_cursor(self, func):
        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            return func()
        finally:
            try:
                slicer.app.restoreOverrideCursor()
            except Exception:
                pass

    def _update_dependency_ui(self):
        available, message = self.logic.core_runtime_status()
        self.statusLabel.text = message
        self.runButton.enabled = available

    def _install_core(self):
        try:
            self._with_wait_cursor(self.logic.install_or_update_core)
        except Exception as exc:
            slicer.util.errorDisplay(f"Microarchitecture core installation failed:\n{exc}")
            self._log(f"[setup] microarchitecture core installation failed: {exc}")
            return
        self._log("[setup] microarchitecture core installed or updated.")
        self._update_dependency_ui()

    def _check_toolbox_updates(self):
        try:
            run_toolbox_update_dialog(__file__, parent=slicer.util.mainWindow())
        except Exception as exc:
            slicer.util.errorDisplay(f"Toolbox update check failed:\n{exc}")
            self._log(f"[toolbox] update check failed: {exc}")

    def _show_measurement_table(self, table_node):
        try:
            slicer.util.selectModule("Tables")
            tables_widget = slicer.modules.tables.widgetRepresentation()
            tables_widget.setCurrentTableNode(table_node)
        except Exception as exc:
            self._log(f"[microarchitecture] table created but could not switch to Tables module: {exc}")

    def _export_measurements_csv(self):
        if not self._lastMetrics:
            slicer.util.errorDisplay("Run microarchitecture before exporting measurements.")
            return
        path = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Export microarchitecture measurements",
            "",
            "CSV files (*.csv)",
        )
        if isinstance(path, tuple):
            path = path[0]
        path = str(path or "").strip()
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path = f"{path}.csv"
        try:
            from bone_microarchitecture.results import write_measurement_csv

            write_measurement_csv(path, self._lastMetrics, self._lastMaps)
        except Exception as exc:
            slicer.util.errorDisplay(f"Measurement CSV export failed:\n{exc}")
            self._log(f"[export] failed: {exc}")
            return
        self._log(f"[export] wrote measurements CSV: {path}")

    def _run_microarchitecture(self):
        try:
            if str(self.thicknessBackendCombo.currentData) == "mps":
                self._log("[microarchitecture] Apple MPS sphere fitting is experimental and may keep Slicer busy.")
            table_node, output_nodes, metrics, maps = self._with_wait_cursor(
                lambda: self.logic.compute_trabecular_microarchitecture(
                    self.trabecularSegmentationSelector.currentNode(),
                    self.periostealMaskSelector.currentNode(),
                    grayscale_node=self.grayscaleSelector.currentNode(),
                    bone_segmentation_node=self.boneSegmentationSelector.currentNode(),
                    cortical_mask_node=self.corticalMaskSelector.currentNode(),
                    trabecular_segment_id=self._selected_segment_id(self.trabecularSegmentationSegmentCombo),
                    periosteal_segment_id=self._selected_segment_id(self.periostealMaskSegmentCombo),
                    bone_segment_id=self._selected_segment_id(self.boneSegmentationSegmentCombo),
                    cortical_segment_id=self._selected_segment_id(self.corticalMaskSegmentCombo),
                    image_units=self.imageUnitsCombo.currentData,
                    prefer_aimio_grayscale=True,
                    mu_scaling=self.muScalingSpin.value,
                    mu_water=self.muWaterSpin.value,
                    rescale_slope=self.rescaleSlopeSpin.value,
                    rescale_intercept=self.rescaleInterceptSpin.value,
                    thickness_method=str(self.thicknessMethodCombo.currentData),
                    thickness_backend=str(self.thicknessBackendCombo.currentData),
                    output_prefix=self.outputPrefixEdit.text,
                    create_maps=True,
                )
            )
        except Exception as exc:
            slicer.util.errorDisplay(f"Microarchitecture failed:\n{exc}")
            self._log(f"[microarchitecture] failed: {exc}")
            return

        self._lastMetrics = dict(metrics)
        self._lastMaps = dict(maps)
        self._lastTableNode = table_node
        self.exportCsvButton.enabled = True
        self._show_measurement_table(table_node)
        self._log(f"[microarchitecture] wrote table: {table_node.GetName()}")
        for key, node in output_nodes.items():
            if key != "table":
                self._log(f"[microarchitecture] wrote {key}: {node.GetName()}")
        from bone_microarchitecture.results import measurement_rows

        for row in measurement_rows(metrics, maps):
            self._log(f"{row['Parameter']}: {float(row['Mean']):.6g} {row['Units']}")


class MicroarchitectureHRpQCTTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.test_MicroarchitectureHRpQCT1()

    def test_MicroarchitectureHRpQCT1(self):
        self.delayDisplay("Microarchitecture HR-pQCT smoke test")
        logic = MicroarchitectureHRpQCTLogic()
        if not logic.is_core_available():
            self.skipTest("Microarchitecture core is not installed")
        self.assertTrue(logic.is_core_available())
