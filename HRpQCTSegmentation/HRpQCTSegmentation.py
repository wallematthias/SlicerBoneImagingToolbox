import tempfile
import sys
import importlib
import inspect
import json
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
AIM_METADATA_ATTRIBUTE = "HRpQCT.AIMMetadata"
AIM_SOURCE_ATTRIBUTE = "HRpQCT.AIMSourcePath"
AIM_SCALING_ATTRIBUTE = "HRpQCT.AIMScaling"
CORE_PIP_CONSTRAINTS = ("numpy>=1.26,<2.0", "scikit-image>=0.24,<0.26", "tifffile<2026")

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


def _same_shape_or_raise(named_arrays):
    present = [(name, np.asarray(array)) for name, array in named_arrays if array is not None]
    if not present:
        raise ValueError("Select at least one mask.")
    shape = present[0][1].shape
    mismatches = [f"{name}={array.shape}" for name, array in present if array.shape != shape]
    if mismatches:
        raise ValueError(f"Selected masks must have the same shape. Expected {shape}; got {', '.join(mismatches)}.")
    return shape


def _derive_compartment_mask_arrays(*, full=None, trab=None, cort=None, output_role=None):
    provided = {
        "full": None if full is None else np.asarray(full, dtype=bool),
        "trab": None if trab is None else np.asarray(trab, dtype=bool),
        "cort": None if cort is None else np.asarray(cort, dtype=bool),
    }
    _same_shape_or_raise(provided.items())
    provided_roles = [role for role, array in provided.items() if array is not None]
    if len(provided_roles) < 2:
        raise ValueError("Select any two of full, trabecular, and cortical masks.")

    missing_roles = [role for role, array in provided.items() if array is None]
    if output_role in (None, "auto"):
        output_role = missing_roles[0] if len(missing_roles) == 1 else "none"
    output_role = str(output_role)
    if output_role not in {"full", "trab", "cort", "none"}:
        raise ValueError(f"Unsupported output mask role: {output_role}")

    full_array = provided["full"]
    trab_array = provided["trab"]
    cort_array = provided["cort"]

    if full_array is None:
        full_array = trab_array | cort_array
    if trab_array is None:
        trab_array = full_array & ~cort_array
    if cort_array is None:
        cort_array = full_array & ~trab_array

    return {
        "full": full_array.astype(bool, copy=False),
        "trab": trab_array.astype(bool, copy=False),
        "cort": cort_array.astype(bool, copy=False),
        "derived_role": output_role,
    }


def _validate_compartment_mask_arrays(*, full=None, trab=None, cort=None):
    masks = _derive_compartment_mask_arrays(full=full, trab=trab, cort=cort, output_role="auto")
    full_array = masks["full"]
    trab_array = masks["trab"]
    cort_array = masks["cort"]
    union = trab_array | cort_array
    overlap = trab_array & cort_array
    outside = union & ~full_array
    missing = full_array & ~union
    return {
        "full": int(np.count_nonzero(full_array)),
        "trab": int(np.count_nonzero(trab_array)),
        "cort": int(np.count_nonzero(cort_array)),
        "overlap": int(np.count_nonzero(overlap)),
        "outside_full": int(np.count_nonzero(outside)),
        "full_not_compartment": int(np.count_nonzero(missing)),
        "valid": bool(not np.any(overlap) and not np.any(outside) and not np.any(missing)),
    }


def _binary_mask_operation_arrays(mask_a, mask_b, operation):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    _same_shape_or_raise([("mask_a", mask_a), ("mask_b", mask_b)])
    operation = str(operation)
    if operation == "union":
        return mask_a | mask_b
    if operation == "intersection":
        return mask_a & mask_b
    if operation == "difference":
        return mask_a & ~mask_b
    if operation == "xor":
        return np.logical_xor(mask_a, mask_b)
    raise ValueError(f"Unsupported mask operation: {operation}")


def _relabel_nonzero_array(array, label):
    label = int(label)
    if label < 1:
        raise ValueError("Output label must be greater than zero.")
    dtype = np.uint8 if label <= 255 else np.uint16
    relabelled = np.zeros(np.asarray(array).shape, dtype=dtype)
    relabelled[np.asarray(array) > 0] = label
    return relabelled


def _material_labels_from_arrays(seg, trab, cort, *, trab_label=126, cort_label=127, cort_source="cort_mask"):
    seg = np.asarray(seg, dtype=bool)
    trab = np.asarray(trab, dtype=bool)
    cort = np.asarray(cort, dtype=bool)
    _same_shape_or_raise([("bone segmentation", seg), ("trabecular mask", trab), ("cortical mask", cort)])
    material = np.zeros(seg.shape, dtype=np.uint8)
    material[seg & trab] = int(trab_label)
    material[seg & cort] = int(cort_label)
    return material, {
        "trab": int(np.count_nonzero(material == int(trab_label))),
        "cort": int(np.count_nonzero(material == int(cort_label))),
        "cort_source": str(cort_source),
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

    def is_geodesic_contour_available(self):
        try:
            import hrpqct_geodesic_contour  # noqa: F401

            return True
        except Exception:
            return False

    def install_or_update_pipeline(self):
        self._remove_incompatible_optional_packages()
        packages = " ".join(["timelapsed-hrpqct", *CORE_PIP_CONSTRAINTS])
        slicer.util.pip_install(f"--upgrade --force-reinstall --no-cache-dir {packages}")

    def _remove_incompatible_optional_packages(self):
        if not hasattr(slicer.util, "pip_uninstall"):
            return
        try:
            slicer.util.pip_uninstall("pyjpegls")
        except Exception:
            pass

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

    def _node_aim_metadata(self, volume_node):
        if volume_node is None:
            return {}
        metadata_text = volume_node.GetAttribute(AIM_METADATA_ATTRIBUTE)
        if not metadata_text:
            return {}
        try:
            metadata = json.loads(metadata_text)
        except Exception:
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def _copy_aim_attributes(self, source_node, target_node):
        if source_node is None or target_node is None:
            return
        for attribute in (AIM_METADATA_ATTRIBUTE, AIM_SOURCE_ATTRIBUTE, AIM_SCALING_ATTRIBUTE):
            value = source_node.GetAttribute(attribute)
            if value:
                target_node.SetAttribute(attribute, value)

    def _processing_log_from_metadata(self, metadata):
        for raw in (
            metadata.get("processing_log_raw"),
            metadata.get("processing_log"),
        ):
            if isinstance(raw, str) and raw.strip():
                return raw
        raw = metadata.get("processing_log")
        if isinstance(raw, dict) and raw:
            try:
                from timelapsedhrpqct.io.metadata import dicttolog

                return dicttolog(raw)
            except Exception:
                pass
        raise ValueError("AIM metadata does not contain calibration processing_log.")

    def _density_image_to_laplace_hamming_native(self, reference_image, metadata):
        from timelapsedhrpqct.io.aim import density_to_native_int16

        arr_zyx = sitk.GetArrayFromImage(reference_image)
        native_zyx = density_to_native_int16(
            arr_zyx,
            self._processing_log_from_metadata(metadata),
        )
        image = sitk.GetImageFromArray(native_zyx)
        image.CopyInformation(reference_image)
        return image

    def _selected_volume_native_image(self, reference_image):
        arr_zyx = np.rint(sitk.GetArrayFromImage(reference_image)).astype(np.int16, copy=False)
        image = sitk.GetImageFromArray(arr_zyx)
        image.CopyInformation(reference_image)
        return image

    def _read_laplace_hamming_native_aim(self, source_path, reference_image):
        from timelapsedhrpqct.io.aim import read_aim

        native_image, _metadata = read_aim(source_path, scaling="native")
        native_arr = np.rint(sitk.GetArrayFromImage(native_image)).astype(np.int16, copy=False)
        image = sitk.GetImageFromArray(native_arr)
        if image.GetSize() != reference_image.GetSize():
            raise ValueError(
                "Original AIM source size does not match the selected Slicer volume. "
                f"AIM size={image.GetSize()}, selected volume size={reference_image.GetSize()}."
            )
        image.CopyInformation(reference_image)
        return image

    def _laplace_hamming_support_image(self, volume_node, reference_image):
        metadata = self._node_aim_metadata(volume_node)
        scaling = str(volume_node.GetAttribute(AIM_SCALING_ATTRIBUTE) or "").strip().lower() if volume_node is not None else ""
        if scaling in {"native", "none"}:
            return self._selected_volume_native_image(reference_image), {
                "segmentation_input_unit": "scanco_native_int16",
                "segmentation_input_reader": "selected_volume_native_int16",
                "segmentation_input_path": str(volume_node.GetName()) if volume_node is not None else "",
                "segmentation_input_reason": "Laplace-Hamming threshold is calibrated for native Scanco attenuation values.",
            }
        try:
            return self._density_image_to_laplace_hamming_native(reference_image, metadata), {
                "segmentation_input_unit": "scanco_native_int16",
                "segmentation_input_reader": "imported_density_to_native_int16",
                "segmentation_input_path": str(volume_node.GetName()) if volume_node is not None else "",
                "segmentation_input_reason": "Laplace-Hamming threshold is calibrated for native Scanco attenuation values.",
            }
        except ValueError:
            pass

        source_path = volume_node.GetAttribute(AIM_SOURCE_ATTRIBUTE) if volume_node is not None else None
        if not source_path:
            raise ValueError(
                "Laplace-Hamming segmentation needs the original AIM source. "
                "Load the image with the Scanco I/O module first so scanner-source metadata is attached, "
                "or use a volume with AIM calibration metadata so density can be converted back to native Scanco units."
            )
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Original AIM source for Laplace-Hamming does not exist: {source_path}")

        return self._read_laplace_hamming_native_aim(source_path, reference_image), {
            "segmentation_input_unit": "scanco_native_int16",
            "segmentation_input_reader": "py_aimio_native_int16",
            "segmentation_input_path": str(source_path),
            "segmentation_input_reason": "Laplace-Hamming threshold is calibrated for native Scanco attenuation values.",
        }

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
        self._copy_aim_attributes(reference_node, label_node)
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

    def _mask_array_from_node(self, node, role):
        if node is None:
            return None
        return np.asarray(slicer.util.arrayFromVolume(node)) > 0

    def _labelmap_from_array(self, array, reference_node, name, *, attributes=None):
        if reference_node is None:
            raise ValueError("Select a reference labelmap.")
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", str(name).strip() or "HRpQCT_mask")
        slicer.util.updateVolumeFromArray(node, np.asarray(array))
        node.CopyOrientation(reference_node)
        node.CreateDefaultDisplayNodes()
        for key, value in (attributes or {}).items():
            node.SetAttribute(str(key), str(value))
        return node

    def _first_selected_node(self, *nodes):
        for node in nodes:
            if node is not None:
                return node
        return None

    def create_missing_mask_volume(
        self,
        *,
        full_mask_node=None,
        trab_mask_node=None,
        cort_mask_node=None,
        output_role="auto",
        output_name="HRpQCT_derived_mask",
    ):
        masks = _derive_compartment_mask_arrays(
            full=self._mask_array_from_node(full_mask_node, "full"),
            trab=self._mask_array_from_node(trab_mask_node, "trab"),
            cort=self._mask_array_from_node(cort_mask_node, "cort"),
            output_role=output_role,
        )
        role = masks["derived_role"]
        if role == "none":
            raise ValueError("Choose which mask to generate when all three compartment masks are selected.")
        reference = self._first_selected_node(full_mask_node, trab_mask_node, cort_mask_node)
        node = self._labelmap_from_array(
            masks[role].astype(np.uint8),
            reference,
            output_name or f"HRpQCT_{role}_derived",
            attributes={
                "HRpQCT.MaskRole": role,
                "HRpQCT.MaskDerived": "1",
            },
        )
        return node, {"role": role, "voxels": int(np.count_nonzero(masks[role]))}

    def validate_compartment_masks(self, *, full_mask_node=None, trab_mask_node=None, cort_mask_node=None):
        return _validate_compartment_mask_arrays(
            full=self._mask_array_from_node(full_mask_node, "full"),
            trab=self._mask_array_from_node(trab_mask_node, "trab"),
            cort=self._mask_array_from_node(cort_mask_node, "cort"),
        )

    def create_boolean_mask_volume(self, mask_a_node, mask_b_node, operation, output_name="HRpQCT_mask_operation"):
        if mask_a_node is None or mask_b_node is None:
            raise ValueError("Select both input masks.")
        result = _binary_mask_operation_arrays(
            self._mask_array_from_node(mask_a_node, "mask A"),
            self._mask_array_from_node(mask_b_node, "mask B"),
            operation,
        )
        node = self._labelmap_from_array(
            result.astype(np.uint8),
            mask_a_node,
            output_name or f"HRpQCT_{operation}",
            attributes={
                "HRpQCT.MaskOperation": str(operation),
            },
        )
        return node, {"voxels": int(np.count_nonzero(result)), "operation": str(operation)}

    def relabel_mask_volume(self, source_node, label, output_name="HRpQCT_relabelled"):
        if source_node is None:
            raise ValueError("Select a source mask.")
        result = _relabel_nonzero_array(slicer.util.arrayFromVolume(source_node), int(label))
        node = self._labelmap_from_array(
            result,
            source_node,
            output_name or "HRpQCT_relabelled",
            attributes={
                "HRpQCT.RelabelValue": int(label),
            },
        )
        return node, {"voxels": int(np.count_nonzero(result)), "label": int(label)}

    def mask_voxel_counts(self, **nodes):
        counts = {}
        for role, node in nodes.items():
            if node is not None:
                counts[role] = int(np.count_nonzero(slicer.util.arrayFromVolume(node)))
        if not counts:
            raise ValueError("Select at least one mask.")
        return counts

    def create_material_label_volume(
        self,
        bone_segmentation_node,
        trab_mask_node=None,
        cort_mask_node=None,
        full_mask_node=None,
        *,
        trab_label=126,
        cort_label=127,
        output_name="HRpQCT_HOM_material_labels",
    ):
        if bone_segmentation_node is None:
            raise ValueError("Select a bone segmentation labelmap.")

        seg = self._mask_array_from_node(bone_segmentation_node, "bone segmentation")
        masks = _derive_compartment_mask_arrays(
            full=self._mask_array_from_node(full_mask_node, "full"),
            trab=self._mask_array_from_node(trab_mask_node, "trab"),
            cort=self._mask_array_from_node(cort_mask_node, "cort"),
            output_role="auto",
        )
        if seg.shape != masks["trab"].shape:
            raise ValueError(
                f"Bone segmentation shape {seg.shape} does not match compartment mask shape {masks['trab'].shape}."
            )
        cort_source = (
            "cort_mask"
            if cort_mask_node is not None
            else "derived_from_full_minus_trab"
            if full_mask_node is not None and trab_mask_node is not None
            else "derived_from_full_minus_trab"
        )

        material, counts = _material_labels_from_arrays(
            seg,
            masks["trab"],
            masks["cort"],
            trab_label=int(trab_label),
            cort_label=int(cort_label),
            cort_source=cort_source,
        )
        if not np.any(material):
            raise ValueError("The selected segmentation and compartment masks do not overlap.")

        node = self._labelmap_from_array(
            material,
            bone_segmentation_node,
            output_name or "HRpQCT_HOM_material_labels",
            attributes={
                "HRpQCT.MaterialLabels": "1",
                "HRpQCT.MaterialLabel.Trabecular": int(trab_label),
                "HRpQCT.MaterialLabel.Cortical": int(cort_label),
                "HRpQCT.MaterialLabel.CorticalSource": cort_source,
                "HRpQCT.MaterialLabel.DerivedMaskRole": masks["derived_role"],
            },
        )
        return node, counts

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
        create_labelmaps=False,
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
        segmentation_source_meta = {}
        if segmentation_method == "laplace_hamming":
            segmentation_image, segmentation_source_meta = self._laplace_hamming_support_image(volume_node, image)

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
                full_xyz = np.ones_like(image_xyz, dtype=bool)

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
                cort_xyz = np.zeros_like(full_xyz, dtype=bool)

            full_xyz = _ensure_bool(full_xyz)
            trab_xyz = _ensure_bool(trab_xyz) & full_xyz
            cort_xyz = _ensure_bool(cort_xyz) & full_xyz
            if segmentation_method == "none":
                seg_xyz = np.zeros_like(full_xyz, dtype=bool)
            elif segmentation_method == "laplace_hamming" and inner_support_xyz is not None:
                seg_xyz = _ensure_bool(inner_support_xyz) & full_xyz
            elif segmentation_method == "adaptive" and inner_support_xyz is not None:
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

        periosteal_contour_generated = periosteal_contour_method != "none"
        compartment_split_generated = endosteal_contour_method == "standard"
        if not compartment_split_generated:
            full_xyz = sitk_to_numpy_xyz(generated.full) > 0
            empty_xyz = np.zeros_like(full_xyz, dtype=bool)
            generated.trab = numpy_xyz_to_sitk_binary(empty_xyz, image)
            generated.cort = numpy_xyz_to_sitk_binary(empty_xyz, image)

        if segmentation_method == "laplace_hamming" and segmentation_image is not None:
            full_xyz = sitk_to_numpy_xyz(generated.full) > 0
            segmentation_image_xyz = sitk_to_numpy_xyz(segmentation_image)
            spacing_xyz = tuple(float(value) for value in image.GetSpacing())
            lh_support_xyz = _contour_support_binarization_xyz(
                segmentation_image_xyz,
                params=segmentation_support_params,
                spacing_xyz=spacing_xyz,
                full_mask_xyz=full_xyz,
                role="inner",
            )
            if lh_support_xyz is not None:
                seg_xyz = _ensure_bool(lh_support_xyz) & full_xyz
                generated.seg = numpy_xyz_to_sitk_binary(seg_xyz, image)
                generated.metadata.setdefault("voxel_counts", {})
                generated.metadata["voxel_counts"]["seg"] = int(seg_xyz.sum())

        generated.metadata["segmentation_method"] = segmentation_method
        generated.metadata["periosteal_contour_method"] = periosteal_contour_method
        generated.metadata["endosteal_contour_method"] = endosteal_contour_method
        generated.metadata["periosteal_contour_generated"] = bool(periosteal_contour_generated)
        generated.metadata["compartment_split_generated"] = bool(compartment_split_generated)
        if not periosteal_contour_generated:
            full_xyz = sitk_to_numpy_xyz(generated.full) > 0
            empty_xyz = np.zeros_like(full_xyz, dtype=bool)
            generated.full = numpy_xyz_to_sitk_binary(empty_xyz, image)
            generated.metadata["periosteal_contour_reason"] = "periosteal_contour_method_none"
            generated.metadata.setdefault("voxel_counts", {})
            generated.metadata["voxel_counts"]["full"] = 0
        if not compartment_split_generated:
            generated.metadata["compartment_split_reason"] = "endosteal_contour_method_none"
            generated.metadata.setdefault("voxel_counts", {})
            generated.metadata["voxel_counts"]["trab"] = 0
            generated.metadata["voxel_counts"]["cort"] = 0
        if segmentation_method == "laplace_hamming":
            generated.metadata["segmentation_method"] = "laplace_hamming"
            generated.metadata.update(segmentation_source_meta)
            generated.metadata["voxel_counts"]["seg"] = int(
                sitk.GetArrayFromImage(generated.seg).astype(bool, copy=False).sum()
            )
            if generated.metadata["voxel_counts"]["seg"] == 0:
                raise RuntimeError(
                    "Laplace-Hamming produced an empty bone segmentation. "
                    "Check that the selected volume has valid AIM calibration metadata "
                    "or an original AIM source, and that LH parameters match native Scanco units."
                )

        prefix = output_prefix.strip() if output_prefix else volume_node.GetName()
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            f"{prefix}_HRpQCT_segmentation",
        )
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
        segmentation_node.CreateDefaultDisplayNodes()
        self._copy_aim_attributes(volume_node, segmentation_node)
        for key in (
            "segmentation_method",
            "segmentation_input_unit",
            "segmentation_input_reader",
            "segmentation_input_path",
            "periosteal_contour_method",
            "periosteal_contour_generated",
            "periosteal_contour_reason",
            "endosteal_contour_method",
            "compartment_split_generated",
        ):
            if key in generated.metadata:
                segmentation_node.SetAttribute(f"HRpQCT.{key}", str(generated.metadata[key]))

        outputs = {}
        output_specs = [
            ("full", generated.full, "Full mask"),
            ("trab", generated.trab, "Trabecular mask"),
            ("cort", generated.cort, "Cortical mask"),
            ("seg", generated.seg, "Bone segmentation"),
        ]
        if not periosteal_contour_generated:
            output_specs = [spec for spec in output_specs if spec[0] != "full"]
        if not compartment_split_generated:
            output_specs = [spec for spec in output_specs if spec[0] in {"full", "seg"}]
        generated.metadata["emitted_roles"] = [role for role, _image_out, _segment_name in output_specs]
        for role, image_out, segment_name in output_specs:
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
    def _tip(self, widget, text):
        widget.toolTip = str(text)
        return widget

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
        self.toolTabs = qt.QTabWidget()
        self.layout.addWidget(self.toolTabs)

        generate_tab = qt.QWidget()
        generate_layout = qt.QVBoxLayout(generate_tab)
        form = qt.QFormLayout()
        generate_layout.addLayout(form)

        self.pipelineStatusLabel = qt.QLabel()
        self.installButton = qt.QPushButton("Install / Update contouring dependencies")
        self.updateToolboxButton = qt.QPushButton("Check toolbox updates")
        self._tip(self.pipelineStatusLabel, "Shows whether the core contouring packages are available in Slicer Python.")
        self._tip(self.installButton, "Install or update timelapsed-hrpqct and hrpqct-geodesic-contour in Slicer Python.")
        self._tip(self.updateToolboxButton, "Check whether this local Slicer toolbox checkout has upstream updates.")
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
        self._tip(self.volumeSelector, "Input HR-pQCT volume used to generate masks and bone segmentation.")
        form.addRow("Input volume", self.volumeSelector)

        self.siteCombo = qt.QComboBox()
        for label, value in [("Radius", "radius"), ("Tibia", "tibia"), ("Knee", "knee")]:
            self.siteCombo.addItem(label, value)
        self.siteCombo.currentIndexChanged.connect(self._apply_site_preset)
        self._tip(self.siteCombo, "Applies radius, tibia, or knee defaults for contour thresholds and morphology.")
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
        self._tip(
            self.segmentationMethodCombo,
            "Bone binarization method. Laplace-Hamming uses native Scanco attenuation values from AIM metadata/source.",
        )
        form.addRow("Bone segmentation", self.segmentationMethodCombo)

        self.periostealContourCombo = qt.QComboBox()
        for label, value in [
            ("Standard", "standard"),
            ("Geodesic fracture", "geodesic_fracture"),
            ("None", "none"),
        ]:
            self.periostealContourCombo.addItem(label, value)
        self._tip(self.periostealContourCombo, "Outer contour method for the full bone mask.")
        form.addRow("Periosteal (outer) contour", self.periostealContourCombo)

        self.endostealContourCombo = qt.QComboBox()
        for label, value in [
            ("Standard", "standard"),
            ("None", "none"),
        ]:
            self.endostealContourCombo.addItem(label, value)
        self._tip(self.endostealContourCombo, "Inner contour method used to split full mask into trabecular and cortical compartments.")
        form.addRow("Endosteal (inner) contour", self.endostealContourCombo)

        self.outputPrefixEdit = qt.QLineEdit()
        self._tip(self.outputPrefixEdit, "Optional prefix for generated Slicer nodes. Leave empty to use the input volume name.")
        form.addRow("Output prefix", self.outputPrefixEdit)

        self.openEditorCheck = qt.QCheckBox()
        self.openEditorCheck.checked = False
        self._tip(self.openEditorCheck, "Open Slicer's Segment Editor after generating masks for manual cleanup.")
        form.addRow("Open Segment Editor", self.openEditorCheck)

        expert = ctk.ctkCollapsibleButton()
        expert.text = "Expert Settings"
        expert.collapsed = True
        generate_layout.addWidget(expert)
        expert_form = qt.QFormLayout(expert)

        self.trabThresholdSpin = self._double_spin(0, 5000, 1, 320.0)
        self.cortThresholdSpin = self._double_spin(0, 5000, 1, 450.0)
        self.gaussSigmaSpin = self._double_spin(0, 10, 2, 0.8)
        self._tip(self.trabThresholdSpin, "Trabecular threshold used by Gaussian/adaptive segmentation support generation.")
        self._tip(self.cortThresholdSpin, "Cortical threshold used by Gaussian/adaptive segmentation support generation.")
        self._tip(self.gaussSigmaSpin, "Gaussian smoothing sigma applied before threshold-based segmentation.")
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
        self._tip(self.adaptiveLowSpin, "Lower local threshold for adaptive bone segmentation.")
        self._tip(self.adaptiveHighSpin, "Upper local threshold for adaptive bone segmentation.")
        self._tip(self.adaptiveBlockSpin, "Odd local window size for adaptive thresholding.")
        expert_form.addRow("Adaptive low", self.adaptiveLowSpin)
        expert_form.addRow("Adaptive high", self.adaptiveHighSpin)
        expert_form.addRow("Adaptive block size", self.adaptiveBlockSpin)

        self.lhThresholdSpin = self._double_spin(0, 100000, 1, 15564.0)
        self.lhBackendCombo = qt.QComboBox()
        for label, value in [("CPU", "cpu"), ("Auto", "auto"), ("Torch MPS", "torch_mps")]:
            self.lhBackendCombo.addItem(label, value)
        self._tip(self.lhThresholdSpin, "Laplace-Hamming threshold in native Scanco attenuation units.")
        self._tip(self.lhBackendCombo, "Execution backend for Laplace-Hamming support calculation when available.")
        expert_form.addRow("LH threshold", self.lhThresholdSpin)
        expert_form.addRow("LH backend", self.lhBackendCombo)

        self.minSizeSpin = qt.QSpinBox()
        self.minSizeSpin.minimum = 0
        self.minSizeSpin.maximum = 1000000
        self.minSizeSpin.value = 64
        self.keepLargestCheck = qt.QCheckBox()
        self.keepLargestCheck.checked = True
        self._tip(self.minSizeSpin, "Remove connected bone components smaller than this voxel count.")
        self._tip(self.keepLargestCheck, "Keep only the largest connected segmentation component.")
        expert_form.addRow("Min component voxels", self.minSizeSpin)
        expert_form.addRow("Keep largest", self.keepLargestCheck)

        self.geodesicBoneThresholdSpin = self._double_spin(0, 5000, 1, 250.0)
        self.geodesicFillHolesCheck = qt.QCheckBox()
        self.geodesicFillHolesCheck.checked = True
        self._tip(self.geodesicBoneThresholdSpin, "Threshold passed to the geodesic fracture periosteal contour.")
        self._tip(self.geodesicFillHolesCheck, "Fill holes inside the geodesic full-mask contour.")
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
        self._tip(self.trabCloseSpin, "Morphological close radius for trabecular compartment cleanup.")
        self._tip(self.outerKernelSpin, "Kernel size for periosteal contour smoothing/refinement.")
        self._tip(self.innerKernelSpin, "Kernel size for endosteal contour smoothing/refinement.")
        self._tip(self.outerOpenSpin, "Opening radius for full-mask contour cleanup.")
        self._tip(self.peelSpin, "Number of voxels peeled near the cortex for trabecular mask separation.")
        expert_form.addRow("Trab close radius", self.trabCloseSpin)
        expert_form.addRow("Periosteal kernel", self.outerKernelSpin)
        expert_form.addRow("Endosteal kernel", self.innerKernelSpin)
        expert_form.addRow("Periosteal open radius", self.outerOpenSpin)
        expert_form.addRow("Peel", self.peelSpin)

        self.createButton = qt.QPushButton("Generate Masks And Segmentation")
        self.createButton.clicked.connect(self._create_segmentation)
        self._tip(self.createButton, "Run contour and segmentation generation with the selected methods and expert settings.")
        form.addRow(self.createButton)

        self.openEditorButton = qt.QPushButton("Open Segment Editor")
        self.openEditorButton.clicked.connect(self._open_segment_editor)
        self._tip(self.openEditorButton, "Open Slicer's Segment Editor for manual inspection or cleanup of generated segments.")
        form.addRow(self.openEditorButton)
        generate_layout.addStretch(1)
        self.toolTabs.addTab(generate_tab, "Generate")
        self._build_derive_labels_tab()

    def _labelmap_selector(self):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"]
        selector.selectNodeUponCreation = False
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = True
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def _build_derive_labels_tab(self):
        derive_tab = qt.QWidget()
        derive_layout = qt.QVBoxLayout(derive_tab)
        form = qt.QFormLayout()
        derive_layout.addLayout(form)

        self.materialSegSelector = self._labelmap_selector()
        self.materialTrabSelector = self._labelmap_selector()
        self.materialCortSelector = self._labelmap_selector()
        self.materialFullSelector = self._labelmap_selector()
        self._tip(self.materialSegSelector, "Bone segmentation labelmap used to restrict HOM material labels to segmented bone voxels.")
        self._tip(self.materialTrabSelector, "Trabecular compartment mask. Can be paired with full or cortical mask to derive the missing compartment.")
        self._tip(self.materialCortSelector, "Cortical compartment mask. Can be paired with full or trabecular mask to derive the missing compartment.")
        self._tip(self.materialFullSelector, "Full periosteal mask. Can be paired with trabecular or cortical mask to derive the missing compartment.")
        form.addRow("Bone segmentation", self.materialSegSelector)
        form.addRow("Trabecular mask", self.materialTrabSelector)
        form.addRow("Cortical mask", self.materialCortSelector)
        form.addRow("Full mask", self.materialFullSelector)

        missing = ctk.ctkCollapsibleButton()
        missing.text = "Generate Missing Mask"
        missing.collapsed = False
        derive_layout.addWidget(missing)
        missing_form = qt.QFormLayout(missing)
        self.missingMaskRoleCombo = qt.QComboBox()
        for label, value in [
            ("Auto", "auto"),
            ("Full", "full"),
            ("Trabecular", "trab"),
            ("Cortical", "cort"),
        ]:
            self.missingMaskRoleCombo.addItem(label, value)
        self.missingMaskOutputNameEdit = qt.QLineEdit("HRpQCT_derived_mask")
        self.generateMissingMaskButton = qt.QPushButton("Generate Missing Mask")
        self.generateMissingMaskButton.clicked.connect(self._generate_missing_mask)
        self._tip(self.missingMaskRoleCombo, "Choose the missing mask to generate, or Auto when exactly one of full/trab/cort is missing.")
        self._tip(self.missingMaskOutputNameEdit, "Name for the generated labelmap node.")
        self._tip(self.generateMissingMaskButton, "Create full, trabecular, or cortical mask from the other two compartment masks.")
        missing_form.addRow("Output role", self.missingMaskRoleCombo)
        missing_form.addRow("Output name", self.missingMaskOutputNameEdit)
        missing_form.addRow(self.generateMissingMaskButton)

        hom = ctk.ctkCollapsibleButton()
        hom.text = "HOM Material Labels"
        hom.collapsed = False
        derive_layout.addWidget(hom)
        hom_form = qt.QFormLayout(hom)
        self.materialTrabLabelSpin = qt.QSpinBox()
        self.materialTrabLabelSpin.minimum = 1
        self.materialTrabLabelSpin.maximum = 255
        self.materialTrabLabelSpin.value = 126
        self.materialCortLabelSpin = qt.QSpinBox()
        self.materialCortLabelSpin.minimum = 1
        self.materialCortLabelSpin.maximum = 255
        self.materialCortLabelSpin.value = 127
        self._tip(self.materialTrabLabelSpin, "Label value assigned to segmented trabecular bone voxels in the HOM/material labelmap.")
        self._tip(self.materialCortLabelSpin, "Label value assigned to segmented cortical bone voxels in the HOM/material labelmap.")
        hom_form.addRow("Trab label", self.materialTrabLabelSpin)
        hom_form.addRow("Cort label", self.materialCortLabelSpin)

        self.materialOutputNameEdit = qt.QLineEdit("HRpQCT_HOM_material_labels")
        self._tip(self.materialOutputNameEdit, "Name for the generated HOM/material labelmap node.")
        hom_form.addRow("Output name", self.materialOutputNameEdit)

        self.createMaterialLabelsButton = qt.QPushButton("Create HOM Material Labels")
        self.createMaterialLabelsButton.clicked.connect(self._create_material_labels)
        self._tip(self.createMaterialLabelsButton, "Create one material labelmap from bone segmentation plus any two compartment masks.")
        hom_form.addRow(self.createMaterialLabelsButton)

        operations = ctk.ctkCollapsibleButton()
        operations.text = "Mask Operations"
        operations.collapsed = True
        derive_layout.addWidget(operations)
        operations_form = qt.QFormLayout(operations)
        self.maskASelector = self._labelmap_selector()
        self.maskBSelector = self._labelmap_selector()
        self.maskOperationCombo = qt.QComboBox()
        for label, value in [
            ("Union", "union"),
            ("Intersection", "intersection"),
            ("A minus B", "difference"),
            ("XOR", "xor"),
        ]:
            self.maskOperationCombo.addItem(label, value)
        self.maskOperationOutputNameEdit = qt.QLineEdit("HRpQCT_mask_operation")
        self.createMaskOperationButton = qt.QPushButton("Create Mask Operation")
        self.createMaskOperationButton.clicked.connect(self._create_mask_operation)
        self._tip(self.maskASelector, "First input mask for boolean operations.")
        self._tip(self.maskBSelector, "Second input mask for boolean operations.")
        self._tip(self.maskOperationCombo, "Boolean mask operation. A minus B keeps voxels in Mask A that are not in Mask B.")
        self._tip(self.maskOperationOutputNameEdit, "Name for the generated mask operation labelmap.")
        self._tip(self.createMaskOperationButton, "Create a new labelmap from the selected boolean mask operation.")
        operations_form.addRow("Mask A", self.maskASelector)
        operations_form.addRow("Mask B", self.maskBSelector)
        operations_form.addRow("Operation", self.maskOperationCombo)
        operations_form.addRow("Output name", self.maskOperationOutputNameEdit)
        operations_form.addRow(self.createMaskOperationButton)

        relabel = ctk.ctkCollapsibleButton()
        relabel.text = "Relabel And Validate"
        relabel.collapsed = True
        derive_layout.addWidget(relabel)
        relabel_form = qt.QFormLayout(relabel)
        self.relabelSourceSelector = self._labelmap_selector()
        self.relabelValueSpin = qt.QSpinBox()
        self.relabelValueSpin.minimum = 1
        self.relabelValueSpin.maximum = 65535
        self.relabelValueSpin.value = 126
        self.relabelOutputNameEdit = qt.QLineEdit("HRpQCT_relabelled")
        self.relabelButton = qt.QPushButton("Relabel Nonzero Voxels")
        self.relabelButton.clicked.connect(self._relabel_mask)
        self.validateMasksButton = qt.QPushButton("Validate Mask Set")
        self.validateMasksButton.clicked.connect(self._validate_mask_set)
        self.countMasksButton = qt.QPushButton("Count Selected Masks")
        self.countMasksButton.clicked.connect(self._count_selected_masks)
        self._tip(self.relabelSourceSelector, "Source labelmap whose nonzero voxels will be assigned a single output label.")
        self._tip(self.relabelValueSpin, "Output label value for all nonzero source voxels.")
        self._tip(self.relabelOutputNameEdit, "Name for the relabelled output labelmap.")
        self._tip(self.relabelButton, "Create a copy where every nonzero source voxel has the selected label value.")
        self._tip(self.validateMasksButton, "Check full/trab/cort consistency: overlap, outside-full voxels, and missing compartment voxels.")
        self._tip(self.countMasksButton, "Report nonzero voxel counts for the selected bone, full, trabecular, and cortical masks.")
        relabel_form.addRow("Source", self.relabelSourceSelector)
        relabel_form.addRow("Label", self.relabelValueSpin)
        relabel_form.addRow("Output name", self.relabelOutputNameEdit)
        relabel_form.addRow(self.relabelButton)
        relabel_form.addRow(self.validateMasksButton)
        relabel_form.addRow(self.countMasksButton)

        derive_layout.addStretch(1)
        self.toolTabs.addTab(derive_tab, "Derive Labels")

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

    def _generate_missing_mask(self):
        try:
            node, counts = self.logic.create_missing_mask_volume(
                full_mask_node=self.materialFullSelector.currentNode(),
                trab_mask_node=self.materialTrabSelector.currentNode(),
                cort_mask_node=self.materialCortSelector.currentNode(),
                output_role=str(self.missingMaskRoleCombo.currentData),
                output_name=self.missingMaskOutputNameEdit.text.strip() or "HRpQCT_derived_mask",
            )
            self._log(f"Created {node.GetName()}. Role={counts['role']}, voxels={counts['voxels']}.")
        except Exception as exc:
            self._error(exc)

    def _create_material_labels(self):
        try:
            node, counts = self.logic.create_material_label_volume(
                self.materialSegSelector.currentNode(),
                self.materialTrabSelector.currentNode(),
                self.materialCortSelector.currentNode(),
                self.materialFullSelector.currentNode(),
                trab_label=int(self.materialTrabLabelSpin.value),
                cort_label=int(self.materialCortLabelSpin.value),
                output_name=self.materialOutputNameEdit.text.strip() or "HRpQCT_HOM_material_labels",
            )
            self._log(
                f"Created {node.GetName()}. Material voxels: "
                f"trab={counts.get('trab')}, cort={counts.get('cort')} "
                f"({counts.get('cort_source')})."
            )
        except Exception as exc:
            self._error(exc)

    def _create_mask_operation(self):
        try:
            node, counts = self.logic.create_boolean_mask_volume(
                self.maskASelector.currentNode(),
                self.maskBSelector.currentNode(),
                str(self.maskOperationCombo.currentData),
                output_name=self.maskOperationOutputNameEdit.text.strip() or "HRpQCT_mask_operation",
            )
            self._log(f"Created {node.GetName()}. Operation={counts['operation']}, voxels={counts['voxels']}.")
        except Exception as exc:
            self._error(exc)

    def _relabel_mask(self):
        try:
            node, counts = self.logic.relabel_mask_volume(
                self.relabelSourceSelector.currentNode(),
                int(self.relabelValueSpin.value),
                output_name=self.relabelOutputNameEdit.text.strip() or "HRpQCT_relabelled",
            )
            self._log(f"Created {node.GetName()}. Label={counts['label']}, voxels={counts['voxels']}.")
        except Exception as exc:
            self._error(exc)

    def _validate_mask_set(self):
        try:
            counts = self.logic.validate_compartment_masks(
                full_mask_node=self.materialFullSelector.currentNode(),
                trab_mask_node=self.materialTrabSelector.currentNode(),
                cort_mask_node=self.materialCortSelector.currentNode(),
            )
            status = "valid" if counts["valid"] else "not valid"
            self._log(
                f"Mask set {status}. full={counts['full']}, trab={counts['trab']}, cort={counts['cort']}, "
                f"overlap={counts['overlap']}, outside_full={counts['outside_full']}, "
                f"full_not_compartment={counts['full_not_compartment']}."
            )
        except Exception as exc:
            self._error(exc)

    def _count_selected_masks(self):
        try:
            counts = self.logic.mask_voxel_counts(
                seg=self.materialSegSelector.currentNode(),
                full=self.materialFullSelector.currentNode(),
                trab=self.materialTrabSelector.currentNode(),
                cort=self.materialCortSelector.currentNode(),
            )
            self._log("Voxel counts: " + ", ".join(f"{role}={count}" for role, count in counts.items()) + ".")
        except Exception as exc:
            self._error(exc)

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
            elif segmentation_method == "laplace_hamming":
                progress_dialog = self._create_busy_progress_dialog(
                    "Running Laplace-Hamming bone segmentation...",
                    "Laplace-Hamming Segmentation",
                )
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
                    create_labelmaps=False,
                    open_segment_editor=bool(self.openEditorCheck.checked),
                    params=self._collect_params(),
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                )
            finally:
                if progress_dialog is not None:
                    progress_dialog.close()
            counts = metadata.get("voxel_counts", {})
            label_text = ""
            provenance_text = f" Method={metadata.get('segmentation_method')}."
            if metadata.get("segmentation_method") == "laplace_hamming":
                provenance_text = (
                    f" Method=laplace_hamming; input={metadata.get('segmentation_input_unit')} "
                    f"via {metadata.get('segmentation_input_reader')}."
                )
            self._log(
                f"Created {segmentation_node.GetName()}.{label_text} "
                f"Voxel counts: full={counts.get('full')}, trab={counts.get('trab')}, "
                f"cort={counts.get('cort')}, seg={counts.get('seg')}.{provenance_text}"
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

    def _create_busy_progress_dialog(self, message, title):
        dialog = qt.QProgressDialog(
            str(message),
            None,
            0,
            0,
            slicer.util.mainWindow(),
        )
        dialog.setWindowTitle(str(title))
        dialog.setWindowModality(qt.Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setCancelButton(None)
        dialog.show()
        slicer.app.processEvents()
        return dialog

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
