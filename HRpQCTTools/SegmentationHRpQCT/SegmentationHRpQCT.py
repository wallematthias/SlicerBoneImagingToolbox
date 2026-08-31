import tempfile
import sys
import importlib
import inspect
import json
import os
import re
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import ctk
import numpy as np
import qt
import slicer
import SimpleITK as sitk
import vtk

_TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(_TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLBOX_ROOT))

def _resolve_sibling_repo(repo_name):
    candidates = [
        _TOOLBOX_ROOT.parent / repo_name,
        _TOOLBOX_ROOT.parent.parent / repo_name,
        _TOOLBOX_ROOT.parent.parent.parent / repo_name,
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() or (candidate / "setup.py").exists():
            return candidate
    return candidates[0]


_GEODESIC_CONTOUR_LOCAL_REPO = _resolve_sibling_repo("hrpqct-geodesic-contour")
_GEODESIC_CONTOUR_LOCAL_SRC = _GEODESIC_CONTOUR_LOCAL_REPO / "src"
if _GEODESIC_CONTOUR_LOCAL_SRC.exists() and str(_GEODESIC_CONTOUR_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(_GEODESIC_CONTOUR_LOCAL_SRC))
_BONE_CONTOURING_LOCAL_REPO = _resolve_sibling_repo("bone-contouring")
_BONE_CONTOURING_LOCAL_SRC = _BONE_CONTOURING_LOCAL_REPO / "src"
if _BONE_CONTOURING_LOCAL_SRC.exists() and str(_BONE_CONTOURING_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(_BONE_CONTOURING_LOCAL_SRC))
_SCANCO_IO_DIR = _TOOLBOX_ROOT / "IOTools" / "ScancoIO"
if _SCANCO_IO_DIR.exists() and str(_SCANCO_IO_DIR) not in sys.path:
    sys.path.insert(0, str(_SCANCO_IO_DIR))

from SlicerBoneImagingToolboxLib.segmentation_methods import (
    BONE_SEGMENTATION_METHODS,
    ENDOSTEAL_CONTOUR_METHODS,
    PERIOSTEAL_CONTOUR_METHODS,
    method_supports_site,
    selected_parameter_groups,
)

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
BONE_CONTOURING_PIP_CONSTRAINTS = ("numpy>=1.26,<3.0", "SimpleITK>=2.3")
SCENE_CONTOUR_DEBUG_ENV = "SLICER_BONE_CONTOUR_DEBUG"
SCENE_CONTOUR_DEBUG_DIR_ENV = "SLICER_BONE_CONTOUR_DEBUG_DIR"
SCENE_CONTOUR_DEBUG_DEFAULT_DIR = Path("/Users/matthias.walle/Documents/10_Data/STRAMBO/test/tmp-debug")

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
            "use_adaptive_threshold": False,
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

SEGMENTATION_METHODS = set(BONE_SEGMENTATION_METHODS)
PERIOSTEAL_CONTOUR_METHOD_IDS = set(PERIOSTEAL_CONTOUR_METHODS)
ENDOSTEAL_CONTOUR_METHOD_IDS = set(ENDOSTEAL_CONTOUR_METHODS)


def _image_output_stem(path):
    path = Path(path)
    name = path.name
    upper = name.upper()
    aim_index = upper.find(".AIM")
    if aim_index >= 0:
        return name[:aim_index]
    lower = name.lower()
    for suffix in (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _is_aim_path(path):
    return ".aim" in Path(path).name.lower()


def _truthy_env(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "debug"}


def _scene_contour_debug_dir():
    if not _truthy_env(os.environ.get(SCENE_CONTOUR_DEBUG_ENV)):
        return None
    return Path(os.environ.get(SCENE_CONTOUR_DEBUG_DIR_ENV) or SCENE_CONTOUR_DEBUG_DEFAULT_DIR)


def _sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._") or "scene_volume"


def _clean_method_label(label):
    text = str(label)
    for prefix in (
        "XCT2 - ",
        "XCT2 ",
        "XCT 2 - ",
        "XCT 2 ",
        "XtremeCT II - ",
        "XtremeCT II ",
        "XCT1 - ",
        "XCT1 ",
        "XCT 1 - ",
        "XCT 1 ",
        "XtremeCT I - ",
        "XtremeCT I ",
    ):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _mask_sidecar_path(mask_path):
    mask_path = Path(mask_path)
    if mask_path.name.lower().endswith(".nii.gz"):
        return mask_path.with_name(mask_path.name[:-7] + ".json")
    return mask_path.with_suffix(".json")


def _set_slicer_volume_geometry_from_sitk_image(node, image):
    direction_lps = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin_lps = np.asarray(image.GetOrigin(), dtype=float)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    ijk_to_ras = np.eye(4, dtype=float)
    ijk_to_ras[:3, :3] = lps_to_ras @ direction_lps @ np.diag(spacing)
    ijk_to_ras[:3, 3] = lps_to_ras @ origin_lps
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for row in range(4):
        for column in range(4):
            matrix.SetElement(row, column, float(ijk_to_ras[row, column]))
    node.SetIJKToRASMatrix(matrix)
    node.Modified()


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


class SegmentationHRpQCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Bone Contours"
        parent.categories = ["Bone Imaging.Segmentation Methods"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Generate bone full, trabecular, cortical, and binary segmentation "
            f"masks using site presets and standard segmentation methods. Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = "Part of the Bone Imaging Toolbox for 3D Slicer."


class SegmentationHRpQCTLogic(ScriptedLoadableModuleLogic):
    def is_pipeline_available(self):
        try:
            import bone_contouring  # noqa: F401

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
        if _BONE_CONTOURING_LOCAL_REPO.exists():
            slicer.util.pip_install(f"--no-deps -e {_BONE_CONTOURING_LOCAL_REPO}")
        else:
            packages = " ".join(["bone-contouring", *BONE_CONTOURING_PIP_CONSTRAINTS])
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
        self._lastProcessingImageReader = "selected_slicer_volume"
        with tempfile.TemporaryDirectory(prefix="hrpqct_seg_in_") as temp_dir:
            path = Path(temp_dir) / "input.nrrd"
            if not slicer.util.saveNode(volume_node, str(path)):
                raise RuntimeError("Could not save selected Slicer volume for processing.")
            selected_image = sitk.ReadImage(str(path))
        source_path = self._volume_source_aim_path(volume_node)
        if source_path is not None and source_path.exists():
            try:
                from ScancoIOLib import aim_io

                image, _metadata = aim_io.read_aim(source_path, scaling="density")
                if image.GetSize() == selected_image.GetSize():
                    image.CopyInformation(selected_image)
                    self._lastProcessingImageReader = "py_aimio_density"
                    return image
            except Exception:
                pass
        self._lastProcessingImageReader = "selected_slicer_volume"
        return selected_image

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

    def _aim_metadata_from_source(self, source_path):
        try:
            from ScancoIOLib import aim_io

            _image, metadata = aim_io.read_aim(source_path, scaling="native")
            return metadata if isinstance(metadata, dict) else {}
        except Exception:
            return {}

    def _write_mask_aim_if_supported(self, mask_image, source_path, output_path, metadata=None, role=None):
        source_path = Path(source_path)
        if ".aim" not in source_path.name.lower():
            return None
        try:
            from ScancoIOLib import aim_io

            metadata = dict(metadata if metadata is not None else self._aim_metadata_from_source(source_path))
            metadata["source_file"] = str(source_path)
            if role:
                metadata["mask_role"] = str(role)
            aim_io.write_aim(
                sitk.Cast(mask_image > 0, sitk.sitkUInt8),
                Path(output_path),
                metadata=metadata,
                unit="native",
                mask=True,
            )
            return str(output_path)
        except Exception as exc:
            return {"error": str(exc)}

    def _write_mask_sidecar(self, mask_path, *, role, source_path, site, segmentation_method, periosteal_method, endosteal_method, output_format, params, metadata, source_metadata):
        sidecar_path = _mask_sidecar_path(mask_path)
        sidecar = {
            "schema": "bone-contour-mask-provenance-v1",
            "role": str(role),
            "mask_path": str(mask_path),
            "source_image": str(source_path),
            "site": str(site),
            "segmentation_method": str(segmentation_method),
            "periosteal_contour_method": str(periosteal_method),
            "endosteal_contour_method": str(endosteal_method),
            "output_format": str(output_format),
            "parameters": dict(params or {}),
            "algorithm_metadata": dict(metadata or {}),
            "source_metadata": dict(source_metadata or {}),
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return str(sidecar_path)

    def read_mask_image_file(self, mask_path):
        mask_path = Path(mask_path)
        if _is_aim_path(mask_path):
            from ScancoIOLib import aim_io

            image, metadata = aim_io.read_aim(mask_path, scaling="native")
            return sitk.Cast(image > 0, sitk.sitkUInt8), metadata
        return sitk.Cast(sitk.ReadImage(str(mask_path)) > 0, sitk.sitkUInt8), {}

    def _volume_source_aim_path(self, volume_node):
        if volume_node is None:
            return None
        source_path = volume_node.GetAttribute(AIM_SOURCE_ATTRIBUTE)
        if not source_path:
            storage_node = volume_node.GetStorageNode()
            if storage_node is not None:
                file_name = storage_node.GetFileName()
                if file_name and ".aim" in str(file_name).lower():
                    source_path = file_name
        if not source_path:
            return None
        source_path = Path(source_path)
        if ".aim" not in source_path.name.lower():
            return None
        return source_path

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

        source_path = self._volume_source_aim_path(volume_node)
        if source_path is not None and source_path.exists():
            return self._read_laplace_hamming_native_aim(source_path, reference_image), {
                "segmentation_input_unit": "scanco_native_int16",
                "segmentation_input_reader": "py_aimio_native_int16",
                "segmentation_input_path": str(source_path),
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

        if not source_path:
            raise ValueError(
                "Laplace-Hamming segmentation needs the original AIM source. "
                "Load the image with the Scanco I/O module first so scanner-source metadata is attached, "
                "or run Batch mode directly on AIM files so the source path can be attached automatically."
            )
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
            loaded = slicer.util.loadLabelVolume(str(path), {"name": name})
        if isinstance(loaded, tuple):
            success, label_node = loaded
        else:
            success, label_node = bool(loaded), loaded
        if not success or label_node is None:
            raise RuntimeError(f"Could not load generated labelmap: {name}")
        label_node.CopyOrientation(reference_node)
        self._copy_aim_attributes(reference_node, label_node)
        return label_node

    def _finalize_segment(self, segmentation_node, segment_id, segment_name, role=None):
        segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
        if segment is None:
            return
        segment.SetName(str(segment_name))
        if role is not None and hasattr(segment, "SetTag"):
            segment.SetTag("HRpQCT.Role", str(role))

    def _add_labelmap_segment(self, label_node, segmentation_node, segment_name, role=None):
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            label_node,
            segmentation_node,
        )
        segmentation = segmentation_node.GetSegmentation()
        if segmentation.GetNumberOfSegments() > 0:
            segment_id = segmentation.GetNthSegmentID(segmentation.GetNumberOfSegments() - 1)
            self._finalize_segment(segmentation_node, segment_id, segment_name, role)

    def _add_sitk_segment(self, image, segmentation_node, segment_name, reference_node, role=None):
        array_zyx = sitk.GetArrayFromImage(image).astype(np.uint8, copy=False)
        array_zyx = (array_zyx > 0).astype(np.uint8, copy=False)
        segment_id = segmentation_node.GetSegmentation().AddEmptySegment(segment_name)
        self._finalize_segment(segmentation_node, segment_id, segment_name, role)
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            array_zyx,
            segmentation_node,
            segment_id,
            reference_node,
        )
        self._finalize_segment(segmentation_node, segment_id, segment_name, role)

    def _configure_segmentation_display(self, segmentation_node):
        display_node = segmentation_node.GetDisplayNode()
        if display_node is None:
            return
        display_node.SetVisibility(True)
        display_node.SetVisibility2DFill(True)
        display_node.SetVisibility2DOutline(True)
        if hasattr(display_node, "SetAllSegmentsVisibility2DFill"):
            display_node.SetAllSegmentsVisibility2DFill(True)
        if hasattr(display_node, "SetAllSegmentsVisibility2DOutline"):
            display_node.SetAllSegmentsVisibility2DOutline(True)
        if hasattr(display_node, "SetOpacity"):
            display_node.SetOpacity(0.5)
        if hasattr(display_node, "SetOpacity2DFill"):
            display_node.SetOpacity2DFill(0.85)
        if hasattr(display_node, "SetOpacity2DOutline"):
            display_node.SetOpacity2DOutline(1.0)
        if hasattr(display_node, "SetOpacity3D"):
            display_node.SetOpacity3D(0.4)
        if hasattr(display_node, "SetAllSegmentsOpacity2DFill"):
            display_node.SetAllSegmentsOpacity2DFill(0.85)
        if hasattr(display_node, "SetAllSegmentsOpacity2DOutline"):
            display_node.SetAllSegmentsOpacity2DOutline(1.0)

    def _remove_empty_duplicate_segmentation_nodes(self, segmentation_node):
        nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
        nodes.UnRegister(None)
        for index in range(nodes.GetNumberOfItems()):
            node = nodes.GetItemAsObject(index)
            if (
                node is not None
                and node is not segmentation_node
                and node.IsA("vtkMRMLSegmentationNode")
                and node.GetName() == segmentation_node.GetName()
                and node.GetSegmentation().GetNumberOfSegments() == 0
            ):
                slicer.mrmlScene.RemoveNode(node)

    def _geodesic_full_mask_xyz(
        self,
        image,
        *,
        params=None,
        progress_callback=None,
        cancel_callback=None,
        debug_output_dir=None,
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

    def _import_bone_contouring(self):
        import importlib

        module = sys.modules.get("bone_contouring")
        if _BONE_CONTOURING_LOCAL_SRC.exists():
            for name in list(sys.modules):
                if name == "bone_contouring" or name.startswith("bone_contouring."):
                    del sys.modules[name]
            return importlib.import_module("bone_contouring")
        if module is not None:
            module_path = Path(getattr(module, "__file__", "")).resolve()
            try:
                module_path.relative_to(_TOOLBOX_ROOT.resolve())
                return module
            except ValueError:
                for name in list(sys.modules):
                    if name == "bone_contouring" or name.startswith("bone_contouring."):
                        del sys.modules[name]
        return importlib.import_module("bone_contouring")

    def _empty_mask_like(self, reference_image):
        empty = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8)
        empty.CopyInformation(reference_image)
        return empty

    def _mask_voxel_count(self, mask_image):
        return int(sitk.GetArrayFromImage(mask_image).astype(bool, copy=False).sum())

    def _write_debug_sitk_image(self, image, path, *, binary=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if binary:
            sitk.WriteImage(sitk.Cast(image > 0, sitk.sitkUInt8), str(path))
        else:
            sitk.WriteImage(image, str(path))
        return str(path)

    def _write_scene_debug_artifacts(
        self,
        debug_dir,
        *,
        prefix,
        source_path,
        image,
        segmentation_input_image,
        generated,
        roles,
        config,
        metadata,
    ):
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = _sanitize_filename(prefix)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = debug_dir / f"{safe_prefix}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "processing_image": self._write_debug_sitk_image(image, run_dir / f"{safe_prefix}_processing-density.nrrd"),
            "masks": {},
        }
        source_metadata = self._aim_metadata_from_source(source_path) if source_path and _is_aim_path(source_path) else {}
        write_aim_masks = bool(source_path and _is_aim_path(source_path))
        if segmentation_input_image is not None:
            artifacts["laplace_hamming_input"] = self._write_debug_sitk_image(
                segmentation_input_image,
                run_dir / f"{safe_prefix}_laplace-hamming-native-input.nrrd",
            )
            if write_aim_masks:
                try:
                    from ScancoIOLib import aim_io

                    lh_input_aim = run_dir / f"{safe_prefix}_laplace-hamming-native-input.AIM"
                    aim_io.write_aim(
                        sitk.Cast(segmentation_input_image, sitk.sitkInt16),
                        lh_input_aim,
                        metadata={
                            **source_metadata,
                            "source_file": str(source_path),
                            "debug_role": "laplace_hamming_native_input",
                        },
                        unit="native",
                        mask=False,
                    )
                    artifacts["laplace_hamming_input_aim"] = str(lh_input_aim)
                except Exception as exc:
                    artifacts["laplace_hamming_input_aim_error"] = str(exc)

        for role in roles:
            mask_image = getattr(generated, role)
            if write_aim_masks:
                aim_result = self._write_mask_aim_if_supported(
                    mask_image,
                    source_path,
                    run_dir / f"{safe_prefix}_mask-{role}.AIM",
                    metadata=source_metadata,
                    role=role,
                )
                if isinstance(aim_result, dict) or aim_result is None:
                    artifacts["masks"][role] = self._write_debug_sitk_image(
                        mask_image,
                        run_dir / f"{safe_prefix}_mask-{role}.nii.gz",
                        binary=True,
                    )
                else:
                    artifacts["masks"][role] = aim_result
            else:
                artifacts["masks"][role] = self._write_debug_sitk_image(
                    mask_image,
                    run_dir / f"{safe_prefix}_mask-{role}.nii.gz",
                    binary=True,
                )

        manifest_path = run_dir / f"{safe_prefix}_scene_contour_debug.json"
        payload = {
            "schema": "bone-contour-scene-debug-v1",
            "source_image": str(source_path) if source_path else None,
            "config": dict(config or {}),
            "algorithm_metadata": dict(metadata or {}),
            "artifacts": artifacts,
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return str(manifest_path)

    def _generate_bone_masks_with_bone_contouring(
        self,
        volume_node,
        image,
        *,
        site,
        segmentation_method,
        periosteal_contour_method,
        endosteal_contour_method,
        output_prefix=None,
        create_labelmaps=False,
        open_segment_editor=False,
        params=None,
        debug_output_dir=None,
    ):
        bone_contouring = self._import_bone_contouring()
        ContourParameters = bone_contouring.ContourParameters
        InnerContourParameters = bone_contouring.InnerContourParameters
        OuterContourParameters = bone_contouring.OuterContourParameters
        SegmentationParameters = bone_contouring.SegmentationParameters
        generate_masks_from_image = bone_contouring.generate_masks_from_image

        params = dict(params or {})
        site_defaults = SITE_PRESETS.get(str(site), SITE_PRESETS["radius"])
        method_defaults = METHOD_PRESETS.get(segmentation_method, METHOD_PRESETS["seg_gauss"])

        inner_params = dict(site_defaults["inner"])
        outer_params = dict(site_defaults["outer"])
        segmentation_params = dict(method_defaults)
        inner_params.update(params.get("inner", {}))
        outer_params.update(params.get("outer", {}))
        segmentation_params.update(params.get("segmentation", {}))

        def _int_param(source, new_key, legacy_key, default):
            return int(source.get(new_key, source.get(legacy_key, default)))

        def _float_param(source, key, default):
            return float(source.get(key, default))

        def _bool_param(source, key, default):
            return bool(source.get(key, default))

        outer_method = "geodesic" if periosteal_contour_method == "geodesic_fracture" else periosteal_contour_method
        inner_method = endosteal_contour_method
        segmentation_package_method = {
            "seg_gauss": "gauss",
            "none": "gauss",
        }.get(segmentation_method, segmentation_method)
        geodesic_params = dict(params.get("geodesic", {}))

        contour_params = ContourParameters(
            modality=str(params.get("modality", "xct2")),
            site=str(site),
            outer=OuterContourParameters(
                contour_method=outer_method,
                periosteal_threshold=_float_param(outer_params, "periosteal_threshold", 300.0),
                periosteal_kernel_size=_int_param(outer_params, "periosteal_kernel_size", "periosteal_kernelsize", 5),
                periosteal_open_radius=_int_param(outer_params, "periosteal_open_radius", "periosteal_openradius", 2),
                gaussian_sigma=_float_param(outer_params, "gaussian_sigma", 1.5),
                use_adaptive_threshold=_bool_param(outer_params, "use_adaptive_threshold", False),
                fill_holes=_bool_param(outer_params, "fill_holes", True),
                geodesic_bone_threshold=float(
                    geodesic_params.get("bone_threshold", outer_params.get("geodesic_bone_threshold", 250.0))
                ),
                geodesic_fill_holes=bool(
                    geodesic_params.get("fill_holes", outer_params.get("geodesic_fill_holes", True))
                ),
            ),
            inner=InnerContourParameters(
                contour_method=inner_method,
                site=str(site),
                endosteal_threshold=_float_param(inner_params, "endosteal_threshold", 500.0),
                endosteal_kernel_size=_int_param(inner_params, "endosteal_kernel_size", "endosteal_kernelsize", 3),
                gaussian_sigma=_float_param(inner_params, "gaussian_sigma", 1.5),
                use_adaptive_threshold=_bool_param(inner_params, "use_adaptive_threshold", False),
                peel=_int_param(inner_params, "peel", "peel", 3),
                trabecular_close_radius=inner_params.get("trabecular_close_radius"),
            ),
            segmentation=SegmentationParameters(
                enabled=segmentation_method != "none",
                method=segmentation_package_method,
                gaussian_sigma=_float_param(segmentation_params, "gaussian_sigma", 0.8),
                trab_threshold=_float_param(segmentation_params, "trab_threshold", 320.0),
                cort_threshold=_float_param(segmentation_params, "cort_threshold", 450.0),
                adaptive_low_threshold=_float_param(segmentation_params, "adaptive_low_threshold", 100.0),
                adaptive_high_threshold=_float_param(segmentation_params, "adaptive_high_threshold", 300.0),
                adaptive_block_size=_int_param(segmentation_params, "adaptive_block_size", "adaptive_block_size", 13),
                min_size_voxels=_int_param(segmentation_params, "min_size_voxels", "min_size_voxels", 64),
                keep_largest_component=_bool_param(segmentation_params, "keep_largest_component", True),
                laplace_hamming_low_pass_cutoff=_float_param(
                    segmentation_params, "laplace_hamming_low_pass_cutoff", 0.3
                ),
                laplace_hamming_high_pass_cutoff=_float_param(
                    segmentation_params, "laplace_hamming_high_pass_cutoff", 0.0
                ),
                laplace_hamming_threshold=_float_param(segmentation_params, "laplace_hamming_threshold", 15564.0),
                laplace_hamming_epsilon=_float_param(segmentation_params, "laplace_hamming_epsilon", 0.45),
                laplace_hamming_amplitude=_float_param(segmentation_params, "laplace_hamming_amplitude", 1.0),
                laplace_hamming_amplification=_float_param(segmentation_params, "laplace_hamming_amplification", 1.0),
                laplace_hamming_input_offset=_float_param(segmentation_params, "laplace_hamming_input_offset", 0.0),
                laplace_hamming_ipl_float_max=_float_param(
                    segmentation_params, "laplace_hamming_ipl_float_max", 200000.0
                ),
                laplace_hamming_int16_max=_float_param(segmentation_params, "laplace_hamming_int16_max", 32767.0),
                laplace_hamming_min_size_voxels=_int_param(
                    segmentation_params, "laplace_hamming_min_size_voxels", "laplace_hamming_min_size_voxels", 70
                ),
                laplace_hamming_backend=str(segmentation_params.get("laplace_hamming_backend", "cpu")),
                use_segmentation_aligned_contour_support=_bool_param(
                    segmentation_params, "use_segmentation_aligned_contour_support", False
                ),
            ),
        )

        segmentation_image = None
        segmentation_source_meta = {}
        if segmentation_method == "laplace_hamming":
            segmentation_image, segmentation_source_meta = self._laplace_hamming_support_image(volume_node, image)

        generated = generate_masks_from_image(image, contour_params, segmentation_image=segmentation_image)
        generated.metadata.update(segmentation_source_meta)

        periosteal_contour_generated = periosteal_contour_method != "none"
        compartment_split_generated = endosteal_contour_method == "standard"
        if segmentation_method == "none":
            generated.seg = self._empty_mask_like(image)
        if not compartment_split_generated:
            generated.trab = self._empty_mask_like(image)
            generated.cort = self._empty_mask_like(image)
        if not periosteal_contour_generated:
            generated.full = self._empty_mask_like(image)

        generated.metadata.update(
            {
                "segmentation_method": segmentation_method,
                "processing_image_reader": str(getattr(self, "_lastProcessingImageReader", "selected_slicer_volume")),
                "bone_contouring_path": str(Path(getattr(bone_contouring, "__file__", "")).resolve()),
                "segmentation_aligned_contour_support": bool(
                    contour_params.segmentation.use_segmentation_aligned_contour_support
                ),
                "periosteal_contour_method": periosteal_contour_method,
                "endosteal_contour_method": endosteal_contour_method,
                "internal_periosteal_contour_method": periosteal_contour_method,
                "internal_endosteal_contour_method": endosteal_contour_method,
                "periosteal_contour_generated": bool(periosteal_contour_generated),
                "compartment_split_generated": bool(compartment_split_generated),
                "voxel_counts": {
                    "seg": self._mask_voxel_count(generated.seg),
                    "full": self._mask_voxel_count(generated.full),
                    "trab": self._mask_voxel_count(generated.trab),
                    "cort": self._mask_voxel_count(generated.cort),
                },
            }
        )
        if not compartment_split_generated:
            generated.metadata["compartment_split_reason"] = "endosteal_contour_method_none"
        if segmentation_method == "seg_gauss" and not compartment_split_generated:
            trab_threshold = float(contour_params.segmentation.trab_threshold)
            generated.metadata["segmentation_warning"] = (
                "No cortical mask was provided; Gaussian segmentation used the trabecular threshold "
                f"{trab_threshold:g} globally."
            )
            generated.metadata["segmentation_threshold_applied_global"] = trab_threshold
        if segmentation_method == "laplace_hamming" and generated.metadata["voxel_counts"]["seg"] == 0:
            raise RuntimeError(
                "Laplace-Hamming produced an empty bone segmentation. "
                "Check that the selected volume has valid AIM calibration metadata "
                "or an original AIM source, and that LH parameters match native Scanco units."
            )

        if debug_output_dir is not None:
            prefix = output_prefix.strip() if output_prefix else volume_node.GetName()
            debug_config = {
                "site": str(site),
                "segmentation_method": str(segmentation_method),
                "periosteal_contour_method": str(periosteal_contour_method),
                "endosteal_contour_method": str(endosteal_contour_method),
                "parameters": dict(params or {}),
            }
            generated.metadata["scene_debug_manifest"] = self._write_scene_debug_artifacts(
                debug_output_dir,
                prefix=prefix,
                source_path=self._volume_source_aim_path(volume_node),
                image=image,
                segmentation_input_image=segmentation_image,
                generated=generated,
                roles=["full", "trab", "cort", "seg"],
                config=debug_config,
                metadata=generated.metadata,
            )

        prefix = output_prefix.strip() if output_prefix else volume_node.GetName()
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            f"{prefix}_HRpQCT_segmentation",
        )
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
        segmentation_node.CreateDefaultDisplayNodes()
        self._configure_segmentation_display(segmentation_node)
        self._copy_aim_attributes(volume_node, segmentation_node)
        for key in (
            "segmentation_method",
            "processing_image_reader",
            "bone_contouring_path",
            "segmentation_input_unit",
            "segmentation_input_reader",
            "segmentation_input_path",
            "segmentation_aligned_contour_support",
            "segmentation_warning",
            "segmentation_threshold_applied_global",
            "periosteal_contour_method",
            "periosteal_contour_generated",
            "periosteal_contour_reason",
            "endosteal_contour_method",
            "internal_periosteal_contour_method",
            "internal_endosteal_contour_method",
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
            self._add_sitk_segment(image_out, segmentation_node, segment_name, volume_node, role)
            if create_labelmaps:
                label_node = self._sitk_to_labelmap(image_out, f"{prefix}_{role}", volume_node)
                outputs[role] = label_node

        self._remove_empty_duplicate_segmentation_nodes(segmentation_node)
        if open_segment_editor:
            slicer.util.selectModule("SegmentEditor")
        return segmentation_node, outputs, generated.metadata

    def generate_bone_masks(
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
        debug_output_dir=None,
    ):
        image = self._volume_to_sitk(volume_node)
        if segmentation_method is None:
            segmentation_method = "seg_gauss" if method is None else str(method)
        segmentation_method = str(segmentation_method)
        periosteal_contour_method = str(periosteal_contour_method)
        endosteal_contour_method = str(endosteal_contour_method)
        requested_periosteal_contour_method = periosteal_contour_method
        requested_endosteal_contour_method = endosteal_contour_method
        if segmentation_method not in SEGMENTATION_METHODS:
            raise ValueError(f"Unsupported bone segmentation method: {segmentation_method}")
        if periosteal_contour_method not in PERIOSTEAL_CONTOUR_METHOD_IDS:
            raise ValueError(f"Unsupported periosteal contour method: {periosteal_contour_method}")
        if endosteal_contour_method not in ENDOSTEAL_CONTOUR_METHOD_IDS:
            raise ValueError(f"Unsupported endosteal contour method: {endosteal_contour_method}")
        if periosteal_contour_method == "none" and endosteal_contour_method == "standard":
            raise ValueError("Standard endosteal contour requires a periosteal contour.")
        if not method_supports_site(PERIOSTEAL_CONTOUR_METHODS[periosteal_contour_method], site):
            raise ValueError(f"{PERIOSTEAL_CONTOUR_METHODS[periosteal_contour_method].label} only supports knee scans.")
        if periosteal_contour_method != "none":
            return self._generate_bone_masks_with_bone_contouring(
                volume_node,
                image,
                site=site,
                segmentation_method=segmentation_method,
                periosteal_contour_method=periosteal_contour_method,
                endosteal_contour_method=endosteal_contour_method,
                output_prefix=output_prefix,
                create_labelmaps=create_labelmaps,
                open_segment_editor=open_segment_editor,
                params=params,
                debug_output_dir=debug_output_dir,
            )

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
        site_defaults = SITE_PRESETS.get(str(site), SITE_PRESETS["radius"])
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
        compartment_split_requested = endosteal_contour_method == "standard"

        contour_params = ContourGenerationParams(
            outer=OuterContourParams(**outer_params),
            inner=InnerContourParams(**inner_params),
            segmentation=SegmentationParams(**segmentation_params),
        )
        outer_options = asdict(contour_params.outer)
        inner_options = asdict(contour_params.inner)
        segmentation_support_params = contour_params.segmentation
        use_aligned_support = bool(segmentation_support_params.use_segmentation_aligned_contour_support)

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
            contour_support_source = (
                segmentation_image
                if use_aligned_support and segmentation_image is not None
                else image
            )
            contour_support_image_xyz = sitk_to_numpy_xyz(contour_support_source)
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
                    contour_support_image_xyz,
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
                contour_support_image_xyz,
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
            global_threshold_without_compartments = (
                segmentation_method == "seg_gauss" and not compartment_split_requested
            )
            if segmentation_method == "none":
                seg_xyz = np.zeros_like(full_xyz, dtype=bool)
            elif (
                use_aligned_support
                and segmentation_method == "laplace_hamming"
                and inner_support_xyz is not None
            ):
                seg_xyz = _ensure_bool(inner_support_xyz) & full_xyz
            elif (
                use_aligned_support
                and segmentation_method == "adaptive"
                and inner_support_xyz is not None
            ):
                seg_xyz = _ensure_bool(inner_support_xyz) & full_xyz
            elif global_threshold_without_compartments:
                trab_threshold = float(segmentation_support_params.trab_threshold)
                seg_xyz = (segmentation_image_xyz >= trab_threshold) & full_xyz
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
                    "segmentation_aligned_contour_support": bool(use_aligned_support),
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
            if global_threshold_without_compartments:
                generated.metadata["segmentation_warning"] = (
                    "No cortical mask was provided; Gaussian segmentation used the trabecular threshold "
                    f"{trab_threshold:g} globally."
                )
                generated.metadata["segmentation_threshold_applied_global"] = float(trab_threshold)

        periosteal_contour_generated = periosteal_contour_method != "none"
        compartment_split_generated = compartment_split_requested
        if not compartment_split_generated:
            full_xyz = sitk_to_numpy_xyz(generated.full) > 0
            empty_xyz = np.zeros_like(full_xyz, dtype=bool)
            generated.trab = numpy_xyz_to_sitk_binary(empty_xyz, image)
            generated.cort = numpy_xyz_to_sitk_binary(empty_xyz, image)

        if (
            use_aligned_support
            and segmentation_method == "laplace_hamming"
            and segmentation_image is not None
        ):
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
        generated.metadata["processing_image_reader"] = str(
            getattr(self, "_lastProcessingImageReader", "selected_slicer_volume")
        )
        generated.metadata["segmentation_aligned_contour_support"] = bool(use_aligned_support)
        generated.metadata["periosteal_contour_method"] = requested_periosteal_contour_method
        generated.metadata["endosteal_contour_method"] = requested_endosteal_contour_method
        generated.metadata["internal_periosteal_contour_method"] = periosteal_contour_method
        generated.metadata["internal_endosteal_contour_method"] = endosteal_contour_method
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

        if debug_output_dir is not None:
            prefix = output_prefix.strip() if output_prefix else volume_node.GetName()
            debug_config = {
                "site": str(site),
                "segmentation_method": str(segmentation_method),
                "periosteal_contour_method": str(requested_periosteal_contour_method),
                "endosteal_contour_method": str(requested_endosteal_contour_method),
                "parameters": dict(params or {}),
            }
            generated.metadata["scene_debug_manifest"] = self._write_scene_debug_artifacts(
                debug_output_dir,
                prefix=prefix,
                source_path=self._volume_source_aim_path(volume_node),
                image=image,
                segmentation_input_image=segmentation_image,
                generated=generated,
                roles=["full", "trab", "cort", "seg"],
                config=debug_config,
                metadata=generated.metadata,
            )
        prefix = output_prefix.strip() if output_prefix else volume_node.GetName()
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            f"{prefix}_HRpQCT_segmentation",
        )
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
        segmentation_node.CreateDefaultDisplayNodes()
        self._configure_segmentation_display(segmentation_node)
        self._copy_aim_attributes(volume_node, segmentation_node)
        for key in (
            "segmentation_method",
            "processing_image_reader",
            "segmentation_input_unit",
            "segmentation_input_reader",
            "segmentation_input_path",
            "segmentation_aligned_contour_support",
            "segmentation_warning",
            "segmentation_threshold_applied_global",
            "periosteal_contour_method",
            "periosteal_contour_generated",
            "periosteal_contour_reason",
            "endosteal_contour_method",
            "internal_periosteal_contour_method",
            "internal_endosteal_contour_method",
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
            self._add_sitk_segment(image_out, segmentation_node, segment_name, volume_node, role)
            if create_labelmaps:
                label_node = self._sitk_to_labelmap(image_out, f"{prefix}_{role}", volume_node)
                outputs[role] = label_node

        self._remove_empty_duplicate_segmentation_nodes(segmentation_node)

        if open_segment_editor:
            slicer.util.selectModule("SegmentEditor")

        return segmentation_node, outputs, generated.metadata

    def generate_hrpqct_masks(self, *args, **kwargs):
        return self.generate_bone_masks(*args, **kwargs)

    def write_bone_mask_files(
        self,
        image_path,
        output_dir,
        *,
        site,
        segmentation_method,
        periosteal_contour_method,
        endosteal_contour_method,
        output_prefix=None,
        output_format="auto",
        keep_loaded=False,
        params=None,
    ):
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        volume_node = None
        segmentation_node = None
        try:
            volume_node = slicer.util.loadVolume(str(image_path), returnNode=True)[1]
            if ".aim" in image_path.name.lower():
                volume_node.SetAttribute(AIM_SOURCE_ATTRIBUTE, str(image_path))
            prefix = output_prefix or _image_output_stem(image_path)
            segmentation_node, outputs, metadata = self.generate_bone_masks(
                volume_node,
                site=site,
                segmentation_method=segmentation_method,
                periosteal_contour_method=periosteal_contour_method,
                endosteal_contour_method=endosteal_contour_method,
                output_prefix=prefix,
                create_labelmaps=True,
                open_segment_editor=False,
                params=params,
            )
            written = {}
            aim_written = {}
            sidecars = {}
            is_aim_input = _is_aim_path(image_path)
            output_format = str(output_format or "auto").strip().lower()
            if output_format not in {"auto", "aim", "nifti"}:
                raise ValueError("Output format must be one of: auto, aim, nifti.")
            write_aim_output = output_format == "aim" or (output_format == "auto" and is_aim_input)
            if write_aim_output and not is_aim_input:
                raise ValueError("AIM output requires an AIM input so source scanner metadata can be preserved.")
            source_aim_metadata = self._aim_metadata_from_source(image_path) if is_aim_input else None
            stem = _image_output_stem(image_path)
            for role in metadata.get("emitted_roles", []):
                label_node = outputs.get(role)
                if label_node is None:
                    raise RuntimeError(f"Generated {role} mask labelmap was not returned by the contour pipeline.")
                mask_image = self._volume_to_sitk(label_node)
                if write_aim_output:
                    aim_result = self._write_mask_aim_if_supported(
                        mask_image,
                        image_path,
                        output_dir / f"{stem}_mask-{role}.AIM",
                        metadata=source_aim_metadata,
                        role=role,
                    )
                    if isinstance(aim_result, dict):
                        raise RuntimeError(f"Could not write AIM {role} mask: {aim_result['error']}")
                    aim_written[role] = aim_result
                    written[role] = aim_result
                    mask_path = Path(aim_result)
                else:
                    out_path = output_dir / f"{stem}_mask-{role}.nii.gz"
                    slicer.util.saveNode(label_node, str(out_path))
                    written[role] = str(out_path)
                    mask_path = out_path
                sidecars[role] = self._write_mask_sidecar(
                    mask_path,
                    role=role,
                    source_path=image_path,
                    site=site,
                    segmentation_method=segmentation_method,
                    periosteal_method=periosteal_contour_method,
                    endosteal_method=endosteal_contour_method,
                    output_format="aim" if write_aim_output else "nifti",
                    params=params,
                    metadata=metadata,
                    source_metadata=source_aim_metadata or {},
                )
                if not keep_loaded:
                    slicer.mrmlScene.RemoveNode(label_node)
            if aim_written:
                metadata["aim_outputs"] = aim_written
                metadata["output_format"] = "aim"
            else:
                metadata["output_format"] = "nifti"
            metadata["provenance_sidecars"] = sidecars
            return metadata, written
        finally:
            if segmentation_node is not None and not keep_loaded:
                slicer.mrmlScene.RemoveNode(segmentation_node)
            if volume_node is not None and not keep_loaded:
                slicer.mrmlScene.RemoveNode(volume_node)


class SegmentationHRpQCTWidget(ScriptedLoadableModuleWidget):
    def _tip(self, widget, text):
        widget.toolTip = str(text)
        return widget

    def setup(self):
        super().setup()
        self.logic = SegmentationHRpQCTLogic()
        self._geodesic_cancel_requested = False
        self._batchImagePaths = []
        self._batchImageRows = []
        self._batchRowOutputs = {}
        self._batchQueue = []
        self._batchQueueRunning = False
        self._batchProcess = None
        self._batchRunningRow = None
        self._batchProcessStdout = ""
        self._batchProcessStderr = ""
        self._batchCancelRequested = False
        self._build_segmentation_section()
        self._build_log_section()
        self.layout.addStretch(1)
        self._apply_modality_preset()
        self._apply_preset_values(update_segmentation_method=False)
        self._refresh_parameter_mode_ui()
        self._refresh_method_dependent_ui()
        self._update_dependency_ui()
        self._log("Ready.")

    def _build_segmentation_section(self):
        self.toolTabs = qt.QTabWidget()
        self.layout.addWidget(self.toolTabs)

        generate_tab = qt.QWidget()
        generate_layout = qt.QVBoxLayout(generate_tab)
        form = qt.QFormLayout()
        self._generateForm = form
        self._extraInputRows = {}
        self._topRows = {}
        generate_layout.addLayout(form)

        self.pipelineStatusLabel = qt.QLabel()
        self._tip(self.pipelineStatusLabel, "Shows whether the core contouring packages are available in Slicer Python.")
        form.addRow("Status", self.pipelineStatusLabel)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = False
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self._tip(self.volumeSelector, "Input volume used to generate masks and bone segmentation.")
        form.addRow("Input volume", self.volumeSelector)

        self.parameterModeCombo = qt.QComboBox()
        for label, value in [("Preset", "preset"), ("Custom", "custom")]:
            self.parameterModeCombo.addItem(label, value)
        self.parameterModeCombo.currentIndexChanged.connect(self._on_parameter_mode_changed)
        self.parameterModeCombo.currentIndexChanged.connect(self._refresh_parameter_mode_ui)
        self.parameterModeCombo.currentIndexChanged.connect(self._update_batch_options_summary)
        self._tip(
            self.parameterModeCombo,
            "Preset shows modality/site controls and applies defaults. Custom hides preset controls and preserves expert settings.",
        )
        form.addRow("Parameters", self.parameterModeCombo)

        self.modalityCombo = qt.QComboBox()
        for label, value in [("XtremeCT I", "xct1"), ("XtremeCT II", "xct2")]:
            self.modalityCombo.addItem(label, value)
        self.modalityCombo.currentIndexChanged.connect(self._on_modality_changed)
        self.modalityCombo.currentIndexChanged.connect(self._update_batch_options_summary)
        self._tip(self.modalityCombo, "Applies scanner-specific defaults while keeping the segmentation method names general.")
        form.addRow("Modality preset", self.modalityCombo)
        self._topRows["modality"] = (form.labelForField(self.modalityCombo), self.modalityCombo)

        self.siteCombo = qt.QComboBox()
        for label, value in [("Auto", "auto"), ("Radius", "radius"), ("Tibia", "tibia"), ("Knee", "knee")]:
            self.siteCombo.addItem(label, value)
        self.siteCombo.currentIndexChanged.connect(self._apply_site_preset)
        self.siteCombo.currentIndexChanged.connect(self._refresh_method_dependent_ui)
        self.siteCombo.currentIndexChanged.connect(self._update_batch_options_summary)
        self._tip(self.siteCombo, "Auto detects radius, tibia, or knee from loaded/discovered filenames; concrete sites apply that site everywhere.")
        form.addRow("Site preset", self.siteCombo)
        self._topRows["site"] = (form.labelForField(self.siteCombo), self.siteCombo)

        self.segmentationMethodCombo = qt.QComboBox()
        for value, descriptor in BONE_SEGMENTATION_METHODS.items():
            self.segmentationMethodCombo.addItem(_clean_method_label(descriptor.label), value)
        self.segmentationMethodCombo.currentIndexChanged.connect(self._on_segmentation_method_changed)
        self.segmentationMethodCombo.currentIndexChanged.connect(self._refresh_method_dependent_ui)
        self.segmentationMethodCombo.currentIndexChanged.connect(self._update_batch_options_summary)
        self._tip(
            self.segmentationMethodCombo,
            "Bone binarization method. Laplace-Hamming uses native Scanco attenuation values from AIM metadata/source.",
        )
        form.addRow("Bone segmentation", self.segmentationMethodCombo)

        self.periostealContourCombo = qt.QComboBox()
        for value, descriptor in PERIOSTEAL_CONTOUR_METHODS.items():
            self.periostealContourCombo.addItem(_clean_method_label(descriptor.label), value)
        self.periostealContourCombo.currentIndexChanged.connect(self._refresh_method_dependent_ui)
        self.periostealContourCombo.currentIndexChanged.connect(self._update_batch_options_summary)
        self._tip(self.periostealContourCombo, "Outer contour method for the full bone mask.")
        form.addRow("Periosteal (outer) contour", self.periostealContourCombo)

        self.endostealContourCombo = qt.QComboBox()
        for value, descriptor in ENDOSTEAL_CONTOUR_METHODS.items():
            self.endostealContourCombo.addItem(_clean_method_label(descriptor.label), value)
        self.endostealContourCombo.currentIndexChanged.connect(self._refresh_method_dependent_ui)
        self.endostealContourCombo.currentIndexChanged.connect(self._update_batch_options_summary)
        self._tip(self.endostealContourCombo, "Inner contour method used to split full mask into trabecular and cortical compartments.")
        form.addRow("Endosteal (inner) contour", self.endostealContourCombo)

        self.outputPrefixEdit = qt.QLineEdit()
        self._tip(self.outputPrefixEdit, "Optional prefix for generated Slicer nodes. Leave empty to use the input volume name.")
        form.addRow("Output prefix", self.outputPrefixEdit)

        self.customRecipeRowWidget = qt.QWidget()
        custom_recipe_layout = qt.QHBoxLayout(self.customRecipeRowWidget)
        custom_recipe_layout.setContentsMargins(0, 0, 0, 0)
        self.loadRecipeButton = qt.QPushButton("Load Recipe")
        self.saveRecipeButton = qt.QPushButton("Save Recipe")
        self.loadRecipeButton.clicked.connect(self._load_custom_recipe)
        self.saveRecipeButton.clicked.connect(self._save_custom_recipe)
        self._tip(self.loadRecipeButton, "Load a saved custom contouring recipe JSON file.")
        self._tip(self.saveRecipeButton, "Save the current methods and expert settings as a reusable recipe JSON file.")
        custom_recipe_layout.addWidget(self.loadRecipeButton)
        custom_recipe_layout.addWidget(self.saveRecipeButton)
        form.addRow("Custom recipe", self.customRecipeRowWidget)
        self.customRecipeLabel = form.labelForField(self.customRecipeRowWidget)

        self.expertSettingsButton = ctk.ctkCollapsibleButton()
        self.expertSettingsButton.text = "Expert Settings"
        self.expertSettingsButton.collapsed = True
        generate_layout.addWidget(self.expertSettingsButton)
        self._expertRows = {}
        self._expertSections = {}

        expert_layout = qt.QVBoxLayout(self.expertSettingsButton)
        segmentation_expert = ctk.ctkCollapsibleButton()
        segmentation_expert.text = "Segmentation Settings"
        segmentation_expert.collapsed = False
        expert_layout.addWidget(segmentation_expert)
        segmentation_form = qt.QFormLayout(segmentation_expert)
        self._expertSections["Bone segmentation"] = segmentation_expert

        periosteal_expert = ctk.ctkCollapsibleButton()
        periosteal_expert.text = "Periosteal Contour Settings"
        periosteal_expert.collapsed = False
        expert_layout.addWidget(periosteal_expert)
        periosteal_form = qt.QFormLayout(periosteal_expert)
        self._expertSections["Periosteal contour"] = periosteal_expert

        endosteal_expert = ctk.ctkCollapsibleButton()
        endosteal_expert.text = "Endosteal Contour Settings"
        endosteal_expert.collapsed = False
        expert_layout.addWidget(endosteal_expert)
        endosteal_form = qt.QFormLayout(endosteal_expert)
        self._expertSections["Endosteal contour"] = endosteal_expert
        self._expertForm = segmentation_form

        self.trabThresholdSpin = self._double_spin(0, 5000, 1, 320.0)
        self.cortThresholdSpin = self._double_spin(0, 5000, 1, 450.0)
        self.gaussSigmaSpin = self._double_spin(0, 10, 2, 0.8)
        self._tip(self.trabThresholdSpin, "Trabecular threshold used by Gaussian/adaptive segmentation support generation.")
        self._tip(self.cortThresholdSpin, "Cortical threshold used by Gaussian/adaptive segmentation support generation.")
        self._tip(self.gaussSigmaSpin, "Gaussian smoothing sigma applied before threshold-based segmentation.")
        segmentation_form.addRow("Trab threshold", self.trabThresholdSpin)
        self._remember_expert_row("trab_threshold", self.trabThresholdSpin, form=segmentation_form, group="Bone segmentation")
        segmentation_form.addRow("Cort threshold", self.cortThresholdSpin)
        self._remember_expert_row("cort_threshold", self.cortThresholdSpin, form=segmentation_form, group="Bone segmentation")
        segmentation_form.addRow("Gaussian sigma", self.gaussSigmaSpin)
        self._remember_expert_row("gaussian_sigma", self.gaussSigmaSpin, form=segmentation_form, group="Bone segmentation")

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
        segmentation_form.addRow("Adaptive low", self.adaptiveLowSpin)
        self._remember_expert_row("adaptive_low_threshold", self.adaptiveLowSpin, form=segmentation_form, group="Bone segmentation")
        segmentation_form.addRow("Adaptive high", self.adaptiveHighSpin)
        self._remember_expert_row("adaptive_high_threshold", self.adaptiveHighSpin, form=segmentation_form, group="Bone segmentation")
        segmentation_form.addRow("Adaptive block size", self.adaptiveBlockSpin)
        self._remember_expert_row("adaptive_block_size", self.adaptiveBlockSpin, form=segmentation_form, group="Bone segmentation")

        self.lhThresholdSpin = self._double_spin(0, 100000, 1, 15564.0)
        self.lhBackendCombo = qt.QComboBox()
        for label, value in [("CPU", "cpu"), ("Auto", "auto"), ("Torch MPS", "torch_mps")]:
            self.lhBackendCombo.addItem(label, value)
        self._tip(self.lhThresholdSpin, "Laplace-Hamming threshold in native Scanco attenuation units.")
        self._tip(self.lhBackendCombo, "Execution backend for Laplace-Hamming support calculation when available.")
        segmentation_form.addRow("LH threshold", self.lhThresholdSpin)
        self._remember_expert_row("laplace_hamming_threshold", self.lhThresholdSpin, form=segmentation_form, group="Bone segmentation")
        segmentation_form.addRow("LH backend", self.lhBackendCombo)
        self._remember_expert_row("laplace_hamming_backend", self.lhBackendCombo, form=segmentation_form, group="Bone segmentation")

        self.lhLowPassSpin = self._double_spin(0, 1, 2, 0.3)
        self.lhEpsilonSpin = self._double_spin(0, 1, 2, 0.45)
        self.lhMinSizeSpin = qt.QSpinBox()
        self.lhMinSizeSpin.minimum = 0
        self.lhMinSizeSpin.maximum = 1000000
        self.lhMinSizeSpin.value = 70
        self._tip(self.lhLowPassSpin, "Laplace-Hamming low-pass cutoff for the frequency-domain Hamming filter.")
        self._tip(self.lhEpsilonSpin, "Laplace-Hamming edge-enhancement weight.")
        self._tip(self.lhMinSizeSpin, "Remove Laplace-Hamming components smaller than this voxel count.")
        segmentation_form.addRow("LH low-pass cutoff", self.lhLowPassSpin)
        self._remember_expert_row("laplace_hamming_low_pass_cutoff", self.lhLowPassSpin, form=segmentation_form, group="Bone segmentation")
        segmentation_form.addRow("LH epsilon", self.lhEpsilonSpin)
        self._remember_expert_row("laplace_hamming_epsilon", self.lhEpsilonSpin, form=segmentation_form, group="Bone segmentation")
        segmentation_form.addRow("LH min component voxels", self.lhMinSizeSpin)
        self._remember_expert_row("laplace_hamming_min_size_voxels", self.lhMinSizeSpin, form=segmentation_form, group="Bone segmentation")

        self.minSizeSpin = qt.QSpinBox()
        self.minSizeSpin.minimum = 0
        self.minSizeSpin.maximum = 1000000
        self.minSizeSpin.value = 64
        self.keepLargestCheck = qt.QCheckBox()
        self.keepLargestCheck.checked = True
        self._tip(self.minSizeSpin, "Remove connected bone components smaller than this voxel count.")
        self._tip(self.keepLargestCheck, "Keep only the largest connected segmentation component.")
        segmentation_form.addRow("Min component voxels", self.minSizeSpin)
        self._remember_expert_row("min_size_voxels", self.minSizeSpin, form=segmentation_form, group="Bone segmentation")
        segmentation_form.addRow("Keep largest", self.keepLargestCheck)
        self._remember_expert_row("keep_largest_component", self.keepLargestCheck, form=segmentation_form, group="Bone segmentation")

        self.segmentationAlignedSupportCheck = qt.QCheckBox()
        self.segmentationAlignedSupportCheck.checked = False
        self._tip(
            self.segmentationAlignedSupportCheck,
            "When enabled, periosteal/endosteal contour-support binarization follows the selected segmentation method. "
            "Leave this off to keep full/trab/cort masks more stable across scans.",
        )
        periosteal_form.addRow("Aligned contour support", self.segmentationAlignedSupportCheck)
        self._remember_expert_row(
            "segmentation_aligned_contour_support",
            self.segmentationAlignedSupportCheck,
            form=periosteal_form,
            group="Periosteal contour",
        )

        self.geodesicBoneThresholdSpin = self._double_spin(0, 5000, 1, 250.0)
        self.geodesicFillHolesCheck = qt.QCheckBox()
        self.geodesicFillHolesCheck.checked = True
        self._tip(self.geodesicBoneThresholdSpin, "Threshold passed to the geodesic fracture periosteal contour.")
        self._tip(self.geodesicFillHolesCheck, "Fill holes inside generated full/periosteal masks.")
        periosteal_form.addRow("Geodesic bone threshold", self.geodesicBoneThresholdSpin)
        self._remember_expert_row(
            "geodesic_bone_threshold", self.geodesicBoneThresholdSpin, form=periosteal_form, group="Periosteal contour"
        )
        periosteal_form.addRow("Fill full mask holes", self.geodesicFillHolesCheck)
        self._remember_expert_row("fill_holes", self.geodesicFillHolesCheck, form=periosteal_form, group="Periosteal contour")

        self.trabCloseSpin = qt.QSpinBox()
        self.trabCloseSpin.minimum = 0
        self.trabCloseSpin.maximum = 200
        self.trabCloseSpin.value = 25
        self.outerGaussSigmaSpin = self._double_spin(0, 10, 2, 1.5)
        self.innerGaussSigmaSpin = self._double_spin(0, 10, 2, 1.5)
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
        self._tip(self.outerGaussSigmaSpin, "Gaussian smoothing sigma applied before standard periosteal contour thresholding.")
        self._tip(self.innerGaussSigmaSpin, "Gaussian smoothing sigma applied before standard endosteal contour thresholding.")
        self._tip(self.outerKernelSpin, "Kernel size for periosteal contour smoothing/refinement.")
        self._tip(self.innerKernelSpin, "Kernel size for endosteal contour smoothing/refinement.")
        self._tip(self.outerOpenSpin, "Opening radius for full-mask contour cleanup.")
        self._tip(self.peelSpin, "Number of voxels peeled near the cortex for trabecular mask separation.")
        endosteal_form.addRow("Trab close radius", self.trabCloseSpin)
        self._remember_expert_row("trabecular_close_radius", self.trabCloseSpin, form=endosteal_form, group="Endosteal contour")
        periosteal_form.addRow("Periosteal Gaussian sigma", self.outerGaussSigmaSpin)
        self._remember_expert_row("outer_gaussian_sigma", self.outerGaussSigmaSpin, form=periosteal_form, group="Periosteal contour")
        endosteal_form.addRow("Endosteal Gaussian sigma", self.innerGaussSigmaSpin)
        self._remember_expert_row("inner_gaussian_sigma", self.innerGaussSigmaSpin, form=endosteal_form, group="Endosteal contour")
        periosteal_form.addRow("Periosteal kernel", self.outerKernelSpin)
        self._remember_expert_row("periosteal_kernelsize", self.outerKernelSpin, form=periosteal_form, group="Periosteal contour")
        endosteal_form.addRow("Endosteal kernel", self.innerKernelSpin)
        self._remember_expert_row("endosteal_kernelsize", self.innerKernelSpin, form=endosteal_form, group="Endosteal contour")
        periosteal_form.addRow("Periosteal open radius", self.outerOpenSpin)
        self._remember_expert_row("periosteal_open_radius", self.outerOpenSpin, form=periosteal_form, group="Periosteal contour")
        endosteal_form.addRow("Peel", self.peelSpin)
        self._remember_expert_row("peel", self.peelSpin, form=endosteal_form, group="Endosteal contour")

        self.periostealThresholdSpin = self._double_spin(0, 5000, 1, 300.0)
        self.endostealThresholdSpin = self._double_spin(0, 5000, 1, 500.0)
        self._tip(self.periostealThresholdSpin, "Threshold used by the standard periosteal contour.")
        self._tip(self.endostealThresholdSpin, "Threshold used by the standard endosteal contour.")
        periosteal_form.addRow("Periosteal threshold", self.periostealThresholdSpin)
        self._remember_expert_row("periosteal_threshold", self.periostealThresholdSpin, form=periosteal_form, group="Periosteal contour")
        endosteal_form.addRow("Endosteal threshold", self.endostealThresholdSpin)
        self._remember_expert_row("endosteal_threshold", self.endostealThresholdSpin, form=endosteal_form, group="Endosteal contour")

        self.createButton = qt.QPushButton("Generate")
        self.createButton.clicked.connect(self._create_segmentation)
        self._tip(self.createButton, "Run contour and segmentation generation with the selected methods and expert settings.")
        self.createButton.setStyleSheet(
            "QPushButton { background:#1f6feb; color:white; border:1px solid #175cc5; "
            "font-weight:600; padding:7px 14px; border-radius:4px; } "
            "QPushButton:hover { background:#1a5fd0; } "
            "QPushButton:pressed { background:#154ea8; } "
            "QPushButton:disabled { background:#9aaec8; border-color:#8fa2ba; }"
        )
        generate_layout.addWidget(self.createButton)
        self.toolTabs.addTab(generate_tab, "Scene")
        self._build_batch_tab()

    def _build_batch_tab(self):
        batch_tab = qt.QWidget()
        batch_layout = qt.QVBoxLayout(batch_tab)
        batch_form = qt.QFormLayout()
        batch_layout.addLayout(batch_form)

        self.batchInputRootEdit = ctk.ctkPathLineEdit()
        self.batchInputRootEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.batchInputRootEdit.currentPath = ""
        self._tip(self.batchInputRootEdit, "Folder containing images to process.")
        batch_form.addRow("Input folder", self.batchInputRootEdit)

        self.batchOutputRootEdit = ctk.ctkPathLineEdit()
        self.batchOutputRootEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.batchOutputRootEdit.currentPath = ""
        self._tip(self.batchOutputRootEdit, "Folder where generated mask labelmaps will be written.")
        if hasattr(self.batchOutputRootEdit, "currentPathChanged"):
            self.batchOutputRootEdit.currentPathChanged.connect(self._update_batch_options_summary)
        batch_form.addRow("Output folder", self.batchOutputRootEdit)

        batch_advanced = ctk.ctkCollapsibleButton()
        batch_advanced.text = "Advanced"
        batch_advanced.collapsed = True
        batch_layout.addWidget(batch_advanced)
        batch_advanced_form = qt.QFormLayout(batch_advanced)
        self.batchOutputFormatCombo = qt.QComboBox()
        for label, value in [("Auto", "auto"), ("AIM", "aim"), ("NIfTI", "nifti")]:
            self.batchOutputFormatCombo.addItem(label, value)
        self.batchOutputFormatCombo.currentIndexChanged.connect(self._update_batch_options_summary)
        self._tip(
            self.batchOutputFormatCombo,
            "Batch mask output format. Auto writes AIM masks for AIM inputs and NIfTI labelmaps for other image formats.",
        )
        batch_advanced_form.addRow("Output format", self.batchOutputFormatCombo)

        button_row_widget = qt.QWidget()
        button_row = qt.QHBoxLayout(button_row_widget)
        button_row.setContentsMargins(0, 0, 0, 0)
        self.batchDiscoverButton = qt.QPushButton("Discover Images")
        self.batchDiscoverButton.clicked.connect(self._discover_batch_images)
        button_row.addWidget(self.batchDiscoverButton)
        batch_form.addRow(button_row_widget)

        self.batchSummaryTable = qt.QTableWidget()
        self.batchSummaryTable.setColumnCount(6)
        self.batchSummaryTable.setHorizontalHeaderLabels(["Image", "Subject", "Session", "Site", "Action", "Status"])
        self.batchSummaryTable.setMaximumHeight(220)
        self.batchSummaryTable.setMinimumHeight(120)
        self.batchSummaryTable.horizontalHeader().setStretchLastSection(True)
        batch_layout.addWidget(self.batchSummaryTable)
        self.batchRunButton = qt.QPushButton("Run All")
        self.batchRunButton.clicked.connect(self._queue_all_batch_rows)
        self.batchRunButton.setStyleSheet(
            "QPushButton { background:#1f6feb; color:white; border:1px solid #175cc5; "
            "font-weight:600; padding:7px 14px; border-radius:4px; } "
            "QPushButton:hover { background:#1a5fd0; } "
            "QPushButton:pressed { background:#154ea8; } "
            "QPushButton:disabled { background:#9aaec8; border-color:#8fa2ba; }"
        )
        batch_layout.addWidget(self.batchRunButton)
        self.batchOptionsSummaryLabel = qt.QLabel()
        self.batchOptionsSummaryLabel.wordWrap = True
        self._tip(self.batchOptionsSummaryLabel, "Current scene settings that will be applied to queued batch rows.")
        batch_layout.addWidget(self.batchOptionsSummaryLabel)
        batch_layout.addStretch(1)
        self.toolTabs.addTab(batch_tab, "Batch")
        self._update_batch_options_summary()

    def _labelmap_selector(self):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"]
        selector.selectNodeUponCreation = False
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = True
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

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

    def _remember_expert_row(self, parameter_id, widget, *, form=None, group=None):
        label = (form or self._expertForm).labelForField(widget)
        self._expertRows[str(parameter_id)] = (label, widget, group)

    def _remember_extra_input_row(self, input_id, widget):
        label = self._generateForm.labelForField(widget)
        self._extraInputRows[str(input_id)] = (label, widget)

    def _set_widget_row_visible(self, widget, visible):
        widget.visible = bool(visible)
        parent = widget.parent()
        if hasattr(parent, "visible"):
            parent.visible = bool(visible)

    def _set_expert_row_visible(self, parameter_id, visible):
        row = self._expertRows.get(str(parameter_id))
        if not row:
            return
        label, widget, _group = row
        if label is not None:
            label.visible = bool(visible)
        widget.visible = bool(visible)

    def _combo_count(self, combo):
        count = combo.count
        return int(count() if callable(count) else count)

    def _combo_data(self, combo, default=None):
        if combo is None:
            return default
        data = combo.currentData
        if callable(data):
            data = data()
        return default if data is None else data

    def _combo_value_index(self, combo, value):
        for index in range(self._combo_count(combo)):
            if str(combo.itemData(index)) == str(value):
                return index
        return -1

    def _refresh_site_limited_combo(self, combo, descriptors):
        site = str(self.siteCombo.currentData)
        selected_value = str(combo.currentData)
        first_supported = -1
        selected_supported = True
        for index in range(self._combo_count(combo)):
            method_id = str(combo.itemData(index))
            supported = True if site == "auto" else method_supports_site(descriptors[method_id], site)
            model = combo.model()
            item = model.item(index) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(bool(supported))
                item.setToolTip("" if supported else "This method is only available for knee scans.")
            if supported and first_supported < 0:
                first_supported = index
            if method_id == selected_value:
                selected_supported = bool(supported)
        if not selected_supported and first_supported >= 0:
            combo.setCurrentIndex(first_supported)

    def _refresh_method_dependent_ui(self):
        if not hasattr(self, "_expertRows"):
            return
        self._refresh_site_limited_combo(self.periostealContourCombo, PERIOSTEAL_CONTOUR_METHODS)
        periosteal_method = str(self.periostealContourCombo.currentData)
        extra_inputs = PERIOSTEAL_CONTOUR_METHODS[periosteal_method].extra_inputs
        for input_id, (label, widget) in self._extraInputRows.items():
            visible = input_id in extra_inputs
            if label is not None:
                label.visible = bool(visible)
            widget.visible = bool(visible)

        groups = selected_parameter_groups(
            bone_method=str(self.segmentationMethodCombo.currentData),
            periosteal_method=periosteal_method,
            endosteal_method=str(self.endostealContourCombo.currentData),
        )
        visible_parameters = {parameter for parameters in groups.values() for parameter in parameters}
        for parameter_id in self._expertRows:
            self._set_expert_row_visible(parameter_id, parameter_id in visible_parameters)
        for group_name, section in getattr(self, "_expertSections", {}).items():
            group_parameters = set(groups.get(group_name, ()))
            section.visible = any(parameter in visible_parameters for parameter in group_parameters)

    def _refresh_parameter_mode_ui(self):
        if not hasattr(self, "parameterModeCombo"):
            return
        preset_mode = str(self._combo_data(self.parameterModeCombo, "preset")) == "preset"
        for label, widget in getattr(self, "_topRows", {}).values():
            if label is not None:
                label.visible = bool(preset_mode)
            widget.visible = bool(preset_mode)
        if hasattr(self, "customRecipeRowWidget"):
            self.customRecipeRowWidget.visible = not preset_mode
        if hasattr(self, "customRecipeLabel") and self.customRecipeLabel is not None:
            self.customRecipeLabel.visible = not preset_mode
        if hasattr(self, "expertSettingsButton") and not preset_mode:
            self.expertSettingsButton.collapsed = False

    def _build_log_section(self):
        self.messageLabel = qt.QLabel()
        self.messageLabel.wordWrap = True
        self.layout.addWidget(self.messageLabel)

    def _default_recipe_dir(self):
        path = Path.home() / ".slicerboneimagingtoolbox" / "bone-contour-recipes"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _set_combo_by_data(self, combo, value):
        count = combo.count
        if callable(count):
            count = count()
        for index in range(int(count)):
            if str(combo.itemData(index)) == str(value):
                combo.setCurrentIndex(index)
                return True
        return False

    def _on_parameter_mode_changed(self):
        self._apply_preset_values(update_segmentation_method=False)

    def _on_modality_changed(self):
        self._apply_preset_values(update_segmentation_method=True)

    def _on_segmentation_method_changed(self):
        self._apply_preset_values(update_segmentation_method=False)

    def _apply_preset_values(self, update_segmentation_method=False):
        if str(self._combo_data(self.parameterModeCombo, "preset")) != "preset":
            return
        if update_segmentation_method:
            self._apply_modality_preset()
        self._apply_segmentation_preset()
        self._apply_site_preset()

    def _apply_modality_preset(self):
        if not hasattr(self, "modalityCombo") or not hasattr(self, "segmentationMethodCombo"):
            return
        if not self._use_site_preset_params():
            return
        modality = str(self._combo_data(self.modalityCombo, "xct2"))
        target_method = "laplace_hamming" if modality == "xct1" else "seg_gauss"
        if str(self.segmentationMethodCombo.currentData) != target_method:
            self._set_combo_by_data(self.segmentationMethodCombo, target_method)

    def _save_custom_recipe(self):
        try:
            site = self._selected_site(volume_node=self.volumeSelector.currentNode(), strict=False)
            if site == "unparsed":
                site = str(self._combo_data(self.siteCombo, "auto"))
            recipe = {
                "schema": "bone-contour-recipe-v1",
                "modality": str(self._combo_data(self.modalityCombo, "xct2")),
                "site": str(site),
                "methods": {
                    "bone_segmentation": str(self.segmentationMethodCombo.currentData),
                    "periosteal_contour": str(self.periostealContourCombo.currentData),
                    "endosteal_contour": str(self.endostealContourCombo.currentData),
                },
                "parameters": self._collect_params(site=site, use_site_defaults=False),
            }
            default_path = str(self._default_recipe_dir() / "bone_contour_recipe.json")
            selected = qt.QFileDialog.getSaveFileName(
                slicer.util.mainWindow(),
                "Save Bone Contour Recipe",
                default_path,
                "JSON files (*.json)",
            )
            path_text = selected[0] if isinstance(selected, tuple) else selected
            if not path_text:
                return
            path = Path(str(path_text))
            if path.suffix.lower() != ".json":
                path = path.with_suffix(".json")
            path.write_text(json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8")
            self._log(f"Saved custom recipe: {path}")
        except Exception as exc:
            self._error(exc)

    def _load_custom_recipe(self):
        try:
            selected = qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(),
                "Load Bone Contour Recipe",
                str(self._default_recipe_dir()),
                "JSON files (*.json)",
            )
            path_text = selected[0] if isinstance(selected, tuple) else selected
            if not path_text:
                return
            path = Path(str(path_text))
            recipe = json.loads(path.read_text(encoding="utf-8"))
            self._apply_recipe(recipe)
            self._log(f"Loaded custom recipe: {path}")
        except Exception as exc:
            self._error(exc)

    def _apply_recipe(self, recipe):
        if recipe.get("schema") != "bone-contour-recipe-v1":
            raise ValueError("Recipe is not a bone-contour-recipe-v1 JSON file.")
        self._set_combo_by_data(self.parameterModeCombo, "custom")
        if recipe.get("modality"):
            self._set_combo_by_data(self.modalityCombo, recipe["modality"])
        if recipe.get("site"):
            self._set_combo_by_data(self.siteCombo, recipe["site"])
        methods = dict(recipe.get("methods") or {})
        if methods.get("bone_segmentation"):
            self._set_combo_by_data(self.segmentationMethodCombo, methods["bone_segmentation"])
        if methods.get("periosteal_contour"):
            self._set_combo_by_data(self.periostealContourCombo, methods["periosteal_contour"])
        if methods.get("endosteal_contour"):
            self._set_combo_by_data(self.endostealContourCombo, methods["endosteal_contour"])
        self._apply_params_to_widgets(recipe.get("parameters") or {})
        self._refresh_parameter_mode_ui()
        self._refresh_method_dependent_ui()
        self._update_batch_options_summary()

    def _apply_params_to_widgets(self, params):
        segmentation = dict(params.get("segmentation") or {})
        outer = dict(params.get("outer") or {})
        inner = dict(params.get("inner") or {})
        geodesic = dict(params.get("geodesic") or {})
        if "gaussian_sigma" in segmentation:
            self.gaussSigmaSpin.value = float(segmentation["gaussian_sigma"])
        if "trab_threshold" in segmentation:
            self.trabThresholdSpin.value = float(segmentation["trab_threshold"])
        if "cort_threshold" in segmentation:
            self.cortThresholdSpin.value = float(segmentation["cort_threshold"])
        if "adaptive_low_threshold" in segmentation:
            self.adaptiveLowSpin.value = float(segmentation["adaptive_low_threshold"])
        if "adaptive_high_threshold" in segmentation:
            self.adaptiveHighSpin.value = float(segmentation["adaptive_high_threshold"])
        if "adaptive_block_size" in segmentation:
            self.adaptiveBlockSpin.value = int(segmentation["adaptive_block_size"])
        if "min_size_voxels" in segmentation:
            self.minSizeSpin.value = int(segmentation["min_size_voxels"])
        if "keep_largest_component" in segmentation:
            self.keepLargestCheck.checked = bool(segmentation["keep_largest_component"])
        if "use_segmentation_aligned_contour_support" in segmentation:
            self.segmentationAlignedSupportCheck.checked = bool(segmentation["use_segmentation_aligned_contour_support"])
        if "laplace_hamming_threshold" in segmentation:
            self.lhThresholdSpin.value = float(segmentation["laplace_hamming_threshold"])
        if "laplace_hamming_low_pass_cutoff" in segmentation:
            self.lhLowPassSpin.value = float(segmentation["laplace_hamming_low_pass_cutoff"])
        if "laplace_hamming_epsilon" in segmentation:
            self.lhEpsilonSpin.value = float(segmentation["laplace_hamming_epsilon"])
        if "laplace_hamming_min_size_voxels" in segmentation:
            self.lhMinSizeSpin.value = int(segmentation["laplace_hamming_min_size_voxels"])
        if "laplace_hamming_backend" in segmentation:
            self._set_combo_by_data(self.lhBackendCombo, segmentation["laplace_hamming_backend"])
        if "periosteal_threshold" in outer:
            self.periostealThresholdSpin.value = float(outer["periosteal_threshold"])
        if "gaussian_sigma" in outer:
            self.outerGaussSigmaSpin.value = float(outer["gaussian_sigma"])
        if "periosteal_kernelsize" in outer:
            self.outerKernelSpin.value = int(outer["periosteal_kernelsize"])
        if "periosteal_open_radius" in outer:
            self.outerOpenSpin.value = int(outer["periosteal_open_radius"])
        if "fill_holes" in outer:
            self.geodesicFillHolesCheck.checked = bool(outer["fill_holes"])
        if "endosteal_threshold" in inner:
            self.endostealThresholdSpin.value = float(inner["endosteal_threshold"])
        if "gaussian_sigma" in inner:
            self.innerGaussSigmaSpin.value = float(inner["gaussian_sigma"])
        if "endosteal_kernelsize" in inner:
            self.innerKernelSpin.value = int(inner["endosteal_kernelsize"])
        if "peel" in inner:
            self.peelSpin.value = int(inner["peel"])
        if "trabecular_close_radius" in inner:
            self.trabCloseSpin.value = int(inner["trabecular_close_radius"])
        if "bone_threshold" in geodesic:
            self.geodesicBoneThresholdSpin.value = float(geodesic["bone_threshold"])

    def _apply_site_preset(self):
        if not hasattr(self, "siteCombo"):
            return
        if str(self._combo_data(self.parameterModeCombo, "preset")) != "preset":
            return
        site = str(self.siteCombo.currentData)
        if site not in SITE_PRESETS:
            site = self._selected_site(volume_node=self.volumeSelector.currentNode(), strict=False)
        if site not in SITE_PRESETS:
            site = "radius"
        preset = SITE_PRESETS[site]
        inner = preset["inner"]
        outer = preset["outer"]
        modality = str(self.modalityCombo.currentData) if hasattr(self, "modalityCombo") else "xct2"
        self.periostealThresholdSpin.value = float(outer["periosteal_threshold"])
        self.endostealThresholdSpin.value = float(inner["endosteal_threshold"])
        self.outerGaussSigmaSpin.value = float(outer["gaussian_sigma"])
        self.innerGaussSigmaSpin.value = float(inner["gaussian_sigma"])
        self.trabCloseSpin.value = int(inner["trabecular_close_radius"])
        self.segmentationAlignedSupportCheck.checked = True
        self.outerKernelSpin.value = 12 if modality == "xct1" else int(outer["periosteal_kernelsize"])
        self.innerKernelSpin.value = int(inner["endosteal_kernelsize"])
        self.outerOpenSpin.value = 1 if modality == "xct1" else int(outer["periosteal_open_radius"])
        if str(self.segmentationMethodCombo.currentData) == "laplace_hamming":
            self.lhThresholdSpin.value = self._lh_threshold(site, modality)
        self.peelSpin.value = int(inner["peel"])

    def _lh_threshold(self, site, modality):
        if str(modality) == "xct1" and str(site) in {"radius", "tibia"}:
            return 15000.0
        return 15564.0

    def _apply_segmentation_preset(self):
        if not hasattr(self, "segmentationMethodCombo"):
            return
        if str(self.segmentationMethodCombo.currentData) == "none":
            return
        preset = METHOD_PRESETS.get(self.segmentationMethodCombo.currentData)
        if not preset:
            return
        self.gaussSigmaSpin.value = float(preset["gaussian_sigma"])
        self.trabThresholdSpin.value = float(preset["trab_threshold"])
        self.cortThresholdSpin.value = float(preset["cort_threshold"])
        self.adaptiveLowSpin.value = float(preset["adaptive_low_threshold"])
        self.adaptiveHighSpin.value = float(preset["adaptive_high_threshold"])
        self.adaptiveBlockSpin.value = int(preset["adaptive_block_size"])
        self.minSizeSpin.value = int(preset["min_size_voxels"])
        self.keepLargestCheck.checked = bool(preset["keep_largest_component"])
        self.lhThresholdSpin.value = float(preset["laplace_hamming_threshold"])
        self.lhLowPassSpin.value = float(preset.get("laplace_hamming_low_pass_cutoff", 0.3))
        self.lhEpsilonSpin.value = float(preset.get("laplace_hamming_epsilon", 0.45))
        self.lhMinSizeSpin.value = int(preset.get("laplace_hamming_min_size_voxels", 70))
        backend = str(preset["laplace_hamming_backend"])
        count = self.lhBackendCombo.count
        if callable(count):
            count = count()
        for i in range(int(count)):
            if self.lhBackendCombo.itemData(i) == backend:
                self.lhBackendCombo.setCurrentIndex(i)
                break

    def _site_contour_params(self, site):
        preset = SITE_PRESETS[str(site)]
        inner = dict(preset["inner"])
        outer = dict(preset["outer"])
        modality = str(self.modalityCombo.currentData) if hasattr(self, "modalityCombo") else "xct2"
        if modality == "xct1":
            outer["periosteal_kernelsize"] = 12
            outer["periosteal_open_radius"] = 1
        return {
            "outer": {
                "periosteal_threshold": float(outer["periosteal_threshold"]),
                "gaussian_sigma": float(outer["gaussian_sigma"]),
                "periosteal_kernelsize": int(outer["periosteal_kernelsize"]),
                "periosteal_open_radius": int(outer["periosteal_open_radius"]),
                "use_adaptive_threshold": bool(outer["use_adaptive_threshold"]),
                "fill_holes": bool(self.geodesicFillHolesCheck.checked),
            },
            "inner": {
                "endosteal_threshold": float(inner["endosteal_threshold"]),
                "gaussian_sigma": float(inner["gaussian_sigma"]),
                "endosteal_kernelsize": int(inner["endosteal_kernelsize"]),
                "use_adaptive_threshold": bool(inner["use_adaptive_threshold"]),
                "peel": int(inner["peel"]),
                "trabecular_close_radius": int(inner["trabecular_close_radius"]),
            },
        }

    def _collect_params(self, site=None, use_site_defaults=False):
        block_size = int(self.adaptiveBlockSpin.value)
        if block_size % 2 == 0:
            block_size += 1
        modality = str(self.modalityCombo.currentData) if hasattr(self, "modalityCombo") else "xct2"
        selected_site = str(site) if site else str(self.siteCombo.currentData) if hasattr(self, "siteCombo") else "radius"
        lh_threshold = self._lh_threshold(selected_site, modality) if use_site_defaults else float(self.lhThresholdSpin.value)
        params = {
            "modality": modality,
            "segmentation": {
                "gaussian_sigma": float(self.gaussSigmaSpin.value),
                "trab_threshold": float(self.trabThresholdSpin.value),
                "cort_threshold": float(self.cortThresholdSpin.value),
                "adaptive_low_threshold": float(self.adaptiveLowSpin.value),
                "adaptive_high_threshold": float(self.adaptiveHighSpin.value),
                "adaptive_block_size": block_size,
                "min_size_voxels": int(self.minSizeSpin.value),
                "keep_largest_component": bool(self.keepLargestCheck.checked),
                "use_segmentation_aligned_contour_support": bool(self.segmentationAlignedSupportCheck.checked),
                "laplace_hamming_threshold": float(lh_threshold),
                "laplace_hamming_low_pass_cutoff": float(self.lhLowPassSpin.value),
                "laplace_hamming_epsilon": float(self.lhEpsilonSpin.value),
                "laplace_hamming_min_size_voxels": int(self.lhMinSizeSpin.value),
                "laplace_hamming_backend": str(self.lhBackendCombo.currentData),
            },
            "outer": {
                "periosteal_threshold": float(self.periostealThresholdSpin.value),
                "gaussian_sigma": float(self.outerGaussSigmaSpin.value),
                "periosteal_kernelsize": int(self.outerKernelSpin.value),
                "periosteal_open_radius": int(self.outerOpenSpin.value),
                "fill_holes": bool(self.geodesicFillHolesCheck.checked),
            },
            "inner": {
                "endosteal_threshold": float(self.endostealThresholdSpin.value),
                "gaussian_sigma": float(self.innerGaussSigmaSpin.value),
                "endosteal_kernelsize": int(self.innerKernelSpin.value),
                "peel": int(self.peelSpin.value),
                "trabecular_close_radius": int(self.trabCloseSpin.value),
            },
            "geodesic": {
                "bone_threshold": float(self.geodesicBoneThresholdSpin.value),
                "fill_holes": bool(self.geodesicFillHolesCheck.checked),
            },
        }
        if use_site_defaults and site in SITE_PRESETS:
            params.update(self._site_contour_params(site))
        return params

    def _is_batch_image_path(self, path):
        name = path.name.lower()
        if "_mask-" in name or "_seg" in name:
            return False
        if name.endswith((".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")):
            return True
        return path.name.upper().startswith("ISQ") or ".AIM" in path.name.upper()

    def _parse_batch_image_path(self, image_path):
        try:
            from timelapsedhrpqct.config.models import DiscoveryConfig
            from timelapsedhrpqct.dataset.filename_decoder import decode_filename

            decoded = decode_filename(Path(image_path), DiscoveryConfig())
            return {
                "subject": str(decoded.subject_id),
                "session": str(decoded.session_id),
                "site": str(decoded.site),
            }
        except Exception:
            return {"subject": "", "session": "", "site": ""}

    def _site_from_text(self, text):
        lower = str(text or "").lower()
        for site in SITE_PRESETS:
            if site in lower:
                return site
        for short, site in (("_rl", "radius"), ("_rr", "radius"), ("_tl", "tibia"), ("_tr", "tibia")):
            if short in lower:
                return site
        return ""

    def _resolve_site_from_volume(self, volume_node):
        if volume_node is None:
            return ""
        for value in (
            volume_node.GetAttribute(AIM_SOURCE_ATTRIBUTE),
            volume_node.GetName(),
        ):
            site = self._site_from_text(value)
            if site:
                return site
        storage_node = volume_node.GetStorageNode()
        if storage_node is not None:
            site = self._site_from_text(storage_node.GetFileName())
            if site:
                return site
        return ""

    def _selected_site(self, *, item=None, volume_node=None, strict=False):
        selected = str(self._combo_data(self.siteCombo, "auto"))
        if selected in SITE_PRESETS:
            return selected
        candidates = []
        if item is not None:
            candidates.extend([item.get("site"), item.get("path")])
        if volume_node is not None:
            candidates.extend(
                [
                    volume_node.GetAttribute(AIM_SOURCE_ATTRIBUTE),
                    volume_node.GetName(),
                ]
            )
            storage_node = volume_node.GetStorageNode()
            if storage_node is not None:
                candidates.append(storage_node.GetFileName())
        for candidate in candidates:
            site = self._site_from_text(candidate)
            if site:
                return site
        if strict:
            raise ValueError("Site preset is Auto, but the site could not be detected. Select Radius, Tibia, or Knee.")
        return "unparsed"

    def _combo_label(self, combo):
        text = combo.currentText
        return str(text() if callable(text) else text)

    def _update_batch_options_summary(self):
        if not hasattr(self, "batchOptionsSummaryLabel"):
            return
        output_root_text = str(self.batchOutputRootEdit.currentPath or "").strip()
        output_text = output_root_text or "input folder"
        self.batchOptionsSummaryLabel.text = (
            "Batch settings: "
            f"modality={self._combo_label(self.modalityCombo)}, "
            f"site={self._combo_label(self.siteCombo)}, "
            f"parameters={self._combo_label(self.parameterModeCombo) if hasattr(self, 'parameterModeCombo') else 'Preset'}, "
            f"bone segmentation={self._combo_label(self.segmentationMethodCombo)}, "
            f"periosteal={self._combo_label(self.periostealContourCombo)}, "
            f"endosteal={self._combo_label(self.endostealContourCombo)}, "
            f"format={self._combo_label(self.batchOutputFormatCombo) if hasattr(self, 'batchOutputFormatCombo') else 'Auto'}, "
            f"output={output_text}."
        )

    def _set_batch_row(self, row, image_path, status, parsed=None):
        parsed = parsed or {}
        self.batchSummaryTable.setItem(row, 0, qt.QTableWidgetItem(Path(image_path).name))
        self.batchSummaryTable.setItem(row, 1, qt.QTableWidgetItem(str(parsed.get("subject") or "Unparsed")))
        self.batchSummaryTable.setItem(row, 2, qt.QTableWidgetItem(str(parsed.get("session") or "Unparsed")))
        self.batchSummaryTable.setItem(row, 3, qt.QTableWidgetItem(str(parsed.get("site") or "Use selected")))
        self.batchSummaryTable.setItem(row, 5, qt.QTableWidgetItem(str(status)))
        action = "Load" if self._batchRowOutputs.get(row) else "Run"
        self._set_batch_action(row, action)

    def _set_batch_action(self, row, action):
        button = qt.QPushButton(str(action))
        if str(action) == "Load":
            button.clicked.connect(lambda _checked=False, row=row: self._load_batch_row_outputs(row))
        elif str(action) == "Cancel":
            button.clicked.connect(lambda _checked=False, row=row: self._cancel_batch_row(row))
        else:
            button.clicked.connect(lambda _checked=False, row=row: self._queue_batch_row(row))
        self.batchSummaryTable.setCellWidget(row, 4, button)

    def _resize_batch_table_columns(self):
        self.batchSummaryTable.resizeColumnsToContents()
        self.batchSummaryTable.setColumnWidth(4, 80)

    def _discover_batch_images(self):
        try:
            input_root = Path(self.batchInputRootEdit.currentPath or "").expanduser()
            if not input_root.is_dir():
                raise ValueError("Select an input folder.")
            image_paths = sorted(path for path in input_root.rglob("*") if path.is_file() and self._is_batch_image_path(path))
            self._batchImagePaths = image_paths
            self._batchImageRows = [
                {"path": path, **self._parse_batch_image_path(path)}
                for path in image_paths
            ]
            self._batchRowOutputs = {}
            self._batchQueue = []
            self._batchQueueRunning = False
            self.batchSummaryTable.setRowCount(len(image_paths))
            for row, item in enumerate(self._batchImageRows):
                outputs, format_label = self._preferred_existing_batch_outputs(item)
                if outputs:
                    self._batchRowOutputs[row] = outputs
                    self._set_batch_row(row, item["path"], f"Finished {format_label} ({','.join(outputs)})", item)
                else:
                    self._set_batch_row(row, item["path"], "Ready", item)
            self._resize_batch_table_columns()
            self._log(f"Discovered {len(image_paths)} image(s) for batch contouring.")
        except Exception as exc:
            self._error(exc)

    def _ensure_batch_rows(self):
        if not self._batchImagePaths:
            self._discover_batch_images()
        if not self._batchImagePaths:
            raise ValueError("No images discovered.")
        if not self._batchImageRows:
            self._batchImageRows = [
                {"path": path, **self._parse_batch_image_path(path)}
                for path in self._batchImagePaths
            ]

    def _batch_output_root_text(self):
        output_root_text = str(self.batchOutputRootEdit.currentPath or "").strip()
        if not output_root_text:
            output_root_text = str(self.batchInputRootEdit.currentPath or "").strip()
        return output_root_text

    def _batch_output_root(self):
        output_root_text = self._batch_output_root_text()
        output_root = Path(output_root_text).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root

    def _batch_item_site(self, item):
        return self._selected_site(item=item)

    def _batch_item_output_dir(self, item, output_root=None):
        output_root = Path(output_root) if output_root is not None else self._batch_output_root()
        image_path = Path(item["path"])
        subject = item.get("subject") or "unparsed"
        site = self._batch_item_site(item)
        session = item.get("session") or _image_output_stem(image_path)
        return output_root / "BoneContours" / f"sub-{subject}" / f"site-{site}" / f"ses-{session}" / "masks"

    def _find_existing_batch_outputs(self, item):
        output_root_text = self._batch_output_root_text()
        if not output_root_text:
            return {}, {}
        image_path = Path(item["path"])
        output_dir = self._batch_item_output_dir(item, Path(output_root_text).expanduser())
        stem = _image_output_stem(image_path)
        nifti_outputs = {}
        aim_outputs = {}
        for role in ("full", "trab", "cort", "seg"):
            nifti_path = output_dir / f"{stem}_mask-{role}.nii.gz"
            aim_path = output_dir / f"{stem}_mask-{role}.AIM"
            if nifti_path.exists():
                nifti_outputs[role] = str(nifti_path)
            if aim_path.exists():
                aim_outputs[role] = str(aim_path)
        return nifti_outputs, aim_outputs

    def _batch_output_format(self):
        if not hasattr(self, "batchOutputFormatCombo"):
            return "auto"
        return str(self.batchOutputFormatCombo.currentData or "auto").strip().lower()

    def _use_site_preset_params(self):
        if not hasattr(self, "parameterModeCombo"):
            return True
        return str(self._combo_data(self.parameterModeCombo, "preset")) == "preset"

    def _preferred_existing_batch_outputs(self, item):
        nifti_outputs, aim_outputs = self._find_existing_batch_outputs(item)
        output_format = self._batch_output_format()
        prefer_aim = output_format == "aim" or (output_format == "auto" and _is_aim_path(item["path"]))
        if prefer_aim:
            return aim_outputs, "AIM"
        return nifti_outputs, "NIfTI"

    def _queue_batch_row(self, row):
        try:
            self._ensure_batch_rows()
            if row < 0 or row >= len(self._batchImageRows):
                raise ValueError("Batch row is no longer available.")
            if row not in self._batchQueue and row not in self._batchRowOutputs:
                self._batchQueue.append(row)
                self._set_batch_row(row, self._batchImageRows[row]["path"], "Queued", self._batchImageRows[row])
                self._set_batch_action(row, "Cancel")
            if not self._batchQueueRunning:
                qt.QTimer.singleShot(0, self._process_next_batch_job)
        except Exception as exc:
            self._error(exc)

    def _queue_all_batch_rows(self):
        try:
            self._ensure_batch_rows()
            queued = 0
            for row, item in enumerate(self._batchImageRows):
                if row in self._batchRowOutputs or row in self._batchQueue:
                    continue
                self._batchQueue.append(row)
                self._set_batch_row(row, item["path"], "Queued", item)
                self._set_batch_action(row, "Cancel")
                queued += 1
            self._log(f"Queued {queued} batch job(s).")
            if queued and not self._batchQueueRunning:
                qt.QTimer.singleShot(0, self._process_next_batch_job)
        except Exception as exc:
            self._error(exc)

    def _process_next_batch_job(self):
        if self._batchProcess is not None:
            self._batchQueueRunning = True
            return
        if not self._batchQueue:
            self._batchQueueRunning = False
            self.batchRunButton.enabled = True
            self._log("Batch queue complete.")
            return
        self._batchQueueRunning = True
        self.batchRunButton.enabled = False
        row = self._batchQueue.pop(0)
        self._start_batch_worker(row)

    def _batch_worker_environment(self):
        environment = qt.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        for key in ("ITK_AUTOLOAD_PATH", "SITK_AUTOLOAD_PATH"):
            if environment.contains(key):
                environment.remove(key)
            environment.insert(key, "")
        python_paths = [
            str(_TOOLBOX_ROOT),
            str(_SCANCO_IO_DIR),
            str(_BONE_CONTOURING_LOCAL_SRC),
            str(_GEODESIC_CONTOUR_LOCAL_SRC),
        ]
        existing = str(environment.value("PYTHONPATH") or "")
        if existing:
            python_paths.append(existing)
        environment.insert("PYTHONPATH", ":".join(path for path in python_paths if path))
        return environment

    def _python_slicer_executable(self):
        executable = Path(sys.executable)
        candidates = []
        if executable.name == "python-real":
            candidates.append(executable.with_name("PythonSlicer"))
        try:
            app_dir = Path(slicer.app.applicationDirPath())
            candidates.extend(
                [
                    app_dir / "PythonSlicer",
                    app_dir.parent / "bin" / "PythonSlicer",
                    app_dir.parent / "MacOS" / "PythonSlicer",
                ]
            )
        except Exception:
            pass
        candidates.append(executable)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(executable)

    def _qbytearray_to_text(self, raw):
        if isinstance(raw, (bytes, bytearray)):
            data = bytes(raw)
        else:
            try:
                data = raw.data()
                if isinstance(data, str):
                    data = data.encode("utf-8", errors="replace")
                else:
                    data = bytes(data)
            except Exception:
                try:
                    data = bytes(raw)
                except Exception:
                    data = str(raw).encode("utf-8", errors="replace")
        return data.decode("utf-8", errors="replace")

    def _append_batch_worker_output(self, process, stream_name):
        if stream_name == "stdout":
            text = self._qbytearray_to_text(process.readAllStandardOutput())
            self._batchProcessStdout += text
        else:
            text = self._qbytearray_to_text(process.readAllStandardError())
            self._batchProcessStderr += text
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("{"):
                self._log(f"[contour-worker] {line}")

    def _cancel_batch_row(self, row):
        try:
            if row in self._batchQueue:
                self._batchQueue = [queued_row for queued_row in self._batchQueue if queued_row != row]
                self._set_batch_row(row, self._batchImageRows[row]["path"], "Cancelled", self._batchImageRows[row])
                self._log(f"Cancelled queued batch row: {Path(self._batchImageRows[row]['path']).name}.")
                return
            if row == self._batchRunningRow and self._batchProcess is not None:
                self._batchCancelRequested = True
                self._set_batch_row(row, self._batchImageRows[row]["path"], "Cancelling", self._batchImageRows[row])
                self._batchProcess.terminate()
                if not self._batchProcess.waitForFinished(1500):
                    self._batchProcess.kill()
                return
        except Exception as exc:
            self._error(exc)

    def _start_batch_worker(self, row):
        try:
            self._ensure_batch_rows()
            if row < 0 or row >= len(self._batchImageRows):
                raise ValueError("Batch row is no longer available.")
            item = self._batchImageRows[row]
            image_path = Path(item["path"])
            output_root = self._batch_output_root()
            site = self._selected_site(item=item, strict=self._use_site_preset_params())
            image_output_dir = self._batch_item_output_dir(item, output_root)
            config_dir = output_root / "BoneContours" / "slicer_run_configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / f"{_image_output_stem(image_path)}_batch_contour.json"
            config = {
                "image_path": str(image_path),
                "output_dir": str(image_output_dir),
                "site": site,
                "segmentation_method": str(self.segmentationMethodCombo.currentData),
                "periosteal_contour_method": str(self.periostealContourCombo.currentData),
                "endosteal_contour_method": str(self.endostealContourCombo.currentData),
                "output_prefix": _image_output_stem(image_path),
                "output_format": self._batch_output_format(),
                "params": self._collect_params(site=site, use_site_defaults=self._use_site_preset_params()),
            }
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
            worker_path = _TOOLBOX_ROOT / "SlicerBoneImagingToolboxLib" / "bone_contour_batch_worker.py"
            process = qt.QProcess()
            process.setProcessEnvironment(self._batch_worker_environment())
            process.readyReadStandardOutput.connect(
                lambda process=process: self._append_batch_worker_output(process, "stdout")
            )
            process.readyReadStandardError.connect(
                lambda process=process: self._append_batch_worker_output(process, "stderr")
            )
            process.finished.connect(
                lambda *signal_args, row=row, process=process: self._batch_worker_finished(
                    row,
                    process,
                    *signal_args,
                )
            )
            self._batchProcess = process
            self._batchRunningRow = row
            self._batchProcessStdout = ""
            self._batchProcessStderr = ""
            self._batchCancelRequested = False
            self._set_batch_row(row, image_path, "Running", item)
            self._set_batch_action(row, "Cancel")
            program = self._python_slicer_executable()
            process.start(program, [str(worker_path), "--config", str(config_path)])
            if not process.waitForStarted(1000):
                raise RuntimeError("Could not start batch contour worker.")
            self._log(f"Started batch contour worker: {image_path.name} ({program}).")
        except Exception as exc:
            if 0 <= row < len(self._batchImageRows):
                self._set_batch_row(row, self._batchImageRows[row]["path"], f"Error: {exc}", self._batchImageRows[row])
            self._batchProcess = None
            self._batchRunningRow = None
            self._error(exc)
            qt.QTimer.singleShot(0, self._process_next_batch_job)

    def _batch_worker_finished(self, row, process, *signal_args):
        try:
            self._append_batch_worker_output(process, "stdout")
            self._append_batch_worker_output(process, "stderr")
            stdout = self._batchProcessStdout.strip()
            stderr = self._batchProcessStderr.strip()
            if self._batchCancelRequested:
                self._set_batch_row(row, self._batchImageRows[row]["path"], "Cancelled", self._batchImageRows[row])
                self._log(f"Cancelled running batch row: {Path(self._batchImageRows[row]['path']).name}.")
                return
            if len(signal_args) >= 1:
                exit_code = int(signal_args[0])
            else:
                exit_code = int(process.exitCode())
            if exit_code != 0:
                raise RuntimeError(stderr or stdout or f"Batch contour worker exited with code {exit_code}.")
            lines = [line for line in stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError("Batch contour worker did not return a result.")
            result = json.loads(lines[-1])
            written = result.get("written") or {}
            metadata = result.get("metadata") or {}
            self._batchRowOutputs[row] = written
            roles = ",".join(written) or ",".join(metadata.get("emitted_roles", []))
            if metadata.get("output_format") == "aim":
                roles = f"AIM {roles}"
            self._set_batch_row(row, self._batchImageRows[row]["path"], f"Wrote {roles}", self._batchImageRows[row])
            self._resize_batch_table_columns()
            self._log(f"Batch row complete: {Path(self._batchImageRows[row]['path']).name}.")
        except Exception as exc:
            if 0 <= row < len(self._batchImageRows):
                self._set_batch_row(row, self._batchImageRows[row]["path"], f"Error: {exc}", self._batchImageRows[row])
            self._error(exc)
        finally:
            process.deleteLater()
            if self._batchProcess is process:
                self._batchProcess = None
                self._batchRunningRow = None
                self._batchProcessStdout = ""
                self._batchProcessStderr = ""
                self._batchCancelRequested = False
            qt.QTimer.singleShot(0, self._process_next_batch_job)

    def _execute_batch_row(self, row):
        try:
            self._ensure_batch_rows()
            if row < 0 or row >= len(self._batchImageRows):
                raise ValueError("Batch row is no longer available.")
            if not self.logic.is_pipeline_available():
                raise RuntimeError("Install or update bone-contouring first.")
            item = self._batchImageRows[row]
            image_path = item["path"]
            self._set_batch_row(row, image_path, "Running", item)
            slicer.app.processEvents()
            output_root = self._batch_output_root()
            site = self._selected_site(item=item, strict=self._use_site_preset_params())
            image_output_dir = self._batch_item_output_dir(item, output_root)
            metadata, written = self.logic.write_bone_mask_files(
                image_path,
                image_output_dir,
                site=site,
                segmentation_method=str(self.segmentationMethodCombo.currentData),
                periosteal_contour_method=str(self.periostealContourCombo.currentData),
                endosteal_contour_method=str(self.endostealContourCombo.currentData),
                output_prefix=_image_output_stem(image_path),
                output_format=self._batch_output_format(),
                keep_loaded=False,
                params=self._collect_params(site=site, use_site_defaults=self._use_site_preset_params()),
            )
            self._batchRowOutputs[row] = written
            roles = ",".join(written) or ",".join(metadata.get("emitted_roles", []))
            aim_outputs = metadata.get("aim_outputs", {})
            if aim_outputs:
                roles = f"AIM {','.join(aim_outputs)}"
            self._set_batch_row(row, image_path, f"Wrote {roles}", item)
            self._resize_batch_table_columns()
            self._log(f"Batch row complete: {image_path.name}.")
        except Exception as exc:
            if 0 <= row < len(self._batchImageRows):
                self._set_batch_row(row, self._batchImageRows[row]["path"], f"Error: {exc}", self._batchImageRows[row])
            self._error(exc)

    def _run_batch(self):
        self._queue_all_batch_rows()

    def _find_loaded_batch_source_volume(self, item):
        source_path = str((item or {}).get("path") or "")
        source_stem = _image_output_stem(source_path) if source_path else ""
        nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        for index in range(nodes.GetNumberOfItems()):
            node = nodes.GetItemAsObject(index)
            if node is None:
                continue
            candidates = [
                node.GetAttribute(AIM_SOURCE_ATTRIBUTE),
                node.GetName(),
            ]
            storage_node = node.GetStorageNode()
            if storage_node is not None:
                candidates.append(storage_node.GetFileName())
            for candidate in candidates:
                if not candidate:
                    continue
                candidate_text = str(candidate)
                if source_path and candidate_text == source_path:
                    return node
                if source_path and Path(candidate_text).name == Path(source_path).name:
                    return node
                if source_stem and _image_output_stem(candidate_text) == source_stem:
                    return node
        return None

    def _load_batch_row_outputs(self, row):
        try:
            written = self._batchRowOutputs.get(row) or {}
            if not written:
                raise ValueError("Run this row before loading outputs.")
            item = self._batchImageRows[row] if 0 <= row < len(self._batchImageRows) else {}
            source_name = _image_output_stem(item.get("path") or next(iter(written.values())))
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode",
                f"{source_name}_HRpQCT_segmentation",
            )
            segmentation_node.CreateDefaultDisplayNodes()
            self.logic._configure_segmentation_display(segmentation_node)
            segment_names = {
                "full": "Full mask",
                "trab": "Trabecular mask",
                "cort": "Cortical mask",
                "seg": "Bone segmentation",
            }
            loaded = []
            reference_node = None
            source_volume = self._find_loaded_batch_source_volume(item)
            for role in ("full", "trab", "cort", "seg"):
                path = written.get(role)
                if not path:
                    continue
                mask_image, mask_metadata = self.logic.read_mask_image_file(path)
                if reference_node is None:
                    reference_node = slicer.mrmlScene.AddNewNodeByClass(
                        "vtkMRMLScalarVolumeNode",
                        f"__{source_name}_batch_mask_reference__",
                    )
                    reference_node.SetHideFromEditors(True)
                    slicer.util.updateVolumeFromArray(reference_node, sitk.GetArrayFromImage(mask_image))
                    _set_slicer_volume_geometry_from_sitk_image(reference_node, mask_image)
                    if mask_metadata:
                        reference_node.SetAttribute(AIM_METADATA_ATTRIBUTE, json.dumps(mask_metadata, sort_keys=True, default=str))
                        reference_node.SetAttribute(AIM_SOURCE_ATTRIBUTE, str(path))
                        reference_node.SetAttribute(AIM_SCALING_ATTRIBUTE, "native")
                    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_node)
                self.logic._add_sitk_segment(
                    mask_image,
                    segmentation_node,
                    segment_names.get(role, role),
                    reference_node,
                    role,
                )
                loaded.append(role)
            if reference_node is not None:
                self.logic._copy_aim_attributes(reference_node, segmentation_node)
            if reference_node is not None and source_volume is not None:
                slicer.mrmlScene.RemoveNode(reference_node)
            segmentation_node.SetAttribute("BoneImaging.MaskRoles", ",".join(loaded))
            self.logic._configure_segmentation_display(segmentation_node)
            background_node = source_volume if source_volume is not None else reference_node
            if background_node is not None:
                slicer.util.setSliceViewerLayers(background=background_node, fit=False)
            self._center_slices_on_node(background_node or segmentation_node)
            self._log(f"Loaded batch segmentation with masks: {', '.join(loaded)}.")
        except Exception as exc:
            self._error(exc)

    def _center_slices_on_node(self, node_to_center):
        if node_to_center is None:
            return
        try:
            bounds = [0.0] * 6
            node_to_center.GetRASBounds(bounds)
            if not all(np.isfinite(bounds)):
                return
            cx = 0.5 * (bounds[0] + bounds[1])
            cy = 0.5 * (bounds[2] + bounds[3])
            cz = 0.5 * (bounds[4] + bounds[5])
            layout_manager = slicer.app.layoutManager()
            if layout_manager is None:
                return
            for view_name in ("Red", "Yellow", "Green"):
                widget = layout_manager.sliceWidget(view_name)
                if widget is None:
                    continue
                slice_node = widget.mrmlSliceNode()
                if slice_node is not None:
                    if hasattr(slice_node, "JumpSliceByOffsetting"):
                        slice_node.JumpSliceByOffsetting(cx, cy, cz)
                    else:
                        slice_node.JumpSliceByCentering(cx, cy, cz)
                    self._fit_slice_node_to_bounds(slice_node, widget, bounds, view_name)
        except Exception:
            pass

    def _fit_slice_node_to_bounds(self, slice_node, widget, bounds, view_name):
        try:
            dims = {
                "Red": (abs(bounds[1] - bounds[0]), abs(bounds[3] - bounds[2])),
                "Yellow": (abs(bounds[1] - bounds[0]), abs(bounds[5] - bounds[4])),
                "Green": (abs(bounds[3] - bounds[2]), abs(bounds[5] - bounds[4])),
            }.get(str(view_name), (abs(bounds[1] - bounds[0]), abs(bounds[3] - bounds[2])))
            dim_x = max(float(dims[0]), 1.0)
            dim_y = max(float(dims[1]), 1.0)
            view = widget.sliceView() if widget is not None else None
            render_window = view.renderWindow() if view is not None else None
            size = render_window.GetSize() if render_window is not None else (1, 1)
            aspect = max(0.1, float(max(1, int(size[0]))) / float(max(1, int(size[1]))))
            target_x = max(dim_x * 1.18, 8.0)
            target_y = max(dim_y * 1.18, 8.0)
            if target_x / target_y < aspect:
                target_x = target_y * aspect
            else:
                target_y = target_x / aspect
            current_fov = slice_node.GetFieldOfView()
            z_fov = float(current_fov[2]) if current_fov is not None and len(current_fov) >= 3 else 1.0
            slice_node.SetFieldOfView(float(target_x), float(target_y), z_fov)
        except Exception:
            pass

    def _create_segmentation(self):
        try:
            segmentation_method = str(self.segmentationMethodCombo.currentData)
            periosteal_method = str(self.periostealContourCombo.currentData)
            endosteal_method = str(self.endostealContourCombo.currentData)
            volume_node = self.volumeSelector.currentNode()
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
                raise RuntimeError("Install or update bone-contouring first.")
            try:
                selected_site = self._selected_site(
                    volume_node=volume_node,
                    strict=self._use_site_preset_params(),
                )
                params = self._collect_params(site=selected_site, use_site_defaults=self._use_site_preset_params())
                debug_dir = _scene_contour_debug_dir()
                segmentation_node, labelmaps, metadata = self.logic.generate_bone_masks(
                    volume_node,
                    site=selected_site,
                    segmentation_method=segmentation_method,
                    periosteal_contour_method=periosteal_method,
                    endosteal_contour_method=endosteal_method,
                    output_prefix=self.outputPrefixEdit.text.strip() or None,
                    create_labelmaps=False,
                    open_segment_editor=False,
                    params=params,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                    debug_output_dir=debug_dir,
                )
            finally:
                if progress_dialog is not None:
                    progress_dialog.close()
            counts = metadata.get("voxel_counts", {})
            label_text = ""
            processing_reader = metadata.get("processing_image_reader") or "selected_slicer_volume"
            provenance_text = f" Method={metadata.get('segmentation_method')}; image={processing_reader}."
            if metadata.get("segmentation_method") == "laplace_hamming":
                provenance_text = (
                    f" Method=laplace_hamming; image={processing_reader}; input={metadata.get('segmentation_input_unit')} "
                    f"via {metadata.get('segmentation_input_reader')}."
                )
            warning_text = ""
            if metadata.get("segmentation_warning"):
                warning_text = f" Warning: {metadata.get('segmentation_warning')}"
            emitted_roles = metadata.get("emitted_roles", [])
            emitted_text = f" Emitted={','.join(emitted_roles)}." if emitted_roles else ""
            debug_path = metadata.get("scene_debug_manifest")
            debug_text = f" Debug={debug_path}." if debug_path else ""
            self._log(
                f"Created {segmentation_node.GetName()}.{label_text} "
                f"Voxel counts: full={counts.get('full')}, trab={counts.get('trab')}, "
                f"cort={counts.get('cort')}, seg={counts.get('seg')}.{provenance_text} "
                f"Periosteal={metadata.get('periosteal_contour_method')}; "
                f"Endosteal={metadata.get('endosteal_contour_method')}; "
                f"Compartments={metadata.get('compartment_split_generated')}.{emitted_text}{debug_text}{warning_text}"
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

    def _update_dependency_ui(self):
        pipeline_available = self.logic.is_pipeline_available()
        geodesic_available = self.logic.is_geodesic_contour_available()
        if pipeline_available and geodesic_available:
            self.pipelineStatusLabel.text = "Installed"
            self.pipelineStatusLabel.styleSheet = "color: #228b22;"
        elif pipeline_available or geodesic_available:
            missing = "geodesic contour" if pipeline_available else "bone-contouring"
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


class SegmentationHRpQCTTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("SegmentationHRpQCT smoke test passed.")
