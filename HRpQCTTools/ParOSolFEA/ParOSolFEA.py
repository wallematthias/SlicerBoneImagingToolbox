from __future__ import annotations

import contextlib
import json
import math
import os
import csv
import copy
import html
import importlib
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import ctk
import numpy as np
import qt
import SimpleITK as sitk
import slicer
import vtk


def _active_repositories_root(toolbox_root):
    root = Path(toolbox_root).resolve()
    if root.parent.name == ".worktrees":
        return root.parent.parent.parent
    return root.parent


def _bootstrap_parosol_source_import_paths():
    module_path = Path(__file__).resolve()
    extension_root = module_path.parents[2]
    candidates = []
    env_source = str(os.environ.get("SLICER_PAROSOL_SOURCE", "") or "").strip()
    if env_source:
        candidates.append(Path(env_source).expanduser())

    import_paths = []
    seen = set()
    for candidate in candidates:
        for path in (candidate, candidate / "src"):
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            key = str(resolved)
            if key in seen:
                continue
            if (resolved / "parosol_py").is_dir():
                import_paths.append(resolved)
                seen.add(key)
    return tuple(import_paths)


def _load_workflow_geometry_module():
    for source_path in (*_bootstrap_parosol_source_import_paths(), None):
        source_path_text = None if source_path is None else str(source_path)
        if source_path_text:
            while source_path_text in sys.path:
                sys.path.remove(source_path_text)
            sys.path.insert(0, source_path_text)
            sys.modules.pop("parosol_py.workflow_geometry", None)
            sys.modules.pop("parosol_py", None)
        try:
            module = importlib.import_module("parosol_py.workflow_geometry")
        except Exception:
            continue
        required = (
            "estimate_reference_to_sample_transform",
            "invert_rigid_transform",
            "read_reference_points",
            "prealign_reference_points_to_sample",
            "resolve_reference_space_editor",
            "scale_reference_points_preserving_pose",
        )
        if all(hasattr(module, name) for name in required):
            return module

    raise ImportError(
        "SlicerParOSol requires parosol_py.workflow_geometry. "
        "Install ParOSol-py or set SLICER_PAROSOL_SOURCE to a current source checkout."
    )


def _purge_parosol_py_modules():
    for name in list(sys.modules):
        if name == "parosol_py" or str(name).startswith("parosol_py."):
            sys.modules.pop(name, None)


def _prepare_parosol_py_runtime_import():
    source_paths = _bootstrap_parosol_source_import_paths()
    if not source_paths:
        return None
    source_path = str(source_paths[0])
    while source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)
    _purge_parosol_py_modules()
    return source_path


try:
    _workflow_geometry = _load_workflow_geometry_module()
    _WORKFLOW_GEOMETRY_IMPORT_ERROR = None
except Exception as exc:
    _workflow_geometry = None
    _WORKFLOW_GEOMETRY_IMPORT_ERROR = exc


def _missing_workflow_geometry(*_args, **_kwargs):
    raise RuntimeError(
        "ParOSol-py workflow geometry is not available. "
        "Open Bone Imaging > Setup and install/update runtime packages, or set SLICER_PAROSOL_SOURCE."
    ) from _WORKFLOW_GEOMETRY_IMPORT_ERROR


estimate_reference_to_sample_transform = (
    _workflow_geometry.estimate_reference_to_sample_transform
    if _workflow_geometry is not None
    else _missing_workflow_geometry
)
invert_rigid_transform = _workflow_geometry.invert_rigid_transform if _workflow_geometry is not None else _missing_workflow_geometry
read_reference_points = _workflow_geometry.read_reference_points if _workflow_geometry is not None else _missing_workflow_geometry
prealign_reference_points_to_sample = (
    _workflow_geometry.prealign_reference_points_to_sample
    if _workflow_geometry is not None
    else _missing_workflow_geometry
)
resolve_reference_space_editor = (
    _workflow_geometry.resolve_reference_space_editor
    if _workflow_geometry is not None
    else _missing_workflow_geometry
)
scale_reference_points_preserving_pose = (
    _workflow_geometry.scale_reference_points_preserving_pose
    if _workflow_geometry is not None
    else _missing_workflow_geometry
)
generate_slicer_disk_and_nodeset_geometry = (
    getattr(_workflow_geometry, "generate_slicer_disk_and_nodeset_geometry", None)
    if _workflow_geometry is not None
    else None
)

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

from ParOSolFEALib.slicer_adapters import (
    node_array_shape as _node_array_shape,
    node_storage_file as _node_storage_file,
    volume_ijk_to_ras_array as _volume_ijk_to_ras_array,
)
from ParOSolFEALib.workflow_controllers import (
    BoundaryPreviewController,
    ExecutionController,
    LightweightEditorController,
    LoadPreviewController,
    PreprocessController,
)
from ParOSolFEALib.workflow_stage import WorkflowStageController, WorkflowStageState

TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from SlicerBoneImagingToolboxLib.fea_batch import (
    batch_profile_support_status,
    build_parosol_case_commands,
    case_readiness,
    discover_fea_batch_cases,
    parosol_command_derivative_context,
    role_options_for_workflow,
)


MODULE_VERSION = "0.1.0"
DEFAULT_PAROSOL = ""
DEFAULT_PAROSOL_SOURCE = ""
COMMON_MPI_LAUNCHER_CANDIDATES = (
    Path("/opt/homebrew/bin/mpirun"),
    Path("/opt/homebrew/bin/mpiexec"),
    Path("/usr/local/bin/mpirun"),
    Path("/usr/local/bin/mpiexec"),
    Path.home() / "miniforge3/envs/parosol/bin/mpirun",
    Path.home() / "miniforge3/envs/parosol/bin/mpiexec",
    Path.home() / "mambaforge/envs/parosol/bin/mpirun",
    Path.home() / "mambaforge/envs/parosol/bin/mpiexec",
    Path.home() / "anaconda3/envs/parosol/bin/mpirun",
    Path.home() / "anaconda3/envs/parosol/bin/mpiexec",
)
def _shared_profile_tool_root(tool):
    try:
        from bone_imaging_derivatives import tool_profile_dir

        return tool_profile_dir(tool)
    except Exception:
        path = Path.home() / ".slicerboneimagingtoolbox" / "profiles" / str(tool)
        path.mkdir(parents=True, exist_ok=True)
        return path


USER_WORKFLOW_ROOT = _shared_profile_tool_root("parosol-fea")
WORKFLOW_SEARCH_ROOTS = (USER_WORKFLOW_ROOT,)
WORKFLOW_BUNDLE_EXCLUDED_FILES = {"slicer_input.nii.gz", "slicer_mask.nii.gz"}
PREFERRED_WORKFLOW = "interactive_custom"
SLICER_PAROSOL_BUILD = "2026-07-16-portable-load-history-bundle"
CUSTOM_PREPROCESSING_CREATE_TOKEN = "__create__"
CUSTOM_PREPROCESSING_SCAFFOLD_TEMPLATE = '''"""Custom preprocessing hook for ParOSol-py/Slicer workflows.

This file is bundled with the .parosol-workflow file and runs after standard
Image Prep preprocessing and ICP registration, on the aligned model grid.
"""

import numpy as np

from parosol_py.images import ImageGrid


def __FUNCTION_NAME__(image, mask=None):
    """Return a preprocessed (image, mask) pair.

    Parameters
    ----------
    image : parosol_py.images.ImageGrid
        Density/grayscale image with array_xyz, spacing, and origin.
    mask : parosol_py.images.ImageGrid | None
        Label/binary mask on the same grid as image.

    Returns
    -------
    tuple
        Return (image, mask) or (image, mask, metadata).
    """
    if mask is None:
        return image, mask

    image_array = np.asarray(image.array_xyz)
    mask_array = np.asarray(mask.array_xyz)

    # Example: replace this block with workflow-specific cropping/filtering.
    out_image = ImageGrid(
        array_xyz=image_array.copy(),
        spacing=image.spacing,
        origin=image.origin,
    )
    out_mask = ImageGrid(
        array_xyz=mask_array.copy(),
        spacing=mask.spacing,
        origin=mask.origin,
    )
    metadata = {"name": "__FUNCTION_NAME__"}
    return out_image, out_mask, metadata
'''
GENERATED_FIXED_LABEL_BASE = 11001
GENERATED_DIRICHLET_LABEL_BASE = 12001
GENERATED_NEUMANN_LABEL_BASE = 13001
GENERATED_INACTIVE_LABEL_BASE = 14001
LOAD_FIXED_DOFS_COLUMN = 8
LOAD_NODESET_LABEL_COLUMN = 9
LOAD_TABLE_COLUMN_COUNT = LOAD_NODESET_LABEL_COLUMN + 1
PAROSOL_SOURCE_CLI_BOOTSTRAP = (
    "import os, runpy, sys; "
    "source_path = os.environ.get(\"SLICER_PAROSOL_SOURCE\"); "
    "sys.path.remove(source_path) if source_path in sys.path else None; "
    "sys.path.insert(0, source_path) if source_path else None; "
    "sys.argv[0] = \"parosol\"; "
    "runpy.run_module(\"parosol_py.cli\", run_name=\"__main__\")"
)
SEGMENT_SELECTION_ATTRIBUTE = "SlicerParOSol.SelectedSegmentID"
SEGMENT_SELECTION_IDS_ATTRIBUTE = "SlicerParOSol.SelectedSegmentIDs"
LABEL_SELECTION_VALUES_ATTRIBUTE = "SlicerParOSol.SelectedLabelValues"
SEGMENT_LABEL_VALUE_MAP_ATTRIBUTE = "SlicerParOSol.SegmentLabelValueMap"
PREVIEW_SEGMENT_SIGNATURE_ATTRIBUTE = "SlicerParOSol.PreviewSourceSignature"
SEGMENT_SELECTION_ALL = "__all__"
SEGMENT_SELECTION_SUBSET = "__subset__"
WORKFLOW_ORDER = (
    "spine-compression",
    "spine-compression-nonlinear",
    "hip-sideways-fall-left",
    "hip-sideways-fall-left-nonlinear",
    "hip-sideways-fall-right",
    "hip-sideways-fall-right-nonlinear",
    "XtremeCTII",
    "XtremeCTI",
    "load_history_3",
    "load_history_6",
)
_PAROSOL_WORKFLOW_REGISTRY = None


def _available_builtin_workflows():
    parosol_names = set(_parosol_py_workflow_names())
    names = [name for name in WORKFLOW_ORDER if name in parosol_names]
    for name in sorted(parosol_names):
        if name not in names:
            names.append(name)
    return tuple(names)


def _parosol_workflow_registry():
    global _PAROSOL_WORKFLOW_REGISTRY

    if _PAROSOL_WORKFLOW_REGISTRY is not None:
        return _PAROSOL_WORKFLOW_REGISTRY

    source_paths = _parosol_source_checkout_import_paths()
    import_attempts = [*source_paths, None]
    for source_path in import_attempts:
        inserted = False
        source_path_text = None
        if source_path is not None:
            source_path_text = str(source_path)
            while source_path_text in sys.path:
                sys.path.remove(source_path_text)
            sys.path.insert(0, source_path_text)
            inserted = True
            sys.modules.pop("parosol_py.workflow_registry", None)
            sys.modules.pop("parosol_py", None)
        try:
            registry = importlib.import_module("parosol_py.workflow_registry")
        except Exception:
            if inserted and source_path_text in sys.path:
                sys.path.remove(source_path_text)
            continue
        available_profiles = getattr(registry, "available_profiles", None)
        builtin_workflow_path = getattr(registry, "builtin_workflow_path", None)
        if callable(available_profiles) and callable(builtin_workflow_path):
            _PAROSOL_WORKFLOW_REGISTRY = (available_profiles, builtin_workflow_path)
            return _PAROSOL_WORKFLOW_REGISTRY
    return None


def _parosol_source_checkout_import_paths():
    candidates = []
    env_source = str(os.environ.get("SLICER_PAROSOL_SOURCE", "") or "").strip()
    if env_source:
        candidates.append(Path(env_source).expanduser())

    source_setting = str(
        _slicer_setting_value("SlicerParOSol/useSourceCheckout", "false") or ""
    ).strip().lower()
    if source_setting in {"1", "true", "yes", "on", "source"}:
        configured_source = str(
            _slicer_setting_value(
                "SlicerParOSol/parosolSource",
                os.environ.get("SLICER_PAROSOL_SOURCE", DEFAULT_PAROSOL_SOURCE),
            )
            or ""
        ).strip()
        if configured_source:
            candidates.append(Path(configured_source).expanduser())
        candidates.extend(_local_parosol_source_checkout_paths())

    return _parosol_import_paths_from_candidates(candidates)


def _local_parosol_source_checkout_paths():
    module_path = Path(__file__).resolve()
    extension_root = module_path.parents[2]
    sibling_root = _active_repositories_root(extension_root) / "parosol-py"
    nested_root = extension_root / "parosol-py"
    return (sibling_root, nested_root)


def _parosol_import_paths_from_candidates(candidates):
    import_paths = []
    seen = set()
    for candidate in candidates:
        for path in (candidate, candidate / "src"):
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            key = str(resolved)
            if key in seen:
                continue
            if (resolved / "parosol_py").is_dir():
                import_paths.append(resolved)
                seen.add(key)
    return tuple(import_paths)


def _slicer_setting_value(key, default=""):
    try:
        settings_class = getattr(qt, "QSettings", None)
        if settings_class is not None:
            value = settings_class().value(key, default)
            return default if value is None else value
    except Exception:
        pass
    try:
        app = getattr(slicer, "app", None)
        settings = app.settings() if app is not None and hasattr(app, "settings") else None
        if settings is not None:
            value = settings.value(key, default)
            return default if value is None else value
    except Exception:
        pass
    return default


def _parosol_py_workflow_names():
    registry = _parosol_workflow_registry()
    if registry is None:
        return ()
    available_profiles, _builtin_workflow_path = registry
    try:
        return tuple(str(name) for name in available_profiles())
    except Exception:
        return ()


def _parosol_py_workflow_path(name):
    registry = _parosol_workflow_registry()
    if registry is None:
        return None
    _available_profiles, builtin_workflow_path = registry
    try:
        path = builtin_workflow_path(str(name or "").strip())
    except Exception:
        return None
    if path is None:
        return None
    return Path(path)


def _available_user_workflows():
    try:
        from bone_imaging_derivatives import list_profiles

        return tuple(
            str(record.name)
            for record in list_profiles("parosol-fea")
            if str(record.kind) == "parosol-workflow" and Path(record.path).expanduser().is_file()
        )
    except Exception:
        return ()


def _registered_user_workflow_path(name):
    token = str(name or "").strip()
    if not token:
        return None
    try:
        from bone_imaging_derivatives import list_profiles

        for record in list_profiles("parosol-fea"):
            if str(record.kind) != "parosol-workflow":
                continue
            if str(record.name).strip() != token:
                continue
            path = Path(record.path).expanduser()
            if path.is_file():
                return path
    except Exception:
        return None
    return None


def _workflow_path_in_roots(name):
    token = str(name or "").strip()
    if not token:
        return None
    lowered = token.lower()
    for root in WORKFLOW_SEARCH_ROOTS:
        for candidate in (token, token.lower(), token.replace(" ", "_")):
            bundle = root / f"{candidate}.parosol-workflow"
            if bundle.is_file():
                return bundle
            workflow = root / candidate / "workflow.yaml"
            if workflow.is_file():
                return workflow
        if root.is_dir():
            for path in root.iterdir():
                if path.is_file() and path.name.lower() == f"{lowered}.parosol-workflow":
                    return path
                if path.is_dir() and path.name.lower() == lowered:
                    workflow = path / "workflow.yaml"
                    if workflow.is_file():
                        return workflow
    return None


def _builtin_workflow_path(name):
    path = _parosol_py_workflow_path(name)
    if path is not None:
        return path
    path = _registered_user_workflow_path(name)
    if path is not None:
        return path
    return _workflow_path_in_roots(name)


def _is_backup_workflow_name(name):
    token = str(name).lower()
    return ".backup-" in token or ".pre-" in token


def _default_profiles():
    profiles = [*_available_builtin_workflows(), PREFERRED_WORKFLOW]
    for name in _available_user_workflows():
        if name not in profiles:
            profiles.append(name)
    return tuple(profiles)


DEFAULT_PROFILES = (PREFERRED_WORKFLOW,)


class ParOSolFEA(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "ParOsol-FEA"
        parent.categories = ["Bone Imaging.FE Analysis"]
        parent.icon = qt.QIcon(str(Path(__file__).with_name("Resources") / "Icons" / "ParOSolFEA.png"))
        parent.index = 10
        parent.dependencies = ["Segmentations"]
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "ParOSol-py finite-element model editor and runner.\n"
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = "Author: Matthias Walle. Built for visual ParOSol-py model authoring."


class ParOSolFEALogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._proc = None
        self._user_terminated = False

    def setting_value(self, key, default=""):
        if hasattr(qt, "QSettings"):
            return qt.QSettings().value(key, default)
        return default

    def set_setting_value(self, key, value):
        if hasattr(qt, "QSettings"):
            qt.QSettings().setValue(key, value)

    def discover_fea_batch_cases(self, dataset_root, **filters):
        return discover_fea_batch_cases(dataset_root, **filters)

    def build_fea_batch_commands(
        self,
        dataset_root,
        cases,
        *,
        workflow,
        selected_roles=None,
        dry_run=False,
    ):
        return build_parosol_case_commands(
            dataset_root,
            cases,
            workflow=workflow,
            selected_roles=selected_roles,
            dry_run=dry_run,
        )

    def python_launcher(self):
        exe = Path(sys.executable).resolve()
        wrapper = exe.parent / "PythonSlicer"
        if wrapper.exists():
            return str(wrapper)
        return str(exe)

    def discover_mpi_launcher(self):
        configured = str(self.setting_value("SlicerParOSol/mpiLauncher", "") or "").strip()
        candidates = []
        if configured:
            candidates.append(Path(configured).expanduser())
        path_env = self.parosol_environment().get("PATH", "")
        for name in ("mpirun", "mpiexec"):
            found = shutil.which(name, path=path_env)
            if found:
                candidates.append(Path(found))
        candidates.extend(COMMON_MPI_LAUNCHER_CANDIDATES)
        for candidate in candidates:
            try:
                if candidate.expanduser().exists():
                    return str(candidate.expanduser())
            except Exception:
                continue
        return ""

    def create_plane(self, name, volume_node=None, side="top", bounds_node=None):
        plane = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", name)
        if volume_node is not None:
            bounds = self.node_foreground_ras_bounds(bounds_node, volume_node) if bounds_node else None
            if bounds is None:
                bounds = [0.0] * 6
                volume_node.GetRASBounds(bounds)
            x_center = 0.5 * (bounds[0] + bounds[1])
            y_center = 0.5 * (bounds[2] + bounds[3])
            z_value = bounds[5] if side == "top" else bounds[4]
            plane.SetCenter([x_center, y_center, z_value])
            plane.SetNormal([0.0, 0.0, -1.0] if side == "top" else [0.0, 0.0, 1.0])
            size_x = max(bounds[1] - bounds[0], 1.0)
            size_y = max(bounds[3] - bounds[2], 1.0)
            if hasattr(plane, "SetSize"):
                plane.SetSize(size_x, size_y)
        _style_interactive_plane(plane)
        display = plane.GetDisplayNode()
        if display is not None:
            display.SetSelectedColor(0.1, 0.6, 1.0)
        self.group_node(plane, "Planes")
        return plane

    def create_axis_plane(
        self,
        name,
        volume_node=None,
        *,
        axis="z",
        normal_sign=1,
        bounds_node=None,
    ):
        plane = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", name)
        axis = str(axis).strip().lower()
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        normal_sign = 1 if int(normal_sign) >= 0 else -1
        normal = [0.0, 0.0, 0.0]
        normal[axis_index] = float(normal_sign)
        plane.SetNormal(normal)
        if volume_node is not None:
            bounds = self.node_foreground_ras_bounds(bounds_node, volume_node) if bounds_node else None
            if bounds is None:
                bounds = [0.0] * 6
                volume_node.GetRASBounds(bounds)
            center = [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ]
            center[axis_index] = bounds[2 * axis_index] if normal_sign > 0 else bounds[2 * axis_index + 1]
            plane.SetCenter(center)
            lateral_axes = [idx for idx in range(3) if idx != axis_index]
            size_0 = max(bounds[2 * lateral_axes[0] + 1] - bounds[2 * lateral_axes[0]], 1.0)
            size_1 = max(bounds[2 * lateral_axes[1] + 1] - bounds[2 * lateral_axes[1]], 1.0)
            if hasattr(plane, "SetSize"):
                plane.SetSize(size_0, size_1)
        _style_interactive_plane(plane)
        display = plane.GetDisplayNode()
        if display is not None:
            display.SetSelectedColor(0.1, 0.6, 1.0)
        self.group_node(plane, "Planes")
        return plane

    def node_foreground_ras_bounds(self, node, reference_node=None):
        if node is None:
            return None
        try:
            array = _array_from_mask_like(node, reference_node)
        except Exception:
            return None
        foreground = np.argwhere(np.asarray(array) != 0)
        if foreground.size == 0:
            return None
        min_zyx = foreground.min(axis=0)
        max_zyx = foreground.max(axis=0) + 1
        ijk_corners = []
        for k in (int(min_zyx[0]), int(max_zyx[0])):
            for j in (int(min_zyx[1]), int(max_zyx[1])):
                for i in (int(min_zyx[2]), int(max_zyx[2])):
                    ijk_corners.append((i, j, k))
        geometry_node = reference_node if _is_segmentation_node(node) and reference_node is not None else node
        ijk_to_ras = vtk.vtkMatrix4x4()
        geometry_node.GetIJKToRASMatrix(ijk_to_ras)
        ras_corners = [
            ijk_to_ras.MultiplyPoint([i, j, k, 1.0])[:3]
            for i, j, k in ijk_corners
        ]
        xs = [p[0] for p in ras_corners]
        ys = [p[1] for p in ras_corners]
        zs = [p[2] for p in ras_corners]
        return [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)]

    def create_labelmap_like(self, volume_node, name):
        if volume_node is None:
            raise ValueError("Input volume is required")
        label_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
        reference = slicer.util.arrayFromVolume(volume_node)
        slicer.util.updateVolumeFromArray(
            label_node, np.zeros(reference.shape, dtype=np.uint16)
        )
        label_node.CopyOrientation(volume_node)
        label_node.CreateDefaultDisplayNodes()
        label_node.Modified()
        self.group_node(label_node, _parosol_folder_for_node_name(name))
        return label_node

    def style_labelmap(self, label_node, kind):
        if label_node is None:
            return
        label_node.CreateDefaultDisplayNodes()
        display = label_node.GetDisplayNode()
        if display is None:
            return
        color_node = _color_table(kind)
        if color_node is not None and kind == "disks":
            try:
                max_label = int(np.max(slicer.util.arrayFromVolume(label_node)))
                if max_label >= color_node.GetNumberOfColors():
                    color_node.SetNumberOfColors(max_label + 1)
                for label in np.unique(slicer.util.arrayFromVolume(label_node)):
                    label = int(label)
                    if label > 0:
                        shade = 0.45 + 0.35 * ((label * 37) % 100) / 100.0
                        color_node.SetColor(label, f"cap_{label}", shade, shade, shade, 0.75)
            except Exception:
                pass
        elif color_node is not None:
            try:
                _extend_label_color_table(
                    color_node,
                    (int(label) for label in np.unique(slicer.util.arrayFromVolume(label_node))),
                    kind=kind,
                )
            except Exception:
                pass
        if color_node is not None:
            display.SetAndObserveColorNodeID(color_node.GetID())
        display.SetOpacity(0.65)

    def remove_named_node(self, name):
        nodes = [
            slicer.mrmlScene.GetNthNode(index)
            for index in range(slicer.mrmlScene.GetNumberOfNodes())
        ]
        matching_nodes = [
            node for node in nodes
            if node is not None
            and str(node.GetName() or "") == str(name)
            and not _is_mrml_infrastructure_node(node)
        ]
        removed = 0
        for node in matching_nodes:
            if slicer.mrmlScene.IsNodePresent(node):
                self.remove_node(node)
                removed += 1
        return removed

    def group_node(self, node, folder_name=None):
        if node is None:
            return
        try:
            sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
                slicer.mrmlScene
            )
            if sh_node is None:
                return
            scene_item_id = sh_node.GetSceneItemID()
            root_id = _subject_hierarchy_folder(sh_node, scene_item_id, "SlicerParOSol")
            if folder_name:
                parent_id = _subject_hierarchy_folder(sh_node, root_id, str(folder_name))
            else:
                parent_id = root_id
            item_id = sh_node.GetItemByDataNode(node)
            if item_id:
                sh_node.SetItemParent(item_id, parent_id)
        except Exception:
            return

    def clear_generated_nodes(self):
        nodes = [
            slicer.mrmlScene.GetNthNode(index)
            for index in range(slicer.mrmlScene.GetNumberOfNodes())
        ]
        generated = [
            node for node in nodes
            if node is not None and _is_removable_generated_parosol_node(node)
        ]
        generated.sort(key=_generated_node_removal_priority)
        removed = 0
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        try:
            _clear_parosol_viewer_references()
            for node in generated:
                if node is not None and slicer.mrmlScene.IsNodePresent(node):
                    try:
                        self.remove_node(node)
                        removed += 1
                    except Exception:
                        pass
        finally:
            slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)
        _remove_empty_parosol_subject_hierarchy_folders()
        return removed

    def show_mask_3d(self, node, reference_node=None, *, active_values=None):
        mask_preview_name = "ParOSol_input_mask_3D"
        volume_preview_name = "ParOSol_input_volume_3D"
        if node is None:
            self.remove_named_node(mask_preview_name)
            self.remove_named_node(volume_preview_name)
            if reference_node is None:
                return None
            self.hide_other_mask_3d(reference_node)
            preview_labelmap = self.input_volume_preview_labelmap(
                reference_node,
                active_values=active_values,
            )
            if preview_labelmap is None:
                return None
            try:
                return self.labelmap_to_3d_segmentation(
                    preview_labelmap,
                    volume_preview_name,
                    reference_node=reference_node,
                    kind="mask",
                )
            finally:
                if slicer.mrmlScene.IsNodePresent(preview_labelmap):
                    self.remove_node(preview_labelmap)
        if _is_segmentation_node(node):
            self.remove_named_node(volume_preview_name)
            self.hide_other_mask_3d(None)
            self.remove_named_node(mask_preview_name)
            return self.labelmap_to_3d_segmentation(
                node,
                mask_preview_name,
                reference_node=reference_node,
                kind="mask",
            )
        self.remove_named_node(volume_preview_name)
        self.hide_other_mask_3d(node)
        self.remove_named_node(mask_preview_name)
        return self.labelmap_to_3d_segmentation(
            node,
            mask_preview_name,
            reference_node=reference_node,
            kind="mask",
        )

    def input_volume_preview_labelmap(self, volume_node, *, active_values=None):
        if volume_node is None:
            return None
        try:
            array = np.asarray(slicer.util.arrayFromVolume(volume_node))
        except Exception:
            return None
        if array.size == 0:
            return None
        if active_values is not None:
            values = tuple(int(value) for value in active_values)
            mask = np.isin(array, values) if values else array != 0
            preview = np.where(mask, np.rint(array), 0)
        else:
            mask = array != 0
            if not np.any(mask):
                return None
            nonzero_values = np.unique(array[mask])
            label_like = (
                len(nonzero_values) <= 64
                and np.all(np.isfinite(nonzero_values))
                and np.allclose(nonzero_values, np.rint(nonzero_values))
                and float(np.min(nonzero_values)) >= 0.0
                and float(np.max(nonzero_values)) <= 65535.0
            )
            preview = np.where(mask, np.rint(array), 0) if label_like else mask.astype(np.uint16)
        if not np.any(preview):
            return None
        label_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            "ParOSol_input_volume_preview_labelmap",
        )
        slicer.util.updateVolumeFromArray(
            label_node,
            np.asarray(np.clip(preview, 0, 65535), dtype=np.uint16),
        )
        label_node.CopyOrientation(volume_node)
        label_node.CreateDefaultDisplayNodes()
        label_node.Modified()
        self.style_labelmap(label_node, "mask")
        self.group_node(label_node, "Inputs")
        return label_node

    def hide_other_mask_3d(self, active_node):
        keep_visible = {"ParOSol_contact_caps_3D", "ParOSol_boundary_conditions_3D"}
        parosol_preview_names = {"ParOSol_input_mask_3D", "ParOSol_input_volume_3D"}
        for index in range(slicer.mrmlScene.GetNumberOfNodes()):
            node = slicer.mrmlScene.GetNthNode(index)
            if node is None or node is active_node:
                continue
            if str(node.GetName()) in keep_visible:
                continue
            if str(node.GetName()) in parosol_preview_names or _is_segmentation_node(node):
                _hide_display_node(node.GetDisplayNode())

    def labelmap_to_3d_segmentation(self, label_node, name, *, reference_node=None, kind="mask"):
        if label_node is None:
            return None
        import_node = self._as_labelmap_volume(label_node, reference_node=reference_node)
        if import_node is None:
            return None
        temporary_imports = []
        if import_node is not label_node:
            temporary_imports.append(import_node)
        reference_import = _labelmap_preview_in_reference_geometry(
            import_node,
            reference_node,
            name=f"{name}_reference_geometry_labelmap",
        )
        if reference_import is not import_node:
            import_node = reference_import
            temporary_imports.append(reference_import)
        source_import = import_node
        import_node = _integer_labelmap_for_segmentation_import(import_node)
        if import_node is not source_import:
            temporary_imports.append(import_node)
        self.style_labelmap(import_node, kind if kind in {"disks", "nodesets", "mask"} else "nodesets")
        signature = _preview_segmentation_signature(import_node, reference_node, kind=kind)
        existing = _cached_preview_segmentation(name, signature)
        if existing is not None:
            self._remove_temporary_preview_nodes(temporary_imports)
            _style_segmentation_3d(existing, kind=kind)
            self.group_node(existing, _parosol_folder_for_node_name(name))
            return existing
        self.remove_named_node(name)
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
        segmentation_node.CreateDefaultDisplayNodes()
        try:
            segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
                reference_node or import_node
            )
        except Exception:
            pass
        try:
            imported = _segmentations_logic().ImportLabelmapToSegmentationNode(
                import_node,
                segmentation_node,
            )
        except TypeError:
            segment_ids = vtk.vtkStringArray()
            imported = _segmentations_logic().ImportLabelmapToSegmentationNode(
                import_node,
                segmentation_node,
                segment_ids,
            )
        finally:
            self._remove_temporary_preview_nodes(temporary_imports)
        if imported is False:
            raise RuntimeError(f"Could not import {label_node.GetName()} into a 3D segmentation")
        _ensure_closed_surface_representation(segmentation_node)
        segmentation_node.SetAttribute(PREVIEW_SEGMENT_SIGNATURE_ATTRIBUTE, signature)
        _style_segmentation_3d(segmentation_node, kind=kind)
        self.group_node(segmentation_node, _parosol_folder_for_node_name(name))
        return segmentation_node

    def _remove_temporary_preview_nodes(self, nodes):
        for node in nodes or ():
            try:
                if node is not None and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)
            except Exception:
                pass

    def generate_fast_face_nodesets(self, volume_node, active_node, rows, *, active_values=None):
        if volume_node is None:
            raise ValueError("Input volume is required")
        output_labelmap = self.create_labelmap_like(volume_node, "ParOSol_nodesets")
        nodesets = self.fast_face_nodeset_array(
            volume_node,
            active_node,
            rows,
            active_values=active_values,
        )
        slicer.util.updateVolumeFromArray(output_labelmap, nodesets)
        output_labelmap.CopyOrientation(volume_node)
        output_labelmap.Modified()
        self.style_labelmap(output_labelmap, "nodesets")
        return output_labelmap

    def fast_face_nodeset_array(self, volume_node, active_node, rows, *, active_values=None):
        active = _target_mask_array(active_node or volume_node, volume_node, active_values=active_values)
        if active is None or not np.any(active):
            try:
                return np.zeros_like(slicer.util.arrayFromVolume(volume_node), dtype=np.uint16)
            except Exception:
                return np.zeros((0, 0, 0), dtype=np.uint16)
        nodesets = np.zeros_like(active, dtype=np.uint16)
        axis_to_array_dim = {"x": 2, "y": 1, "z": 0}
        for row in rows:
            axis = str(row.get("axis", "z")).strip().lower()
            array_dim = axis_to_array_dim.get(axis, 0)
            normal = str(row.get("normal", "-")).strip()
            label = int(row.get("label", 0))
            if label <= 0:
                continue
            plane = row.get("plane")
            if _looks_like_plane_node(plane):
                plane_nodes = _intersect_plane_nodeset_array(
                    active,
                    volume_node,
                    plane,
                    label,
                    shape=row.get("shape", "anatomy"),
                    radius_mm=row.get("radius", 12.0),
                    use_plane_size=bool(row.get("use_plane_size", True)),
                )
                if np.any(plane_nodes == label):
                    nodesets[plane_nodes == label] = label
                continue
            active_moved = np.moveaxis(active, array_dim, 0)
            nodesets_moved = np.moveaxis(nodesets, array_dim, 0)
            footprint = np.any(active_moved, axis=0)
            if not np.any(footprint):
                continue
            if normal == "-":
                surface = active_moved.shape[0] - 1 - np.argmax(active_moved[::-1], axis=0)
            else:
                surface = np.argmax(active_moved, axis=0)
            other_indices = np.nonzero(footprint)
            nodesets_moved[(surface[other_indices], *other_indices)] = label
        return nodesets

    def _as_labelmap_volume(self, node, *, reference_node=None):
        if node is None:
            return None
        if node.IsA("vtkMRMLLabelMapVolumeNode"):
            return node
        if _is_segmentation_node(node):
            return _segmentation_to_labelmap_node(node, reference_node)
        if not node.IsA("vtkMRMLScalarVolumeNode"):
            return None
        label_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{node.GetName()}_labelmap_preview",
        )
        try:
            array = np.asarray(slicer.util.arrayFromVolume(node))
            slicer.util.updateVolumeFromArray(label_node, array.astype(np.uint16, copy=False))
            label_node.CopyOrientation(reference_node or node)
            label_node.CreateDefaultDisplayNodes()
            label_node.Modified()
            return label_node
        except Exception:
            if slicer.mrmlScene.IsNodePresent(label_node):
                slicer.mrmlScene.RemoveNode(label_node)
            raise

    def create_arrow_model(self, name, start, direction, length_mm, color):
        direction = _normalized(direction)
        if _vector_length(direction) <= 0:
            return None
        self.remove_named_node(name)
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        model.SetAndObservePolyData(
            _arrow_polydata(start, direction, max(float(length_mm), 1.0))
        )
        model.CreateDefaultDisplayNodes()
        display = model.GetDisplayNode()
        if display is not None:
            _set_display_color(display, color)
            display.SetOpacity(0.95)
            display.SetVisibility(True)
            try:
                display.SetVisibility2D(True)
            except Exception:
                pass
            try:
                display.SetVisibility3D(True)
            except Exception:
                pass
        self.group_node(model, "Loads")
        return model

    def create_arrow_glyph_model(self, name, vectors, color, *, folder_name="Loads"):
        self.remove_named_node(name)
        append = vtk.vtkAppendPolyData()
        count = 0
        for start, direction, length_mm in vectors:
            if _vector_length(direction) <= 1e-6 or float(length_mm) <= 1e-6:
                continue
            append.AddInputData(_arrow_polydata(start, direction, float(length_mm)))
            count += 1
        if count == 0:
            return None
        append.Update()
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        model.SetAndObservePolyData(append.GetOutput())
        model.CreateDefaultDisplayNodes()
        display = model.GetDisplayNode()
        if display is not None:
            _set_display_color(display, color)
            display.SetOpacity(0.95)
            display.SetVisibility(True)
            try:
                display.SetVisibility2D(True)
                display.SetVisibility3D(True)
            except Exception:
                pass
        self.group_node(model, folder_name)
        return model

    def create_vector_line(self, name, start, direction, length_mm, color, *, thickness=0.6, glyph_scale=2.0):
        direction = _normalized(direction)
        if _vector_length(direction) <= 0:
            return None
        self.remove_named_node(name)
        end = [
            float(start[index]) + float(length_mm) * float(direction[index])
            for index in range(3)
        ]
        line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
        try:
            line.AddControlPointWorld(vtk.vtkVector3d(*start))
            line.AddControlPointWorld(vtk.vtkVector3d(*end))
        except Exception:
            line.AddControlPoint(vtk.vtkVector3d(*start))
            line.AddControlPoint(vtk.vtkVector3d(*end))
        line.CreateDefaultDisplayNodes()
        display = line.GetDisplayNode()
        if display is not None:
            _set_display_color(display, color)
            display.SetVisibility(True)
            try:
                display.SetVisibility2D(True)
            except Exception:
                pass
            try:
                display.SetVisibility3D(True)
            except Exception:
                pass
            for method, value in (
                ("SetLineThickness", float(thickness)),
                ("SetGlyphScale", float(glyph_scale)),
                ("SetTextScale", 0.0),
            ):
                if hasattr(display, method):
                    getattr(display, method)(value)
            for method in ("SetPointLabelsVisibility", "SetPropertiesLabelVisibility"):
                if hasattr(display, method):
                    getattr(display, method)(False)
        self.group_node(line, "Loads")
        return line

    def label_centroid_ras(self, label_node, label, reference_node=None):
        if label_node is None:
            return None
        try:
            array = np.asarray(slicer.util.arrayFromVolume(label_node))
        except Exception:
            return None
        indices_zyx = np.argwhere(array == int(label))
        if indices_zyx.size == 0:
            return None
        center_zyx = indices_zyx.mean(axis=0)
        ijk = (float(center_zyx[2]), float(center_zyx[1]), float(center_zyx[0]))
        geometry_node = reference_node or label_node
        ijk_to_ras = vtk.vtkMatrix4x4()
        geometry_node.GetIJKToRASMatrix(ijk_to_ras)
        return tuple(float(value) for value in ijk_to_ras.MultiplyPoint([*ijk, 1.0])[:3])

    def label_sample_points_ras(self, label_node, label, reference_node=None, *, max_points=384):
        if label_node is None:
            return []
        try:
            array = np.asarray(slicer.util.arrayFromVolume(label_node))
        except Exception:
            return []
        indices_zyx = np.argwhere(array == int(label))
        if indices_zyx.size == 0:
            return []
        max_points = max(1, int(max_points))
        indices_zyx = _grid_sample_indices_zyx(indices_zyx, max_points=max_points)
        geometry_node = reference_node or label_node
        ijk_to_ras = vtk.vtkMatrix4x4()
        geometry_node.GetIJKToRASMatrix(ijk_to_ras)
        points = []
        for index in indices_zyx:
            ijk = (float(index[2]), float(index[1]), float(index[0]))
            points.append(tuple(float(value) for value in ijk_to_ras.MultiplyPoint([*ijk, 1.0])[:3]))
        return points

    def create_point_markers(self, name, points, color, *, glyph_scale=1.5):
        self.remove_named_node(name)
        if not points:
            return None
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        for point in points:
            try:
                node.AddControlPointWorld(vtk.vtkVector3d(*point))
            except Exception:
                node.AddControlPoint(vtk.vtkVector3d(*point))
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        if display is not None:
            _set_display_color(display, color)
            display.SetVisibility(True)
            try:
                display.SetVisibility2D(True)
                display.SetVisibility3D(True)
            except Exception:
                pass
            for method, value in (
                ("SetGlyphScale", float(glyph_scale)),
                ("SetTextScale", 0.0),
            ):
                if hasattr(display, method):
                    getattr(display, method)(value)
            if hasattr(display, "SetPointLabelsVisibility"):
                display.SetPointLabelsVisibility(False)
        self.group_node(node, "Loads")
        return node

    def create_point_cloud_model(
        self,
        name,
        points,
        color,
        *,
        folder_name="Debug",
        point_size=5.0,
        opacity=0.95,
    ):
        self.remove_named_node(name)
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("point cloud points must have shape (n, 3)")
        array = array[np.all(np.isfinite(array), axis=1)]
        if array.shape[0] == 0:
            return None

        vtk_points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        for point in array:
            point_id = vtk_points.InsertNextPoint(
                float(point[0]),
                float(point[1]),
                float(point[2]),
            )
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(point_id)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetVerts(vertices)

        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        model.SetAndObservePolyData(polydata)
        model.CreateDefaultDisplayNodes()
        display = model.GetDisplayNode()
        if display is not None:
            _set_display_color(display, color)
            display.SetOpacity(float(opacity))
            display.SetVisibility(True)
            try:
                display.SetVisibility2D(True)
                display.SetVisibility3D(True)
            except Exception:
                pass
            if hasattr(display, "SetPointSize"):
                display.SetPointSize(float(point_size))
        self.group_node(model, folder_name)
        return model

    def create_deformed_point_cloud(
        self,
        volume_node,
        component_paths,
        *,
        scale=10.0,
        max_points=20000,
    ):
        if volume_node is None:
            raise ValueError("Select the input volume before showing the deformed model.")
        self.remove_named_node("ParOSol_deformed_model")
        material = np.asarray(slicer.util.arrayFromVolume(volume_node))
        components = self._read_displacement_components(component_paths)
        for axis, component in components.items():
            if component.shape != material.shape:
                raise ValueError(
                    f"displacement_{axis} shape {component.shape} does not match "
                    f"input image shape {material.shape}."
                )

        displacement_norm = np.sqrt(
            components["x"] ** 2 + components["y"] ** 2 + components["z"] ** 2
        )
        active_zyx = np.argwhere(displacement_norm > 0)
        if active_zyx.size == 0:
            active_zyx = np.argwhere(material != 0)
        if active_zyx.size == 0:
            raise ValueError("No non-zero voxels found for the deformed model preview.")
        sampled_zyx = _grid_sample_indices_zyx(active_zyx, max_points=int(max_points))

        ijk_to_ras = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras)
        axis_dirs = _ijk_axis_directions_ras(ijk_to_ras)

        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        scale = float(scale)
        for k, j, i in sampled_zyx:
            ras = ijk_to_ras.MultiplyPoint([float(i), float(j), float(k), 1.0])[:3]
            displacement = (
                float(components["x"][k, j, i]),
                float(components["y"][k, j, i]),
                float(components["z"][k, j, i]),
            )
            offset = [
                scale
                * sum(displacement[axis] * axis_dirs[axis][coord] for axis in range(3))
                for coord in range(3)
            ]
            point_id = points.InsertNextPoint(
                float(ras[0]) + offset[0],
                float(ras[1]) + offset[1],
                float(ras[2]) + offset[2],
            )
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(point_id)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)

        model = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "ParOSol_deformed_model"
        )
        model.SetAndObservePolyData(polydata)
        model.CreateDefaultDisplayNodes()
        display = model.GetDisplayNode()
        if display is not None:
            _set_display_color(display, (1.0, 0.45, 0.05))
            display.SetOpacity(0.85)
            display.SetVisibility(True)
            try:
                display.SetVisibility2D(True)
                display.SetVisibility3D(True)
            except Exception:
                pass
            if hasattr(display, "SetPointSize"):
                display.SetPointSize(3.0)
        self.group_node(model, "Results")
        return model

    def create_deformation_arrow_glyphs(
        self,
        volume_node,
        component_paths,
        *,
        scale=10.0,
        max_points=1200,
        use_material_mask=True,
        color_field_path=None,
        color_field_name="SED",
    ):
        if volume_node is None:
            raise ValueError("Select the input volume before showing deformation arrows.")
        self.remove_named_node("ParOSol_deformed_model")
        self.remove_named_node("ParOSol_deformation_arrows")
        for bin_index in range(5):
            self.remove_named_node(f"ParOSol_deformation_arrows_bin_{bin_index + 1}")
        material = np.asarray(slicer.util.arrayFromVolume(volume_node))
        components = self._read_displacement_components(component_paths)
        color_field = None
        if color_field_path is not None and Path(color_field_path).exists():
            color_node = _load_volume_node(
                str(color_field_path),
                {"name": "ParOSol_deformation_color_probe", "show": False},
            )
            try:
                color_field = np.array(slicer.util.arrayFromVolume(color_node), copy=True)
            finally:
                self.remove_node(color_node)
        for axis, component in components.items():
            if component.shape != material.shape:
                raise ValueError(
                    f"displacement_{axis} shape {component.shape} does not match "
                    f"input image shape {material.shape}."
                )
        if color_field is not None and color_field.shape != material.shape:
            color_field = None

        displacement_norm = np.sqrt(
            components["x"] ** 2 + components["y"] ** 2 + components["z"] ** 2
        )
        if use_material_mask:
            active_zyx = np.argwhere((displacement_norm > 0) & (material != 0))
        else:
            active_zyx = np.empty((0, 3), dtype=np.int64)
        if active_zyx.size == 0:
            active_zyx = np.argwhere(displacement_norm > 0)
        if active_zyx.size == 0:
            raise ValueError("No non-zero displacement voxels found for deformation arrows.")
        sampled_zyx = _grid_sample_indices_zyx(active_zyx, max_points=int(max_points))

        ijk_to_ras = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras)
        axis_dirs = _ijk_axis_directions_ras(ijk_to_ras)
        scale = float(scale)
        raw_vectors = []
        for k, j, i in sampled_zyx:
            displacement = (
                float(components["x"][k, j, i]),
                float(components["y"][k, j, i]),
                float(components["z"][k, j, i]),
            )
            ras_vector = [
                sum(displacement[axis] * axis_dirs[axis][coord] for axis in range(3))
                for coord in range(3)
            ]
            raw_length = _vector_length(ras_vector)
            if raw_length <= 1.0e-12:
                continue
            start = ijk_to_ras.MultiplyPoint([float(i), float(j), float(k), 1.0])[:3]
            color_value = float(color_field[k, j, i]) if color_field is not None else raw_length
            raw_vectors.append((start, ras_vector, raw_length, color_value))
        if not raw_vectors:
            raise ValueError("No non-zero displacement vectors found for deformation arrows.")
        lengths = np.asarray([item[2] for item in raw_vectors], dtype=np.float64)
        color_values = np.asarray([item[3] for item in raw_vectors], dtype=np.float64)
        finite_color_values = color_values[np.isfinite(color_values)]
        use_color_field = color_field is not None and finite_color_values.size > 0
        max_raw_length = float(np.max(lengths))
        spacing = [abs(float(value)) for value in volume_node.GetSpacing() if abs(float(value)) > 0]
        max_display_length = max(float(scale) * min(spacing or [1.0]), min(spacing or [1.0]))
        binned_vectors = [[] for _ in range(5)]
        bin_source = finite_color_values if use_color_field else lengths
        bin_edges = np.quantile(bin_source, [0.2, 0.4, 0.6, 0.8])
        for start, ras_vector, raw_length, color_value in raw_vectors:
            display_length = (float(raw_length) / max_raw_length) * max_display_length
            if display_length <= 1.0e-6:
                continue
            bin_value = float(color_value) if use_color_field and np.isfinite(color_value) else float(raw_length)
            bin_index = int(np.searchsorted(bin_edges, bin_value, side="right"))
            binned_vectors[min(bin_index, 4)].append((start, ras_vector, display_length))
        colors = [
            (0.20, 0.35, 1.00),
            (0.00, 0.70, 0.85),
            (0.10, 0.75, 0.25),
            (1.00, 0.70, 0.05),
            (1.00, 0.18, 0.05),
        ]
        models = []
        for bin_index, vectors in enumerate(binned_vectors):
            if not vectors:
                continue
            model = self.create_arrow_glyph_model(
                f"ParOSol_deformation_arrows_bin_{bin_index + 1}",
                vectors,
                colors[bin_index],
                folder_name="Results",
            )
            if model is not None:
                models.append(model)
        stats = {
            "count": int(sum(len(vectors) for vectors in binned_vectors)),
            "min_mm": float(np.min(lengths)),
            "median_mm": float(np.median(lengths)),
            "max_mm": max_raw_length,
            "max_display_mm": float(max_display_length),
            "bin_counts": [int(len(vectors)) for vectors in binned_vectors],
            "color_mode": str(color_field_name) if use_color_field else "displacement magnitude",
            "color_min": float(np.min(bin_source)),
            "color_median": float(np.median(bin_source)),
            "color_max": float(np.max(bin_source)),
        }
        return (models[0] if models else None), stats

    def _read_displacement_components(self, component_paths):
        components = {}
        temporary_nodes = []
        try:
            for axis in ("x", "y", "z"):
                path = Path(component_paths[axis])
                if not path.exists():
                    raise ValueError(f"Missing displacement component: {path}")
                node = _load_volume_node(
                    str(path),
                    {"name": f"ParOSol_displacement_{axis}", "show": False},
                )
                if node is None:
                    raise ValueError(f"Could not load displacement component: {path}")
                temporary_nodes.append(node)
                components[axis] = np.array(slicer.util.arrayFromVolume(node), copy=True)
        finally:
            for node in temporary_nodes:
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass
        return components

    def reset_labelmap_like(self, label_node, volume_node):
        if label_node is None:
            return self.create_labelmap_like(volume_node, "ParOSol_labelmap")
        reference = slicer.util.arrayFromVolume(volume_node)
        slicer.util.updateVolumeFromArray(
            label_node, np.zeros(reference.shape, dtype=np.uint16)
        )
        label_node.CopyOrientation(volume_node)
        label_node.CreateDefaultDisplayNodes()
        label_node.Modified()
        return label_node

    def pad_volume_for_projected_contacts(
        self,
        volume_node,
        mask_node,
        rows,
        *,
        target_values=None,
    ):
        if volume_node is None:
            raise ValueError("Input volume is required")
        pad_before, pad_after = _required_projected_contact_padding_ijk(
            volume_node,
            mask_node or volume_node,
            rows,
            target_values=target_values,
        )
        if not np.any(pad_before) and not np.any(pad_after):
            return volume_node, mask_node, False

        padded_image = _pad_volume_node(
            volume_node,
            "ParOSol_padded_image",
            pad_before,
            pad_after,
            label=volume_node.IsA("vtkMRMLLabelMapVolumeNode"),
        )
        self.group_node(padded_image, "Inputs")

        padded_mask = None
        if mask_node is not None:
            mask_array = np.asarray(_array_from_mask_like(mask_node, volume_node))
            padded_mask = _pad_volume_node(
                volume_node,
                "ParOSol_padded_mask",
                pad_before,
                pad_after,
                label=True,
                source_array=mask_array.astype(np.uint16, copy=False),
            )
            self.style_labelmap(padded_mask, "mask")
            self.group_node(padded_mask, "Inputs")

        return padded_image, padded_mask, True

    def crop_to_mask(self, volume_node, mask_node, *, margin_voxels=2, padding_mm=None):
        if volume_node is None:
            raise ValueError("Select an image/material volume first.")
        if mask_node is None:
            raise ValueError("Select a mask or segmentation before cropping.")
        image_array = np.asarray(slicer.util.arrayFromVolume(volume_node))
        mask_array = np.asarray(_array_from_mask_like(mask_node, volume_node))
        if mask_array.shape != image_array.shape:
            raise ValueError(
                f"Mask shape {mask_array.shape} does not match image shape {image_array.shape}."
            )
        foreground = np.argwhere(mask_array != 0)
        if foreground.size == 0:
            raise ValueError("Mask is empty; cannot crop to mask.")

        margin_zyx = _crop_margin_zyx(volume_node, margin_voxels=margin_voxels, padding_mm=padding_mm)
        requested_mins = foreground.min(axis=0) - margin_zyx
        requested_maxs = foreground.max(axis=0) + margin_zyx + 1
        cropped_image_array, source_slices, target_slices = _padded_crop_array(
            image_array,
            requested_mins,
            requested_maxs,
        )
        cropped_mask_array, _mask_source_slices, _mask_target_slices = _padded_crop_array(
            mask_array,
            requested_mins,
            requested_maxs,
        )
        z0, y0, x0 = (int(value) for value in requested_mins)
        offset_ijk = (x0, y0, z0)

        self.remove_named_node("ParOSol_cropped_image")
        self.remove_named_node("ParOSol_cropped_mask")

        image_crop = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", "ParOSol_cropped_image"
        )
        slicer.util.updateVolumeFromArray(image_crop, cropped_image_array)
        _copy_cropped_geometry(volume_node, image_crop, offset_ijk)
        image_crop.CreateDefaultDisplayNodes()
        self.group_node(image_crop, "Inputs")

        mask_crop = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "ParOSol_cropped_mask"
        )
        slicer.util.updateVolumeFromArray(
            mask_crop, cropped_mask_array.astype(np.uint16, copy=False)
        )
        _copy_cropped_geometry(volume_node, mask_crop, offset_ijk)
        mask_crop.CreateDefaultDisplayNodes()
        self.style_labelmap(mask_crop, "mask")
        self.group_node(mask_crop, "Inputs")

        return image_crop, mask_crop, {
            "offset_ijk": offset_ijk,
            "shape_before_zyx": tuple(int(value) for value in image_array.shape),
            "shape_after_zyx": tuple(int(value) for value in cropped_image_array.shape),
            "source_slices_zyx": tuple((sl.start, sl.stop) for sl in source_slices),
            "target_slices_zyx": tuple((sl.start, sl.stop) for sl in target_slices),
        }

    def crop_to_mask_aspect_ratio(
        self,
        volume_node,
        mask_node,
        *,
        ratio_zyx,
        crop_from_zyx=None,
    ):
        if volume_node is None:
            raise ValueError("Select an image/material volume first.")
        if mask_node is None:
            raise ValueError("Select a mask or segmentation before aspect-ratio cropping.")
        image_array = np.asarray(slicer.util.arrayFromVolume(volume_node))
        mask_array = np.asarray(_array_from_mask_like(mask_node, volume_node))
        if mask_array.shape != image_array.shape:
            raise ValueError(
                f"Mask shape {mask_array.shape} does not match image shape {image_array.shape}."
            )
        ratio_zyx = _aspect_ratio_zyx(ratio_zyx)
        slices, offset_ijk, bounds_info = _aspect_ratio_crop_slices_zyx(
            mask_array,
            ratio_zyx,
            crop_from_zyx=crop_from_zyx,
            spacing_xyz=volume_node.GetSpacing(),
        )
        cropped_image_array = image_array[slices]
        cropped_mask_array = mask_array[slices]

        self.remove_named_node("ParOSol_aspect_ratio_image")
        self.remove_named_node("ParOSol_aspect_ratio_mask")

        image_crop = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", "ParOSol_aspect_ratio_image"
        )
        slicer.util.updateVolumeFromArray(image_crop, cropped_image_array)
        _copy_cropped_geometry(volume_node, image_crop, offset_ijk)
        image_crop.CreateDefaultDisplayNodes()
        self.group_node(image_crop, "Inputs")

        mask_crop = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "ParOSol_aspect_ratio_mask"
        )
        slicer.util.updateVolumeFromArray(
            mask_crop, cropped_mask_array.astype(np.uint16, copy=False)
        )
        _copy_cropped_geometry(volume_node, mask_crop, offset_ijk)
        mask_crop.CreateDefaultDisplayNodes()
        self.style_labelmap(mask_crop, "mask")
        self.group_node(mask_crop, "Inputs")

        info = {
            "offset_ijk": offset_ijk,
            "ratio_zyx": ratio_zyx,
            "crop_from_zyx": crop_from_zyx,
            "shape_before_zyx": tuple(int(value) for value in image_array.shape),
            "shape_after_zyx": tuple(int(value) for value in cropped_image_array.shape),
        }
        info.update(bounds_info)
        return image_crop, mask_crop, info

    def foreground_mask_from_volume(self, volume_node, name):
        if volume_node is None:
            raise ValueError("Select an image/material volume first.")
        array = np.asarray(slicer.util.arrayFromVolume(volume_node))
        mask = (array != 0).astype(np.uint16)
        self.remove_named_node(name)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
        slicer.util.updateVolumeFromArray(node, mask)
        node.CopyOrientation(volume_node)
        node.CreateDefaultDisplayNodes()
        self.style_labelmap(node, "mask")
        self.group_node(node, "Inputs")
        return node

    def keep_largest_connected_component(self, mask_node, reference_node, *, name):
        if mask_node is None:
            raise ValueError("Largest connected component requires a mask/label node.")
        array = np.asarray(_array_from_mask_like(mask_node, reference_node))
        foreground = array != 0
        largest = _largest_connected_component_mask(foreground)
        kept = int(np.count_nonzero(largest))
        filtered = np.where(largest, array, 0).astype(np.asarray(array).dtype, copy=False)
        self.remove_named_node(name)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
        slicer.util.updateVolumeFromArray(node, filtered.astype(np.uint16, copy=False))
        node.CopyOrientation(reference_node)
        node.CreateDefaultDisplayNodes()
        self.style_labelmap(node, "mask")
        self.group_node(node, "Inputs")
        return node, kept

    def smooth_volume_node(self, source_node, name, *, sigma_mm, label=False):
        node = _smooth_volume_node(source_node, name, sigma_mm=sigma_mm, label=label)
        self.group_node(node, "Inputs")
        return node

    def smooth_mask_node(self, mask_node, reference_node, name, *, sigma_mm):
        if mask_node is None:
            raise ValueError("Smooth mask requires a mask/label node.")
        array = np.asarray(_array_from_mask_like(mask_node, reference_node))
        binary_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", f"{name}_binary"
        )
        try:
            slicer.util.updateVolumeFromArray(binary_node, (array != 0).astype(np.float32))
            binary_node.CopyOrientation(reference_node)
            smoothed_binary = _smooth_volume_node(
                binary_node,
                f"{name}_binary_smooth",
                sigma_mm=sigma_mm,
                label=False,
            )
            try:
                smooth_array = np.asarray(slicer.util.arrayFromVolume(smoothed_binary))
                keep = smooth_array >= 0.5
            finally:
                self.remove_node(smoothed_binary)
        finally:
            self.remove_node(binary_node)
        filtered = np.where(keep, array, 0).astype(np.asarray(array).dtype, copy=False)
        self.remove_named_node(name)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
        slicer.util.updateVolumeFromArray(node, filtered.astype(np.uint16, copy=False))
        node.CopyOrientation(reference_node)
        node.CreateDefaultDisplayNodes()
        self.style_labelmap(node, "mask")
        self.group_node(node, "Inputs")
        return node

    def ensure_isotropic_inputs(
        self,
        volume_node,
        *,
        target_spacing_mm,
        spacing_tolerance_mm=1.0e-6,
        spacing_tolerance_relative=1.0e-3,
        canonicalize_within_tolerance=False,
        image_is_label=False,
        mask_node=None,
        disk_labelmap=None,
        nodeset_labelmap=None,
    ):
        if volume_node is None:
            raise ValueError("Select an image/material volume first.")
        target_spacing = float(target_spacing_mm)
        if target_spacing <= 0:
            raise ValueError("Target isotropic spacing must be positive.")
        spacing = tuple(abs(float(value)) for value in volume_node.GetSpacing())
        target_tuple = (target_spacing, target_spacing, target_spacing)
        if np.allclose(
            spacing,
            target_tuple,
            rtol=float(spacing_tolerance_relative),
            atol=float(spacing_tolerance_mm),
        ):
            return volume_node, mask_node, disk_labelmap, nodeset_labelmap, False

        self.remove_named_node("ParOSol_isotropic_image")
        self.remove_named_node("ParOSol_isotropic_mask")
        self.remove_named_node("ParOSol_isotropic_disks")
        self.remove_named_node("ParOSol_isotropic_nodesets")

        image_iso = _resample_volume_node(
            volume_node,
            "ParOSol_isotropic_image",
            target_spacing_mm=target_spacing,
            label=bool(image_is_label),
            reference_node=volume_node,
        )
        self.group_node(image_iso, "Inputs")

        mask_iso = None
        if mask_node is not None:
            mask_iso = _resample_mask_like_node(
                mask_node,
                volume_node,
                "ParOSol_isotropic_mask",
                target_spacing_mm=target_spacing,
                reference_node=image_iso,
            )
            self.style_labelmap(mask_iso, "mask")
            self.group_node(mask_iso, "Inputs")

        disk_iso = None
        if disk_labelmap is not None:
            disk_iso = _resample_volume_node(
                disk_labelmap,
                "ParOSol_isotropic_disks",
                target_spacing_mm=target_spacing,
                label=True,
                reference_node=image_iso,
            )
            self.style_labelmap(disk_iso, "disks")
            self.group_node(disk_iso, "Disks")

        nodeset_iso = None
        if nodeset_labelmap is not None:
            nodeset_iso = _resample_volume_node(
                nodeset_labelmap,
                "ParOSol_isotropic_nodesets",
                target_spacing_mm=target_spacing,
                label=True,
                reference_node=image_iso,
            )
            self.style_labelmap(nodeset_iso, "nodesets")
            self.group_node(nodeset_iso, "Loads")

        return image_iso, mask_iso, disk_iso, nodeset_iso, True

    def generate_workflow_contact_labelmaps(
        self,
        volume_node,
        contact_rows,
        *,
        target_mask_node=None,
        target_values=None,
    ):
        if generate_slicer_disk_and_nodeset_geometry is None:
            raise RuntimeError(
                "The active ParOSol-py runtime does not provide workflow geometry generation. "
                "Update ParOSol-py or set SLICER_PAROSOL_SOURCE to a current source checkout."
            )
        if volume_node is None:
            raise ValueError("Input volume is required")

        reference = np.asarray(slicer.util.arrayFromVolume(volume_node))
        disk_array = np.zeros(reference.shape, dtype=np.uint16)
        nodeset_array = np.zeros(reference.shape, dtype=np.uint16)
        base_material = np.asarray(
            _array_from_mask_like(volume_node, volume_node, apply_selection=False)
        )
        ijk_to_ras = _volume_ijk_to_ras_array(volume_node)
        has_disk_rows = False
        grouped_rows = {}

        for row, row_spec in enumerate(contact_rows or ()):
            plane = row_spec.get("plane") if isinstance(row_spec, dict) else None
            if plane is None:
                continue
            row_index = int(row_spec.get("row_index", row))
            target_for_row = row_spec.get("disk_target_values") or target_values
            target_key = _target_values_tuple(target_for_row)
            grouped_rows.setdefault(target_key, []).append((row, row_index, row_spec, plane))

        for target_key, group_rows in grouped_rows.items():
            mask = _target_mask_array(
                target_mask_node or volume_node,
                volume_node,
                active_values=target_key,
                fallback_to_nonzero=bool(target_key),
            )
            if mask is None or not np.any(mask):
                continue

            material_for_geometry = np.where(mask, np.maximum(base_material, 1), 0)
            editor = {"planes": []}
            nodeset_specs = {}
            nodeset_labels = {}
            nodeset_names = {}
            disk_labels = {}
            used_names = {}

            for row, row_index, row_spec, plane in group_rows:
                plane_spec = _workflow_plane_spec_from_slicer_row(row_spec, plane)
                base_name = str(plane_spec["name"])
                name_count = int(used_names.get(base_name, 0))
                used_names[base_name] = name_count + 1
                if name_count:
                    plane_spec = dict(plane_spec)
                    plane_spec["name"] = f"{base_name} {row_index + 1}"
                name = str(plane_spec["name"])
                nodeset_name = _safe_identifier(name or f"nodeset_{row + 1}")
                editor["planes"].append(plane_spec)

                contact = str(row_spec.get("contact", "Material disks")).strip()
                if contact == "PMMA caps":
                    contact = "Material disks"
                if contact in {"Material disks", "Connective disk"}:
                    has_disk_rows = True
                    disk_labels[name] = int(row_spec.get("disk_label", 0) or 0)

                bc_type = str(row_spec.get("bc_type", "None")).strip()
                if bc_type != "None" and contact != "Connective disk":
                    label = int(row_spec.get("nodeset_label") or row_spec.get("preview_label") or 0)
                    if label <= 0:
                        label = _generated_boundary_label_for_row(bc_type, row_index)
                    nodeset_names[name] = nodeset_name
                    nodeset_labels[nodeset_name] = label
                    nodeset_specs[nodeset_name] = {
                        "label": label,
                        "selection": "outer_face_nodes"
                        if contact == "Material disks"
                        else "interface_nodes",
                    }

            if not editor["planes"]:
                continue

            result = generate_slicer_disk_and_nodeset_geometry(
                editor,
                mask_zyx=mask,
                material_zyx=material_for_geometry,
                ijk_to_ras=ijk_to_ras,
                nodeset_specs=nodeset_specs,
                nodeset_labels=nodeset_labels,
                nodeset_names=nodeset_names,
                disk_labels=disk_labels,
            )
            row_disks = np.asarray(result.disk_labels_zyx, dtype=np.uint16)
            row_nodesets = np.asarray(result.nodeset_labels_zyx, dtype=np.uint16)
            disk_array[row_disks != 0] = row_disks[row_disks != 0]
            nodeset_array[row_nodesets != 0] = row_nodesets[row_nodesets != 0]

        disk_labelmap = None
        if has_disk_rows:
            disk_labelmap = self.create_labelmap_like(volume_node, "ParOSol_disks")
            slicer.util.updateVolumeFromArray(disk_labelmap, disk_array)
            disk_labelmap.CopyOrientation(volume_node)
            disk_labelmap.Modified()
            self.style_labelmap(disk_labelmap, "disks")

        nodesets = self.create_labelmap_like(volume_node, "ParOSol_nodesets")
        slicer.util.updateVolumeFromArray(nodesets, nodeset_array)
        nodesets.CopyOrientation(volume_node)
        nodesets.Modified()
        self.style_labelmap(nodesets, "nodesets")
        return disk_labelmap, nodesets

    def generate_disks(
        self,
        volume_node,
        output_labelmap,
        top_plane=None,
        bottom_plane=None,
        *,
        shape="anatomy",
        thickness_mm=3.0,
        intrusion_depth_mm=2.0,
        radius_mm=12.0,
        square_width_mm=24.0,
        hex_radius_mm=12.0,
        use_plane_size=True,
        anatomy_constrained=False,
        cap_mode="projected_cap",
        target_mask_node=None,
        target_values=None,
        top_label=201,
        bottom_label=202,
        projection_mode="project",
        clear=True,
    ):
        if volume_node is None:
            raise ValueError("Input volume is required")
        contact = (
            "Connective disk"
            if str(cap_mode).strip().lower() == "connective_disk"
            else "Material disks"
        )
        rows = []
        for plane, label, row_index, name in (
            (top_plane, top_label, 0, "Top disk"),
            (bottom_plane, bottom_label, 1, "Bottom disk"),
        ):
            if plane is None:
                continue
            rows.append(
                {
                    "name": name,
                    "plane": plane,
                    "row_index": row_index,
                    "contact": contact,
                    "surface_mode": projection_mode,
                    "bc_type": "None",
                    "shape": shape,
                    "anatomy_constrained": anatomy_constrained,
                    "thickness": thickness_mm,
                    "intrusion": intrusion_depth_mm,
                    "radius": radius_mm,
                    "square_width": square_width_mm,
                    "hex_radius": hex_radius_mm,
                    "use_plane_size": use_plane_size,
                    "disk_label": int(label),
                    "disk_target_values": target_values,
                }
            )
        generated_disk, generated_nodesets = self.generate_workflow_contact_labelmaps(
            volume_node,
            rows,
            target_mask_node=target_mask_node,
            target_values=target_values,
        )
        try:
            if output_labelmap is None:
                output_labelmap = self.create_labelmap_like(volume_node, "ParOSol_disks")
            elif clear:
                output_labelmap = self.reset_labelmap_like(output_labelmap, volume_node)
            existing = np.asarray(slicer.util.arrayFromVolume(output_labelmap)).copy()
            if generated_disk is not None:
                generated = np.asarray(slicer.util.arrayFromVolume(generated_disk), dtype=np.uint16)
                if clear:
                    existing = generated
                else:
                    existing[generated != 0] = generated[generated != 0]
            slicer.util.updateVolumeFromArray(output_labelmap, existing.astype(np.uint16, copy=False))
            output_labelmap.CopyOrientation(volume_node)
            output_labelmap.Modified()
            self.style_labelmap(output_labelmap, "disks")
            return output_labelmap
        finally:
            self.remove_node(generated_disk)
            self.remove_node(generated_nodesets)

    def generate_nodesets_from_disks_or_surface(
        self,
        volume_node,
        output_labelmap,
        top_plane=None,
        bottom_plane=None,
        *,
        disk_labelmap=None,
        target_mask_node=None,
        target_values=None,
        contact_target="cap_disks",
        shape="anatomy",
        thickness_mm=3.0,
        intrusion_depth_mm=2.0,
        radius_mm=12.0,
        square_width_mm=24.0,
        hex_radius_mm=12.0,
        use_plane_size=True,
        fixed_label=1,
        loaded_label=2,
        cap_disk_label=None,
        cap_nodeset_label=None,
        projection_mode="project",
        clear=True,
    ):
        if volume_node is None:
            raise ValueError("Input volume is required")
        contact = (
            "Material disks"
            if str(contact_target).strip().lower() == "cap_disks"
            else "Bone surface"
        )
        disk_source_labelmap = (
            disk_labelmap
            if contact == "Material disks" and target_mask_node is None and disk_labelmap is not None
            else None
        )
        rows = []
        if contact == "Material disks" and cap_disk_label is not None:
            for row_index, plane in enumerate((top_plane, bottom_plane)):
                if plane is None:
                    continue
                row_target_values = (
                    [int(cap_disk_label)] if disk_source_labelmap is not None else target_values
                )
                rows.append(
                    {
                        "name": f"Material disk {row_index + 1}",
                        "plane": plane,
                        "row_index": row_index,
                        "contact": contact,
                        "surface_mode": projection_mode,
                        "bc_type": "Dirichlet",
                        "shape": shape,
                        "thickness": thickness_mm,
                        "intrusion": intrusion_depth_mm,
                        "radius": radius_mm,
                        "square_width": square_width_mm,
                        "hex_radius": hex_radius_mm,
                        "use_plane_size": use_plane_size,
                        "disk_label": int(cap_disk_label),
                        "nodeset_label": int(cap_nodeset_label or loaded_label),
                        "disk_target_values": row_target_values,
                    }
                )
        elif contact == "Material disks":
            for plane, disk_label, nodeset_label, row_index, name in (
                (top_plane, 201, loaded_label, 0, "Top disk"),
                (bottom_plane, 202, fixed_label, 1, "Bottom disk"),
            ):
                if plane is None:
                    continue
                row_target_values = (
                    [int(disk_label)] if disk_source_labelmap is not None else target_values
                )
                rows.append(
                    {
                        "name": name,
                        "plane": plane,
                        "row_index": row_index,
                        "contact": contact,
                        "surface_mode": projection_mode,
                        "bc_type": "Dirichlet",
                        "shape": shape,
                        "thickness": thickness_mm,
                        "intrusion": intrusion_depth_mm,
                        "radius": radius_mm,
                        "square_width": square_width_mm,
                        "hex_radius": hex_radius_mm,
                        "use_plane_size": use_plane_size,
                        "disk_label": int(disk_label),
                        "nodeset_label": int(nodeset_label),
                        "disk_target_values": row_target_values,
                    }
                )
        else:
            for plane, nodeset_label, row_index, name in (
                (top_plane, loaded_label, 0, "Top surface"),
                (bottom_plane, fixed_label, 1, "Bottom surface"),
            ):
                if plane is None:
                    continue
                rows.append(
                    {
                        "name": name,
                        "plane": plane,
                        "row_index": row_index,
                        "contact": contact,
                        "surface_mode": projection_mode,
                        "bc_type": "Dirichlet",
                        "shape": shape,
                        "thickness": thickness_mm,
                        "intrusion": intrusion_depth_mm,
                        "radius": radius_mm,
                        "square_width": square_width_mm,
                        "hex_radius": hex_radius_mm,
                        "use_plane_size": use_plane_size,
                        "nodeset_label": int(nodeset_label),
                        "disk_target_values": target_values,
                    }
                )
        generated_disk, generated_nodesets = self.generate_workflow_contact_labelmaps(
            volume_node,
            rows,
            target_mask_node=disk_source_labelmap or target_mask_node,
            target_values=target_values,
        )
        try:
            if output_labelmap is None:
                output_labelmap = self.create_labelmap_like(volume_node, "ParOSol_nodesets")
            elif clear:
                output_labelmap = self.reset_labelmap_like(output_labelmap, volume_node)
            existing = np.asarray(slicer.util.arrayFromVolume(output_labelmap)).copy()
            generated = np.asarray(slicer.util.arrayFromVolume(generated_nodesets), dtype=np.uint16)
            if clear:
                existing = generated
            else:
                existing[generated != 0] = generated[generated != 0]
            slicer.util.updateVolumeFromArray(output_labelmap, existing.astype(np.uint16, copy=False))
            output_labelmap.CopyOrientation(volume_node)
            output_labelmap.Modified()
            self.style_labelmap(output_labelmap, "nodesets")
            return output_labelmap
        finally:
            self.remove_node(generated_disk)
            self.remove_node(generated_nodesets)

    def generate_bc_labels(
        self,
        volume_node,
        output_labelmap,
        fixed_plane=None,
        loaded_plane=None,
        *,
        thickness_mm=1.0,
        fixed_label=1,
        loaded_label=2,
    ):
        if volume_node is None:
            raise ValueError("Input volume is required")
        if output_labelmap is None:
            output_labelmap = self.create_labelmap_like(volume_node, "ParOSol_nodesets")
        else:
            output_labelmap = self.reset_labelmap_like(output_labelmap, volume_node)
        image = output_labelmap.GetImageData()
        dims = image.GetDimensions()
        scalars = image.GetPointData().GetScalars()
        scalars.Fill(0)
        ijk_to_ras = vtk.vtkMatrix4x4()
        output_labelmap.GetIJKToRASMatrix(ijk_to_ras)

        for plane, label in ((fixed_plane, fixed_label), (loaded_plane, loaded_label)):
            if plane is None:
                continue
            center = [0.0, 0.0, 0.0]
            normal = [0.0, 0.0, 1.0]
            plane.GetCenter(center)
            plane.GetNormal(normal)
            normal = _normalized(normal)
            for k in range(dims[2]):
                for j in range(dims[1]):
                    for i in range(dims[0]):
                        ras = ijk_to_ras.MultiplyPoint([i, j, k, 1.0])[:3]
                        rel = [ras[d] - center[d] for d in range(3)]
                        distance = abs(sum(rel[d] * normal[d] for d in range(3)))
                        if distance <= float(thickness_mm):
                            scalars.SetTuple1(image.ComputePointId([i, j, k]), label)
        output_labelmap.Modified()
        return output_labelmap

    def export_volume(self, node, path):
        if node is None:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        original_storage = node.GetStorageNode()
        original_storage_id = original_storage.GetID() if original_storage is not None else None
        export_storage = node.CreateDefaultStorageNode()
        if export_storage is None:
            raise RuntimeError(f"Could not create a storage node for {node.GetName()}")
        slicer.mrmlScene.AddNode(export_storage)
        try:
            node.SetAndObserveStorageNodeID(export_storage.GetID())
            export_storage.SetFileName(str(path))
            if hasattr(export_storage, "SetUseCompression"):
                export_storage.SetUseCompression(True)
            if not slicer.util.saveNode(node, str(path)):
                raise RuntimeError(f"Failed to save {node.GetName()} to {path}")
        finally:
            try:
                node.SetAndObserveStorageNodeID(original_storage_id)
            except TypeError:
                node.SetAndObserveStorageNodeID(original_storage_id or "")
            if slicer.mrmlScene.IsNodePresent(export_storage):
                slicer.mrmlScene.RemoveNode(export_storage)
        return path

    def export_mask_like(self, node, reference_node, path):
        if node is None:
            return None
        if not _is_segmentation_node(node):
            return self.export_volume(node, path)
        label = _segmentation_to_labelmap_node(node, reference_node)
        try:
            return self.export_volume(label, path)
        finally:
            self.remove_node(label)

    def build_config(
        self,
        *,
        image_path,
        mask_path=None,
        nodeset_path=None,
        profile="XtremeCTII",
        output_dir,
        force_n=None,
        displacement_value=1.0,
        direction_vector=(0.0, 0.0, -1.0),
        loaded_label=2,
        disk_labels=None,
        disk_materials=None,
        disk_e_mpa=3000.0,
        disk_nu=0.3,
        material_override=None,
        nodeset_specs=None,
        load_case_override=None,
        mpi_processes=None,
        mpi_launcher=None,
        tolerance=None,
        export_displacements=False,
        output_fields=None,
        postprocess_config=None,
    ):
        profile_config = _profile_defaults(profile)
        output_fields = list(output_fields or (["sed", "displacements"] if bool(export_displacements) else ["sed"]))
        config = {
            "case": {
                "name": Path(output_dir).name or "slicer_case",
                "work_dir": str(Path(output_dir)),
            },
            "input": {
                "image": str(image_path),
                "image_type": profile_config["image_type"],
                "spacing": "auto",
                "origin": "auto",
            },
            "materials": profile_config["materials"],
            "load_case": profile_config["load_case"],
            "output": {
                "result": str(Path(output_dir) / "result.json"),
                "run_summary": str(Path(output_dir) / "summary.json"),
                "fields": output_fields,
                "export_fields": True,
                "fields_dir": str(Path(output_dir) / "fields"),
                "visualize": True,
                "visualization": str(Path(output_dir) / "overview.png"),
            },
            "solver": self.default_solver_config(),
            "postprocess": {},
        }
        config["postprocess"] = copy.deepcopy(postprocess_config) if postprocess_config is not None else _postprocess_preset_config("pistoia")
        config["solver"]["outputs"] = output_fields
        if material_override:
            if "image_type" in material_override:
                config["input"]["image_type"] = material_override["image_type"]
            if "materials" in material_override:
                config["materials"] = material_override["materials"]
            if "solver" in material_override:
                config["solver"] = _deep_merge_workflow_config(
                    config.get("solver", {}),
                    material_override["solver"],
                )
        if mpi_processes is not None:
            config["solver"]["mpi_processes"] = max(1, int(mpi_processes))
            if int(config["solver"]["mpi_processes"]) <= 1:
                config["solver"]["mpi_launcher"] = ""
            elif mpi_launcher:
                config["solver"]["mpi_launcher"] = str(mpi_launcher)
        if tolerance is not None:
            config["solver"]["tolerance"] = float(tolerance)
        if disk_materials:
            labels = config["materials"].setdefault("labels", {})
            for label, material in disk_materials.items():
                labels.setdefault(
                    int(label),
                    {
                        "name": material.get("name", "boundary_disk"),
                        "E": float(material.get("E", disk_e_mpa)),
                        "nu": float(material.get("nu", disk_nu)),
                    },
                )
        elif disk_labels:
            labels = config["materials"].setdefault("labels", {})
            for label in disk_labels:
                labels.setdefault(
                    int(label),
                    {
                        "name": "boundary_disk",
                        "E": float(disk_e_mpa),
                        "nu": float(disk_nu),
                    },
                )
        if mask_path:
            config["input"]["mask"] = str(mask_path)
        if nodeset_path and nodeset_specs is not None and load_case_override is not None:
            config["nodesets"] = nodeset_specs
            config["load_case"] = load_case_override
            return config
        if nodeset_path:
            config["nodesets"] = {
                "fixed": {
                    "type": "label_image",
                    "image": str(nodeset_path),
                    "label": 1,
                    "selection": "surface_nodes",
                },
                "loaded": {
                    "type": "label_image",
                    "image": str(nodeset_path),
                    "label": int(loaded_label),
                    "selection": "surface_nodes",
                },
            }
            direction = _normalized(direction_vector)
            nonzero_components = []
            for axis, component in zip(("x", "y", "z"), direction):
                if abs(component) > 1e-12:
                    nonzero_components.append((axis, component))
            if not nonzero_components:
                nonzero_components = [("z", -1.0)]
            config["load_case"] = {
                "type": "nodeset",
                "fixed": [{"nodeset": "fixed", "dofs": ["x", "y", "z"], "value": 0.0}],
            }
            if force_n is None:
                config["load_case"]["prescribed"] = [
                    {
                        "nodeset": "loaded",
                        "dof": axis,
                        "value": float(displacement_value) * float(component),
                    }
                    for axis, component in nonzero_components
                ]
            else:
                config["load_case"]["loaded"] = [
                    {
                        "nodeset": "loaded",
                        "dof": axis,
                        "value": float(force_n) * float(component),
                        "distribute": True,
                    }
                    for axis, component in nonzero_components
                ]
        return config

    def default_solver_config(self):
        return {
            "mpi_processes": 4,
            "mpi_launcher": "",
            "tolerance": 1e-4,
            "outputs": ["sed"],
        }

    def write_config(self, config, path):
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def export_displacement_components_from_run(self, output_dir, *, on_output=None):
        output_dir = Path(output_dir).expanduser().resolve()
        input_h5 = output_dir / "parosol_input.h5"
        material_image = output_dir / "slicer_input.nii.gz"
        mask_image = output_dir / "slicer_mask.nii.gz"
        fields_dir = output_dir / "fields"
        if not input_h5.exists():
            raise FileNotFoundError(f"Missing ParOSol input/output H5: {input_h5}")
        if not material_image.exists():
            raise FileNotFoundError(f"Missing exported material image: {material_image}")
        python = Path(DEFAULT_PAROSOL).resolve().parent / "python"
        if not python.exists():
            python = Path(sys.executable)
        script = r"""
from pathlib import Path
import sys
import h5py
import numpy as np
import SimpleITK as sitk
from parosol_py.field_export import NativeFieldMapper
from parosol_py.images import ImageGrid, export_scalar_image

input_h5 = Path(sys.argv[1])
material_image = Path(sys.argv[2])
mask_image = Path(sys.argv[3])
fields_dir = Path(sys.argv[4])
fields_dir.mkdir(parents=True, exist_ok=True)

image = sitk.ReadImage(str(material_image))
material_zyx = sitk.GetArrayFromImage(image)
stiffness_xyz = np.transpose(np.asarray(material_zyx), (2, 1, 0))
with h5py.File(input_h5, "r") as h5:
    disp = np.asarray(h5["Solution"]["disp"][...])
    mesh_coords = np.asarray(h5["Mesh"]["Coordinates"][...]) if "Mesh" in h5 and "Coordinates" in h5["Mesh"] else None
    mesh_elements = np.asarray(h5["Mesh"]["Elements"][...]) if "Mesh" in h5 and "Elements" in h5["Mesh"] else None
mapper = NativeFieldMapper(stiffness_xyz)
if mesh_coords is not None and mesh_elements is not None and disp.shape[0] == mesh_coords.shape[0]:
    corner_coords = mesh_coords[mesh_elements]
    element_coords = np.floor(corner_coords.min(axis=1)).astype(np.int64)
    in_bounds = (
        (element_coords[:, 0] >= 0)
        & (element_coords[:, 0] < stiffness_xyz.shape[0])
        & (element_coords[:, 1] >= 0)
        & (element_coords[:, 1] < stiffness_xyz.shape[1])
        & (element_coords[:, 2] >= 0)
        & (element_coords[:, 2] < stiffness_xyz.shape[2])
    )
    if not np.all(in_bounds):
        raise ValueError("mesh element coordinates fall outside dense image bounds")
    dense = np.zeros((*stiffness_xyz.shape, 3), dtype=disp.dtype)
    dense[
        element_coords[:, 0],
        element_coords[:, 1],
        element_coords[:, 2],
    ] = np.mean(disp[mesh_elements], axis=1)
else:
    dense = mapper.nodal_vector_to_dense_element(disp)
mask_xyz = None
if mask_image.exists():
    mask_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_image))) != 0
    mask_xyz = np.transpose(np.asarray(mask_zyx), (2, 1, 0))
for component, axis in enumerate(("x", "y", "z")):
    array_xyz = dense[..., component]
    if mask_xyz is not None:
        array_xyz = np.where(mask_xyz, array_xyz, 0.0)
    export_scalar_image(
        ImageGrid(
            array_xyz=array_xyz,
            spacing=tuple(float(v) for v in image.GetSpacing()),
            origin=tuple(float(v) for v in image.GetOrigin()),
        ),
        fields_dir / f"displacement_{axis}.nii.gz",
    )
"""
        completed = subprocess.run(
            [str(python), "-c", script, str(input_h5), str(material_image), str(mask_image), str(fields_dir)],
            cwd=str(output_dir),
            env=self.parosol_environment(),
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to export displacement components from ParOSol H5\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        if on_output:
            on_output("Exported displacement_x/y/z fields from existing ParOSol H5.\n")
        return {
            axis: fields_dir / f"displacement_{axis}.nii.gz"
            for axis in ("x", "y", "z")
        }

    def slicer_python_tags(self):
        version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        soabi = str(sysconfig.get_config_var("SOABI") or "")
        return {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "implementation": version,
            "abi": version if version in soabi else soabi,
            "platform": platform_tag,
            "machine": os.uname().machine if hasattr(os, "uname") else "",
        }

    def expected_wheel_patterns(self):
        tags = self.slicer_python_tags()
        implementation = tags["implementation"]
        platform_tag = tags["platform"]
        return [
            f"parosol_py-*-{implementation}-{implementation}-{platform_tag}.whl",
            f"parosol_py-*-{implementation}-abi3-{platform_tag}.whl",
        ]

    def slicer_local_parosol(self):
        exe_dir = Path(sys.executable).resolve().parent
        scripts_dir = Path(sysconfig.get_path("scripts") or exe_dir)
        candidates = [
            scripts_dir / "parosol",
            scripts_dir / "parosol.exe",
            exe_dir / "parosol",
            exe_dir / "Scripts" / "parosol.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def parosol_executable(self):
        return " ".join(self.parosol_command_base())

    def parosol_command_base(self):
        configured = self.setting_value("SlicerParOSol/parosolExecutable", "")
        candidates = [configured]
        for candidate in candidates:
            if candidate and Path(str(candidate)).exists():
                return [str(candidate)]
        slicer_local = self.slicer_local_parosol()
        if slicer_local:
            if self._parosol_source_checkout_path() is not None:
                return [self.python_launcher(), "-c", PAROSOL_SOURCE_CLI_BOOTSTRAP]
            return [self.python_launcher(), "-m", "parosol_py.cli"]
        path_candidate = shutil.which("parosol", path=self.parosol_environment().get("PATH"))
        if path_candidate:
            return [path_candidate]
        return ["parosol"]

    def run_parosol(self, args, on_output=None, on_finished=None, cwd=None):
        if self._proc is not None:
            raise RuntimeError("A ParOSol process is already running")
        self._user_terminated = False
        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.MergedChannels)
        env = qt.QProcessEnvironment()
        for key, value in self.parosol_environment().items():
            env.insert(key, value)
        proc.setProcessEnvironment(env)

        def _read_output():
            raw = proc.readAll()
            text = _decode_process_output(raw)
            text = _filter_runtime_noise(text)
            if on_output and text:
                on_output(text)

        def _finished(*signal_args):
            interrupted = bool(self._user_terminated)
            self._user_terminated = False
            self._proc = None
            exit_code = int(signal_args[0]) if signal_args else int(proc.exitCode())
            if on_finished:
                on_finished(exit_code, interrupted)

        proc.readyRead.connect(_read_output)
        proc.finished.connect(_finished)
        command = self.parosol_command_base() + list(args)
        if cwd is not None:
            proc.setWorkingDirectory(str(Path(cwd)))
        if on_output:
            on_output(f"[process] launching: {' '.join(command)}\n")
        proc.start(command[0], command[1:])
        if not proc.waitForStarted(3000):
            raise RuntimeError("Failed to start ParOSol process")
        self._proc = proc

    def _parosol_source_checkout_path(self):
        for source_path in _parosol_source_checkout_import_paths():
            if source_path.is_dir():
                return source_path
        return None

    def parosol_environment(self):
        env = dict(os.environ)
        for key in (
            "PYTHONHOME",
            "PYTHONEXECUTABLE",
            "PYTHONUSERBASE",
            "PYTHONSAFEPATH",
            "__PYVENV_LAUNCHER__",
        ):
            env.pop(key, None)
        path_parts = [str(Path(sys.executable).resolve().parent)]
        configured = self.setting_value("SlicerParOSol/parosolExecutable", "")
        for executable in (configured, DEFAULT_PAROSOL):
            if executable:
                executable_path = Path(str(executable))
                if executable_path.exists():
                    path_parts.append(str(executable_path.resolve().parent))
        current_path = env.get("PATH", "")
        if current_path:
            path_parts.append(current_path)
        env["PATH"] = os.pathsep.join(_unique_paths(path_parts))
        source_path = self._parosol_source_checkout_path()
        if source_path is not None:
            env["SLICER_PAROSOL_SOURCE"] = str(source_path)
            env["PYTHONPATH"] = str(source_path)
        else:
            env.pop("SLICER_PAROSOL_SOURCE", None)
            env.pop("PYTHONPATH", None)
        env["PYTHONUNBUFFERED"] = "1"
        env["ITK_AUTOLOAD_PATH"] = ""
        env["SITK_AUTOLOAD_PATH"] = ""
        return env

    def compatible_wheel_message(self, wheel_path):
        path = Path(wheel_path)
        if path.suffix != ".whl":
            return False, "Selected file is not a .whl file."
        parts = path.name[:-4].split("-")
        if len(parts) < 5:
            return False, "Wheel filename does not contain standard Python tags."
        py_tag, abi_tag, platform_tag = parts[-3:]
        try:
            from pip._vendor.packaging.tags import parse_tag, sys_tags

            wheel_tags = parse_tag("-".join((py_tag, abi_tag, platform_tag)))
            supported = set(sys_tags())
            if wheel_tags & supported:
                return True, "Wheel tags match this Slicer Python."
        except Exception:
            pass
        tags = self.slicer_python_tags()
        python_ok = py_tag == tags["implementation"] or py_tag.startswith("py3")
        abi_ok = abi_tag in {tags["implementation"], "abi3", "none"} or abi_tag == tags["abi"]
        platform_tokens = set(platform_tag.split("."))
        platform_ok = platform_tag == "any" or tags["platform"] in platform_tokens
        if not python_ok:
            return False, f"Wheel Python tag {py_tag!r} does not match Slicer {tags['implementation']}."
        if not abi_ok:
            return False, f"Wheel ABI tag {abi_tag!r} does not match Slicer {tags['abi']}."
        if not platform_ok:
            return False, f"Wheel platform {platform_tag!r} does not match Slicer {tags['platform']}."
        return True, "Wheel tags match this Slicer Python."

    def install_wheel_into_slicer(self, wheel_path, *, on_output=None):
        ok, message = self.compatible_wheel_message(wheel_path)
        if on_output:
            on_output(f"{message}\n")
        if not ok:
            raise RuntimeError(message)
        command = [
            self.python_launcher(),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            str(Path(wheel_path)),
        ]
        if on_output:
            on_output(f"[runtime] installing with Slicer Python: {' '.join(command)}\n")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=self.parosol_environment(),
        )
        stdout = _filter_runtime_noise(completed.stdout)
        stderr = _filter_runtime_noise(completed.stderr)
        if on_output and stdout:
            on_output(stdout)
        if on_output and stderr:
            on_output(stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"Wheel install failed with exit code {completed.returncode}")
        local_parosol = self.slicer_local_parosol()
        if local_parosol:
            if on_output:
                on_output(
                    "[runtime] using Slicer-installed parosol through PythonSlicer -m parosol_py.cli\n"
                )
        return self.check_runtime(on_output=on_output)

    def install_pypi_into_slicer(self, *, on_output=None):
        command = [
            self.python_launcher(),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "parosol-py",
        ]
        if on_output:
            on_output(f"[runtime] installing from PyPI with Slicer Python: {' '.join(command)}\n")
        env = self.parosol_environment()
        env.pop("PYTHONPATH", None)
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        stdout = _filter_runtime_noise(completed.stdout)
        stderr = _filter_runtime_noise(completed.stderr)
        if on_output and stdout:
            on_output(stdout)
        if on_output and stderr:
            on_output(stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"PyPI install failed with exit code {completed.returncode}")
        return self.check_runtime(on_output=on_output)

    def check_runtime(self, *, on_output=None, return_text=False):
        tags = self.slicer_python_tags()
        env = self.parosol_environment()
        command_base = self.parosol_command_base()
        lines = [
            "[runtime] Slicer Python:",
            f"  executable: {tags['executable']}",
            f"  version: {tags['python']}",
            f"  wheel tags: {tags['implementation']} / {tags['abi']} / {tags['platform']}",
            "  install command:",
            "    PythonSlicer -m pip install --upgrade parosol-py",
            f"  parosol executable: {' '.join(command_base)}",
        ]
        mpi = self.discover_mpi_launcher()
        lines.append(f"  external MPI launcher override: {mpi or 'none'}")
        python_runtime = self._python_runtime_for_command(command_base, env)
        if python_runtime is not None:
            command = [
                python_runtime,
                "-c",
                (
                    "import os, sys; "
                    "source_path = os.environ.get(\"SLICER_PAROSOL_SOURCE\"); "
                    "sys.path.remove(source_path) if source_path in sys.path else None; "
                    "sys.path.insert(0, source_path) if source_path else None; "
                    "import parosol_py; "
                    "import inspect; "
                    "import parosol_py.visualization as visualization; "
                    "from parosol_py.runner import "
                    "packaged_executable, packaged_mpi_launcher; "
                    "print('parosol_py', getattr(parosol_py, '__version__', 'unknown')); "
                    "print('source', getattr(parosol_py, '__file__', 'unknown')); "
                    "print('visualization', inspect.getsourcefile(visualization)); "
                    "print('native', packaged_executable()); "
                    "print('packaged MPI launcher', packaged_mpi_launcher())"
                ),
            ]
        else:
            command = command_base + ["--help"]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if completed.returncode == 0:
            lines.append("  import: ok")
            stdout = _filter_runtime_noise(completed.stdout)
            lines.extend(f"  {line}" for line in stdout.strip().splitlines() if line)
        else:
            lines.append("  import: failed")
            detail = _filter_runtime_noise(completed.stderr or completed.stdout).strip()
            if detail:
                lines.extend(f"  {line}" for line in detail.splitlines())
        text = "\n".join(lines) + "\n"
        if on_output:
            on_output(text)
        if return_text:
            return completed.returncode == 0, text
        return completed.returncode == 0

    def _python_runtime_for_command(self, command_base, env):
        if command_base[:3] == [self.python_launcher(), "-c", PAROSOL_SOURCE_CLI_BOOTSTRAP]:
            return self.python_launcher()
        if command_base[:3] == [self.python_launcher(), "-m", "parosol_py.cli"]:
            return self.python_launcher()
        if not command_base:
            return None
        executable = shutil.which(str(command_base[0]), path=env.get("PATH")) or str(command_base[0])
        executable_path = Path(executable)
        sibling_python = executable_path.with_name("python")
        if sibling_python.exists():
            return str(sibling_python)
        sibling_python3 = executable_path.with_name("python3")
        if sibling_python3.exists():
            return str(sibling_python3)
        return None

    def export_parosol_input(self, config_path, *, on_output=None):
        command_name = "batch" if _config_file_has_batch(config_path) else "run"
        command = self.parosol_command_base() + [command_name, str(config_path), "--dry-run"]
        if on_output:
            on_output(f"[process] exporting ParOSol input: {' '.join(command)}\n")
        completed = subprocess.run(
            command,
            cwd=str(Path(config_path).parent),
            env=self.parosol_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        stdout = _filter_runtime_noise(completed.stdout)
        stderr = _filter_runtime_noise(completed.stderr)
        if on_output and stdout:
            on_output(stdout)
        if on_output and stderr:
            on_output(stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"ParOSol input export failed with return code {completed.returncode}"
            )
        if command_name == "batch":
            summary_path = _config_file_batch_summary(config_path)
            if not summary_path.exists():
                raise RuntimeError(
                    f"ParOSol batch dry-run did not write summary: {summary_path}"
                )
            if on_output:
                on_output(f"Exported ParOSol batch dry-run summary: {summary_path}\n")
            return summary_path
        summary_path = Path(config_path).parent / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"ParOSol dry-run did not write summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        input_file = Path(summary.get("outputs", {}).get("input_file", ""))
        if not input_file.exists():
            raise RuntimeError(f"ParOSol dry-run did not write input h5: {input_file}")
        if on_output:
            on_output(f"Exported ParOSol input H5: {input_file}\n")
        return input_file

    def create_portable_bundle(self, config_path, bundle_path, *, on_output=None):
        bundle_path = Path(bundle_path).expanduser().resolve()
        command = self.parosol_command_base() + [
            "bundle",
            "create",
            str(config_path),
            "--output",
            str(bundle_path),
        ]
        if on_output:
            on_output(f"[process] creating portable bundle: {' '.join(command)}\n")
        completed = subprocess.run(
            command,
            cwd=str(Path(config_path).parent),
            env=self.parosol_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        stdout = _filter_runtime_noise(completed.stdout)
        stderr = _filter_runtime_noise(completed.stderr)
        if on_output and stdout:
            on_output(stdout)
        if on_output and stderr:
            on_output(stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Portable bundle export failed with return code {completed.returncode}"
            )
        if not bundle_path.exists():
            raise RuntimeError(f"Portable bundle was not written: {bundle_path}")
        if on_output:
            on_output(f"Exported portable bundle: {bundle_path}\n")
        return bundle_path

    def interrupt(self):
        proc = self._proc
        if proc is None:
            return False
        self._user_terminated = True
        proc.terminate()
        qt.QTimer.singleShot(1500, lambda: proc.kill() if proc.state() != qt.QProcess.NotRunning else None)
        return True

    def solver_input_volume(self, volume_node, disk_labelmap=None, *, disk_material_value=None, mask_node=None):
        if volume_node is None:
            raise ValueError("Input volume is required")
        if disk_labelmap is None and mask_node is None:
            return volume_node, []
        clone = slicer.modules.volumes.logic().CloneVolume(
            slicer.mrmlScene, volume_node, f"{volume_node.GetName()}_ParOSol_solver_input"
        )
        material_array = np.asarray(slicer.util.arrayFromVolume(clone)).copy()
        changed = False
        if mask_node is not None:
            mask_array = np.asarray(_array_from_mask_like(mask_node, volume_node)) != 0
            if mask_array.shape != material_array.shape:
                raise ValueError(
                    f"Mask shape {mask_array.shape} does not match image shape {material_array.shape}."
                )
            material_array = np.where(mask_array, material_array, 0)
            changed = True
        if disk_labelmap is None:
            if changed:
                slicer.util.updateVolumeFromArray(clone, material_array)
                clone.CopyOrientation(volume_node)
                clone.Modified()
            return clone, []
        try:
            disk_array = np.asarray(slicer.util.arrayFromVolume(disk_labelmap))
        except Exception:
            return clone, []
        if disk_array.shape != material_array.shape:
            raise ValueError(
                f"Disk labelmap shape {disk_array.shape} does not match image shape {material_array.shape}."
            )
        disk_mask = disk_array > 0
        disk_labels = []
        if np.any(disk_mask):
            if disk_material_value is not None:
                disk_material_array = np.full(
                    int(np.count_nonzero(disk_mask)),
                    int(disk_material_value),
                    dtype=np.int64,
                )
            else:
                disk_material_array = np.rint(disk_array[disk_mask]).astype(np.int64, copy=False)
            material_array = _promote_material_label_array_for_values(material_array, disk_material_array)
            material_array[disk_mask] = disk_material_array.astype(material_array.dtype, copy=False)
            disk_labels = sorted(int(value) for value in np.unique(disk_material_array))
            changed = True
        if changed:
            slicer.util.updateVolumeFromArray(clone, material_array)
            clone.CopyOrientation(volume_node)
            clone.Modified()
        return clone, sorted(disk_labels)

    def normalize_xtremect_material_labels(self, volume_node):
        """Undo Slicer/ITK one-step-low AIM label values for XtremeCT label profiles."""
        if volume_node is None:
            raise ValueError("Input volume is required")
        array = np.asarray(slicer.util.arrayFromVolume(volume_node))
        needs_label_fix = bool(np.any(np.rint(array) == 99) or np.any(np.rint(array) == 126))
        needs_scalar_fix = array.dtype != np.dtype(np.int16)
        if not needs_label_fix and not needs_scalar_fix:
            return volume_node
        fixed = np.rint(array).astype(np.int16, copy=True)
        fixed[fixed == 99] = 100
        fixed[fixed == 126] = 127
        clone = slicer.modules.volumes.logic().CloneVolume(
            slicer.mrmlScene,
            volume_node,
            f"{volume_node.GetName()}_ParOSol_xtremect_labels",
        )
        slicer.util.updateVolumeFromArray(clone, fixed)
        clone.CopyOrientation(volume_node)
        clone.Modified()
        return clone

    def density_material_input_volume(
        self,
        volume_node,
        disk_labelmap=None,
        *,
        material_override,
        disk_materials=None,
        cap_e_mpa=3000.0,
        mask_node=None,
    ):
        if volume_node is None:
            raise ValueError("Input volume is required")
        density = np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=np.float64)
        active_mask = np.ones(density.shape, dtype=bool)
        if mask_node is not None:
            active_mask = np.asarray(_array_from_mask_like(mask_node, volume_node)) != 0
            if active_mask.shape != density.shape:
                raise ValueError(
                    f"Mask shape {active_mask.shape} does not match image shape {density.shape}."
                )
        density_cfg = material_override.get("materials", {}).get("density", {})
        e_cfg = density_cfg.get("E", density_cfg)
        threshold = float(density_cfg.get("active_threshold", density_cfg.get("mask_threshold", 0.0)))
        active = (density > threshold) & active_mask
        density_for_material = density
        if _enabled_value(
            density_cfg.get(
                "bin_material",
                e_cfg.get("bin_material", e_cfg.get("binned_material", False)),
            )
        ):
            density_for_material, _, _ = _ogo_binned_density_values(
                density,
                active=active,
                number_bins=density_cfg.get(
                    "number_bins",
                    density_cfg.get("bins", e_cfg.get("number_bins", e_cfg.get("bins", 128))),
                ),
            )
        youngs = _density_to_e_mpa(density_for_material, e_cfg)
        youngs = np.where(active, youngs, 0.0)
        minimum_e = _density_floor_config_value(e_cfg, density_cfg)
        maximum_e = e_cfg.get("maximum_e_mpa", density_cfg.get("maximum_e_mpa"))
        if minimum_e is not None:
            youngs = np.where(active, np.maximum(youngs, float(minimum_e)), 0.0)
        if maximum_e is not None:
            youngs = np.minimum(youngs, float(maximum_e))

        if disk_labelmap is not None:
            disk_array = np.asarray(slicer.util.arrayFromVolume(disk_labelmap))
            if disk_materials:
                for label, material in disk_materials.items():
                    youngs = np.where(
                        disk_array == int(label),
                        float(material.get("E", cap_e_mpa)),
                        youngs,
                    )
            else:
                youngs = np.where(disk_array > 0, float(cap_e_mpa), youngs)

        clone = slicer.modules.volumes.logic().CloneVolume(
            slicer.mrmlScene,
            volume_node,
            f"{volume_node.GetName()}_ParOSol_material_mpa",
        )
        slicer.util.updateVolumeFromArray(clone, youngs.astype(np.float32, copy=False))
        clone.CopyOrientation(volume_node)
        clone.Modified()
        return clone

    def remove_node(self, node):
        if node is not None and slicer.mrmlScene.IsNodePresent(node):
            _prepare_node_for_removal(node)
            slicer.mrmlScene.RemoveNode(node)

    def move_plane_along_normal(self, plane, distance_mm):
        if plane is None:
            return
        center = [0.0, 0.0, 0.0]
        normal = [0.0, 0.0, 1.0]
        plane.GetCenter(center)
        try:
            plane.GetNormalWorld(normal)
        except Exception:
            plane.GetNormal(normal)
        normal = _normalized(normal)
        plane.SetCenter(
            [
                center[index] + float(distance_mm) * normal[index]
                for index in range(3)
            ]
        )

    def scale_plane_size(self, plane, factor):
        if plane is None or not hasattr(plane, "GetSize") or not hasattr(plane, "SetSize"):
            return
        size = [0.0, 0.0]
        plane.GetSize(size)
        plane.SetSize(
            max(float(size[0]) * float(factor), 0.1),
            max(float(size[1]) * float(factor), 0.1),
        )

    def move_plane_in_plane(self, plane, axis="u", distance_mm=1.0):
        if plane is None:
            return
        center = [0.0, 0.0, 0.0]
        normal = [0.0, 0.0, 1.0]
        plane.GetCenter(center)
        try:
            plane.GetNormalWorld(normal)
        except Exception:
            plane.GetNormal(normal)
        normal = _normalized(normal)
        u_axis, v_axis = _plane_axes_from_plane(plane, normal)
        direction = u_axis if str(axis).lower() == "u" else v_axis
        plane.SetCenter(
            [
                center[index] + float(distance_mm) * float(direction[index])
                for index in range(3)
            ]
        )

    def rotate_plane(self, plane, axis="u", angle_degrees=1.0):
        if plane is None:
            return
        normal = [0.0, 0.0, 1.0]
        try:
            plane.GetNormalWorld(normal)
        except Exception:
            plane.GetNormal(normal)
        normal = _normalized(normal)
        u_axis, v_axis = _plane_axes_from_plane(plane, normal)
        axis = str(axis).strip().lower()
        if axis in {"u", "plane_u"}:
            rotation_axis = u_axis
        elif axis in {"v", "plane_v"}:
            rotation_axis = v_axis
        else:
            rotation_axis = normal
        angle_radians = math.radians(float(angle_degrees))
        rotated_u = _rotate_vector_about_axis(u_axis, rotation_axis, angle_radians)
        rotated_v = _rotate_vector_about_axis(v_axis, rotation_axis, angle_radians)
        rotated_n = _rotate_vector_about_axis(normal, rotation_axis, angle_radians)
        _set_plane_axes_world(plane, rotated_u, rotated_v, rotated_n)

    def snap_plane_rotation(self, plane, *, step_degrees=5.0):
        if plane is None:
            return
        normal = [0.0, 0.0, 1.0]
        try:
            plane.GetNormalWorld(normal)
        except Exception:
            plane.GetNormal(normal)
        normal = _snap_vector_to_angular_step(normal, step_degrees)
        current_u, _current_v = _plane_axes_from_plane(plane, normal)
        projected_u = _subtract(current_u, [normal[index] * _dot(current_u, normal) for index in range(3)])
        if _vector_length(projected_u) <= 1e-6:
            projected_u, projected_v = _plane_axes(normal)
        else:
            projected_u = _normalized(projected_u)
            projected_v = _normalized(_cross(normal, projected_u))
        _set_plane_axes_world(plane, projected_u, projected_v, normal)


class ParOSolFEAWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        self.logic = ParOSolFEALogic()
        self.topDiskPlane = None
        self.bottomDiskPlane = None
        self.fixedPlane = None
        self.loadedPlane = None
        self.contactPlaneRows = []
        self.bcArrowNodes = []
        self.bcMarkerNodes = []
        self._updating_isotropic_spacing = False
        self._isotropic_spacing_user_override = False
        self._isotropic_spacing_workflow_override = False
        self._resample_spacing_tolerance_mm = 1.0e-6
        self._resample_spacing_tolerance_relative = 1.0e-3
        self._resample_canonicalize_within_tolerance = False
        self._resample_density_interpolation = "linear"
        self._resample_density_interpolation = "linear"
        self._updating_preprocess_spacing_defaults = False
        self._preprocess_spacing_user_override = False
        self._appliedProfileName = None
        self._profileHasGeneratedBoundaryConditions = False
        self._workflowStageState = WorkflowStageState()
        self._stageController = WorkflowStageController(self._workflowStageState)
        self._lightweightEditorController = LightweightEditorController(self)
        self._preprocessController = PreprocessController(self)
        self._boundaryPreviewController = BoundaryPreviewController(self)
        self._loadPreviewController = LoadPreviewController(self)
        self._executionController = ExecutionController(self)
        self._preprocessingAppliedToInputs = False
        self._workflowReplayContractEditor = None
        self._workflowReplayResolvedEditor = None
        self._workflowReplayResolvedEditorDirty = False
        self._suppressWorkflowReplayEditorDirty = 0
        self._suppressWorkflowSelectionUpdate = False
        self._suppressInputNodeChanged = 0
        self._preprocessPreviewImageNode = None
        self._preprocessPreviewMaskNode = None
        self._preprocessPreviewMaterialNode = None
        super().__init__(parent)

    def _workflow_tab_page(self, title):
        page = qt.QWidget()
        page.setMinimumWidth(0)
        page_layout = qt.QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)
        page_layout.addStretch(1)
        scroll = qt.QScrollArea()
        scroll.setMinimumWidth(0)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
        try:
            scroll.setFrameShape(qt.QFrame.NoFrame)
        except Exception:
            pass
        scroll.setWidget(page)
        self.workflowTabs.addTab(scroll, title)
        return page, page_layout

    def _set_workflow_tab_enabled(self, index, enabled):
        tabs = getattr(self, "workflowTabs", None)
        if tabs is None:
            return
        try:
            tabs.setTabEnabled(int(index), bool(enabled))
        except Exception:
            pass

    def setup(self):
        super().setup()

        self.runtimeCollapsible = ctk.ctkCollapsibleButton()
        self.runtimeCollapsible.text = "Advanced Runtime"
        self.runtimeCollapsible.collapsed = True
        self.layout.addWidget(self.runtimeCollapsible)
        runtime_layout = qt.QVBoxLayout(self.runtimeCollapsible)
        self.runtimeStatusLabel = qt.QLabel("Runtime not checked yet.")
        self.runtimeStatusLabel.wordWrap = True
        runtime_layout.addWidget(self.runtimeStatusLabel)
        runtime_buttons = qt.QHBoxLayout()
        self.checkRuntimeButton = qt.QPushButton("Check Runtime")
        runtime_buttons.addWidget(self.checkRuntimeButton)
        runtime_layout.addLayout(runtime_buttons)

        self.workflowTabs = qt.QTabWidget()
        self.workflowTabs.setTabPosition(qt.QTabWidget.North)
        self.workflowTabs.setMinimumWidth(0)
        try:
            self.workflowTabs.setUsesScrollButtons(True)
            self.workflowTabs.setElideMode(qt.Qt.ElideRight)
            tab_bar = self.workflowTabs.tabBar()
            if tab_bar is not None:
                tab_bar.setExpanding(False)
        except Exception:
            pass
        try:
            self.workflowTabs.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        except Exception:
            pass
        self.layout.addWidget(self.workflowTabs, 1)

        self.inputPage, input_page_layout = self._workflow_tab_page("1 Inputs")
        self.inputCollapsible = qt.QWidget()
        input_page_layout.insertWidget(input_page_layout.count() - 1, self.inputCollapsible)
        input_layout = qt.QVBoxLayout(self.inputCollapsible)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        self.inputSetupGroup = qt.QGroupBox("Setup")
        input_form = qt.QFormLayout(self.inputSetupGroup)
        input_layout.addWidget(self.inputSetupGroup)

        self.imageSelector = slicer.qMRMLNodeComboBox()
        self.imageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.imageSelector.selectNodeUponCreation = True
        self.imageSelector.addEnabled = False
        self.imageSelector.removeEnabled = False
        self.imageSelector.setMRMLScene(slicer.mrmlScene)
        input_form.addRow("Image / material volume", self.imageSelector)

        self.maskSelector = slicer.qMRMLNodeComboBox()
        self.maskSelector.nodeTypes = [
            "vtkMRMLSegmentationNode",
            "vtkMRMLLabelMapVolumeNode",
            "vtkMRMLScalarVolumeNode",
        ]
        self.maskSelector.selectNodeUponCreation = False
        self.maskSelector.noneEnabled = True
        self.maskSelector.addEnabled = False
        self.maskSelector.removeEnabled = False
        self.maskSelector.setMRMLScene(slicer.mrmlScene)
        input_form.addRow("Optional mask / labels", self.maskSelector)
        self.maskRequirementLabel = qt.QLabel("")
        self.maskRequirementLabel.wordWrap = True
        self.maskRequirementLabel.maximumWidth = 520
        input_form.addRow("", self.maskRequirementLabel)

        self.maskSegmentBox = qt.QComboBox()
        self.maskSegmentBox.enabled = False
        self.maskSegmentBox.visible = False
        self.maskSegmentBox.toolTip = "Choose all labels or a checked subset from the selected mask/segmentation."
        input_form.addRow("Mask segment", self.maskSegmentBox)
        self.maskSegmentLabel = input_form.labelForField(self.maskSegmentBox)
        if self.maskSegmentLabel is not None:
            self.maskSegmentLabel.visible = False
        self.maskSegmentChecklist = qt.QListWidget()
        self.maskSegmentChecklist.visible = False
        self.maskSegmentChecklist.minimumHeight = 78
        self.maskSegmentChecklist.maximumHeight = 150
        input_form.addRow("Mask subset", self.maskSegmentChecklist)
        self.maskSubsetLabel = input_form.labelForField(self.maskSegmentChecklist)
        if self.maskSubsetLabel is not None:
            self.maskSubsetLabel.visible = False

        self.resetButton = qt.QPushButton("Reset")
        self.resetButton.toolTip = "Remove generated ParOSol scene nodes and clear previews. Output files on disk are not deleted."

        self.profileBox = qt.QComboBox()
        self.profileBox.setEditable(True)
        profiles = _default_profiles()
        self.profileBox.addItems(profiles)
        if PREFERRED_WORKFLOW in profiles:
            self.profileBox.setCurrentText(PREFERRED_WORKFLOW)
        self.applyProfileButton = qt.QPushButton("Apply Workflow")
        self.showWorkflowButton = qt.QPushButton("Show")
        self.loadProfileButton = qt.QPushButton("Load...")
        self.applyProfileButton.toolTip = "Expand the selected built-in workflow or workflow file into editable Slicer tables."
        self.showWorkflowButton.toolTip = "Show the resolved workflow file and key registration settings for the selected workflow."
        self.loadProfileButton.toolTip = "Select a SlicerParOSol .parosol-workflow or YAML workflow file to apply."
        self.workflowRowWidget = qt.QWidget()
        workflow_row = qt.QHBoxLayout(self.workflowRowWidget)
        workflow_row.setContentsMargins(0, 0, 0, 0)
        workflow_row.setSpacing(6)
        self.workflowLabel = qt.QLabel("Workflow")
        self.workflowLabel.maximumWidth = 72
        workflow_row.addWidget(self.workflowLabel)
        workflow_row.addWidget(self.profileBox, 1)
        input_form.addRow(self.workflowRowWidget)
        self.profileStatusLabel = qt.QLabel("")
        self.profileStatusLabel.wordWrap = True
        self.profileStatusLabel.maximumWidth = 460
        input_form.addRow("", self.profileStatusLabel)
        self.workflowInstructionLabel = qt.QLabel("")
        self.workflowInstructionLabel.wordWrap = True
        self.workflowInstructionLabel.maximumWidth = 520
        input_form.addRow("", self.workflowInstructionLabel)

        self.outputDirectory = ctk.ctkDirectoryButton()
        self.outputDirectory.directory = str(Path.home() / "SlicerParOSolRuns")
        input_form.addRow("Output directory", self.outputDirectory)

        self.derivativeDatasetRootSelector = ctk.ctkPathLineEdit()
        self.derivativeDatasetRootSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.derivativeSubjectEdit = qt.QLineEdit()
        self.derivativeSiteEdit = qt.QLineEdit()
        self.derivativeSessionEdit = qt.QLineEdit()
        input_form.addRow("Derivative dataset", self.derivativeDatasetRootSelector)
        input_form.addRow("Derivative subject", self.derivativeSubjectEdit)
        input_form.addRow("Derivative site", self.derivativeSiteEdit)
        input_form.addRow("Derivative session", self.derivativeSessionEdit)

        self.inputActionGroup = qt.QGroupBox("Run")
        run_layout = qt.QVBoxLayout(self.inputActionGroup)
        input_layout.addWidget(self.inputActionGroup)

        self.inputReadinessLabel = qt.QLabel("")
        self.inputReadinessLabel.wordWrap = True
        self.inputReadinessLabel.maximumWidth = 520
        run_layout.addWidget(self.inputReadinessLabel)

        self.runStatusLabel = qt.QLabel("Idle.")
        self.runStatusLabel.wordWrap = True
        self.runStatusLabel.maximumWidth = 520
        run_layout.addWidget(self.runStatusLabel)

        self.runProgressBar = qt.QProgressBar()
        self.runProgressBar.visible = False
        self.runProgressBar.setTextVisible(False)
        run_layout.addWidget(self.runProgressBar)
        self.runElapsedTimer = qt.QTimer()

        self.quickRunButton = qt.QPushButton("Run Selected Workflow")
        self.quickStopButton = qt.QPushButton("Stop")
        self.quickStopButton.enabled = False
        self.fastRecipeRunCheckBox = qt.QCheckBox("Fast recipe run")
        self.fastRecipeRunCheckBox.checked = False
        self.fastRecipeRunCheckBox.toolTip = (
            "Skip visual stage preparation and replay the selected recipe directly. "
            "Use for trusted workflows when you do not need Slicer to recreate "
            "contact regions and load arrows before solving."
        )
        self.additionalOptionsButton = qt.QPushButton("Additional Options")
        self.quickRunButton.minimumHeight = 46
        self.quickRunButton.setStyleSheet(
            "QPushButton { font-weight: 700; background-color: #2f7ed8; color: white; "
            "border: 1px solid #1f5fa8; border-radius: 4px; padding: 8px 12px; }"
            "QPushButton:disabled { background-color: #9aa9b8; color: #f0f0f0; border-color: #8794a0; }"
        )
        run_layout.addWidget(self.quickRunButton)
        run_layout.addWidget(self.quickStopButton)

        run_options_row = qt.QHBoxLayout()
        run_options_row.addWidget(self.fastRecipeRunCheckBox)
        run_options_row.addStretch(1)
        run_layout.addLayout(run_options_row)

        workflow_tools_row = qt.QHBoxLayout()
        workflow_tools_row.addWidget(self.additionalOptionsButton)
        workflow_tools_row.addWidget(self.applyProfileButton)
        workflow_tools_row.addWidget(self.loadProfileButton)
        workflow_tools_row.addWidget(self.showWorkflowButton)
        workflow_tools_row.addWidget(self.resetButton)
        workflow_tools_row.addStretch(1)
        run_layout.addLayout(workflow_tools_row)

        self.materialPage, material_page_layout = self._workflow_tab_page("2 Materials")
        self.materialCollapsible = qt.QWidget()
        material_page_layout.insertWidget(
            material_page_layout.count() - 1,
            self.materialCollapsible,
        )
        material_section_layout = qt.QVBoxLayout(self.materialCollapsible)

        material_group = qt.QGroupBox("Material Properties")
        material_form = qt.QFormLayout(material_group)
        material_section_layout.addWidget(material_group)

        self.materialModeBox = qt.QComboBox()
        self.materialModeBox.addItems(["Linear label-based", "Linear density formula", "Nonlinear density formula"])
        material_form.addRow("Mode", self.materialModeBox)

        self.materialPresetBox = qt.QComboBox()
        self.materialPresetBox.addItems(
            [
                "Manual",
                "XtremeCTI labels",
                "XtremeCTII labels",
                "Mulder 2007 framework density",
                "Kopperdahl density",
                "Michalski density power law",
                "Morgan trabecular density",
                "Crawford voxel density",
                "Bayraktar trabecular constant",
            ]
        )
        material_form.addRow("Preset", self.materialPresetBox)
        self.materialPresetLabel = material_form.labelForField(self.materialPresetBox)

        self.nonlinearPresetBox = qt.QComboBox()
        self.nonlinearPresetBox.addItems(["Spine nonlinear", "Hip nonlinear", "Manual"])
        material_form.addRow("Nonlinear preset", self.nonlinearPresetBox)
        self.nonlinearPresetLabel = material_form.labelForField(self.nonlinearPresetBox)

        self.nonlinearElasticCoeffSpin = self._material_law_spinbox(3814.4)
        self.nonlinearElasticExponentSpin = self._material_law_spinbox(1.05, maximum=10.0)
        self.nonlinearElasticReferenceSpin = self._material_law_spinbox(1000.0, minimum=1e-6)
        self.nonlinearCompressionCoeffSpin = self._material_law_spinbox(57.4464)
        self.nonlinearCompressionExponentSpin = self._material_law_spinbox(1.39, maximum=10.0)
        self.nonlinearCompressionReferenceSpin = self._material_law_spinbox(1000.0, minimum=1e-6)
        self.nonlinearTensionCoeffSpin = self._material_law_spinbox(57.4464)
        self.nonlinearTensionExponentSpin = self._material_law_spinbox(1.39, maximum=10.0)
        self.nonlinearTensionReferenceSpin = self._material_law_spinbox(1000.0, minimum=1e-6)
        self.nonlinearManualRows = {}
        for key, label, widget in (
            ("elastic_coeff", "Elastic coeff", self.nonlinearElasticCoeffSpin),
            ("elastic_exponent", "Elastic exponent", self.nonlinearElasticExponentSpin),
            ("elastic_reference", "Elastic ref density", self.nonlinearElasticReferenceSpin),
            ("compression_coeff", "Compression coeff", self.nonlinearCompressionCoeffSpin),
            ("compression_exponent", "Compression exponent", self.nonlinearCompressionExponentSpin),
            ("compression_reference", "Compression ref density", self.nonlinearCompressionReferenceSpin),
            ("tension_coeff", "Tension coeff", self.nonlinearTensionCoeffSpin),
            ("tension_exponent", "Tension exponent", self.nonlinearTensionExponentSpin),
            ("tension_reference", "Tension ref density", self.nonlinearTensionReferenceSpin),
        ):
            material_form.addRow(label, widget)
            self.nonlinearManualRows[key] = (material_form.labelForField(widget), widget)

        self.materialTable = qt.QTableWidget()
        self.materialTable.setColumnCount(4)
        self.materialTable.setHorizontalHeaderLabels(["Label", "Name", "E MPa", "nu"])
        self.materialTable.minimumHeight = 100
        _configure_resizable_table(self.materialTable)
        material_form.addRow("Label materials", self.materialTable)
        self.materialTableLabel = material_form.labelForField(self.materialTable)

        material_buttons = qt.QHBoxLayout()
        self.seedMaterialsButton = qt.QPushButton("Generate Automatically")
        self.addMaterialButton = qt.QPushButton("Add Material")
        self.deleteMaterialButton = qt.QPushButton("Delete Material")
        self.seedMaterialsButton.minimumHeight = 32
        material_buttons.addWidget(self.seedMaterialsButton)
        material_buttons.addWidget(self.addMaterialButton)
        material_buttons.addWidget(self.deleteMaterialButton)
        material_form.addRow(material_buttons)
        self.materialButtons = [
            self.seedMaterialsButton,
            self.addMaterialButton,
            self.deleteMaterialButton,
        ]

        self.densityEquationBox = qt.QComboBox()
        self.densityEquationBox.addItems(["kopperdahl", "mulder2007", "power", "linear", "polynomial"])
        material_form.addRow("E formula", self.densityEquationBox)

        self.densityFormulaLabel = qt.QLabel("")
        material_form.addRow("Equation", self.densityFormulaLabel)

        self.densitySlopeSpin = qt.QDoubleSpinBox()
        self.densitySlopeSpin.minimum = -1e7
        self.densitySlopeSpin.maximum = 1e7
        self.densitySlopeSpin.value = 10.0
        material_form.addRow("Linear slope", self.densitySlopeSpin)

        self.densityInterceptSpin = qt.QDoubleSpinBox()
        self.densityInterceptSpin.minimum = -1e7
        self.densityInterceptSpin.maximum = 1e7
        self.densityInterceptSpin.value = 0.0
        material_form.addRow("Linear intercept", self.densityInterceptSpin)

        self.densityCoeffSpin = qt.QDoubleSpinBox()
        self.densityCoeffSpin.minimum = 0.0
        self.densityCoeffSpin.maximum = 1e7
        self.densityCoeffSpin.value = 2980.0
        material_form.addRow("Power coefficient", self.densityCoeffSpin)

        self.densityExponentSpin = qt.QDoubleSpinBox()
        self.densityExponentSpin.minimum = 0.0
        self.densityExponentSpin.maximum = 10.0
        self.densityExponentSpin.value = 1.05
        material_form.addRow("Power exponent", self.densityExponentSpin)

        self.densityReferenceSpin = qt.QDoubleSpinBox()
        self.densityReferenceSpin.minimum = 1e-6
        self.densityReferenceSpin.maximum = 1e7
        self.densityReferenceSpin.value = 1000.0
        material_form.addRow("Reference density", self.densityReferenceSpin)

        self.densityQuadSpin = qt.QDoubleSpinBox()
        self.densityQuadSpin.minimum = -1e7
        self.densityQuadSpin.maximum = 1e7
        self.densityQuadSpin.value = 0.0
        material_form.addRow("Polynomial a2", self.densityQuadSpin)

        self.densityFloorSpin = qt.QDoubleSpinBox()
        self.densityFloorSpin.minimum = 0.0
        self.densityFloorSpin.maximum = 1e7
        self.densityFloorSpin.decimals = 6
        self.densityFloorSpin.value = 0.0
        self.densityFloorSpin.suffix = " MPa"
        material_form.addRow("Minimum E", self.densityFloorSpin)

        self.binMaterialCheckBox = qt.QCheckBox("bin_material")
        self.binMaterialCheckBox.checked = False
        material_form.addRow("Binned material", self.binMaterialCheckBox)

        self.numberBinsSpin = qt.QSpinBox()
        self.numberBinsSpin.minimum = 1
        self.numberBinsSpin.maximum = 4096
        self.numberBinsSpin.value = 128
        material_form.addRow("number bins", self.numberBinsSpin)

        self.densityTestSpin = qt.QDoubleSpinBox()
        self.densityTestSpin.minimum = -1e7
        self.densityTestSpin.maximum = 1e7
        self.densityTestSpin.value = 1000.0
        material_form.addRow("Test density", self.densityTestSpin)

        self.densityResultLabel = qt.QLabel("")
        material_form.addRow("Predicted E", self.densityResultLabel)

        self._density_formula_rows = {
            "equation": (material_form.labelForField(self.densityEquationBox), self.densityEquationBox),
            "formula": (material_form.labelForField(self.densityFormulaLabel), self.densityFormulaLabel),
            "slope": (material_form.labelForField(self.densitySlopeSpin), self.densitySlopeSpin),
            "intercept": (material_form.labelForField(self.densityInterceptSpin), self.densityInterceptSpin),
            "coefficient": (material_form.labelForField(self.densityCoeffSpin), self.densityCoeffSpin),
            "exponent": (material_form.labelForField(self.densityExponentSpin), self.densityExponentSpin),
            "reference": (material_form.labelForField(self.densityReferenceSpin), self.densityReferenceSpin),
            "quad": (material_form.labelForField(self.densityQuadSpin), self.densityQuadSpin),
            "floor": (material_form.labelForField(self.densityFloorSpin), self.densityFloorSpin),
            "bin_material": (material_form.labelForField(self.binMaterialCheckBox), self.binMaterialCheckBox),
            "number_bins": (material_form.labelForField(self.numberBinsSpin), self.numberBinsSpin),
            "test": (material_form.labelForField(self.densityTestSpin), self.densityTestSpin),
            "result": (material_form.labelForField(self.densityResultLabel), self.densityResultLabel),
        }

        self.materialNuSpin = qt.QDoubleSpinBox()
        self.materialNuSpin.minimum = 0.0
        self.materialNuSpin.maximum = 0.49
        self.materialNuSpin.value = 0.3
        material_form.addRow("Density nu", self.materialNuSpin)
        self.materialNuLabel = material_form.labelForField(self.materialNuSpin)

        material_actions = qt.QHBoxLayout()
        self.applyMaterialsButton = qt.QPushButton("Apply Materials")
        self.applyMaterialsNextButton = qt.QPushButton("Apply & Next")
        self.applyMaterialsButton.minimumHeight = 34
        self.applyMaterialsNextButton.minimumHeight = 38
        self.applyMaterialsNextButton.setStyleSheet(
            "QPushButton { font-weight: 700; background-color: #2f7ed8; color: white; "
            "border: 1px solid #1f5fa8; border-radius: 4px; padding: 8px 12px; }"
            "QPushButton:disabled { background-color: #9aa9b8; color: #f0f0f0; border-color: #8794a0; }"
        )
        material_actions.addStretch(1)
        material_actions.addWidget(self.applyMaterialsButton)
        material_actions.addWidget(self.applyMaterialsNextButton)
        material_section_layout.addLayout(material_actions)

        self.preprocessingPage, preprocessing_page_layout = self._workflow_tab_page("3 Image Prep")
        self.preprocessingCollapsible = qt.QWidget()
        preprocessing_page_layout.insertWidget(
            preprocessing_page_layout.count() - 1,
            self.preprocessingCollapsible,
        )
        preprocessing_form = qt.QFormLayout(self.preprocessingCollapsible)

        self.preprocessPreparationGroup = qt.QGroupBox("Image Preparation")
        preparation_form = qt.QFormLayout(self.preprocessPreparationGroup)
        preprocessing_form.addRow(self.preprocessPreparationGroup)

        crop_row = qt.QHBoxLayout()
        self.cropToMaskCheckBox = qt.QCheckBox("Crop to mask")
        self.cropToMaskCheckBox.checked = False
        crop_row.addWidget(self.cropToMaskCheckBox)

        self.cropPaddingSpin = qt.QDoubleSpinBox()
        self.cropPaddingSpin.minimum = 0.0
        self.cropPaddingSpin.maximum = 100.0
        self.cropPaddingSpin.decimals = 1
        self.cropPaddingSpin.value = 5.0
        self.cropPaddingSpin.suffix = " mm"
        crop_row.addWidget(self.cropPaddingSpin)
        preparation_form.addRow("", crop_row)

        self.largestComponentCheckBox = qt.QCheckBox("Keep largest connected component")
        self.largestComponentCheckBox.checked = False
        preparation_form.addRow("", self.largestComponentCheckBox)

        self.smoothDensityCheckBox = qt.QCheckBox("Smooth density / grayscale")
        self.smoothDensityCheckBox.checked = False
        preparation_form.addRow("", self.smoothDensityCheckBox)

        self.smoothLabelsCheckBox = qt.QCheckBox("Smooth mask / labels")
        self.smoothLabelsCheckBox.checked = False
        preparation_form.addRow("", self.smoothLabelsCheckBox)

        self.smoothSigmaSpin = qt.QDoubleSpinBox()
        self.smoothSigmaSpin.minimum = 0.0
        self.smoothSigmaSpin.maximum = 10.0
        self.smoothSigmaSpin.decimals = 2
        self.smoothSigmaSpin.value = 1.0
        self.smoothSigmaSpin.suffix = " mm"
        preparation_form.addRow("Smoothing sigma", self.smoothSigmaSpin)

        self.resampleIsotropicCheckBox = qt.QCheckBox("Resample to isotropic spacing before solve")
        self.resampleIsotropicCheckBox.checked = True
        preparation_form.addRow("", self.resampleIsotropicCheckBox)

        self.isotropicSpacingSpin = qt.QDoubleSpinBox()
        self.isotropicSpacingSpin.minimum = 0.001
        self.isotropicSpacingSpin.maximum = 20.0
        self.isotropicSpacingSpin.decimals = 4
        self.isotropicSpacingSpin.value = 1.0
        self.isotropicSpacingSpin.suffix = " mm"
        preparation_form.addRow("Target spacing", self.isotropicSpacingSpin)

        self.resampleDensityInterpolationBox = qt.QComboBox()
        self.resampleDensityInterpolationBox.addItem("Linear", "linear")
        self.resampleDensityInterpolationBox.addItem("B-spline / cubic", "bspline")
        preparation_form.addRow("Density interpolation", self.resampleDensityInterpolationBox)

        custom_preprocessing_row = qt.QHBoxLayout()
        self.customPreprocessingBox = qt.QComboBox()
        self.customPreprocessingBox.addItem("None", "")
        self.customPreprocessingBox.addItem("Create new...", CUSTOM_PREPROCESSING_CREATE_TOKEN)
        custom_preprocessing_row.addWidget(self.customPreprocessingBox, 1)
        self.editCustomPreprocessingButton = qt.QPushButton("Edit...")
        custom_preprocessing_row.addWidget(self.editCustomPreprocessingButton)
        self.preprocessRegistrationGroup = qt.QGroupBox("Registration")
        registration_form = qt.QFormLayout(self.preprocessRegistrationGroup)
        preprocessing_form.addRow(self.preprocessRegistrationGroup)

        self.icpRegistrationCheckBox = qt.QCheckBox("Use ICP registration")
        self.icpRegistrationCheckBox.checked = False
        self.icpRegistrationCheckBox.toolTip = (
            "When enabled, workflow saving writes a reference point cloud from "
            "the current sample. Workflow runs can register new samples to it."
        )
        self.showIcpReferenceButton = qt.QPushButton("Show ICP Ref")
        self.showIcpReferenceButton.toolTip = (
            "Load the active ICP reference point cloud into Slicer as a debug overlay. "
            "Uses the current self-reference preview when available."
        )
        icp_row = qt.QHBoxLayout()
        icp_row.addWidget(self.icpRegistrationCheckBox)
        icp_row.addWidget(self.showIcpReferenceButton)
        icp_row.addStretch(1)
        registration_form.addRow("", icp_row)
        registration_form.addRow("Post-registration custom preprocessing", custom_preprocessing_row)
        self.icpTargetImageBox = qt.QComboBox()
        self.icpTargetImageBox.addItem("Workflow reference", "workflow-reference")
        self.icpTargetImageBox.addItem("Self", "self")
        self.icpTargetImageBox.addItem("Slicer node", "slicer-node")
        self.icpTargetImageBox.visible = False
        self.icpTargetImageBox.toolTip = (
            "Select where ICP reference points come from: a saved workflow reference, "
            "the current preprocessed sample, or another Slicer node."
        )
        registration_form.addRow("ICP target image", self.icpTargetImageBox)
        self.icpTargetImageLabel = registration_form.labelForField(self.icpTargetImageBox)
        if self.icpTargetImageLabel is not None:
            self.icpTargetImageLabel.visible = False

        self.icpTargetNodeSelector = slicer.qMRMLNodeComboBox()
        self.icpTargetNodeSelector.nodeTypes = [
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLLabelMapVolumeNode",
            "vtkMRMLSegmentationNode",
        ]
        self.icpTargetNodeSelector.selectNodeUponCreation = False
        self.icpTargetNodeSelector.noneEnabled = True
        self.icpTargetNodeSelector.addEnabled = False
        self.icpTargetNodeSelector.removeEnabled = False
        self.icpTargetNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.icpTargetNodeSelector.visible = False
        self.icpTargetNodeSelector.toolTip = (
            "Optional Slicer volume, labelmap, or segmentation used as the ICP reference source."
        )
        registration_form.addRow("ICP target node", self.icpTargetNodeSelector)
        self.icpTargetNodeLabel = registration_form.labelForField(self.icpTargetNodeSelector)
        if self.icpTargetNodeLabel is not None:
            self.icpTargetNodeLabel.visible = False

        self.icpTargetLabelBox = qt.QComboBox()
        self.icpTargetLabelBox.visible = False
        self.icpTargetLabelBox.toolTip = "Select the label used to sample the ICP point cloud. Use All for binary masks."
        registration_form.addRow("ICP target label", self.icpTargetLabelBox)
        self.icpTargetLabel = registration_form.labelForField(self.icpTargetLabelBox)
        if self.icpTargetLabel is not None:
            self.icpTargetLabel.visible = False

        self.preprocessPreviewGroup = qt.QGroupBox("Preview")
        preview_form = qt.QFormLayout(self.preprocessPreviewGroup)
        preprocessing_form.addRow(self.preprocessPreviewGroup)

        self.preprocessPreviewModeBox = qt.QComboBox()
        self.preprocessPreviewModeBox.addItems(["Image + mask", "Mask", "Material"])
        preview_form.addRow("Display", self.preprocessPreviewModeBox)

        preprocess_actions = qt.QHBoxLayout()
        self.preprocessButton = qt.QPushButton("Prepare Image")
        self.preprocessNextButton = qt.QPushButton("Prepare & Next")
        self.preprocessButton.minimumHeight = 34
        self.preprocessNextButton.minimumHeight = 38
        self.preprocessNextButton.setStyleSheet(
            "QPushButton { font-weight: 700; background-color: #208a5d; color: white; "
            "border: 1px solid #146844; border-radius: 4px; padding: 8px 12px; }"
            "QPushButton:disabled { background-color: #9aa9b8; color: #f0f0f0; border-color: #8794a0; }"
        )
        self.preprocessButton.toolTip = (
            "Apply selected preprocessing to the current Slicer inputs for visual "
            "authoring: crop, largest connected component, smoothing, isotropic resampling, "
            "material preview, and ICP reference bookkeeping."
        )
        self.preprocessNextButton.toolTip = self.preprocessButton.toolTip
        preprocess_actions.addStretch(1)
        preprocess_actions.addWidget(self.preprocessButton)
        preprocess_actions.addWidget(self.preprocessNextButton)
        preprocessing_form.addRow("", preprocess_actions)

        self.editorPage, editor_page_layout = self._workflow_tab_page("4 Contact Regions")
        self.editorCollapsible = qt.QWidget()
        editor_page_layout.insertWidget(editor_page_layout.count() - 1, self.editorCollapsible)
        editor_layout = qt.QVBoxLayout(self.editorCollapsible)

        self.editorGuideLabel = qt.QLabel(
            "Move planes in the views, create contact regions, then preview loads before running."
        )
        self.editorGuideLabel.wordWrap = True
        editor_layout.addWidget(self.editorGuideLabel)

        disk_group = qt.QGroupBox("Contact Regions")
        disk_form = qt.QFormLayout(disk_group)
        editor_layout.addWidget(disk_group)

        self.contactModelBox = qt.QComboBox()
        self.contactModelBox.addItems(["Material disks", "Bone surface"])
        self.contactModelBox.visible = False

        self.planeNudgeStep = qt.QDoubleSpinBox()
        self.planeNudgeStep.minimum = 0.1
        self.planeNudgeStep.maximum = 50.0
        self.planeNudgeStep.value = 1.0
        self.planeNudgeStep.suffix = " mm"
        self.planeNudgeStep.visible = False

        self.planeRotateStep = qt.QDoubleSpinBox()
        self.planeRotateStep.minimum = 0.1
        self.planeRotateStep.maximum = 45.0
        self.planeRotateStep.value = 5.0
        self.planeRotateStep.suffix = " deg"
        self.planeRotateStep.visible = False

        self.contactGuideLabel = qt.QLabel(
            "Contact regions are generated from the current planes. Edit labels, axes, cap thickness, or surface mode in the table below."
        )
        self.contactGuideLabel.wordWrap = True
        disk_form.addRow(self.contactGuideLabel)

        self.contactTableGroup = qt.QGroupBox("Contact Region Table")
        contact_table_form = qt.QFormLayout(self.contactTableGroup)
        disk_form.addRow(self.contactTableGroup)

        self.planeTable = qt.QTableWidget()
        self.planeTable.setColumnCount(19)
        self.planeTable.setHorizontalHeaderLabels(
            [
                "Name",
                "Axis",
                "Normal",
                "Contact",
                "Surface mode",
                "BC mode",
                "Direction frame",
                "Custom R",
                "Custom A",
                "Custom S",
                "Value",
                "Shape",
                "Thickness",
                "Intrusion",
                "Radius",
                "Plane size",
                "Disk target",
                "Disk E MPa",
                "Disk nu",
            ]
        )
        self.planeTable.selectionBehavior = qt.QAbstractItemView.SelectRows
        self.planeTable.selectionMode = qt.QAbstractItemView.SingleSelection
        self.planeTable.minimumHeight = 120
        _configure_resizable_table(self.planeTable)
        try:
            self.planeTable.horizontalHeader().setStretchLastSection(True)
        except Exception:
            pass
        _configure_plane_table_visibility(self.planeTable)
        contact_table_form.addRow(self.planeTable)

        inspector_group = qt.QGroupBox("Selected Plane Normal")
        inspector_layout = qt.QGridLayout(inspector_group)
        self.planeStatusLabel = qt.QLabel("Select a plane row.")
        self.planeStatusLabel.wordWrap = True
        self.planeStatusLabel.maximumWidth = 260
        inspector_layout.addWidget(self.planeStatusLabel, 0, 0, 1, 6)
        self.planeNormalREdit = qt.QLineEdit("0")
        self.planeNormalAEdit = qt.QLineEdit("0")
        self.planeNormalSEdit = qt.QLineEdit("1")
        for edit in (self.planeNormalREdit, self.planeNormalAEdit, self.planeNormalSEdit):
            edit.maximumWidth = 58
        self.applyPlaneNormalButton = qt.QPushButton("Set")
        self.planeNormalRButton = qt.QPushButton("R/L")
        self.planeNormalAButton = qt.QPushButton("A/P")
        self.planeNormalSButton = qt.QPushButton("S/I")
        self.applyPlaneNormalButton.toolTip = "Apply the typed RAS normal to the selected plane."
        self.planeNormalRButton.toolTip = "Set selected plane normal to the R/L axis."
        self.planeNormalAButton.toolTip = "Set selected plane normal to the A/P axis."
        self.planeNormalSButton.toolTip = "Set selected plane normal to the S/I axis."
        inspector_layout.addWidget(qt.QLabel("Normal RAS"), 1, 0)
        inspector_layout.addWidget(self.planeNormalREdit, 1, 1)
        inspector_layout.addWidget(self.planeNormalAEdit, 1, 2)
        inspector_layout.addWidget(self.planeNormalSEdit, 1, 3)
        inspector_layout.addWidget(self.applyPlaneNormalButton, 1, 4)
        inspector_layout.addWidget(self.planeNormalRButton, 2, 1)
        inspector_layout.addWidget(self.planeNormalAButton, 2, 2)
        inspector_layout.addWidget(self.planeNormalSButton, 2, 3)
        self.coordinateHelpLabel = qt.QLabel(
            "RAS axes: R/L, A/P, S/I."
        )
        self.coordinateHelpLabel.wordWrap = True
        self.coordinateHelpLabel.maximumWidth = 260
        self.coordinateHelpLabel.toolTip = (
            "Coordinates use Slicer RAS. Red/Green/Yellow are slice views, not load axes. "
            "Fixed has no value; displacement uses mm or %, force uses total N, bending and torsion use degrees."
        )
        inspector_layout.addWidget(self.coordinateHelpLabel, 3, 0, 1, 6)
        contact_table_form.addRow(inspector_group)
        inspector_group.visible = True

        self.loadTable = qt.QTableWidget()
        self.loadTable.setColumnCount(LOAD_TABLE_COLUMN_COUNT)
        self.loadTable.setHorizontalHeaderLabels(
            [
                "Region",
                "BC mode",
                "Direction frame",
                "R",
                "A",
                "S",
                "Value",
                "Units",
                "Fixed axes (RAS)",
                "Label",
            ]
        )
        self.loadTable.selectionBehavior = qt.QAbstractItemView.SelectRows
        self.loadTable.selectionMode = qt.QAbstractItemView.SingleSelection
        self.loadTable.minimumHeight = 95
        _configure_resizable_table(self.loadTable)
        try:
            self.loadTable.horizontalHeader().setStretchLastSection(True)
        except Exception:
            pass
        table_buttons = qt.QHBoxLayout()
        self.autoTopBottomButton = qt.QPushButton("Auto Top/Bottom")
        self.addPlaneButton = qt.QPushButton("Add Plane")
        self.deletePlaneButton = qt.QPushButton("Delete Plane")
        table_buttons.addWidget(self.autoTopBottomButton)
        table_buttons.addWidget(self.addPlaneButton)
        table_buttons.addWidget(self.deletePlaneButton)
        contact_table_form.addRow(table_buttons)

        plane_buttons = qt.QHBoxLayout()
        self.createTopDiskButton = qt.QPushButton("Create Top Plane")
        self.createBottomDiskButton = qt.QPushButton("Create Bottom Plane")
        self.previewDisksButton = qt.QPushButton("Create Regions")
        self.previewDisksNextButton = qt.QPushButton("Create & Next")
        self.deleteDisksButton = qt.QPushButton("Remove Contact Regions")
        self.previewDisksButton.minimumHeight = 34
        self.previewDisksNextButton.minimumHeight = 38
        self.deleteDisksButton.minimumHeight = 32
        self.previewDisksNextButton.setStyleSheet(
            "QPushButton { font-weight: 700; background-color: #2f7ed8; color: white; "
            "border: 1px solid #1f5fa8; border-radius: 4px; padding: 8px 12px; }"
            "QPushButton:disabled { background-color: #9aa9b8; color: #f0f0f0; border-color: #8794a0; }"
        )
        self.deleteDisksButton.setStyleSheet(
            "QPushButton { font-weight: 600; background-color: #f3f6f9; color: #34495e; "
            "border: 1px solid #b9c5d1; border-radius: 4px; padding: 6px 12px; }"
        )
        plane_buttons.addWidget(self.createTopDiskButton)
        plane_buttons.addWidget(self.createBottomDiskButton)
        disk_form.addRow(plane_buttons)
        disk_form.addRow(self.deleteDisksButton)
        contact_actions = qt.QHBoxLayout()
        contact_actions.addStretch(1)
        contact_actions.addWidget(self.previewDisksButton)
        contact_actions.addWidget(self.previewDisksNextButton)
        disk_form.addRow(contact_actions)
        self.createTopDiskButton.visible = False
        self.createBottomDiskButton.visible = False

        self.loadPage, load_page_layout = self._workflow_tab_page("5 Loads")
        self.loadCollapsible = qt.QWidget()
        load_page_layout.insertWidget(load_page_layout.count() - 1, self.loadCollapsible)
        load_section_layout = qt.QVBoxLayout(self.loadCollapsible)

        load_group = qt.QGroupBox("Loads & Fixities")
        load_layout = qt.QVBoxLayout(load_group)
        self.loadGuideLabel = qt.QLabel(
            "Preview force, displacement, bending, torsion, or load-history arrows from the current contact regions."
        )
        self.loadGuideLabel.wordWrap = True
        load_layout.addWidget(self.loadGuideLabel)
        self.loadTableGroup = qt.QGroupBox("Load/Fixity Table")
        load_table_layout = qt.QVBoxLayout(self.loadTableGroup)
        load_layout.addWidget(self.loadTableGroup)
        load_table_layout.addWidget(self.loadTable)
        load_arrow_layout = qt.QHBoxLayout()
        self.loadArrowScaleSpin = qt.QDoubleSpinBox()
        self.loadArrowScaleSpin.minimum = 0.1
        self.loadArrowScaleSpin.maximum = 50.0
        self.loadArrowScaleSpin.value = 1.0
        self.loadArrowScaleSpin.singleStep = 0.5
        self.loadArrowScaleSpin.suffix = "x"
        load_arrow_layout.addWidget(qt.QLabel("Load arrow scale"))
        load_arrow_layout.addWidget(self.loadArrowScaleSpin)
        load_arrow_layout.addStretch(1)
        load_table_layout.addLayout(load_arrow_layout)
        self.previewLoadsButton = qt.QPushButton("Preview Loads")
        self.previewLoadsNextButton = qt.QPushButton("Preview & Next")
        self.previewLoadsButton.minimumHeight = 34
        self.previewLoadsNextButton.minimumHeight = 38
        self.previewLoadsNextButton.setStyleSheet(
            "QPushButton { font-weight: 700; background-color: #208a5d; color: white; "
            "border: 1px solid #146844; border-radius: 4px; padding: 8px 12px; }"
            "QPushButton:disabled { background-color: #9aa9b8; color: #f0f0f0; border-color: #8794a0; }"
        )
        load_actions = qt.QHBoxLayout()
        load_actions.addStretch(1)
        load_actions.addWidget(self.previewLoadsButton)
        load_actions.addWidget(self.previewLoadsNextButton)
        load_layout.addLayout(load_actions)
        load_section_layout.addWidget(load_group)

        self.diskLabelSelector = slicer.qMRMLNodeComboBox()
        self.diskLabelSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.diskLabelSelector.noneEnabled = True
        self.diskLabelSelector.addEnabled = True
        self.diskLabelSelector.renameEnabled = True
        self.diskLabelSelector.removeEnabled = True
        self.diskLabelSelector.setMRMLScene(slicer.mrmlScene)
        self.diskLabelSelector.visible = False

        bc_group = qt.QGroupBox("Boundary surfaces")
        bc_form = qt.QFormLayout(bc_group)
        editor_layout.addWidget(bc_group)
        bc_group.visible = False

        self.bcThickness = qt.QDoubleSpinBox()
        self.bcThickness.minimum = 0.1
        self.bcThickness.maximum = 1000.0
        self.bcThickness.value = 1.0
        self.bcThickness.suffix = " mm"
        bc_form.addRow("Selection thickness", self.bcThickness)

        bc_buttons = qt.QHBoxLayout()
        self.createFixedPlaneButton = qt.QPushButton("Create Fixed Plane")
        self.createLoadedPlaneButton = qt.QPushButton("Create Loaded Plane")
        self.previewBCButton = qt.QPushButton("Preview BC Labels")
        self.deleteBCButton = qt.QPushButton("Delete BC")
        bc_buttons.addWidget(self.createFixedPlaneButton)
        bc_buttons.addWidget(self.createLoadedPlaneButton)
        bc_buttons.addWidget(self.previewBCButton)
        bc_buttons.addWidget(self.deleteBCButton)
        editor_layout.addLayout(bc_buttons)
        for button in (
            self.createFixedPlaneButton,
            self.createLoadedPlaneButton,
            self.previewBCButton,
            self.deleteBCButton,
        ):
            button.visible = False

        self.bcLabelSelector = slicer.qMRMLNodeComboBox()
        self.bcLabelSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.bcLabelSelector.noneEnabled = True
        self.bcLabelSelector.addEnabled = True
        self.bcLabelSelector.renameEnabled = True
        self.bcLabelSelector.removeEnabled = True
        self.bcLabelSelector.setMRMLScene(slicer.mrmlScene)
        bc_form.addRow("BC labelmap", self.bcLabelSelector)

        self.solvePage, solve_page_layout = self._workflow_tab_page("6 Review & Run")
        self.solveCollapsible = qt.QWidget()
        solve_page_layout.insertWidget(solve_page_layout.count() - 1, self.solveCollapsible)
        solve_layout = qt.QVBoxLayout(self.solveCollapsible)

        self.runReadinessLabel = qt.QLabel("")
        self.runReadinessLabel.wordWrap = True
        solve_layout.addWidget(self.runReadinessLabel)

        self.solverExpertCollapsible = ctk.ctkCollapsibleButton()
        self.solverExpertCollapsible.text = "Expert Solver Settings"
        self.solverExpertCollapsible.collapsed = True
        solve_layout.addWidget(self.solverExpertCollapsible)
        solver_expert_layout = qt.QVBoxLayout(self.solverExpertCollapsible)

        self.solverSettingsGroup = qt.QGroupBox("Solver Settings")
        solver_settings_form = qt.QFormLayout(self.solverSettingsGroup)
        solver_expert_layout.addWidget(self.solverSettingsGroup)

        default_solver = self.logic.default_solver_config()
        self.mpiProcessesSpin = qt.QSpinBox()
        self.mpiProcessesSpin.minimum = 1
        self.mpiProcessesSpin.maximum = 128
        self.mpiProcessesSpin.value = int(default_solver.get("mpi_processes", 1))
        solver_settings_form.addRow("MPI processes", self.mpiProcessesSpin)

        mpi_launcher_row = qt.QHBoxLayout()
        self.mpiLauncherEdit = qt.QLineEdit(str(default_solver.get("mpi_launcher", "")))
        self.mpiLauncherEdit.placeholderText = "Auto: use packaged MPI from parosol-py"
        self.mpiLauncherEdit.toolTip = (
            "Optional external mpirun/mpiexec override. Leave empty to use the "
            "MPI runtime bundled with the parosol-py wheel installed in Slicer's Python."
        )
        self.browseMpiLauncherButton = qt.QPushButton("Browse...")
        self.browseMpiLauncherButton.toolTip = (
            "Select an external mpirun/mpiexec, for example from Homebrew or conda."
        )
        mpi_launcher_row.addWidget(self.mpiLauncherEdit, 1)
        mpi_launcher_row.addWidget(self.browseMpiLauncherButton)
        solver_settings_form.addRow("MPI launcher", mpi_launcher_row)

        self.solverToleranceEdit = qt.QLineEdit(
            _format_scientific_number(default_solver.get("tolerance", 1e-4))
        )
        self.solverToleranceEdit.maximumWidth = 90
        solver_settings_form.addRow("Tolerance", self.solverToleranceEdit)

        self.exportDisplacementsCheckBox = qt.QCheckBox("Export displacement field")
        self.exportDisplacementsCheckBox.checked = False
        self.exportDisplacementsCheckBox.visible = False
        self.exportDisplacementsCheckBox.toolTip = (
            "Request displacement_x/y/z fields for deformation-arrow visualization. "
            "Leave off for faster routine solves."
        )

        self.postprocessCollapsible = ctk.ctkCollapsibleButton()
        self.postprocessCollapsible.text = "Outputs"
        self.postprocessCollapsible.collapsed = False
        solve_layout.addWidget(self.postprocessCollapsible)
        postprocess_form = qt.QFormLayout(self.postprocessCollapsible)

        self.failurePresetBox = qt.QComboBox()
        self.failurePresetBox.addItems(
            [
                "Pistoia EES 0.7% / 2%",
                "Kopperdahl/Crawford 0.68%",
                "None",
            ]
        )
        postprocess_form.addRow("Failure criterion", self.failurePresetBox)

        output_fields_widget = qt.QWidget()
        output_fields_layout = qt.QVBoxLayout(output_fields_widget)
        output_fields_layout.setContentsMargins(0, 0, 0, 0)
        self.outputFieldChecks = {}
        self.nonlinearOutputFieldNames = {
            "stress",
            "plastic_strain",
            "plastic_strain_magnitude",
            "plastic_dissipation",
            "mechanical_work_density",
        }
        for field, label, checked in (
            ("sed", "SED", True),
            ("effective_strain", "Effective strain", False),
            ("von_mises", "Von Mises", False),
            ("strain", "Strain tensor", False),
            ("stress", "Stress tensor", False),
            ("plastic_strain", "Plastic strain tensor", False),
            ("plastic_strain_magnitude", "Plastic strain magnitude", False),
            ("plastic_dissipation", "Plastic dissipation", False),
            ("mechanical_work_density", "Mechanical work density", False),
            ("displacements", "Displacements", False),
        ):
            checkbox = qt.QCheckBox(label)
            checkbox.checked = bool(checked)
            self.outputFieldChecks[field] = checkbox
            output_fields_layout.addWidget(checkbox)
        postprocess_form.addRow("Export fields", output_fields_widget)
        self.outputFieldChecks["displacements"].toggled.connect(
            lambda checked: setattr(self.exportDisplacementsCheckBox, "checked", bool(checked))
        )

        self.exportButton = qt.QPushButton("Package for Cluster")
        self.saveWorkflowButton = qt.QPushButton("Save Workflow")
        self.runButton = qt.QPushButton("Run ParOSol")
        self.runButton.minimumHeight = 44
        self.runButton.default = True
        self.runButton.setStyleSheet(
            "QPushButton {"
            "  font-weight: 700;"
            "  font-size: 15px;"
            "  padding: 10px 14px;"
            "  background-color: #2f7ed8;"
            "  color: white;"
            "  border: 1px solid #1f5fa8;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #9aa9b8;"
            "  color: #f0f0f0;"
            "  border-color: #8794a0;"
            "}"
        )
        self.stopButton = qt.QPushButton("Stop")
        self.stopButton.enabled = False
        solve_layout.addWidget(self.runButton)
        secondary_buttons = qt.QHBoxLayout()
        secondary_buttons.addWidget(self.exportButton)
        secondary_buttons.addWidget(self.saveWorkflowButton)
        secondary_buttons.addWidget(self.stopButton)
        solve_layout.addLayout(secondary_buttons)

        self.resultsPage, results_page_layout = self._workflow_tab_page("7 Results")
        self.resultsCollapsible = qt.QWidget()
        results_page_layout.insertWidget(results_page_layout.count() - 1, self.resultsCollapsible)
        results_layout = qt.QVBoxLayout(self.resultsCollapsible)
        self.resultText = qt.QTextEdit()
        self.resultText.readOnly = True
        results_layout.addWidget(self.resultText)
        result_buttons = qt.QHBoxLayout()
        self.exportResultsCsvButton = qt.QPushButton("Export Result CSV")
        self.saveResultAsButton = qt.QPushButton("Save Result As...")
        self.deleteResultsButton = qt.QPushButton("Delete Results")
        result_buttons.addWidget(self.exportResultsCsvButton)
        result_buttons.addWidget(self.saveResultAsButton)
        result_buttons.addWidget(self.deleteResultsButton)
        result_buttons.addStretch(1)
        results_layout.addLayout(result_buttons)
        deformed_layout = qt.QHBoxLayout()
        self.deformedScaleSpin = qt.QDoubleSpinBox()
        self.deformedScaleSpin.minimum = 0.1
        self.deformedScaleSpin.maximum = 100.0
        self.deformedScaleSpin.value = 10.0
        self.deformedScaleSpin.singleStep = 1.0
        self.deformedScaleSpin.suffix = "x"
        self.deformedMaxArrowsSpin = qt.QSpinBox()
        self.deformedMaxArrowsSpin.minimum = 100
        self.deformedMaxArrowsSpin.maximum = 100000
        self.deformedMaxArrowsSpin.value = 1200
        self.deformedMaxArrowsSpin.singleStep = 500
        self.showDeformedButton = qt.QPushButton("Show Deformation Arrows")
        self.deleteDeformedButton = qt.QPushButton("Delete Deformation")
        deformed_layout.addWidget(qt.QLabel("Deformation scale"))
        deformed_layout.addWidget(self.deformedScaleSpin)
        deformed_layout.addWidget(qt.QLabel("Max arrows"))
        deformed_layout.addWidget(self.deformedMaxArrowsSpin)
        deformed_layout.addWidget(self.showDeformedButton)
        deformed_layout.addWidget(self.deleteDeformedButton)
        results_layout.addLayout(deformed_layout)

        self.consoleCollapsible = ctk.ctkCollapsibleButton()
        self.consoleCollapsible.text = "Console / Logs"
        self.consoleCollapsible.collapsed = True
        self.layout.addWidget(self.consoleCollapsible)
        console_layout = qt.QVBoxLayout(self.consoleCollapsible)
        self.logText = qt.QPlainTextEdit()
        self.logText.readOnly = True
        self.logText.maximumBlockCount = 2000
        self.logText.minimumHeight = 100
        self.logText.maximumHeight = 180
        console_layout.addWidget(self.logText)

        self._install_tooltips()

        self.createTopDiskButton.clicked.connect(lambda: self._create_disk_plane("top"))
        self.createBottomDiskButton.clicked.connect(lambda: self._create_disk_plane("bottom"))
        self.imageSelector.currentNodeChanged.connect(self._on_input_node_changed)
        self.maskSelector.currentNodeChanged.connect(self._on_mask_node_changed)
        try:
            self.imageSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._on_input_node_changed)
            self.maskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._on_mask_node_changed)
        except Exception:
            pass
        self.maskSegmentBox.currentIndexChanged.connect(self._on_mask_segment_changed)
        self.maskSegmentChecklist.itemChanged.connect(self._on_mask_segment_subset_changed)
        self.autoTopBottomButton.clicked.connect(self.auto_top_bottom_contact_planes)
        self.addPlaneButton.clicked.connect(self._add_contact_plane)
        self.deletePlaneButton.clicked.connect(self._delete_selected_contact_plane)
        self.addMaterialButton.clicked.connect(self._add_material_row)
        self.deleteMaterialButton.clicked.connect(self._delete_selected_material_row)
        self.seedMaterialsButton.clicked.connect(self._seed_material_table)
        self.applyMaterialsButton.clicked.connect(self.apply_materials)
        self.applyMaterialsNextButton.clicked.connect(self.apply_materials_and_next)
        self.previewDisksButton.clicked.connect(self.preview_disks)
        self.previewDisksNextButton.clicked.connect(self.preview_disks_and_next)
        self.previewLoadsButton.clicked.connect(self.preview_loads)
        self.previewLoadsNextButton.clicked.connect(self.preview_loads_and_next)
        self.loadArrowScaleSpin.valueChanged.connect(self._on_load_arrow_scale_changed)
        self.deleteDisksButton.clicked.connect(self.delete_disks)
        self.applyPlaneNormalButton.clicked.connect(self._apply_selected_plane_normal_from_fields)
        self.planeNormalREdit.editingFinished.connect(self._apply_selected_plane_normal_from_fields)
        self.planeNormalAEdit.editingFinished.connect(self._apply_selected_plane_normal_from_fields)
        self.planeNormalSEdit.editingFinished.connect(self._apply_selected_plane_normal_from_fields)
        self.planeNormalRButton.clicked.connect(lambda: self._set_selected_plane_normal_vector((1.0, 0.0, 0.0)))
        self.planeNormalAButton.clicked.connect(lambda: self._set_selected_plane_normal_vector((0.0, 1.0, 0.0)))
        self.planeNormalSButton.clicked.connect(lambda: self._set_selected_plane_normal_vector((0.0, 0.0, 1.0)))
        self.createFixedPlaneButton.clicked.connect(lambda: self._create_bc_plane("fixed"))
        self.createLoadedPlaneButton.clicked.connect(lambda: self._create_bc_plane("loaded"))
        self.previewBCButton.clicked.connect(self.preview_bc)
        self.deleteBCButton.clicked.connect(self.delete_bc)
        self.profileBox.currentTextChanged.connect(self._on_workflow_selection_changed)
        self.applyProfileButton.clicked.connect(self.apply_profile)
        self.showWorkflowButton.clicked.connect(self.show_active_workflow_file)
        self.loadProfileButton.clicked.connect(self.load_profile_file)
        self.quickRunButton.clicked.connect(self.run_case)
        self.quickStopButton.clicked.connect(self.stop_case)
        self.additionalOptionsButton.clicked.connect(self._open_additional_options)
        self.contactModelBox.currentTextChanged.connect(self._update_contact_model_mode)
        self.materialModeBox.currentTextChanged.connect(self._update_material_mode)
        self.materialModeBox.currentTextChanged.connect(lambda _text: self._update_output_field_visibility())
        self.materialPresetBox.currentTextChanged.connect(self._apply_material_preset)
        self.nonlinearPresetBox.currentTextChanged.connect(self._update_material_mode)
        self.nonlinearPresetBox.currentTextChanged.connect(self._on_nonlinear_preset_changed)
        self.densityEquationBox.currentTextChanged.connect(self._on_density_equation_changed)
        self.densitySlopeSpin.valueChanged.connect(lambda _value: self._update_material_mode())
        self.densityInterceptSpin.valueChanged.connect(lambda _value: self._update_material_mode())
        self.densityCoeffSpin.valueChanged.connect(lambda _value: self._update_material_mode())
        self.densityExponentSpin.valueChanged.connect(lambda _value: self._update_material_mode())
        self.densityReferenceSpin.valueChanged.connect(lambda _value: self._update_material_mode())
        self.densityQuadSpin.valueChanged.connect(lambda _value: self._update_material_mode())
        self.densityFloorSpin.valueChanged.connect(lambda _value: self._update_material_mode())
        self.binMaterialCheckBox.toggled.connect(lambda _value: self._update_material_mode())
        self.numberBinsSpin.valueChanged.connect(lambda _value: self._update_material_mode())
        self.densityTestSpin.valueChanged.connect(lambda _value: self._update_material_mode())
        self.planeTable.itemSelectionChanged.connect(self._update_selected_plane_status)
        self.planeTable.itemChanged.connect(lambda _item: self._mark_workflow_replay_editor_dirty())
        self.loadTable.itemChanged.connect(lambda _item: self._mark_workflow_replay_loads_dirty())
        self.checkRuntimeButton.clicked.connect(self.check_runtime)
        self.browseMpiLauncherButton.clicked.connect(self.browse_mpi_launcher)
        self.preprocessButton.clicked.connect(self.preprocess_inputs)
        self.preprocessNextButton.clicked.connect(self.preprocess_inputs_and_next)
        self.preprocessPreviewModeBox.currentTextChanged.connect(self._refresh_preprocess_preview_display)
        self.showIcpReferenceButton.clicked.connect(self.show_icp_reference_points)
        self.resetButton.clicked.connect(self.reset_scene)
        self.runElapsedTimer.timeout.connect(self._update_run_status_elapsed)
        self.customPreprocessingBox.currentIndexChanged.connect(self._on_custom_preprocessing_changed)
        self.editCustomPreprocessingButton.clicked.connect(self.edit_custom_preprocessing)
        self.resampleIsotropicCheckBox.toggled.connect(self._update_resample_mode)
        self.icpRegistrationCheckBox.toggled.connect(self._update_icp_target_label_ui)
        self.icpTargetImageBox.currentIndexChanged.connect(self._update_icp_target_label_ui)
        self.icpTargetNodeSelector.currentNodeChanged.connect(lambda _node: self._refresh_target_label_options())
        self.exportDisplacementsCheckBox.toggled.connect(self._sync_displacement_output_checkbox)
        self.cropPaddingSpin.valueChanged.connect(self._on_preprocess_spacing_default_edited)
        self.smoothSigmaSpin.valueChanged.connect(self._on_preprocess_spacing_default_edited)
        self.isotropicSpacingSpin.valueChanged.connect(self._on_isotropic_spacing_edited)
        self.exportButton.clicked.connect(self.export_case_only)
        self.saveWorkflowButton.clicked.connect(self.save_workflow_template)
        self.runButton.clicked.connect(self.run_case)
        self.stopButton.clicked.connect(self.stop_case)
        self.showDeformedButton.clicked.connect(self.show_deformed_result)
        self.deleteDeformedButton.clicked.connect(self.delete_deformed_result)
        self.deleteResultsButton.clicked.connect(self.delete_results)
        self.exportResultsCsvButton.clicked.connect(self.export_results_csv)
        self.saveResultAsButton.clicked.connect(self.save_result_as)
        self._update_profile_mode()
        self._update_contact_model_mode()
        self._update_material_mode()
        self._update_output_field_visibility()
        self._update_custom_preprocessing_ui()
        self._update_resample_mode()
        self._update_icp_target_label_ui()
        self._update_default_preprocess_spacing()
        self._update_default_isotropic_spacing()
        self._update_mask_requirement_ui()
        self._update_input_readiness_status()
        self._refresh_target_label_options()
        self._refresh_mask_segment_options()

    def _install_tooltips(self):
        tooltips = {
            self.checkRuntimeButton: "Check that Slicer's Python can import parosol-py and find the native solver/runtime.",
            self.imageSelector: "Select the image that will be converted to the FE material model.",
            self.maskSelector: "Optional mask, labelmap, or segmentation that defines the active anatomy or material labels.",
            self.maskSegmentBox: "Choose all labels or a subset from the selected mask/segmentation.",
            self.maskSegmentChecklist: "Check individual labels to include in the active mask subset.",
            self.profileBox: "Choose a built-in or loaded workflow. Use Apply Workflow to expand it into editable tables.",
            self.applyProfileButton: "Apply the selected workflow to the current scene and fill the guided preprocessing, material, contact, and load steps.",
            self.showWorkflowButton: "Show where the active workflow was loaded from and which ICP reference it points to.",
            self.loadProfileButton: "Load a .parosol-workflow or workflow YAML file from disk.",
            self.outputDirectory: "Folder where the active run writes YAML, inputs, results, fields, and logs.",
            self.derivativeDatasetRootSelector: "Optional dataset root. When set with subject and site, successful runs are recorded under derivatives/FEA.",
            self.derivativeSubjectEdit: "Subject identifier for derivative records. Either S01 or sub-S01 is accepted.",
            self.derivativeSiteEdit: "Site identifier for derivative records. Either tibia or site-tibia is accepted.",
            self.derivativeSessionEdit: "Optional session identifier for derivative records. Defaults to the output folder name.",
            self.quickRunButton: "Prepare the selected workflow visually, then run it with the generated contact regions and loads.",
            self.fastRecipeRunCheckBox: "Skip visual stage preparation and replay the selected recipe directly.",
            self.additionalOptionsButton: "Apply the workflow and move to materials so contact regions, loads, outputs, and solver settings can be edited.",
            self.resetButton: "Remove generated ParOSol scene nodes and clear previews. Files on disk are not deleted.",
            self.cropToMaskCheckBox: "Crop exported images around the selected mask/segmentation to reduce model size and runtime.",
            self.cropPaddingSpin: "Physical padding kept around the cropped mask. The default is about 5 voxels, displayed in mm.",
            self.customPreprocessingBox: "Choose a workflow-bundled Python preprocessing hook, or create one for special recipe-specific cleanup.",
            self.editCustomPreprocessingButton: "Open the selected custom preprocessing hook in a small Python editor.",
            self.largestComponentCheckBox: "Keep only the largest non-zero foreground component before export.",
            self.smoothDensityCheckBox: "Apply Gaussian smoothing to the density/grayscale image before material conversion.",
            self.smoothLabelsCheckBox: "Smooth the binary foreground and reapply original labels; use cautiously when exact label edges matter.",
            self.smoothSigmaSpin: "Gaussian smoothing sigma in mm. The default is about 1 voxel for the selected image.",
            self.resampleIsotropicCheckBox: "Resample to a near-isotropic voxel grid before solving.",
            self.isotropicSpacingSpin: "Target isotropic spacing in mm for resampling.",
            self.resampleDensityInterpolationBox: "Interpolation for continuous density/grayscale resampling. Masks and labels use nearest-neighbor.",
            self.icpRegistrationCheckBox: "Use workflow reference-point alignment to replay authored planes on the current scan.",
            self.showIcpReferenceButton: "Display the active ICP reference point cloud as a lightweight debug overlay.",
            self.icpTargetImageBox: "Choose whether ICP targets a saved workflow reference, the current sample, or a selected Slicer node.",
            self.icpTargetNodeSelector: "Selected Slicer node used when ICP target image is set to Slicer node.",
            self.icpTargetLabelBox: "Label used for ICP sampling, for example vertebral body label 20 in spine workflows.",
            self.preprocessPreviewModeBox: "Choose whether slice views show the preprocessed image, mask, or material map. This does not change export inputs.",
            self.preprocessButton: "Apply the selected image-preparation steps to the loaded Slicer nodes for visual authoring.",
            self.preprocessNextButton: "Apply image preparation, then move to contact regions.",
            self.materialModeBox: "Choose whether material properties come from label values or a density-to-modulus equation.",
            self.materialPresetBox: "Apply a predefined material table or density equation.",
            self.materialTable: "Map material labels to Young's modulus and Poisson ratio.",
            self.seedMaterialsButton: "Generate material table rows from non-zero labels in the selected image or mask. Use this only for label images, not density images.",
            self.applyMaterialsButton: "Apply the current material table and density equation choices.",
            self.applyMaterialsNextButton: "Apply the material settings, then move to image preparation.",
            self.addMaterialButton: "Add a material-label row.",
            self.deleteMaterialButton: "Delete the selected material-label row.",
            self.densityEquationBox: "Choose the equation used to convert density/grayscale values to Young's modulus.",
            self.densitySlopeSpin: "Slope for the linear density-to-modulus equation.",
            self.densityInterceptSpin: "Intercept for the linear density-to-modulus equation.",
            self.densityCoeffSpin: "Coefficient for power-law density-to-modulus equations.",
            self.densityExponentSpin: "Exponent for power-law density-to-modulus equations.",
            self.densityReferenceSpin: "Reference density used by normalized density-to-modulus equations.",
            self.densityQuadSpin: "Quadratic coefficient for polynomial density-to-modulus equations.",
            self.densityFloorSpin: "Minimum Young's modulus assigned to active voxels.",
            self.binMaterialCheckBox: "Quantize active density values into Ogo-compatible global bins before applying the density equation.",
            self.numberBinsSpin: "Number of global active-density bins; Ogo-style workflows usually use 128.",
            self.densityTestSpin: "Example density value used to preview the current material equation.",
            self.materialNuSpin: "Poisson ratio used for density-based material conversion.",
            self.editorGuideLabel: "Guided sequence for placing planes, generating contact regions, and previewing loads.",
            self.contactGuideLabel: "Short explanation of the current contact-region stage.",
            self.loadGuideLabel: "Short explanation of the current load-preview stage.",
            self.planeTable: "Contact-region table. Rows define where disks, contact regions, fixity, and loads are generated.",
            self.loadTable: "Load/fixity table linked to contact-region rows.",
            self.autoTopBottomButton: "Create default top and bottom contact planes from the selected image or mask bounds.",
            self.addPlaneButton: "Add a new editable contact/boundary plane.",
            self.deletePlaneButton: "Delete the selected contact/boundary plane.",
            self.createTopDiskButton: "Create a top disk plane from current bounds.",
            self.createBottomDiskButton: "Create a bottom disk plane from current bounds.",
            self.previewDisksButton: "Create disk/cap labels and/or bone-surface contact regions from the current plane table.",
            self.previewDisksNextButton: "Create contact regions, then move to loads.",
            self.deleteDisksButton: "Remove generated contact-region labels and previews.",
            self.previewLoadsButton: "Preview the load/fixity table on generated contact regions and draw load arrows.",
            self.previewLoadsNextButton: "Preview loads, then move to review and run.",
            self.loadArrowScaleSpin: "Display-only multiplier for load-arrow length; it does not change exported loads or solver inputs.",
            self.diskLabelSelector: "Generated or selected disk/cap labelmap used during export.",
            self.bcThickness: "Selection thickness for legacy boundary-condition previews.",
            self.bcLabelSelector: "Generated or selected contact-region labelmap.",
            self.runReadinessLabel: "Review status for the staged workflow before export or solve.",
            self.mpiProcessesSpin: "Number of solver processes. Start with 4 on a normal workstation.",
            self.mpiLauncherEdit: "Optional mpirun/mpiexec path. Leave empty to use the packaged parosol-py runtime.",
            self.browseMpiLauncherButton: "Browse for an external mpirun or mpiexec executable.",
            self.solverToleranceEdit: "Solver convergence tolerance. Use 1e-4 for routine desktop runs; tighten only for final checks.",
            self.failurePresetBox: "Failure estimate highlighted in the Slicer results panel.",
            self.exportButton: "Package a portable .parosol bundle that can be solved on another machine or cluster.",
            self.saveWorkflowButton: "Save the current workflow settings as a reusable .parosol-workflow template.",
            self.runButton: "Export the current case and run ParOSol.",
            self.stopButton: "Stop the running ParOSol process.",
            self.resultText: "Compact result summary from result.json.",
            self.exportResultsCsvButton: "Write a one-row CSV summary for the selected result.",
            self.saveResultAsButton: "Copy the current run into a clean renamed result folder for sharing or archiving.",
            self.deleteResultsButton: "Remove loaded/generated result preview nodes from the Slicer scene.",
            self.deformedScaleSpin: "Scale factor for deformation-arrow visualization.",
            self.deformedMaxArrowsSpin: "Maximum sampled deformation arrows. Increase for dense inspection; lower values keep the view responsive.",
            self.showDeformedButton: "Create displacement arrows from exported displacement fields.",
            self.deleteDeformedButton: "Remove deformation-arrow preview nodes.",
            self.logText: "Runtime log from setup, export, solver, and result-loading actions.",
        }
        for widget, text in tooltips.items():
            try:
                widget.toolTip = text
            except Exception:
                pass

    def _volume(self):
        return self.imageSelector.currentNode()

    def _current_input_storage_paths(self):
        return {
            "image": _node_storage_file(self.imageSelector.currentNode()),
            "mask": _node_storage_file(self.maskSelector.currentNode()),
        }

    def _workflow_replay_source_inputs(self):
        paths = getattr(self, "_workflowReplaySourceInputs", None)
        if not isinstance(paths, dict):
            paths = self._current_input_storage_paths()
        image = paths.get("image")
        if not image:
            return None
        resolved = {"image": str(image)}
        mask = paths.get("mask")
        if mask:
            resolved["mask"] = str(mask)
        return resolved

    def _canonicalize_inputs_to_parosol_ras_grid(self, image_node, mask_node):
        if not self._has_applied_workflow_replay_model():
            return image_node, mask_node, False
        if image_node is None:
            return image_node, mask_node, False
        material_override = self._material_override()
        image_is_label = bool(
            isinstance(material_override, dict)
            and str(material_override.get("image_type", "")).strip().lower()
            in {"material_labels", "labels", "segmentation"}
        )
        canonicalized = False
        if not _volume_node_has_parosol_ras_grid(image_node):
            image_node = _canonicalize_volume_node_to_parosol_ras_grid(
                image_node,
                "ParOSol_ras_image",
                label=image_is_label,
            )
            self.logic.group_node(image_node, "Inputs")
            canonicalized = True
        if mask_node is not None and _is_segmentation_node(mask_node):
            mask_node = _segmentation_to_labelmap_node(mask_node, image_node)
            self.logic.style_labelmap(mask_node, "mask")
            self.logic.group_node(mask_node, "Inputs")
            canonicalized = True
        if mask_node is not None and not _volume_node_has_parosol_ras_grid(mask_node):
            mask_node = _canonicalize_volume_node_to_parosol_ras_grid(
                mask_node,
                "ParOSol_ras_mask",
                label=True,
            )
            self.logic.style_labelmap(mask_node, "mask")
            self.logic.group_node(mask_node, "Inputs")
            canonicalized = True
        return image_node, mask_node, canonicalized

    def _workflow_replay_preview_volume(self):
        node = getattr(self, "_workflowReplayPreviewMaterialNode", None)
        try:
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                return node
        except Exception:
            pass
        return self._volume()

    def _is_interactive_profile(self):
        text = self._widget_text(self.profileBox).strip()
        if text.lower() == "interactive_custom":
            return True
        try:
            config = self._load_slicer_profile_config(text)
        except Exception:
            return False
        if _workflow_template_type(config) == "load_history":
            return False
        return isinstance(config.get("slicer_editor"), dict)

    def _source_profile_for_fast_run(self):
        profile = str(self._appliedProfileName or self._widget_text(self.profileBox, "")).strip()
        if not profile:
            return None
        token = Path(profile).name if ("/" in profile or "\\" in profile) else profile
        token = token.strip()
        if token.lower() in {"interactive_custom", "custom", ""}:
            return None
        if Path(profile).exists():
            return None
        return token

    def _fast_recipe_run_enabled(self):
        return bool(getattr(getattr(self, "fastRecipeRunCheckBox", None), "checked", False))

    def _workflow_mask_requirement(self, config=None, profile_text=None):
        token = str(
            profile_text
            if profile_text is not None
            else self._widget_text(getattr(self, "profileBox", None), "")
        ).strip()
        key = Path(token).stem.lower() if ("/" in token or "\\" in token) else token.lower()
        if key in {"interactive_custom", "custom", ""}:
            return "optional"
        if key in {"xtremecti", "xtremectii"}:
            return "not_used"
        if not isinstance(config, dict):
            config = self._active_workflow_config()
        model = config.get("model", {}) if isinstance(config, dict) else {}
        model_type = str(model.get("type", "")).strip().lower() if isinstance(model, dict) else ""
        if model_type in {"spine_compression", "proximal_femur_sideways_fall"}:
            return "required"
        registration = model.get("registration", {}) if isinstance(model, dict) else {}
        if (
            model_type == "workflow_replay"
            and isinstance(model.get("labels"), dict)
            and isinstance(registration, dict)
            and _enabled_value(registration.get("enabled", False))
        ):
            return "required"
        if isinstance(config, dict):
            input_cfg = config.get("input", {})
            image_type = str(
                config.get("image_type", "")
                or (input_cfg.get("image_type", "") if isinstance(input_cfg, dict) else "")
            ).strip().lower()
            if image_type == "material_labels" and not isinstance(model.get("labels"), dict):
                return "not_used"
            if isinstance(input_cfg, dict) and bool(input_cfg.get("mask_required", False)):
                return "required"
            if bool(config.get("mask_required", False)):
                return "required"
        return "optional"

    def _workflow_requires_mask(self, config=None, profile_text=None):
        return self._workflow_mask_requirement(config, profile_text) == "required"

    def _update_mask_subset_visibility(self, segment_visible, *, subset_visible=False):
        segment_visible = bool(segment_visible)
        subset_visible = bool(subset_visible) and segment_visible
        for attr in ("maskSegmentBox", "maskSegmentLabel"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.visible = segment_visible
        for attr in ("maskSegmentChecklist", "maskSubsetLabel"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.visible = subset_visible

    def _workflow_input_requirements(self, config=None, profile_text=None):
        if not isinstance(config, dict):
            config = self._active_workflow_config()
        profile_text = str(
            profile_text
            if profile_text is not None
            else self._widget_text(getattr(self, "profileBox", None), "")
        ).strip()
        key = Path(profile_text).stem.lower() if ("/" in profile_text or "\\" in profile_text) else profile_text.lower()
        requirement = self._workflow_mask_requirement(config, profile_text)
        requirements = {
            "version": 1,
            "mask": requirement,
            "summary": "",
            "image_labels": {},
            "mask_labels": {},
        }
        if key in {"xtremecti", "xtremectii"}:
            requirements["summary"] = (
                "Image-only HR-pQCT workflow. Select a pre-segmented material-label image; "
                "label 100 is trabecular bone and label 127 is cortical bone. No separate mask is used."
            )
            requirements["image_labels"] = {
                "100": "trabecular bone",
                "127": "cortical bone",
            }
            return requirements
        model = config.get("model", {}) if isinstance(config, dict) else {}
        labels = model.get("labels", {}) if isinstance(model, dict) else {}
        if isinstance(labels, dict):
            requirements["mask_labels"] = {str(key): value for key, value in labels.items()}
        if requirement == "required":
            requirements["summary"] = (
                "Select the density image and a mask/label segmentation containing the workflow anatomy labels."
            )
        elif requirement == "not_used":
            requirements["summary"] = "Image-only material-label workflow. No separate mask is used."
        else:
            requirements["summary"] = "Mask/label input is optional for this editable workflow."
        return requirements

    def _update_mask_requirement_ui(self):
        if not hasattr(self, "maskSelector"):
            return
        profile_text = self._widget_text(getattr(self, "profileBox", None), "")
        config = self._active_workflow_config()
        requirement = self._workflow_mask_requirement(config, profile_text)
        input_requirements = self._workflow_input_requirements(config, profile_text)
        mask_enabled = requirement != "not_used"
        self.maskSelector.enabled = mask_enabled
        if not mask_enabled:
            try:
                self.maskSelector.setCurrentNode(None)
            except Exception:
                pass
        mask_has_node = mask_enabled and self.maskSelector.currentNode() is not None
        if hasattr(self, "maskSegmentBox"):
            self.maskSegmentBox.enabled = mask_has_node
        if hasattr(self, "maskSegmentChecklist"):
            self.maskSegmentChecklist.enabled = mask_has_node
        self._update_mask_subset_visibility(mask_has_node)
        if hasattr(self, "maskRequirementLabel"):
            if requirement == "required":
                self.maskRequirementLabel.text = "Required for this workflow."
            elif requirement == "not_used":
                self.maskRequirementLabel.text = "Not used for this image-only workflow."
            else:
                self.maskRequirementLabel.text = ""
            self.maskRequirementLabel.visible = bool(self.maskRequirementLabel.text)
        if hasattr(self, "workflowInstructionLabel"):
            self.workflowInstructionLabel.text = str(input_requirements.get("summary", ""))
        self._update_input_readiness_status()

    def _update_input_readiness_status(self):
        if not hasattr(self, "inputReadinessLabel"):
            return
        image_node = self._volume() if hasattr(self, "imageSelector") else None
        profile_text = self._widget_text(getattr(self, "profileBox", None), "interactive_custom")
        config = self._active_workflow_config()
        requirement = self._workflow_mask_requirement(config, profile_text)
        if image_node is None:
            message = "Select an image or material-label volume."
        elif requirement == "required" and self.maskSelector.currentNode() is None:
            message = "Mask/label segmentation required before this workflow can run."
        elif requirement == "not_used":
            message = "Ready: this workflow uses the image/material labels only."
        else:
            message = "Ready to run or edit the selected workflow."
        self.inputReadinessLabel.text = message

    def _format_elapsed_seconds(self, seconds):
        seconds = max(0, int(seconds))
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _set_run_status(self, message, *, error=False, update_stage=True):
        if update_stage and getattr(self, "_runStartedAt", None) is not None and not error:
            self._runStatusStage = str(message)
        if hasattr(self, "runStatusLabel"):
            self.runStatusLabel.text = str(message)
        if hasattr(self, "runProgressBar") and error:
            self.runProgressBar.visible = False

    def _start_run_status(self, message):
        self._runStartedAt = time.monotonic()
        self._runStatusStage = str(message)
        if hasattr(self, "runProgressBar"):
            self.runProgressBar.setRange(0, 0)
            self.runProgressBar.visible = True
        if hasattr(self, "runElapsedTimer"):
            self.runElapsedTimer.start(1000)
        self._update_run_status_elapsed()
        self._flush_run_status_ui()

    def _flush_run_status_ui(self):
        for widget_name in ("runStatusLabel", "runProgressBar", "inputActionGroup"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.repaint()
            except Exception:
                pass
        try:
            qt.QApplication.processEvents()
        except Exception:
            pass

    def _finish_run_status(self, message, *, error=False):
        elapsed = None
        if getattr(self, "_runStartedAt", None) is not None:
            elapsed = self._format_elapsed_seconds(time.monotonic() - float(self._runStartedAt))
        self._runStartedAt = None
        if hasattr(self, "runElapsedTimer"):
            self.runElapsedTimer.stop()
        if hasattr(self, "runProgressBar"):
            self.runProgressBar.setRange(0, 1)
            self.runProgressBar.value = 0 if error else 1
            self.runProgressBar.visible = False
        suffix = f" Elapsed {elapsed}." if elapsed is not None else ""
        self._set_run_status(f"{message}{suffix}", error=error)

    def _update_run_status_elapsed(self):
        if getattr(self, "_runStartedAt", None) is None:
            return
        elapsed = self._format_elapsed_seconds(time.monotonic() - float(self._runStartedAt))
        stage = getattr(self, "_runStatusStage", "Running...")
        self._set_run_status(f"Stage: {stage} | Elapsed {elapsed}", update_stage=False)

    def _uses_generated_interactive_model(self):
        return (
            self._is_interactive_profile()
            and (
                bool(self._profileHasGeneratedBoundaryConditions)
                or self._has_applied_workflow_replay_model()
            )
        )

    def _select_workflow_tab(self, tab_name):
        tabs = getattr(self, "workflowTabs", None)
        if tabs is None:
            return
        key = str(tab_name or "").strip().lower().replace("_", "-")
        tab_indexes = {
            "inputs": 0,
            "workflow": 0,
            "materials": 1,
            "material": 1,
            "preprocess": 2,
            "image-prep": 2,
            "image-preparation": 2,
            "anatomy": 2,
            "contact": 3,
            "contact-regions": 3,
            "boundary": 3,
            "loads": 4,
            "review": 5,
            "run": 5,
            "solve": 5,
            "results": 6,
        }
        index = tab_indexes.get(key)
        if index is None:
            return
        try:
            count_attr = getattr(tabs, "count", 0)
            count = count_attr() if callable(count_attr) else int(count_attr)
            if 0 <= int(index) < int(count):
                tabs.setCurrentIndex(int(index))
        except Exception:
            try:
                tabs.setCurrentIndex(int(index))
            except Exception:
                pass

    def _advance_workflow_tab_after(self, stage):
        next_tabs = {
            "workflow": "materials",
            "materials": "preprocess",
            "anatomy": "contact",
            "boundary": "loads",
            "loads": "review",
            "export": "review",
            "results": "results",
        }
        self._select_workflow_tab(next_tabs.get(str(stage).strip().lower(), stage))

    def _open_additional_options(self):
        profile_text = self._widget_text(getattr(self, "profileBox", None), "interactive_custom")
        has_applied_config = isinstance(getattr(self, "_appliedProfileConfig", None), dict)
        if str(profile_text).strip().lower() != "interactive_custom" or not has_applied_config:
            self.apply_profile()
        else:
            self._select_workflow_tab("materials")

    def _workflow_stage_controller(self):
        controller = getattr(self, "_stageController", None)
        if not isinstance(controller, WorkflowStageController):
            self._workflowStageState = getattr(self, "_workflowStageState", WorkflowStageState())
            self._stageController = WorkflowStageController(self._workflowStageState)
            controller = self._stageController
        return controller

    def _stage_state(self):
        controller = self._workflow_stage_controller()
        return controller.state()

    def _mark_boundary_preview_stale(self):
        self._workflow_stage_controller().mark_boundary_preview_stale()
        if hasattr(self, "profileStatusLabel"):
            self._update_profile_mode()

    def _mark_load_preview_stale(self):
        self._workflow_stage_controller().mark_load_preview_stale()
        if hasattr(self, "profileStatusLabel"):
            self._update_profile_mode()

    def _mark_stage_complete(self, stage):
        self._workflow_stage_controller().mark_stage_complete(stage)
        if hasattr(self, "profileStatusLabel"):
            self._update_profile_mode()

    def _has_applied_workflow_replay_model(self):
        source = getattr(self, "_appliedProfileConfig", None)
        if not isinstance(source, dict):
            return False
        model = source.get("model")
        if not isinstance(model, dict):
            return False
        replay = model.get("workflow_replay", {})
        return isinstance(replay, dict) and bool(replay.get("enabled", False))

    def _editable_profile_matches_builtin_xtremect_axial(self):
        if self._applied_profile_key() not in {"xtremecti", "xtremectii"}:
            return False
        if not self._is_interactive_profile():
            return False
        disk_node = None
        try:
            disk_node = self.diskLabelSelector.currentNode()
        except Exception:
            disk_node = None
        if disk_node is not None:
            try:
                if np.count_nonzero(slicer.util.arrayFromVolume(disk_node)):
                    return False
            except Exception:
                return False
        if len(getattr(self, "contactPlaneRows", [])) != 2:
            return False
        try:
            specs = [self._contact_row_spec(0), self._contact_row_spec(1)]
        except Exception:
            return False
        by_name = {str(spec.get("name", "")).strip().lower(): spec for spec in specs}
        top = by_name.get("top")
        bottom = by_name.get("bottom")
        if top is None or bottom is None:
            return False
        if not self._is_default_xtremect_surface_row(top, bc_type="Dirichlet", axis="z"):
            return False
        if not self._is_default_xtremect_surface_row(bottom, bc_type="Fixed", axis="z"):
            return False
        units = str(top.get("units", "")).strip().lower()
        if units not in {"%", "percent", "percentage"}:
            return False
        try:
            return np.isclose(abs(_load_value_number(top)), 1.0)
        except Exception:
            return False

    def _is_default_xtremect_surface_row(self, spec, *, bc_type, axis):
        return (
            str(spec.get("contact", "")).strip().lower() == "bone surface"
            and str(spec.get("surface_mode", "")).strip().lower() == "intersect"
            and str(spec.get("bc_type", "")).strip().lower() == bc_type.strip().lower()
            and str(spec.get("axis", "")).strip().lower() == axis.strip().lower()
        )

    def _update_profile_mode(self):
        interactive = self._is_interactive_profile()
        if not hasattr(self, "editorCollapsible"):
            return
        self.editorCollapsible.visible = interactive
        self.editorCollapsible.enabled = interactive
        if hasattr(self, "loadCollapsible"):
            self.loadCollapsible.visible = interactive
            self.loadCollapsible.enabled = interactive
        self._set_workflow_tab_enabled(3, interactive)
        self._set_workflow_tab_enabled(4, interactive)
        generated = self._uses_generated_interactive_model()
        if hasattr(self, "previewDisksButton"):
            self.previewDisksButton.text = "Create Regions"
            self.previewLoadsButton.text = "Preview Loads"
            self.previewDisksButton.toolTip = "Create disk/cap or bone-surface contact regions for the editable model."
            self.previewLoadsButton.toolTip = "Preview the current load/fixity table on generated contact regions."
            if hasattr(self, "previewDisksNextButton"):
                self.previewDisksNextButton.toolTip = "Create contact regions, then move to loads."
            if hasattr(self, "previewLoadsNextButton"):
                self.previewLoadsNextButton.toolTip = "Preview loads, then move to review and run."
        if hasattr(self, "profileStatusLabel"):
            self.profileStatusLabel.text = self._workflow_stage_controller().status_text(generated=generated)
        stage_summary = self._workflow_stage_controller().stage_summary(generated=generated)
        if hasattr(self, "workflowStageLabel"):
            self.workflowStageLabel.text = stage_summary
        if hasattr(self, "runReadinessLabel"):
            self.runReadinessLabel.text = stage_summary

    def _on_workflow_selection_changed(self, *_args):
        if getattr(self, "_suppressWorkflowSelectionUpdate", False):
            self._update_profile_mode()
            self._update_mask_requirement_ui()
            return
        self._apply_selected_workflow_settings_preview()
        self._update_profile_mode()
        self._update_mask_requirement_ui()

    def _apply_selected_workflow_settings_preview(self):
        token = self._widget_text(self.profileBox, "interactive_custom")
        if str(token).strip().lower() == "interactive_custom":
            return
        try:
            config = self._load_slicer_profile_config(token)
        except Exception:
            return
        try:
            self._selectedWorkflowConfig = config
            self._selectedWorkflowName = token
            self._clear_editable_profile_state()
            self._apply_profile_materials(config.get("materials", {}))
            self._apply_profile_preprocessing(config.get("preprocessing", config.get("preprocess", {})))
            self._apply_profile_solver_settings(config.get("solver", {}))
            self._apply_profile_output_settings(config.get("output", {}), config.get("solver", {}))
            self._apply_profile_postprocess_settings(config.get("postprocess", {}))
            self._apply_profile_model_settings(config.get("model", {}))
            self._appliedWorkflowNodesetLabels = _workflow_nodeset_label_map(config)
            self._appliedProfileConfig = config
            self._preprocessingAppliedToInputs = False
            if _workflow_template_type(config) != "load_history" and not self._workflow_has_preprocessing_to_apply():
                editor = config.get("slicer_editor")
                if not isinstance(editor, dict):
                    editor = _editor_state_from_config(config)
                self._apply_profile_planes_and_loads(editor)
        except Exception:
            return

    def load_profile_file(self):
        selected = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select ParOSol Workflow",
            str(USER_WORKFLOW_ROOT),
            "ParOSol workflows (*.parosol-workflow);;YAML workflows (*.yaml *.yml);;All files (*)",
        )
        path = selected[0] if isinstance(selected, tuple) else selected
        if not path:
            return
        self.profileBox.setCurrentText(str(path))
        self.apply_profile()

    def show_active_workflow_file(self):
        try:
            profile_text = self._widget_text(self.profileBox, "interactive_custom")
            source_path = self._selected_workflow_source_path(profile_text)
            config = self._load_slicer_profile_config(profile_text)
            model = config.get("model", {}) if isinstance(config, dict) else {}
            registration = model.get("registration", {}) if isinstance(model, dict) else {}
            workflow_template = config.get("workflow_template", {}) if isinstance(config, dict) else {}
            reference = registration.get("reference_points", "")
            if reference and source_path is not None:
                workflow_dir = source_path.parent if source_path.is_file() else source_path
                if source_path.name.lower().endswith(".parosol-workflow"):
                    reference_text = str(reference)
                else:
                    reference_text = str((workflow_dir / str(reference)).expanduser())
            else:
                reference_text = str(reference or "not configured")
            lines = [
                f"Workflow: {profile_text}",
                f"Source: {source_path or 'built-in template'}",
                f"Profile: {workflow_template.get('profile', profile_text) if isinstance(workflow_template, dict) else profile_text}",
                f"Model type: {model.get('type', 'not configured') if isinstance(model, dict) else 'not configured'}",
                f"ICP enabled: {registration.get('enabled', False) if isinstance(registration, dict) else False}",
                f"ICP method: {registration.get('method', 'not configured') if isinstance(registration, dict) else 'not configured'}",
                f"Reference: {reference_text}",
                f"Reference coordinate system: {registration.get('reference_coordinate_system', registration.get('coordinate_system', 'not configured')) if isinstance(registration, dict) else 'not configured'}",
            ]
            message = "\n".join(lines)
            self._append_log(message + "\n")
            qt.QMessageBox.information(slicer.util.mainWindow(), "Active ParOSol Workflow", message)
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

    def show_icp_reference_points(self):
        try:
            points, source = self._icp_reference_points_for_display()
            model = self.logic.create_point_cloud_model(
                "ParOSol_icp_reference_points",
                points,
                color=(0.1, 0.85, 1.0),
                folder_name="Debug",
                point_size=5.0,
                opacity=0.95,
            )
            if model is None:
                raise ValueError("ICP reference point cloud is empty.")
            self._append_log(
                f"Displayed ICP reference point cloud: {int(points.shape[0])} points from {source}.\n"
            )
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

    def _icp_reference_points_for_display(self):
        source = self._selected_icp_target_source()
        if source == "slicer-node":
            points = self._icp_reference_points_from_selected_target_node()
            node = self._selected_icp_target_node()
            return np.asarray(points, dtype=np.float32), _node_reference_description(node)

        cached = self._cached_self_reference_points_for_current_preview()
        if source == "self" and cached is not None:
            return np.asarray(cached, dtype=np.float32), "current ICP self-reference preview"

        reference = self._current_workflow_reference_points()
        display_cached = self._cached_icp_reference_points_for_current_preview(reference)
        if source == "workflow-reference" and display_cached is not None:
            return np.asarray(display_cached, dtype=np.float32), "current ICP display-space reference preview"

        if reference:
            config = self._active_workflow_config()
            model = config.get("model", {}) if isinstance(config, dict) else {}
            registration = model.get("registration", {}) if isinstance(model, dict) else {}
            max_points = int(registration.get("max_points", 8000)) if isinstance(registration, dict) else 8000
            points = read_reference_points(
                reference,
                max_points=max_points,
                coordinate_system=self._current_workflow_reference_coordinate_system(),
            )
            if points.size == 0:
                raise ValueError(f"ICP reference point cloud is empty: {reference}")
            return np.asarray(points, dtype=np.float32), str(reference)

        if cached is not None:
            return np.asarray(cached, dtype=np.float32), "current ICP self-reference preview"

        if not reference:
            raise ValueError(
                "No ICP reference point cloud is available. Enable ICP and click Prepare Image "
                "to author a self-reference, or apply a workflow with model.registration.reference_points."
            )

    def _selected_workflow_source_path(self, profile_text):
        token = str(profile_text).strip()
        path = Path(token).expanduser()
        if path.is_dir():
            workflow = path / "workflow.yaml"
            if workflow.is_file():
                return workflow.resolve()
        if path.is_file():
            return path.resolve()
        builtin_path = _builtin_workflow_path(token)
        if builtin_path is not None:
            return Path(builtin_path).expanduser().resolve()
        return None

    def browse_mpi_launcher(self):
        start_dir = Path(str(self.mpiLauncherEdit.text or "") or Path.home()).expanduser()
        if start_dir.is_file():
            start_dir = start_dir.parent
        selected = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select MPI launcher",
            str(start_dir),
            "MPI launchers (mpirun mpiexec);;All files (*)",
        )
        path = selected[0] if isinstance(selected, tuple) else selected
        if not path:
            return
        self.mpiLauncherEdit.setText(str(path))
        self.logic.set_setting_value("SlicerParOSol/mpiLauncher", str(path))

    def _apply_profile_without_tab_advance(self):
        previous = getattr(self, "_suppressWorkflowTabAdvance", False)
        self._suppressWorkflowTabAdvance = True
        try:
            self.apply_profile()
        finally:
            self._suppressWorkflowTabAdvance = previous

    def apply_profile(self):
        try:
            profile_text = self._widget_text(self.profileBox, "interactive_custom")
            config = self._load_slicer_profile_config(profile_text)
            self._apply_profile_config(config)
            self._appliedProfileName = profile_text
            self._profileHasGeneratedBoundaryConditions = False
            self._mark_stage_complete("workflow")
            if self.profileBox.findText("interactive_custom") >= 0:
                self._suppressWorkflowSelectionUpdate = True
                try:
                    self.profileBox.setCurrentText("interactive_custom")
                finally:
                    self._suppressWorkflowSelectionUpdate = False
            self._update_profile_mode()
            self._update_mask_requirement_ui()
            self._append_log(
                f"Applied workflow '{profile_text}' as editable Slicer state. Review Materials, then click Prepare Image to update previews and workflow geometry.\n"
            )
            if not getattr(self, "_suppressWorkflowTabAdvance", False):
                self._advance_workflow_tab_after("workflow")
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

    def _auto_prepare_applied_profile(self):
        if not self._is_interactive_profile() or self._volume() is None:
            return
        if self._workflow_has_preprocessing_to_apply():
            self.preprocess_inputs()
        editor = self._editor_from_applied_profile()
        if isinstance(editor, dict):
            config = self._active_workflow_config()
            if isinstance(config, dict):
                self._appliedWorkflowNodesetLabels = _workflow_nodeset_label_map(config)
                self._apply_profile_custom_preprocessing(config.get("custom_preprocessing"))
            self._clear_editable_profile_state(clear_materials=False)
            self._apply_profile_planes_and_loads(
                self._resolve_reference_space_editor_for_current_sample(editor)
            )
        if not getattr(self, "contactPlaneRows", []):
            self._append_log("Workflow has no editable contact planes; nothing to auto-generate.\n")
            return
        self.preview_disks()
        if self.bcLabelSelector.currentNode() is not None:
            self.preview_loads()

    def _workflow_has_preprocessing_to_apply(self):
        if getattr(self, "_preprocessingAppliedToInputs", False):
            return False
        return any(
            bool(getattr(widget, "checked", False))
            for widget in (
                getattr(self, "cropToMaskCheckBox", None),
                getattr(self, "largestComponentCheckBox", None),
                getattr(self, "smoothDensityCheckBox", None),
                getattr(self, "smoothLabelsCheckBox", None),
                getattr(self, "resampleIsotropicCheckBox", None),
                getattr(self, "icpRegistrationCheckBox", None),
            )
            if widget is not None
        ) or bool(self._custom_preprocessing_config())

    def _editor_from_applied_profile(self):
        config = getattr(self, "_appliedProfileConfig", None)
        if not isinstance(config, dict):
            return None
        editor = config.get("slicer_editor")
        if isinstance(editor, dict):
            return editor
        if _workflow_template_type(config) == "load_history":
            return {"planes": [], "loads": []}
        return _editor_state_from_config(config)

    def _active_workflow_config(self):
        token = self._widget_text(self.profileBox, "")
        if token and str(token).strip().lower() != "interactive_custom":
            selected_name = getattr(self, "_selectedWorkflowName", "")
            selected = getattr(self, "_selectedWorkflowConfig", None)
            if str(selected_name) == str(token) and isinstance(selected, dict):
                return selected
            try:
                config = self._load_slicer_profile_config(token)
            except Exception:
                config = None
            if isinstance(config, dict):
                self._selectedWorkflowConfig = config
                self._selectedWorkflowName = token
                return config
        config = getattr(self, "_appliedProfileConfig", None)
        if isinstance(config, dict):
            return config
        config = getattr(self, "_selectedWorkflowConfig", None)
        if isinstance(config, dict):
            return config
        return None

    def _editor_from_active_workflow(self):
        editor = self._editor_from_applied_profile()
        if isinstance(editor, dict):
            return editor
        config = self._active_workflow_config()
        if not isinstance(config, dict):
            return None
        editor = config.get("slicer_editor")
        if isinstance(editor, dict):
            return editor
        if _workflow_template_type(config) == "load_history":
            return {"planes": [], "loads": []}
        return _editor_state_from_config(config)

    def _load_slicer_profile_config(self, profile_text):
        token = str(profile_text).strip()
        path = Path(token).expanduser()
        if path.is_dir():
            path = path / "workflow.yaml"
        if path.is_file() and path.name.lower().endswith(".parosol-workflow"):
            path = self._extract_workflow_bundle(path)
        builtin_path = _builtin_workflow_path(token)
        if not path.exists() and builtin_path is not None:
            path = builtin_path
        if path.is_file() and path.name.lower().endswith(".parosol-workflow"):
            path = self._extract_workflow_bundle(path)
        if path.exists():
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"Workflow file is not a mapping: {path}")
            return _resolve_workflow_relative_paths(data, path.parent)
        return _slicer_profile_template(token)

    def _extract_workflow_bundle(self, path):
        stage = Path(tempfile.mkdtemp(prefix="slicer_parosol_workflow_"))
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (stage / member.filename).resolve()
                if not str(target).startswith(str(stage.resolve())):
                    raise ValueError(f"Unsafe workflow bundle member: {member.filename}")
                archive.extract(member, stage)
        for name in ("workflow.yaml", "workflow.yml", "parosol_slicer_case.yaml"):
            candidate = stage / name
            if candidate.exists():
                return candidate
        raise ValueError(f"Workflow bundle does not contain workflow.yaml: {path}")

    def _apply_profile_config(self, config):
        if not isinstance(config, dict):
            raise ValueError("Workflow config must be a mapping")
        self._clear_editable_profile_state()
        self._apply_profile_materials(config.get("materials", {}))
        self._apply_profile_preprocessing(config.get("preprocessing", config.get("preprocess", {})))
        self._apply_profile_custom_preprocessing(config.get("custom_preprocessing"))
        self._apply_profile_solver_settings(config.get("solver", {}))
        self._apply_profile_output_settings(config.get("output", {}), config.get("solver", {}))
        self._apply_profile_postprocess_settings(config.get("postprocess", {}))
        self._apply_profile_model_settings(config.get("model", {}))
        self._appliedWorkflowNodesetLabels = _workflow_nodeset_label_map(config)
        editor = config.get("slicer_editor")
        if not isinstance(editor, dict):
            if _workflow_template_type(config) == "load_history":
                editor = {"planes": [], "loads": []}
            else:
                editor = _editor_state_from_config(config)
        self._appliedProfileConfig = config
        self._preprocessingAppliedToInputs = False
        self._workflowReplayContractEditor = None
        self._workflowReplayResolvedEditor = None
        self._workflowReplayResolvedEditorDirty = False
        if self._workflow_has_preprocessing_to_apply():
            self._append_log(
                "Workflow has image preparation enabled; contact planes will be created after Prepare Image.\n"
            )
        else:
            self._apply_profile_planes_and_loads(editor)
        if self._has_applied_workflow_replay_model():
            self._workflowReplayContractEditor = copy.deepcopy(editor)
            self._workflowReplayResolvedEditorDirty = False
        self._update_material_mode()
        self._update_profile_mode()

    def _clear_editable_profile_state(self, *, clear_materials=True):
        for row_data in getattr(self, "contactPlaneRows", []):
            try:
                self.logic.remove_node(row_data.get("plane"))
            except Exception:
                pass
        self.contactPlaneRows = []
        self.planeTable.setRowCount(0)
        self.loadTable.setRowCount(0)
        if clear_materials:
            self.materialTable.setRowCount(0)
        try:
            self._delete_bc_arrow_models()
        except Exception:
            pass

    def _apply_profile_materials(self, materials):
        materials = materials or {}
        density = materials.get("density") if isinstance(materials, dict) else None
        nonlinear = materials.get("nonlinear") if isinstance(materials, dict) else None
        if isinstance(density, dict):
            self.materialModeBox.setCurrentText(
                "Nonlinear density formula" if isinstance(nonlinear, dict) else "Linear density formula"
            )
            if isinstance(nonlinear, dict):
                preset = str(nonlinear.get("preset", "spine_nonlinear")).strip().lower()
                if preset == "hip_nonlinear":
                    self.nonlinearPresetBox.setCurrentText("Hip nonlinear")
                elif preset == "manual":
                    self.nonlinearPresetBox.setCurrentText("Manual")
                    self._apply_manual_nonlinear_law_to_widgets(
                        "elastic", nonlinear.get("elastic", {})
                    )
                    self._apply_manual_nonlinear_law_to_widgets(
                        "compression", nonlinear.get("compressive_yield", {})
                    )
                    self._apply_manual_nonlinear_law_to_widgets(
                        "tension", nonlinear.get("tensile_yield", {})
                    )
                else:
                    self.nonlinearPresetBox.setCurrentText("Spine nonlinear")
            e_config = density.get("E", {})
            equation = str(e_config.get("equation", "linear")).strip().lower()
            if equation in {"mulder", "mulder2007", "mulder_2007", "framework_mulder", "framework_mulder2007"}:
                self._set_density_preset(
                    "mulder2007",
                    slope=e_config.get("slope", e_config.get("a", 25.0)),
                    intercept=e_config.get("intercept", e_config.get("b", -5830.0)),
                    floor=_density_floor_config_value(e_config, density),
                )
            elif equation == "power":
                self._set_density_preset(
                    "power",
                    coefficient=e_config.get("coefficient", 1.0),
                    exponent=e_config.get("exponent", 1.0),
                    reference=e_config.get("reference_density", e_config.get("reference", 1000.0)),
                    floor=_density_floor_config_value(e_config, density),
                )
            elif equation == "polynomial":
                coefficients = list(e_config.get("coefficients", [0.0, 1.0, 0.0]))
                coefficients += [0.0] * max(0, 3 - len(coefficients))
                self._set_density_preset(
                    "polynomial",
                    intercept=coefficients[0],
                    slope=coefficients[1],
                    floor=_density_floor_config_value(e_config, density),
                )
                self.densityQuadSpin.value = float(coefficients[2])
            else:
                self._set_density_preset(
                    "linear",
                    slope=e_config.get("slope", 10.0),
                    intercept=e_config.get("intercept", 0.0),
                    floor=_density_floor_config_value(e_config, density),
                )
            bin_material = _enabled_value(
                density.get(
                    "bin_material",
                    e_config.get("bin_material", e_config.get("binned_material", False)),
                )
            )
            self.binMaterialCheckBox.checked = bin_material
            self.numberBinsSpin.value = int(
                density.get(
                    "number_bins",
                    density.get("bins", e_config.get("number_bins", e_config.get("bins", 128))),
                )
            )
            self.materialNuSpin.value = float(density.get("nu", materials.get("nu", 0.3)))
            self._update_material_mode()
            return
        labels = materials.get("labels", {}) if isinstance(materials, dict) else {}
        if labels:
            rows = []
            for label, spec in labels.items():
                rows.append(
                    (
                        int(label),
                        str(spec.get("name", f"label_{label}")),
                        float(spec.get("E", spec.get("e", 8748.0))),
                        float(spec.get("nu", 0.3)),
                    )
                )
            self._set_label_material_preset(sorted(rows, key=lambda item: item[0]))

    def _apply_profile_preprocessing(self, preprocessing):
        preprocessing = preprocessing or {}
        self._isotropic_spacing_user_override = False
        self._isotropic_spacing_workflow_override = False
        self._resample_spacing_tolerance_mm = 1.0e-6
        self._resample_spacing_tolerance_relative = 1.0e-3
        self._resample_canonicalize_within_tolerance = False
        if hasattr(self, "cropToMaskCheckBox"):
            self.cropToMaskCheckBox.checked = False
        if hasattr(self, "largestComponentCheckBox"):
            self.largestComponentCheckBox.checked = False
        if hasattr(self, "smoothDensityCheckBox"):
            self.smoothDensityCheckBox.checked = False
        if hasattr(self, "smoothLabelsCheckBox"):
            self.smoothLabelsCheckBox.checked = False
        if hasattr(self, "resampleIsotropicCheckBox"):
            self.resampleIsotropicCheckBox.checked = False
        crop_cfg = preprocessing.get("crop_to_bb", preprocessing.get("crop_to_mask", {}))
        if isinstance(crop_cfg, dict):
            if hasattr(self, "cropToMaskCheckBox"):
                self.cropToMaskCheckBox.checked = _enabled_value(crop_cfg.get("enabled", True))
            margin = crop_cfg.get("margin_mm", crop_cfg.get("margin_voxels"))
            if margin is not None and hasattr(self, "cropPaddingSpin"):
                self.cropPaddingSpin.value = float(margin)
        elif hasattr(self, "cropToMaskCheckBox"):
            self.cropToMaskCheckBox.checked = _enabled_value(crop_cfg)
        bbox_cfg = preprocessing.get("bbox_ratio")
        if bbox_cfg is not None:
            if isinstance(bbox_cfg, dict):
                aspect_ratio = bbox_cfg.get(
                    "ratio",
                    bbox_cfg.get("ratios", bbox_cfg.get("bbox_ratio")),
                )
            else:
                aspect_ratio = bbox_cfg
        else:
            aspect_cfg = preprocessing.get(
                "normalize_aspect_ratio",
                preprocessing.get("aspect_ratio", preprocessing.get("aspect-ratio", {})),
            )
            if isinstance(aspect_cfg, dict):
                aspect_ratio = aspect_cfg.get(
                    "ratio",
                    aspect_cfg.get("ratios", aspect_cfg.get("aspect_ratio")),
                )
            else:
                aspect_ratio = aspect_cfg
            if aspect_ratio is not None:
                ratio_zyx = _aspect_ratio_zyx(aspect_ratio)
                aspect_ratio = (ratio_zyx[1], ratio_zyx[0], ratio_zyx[2])
        if aspect_ratio is not None:
            self._append_log("Workflow bbox-ratio preprocessing is preserved in YAML but hidden from the Slicer Image Prep UI.\n")
        crop_from = preprocessing.get(
            "bbox_crop_from",
            preprocessing.get("bbox_crop-from", None),
        )
        if crop_from is not None:
            _format_bbox_crop_from(crop_from)
        smooth_cfg = preprocessing.get("smooth", {})
        if hasattr(self, "largestComponentCheckBox"):
            self.largestComponentCheckBox.checked = _enabled_value(
                preprocessing.get("largest_cc", preprocessing.get("connectivity_filter", False))
            )
        if isinstance(smooth_cfg, dict):
            enabled = _enabled_value(smooth_cfg.get("enabled", True))
            if hasattr(self, "smoothDensityCheckBox"):
                self.smoothDensityCheckBox.checked = enabled and _enabled_value(smooth_cfg.get("density", True))
            if hasattr(self, "smoothLabelsCheckBox"):
                self.smoothLabelsCheckBox.checked = enabled and _enabled_value(
                    smooth_cfg.get("labels", smooth_cfg.get("segmentation", True))
                )
            if hasattr(self, "smoothSigmaSpin") and smooth_cfg.get("sigma_mm") is not None:
                self.smoothSigmaSpin.value = float(smooth_cfg.get("sigma_mm"))
        elif hasattr(self, "smoothDensityCheckBox"):
            enabled = _enabled_value(smooth_cfg)
            self.smoothDensityCheckBox.checked = enabled
            if hasattr(self, "smoothLabelsCheckBox"):
                self.smoothLabelsCheckBox.checked = enabled
        resample = preprocessing.get("resample_isotropic", {})
        target_spacing = None
        resample_enabled = False
        if isinstance(resample, dict):
            target_spacing = resample.get("target_spacing_mm")
            self._resample_spacing_tolerance_mm = float(
                resample.get("spacing_tolerance_mm", self._resample_spacing_tolerance_mm)
            )
            self._resample_spacing_tolerance_relative = float(
                resample.get(
                    "spacing_tolerance_relative",
                    self._resample_spacing_tolerance_relative,
                )
            )
            self._resample_canonicalize_within_tolerance = bool(
                resample.get("canonicalize_within_tolerance", False)
            )
            density_interpolation = _resample_density_interpolation_value(
                resample.get("density_interpolation", resample.get("image_interpolation", "linear"))
            )
            self._resample_density_interpolation = density_interpolation
            if hasattr(self, "resampleDensityInterpolationBox"):
                _set_combo_data(self.resampleDensityInterpolationBox, density_interpolation)
            resample_enabled = _enabled_value(
                resample.get("enabled", target_spacing is not None or bool(resample.get("mode")))
            )
        else:
            resample_enabled = _enabled_value(resample)
        if target_spacing is None and isinstance(preprocessing.get("isotropic_spacing"), (int, float)):
            target_spacing = preprocessing.get("isotropic_spacing")
            resample_enabled = True
        if resample_enabled:
            if hasattr(self, "resampleIsotropicCheckBox"):
                self.resampleIsotropicCheckBox.checked = True
            if hasattr(self, "isotropicSpacingSpin") and target_spacing is not None:
                self._isotropic_spacing_workflow_override = target_spacing is not None
                self._updating_isotropic_spacing = True
                try:
                    self.isotropicSpacingSpin.value = float(target_spacing)
                finally:
                    self._updating_isotropic_spacing = False
        self._update_custom_preprocessing_ui()
        self._update_resample_mode()

    def _apply_profile_model_settings(self, model):
        registration = model.get("registration", {}) if isinstance(model, dict) else {}
        if hasattr(self, "icpRegistrationCheckBox"):
            self.icpRegistrationCheckBox.checked = bool(
                isinstance(registration, dict) and registration.get("enabled", False)
            )
        if hasattr(self, "icpTargetImageBox"):
            _set_combo_data(
                self.icpTargetImageBox,
                _registration_target_image_source(registration),
            )
        if hasattr(self, "icpTargetLabelBox"):
            _set_combo_data(
                self.icpTargetLabelBox,
                _first_int_text(self._current_workflow_registration_values(model, registration)),
            )
        self._update_icp_target_label_ui()

    def _apply_profile_solver_settings(self, solver):
        if not isinstance(solver, dict):
            return
        if hasattr(self, "mpiProcessesSpin") and solver.get("mpi_processes") is not None:
            self.mpiProcessesSpin.value = max(1, int(solver.get("mpi_processes")))
        if hasattr(self, "mpiLauncherEdit") and solver.get("mpi_launcher"):
            self.mpiLauncherEdit.setText(str(solver.get("mpi_launcher")))
        if hasattr(self, "solverToleranceEdit") and solver.get("tolerance") is not None:
            self.solverToleranceEdit.setText(_format_scientific_number(solver.get("tolerance")))

    def _apply_profile_output_settings(self, output, solver=None):
        if not hasattr(self, "outputFieldChecks"):
            return
        output = output if isinstance(output, dict) else {}
        solver = solver if isinstance(solver, dict) else {}
        fields = output.get("fields", solver.get("outputs", ["sed"]))
        if not isinstance(fields, (list, tuple)):
            fields = ["sed"]
        selected = {str(field) for field in fields}
        if hasattr(self, "exportDisplacementsCheckBox"):
            self.exportDisplacementsCheckBox.checked = "displacements" in selected
        for field, checkbox in self.outputFieldChecks.items():
            checkbox.checked = field in selected or (field == "sed" and not selected)
        self._update_output_field_visibility()

    def _sync_displacement_output_checkbox(self, checked=False):
        if not hasattr(self, "outputFieldChecks") or "displacements" not in self.outputFieldChecks:
            return
        self.outputFieldChecks["displacements"].checked = bool(checked)

    def _update_output_field_visibility(self):
        if not hasattr(self, "outputFieldChecks"):
            return
        nonlinear_selected = self._nonlinear_material_selected()
        was_nonlinear_selected = bool(getattr(self, "_nonlinearOutputFieldsVisible", False))
        self._nonlinearOutputFieldsVisible = nonlinear_selected
        if nonlinear_selected and not was_nonlinear_selected:
            self._apply_default_nonlinear_output_fields()
        for field, checkbox in self.outputFieldChecks.items():
            checkbox.visible = nonlinear_selected or field not in self.nonlinearOutputFieldNames
            if field in self.nonlinearOutputFieldNames and not nonlinear_selected:
                checkbox.checked = False

    def _apply_default_nonlinear_output_fields(self):
        for field, checkbox in self.outputFieldChecks.items():
            if field in self.nonlinearOutputFieldNames:
                checkbox.checked = field in {
                    "plastic_strain_magnitude",
                    "mechanical_work_density",
                }
        if "sed" in self.outputFieldChecks:
            self.outputFieldChecks["sed"].checked = True

    def _apply_profile_postprocess_settings(self, postprocess):
        if not hasattr(self, "failurePresetBox"):
            return
        postprocess = postprocess if isinstance(postprocess, dict) else {}
        pistoia = postprocess.get("pistoia", {})
        failure_load = postprocess.get("failure_load", {})
        preset = "Pistoia EES 0.7% / 2%"
        if isinstance(pistoia, dict) and str(pistoia.get("criterion", "pistoia")).strip().lower() == "none":
            preset = "None"
        if failure_load:
            deformation = None
            coefficient = None
            if isinstance(failure_load, dict):
                deformation = failure_load.get("linear_deformation")
                coefficient = failure_load.get("crawford_coefficient")
            preferred = None
            if isinstance(failure_load, dict):
                preferred = str(failure_load.get("preferred", "")).strip().lower()
            if (
                coefficient is not None
                and abs(float(coefficient) - 0.0068) <= 1.0e-9
                and preferred != "linear_reaction_at_deformation"
            ):
                preset = "Kopperdahl/Crawford 0.68%"
        self._set_combo_widget(self.failurePresetBox, preset)

    def _mark_workflow_replay_editor_dirty(self):
        if self._workflow_replay_editor_dirty_suppressed():
            return
        self._lightweightEditorController.mark_editor_dirty()

    def _mark_workflow_replay_loads_dirty(self):
        if self._workflow_replay_editor_dirty_suppressed():
            return
        self._lightweightEditorController.mark_loads_dirty()

    def _workflow_replay_editor_dirty_suppressed(self):
        return int(getattr(self, "_suppressWorkflowReplayEditorDirty", 0) or 0) > 0

    @contextlib.contextmanager
    def _suppress_workflow_replay_editor_dirty(self):
        self._suppressWorkflowReplayEditorDirty = (
            int(getattr(self, "_suppressWorkflowReplayEditorDirty", 0) or 0) + 1
        )
        try:
            yield
        finally:
            self._suppressWorkflowReplayEditorDirty = max(
                int(getattr(self, "_suppressWorkflowReplayEditorDirty", 1) or 1) - 1,
                0,
            )

    def _workflow_replay_editor_dirty_for_export(self):
        if bool(getattr(self, "_workflowReplayResolvedEditorDirty", False)):
            return True
        stored_editor = getattr(self, "_workflowReplayResolvedEditor", None)
        if not isinstance(stored_editor, dict) or not stored_editor.get("planes"):
            return False
        try:
            current_editor = self._editor_state_config()
        except Exception:
            return False
        return not _workflow_editor_pose_equivalent(current_editor, stored_editor)

    def _apply_profile_planes_and_loads(self, editor, *, preserve_existing_sizes=None):
        with self._suppress_workflow_replay_editor_dirty():
            self._apply_profile_planes_and_loads_unsuppressed(
                editor,
                preserve_existing_sizes=preserve_existing_sizes,
            )

    def _apply_profile_planes_and_loads_unsuppressed(self, editor, *, preserve_existing_sizes=None):
        planes = list((editor or {}).get("planes", ()))
        loads = list((editor or {}).get("loads", ()))
        workflow_nodeset_labels = getattr(self, "_appliedWorkflowNodesetLabels", {})
        preserved_sizes = preserve_existing_sizes if isinstance(preserve_existing_sizes, dict) else {}
        loads_by_name = {
            str(load.get("nodeset", load.get("name", ""))): load
            for load in loads
            if isinstance(load, dict)
        }
        for plane_spec in planes:
            if not isinstance(plane_spec, dict):
                continue
            name = str(plane_spec.get("name", f"Plane {self._table_row_count() + 1}"))
            load_spec = loads_by_name.get(name, {})
            mode = _profile_bc_mode(load_spec.get("mode", plane_spec.get("bc_mode", "Displacement")))
            value = load_spec.get("value", plane_spec.get("value", 1.0))
            self._add_contact_plane(
                name=name,
                axis=str(plane_spec.get("axis", "z")),
                normal=str(plane_spec.get("normal", "-")),
                bc="Fixed" if mode == "Fixed" else "Loaded",
                value=str(value),
            )
            row = self._table_row_count() - 1
            self._set_combo_cell(row, 3, str(plane_spec.get("contact", "Material disks")))
            self._set_combo_cell(row, 4, str(plane_spec.get("surface_mode", "project")))
            self._set_combo_cell(row, 5, "Fixed" if mode == "Fixed" else ("Neumann" if mode == "Force" else ("None" if mode == "None" else "Dirichlet")))
            self._set_combo_cell(row, 11, str(plane_spec.get("shape", "anatomy")))
            self._set_spin_cell(row, 12, plane_spec.get("thickness_mm", plane_spec.get("thickness", 3.0)))
            self._set_spin_cell(row, 13, plane_spec.get("intrusion_depth_mm", plane_spec.get("intrusion", 2.0)))
            if "anatomy_constrained" in plane_spec:
                self.contactPlaneRows[row]["anatomy_constrained"] = _enabled_value(
                    plane_spec.get("anatomy_constrained")
                )
            else:
                self.contactPlaneRows[row].pop("anatomy_constrained", None)
            nodeset_label = workflow_nodeset_labels.get(_safe_identifier(name))
            if nodeset_label is not None:
                self.contactPlaneRows[row]["nodeset_label"] = int(nodeset_label)
            else:
                self.contactPlaneRows[row].pop("nodeset_label", None)
            disk_label = plane_spec.get("disk_label")
            if disk_label is None and str(plane_spec.get("contact", "")).strip() in {
                "Material disks",
                "PMMA caps",
            }:
                disk_label = nodeset_label
            if disk_label is not None:
                self.contactPlaneRows[row]["disk_label"] = int(disk_label)
            else:
                self.contactPlaneRows[row].pop("disk_label", None)
            self.contactPlaneRows[row]["reference_nodeset"] = load_spec.get(
                "reference_nodeset",
                plane_spec.get("reference_nodeset"),
            )
            fixed_dofs = _valid_fixed_dofs(
                load_spec.get("fixed_dofs", plane_spec.get("fixed_dofs"))
            )
            if fixed_dofs is not None:
                self.contactPlaneRows[row]["fixed_dofs"] = fixed_dofs
            else:
                self.contactPlaneRows[row].pop("fixed_dofs", None)
            disk = plane_spec.get("disk", plane_spec.get("material", {}))
            target_value = plane_spec.get("disk_target", plane_spec.get("target_label"))
            if target_value is None:
                target_value = _first_int_text(self._current_workflow_disk_projection_values())
            else:
                target_value = _first_int_text(
                    self._workflow_target_values_for_current_mask((target_value,))
                )
            self._set_combo_cell(row, 16, str(target_value or ""))
            self._set_item_text(row, 17, str(float(disk.get("E", plane_spec.get("disk_e", 3000.0)))))
            self._set_item_text(row, 18, str(float(disk.get("nu", plane_spec.get("disk_nu", 0.3)))))
            load_row = self._load_row_for_contact(row)
            if load_row is not None:
                self._set_combo_widget(self.loadTable.cellWidget(load_row, 1), mode)
                self._set_combo_widget(
                    self.loadTable.cellWidget(load_row, 2),
                    str(load_spec.get("direction", plane_spec.get("direction", "Plane normal"))),
                )
                vector = load_spec.get("vector_ras", load_spec.get("vector"))
                if isinstance(vector, (list, tuple)) and len(vector) == 3:
                    for offset, component in enumerate(vector):
                        self.loadTable.setItem(load_row, 3 + offset, qt.QTableWidgetItem(str(component)))
                self.loadTable.setItem(load_row, 6, qt.QTableWidgetItem(str(value)))
                self._update_load_row_units(load_row)
                self._set_combo_widget(
                    self.loadTable.cellWidget(load_row, 7),
                    str(load_spec.get("units", plane_spec.get("units", "mm"))),
                )
                self._set_load_fixed_dofs(load_row, fixed_dofs)
            self._set_load_table_nodeset_label(row)
            preserved_size = preserved_sizes.get(name)
            self._apply_plane_pose_from_profile(
                row,
                plane_spec,
                preserve_existing_size=preserved_size is not None,
            )
            plane = self.contactPlaneRows[row].get("plane")
            if preserved_size is not None and plane is not None and hasattr(plane, "SetSize"):
                plane.SetSize(float(preserved_size[0]), float(preserved_size[1]))

    def _apply_plane_pose_from_profile(self, row, plane_spec, *, preserve_existing_size=False):
        plane = self.contactPlaneRows[row].get("plane")
        if plane is None:
            return
        center = plane_spec.get("center_ras")
        normal = plane_spec.get("normal_ras")
        u_axis = plane_spec.get("u_axis_ras")
        v_axis = plane_spec.get("v_axis_ras")
        size = plane_spec.get("size_mm", plane_spec.get("size"))
        if isinstance(center, (list, tuple)) and len(center) == 3:
            plane.SetCenter([float(v) for v in center])
        if (
            isinstance(normal, (list, tuple))
            and len(normal) == 3
            and isinstance(u_axis, (list, tuple))
            and len(u_axis) == 3
            and isinstance(v_axis, (list, tuple))
            and len(v_axis) == 3
        ):
            _set_plane_axes_world(
                plane,
                [float(v) for v in u_axis],
                [float(v) for v in v_axis],
                [float(v) for v in normal],
            )
        elif isinstance(normal, (list, tuple)) and len(normal) == 3:
            plane.SetNormal([float(v) for v in normal])
        if not preserve_existing_size and (
            isinstance(size, (list, tuple)) and len(size) >= 2 and hasattr(plane, "SetSize")
        ):
            plane.SetSize(float(size[0]), float(size[1]))

    def _current_plane_sizes_by_name(self):
        sizes = {}
        for row, row_data in enumerate(getattr(self, "contactPlaneRows", [])):
            plane = row_data.get("plane") if isinstance(row_data, dict) else None
            if plane is None or not hasattr(plane, "GetSize"):
                continue
            name = ""
            try:
                item = self.planeTable.item(row, 0)
                name = item.text() if item is not None else ""
            except Exception:
                name = ""
            if not name:
                continue
            size = [0.0, 0.0]
            try:
                plane.GetSize(size)
            except TypeError:
                size = list(plane.GetSize())
            sizes[str(name)] = (float(size[0]), float(size[1]))
        return sizes

    def _set_combo_cell(self, row, column, text):
        if int(column) == 4:
            text = _surface_mode_ui_text(text)
        if int(column) == 16:
            _set_combo_data(self.planeTable.cellWidget(row, column), text)
            return
        self._set_combo_widget(self.planeTable.cellWidget(row, column), text)

    def _set_combo_widget(self, widget, text):
        if widget is None:
            return
        text = str(text)
        if widget.findText(text) < 0:
            widget.addItem(text)
        widget.setCurrentText(text)

    def _set_spin_cell(self, row, column, value):
        widget = self.planeTable.cellWidget(row, column)
        if widget is not None:
            widget.value = float(value)

    def _set_item_text(self, row, column, text):
        self.planeTable.setItem(row, column, qt.QTableWidgetItem(str(text)))

    def _contact_model(self):
        text = self._widget_text(self.contactModelBox).strip().lower()
        return "bone_surface" if "bone" in text else "cap_disks"

    def _update_contact_model_mode(self):
        self.previewDisksButton.text = "Create Regions"

    def _update_resample_mode(self):
        if not hasattr(self, "isotropicSpacingSpin") or not hasattr(self, "resampleIsotropicCheckBox"):
            return
        enabled = bool(self.resampleIsotropicCheckBox.checked)
        self.isotropicSpacingSpin.enabled = enabled
        if hasattr(self, "resampleDensityInterpolationBox"):
            self.resampleDensityInterpolationBox.enabled = enabled

    def _apply_profile_custom_preprocessing(self, custom_preprocessing):
        if not hasattr(self, "customPreprocessingBox"):
            return
        if _custom_preprocessing_options(custom_preprocessing):
            self._set_custom_preprocessing_options(custom_preprocessing)
            return
        script = _custom_preprocessing_script_value(custom_preprocessing)
        label = _custom_preprocessing_label_value(custom_preprocessing)
        function_name = _custom_preprocessing_function_value(custom_preprocessing)
        if script:
            self._set_custom_preprocessing_script(script, label=label, function_name=function_name)
        else:
            self._set_custom_preprocessing_script(None)

    def _set_custom_preprocessing_script(self, script, *, label=None, function_name=None):
        if not hasattr(self, "customPreprocessingBox"):
            return
        current = "" if not script else str(Path(str(script)).expanduser())
        display_name = str(label or Path(current).stem).strip() if current else ""
        function_name = str(function_name or display_name or "custom_preprocessing").strip()
        self.customPreprocessingBox.blockSignals(True)
        try:
            self.customPreprocessingBox.clear()
            self.customPreprocessingBox.addItem("None", "")
            if current:
                self.customPreprocessingBox.addItem(
                    f"Workflow: {display_name}",
                    current,
                )
            self.customPreprocessingBox.addItem(
                "Create new...",
                CUSTOM_PREPROCESSING_CREATE_TOKEN,
            )
            _set_combo_data(self.customPreprocessingBox, current)
        finally:
            self.customPreprocessingBox.blockSignals(False)
        self._customPreprocessingScriptPath = current
        self._customPreprocessingFunctionName = function_name if current else ""
        self._customPreprocessingLabel = display_name if current else ""
        self._customPreprocessingOptions = []
        self._customPreprocessingSelected = ""
        self._update_custom_preprocessing_ui()

    def _set_custom_preprocessing_options(self, custom_preprocessing):
        options = _custom_preprocessing_options(custom_preprocessing)
        selected = str(
            custom_preprocessing.get("selected", custom_preprocessing.get("default", ""))
            if isinstance(custom_preprocessing, dict)
            else ""
        ).strip()
        if not selected and options:
            selected = _custom_preprocessing_option_id(options[0])
        self.customPreprocessingBox.blockSignals(True)
        try:
            self.customPreprocessingBox.clear()
            self.customPreprocessingBox.addItem("None", "")
            for option in options:
                option_id = _custom_preprocessing_option_id(option)
                label = _custom_preprocessing_label_value(option) or option_id
                self.customPreprocessingBox.addItem(f"Workflow: {label}", option_id)
            self.customPreprocessingBox.addItem(
                "Create new...",
                CUSTOM_PREPROCESSING_CREATE_TOKEN,
            )
            _set_combo_data(self.customPreprocessingBox, selected)
        finally:
            self.customPreprocessingBox.blockSignals(False)
        self._customPreprocessingOptions = [copy.deepcopy(option) for option in options]
        self._customPreprocessingSelected = selected
        selected_option = self._custom_preprocessing_selected_option()
        self._customPreprocessingScriptPath = (
            str(Path(_custom_preprocessing_script_value(selected_option)).expanduser())
            if selected_option and _custom_preprocessing_script_value(selected_option)
            else ""
        )
        self._customPreprocessingFunctionName = (
            _custom_preprocessing_function_value(selected_option) or ""
        )
        self._customPreprocessingLabel = (
            _custom_preprocessing_label_value(selected_option) or ""
        )
        self._update_custom_preprocessing_ui()

    def _on_custom_preprocessing_changed(self, *_args):
        value = str(_combo_data(self.customPreprocessingBox) or "")
        if value == CUSTOM_PREPROCESSING_CREATE_TOKEN:
            spec = self._create_custom_preprocessing_script()
            if spec is None:
                self._set_custom_preprocessing_script(None)
                return
            self._set_custom_preprocessing_script(
                spec["script"],
                label=spec.get("name"),
                function_name=spec.get("function"),
            )
            self.edit_custom_preprocessing()
            return
        options = getattr(self, "_customPreprocessingOptions", [])
        if options:
            self._customPreprocessingSelected = value
            selected_option = self._custom_preprocessing_selected_option()
            self._customPreprocessingScriptPath = (
                str(Path(_custom_preprocessing_script_value(selected_option)).expanduser())
                if selected_option and _custom_preprocessing_script_value(selected_option)
                else ""
            )
            self._customPreprocessingFunctionName = (
                _custom_preprocessing_function_value(selected_option) or ""
            )
            self._customPreprocessingLabel = (
                _custom_preprocessing_label_value(selected_option) or ""
            )
            self._update_custom_preprocessing_ui()
            return
        self._customPreprocessingScriptPath = value
        self._update_custom_preprocessing_ui()

    def _update_custom_preprocessing_ui(self):
        if not hasattr(self, "editCustomPreprocessingButton"):
            return
        self.editCustomPreprocessingButton.enabled = bool(self._custom_preprocessing_script_path())

    def _create_custom_preprocessing_script(self):
        default_dir = USER_WORKFLOW_ROOT / "custom_preprocessing"
        default_dir.mkdir(parents=True, exist_ok=True)
        selected = qt.QInputDialog.getText(
            slicer.util.mainWindow(),
            "Create Custom Preprocessing",
            "Name",
            qt.QLineEdit.Normal,
            "femur_proximal_crop",
        )
        if isinstance(selected, tuple):
            raw_name, accepted = selected[0], bool(selected[1])
        else:
            raw_name, accepted = selected, True
        if not accepted:
            return None
        function_name = _custom_preprocessing_identifier(raw_name)
        path = default_dir / f"{function_name}.py"
        if path.exists():
            overwrite = qt.QMessageBox.question(
                slicer.util.mainWindow(),
                "Overwrite Custom Preprocessing",
                f"{path} already exists. Overwrite it?",
                qt.QMessageBox.Yes | qt.QMessageBox.No,
                qt.QMessageBox.No,
            )
            if overwrite != qt.QMessageBox.Yes:
                return None
        path.write_text(_custom_preprocessing_scaffold(function_name), encoding="utf-8")
        return {"script": str(path), "function": function_name, "name": function_name}

    def edit_custom_preprocessing(self):
        path = self._custom_preprocessing_script_path()
        if path is None:
            spec = self._create_custom_preprocessing_script()
            if spec is None:
                return
            self._set_custom_preprocessing_script(
                spec["script"],
                label=spec.get("name"),
                function_name=spec.get("function"),
            )
            path = Path(spec["script"])
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("Custom Preprocessing")
        layout = qt.QVBoxLayout(dialog)
        editor = qt.QPlainTextEdit()
        editor.setPlainText(
            path.read_text(encoding="utf-8")
            if path.exists()
            else _custom_preprocessing_scaffold(self._custom_preprocessing_function_name())
        )
        editor.minimumWidth = 860
        editor.minimumHeight = 620
        layout.addWidget(editor)
        buttons = qt.QDialogButtonBox(qt.QDialogButtonBox.Save | qt.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != qt.QDialog.Accepted:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(editor.toPlainText(), encoding="utf-8")
        self._set_custom_preprocessing_script(
            path,
            label=getattr(self, "_customPreprocessingLabel", ""),
            function_name=self._custom_preprocessing_function_name(),
        )
        self._append_log(f"Saved custom preprocessing script: {path}\n")

    def _custom_preprocessing_script_path(self):
        value = str(getattr(self, "_customPreprocessingScriptPath", "") or "").strip()
        if not value and hasattr(self, "customPreprocessingBox"):
            value = str(_combo_data(self.customPreprocessingBox) or "").strip()
        if not value or value == CUSTOM_PREPROCESSING_CREATE_TOKEN:
            return None
        return Path(value).expanduser()

    def _custom_preprocessing_function_name(self):
        value = str(getattr(self, "_customPreprocessingFunctionName", "") or "").strip()
        if value:
            return value
        path = self._custom_preprocessing_script_path()
        if path is not None:
            return _custom_preprocessing_identifier(path.stem)
        return "custom_preprocessing"

    def _custom_preprocessing_selected_option(self):
        selected = str(getattr(self, "_customPreprocessingSelected", "") or "").strip()
        for option in getattr(self, "_customPreprocessingOptions", []):
            if _custom_preprocessing_option_id(option) == selected:
                return option
        return None

    def _custom_preprocessing_config(self):
        options = getattr(self, "_customPreprocessingOptions", [])
        selected = str(getattr(self, "_customPreprocessingSelected", "") or "").strip()
        if options and selected:
            return {
                "selected": selected,
                "options": copy.deepcopy(options),
            }
        path = self._custom_preprocessing_script_path()
        if path is None:
            return {}
        function_name = self._custom_preprocessing_function_name()
        return {
            "name": str(getattr(self, "_customPreprocessingLabel", "") or function_name),
            "script": str(path),
            "function": function_name,
            "stage": "post_registration",
        }

    def _preprocessing_config(self, *, force=False):
        if getattr(self, "_preprocessingAppliedToInputs", False) and not force:
            return {}
        preprocessing = {}
        if bool(getattr(self.largestComponentCheckBox, "checked", False)):
            preprocessing["largest_cc"] = True
        if bool(getattr(self.smoothDensityCheckBox, "checked", False)) or bool(
            getattr(self.smoothLabelsCheckBox, "checked", False)
        ):
            preprocessing["smooth"] = {
                "enabled": True,
                "sigma_mm": float(self.smoothSigmaSpin.value),
                "density": bool(self.smoothDensityCheckBox.checked),
                "labels": bool(self.smoothLabelsCheckBox.checked),
            }
        crop_padding = float(self.cropPaddingSpin.value)
        if bool(getattr(self.cropToMaskCheckBox, "checked", False)):
            preprocessing["crop_to_bb"] = {
                "enabled": True,
                "margin_mm": crop_padding,
            }
        if bool(self.resampleIsotropicCheckBox.checked):
            preprocessing["resample_isotropic"] = {
                "enabled": True,
                "mode": "auto",
                "target_spacing_mm": float(self.isotropicSpacingSpin.value),
                "spacing_tolerance_mm": float(self._resample_spacing_tolerance_mm),
                "spacing_tolerance_relative": float(self._resample_spacing_tolerance_relative),
                "canonicalize_within_tolerance": bool(self._resample_canonicalize_within_tolerance),
                "density_interpolation": _resample_density_interpolation_value(
                    _combo_data(getattr(self, "resampleDensityInterpolationBox", None))
                    or self._resample_density_interpolation
                ),
            }
        return preprocessing

    def _on_isotropic_spacing_edited(self, *_args):
        if not self._updating_isotropic_spacing:
            self._isotropic_spacing_user_override = True

    def _on_preprocess_spacing_default_edited(self, *_args):
        if not self._updating_preprocess_spacing_defaults:
            self._preprocess_spacing_user_override = True

    def _update_default_preprocess_spacing(self):
        if self._preprocess_spacing_user_override:
            return
        volume = self._volume()
        if volume is None:
            return
        spacing = [abs(float(value)) for value in volume.GetSpacing() if abs(float(value)) > 0]
        if not spacing:
            return
        voxel_mm = min(spacing)
        sigma_mm = _nice_spacing_default_mm(voxel_mm)
        padding_mm = _nice_spacing_default_mm(5.0 * voxel_mm)
        self._updating_preprocess_spacing_defaults = True
        try:
            self.smoothSigmaSpin.value = float(sigma_mm)
            self.cropPaddingSpin.value = float(padding_mm)
        finally:
            self._updating_preprocess_spacing_defaults = False

    def _update_default_isotropic_spacing(self):
        if self._isotropic_spacing_user_override or self._isotropic_spacing_workflow_override:
            return
        volume = self._volume()
        if volume is None:
            return
        spacing = [abs(float(value)) for value in volume.GetSpacing() if abs(float(value)) > 0]
        if not spacing:
            return
        target = min(spacing)
        self._updating_isotropic_spacing = True
        try:
            self.isotropicSpacingSpin.value = float(target)
        finally:
            self._updating_isotropic_spacing = False

    def _material_mode_key(self):
        text = self._widget_text(self.materialModeBox).strip().lower()
        if "nonlinear" in text:
            return "nonlinear_density"
        if "density" in text:
            return "linear_density"
        return "linear_labels"

    def _nonlinear_material_selected(self):
        return self._material_mode_key() == "nonlinear_density"

    def _manual_nonlinear_material_selected(self):
        return self._nonlinear_material_selected() and self._widget_text(
            getattr(self, "nonlinearPresetBox", None), ""
        ).strip().lower() == "manual"

    def _update_material_mode(self, *_args):
        mode = self._material_mode_key()
        density_mode = mode in {"linear_density", "nonlinear_density"}
        nonlinear_mode = mode == "nonlinear_density"
        formula = self._widget_text(self.densityEquationBox).strip().lower()
        power_mode = formula == "power"
        polynomial_mode = formula == "polynomial"
        self.materialTable.visible = not density_mode
        if self.materialTableLabel is not None:
            self.materialTableLabel.visible = not density_mode
        for button in self.materialButtons:
            button.visible = not density_mode
        self.materialPresetBox.visible = not nonlinear_mode
        if getattr(self, "materialPresetLabel", None) is not None:
            self.materialPresetLabel.visible = not nonlinear_mode
        self.nonlinearPresetBox.visible = nonlinear_mode
        if self.nonlinearPresetLabel is not None:
            self.nonlinearPresetLabel.visible = nonlinear_mode
        self.materialNuSpin.visible = density_mode
        if self.materialNuLabel is not None:
            self.materialNuLabel.visible = density_mode
        visible_rows = set()
        if density_mode and not nonlinear_mode:
            visible_rows.update({"equation", "formula", "floor", "test", "result"})
            visible_rows.add("bin_material")
            if bool(getattr(self.binMaterialCheckBox, "checked", False)):
                visible_rows.add("number_bins")
            if formula == "power":
                visible_rows.update({"coefficient", "exponent", "reference"})
            elif formula == "polynomial":
                visible_rows.update({"quad", "slope", "intercept"})
            elif formula in {"linear", "mulder2007"}:
                visible_rows.update({"slope", "intercept"})
        elif nonlinear_mode:
            visible_rows.add("bin_material")
            if bool(getattr(self.binMaterialCheckBox, "checked", False)):
                visible_rows.add("number_bins")
        for key, widgets in self._density_formula_rows.items():
            self._set_widgets_visible(widgets, density_mode and key in visible_rows)
        for widgets in getattr(self, "nonlinearManualRows", {}).values():
            self._set_widgets_visible(widgets, nonlinear_mode)
        density = float(self.densityTestSpin.value)
        floor = float(self.densityFloorSpin.value)
        if formula == "mulder2007":
            a1 = float(self.densitySlopeSpin.value)
            a0 = float(self.densityInterceptSpin.value)
            predicted = max(a1 * density + a0, floor)
            text = f"Mulder 2007: E = max({floor:g}, {a1:g} * density + {a0:g})"
        elif formula == "kopperdahl":
            coeff = 2980.0
            exponent = 1.05
            ref = 1000.0
            predicted = max(coeff * ((max(density, 0.0) / ref) ** exponent if ref else 0.0), floor)
            text = f"Kopperdahl: E = max({floor:g}, 2980 * (density / 1000)^1.05)"
        elif formula == "power":
            coeff = float(self.densityCoeffSpin.value)
            exponent = float(self.densityExponentSpin.value)
            ref = float(self.densityReferenceSpin.value)
            predicted = max(coeff * ((max(density, 0.0) / ref) ** exponent if ref else 0.0), floor)
            text = f"E = max({floor:g}, {coeff:g} * (density / {ref:g})^{exponent:g})"
        elif formula == "polynomial":
            a2 = float(self.densityQuadSpin.value)
            a1 = float(self.densitySlopeSpin.value)
            a0 = float(self.densityInterceptSpin.value)
            predicted = max(a2 * density * density + a1 * density + a0, floor)
            text = f"E = max({floor:g}, {a2:g} * density^2 + {a1:g} * density + {a0:g})"
        else:
            a1 = float(self.densitySlopeSpin.value)
            a0 = float(self.densityInterceptSpin.value)
            predicted = max(a1 * density + a0, floor)
            text = f"E = max({floor:g}, {a1:g} * density + {a0:g})"
        if bool(getattr(self.binMaterialCheckBox, "checked", False)):
            text = f"{text}; density first binned into {int(self.numberBinsSpin.value)} global active bins"
        self.densityFormulaLabel.text = text
        self.densityResultLabel.text = f"{predicted:g} MPa"
        self._update_output_field_visibility()

    def _on_density_equation_changed(self, *_args):
        formula = self._widget_text(self.densityEquationBox).strip().lower()
        if formula == "mulder2007":
            self.densitySlopeSpin.value = 25.0
            self.densityInterceptSpin.value = -5830.0
            self.densityFloorSpin.value = 2.0
        elif abs(float(self.densityFloorSpin.value) - 2.0) < 1e-12:
            self.densityFloorSpin.value = 0.0
        self._update_material_mode()

    def _on_nonlinear_preset_changed(self, *_args):
        preset = self._widget_text(self.nonlinearPresetBox, "Spine nonlinear").strip().lower()
        if preset in {"spine nonlinear", "hip nonlinear"}:
            self._apply_nonlinear_preset_to_widgets(preset)
        self._update_material_mode()

    def _apply_nonlinear_preset_to_widgets(self, preset):
        preset = str(preset).strip().lower()
        if preset == "hip nonlinear":
            self.nonlinearElasticCoeffSpin.value = 8768.0
            self.nonlinearElasticExponentSpin.value = 1.49
            self.nonlinearElasticReferenceSpin.value = 1.0
            self.nonlinearCompressionCoeffSpin.value = 0.0085 * 8768.0
            self.nonlinearCompressionExponentSpin.value = 1.49
            self.nonlinearCompressionReferenceSpin.value = 1.0
            self.nonlinearTensionCoeffSpin.value = 0.0061 * 8768.0
            self.nonlinearTensionExponentSpin.value = 1.49
            self.nonlinearTensionReferenceSpin.value = 1.0
            return
        self.nonlinearElasticCoeffSpin.value = 3814.4
        self.nonlinearElasticExponentSpin.value = 1.05
        self.nonlinearElasticReferenceSpin.value = 1000.0
        self.nonlinearCompressionCoeffSpin.value = 57.4464
        self.nonlinearCompressionExponentSpin.value = 1.39
        self.nonlinearCompressionReferenceSpin.value = 1000.0
        self.nonlinearTensionCoeffSpin.value = 57.4464
        self.nonlinearTensionExponentSpin.value = 1.39
        self.nonlinearTensionReferenceSpin.value = 1000.0

    def _apply_material_preset(self, *_args):
        preset = self._widget_text(self.materialPresetBox, "Manual").strip().lower()
        if preset == "manual":
            return
        if preset == "xtremecti labels":
            self._set_label_material_preset(
                [
                    (100, "trabecular_bone", 6829.0, 0.3),
                    (127, "cortical_bone", 6829.0, 0.3),
                ]
            )
        elif preset == "xtremectii labels":
            self._set_label_material_preset(
                [
                    (100, "trabecular_bone", 8748.0, 0.3),
                    (127, "cortical_bone", 8748.0, 0.3),
                ]
            )
        elif preset == "mulder 2007 framework density":
            self._set_density_preset("mulder2007", slope=25.0, intercept=-5830.0, floor=2.0)
        elif preset == "kopperdahl density":
            self._set_density_preset("kopperdahl", coefficient=2980.0, exponent=1.05, reference=1000.0, floor=0.0)
        elif preset == "michalski density power law":
            self._set_density_preset("power", coefficient=10500.0, exponent=2.29, reference=1000.0, floor=0.0)
        elif preset == "morgan trabecular density":
            self._set_density_preset("power", coefficient=4730.0, exponent=1.56, reference=1000.0, floor=0.0)
        elif preset == "crawford voxel density":
            self._set_density_preset("linear", slope=3230.0, intercept=-34.7, floor=0.0)
        elif preset == "bayraktar trabecular constant":
            self._set_density_preset("linear", slope=0.0, intercept=18000.0, floor=0.0)
        self._update_material_mode()

    def _set_label_material_preset(self, rows):
        self.materialModeBox.setCurrentText("Linear label-based")
        self.materialTable.setRowCount(0)
        for label, name, e, nu in rows:
            self._add_material_row(label=label, name=name, e=e, nu=nu)

    def _set_density_preset(
        self,
        equation,
        *,
        coefficient=None,
        exponent=None,
        reference=None,
        slope=None,
        intercept=None,
        floor=None,
    ):
        if self._material_mode_key() != "nonlinear_density":
            self.materialModeBox.setCurrentText("Linear density formula")
        self.densityEquationBox.setCurrentText(str(equation))
        if floor is None and str(equation).strip().lower() in {
            "mulder",
            "mulder2007",
            "mulder_2007",
            "framework_mulder",
            "framework_mulder2007",
        }:
            floor = 2.0
        if coefficient is not None:
            self.densityCoeffSpin.value = float(coefficient)
        if exponent is not None:
            self.densityExponentSpin.value = float(exponent)
        if reference is not None:
            self.densityReferenceSpin.value = float(reference)
        if slope is not None:
            self.densitySlopeSpin.value = float(slope)
        if intercept is not None:
            self.densityInterceptSpin.value = float(intercept)
        if floor is not None:
            self.densityFloorSpin.value = float(floor)

    def _apply_manual_nonlinear_law_to_widgets(self, prefix, spec):
        if not isinstance(spec, dict):
            return
        attr_prefix = {
            "elastic": "nonlinearElastic",
            "compression": "nonlinearCompression",
            "tension": "nonlinearTension",
        }.get(str(prefix), "")
        if not attr_prefix:
            return
        coeff = spec.get("coefficient", spec.get("e_max"))
        exponent = spec.get("exponent")
        reference = spec.get("reference_density", spec.get("reference", spec.get("rho_max")))
        if coeff is not None:
            getattr(self, f"{attr_prefix}CoeffSpin").value = float(coeff)
        if exponent is not None:
            getattr(self, f"{attr_prefix}ExponentSpin").value = float(exponent)
        if reference is not None:
            getattr(self, f"{attr_prefix}ReferenceSpin").value = float(reference)

    def _material_law_spinbox(self, value, *, minimum=0.0, maximum=1e7):
        spin = qt.QDoubleSpinBox()
        spin.minimum = float(minimum)
        spin.maximum = float(maximum)
        spin.decimals = 6
        spin.value = float(value)
        return spin

    def _set_widgets_visible(self, widgets, visible):
        for widget in widgets:
            if widget is not None:
                widget.visible = bool(visible)

    def _material_override(self):
        mode = self._material_mode_key()
        if mode in {"linear_density", "nonlinear_density"}:
            equation = self._widget_text(self.densityEquationBox, "linear").strip().lower()
            if mode == "nonlinear_density":
                e_config = {"equation": "linear", "slope": 1.0, "intercept": 0.0}
            elif equation == "kopperdahl":
                e_config = {
                    "equation": "power",
                    "coefficient": 2980.0,
                    "exponent": 1.05,
                    "reference_density": 1000.0,
                }
            elif equation == "mulder2007":
                e_config = {
                    "equation": "mulder2007",
                    "slope": float(self.densitySlopeSpin.value),
                    "intercept": float(self.densityInterceptSpin.value),
                }
            elif equation == "power":
                e_config = {
                    "equation": "power",
                    "coefficient": float(self.densityCoeffSpin.value),
                    "exponent": float(self.densityExponentSpin.value),
                    "reference_density": float(self.densityReferenceSpin.value),
                }
            elif equation == "polynomial":
                e_config = {
                    "equation": "polynomial",
                    "coefficients": [
                        float(self.densityInterceptSpin.value),
                        float(self.densitySlopeSpin.value),
                        float(self.densityQuadSpin.value),
                    ],
                }
            else:
                e_config = {
                    "equation": "linear",
                    "slope": float(self.densitySlopeSpin.value),
                    "intercept": float(self.densityInterceptSpin.value),
                }
            e_config["floor_e_mpa"] = float(self.densityFloorSpin.value)
            density_config = {
                "E": e_config,
                "nu": float(self.materialNuSpin.value),
                "active_threshold": 0,
            }
            if mode == "nonlinear_density":
                density_config["basis"] = (
                    "rho_app"
                    if self._widget_text(self.nonlinearPresetBox).strip().lower() == "hip nonlinear"
                    else "rho_qct_mgcc"
                )
            if bool(getattr(self.binMaterialCheckBox, "checked", False)):
                density_config["bin_material"] = True
                density_config["number_bins"] = int(self.numberBinsSpin.value)
            override = {
                "image_type": "density",
                "materials": {
                    "units": "MPa",
                    "density": density_config,
                },
            }
            if mode == "nonlinear_density":
                override["materials"]["nonlinear"] = self._nonlinear_material_config()
                override["solver"] = {"nonlinear": self._nonlinear_solver_config()}
            return override
        return {
            "image_type": "material_labels",
            "materials": {
                "units": "MPa",
                "labels": self._parse_material_labels(),
            },
        }

    def _nonlinear_material_config(self):
        nonlinear = {
            "preset": "manual",
            "elastic": self._manual_nonlinear_law_config(
                self.nonlinearElasticCoeffSpin,
                self.nonlinearElasticExponentSpin,
                self.nonlinearElasticReferenceSpin,
            ),
            "compressive_yield": self._manual_nonlinear_law_config(
                self.nonlinearCompressionCoeffSpin,
                self.nonlinearCompressionExponentSpin,
                self.nonlinearCompressionReferenceSpin,
            ),
            "tensile_yield": self._manual_nonlinear_law_config(
                self.nonlinearTensionCoeffSpin,
                self.nonlinearTensionExponentSpin,
                self.nonlinearTensionReferenceSpin,
            ),
        }
        if bool(getattr(self.binMaterialCheckBox, "checked", False)):
            nonlinear["bin_material"] = True
            nonlinear["number_bins"] = int(self.numberBinsSpin.value)
        return nonlinear

    def _manual_nonlinear_law_config(self, coefficient_spin, exponent_spin, reference_spin):
        return {
            "equation": "power",
            "coefficient": float(coefficient_spin.value),
            "exponent": float(exponent_spin.value),
            "reference_density": float(reference_spin.value),
        }

    def _nonlinear_solver_config(self):
        return {
            "convergence_tolerance": 1.0e-4,
            "maximum_plastic_iterations": 150,
            "plastic_convergence_window": 2,
        }

    def _parse_material_labels(self):
        labels = {}
        for row in range(self._material_row_count()):
            label_text = self._material_item_text(row, 0, "")
            if not label_text.strip():
                continue
            label = int(label_text)
            name = self._material_item_text(row, 1, f"label_{label}")
            e = float(self._material_item_text(row, 2, "0"))
            nu = float(self._material_item_text(row, 3, str(float(self.materialNuSpin.value))))
            if e <= 0:
                raise ValueError(f"Material label {label} needs E > 0")
            values = {"name": name, "E": e, "nu": nu}
            labels[label] = values
        if not labels:
            raise ValueError("At least one material label is required")
        return labels

    def _active_material_labels_for_preview(self):
        if self._material_mode_key() in {"linear_density", "nonlinear_density"}:
            return None
        labels = []
        for row in range(self._material_row_count()):
            label_text = self._material_item_text(row, 0, "")
            if not label_text.strip():
                continue
            try:
                label = int(label_text)
                e = float(self._material_item_text(row, 2, "0"))
            except Exception:
                continue
            if label != 0 and e > 0:
                labels.append(label)
        return tuple(labels) if labels else None

    def _validated_active_material_labels_for_preview(self):
        labels = self._active_material_labels_for_preview()
        if (
            hasattr(self, "materialModeBox")
            and self._material_mode_key() in {"linear_density", "nonlinear_density"}
        ):
            return None
        if labels is None:
            raise ValueError(
                "No active material labels are selected for this label profile. "
                "Apply the workflow profile or seed the material labels before creating contact regions."
            )
        if labels is None or self.maskSelector.currentNode() is not None:
            return labels
        volume = self._volume()
        if volume is None:
            return labels
        try:
            array = np.asarray(slicer.util.arrayFromVolume(volume))
        except Exception:
            return labels
        unique = np.unique(array)
        nonzero = [int(value) for value in unique if int(value) != 0]
        if not nonzero:
            return labels
        label_set = {int(value) for value in labels}
        outside = [value for value in nonzero if value not in label_set]
        if not outside:
            return labels
        if len(nonzero) <= max(len(label_set) + 4, 12):
            return labels
        matched = int(np.count_nonzero(np.isin(array, tuple(sorted(label_set)))))
        total_nonzero = int(np.count_nonzero(array))
        matched_fraction = matched / max(total_nonzero, 1)
        if matched_fraction >= 0.95:
            return labels
        label_text = ", ".join(str(value) for value in sorted(label_set))
        raise ValueError(
            "The selected image does not look like a material-label segmentation for "
            f"the current material labels ({label_text}). It has {len(nonzero)} non-zero "
            f"values and only {matched_fraction:.1%} of non-zero voxels match the material table. "
            "If this is a density/grayscale AIM, use a density-formula material profile and a mask "
            "or convert it to material labels before using XtremeCT label profiles. This prevents "
            "ordinary intensity values such as 100 or 127 from being used as boundary-condition nodes."
        )

    def _material_row_count(self):
        try:
            return int(self._qt_value(self.materialTable.rowCount))
        except Exception:
            return 0

    def _material_item_text(self, row, column, default=""):
        item = self.materialTable.item(row, column)
        if item is None:
            return default
        try:
            return str(self._qt_value(item.text))
        except Exception:
            return default

    def _add_material_row(self, checked=False, *, label=None, name=None, e=8748.0, nu=0.3):
        row = self._material_row_count()
        self.materialTable.insertRow(row)
        label = int(label) if label is not None else self._next_material_label()
        values = [str(label), name or f"label_{label}", f"{float(e):g}", f"{float(nu):g}"]
        for column, value in enumerate(values):
            self.materialTable.setItem(row, column, qt.QTableWidgetItem(value))

    def _delete_selected_material_row(self):
        rows = self.materialTable.selectionModel().selectedRows()
        if rows:
            self.materialTable.removeRow(int(rows[0].row()))

    def _next_material_label(self):
        used = {
            int(self._material_item_text(row, 0, "0"))
            for row in range(self._material_row_count())
            if self._material_item_text(row, 0, "").strip()
        }
        label = 1
        while label in used:
            label += 1
        return label

    def _seed_default_material_table(self):
        return

    def _seed_material_table(self):
        labels = self._unique_input_labels()
        if not labels:
            self._append_log("No non-zero labels found to seed material table.\n")
            return
        existing = {
            int(self._material_item_text(row, 0, "0"))
            for row in range(self._material_row_count())
            if self._material_item_text(row, 0, "").strip()
        }
        for label in labels:
            if label not in existing:
                self._add_material_row(label=label, name=f"label_{label}", e=8748.0, nu=0.3)
        self._append_log(f"Seeded material labels: {', '.join(str(label) for label in labels)}.\n")

    def _unique_input_labels(self):
        node = self.imageSelector.currentNode() or self.maskSelector.currentNode()
        if node is None:
            return []
        try:
            array = np.asarray(_array_from_mask_like(node, self._volume()))
        except Exception:
            return []
        values = [int(value) for value in np.unique(array) if int(value) != 0]
        return values[:200]

    def _bounds_node(self):
        if hasattr(self, "maskSelector") and self.maskSelector is not None:
            return self.maskSelector.currentNode() or self._volume()
        return self._volume()

    def _add_default_contact_planes(self):
        if self.contactPlaneRows:
            return
        self._add_contact_plane(name="Top", axis="z", normal="-", bc="Loaded", value="1.0")
        self._add_contact_plane(name="Bottom", axis="z", normal="+", bc="Fixed", value="0.0")

    def _ensure_default_contact_planes(self):
        if self._volume() is not None and not self.contactPlaneRows:
            self._add_default_contact_planes()

    def _on_input_node_changed(self, *_args):
        if self._input_node_updates_are_suppressed():
            return
        image_node = self.imageSelector.currentNode()
        if image_node is not None and not _is_generated_parosol_node(image_node):
            paths = self._current_input_storage_paths()
            if paths.get("image"):
                self._workflowReplaySourceInputs = paths
        self._workflowReplayContractEditor = None
        self._workflowReplayResolvedEditor = None
        self._workflowReplayResolvedEditorDirty = False
        state = self._stage_state()
        state.anatomy_dirty = True
        state.boundary_dirty = True
        state.loads_dirty = True
        state.export_dirty = True
        self._update_default_preprocess_spacing()
        self._update_default_isotropic_spacing()
        self._refresh_target_label_options()
        self.logic.remove_named_node("ParOSol_input_mask_3D")
        self.logic.remove_named_node("ParOSol_input_volume_3D")
        self._update_profile_mode()
        self._update_mask_requirement_ui()
        self._update_input_readiness_status()

    def _on_mask_node_changed(self, *_args):
        if self._input_node_updates_are_suppressed():
            return
        image_node = self.imageSelector.currentNode()
        if image_node is not None and not _is_generated_parosol_node(image_node):
            paths = self._current_input_storage_paths()
            if paths.get("image"):
                self._workflowReplaySourceInputs = paths
        self._workflowReplayContractEditor = None
        self._workflowReplayResolvedEditor = None
        self._workflowReplayResolvedEditorDirty = False
        self._refresh_mask_segment_options()
        self._refresh_target_label_options()
        self._on_input_node_changed()

    def _show_mask_3d_preserving_mask_selection(self, mask_node, image_node, *, active_values=None):
        selected_mask = self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None
        signal_states = self._begin_input_node_update_suppression()
        try:
            return self.logic.show_mask_3d(mask_node, image_node, active_values=active_values)
        finally:
            try:
                current = self.maskSelector.currentNode()
                current_name = str(current.GetName()) if current is not None else ""
                if selected_mask is not None and current is not selected_mask and current_name in {
                    "ParOSol_input_mask_3D",
                    "ParOSol_input_volume_3D",
                }:
                    self.maskSelector.setCurrentNode(selected_mask)
            except Exception:
                pass
            self._end_input_node_update_suppression(signal_states)

    def _input_node_updates_are_suppressed(self):
        if int(getattr(self, "_suppressInputNodeChanged", 0)) > 0:
            return True
        try:
            return bool(slicer.mrmlScene.IsBatchProcessing())
        except Exception:
            return False

    def _input_node_selectors(self):
        return [
            selector
            for selector in (
                getattr(self, "imageSelector", None),
                getattr(self, "maskSelector", None),
                getattr(self, "diskLabelSelector", None),
                getattr(self, "bcLabelSelector", None),
            )
            if selector is not None and hasattr(selector, "blockSignals")
        ]

    def _begin_input_node_update_suppression(self):
        self._suppressInputNodeChanged = int(getattr(self, "_suppressInputNodeChanged", 0)) + 1
        states = []
        for selector in self._input_node_selectors():
            try:
                states.append((selector, selector.blockSignals(True)))
            except Exception:
                pass
        return states

    def _end_input_node_update_suppression(self, states):
        for selector, previous in reversed(states or []):
            try:
                selector.blockSignals(previous)
            except Exception:
                pass
        self._suppressInputNodeChanged = max(
            0,
            int(getattr(self, "_suppressInputNodeChanged", 0)) - 1,
        )

    def _refresh_mask_segment_options(self):
        if not hasattr(self, "maskSegmentBox"):
            return
        node = self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None
        self.maskSegmentBox.blockSignals(True)
        if hasattr(self, "maskSegmentChecklist"):
            self.maskSegmentChecklist.blockSignals(True)
        try:
            self.maskSegmentBox.clear()
            if hasattr(self, "maskSegmentChecklist"):
                self.maskSegmentChecklist.clear()
                self.maskSegmentChecklist.visible = False
            if node is None:
                self.maskSegmentBox.enabled = False
                if hasattr(self, "maskSegmentChecklist"):
                    self.maskSegmentChecklist.enabled = False
                self._update_mask_subset_visibility(False)
                return
            options = self._mask_subset_options(node)
            self._update_mask_subset_visibility(bool(options))
            if not options:
                self.maskSegmentBox.enabled = False
                if hasattr(self, "maskSegmentChecklist"):
                    self.maskSegmentChecklist.enabled = False
                return
            self.maskSegmentBox.addItem("All labels", SEGMENT_SELECTION_ALL)
            self.maskSegmentBox.addItem("Subset...", SEGMENT_SELECTION_SUBSET)
            subset_values = _node_csv_attribute(node, SEGMENT_SELECTION_IDS_ATTRIBUTE)
            label_values = _node_csv_attribute(node, LABEL_SELECTION_VALUES_ATTRIBUTE)
            legacy_id = node.GetAttribute(SEGMENT_SELECTION_ATTRIBUTE) if node is not None and hasattr(node, "GetAttribute") else None
            has_subset = bool(subset_values or label_values or (legacy_id and legacy_id != SEGMENT_SELECTION_ALL))
            if hasattr(self, "maskSegmentChecklist"):
                active = set(subset_values or label_values or ([legacy_id] if legacy_id and legacy_id != SEGMENT_SELECTION_ALL else []))
                for option in options:
                    item = qt.QListWidgetItem(str(option["text"]))
                    item.setData(qt.Qt.UserRole, str(option["value"]))
                    item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
                    item.setCheckState(qt.Qt.Checked if str(option["value"]) in active else qt.Qt.Unchecked)
                    self.maskSegmentChecklist.addItem(item)
                self._update_mask_subset_visibility(bool(options), subset_visible=has_subset and bool(options))
            self.maskSegmentBox.setCurrentIndex(1 if has_subset else 0)
            self.maskSegmentBox.enabled = bool(options)
        finally:
            self.maskSegmentBox.blockSignals(False)
            if hasattr(self, "maskSegmentChecklist"):
                self.maskSegmentChecklist.blockSignals(False)
        self._on_mask_segment_changed()

    def _on_mask_segment_changed(self, *_args):
        if not hasattr(self, "maskSegmentBox"):
            return
        node = self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None
        if node is None:
            self._update_mask_subset_visibility(False)
            return
        selected = self.maskSegmentBox.itemData(self.maskSegmentBox.currentIndex)
        subset = str(selected) == SEGMENT_SELECTION_SUBSET
        subset_visible = subset and _list_widget_count(self.maskSegmentChecklist) > 0 if hasattr(self, "maskSegmentChecklist") else False
        if hasattr(self, "maskSegmentChecklist"):
            self._update_mask_subset_visibility(True, subset_visible=subset_visible)
        if not subset:
            node.SetAttribute(SEGMENT_SELECTION_ATTRIBUTE, SEGMENT_SELECTION_ALL)
            node.SetAttribute(SEGMENT_SELECTION_IDS_ATTRIBUTE, "")
            node.SetAttribute(LABEL_SELECTION_VALUES_ATTRIBUTE, "")
        else:
            self._on_mask_segment_subset_changed()

    def _on_mask_segment_subset_changed(self, *_args):
        node = self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None
        if node is None or not hasattr(self, "maskSegmentChecklist"):
            return
        values = []
        for index in range(_list_widget_count(self.maskSegmentChecklist)):
            item = self.maskSegmentChecklist.item(index)
            try:
                if item.checkState() == qt.Qt.Checked:
                    values.append(str(item.data(qt.Qt.UserRole)))
            except Exception:
                pass
        if _is_segmentation_node(node):
            node.SetAttribute(SEGMENT_SELECTION_ATTRIBUTE, SEGMENT_SELECTION_SUBSET if values else SEGMENT_SELECTION_ALL)
            node.SetAttribute(SEGMENT_SELECTION_IDS_ATTRIBUTE, ",".join(values))
            node.SetAttribute(LABEL_SELECTION_VALUES_ATTRIBUTE, "")
        else:
            node.SetAttribute(SEGMENT_SELECTION_ATTRIBUTE, SEGMENT_SELECTION_ALL)
            node.SetAttribute(SEGMENT_SELECTION_IDS_ATTRIBUTE, "")
            node.SetAttribute(LABEL_SELECTION_VALUES_ATTRIBUTE, ",".join(values))

    def _mask_subset_options(self, node):
        if node is None:
            return []
        if _is_segmentation_node(node):
            segmentation = node.GetSegmentation()
            options = []
            for segment_id in _segmentation_segment_ids(node):
                segment = segmentation.GetSegment(segment_id)
                label = segment.GetName() if segment is not None else str(segment_id)
                options.append({"text": label, "value": str(segment_id)})
            return options
        labels = _label_values_from_node(node, self._volume())
        return [{"text": f"Label {value}", "value": str(value)} for value in labels]

    def _refresh_target_label_options(self):
        if hasattr(self, "icpTargetLabelBox"):
            current = _combo_data(self.icpTargetLabelBox)
            self.icpTargetLabelBox.blockSignals(True)
            try:
                self.icpTargetLabelBox.clear()
                for text, value in self._target_label_choices():
                    self.icpTargetLabelBox.addItem(text, value)
                _set_combo_data(self.icpTargetLabelBox, current)
            finally:
                self.icpTargetLabelBox.blockSignals(False)
        for row in range(self._table_row_count() if hasattr(self, "planeTable") else 0):
            combo = self.planeTable.cellWidget(row, 16)
            if combo is not None:
                current = _combo_data(combo)
                combo.blockSignals(True)
                try:
                    combo.clear()
                    for text, value in self._target_label_choices():
                        combo.addItem(text, value)
                    _set_combo_data(combo, current)
                finally:
                    combo.blockSignals(False)

    def _target_label_choices(self):
        label_node = None
        if self._selected_icp_target_source() == "slicer-node":
            label_node = self._selected_icp_target_node()
        if label_node is None:
            label_node = self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None
        labels = _label_values_from_node(label_node, self._volume())
        if not labels:
            labels = _label_values_from_node(self._volume(), self._volume())
        choices = [("All labels", "")]
        choices.extend((f"Label {value}", str(value)) for value in labels)
        return choices

    def _target_label_combo(self, selected=None):
        combo = qt.QComboBox()
        for text, value in self._target_label_choices():
            combo.addItem(text, value)
        _set_combo_data(combo, "" if selected is None else str(selected))
        return combo

    def _selected_icp_target_values(self):
        if not hasattr(self, "icpTargetLabelBox"):
            return None
        return _combo_selected_int_tuple(self.icpTargetLabelBox)

    def _selected_icp_target_source(self):
        if not hasattr(self, "icpTargetImageBox"):
            return "workflow-reference"
        value = _combo_data(self.icpTargetImageBox)
        value = str(value or "").strip().lower()
        if value in {"workflow-reference", "self", "slicer-node"}:
            return value
        return "workflow-reference"

    def _selected_icp_target_node(self):
        if not hasattr(self, "icpTargetNodeSelector"):
            return None
        try:
            return self.icpTargetNodeSelector.currentNode()
        except Exception:
            return None

    def _selected_mask_label_values(self):
        node = self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None
        return _selected_label_values_for_node(node)

    def auto_top_bottom_contact_planes(self):
        if not self._is_interactive_profile():
            slicer.util.errorDisplay("Select profile 'interactive_custom' to use contact planes.")
            return
        if self._volume() is None:
            slicer.util.errorDisplay("Select an image before creating contact planes.")
            return
        self._clear_contact_definitions()
        self._add_default_contact_planes()
        self._show_mask_3d_preserving_mask_selection(self.maskSelector.currentNode(), self._volume())
        bounds_source = "mask/segmentation" if self.maskSelector.currentNode() is not None else "image"
        self._append_log(f"Created top/bottom contact planes from current {bounds_source} bounds.\n")

    def _clear_contact_definitions(self):
        for row_data in self.contactPlaneRows:
            self.logic.remove_node(row_data.get("plane"))
        self.contactPlaneRows = []
        self.planeTable.setRowCount(0)
        self.loadTable.setRowCount(0)
        self.logic.remove_node(self.topDiskPlane)
        self.logic.remove_node(self.bottomDiskPlane)
        self.topDiskPlane = None
        self.bottomDiskPlane = None
        self.logic.remove_node(self.diskLabelSelector.currentNode())
        self.logic.remove_node(self.bcLabelSelector.currentNode())
        self.diskLabelSelector.setCurrentNode(None)
        self.bcLabelSelector.setCurrentNode(None)
        self.logic.remove_named_node("ParOSol_contact_caps_3D")
        self.logic.remove_named_node("ParOSol_boundary_conditions_3D")
        self._delete_bc_arrow_models()

    def _effective_mpi_processes(self):
        requested = max(1, int(self.mpiProcessesSpin.value))
        if requested <= 1:
            return 1
        launcher_text = (
            str(self.mpiLauncherEdit.text or "").strip()
            if hasattr(self, "mpiLauncherEdit")
            else ""
        )
        if launcher_text and not self._selected_mpi_launcher():
            self._append_log(
                f"MPI processes requested ({requested}) but the selected mpirun/mpiexec path is invalid; "
                "using 1 process. Clear the MPI launcher field to use the packaged parosol-py MPI runtime.\n"
            )
            return 1
        return requested

    def _selected_mpi_launcher(self):
        if not hasattr(self, "mpiLauncherEdit"):
            return ""
        launcher = str(self.mpiLauncherEdit.text or "").strip()
        if not launcher:
            return ""
        path = Path(launcher).expanduser()
        if not path.exists():
            self._append_log(
                f"MPI launcher does not exist: {path}; using 1 process. "
                "Install OpenMPI or select a valid mpirun/mpiexec path.\n"
            )
            return ""
        self.logic.set_setting_value("SlicerParOSol/mpiLauncher", str(path))
        return str(path)

    def _effective_solver_tolerance(self):
        text = str(self.solverToleranceEdit.text).strip()
        try:
            value = float(text)
        except Exception as exc:
            raise ValueError("Solver tolerance must be a positive number, for example 1e-4.") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Solver tolerance must be a positive number, for example 1e-4.")
        return value

    def _export_displacements_enabled(self):
        return "displacements" in self._selected_output_fields()

    def _selected_output_fields(self):
        if not hasattr(self, "outputFieldChecks"):
            fields = ["sed"]
            if bool(getattr(getattr(self, "exportDisplacementsCheckBox", None), "checked", False)):
                fields.append("displacements")
            return fields
        fields = [
            field
            for field, checkbox in self.outputFieldChecks.items()
            if bool(getattr(checkbox, "checked", False))
            and (
                field not in getattr(self, "nonlinearOutputFieldNames", set())
                or self._nonlinear_material_selected()
            )
        ]
        if bool(getattr(getattr(self, "exportDisplacementsCheckBox", None), "checked", False)) and "displacements" not in fields:
            fields.append("displacements")
        return fields or ["sed"]

    def _postprocess_config(self):
        preset = self._widget_text(getattr(self, "failurePresetBox", None), "Pistoia EES 0.7% / 2%")
        if preset == "None":
            return _postprocess_preset_config("none")
        elif preset == "Kopperdahl/Crawford 0.68%":
            return _postprocess_preset_config("kopperdahl")
        return _postprocess_preset_config("pistoia")

    def apply_materials(self):
        try:
            self._update_material_mode()
            self._append_log("Applied material settings.\n")
            return True
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
            return False

    def apply_materials_and_next(self):
        if self.apply_materials():
            self._advance_workflow_tab_after("materials")

    def crop_to_mask(self):
        if hasattr(self, "cropToMaskCheckBox"):
            self.cropToMaskCheckBox.checked = True
        self.preprocess_inputs()

    def preprocess_inputs(self):
        return self._preprocessController.preprocess_inputs()

    def preprocess_inputs_and_next(self):
        self._advanceAfterPreprocess = True
        try:
            return self.preprocess_inputs()
        finally:
            self._advanceAfterPreprocess = False

    def _preprocess_inputs_impl(self):
        try:
            image_node = self._volume()
            if image_node is None:
                raise ValueError("Select an image/material volume first.")
            mask_node = self.maskSelector.currentNode()
            self._apply_workflow_label_map_to_segmentation(mask_node)
            source_paths = self._current_input_storage_paths()
            if source_paths.get("image"):
                self._workflowReplaySourceInputs = source_paths
            steps = []
            self._lastIcpAlignment = None

            image_node, mask_node, preview_info = self._build_parosol_py_preprocess_preview()
            self.imageSelector.setCurrentNode(image_node)
            if mask_node is not None:
                self.maskSelector.setCurrentNode(mask_node)
            steps.extend(preview_info.get("steps", []))
            self._finish_preprocessed_inputs(image_node, mask_node, steps)
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

    def _finish_preprocessed_inputs(self, image_node, mask_node, steps):
        self.logic.remove_node(self.diskLabelSelector.currentNode())
        self.diskLabelSelector.setCurrentNode(None)
        self.logic.remove_node(self.bcLabelSelector.currentNode())
        self.bcLabelSelector.setCurrentNode(None)
        self._delete_bc_arrow_models()
        self.logic.remove_named_node("ParOSol_contact_caps_3D")
        self.logic.remove_named_node("ParOSol_boundary_conditions_3D")
        self._profileHasGeneratedBoundaryConditions = False
        self._show_mask_3d_preserving_mask_selection(mask_node, image_node)
        material_node = getattr(self, "_preprocessPreviewMaterialNode", None)
        self._apply_preprocess_display_mode(image_node, mask_node, material_node)
        self._preprocessingAppliedToInputs = True
        self._refresh_workflow_planes_after_preprocess()
        self._mark_stage_complete("anatomy")
        if steps:
            self._append_log("Preprocessed inputs: " + "; ".join(steps) + ".\n")
        else:
            self._append_log("Prepare Image clicked; no image-preparation options were enabled.\n")
        self._append_log(
            self._mask_label_count_summary(mask_node, image_node, prefix="Preprocessed mask") + "\n"
        )
        if getattr(self, "_advanceAfterPreprocess", False):
            self._advance_workflow_tab_after("anatomy")

    def _refresh_preprocess_preview_display(self, *_args):
        image_node = getattr(self, "_preprocessPreviewImageNode", None) or self._volume()
        mask_node = getattr(self, "_preprocessPreviewMaskNode", None) or self.maskSelector.currentNode()
        material_node = getattr(self, "_preprocessPreviewMaterialNode", None)
        self._apply_preprocess_display_mode(image_node, mask_node, material_node)

    def _apply_preprocess_display_mode(self, image_node, mask_node, material_node):
        mode = self._widget_text(
            getattr(self, "preprocessPreviewModeBox", None),
            "Image + mask",
        ).strip().lower()
        if mode == "material" and material_node is not None:
            self._show_in_standard_slice_views(
                material_node,
                label_node=None,
                reset_orientations=False,
            )
            return
        if mode == "mask" and mask_node is not None:
            self._show_in_standard_slice_views(
                mask_node,
                label_node=None,
                reset_orientations=False,
            )
            return
        self._show_in_standard_slice_views(
            image_node,
            label_node=mask_node,
            reset_orientations=False,
        )

    def _should_use_parosol_py_preprocess_preview(self):
        return bool(
            self._has_applied_workflow_replay_model()
            or getattr(self.icpRegistrationCheckBox, "checked", False)
        )

    def _build_parosol_py_preprocess_preview(self):
        output_dir = Path(self.outputDirectory.directory) / "preprocess_preview"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path, mask_path = self._workflow_replay_source_input_paths_for_export(output_dir)
        use_workflow_preview = self._should_use_parosol_py_preprocess_preview()
        config = (
            self._interactive_workflow_replay_config_for_export(
                image_path=image_path,
                mask_path=mask_path,
                output_dir=output_dir,
                material_override=self._material_override(),
                mpi_processes=1,
                mpi_launcher="",
                tolerance=self._effective_solver_tolerance(),
                export_displacements=False,
                output_fields=self._selected_output_fields(),
                postprocess_config=self._postprocess_config(),
            )
            if use_workflow_preview and self._has_applied_workflow_replay_model()
            else None
        )
        if config is None:
            config = self._generic_parosol_py_preprocess_preview_config(
                image_path=image_path,
                mask_path=mask_path,
                output_dir=output_dir,
            )
        _prepare_parosol_py_runtime_import()

        model_config = copy.deepcopy(config["model"])
        if isinstance(config.get("slicer_editor"), dict):
            model_config["slicer_editor"] = copy.deepcopy(config["slicer_editor"])

        if use_workflow_preview:
            from parosol_py.modeling.workflow_replay import build_workflow_replay_preview

            preview = build_workflow_replay_preview(
                model_config,
                base_dir=output_dir,
                preprocessing_config=config.get("preprocessing"),
            )
            image_node, mask_node = self._load_workflow_replay_preprocess_preview(preview)
            registration = copy.deepcopy(preview.metadata.get("registration", {}))
            registration_cfg = model_config.get("registration", {})
            self_reference_authoring = bool(
                isinstance(registration_cfg, dict)
                and registration_cfg.get("reference_authoring")
                and registration_cfg.get("self_reference")
            )
            self_reference_points = None
            if self_reference_authoring:
                self_reference_points = _sample_reference_points_from_parosol_preview_mask(
                    getattr(preview, "registration_mask_zyx", preview.mask_zyx),
                    spacing=preview.spacing,
                    origin=preview.origin,
                    max_points=int(registration_cfg.get("max_points", 8000)),
                    active_values=None,
                )
                if self_reference_points.size == 0:
                    raise ValueError("Cannot author ICP reference: preprocessed mask has no foreground points.")
                self_reference_transform = estimate_reference_to_sample_transform(
                    self_reference_points,
                    self_reference_points,
                    iterations=int(registration_cfg.get("iterations", 50)),
                    tolerance=float(registration_cfg.get("tolerance", 1.0e-4)),
                    allow_scale=False,
                )
                registration = _self_reference_registration_metadata(
                    registration_cfg,
                    point_count=int(self_reference_points.shape[0]),
                    transform=self_reference_transform,
                )
                preview.metadata["registration"] = registration
            self._lastIcpAlignment = {
                "image_id": image_node.GetID(),
                "mask_id": mask_node.GetID() if mask_node is not None else "",
                "reference": self._current_workflow_reference_points(),
                "reference_points": self_reference_points
                if self_reference_points is not None
                else (
                    preview.reference_points
                    if isinstance(preview.reference_points, np.ndarray)
                    else None
                ),
                "metadata": registration,
                "self_reference": self_reference_authoring,
            }
            shape = tuple(int(value) for value in np.asarray(preview.density_zyx).shape)
            steps = [
                "workflow_replay_preview "
                f"model_space={preview.metadata.get('model_space', 'unknown')} "
                f"zyx={shape[0]}x{shape[1]}x{shape[2]}"
            ]
            if registration.get("enabled", False):
                steps.append(
                    "icp_align "
                    f"iterations={int(registration.get('iterations', 0) or 0)} "
                    f"mean_distance={float(registration.get('mean_distance', 0.0) or 0.0):.3g} mm"
                )
            return image_node, mask_node, {"steps": steps, "metadata": preview.metadata}

        from parosol_py.modeling.common import build_preprocessed_inputs_preview

        preview = build_preprocessed_inputs_preview(
            model_config,
            base_dir=output_dir,
            preprocessing_config=config.get("preprocessing"),
            custom_preprocessing_config=config.get("custom_preprocessing"),
        )
        image_node, mask_node = self._load_parosol_py_preprocess_preview(
            preview,
            image_name="ParOSol_preprocessed_image",
            mask_name="ParOSol_preprocessed_mask",
            material_name="ParOSol_preprocess_material_preview",
        )
        shape = tuple(int(value) for value in np.asarray(preview.density_zyx).shape)
        steps = [f"parosol_py_preprocess zyx={shape[0]}x{shape[1]}x{shape[2]}"]
        return image_node, mask_node, {"steps": steps, "metadata": preview.metadata}

    def _generic_parosol_py_preprocess_preview_config(self, *, image_path, mask_path, output_dir):
        source = copy.deepcopy(self._active_workflow_config() or {})
        model_cfg = copy.deepcopy(source.get("model", {})) if isinstance(source, dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        registration_cfg = copy.deepcopy(model_cfg.get("registration", {}))
        if not isinstance(registration_cfg, dict):
            registration_cfg = {}
        icp_enabled = bool(getattr(self.icpRegistrationCheckBox, "checked", False))
        if icp_enabled:
            target_source = self._selected_icp_target_source()
            reference_points = registration_cfg.get("reference_points") or self._current_workflow_reference_points()
            selected_values = self._selected_icp_target_values()
            if selected_values:
                target = int(selected_values[0]) if len(selected_values) == 1 else list(selected_values)
                registration_cfg["target"] = target
                model_cfg.setdefault("targets", {})["registration"] = target
            if target_source == "workflow-reference" and not reference_points:
                target_source = "self"
            registration_cfg["target_image"] = target_source
            if target_source == "self":
                registration_cfg["enabled"] = False
                registration_cfg["reference_authoring"] = True
                registration_cfg["self_reference"] = True
                registration_cfg.pop("reference_points", None)
                registration_cfg.setdefault("method", "lightweight_icp")
                registration_cfg.setdefault("max_points", 8000)
                registration_cfg.setdefault("iterations", 50)
                registration_cfg.setdefault("source_landmark_mode", "linspace")
                registration_cfg.setdefault("reference_landmark_mode", "linspace")
            elif target_source == "slicer-node":
                max_points = int(registration_cfg.get("max_points", 8000))
                points = self._icp_reference_points_from_selected_target_node(max_points=max_points)
                reference_dir = Path(output_dir) / "reference"
                reference_dir.mkdir(parents=True, exist_ok=True)
                reference_path = reference_dir / "slicer_target_points_preview.npz"
                np.savez_compressed(reference_path, points=np.asarray(points, dtype=np.float32))
                registration_cfg["enabled"] = True
                registration_cfg["reference_points"] = str(reference_path)
                registration_cfg["reference_source_node"] = _node_reference_description(
                    self._selected_icp_target_node()
                )
                registration_cfg.setdefault("method", "lightweight_icp")
                registration_cfg.setdefault("max_points", max_points)
                registration_cfg.setdefault("iterations", 50)
            else:
                registration_cfg["enabled"] = True
                registration_cfg["reference_points"] = str(reference_points)
                registration_cfg["target_image"] = "workflow-reference"
                registration_cfg.setdefault("method", "lightweight_icp")
                registration_cfg.setdefault("max_points", 8000)
                registration_cfg.setdefault("iterations", 50)
        else:
            registration_cfg["enabled"] = False
        model_cfg["type"] = "workflow_replay"
        model_cfg["density_image"] = str(image_path)
        if mask_path:
            model_cfg["mask_image"] = str(mask_path)
        else:
            model_cfg.pop("mask_image", None)
        model_cfg["registration"] = registration_cfg
        editor = self._editor_from_active_workflow()
        if isinstance(editor, dict):
            model_cfg["slicer_editor"] = copy.deepcopy(editor)
        replay_cfg = copy.deepcopy(model_cfg.get("workflow_replay", {}))
        if not isinstance(replay_cfg, dict):
            replay_cfg = {}
        replay_cfg["enabled"] = True
        replay_cfg["model_space"] = "reference" if icp_enabled else "sample"
        model_cfg["workflow_replay"] = replay_cfg
        config = {
            "case": {"name": Path(output_dir).name or "preprocess_preview", "work_dir": str(output_dir)},
            "model": model_cfg,
        }
        preprocessing = self._preprocessing_config(force=True)
        if preprocessing:
            config["preprocessing"] = preprocessing
        custom_preprocessing = self._custom_preprocessing_config()
        if custom_preprocessing:
            config["custom_preprocessing"] = custom_preprocessing
        return config

    def _load_workflow_replay_preprocess_preview(self, preview):
        return self._load_parosol_py_preprocess_preview(
            preview,
            image_name="ParOSol_icp_aligned_image",
            mask_name="ParOSol_icp_aligned_mask",
            material_name="ParOSol_icp_aligned_material_preview",
        )

    def _load_parosol_py_preprocess_preview(self, preview, *, image_name, mask_name, material_name):
        self.logic.remove_named_node(str(image_name))
        self.logic.remove_named_node(str(mask_name))
        image_node = _volume_node_from_parosol_ras_array(
            preview.density_zyx,
            str(image_name),
            preview.spacing,
            preview.origin,
            label=False,
        )
        mask_node = _volume_node_from_parosol_ras_array(
            preview.mask_zyx,
            str(mask_name),
            preview.spacing,
            preview.origin,
            label=True,
        )
        self.logic.group_node(image_node, "Inputs")
        self.logic.style_labelmap(mask_node, "mask")
        self.logic.group_node(mask_node, "Inputs")
        material_node = self._preprocess_material_preview_node(
            preview,
            name=str(material_name),
        )
        self._preprocessPreviewImageNode = image_node
        self._preprocessPreviewMaskNode = mask_node
        self._preprocessPreviewMaterialNode = material_node
        return image_node, mask_node

    def _preprocess_material_preview_node(self, preview, *, name):
        self.logic.remove_named_node(str(name))
        material_override = self._material_override()
        if not isinstance(material_override, dict):
            return None
        material_config = material_override.get("materials")
        if not isinstance(material_config, dict):
            return None
        try:
            material_zyx = self._preprocess_material_preview_array(
                preview,
                material_override=material_override,
                material_config=material_config,
            )
        except Exception as exc:
            self._append_log(f"Material preview skipped: {exc}\n")
            return None
        material_node = _volume_node_from_parosol_ras_array(
            material_zyx,
            str(name),
            preview.spacing,
            preview.origin,
            label=False,
        )
        _apply_material_preview_display(material_node)
        self.logic.group_node(material_node, "Inputs")
        return material_node

    def _preprocess_material_preview_array(self, preview, *, material_override, material_config):
        image_type = str(material_override.get("image_type", "density")).strip().lower()
        density_zyx = np.asarray(preview.density_zyx, dtype=np.float64)
        active_mask_zyx = np.asarray(preview.mask_zyx) != 0
        if image_type == "density":
            from parosol_py.modeling.common import material_from_density

            material_zyx, _poisson_ratio = material_from_density(
                density_zyx,
                active_mask_zyx,
                material_config=material_config,
            )
            return np.asarray(material_zyx, dtype=np.float32)
        if image_type in {"material_labels", "labels", "segmentation"}:
            from parosol_py.materials import (
                labels_to_material_map,
                linear_isotropic_materials_from_config,
            )

            table = linear_isotropic_materials_from_config(material_config)
            labels_zyx = np.asarray(np.rint(density_zyx), dtype=np.int64)
            mapped = labels_to_material_map(
                labels_zyx,
                table,
                poisson_ratio=material_config.get(
                    "poisson_ratio",
                    material_config.get("nu"),
                ),
            )
            return np.asarray(mapped.youngs_modulus_mpa, dtype=np.float32)
        if image_type in {"material_mpa", "mpa", "material"}:
            return np.asarray(np.where(active_mask_zyx, density_zyx, 0.0), dtype=np.float32)
        if image_type in {"material_gpa", "gpa"}:
            return np.asarray(np.where(active_mask_zyx, density_zyx * 1000.0, 0.0), dtype=np.float32)
        raise ValueError(
            "material preview supports density, material labels, material_mpa, and material_gpa inputs"
        )

    def _workflow_replay_boundary_preview(self, *, show_load_vectors=False):
        output_dir = Path(self.outputDirectory.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path, mask_path = self._workflow_replay_source_input_paths_for_export(output_dir)
        effective_mpi_processes = self._effective_mpi_processes()
        effective_mpi_launcher = (
            self._selected_mpi_launcher() if effective_mpi_processes > 1 else ""
        )
        config = self._interactive_workflow_replay_config_for_export(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=output_dir,
            load_case_override=None,
            nodeset_specs=None,
            material_override=self._material_override(),
            mpi_processes=effective_mpi_processes,
            mpi_launcher=effective_mpi_launcher,
            tolerance=self._effective_solver_tolerance(),
            export_displacements=self._export_displacements_enabled(),
            output_fields=self._selected_output_fields(),
            postprocess_config=self._postprocess_config(),
        )
        if config is None:
            raise ValueError("The applied workflow is not a workflow-replay model.")
        config_path = self.logic.write_config(config, output_dir / "parosol_slicer_case.yaml")
        self._validate_exported_config_nodeset_files(config_path)

        result = self._run_workflow_replay_dry_run(config_path, output_dir)
        manifest_path = Path(config["model"]["outputs"]["manifest"])
        if not manifest_path.exists():
            manifest_path = Path(result.exported.get("manifest", manifest_path))
        if not manifest_path.exists():
            raise ValueError(f"Workflow replay did not export a model manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolved_editor = self._resolved_workflow_editor_from_manifest(
            manifest,
            manifest_path=manifest_path,
        )
        if not isinstance(resolved_editor, dict):
            raise ValueError("Workflow replay did not report the resolved editor state.")

        outputs = config["model"]["outputs"]
        material_path = Path(outputs["material_image"])
        nodeset_path = Path(outputs["nodeset_image"])
        disk_path_value = outputs.get("disk_label_image")
        disk_path = Path(disk_path_value) if disk_path_value else None
        allowed_nodeset_labels = sorted(
            {
                int(label)
                for label in _workflow_nodeset_label_map(config).values()
                if int(label) > 0
            }
        )
        material_node, nodeset_node = self._load_workflow_replay_model_preview(
            material_path,
            nodeset_path,
            disk_path=disk_path,
            allowed_nodeset_labels=allowed_nodeset_labels,
        )

        preserved_sizes = self._current_plane_sizes_by_name()
        self._appliedWorkflowNodesetLabels = _workflow_nodeset_label_map(config)
        self._workflowReplayContractEditor = None
        self._workflowReplayResolvedEditor = None
        self._workflowReplayResolvedEditorDirty = False
        keep_existing_planes = bool(getattr(self, "contactPlaneRows", []))
        if not keep_existing_planes:
            self._clear_editable_profile_state(clear_materials=False)
            self._apply_profile_planes_and_loads(
                resolved_editor,
                preserve_existing_sizes=preserved_sizes,
            )
        else:
            self._update_load_table_nodeset_labels()
        if show_load_vectors:
            self._refresh_bc_arrow_models(nodeset_node, reference_node=material_node)
        else:
            self._delete_bc_arrow_models()
        self._workflowReplayContractEditor = copy.deepcopy(config.get("slicer_editor", {}))
        if keep_existing_planes:
            self._workflowReplayResolvedEditor = copy.deepcopy(self._editor_state_config())
        else:
            self._workflowReplayResolvedEditor = copy.deepcopy(resolved_editor)
        self._workflowReplayResolvedEditorDirty = False
        self._preprocessingAppliedToInputs = True
        self._profileHasGeneratedBoundaryConditions = True
        self._mark_stage_complete("boundary")
        current_mask_node = self.maskSelector.currentNode()
        self._lastIcpAlignment = {
            "image_id": material_node.GetID() if material_node is not None else "",
            "mask_id": current_mask_node.GetID() if current_mask_node is not None else "",
            "reference": self._current_workflow_reference_points(),
            "metadata": manifest.get("model", {}).get("registration", {}),
        }
        registration = manifest.get("model", {}).get("registration", {})
        replay = manifest.get("model", {}).get("workflow_replay", {})
        shape = manifest.get("shape_zyx", [])
        shape_text = "x".join(str(int(value)) for value in shape) if len(shape) == 3 else "unknown"
        self._append_log(
            "Generated workflow replay boundary preview with shared ParOSol-py builder: "
            f"model_space={replay.get('model_space', 'unknown')}; "
            f"shape_zyx={shape_text}; "
            f"icp_align iterations={int(registration.get('iterations', 0) or 0)} "
            f"mean_distance={float(registration.get('mean_distance', 0.0) or 0.0):.3g} mm.\n"
        )
        self._append_log(
            self._mask_label_count_summary(
                nodeset_node,
                material_node,
                prefix="Generated model labels",
            )
            + "\n"
        )
        self._update_profile_mode()
        return config_path

    def _resolved_workflow_editor_from_manifest(self, manifest, *, manifest_path):
        model = manifest.get("model") if isinstance(manifest, dict) else None
        replay = model.get("workflow_replay") if isinstance(model, dict) else None
        if isinstance(replay, dict) and isinstance(replay.get("resolved_editor"), dict):
            return replay["resolved_editor"]
        replay_keys = sorted(replay.keys()) if isinstance(replay, dict) else []
        raise ValueError(
            "Workflow replay manifest is missing model.workflow_replay.resolved_editor. "
            "This usually means Slicer loaded an older ParOSol-py or an older SlicerParOSol module. "
            f"Manifest: {manifest_path}. "
            f"Available workflow_replay keys: {', '.join(replay_keys) if replay_keys else 'none'}. "
            "Restart Slicer after loading the current SlicerParOSol module and make sure "
            "SLICER_PAROSOL_SOURCE points to the current parosol-py/src checkout."
        )

    def _run_workflow_replay_dry_run(self, config_path, output_dir):
        _prepare_parosol_py_runtime_import()
        from parosol_py.config import run_case_config

        return run_case_config(config_path, dry_run=True, work_dir=output_dir)

    def _workflow_replay_source_input_paths_for_export(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stored = getattr(self, "_workflowReplaySourceInputs", None)
        stored = stored if isinstance(stored, dict) else {}
        image_node = self._volume()
        mask_node = self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None

        image_path = stored.get("image") or _node_storage_file(image_node)
        if image_path and Path(image_path).exists():
            image_path = Path(image_path)
        elif image_node is not None and not _is_generated_parosol_node(image_node):
            image_path = self.logic.export_volume(
                image_node,
                output_dir / "slicer_source_input.nii.gz",
            )
        else:
            raise ValueError(
                "Cannot build workflow replay: the original source image is unavailable. "
                "Reload the input image or reapply the workflow."
            )

        mask_path = stored.get("mask") or _node_storage_file(mask_node)
        if mask_path and Path(mask_path).exists():
            mask_path = Path(mask_path)
        elif mask_node is not None and not _is_generated_parosol_node(mask_node):
            mask_path = self.logic.export_mask_like(
                mask_node,
                image_node,
                output_dir / "slicer_source_mask.nii.gz",
            )
        else:
            mask_path = None

        self._workflowReplaySourceInputs = {
            "image": str(image_path),
            **({"mask": str(mask_path)} if mask_path else {}),
        }
        return image_path, mask_path

    def _load_workflow_replay_model_preview(
        self,
        material_path,
        nodeset_path,
        *,
        disk_path=None,
        allowed_nodeset_labels=None,
    ):
        for name in (
            "ParOSol_workflow_material_preview",
            "ParOSol_workflow_disks_preview",
            "ParOSol_workflow_nodesets_preview",
        ):
            self.logic.remove_named_node(name)
        selected_mask_node = self.maskSelector.currentNode()
        material_node = _load_volume_node(
            material_path,
            {"name": "ParOSol_workflow_material_preview", "show": False},
        )
        self._workflowReplayPreviewMaterialNode = material_node
        disk_node = None
        if disk_path is not None and Path(disk_path).exists():
            disk_node = _load_label_volume_node(
                disk_path,
                {"name": "ParOSol_workflow_disks_preview", "show": False},
            )
        nodeset_node = _load_label_volume_node(
            nodeset_path,
            {"name": "ParOSol_workflow_nodesets_preview", "show": False},
        )
        _filter_labelmap_to_values(nodeset_node, allowed_nodeset_labels)
        if material_node is not None:
            self.logic.group_node(material_node, "Inputs")
        if disk_node is not None:
            self.logic.style_labelmap(disk_node, "disks")
            self.logic.group_node(disk_node, "Contact Regions")
        if nodeset_node is not None:
            self.logic.style_labelmap(nodeset_node, "nodesets")
            self.logic.group_node(nodeset_node, "Contact Regions")
        if disk_node is not None and int(np.count_nonzero(slicer.util.arrayFromVolume(disk_node))) > 0:
            self.logic.labelmap_to_3d_segmentation(
                disk_node,
                "ParOSol_contact_caps_3D",
                reference_node=material_node,
                kind="disks",
            )
        else:
            self.logic.remove_named_node("ParOSol_contact_caps_3D")
        if nodeset_node is not None and int(np.count_nonzero(slicer.util.arrayFromVolume(nodeset_node))) > 0:
            self.logic.labelmap_to_3d_segmentation(
                nodeset_node,
                "ParOSol_boundary_conditions_3D",
                reference_node=material_node,
                kind="nodesets",
            )
        else:
            self.logic.remove_named_node("ParOSol_boundary_conditions_3D")
        signal_states = self._begin_input_node_update_suppression()
        try:
            if nodeset_node is not None:
                self.bcLabelSelector.setCurrentNode(nodeset_node)
            self.diskLabelSelector.setCurrentNode(disk_node)
        finally:
            self._end_input_node_update_suppression(signal_states)
        if selected_mask_node is not None and material_node is not None:
            self._show_mask_3d_preserving_mask_selection(selected_mask_node, material_node)
        self._show_in_standard_slice_views(material_node, label_node=None)
        return material_node, nodeset_node

    def _mask_label_count_summary(self, mask_node, image_node, *, prefix="Mask"):
        if mask_node is None:
            return f"{prefix}: none"
        try:
            array = np.asarray(_array_from_mask_like(mask_node, image_node, apply_selection=False))
        except Exception as exc:
            return f"{prefix}: unreadable ({exc})"
        values, counts = np.unique(array, return_counts=True)
        parts = []
        for value, count in zip(values, counts, strict=True):
            try:
                integer = int(value)
            except (TypeError, ValueError):
                continue
            if integer != 0:
                parts.append(f"{integer}={int(count)}")
        return f"{prefix} labels: {', '.join(parts) if parts else 'empty'}"

    def _current_workflow_reference_points(self):
        config = self._active_workflow_config()
        model = config.get("model", {}) if isinstance(config, dict) else {}
        registration = model.get("registration", {}) if isinstance(model, dict) else {}
        reference = registration.get("reference_points") if isinstance(registration, dict) else None
        return str(reference) if reference else ""

    def _apply_workflow_label_map_to_segmentation(self, mask_node):
        if not _is_segmentation_node(mask_node):
            return
        config = self._active_workflow_config()
        model = config.get("model", {}) if isinstance(config, dict) else {}
        labels = model.get("labels", {}) if isinstance(model, dict) else {}
        if not isinstance(labels, dict) or not labels:
            return
        label_map = {}
        for segment_id in _segmentation_segment_ids(mask_node):
            segment = mask_node.GetSegmentation().GetSegment(str(segment_id))
            name = str(segment.GetName() if segment is not None else segment_id).strip().lower()
            value = None
            if "body" in name or "centrum" in name:
                value = labels.get("body", labels.get("vertebral_body"))
            elif "process" in name or "posterior" in name or "arch" in name:
                value = labels.get("process", labels.get("vertebral_process"))
            elif "left" in name and "femur" in name:
                value = labels.get("left_femur", labels.get("femur"))
            elif "right" in name and "femur" in name:
                value = labels.get("right_femur", labels.get("femur"))
            try:
                if value is not None:
                    label_map[str(segment_id)] = int(value)
            except (TypeError, ValueError):
                pass
        if label_map:
            mask_node.SetAttribute(
                SEGMENT_LABEL_VALUE_MAP_ATTRIBUTE,
                ",".join(f"{key}:{value}" for key, value in sorted(label_map.items())),
            )

    def _current_workflow_reference_coordinate_system(self):
        config = self._active_workflow_config()
        model = config.get("model", {}) if isinstance(config, dict) else {}
        registration = model.get("registration", {}) if isinstance(model, dict) else {}
        if not isinstance(registration, dict):
            return "auto"
        return str(registration.get("coordinate_system", registration.get("reference_coordinate_system", "auto")))

    def _current_workflow_registration_flag(self, key, default=False):
        config = self._active_workflow_config()
        model = config.get("model", {}) if isinstance(config, dict) else {}
        registration = model.get("registration", {}) if isinstance(model, dict) else {}
        if not isinstance(registration, dict):
            return bool(default)
        return _enabled_value(registration.get(key, default))

    def _update_icp_target_label_ui(self, *_args):
        if not hasattr(self, "icpTargetLabelBox"):
            return
        enabled = bool(getattr(self.icpRegistrationCheckBox, "checked", False))
        target_source = self._selected_icp_target_source()
        if enabled and target_source == "workflow-reference" and not self._current_workflow_reference_points():
            target_source = "self"
            if hasattr(self, "icpTargetImageBox"):
                self.icpTargetImageBox.blockSignals(True)
                try:
                    _set_combo_data(self.icpTargetImageBox, target_source)
                finally:
                    self.icpTargetImageBox.blockSignals(False)
        if hasattr(self, "icpTargetImageBox"):
            self.icpTargetImageBox.visible = enabled
        if hasattr(self, "icpTargetImageLabel") and self.icpTargetImageLabel is not None:
            self.icpTargetImageLabel.visible = enabled
        node_visible = enabled and target_source == "slicer-node"
        if hasattr(self, "icpTargetNodeSelector"):
            self.icpTargetNodeSelector.visible = node_visible
        if hasattr(self, "icpTargetNodeLabel") and self.icpTargetNodeLabel is not None:
            self.icpTargetNodeLabel.visible = node_visible
        self.icpTargetLabelBox.visible = enabled
        if hasattr(self, "icpTargetLabel") and self.icpTargetLabel is not None:
            self.icpTargetLabel.visible = enabled
        if enabled:
            self._refresh_target_label_options()

    def _current_workflow_disk_projection_values(self):
        if not hasattr(self, "profileBox"):
            return None
        config = self._active_workflow_config()
        model = config.get("model", {}) if isinstance(config, dict) else {}
        if not isinstance(model, dict):
            return None
        return self._workflow_target_values_for_current_mask(
            _workflow_disk_projection_active_values(model)
        )

    def _current_workflow_registration_values(self, model=None, registration=None):
        if model is None:
            if not hasattr(self, "profileBox"):
                return None
            config = self._active_workflow_config()
            model = config.get("model", {}) if isinstance(config, dict) else {}
        if not isinstance(model, dict):
            return None
        if registration is None:
            registration = model.get("registration", {})
        if not isinstance(registration, dict):
            registration = {}
        return self._workflow_target_values_for_current_mask(
            _workflow_registration_active_values(model, registration)
        )

    def _workflow_target_values_for_current_mask(self, values):
        if not values:
            return None
        try:
            requested = tuple(int(value) for value in values if str(value) != "")
        except (TypeError, ValueError):
            return values
        if not requested:
            return None
        current = set(_label_values_from_node(
            self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None,
            self._volume(),
        ))
        if not current:
            current = set(_label_values_from_node(self._volume(), self._volume()))
        if set(requested).issubset(current):
            return requested
        if len(requested) == 1 and len(current) == 1:
            return (int(next(iter(current))),)
        return requested

    def _icp_transform_for_current_sample(self, image_node, mask_node=None):
        config = self._active_workflow_config()
        model = config.get("model", {}) if isinstance(config, dict) else {}
        registration = model.get("registration", {}) if isinstance(model, dict) else {}
        if not isinstance(registration, dict) or not registration.get("enabled", False):
            raise ValueError("ICP registration is not enabled for the current workflow.")
        active_values = self._selected_icp_target_values() or _workflow_registration_active_values(model, registration)
        reference = self._current_workflow_reference_points()
        if not reference:
            raise ValueError("ICP registration is enabled, but the workflow has no reference_points file.")
        max_points = int(registration.get("max_points", 8000))
        iterations = int(registration.get("iterations", 50))
        tolerance = float(registration.get("tolerance", 1.0e-4))
        mask_like_node = mask_node or image_node
        sample_points = _sample_reference_points_from_mask_like(
            mask_like_node,
            image_node,
            max_points=max_points,
            active_values=active_values,
        )
        if sample_points.size == 0:
            raise ValueError("Cannot run ICP: current mask/image has no non-zero surface points.")
        reference_points = read_reference_points(
            reference,
            max_points=max_points,
            coordinate_system=self._current_workflow_reference_coordinate_system(),
        )
        reference_points, scaling_meta = scale_reference_points_preserving_pose(
            reference_points=reference_points,
            sample_points=sample_points,
            registration_config=registration,
        )
        prealign_reference = _enabled_value(registration.get("prealign_reference_to_sample", False))
        prealign_metadata = {}
        if prealign_reference:
            reference_points, prealign_metadata = prealign_reference_points_to_sample(
                reference_points,
                sample_points,
            )
        sample_to_reference = estimate_reference_to_sample_transform(
            sample_points,
            reference_points,
            iterations=iterations,
            tolerance=tolerance,
            allow_scale=False,
        )
        reference_to_sample = invert_rigid_transform(sample_to_reference)
        extent_points = _foreground_extent_points_from_mask_like(
            mask_like_node,
            image_node,
        )
        if extent_points.size == 0:
            extent_points = sample_points
        return {
            "reference_points": reference_points,
            "sample_points": sample_points,
            "extent_points": extent_points,
            "reference_to_sample": reference_to_sample,
            "sample_to_reference": sample_to_reference,
            "registration": registration,
            "reference_scaling": scaling_meta,
            "prealign_metadata": prealign_metadata,
        }

    def _current_inputs_are_icp_aligned_reference_frame(self, reference):
        info = getattr(self, "_lastIcpAlignment", None)
        if not isinstance(info, dict):
            return False
        image_node = self._volume()
        if image_node is None or image_node.GetID() != info.get("image_id"):
            return False
        if str(reference) != str(info.get("reference", "")):
            return False
        mask_node = self.maskSelector.currentNode()
        stored_mask_id = str(info.get("mask_id", ""))
        if stored_mask_id and (mask_node is None or mask_node.GetID() != stored_mask_id):
            return False
        return True

    def _refresh_workflow_planes_after_preprocess(self):
        editor = self._editor_from_active_workflow()
        if not isinstance(editor, dict):
            return
        config = self._active_workflow_config()
        if isinstance(config, dict):
            self._appliedWorkflowNodesetLabels = _workflow_nodeset_label_map(config)
        preserved_sizes = self._current_plane_sizes_by_name()
        replay_model = self._has_applied_workflow_replay_model()
        if replay_model:
            self._workflowReplayContractEditor = None
            self._workflowReplayResolvedEditor = None
            self._workflowReplayResolvedEditorDirty = False
        self._clear_editable_profile_state(clear_materials=False)
        resolved_editor = (
            self._resolve_reference_space_editor_for_current_sample(editor)
            if self._editor_needs_reference_resolution(editor)
            else editor
        )
        resolved_editor = self._resolve_bbox_relative_editor_for_current_sample(resolved_editor)
        self._apply_profile_planes_and_loads(
            resolved_editor,
            preserve_existing_sizes=preserved_sizes,
        )
        if replay_model:
            self._workflowReplayContractEditor = copy.deepcopy(editor)
            self._workflowReplayResolvedEditor = copy.deepcopy(resolved_editor)
            self._workflowReplayResolvedEditorDirty = False

    def _resolve_bbox_relative_editor_for_current_sample(self, editor):
        if not isinstance(editor, dict):
            return editor
        planes = editor.get("planes", [])
        if not isinstance(planes, list) or not any(
            isinstance(plane, dict)
            and (
                plane.get("center_fraction") is not None
                or plane.get("bbox_fraction_bounds") is not None
            )
            for plane in planes
        ):
            return editor
        mask_node = self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None
        volume_node = self._volume()
        if volume_node is None:
            return editor
        active_values = self._current_workflow_disk_projection_values()
        mask = _target_mask_array(
            mask_node or volume_node,
            volume_node,
            active_values=active_values,
            fallback_to_nonzero=True,
        )
        if mask is None or not np.any(mask):
            return editor
        bounds = _mask_bounds_ras(mask, volume_node)
        if bounds is None:
            return editor
        bounds_min, bounds_max = bounds
        extent = [max(bounds_max[index] - bounds_min[index], 1.0e-6) for index in range(3)]
        resolved = copy.deepcopy(editor)
        for plane in resolved.get("planes", []):
            if not isinstance(plane, dict):
                continue
            if plane.get("center_fraction") is None and plane.get("bbox_fraction_bounds") is None:
                continue
            normal = _normalized(plane.get("normal_ras", [0.0, 0.0, 1.0]))
            u_axis = _normalized(plane.get("u_axis_ras", _default_plane_u_axis(normal)))
            v_axis = _normalized(plane.get("v_axis_ras", _cross(normal, u_axis)))
            fraction_bounds = _bbox_fraction_bounds(plane.get("bbox_fraction_bounds"))
            if fraction_bounds is not None:
                fraction_min = fraction_bounds[:, 0]
                fraction_max = fraction_bounds[:, 1]
                center_fraction = (0.5 * (fraction_min + fraction_max)).tolist()
                span_extent = (
                    np.abs(fraction_max - fraction_min)
                    * np.asarray(extent, dtype=np.float64)
                )
                size_mm = [
                    _extent_along_axis(span_extent, u_axis),
                    _extent_along_axis(span_extent, v_axis),
                ]
                relative_definition = {
                    "relative_to": "model_bbox",
                    "bbox_fraction_bounds": _bbox_fraction_bounds_metadata(
                        plane.get("bbox_fraction_bounds")
                    ),
                }
            else:
                try:
                    center_fraction = [float(value) for value in plane.get("center_fraction")]
                except Exception:
                    continue
                if len(center_fraction) != 3:
                    continue
                size_fraction = plane.get("size_fraction", [1.0, 1.0])
                try:
                    size_fraction = [float(size_fraction[0]), float(size_fraction[1])]
                except Exception:
                    size_fraction = [1.0, 1.0]
                size_mm = [
                    _extent_along_axis(extent, u_axis) * size_fraction[0],
                    _extent_along_axis(extent, v_axis) * size_fraction[1],
                ]
                relative_definition = {
                    "relative_to": "model_bbox",
                    "center_fraction": center_fraction,
                    "size_fraction": size_fraction,
                }
            plane["center_ras"] = [
                bounds_min[index] + center_fraction[index] * extent[index]
                for index in range(3)
            ]
            plane["normal_ras"] = normal
            plane["u_axis_ras"] = u_axis
            plane["v_axis_ras"] = v_axis
            plane["size_mm"] = size_mm
            plane["relative_definition"] = relative_definition
            plane["relative_to"] = "resolved_model_bbox"
        self._append_log("Resolved bbox-relative workflow planes against the current preprocessed mask.\n")
        return resolved

    def _resolve_reference_space_editor_for_current_sample(self, editor):
        if not self._editor_needs_reference_resolution(editor):
            return editor
        config = self._active_workflow_config()
        model = config.get("model", {}) if isinstance(config, dict) else {}
        registration = model.get("registration", {}) if isinstance(model, dict) else {}
        if not isinstance(registration, dict) or not registration.get("enabled", False):
            return editor
        reference = self._current_workflow_reference_points()
        if not reference:
            return editor
        volume_node = self._volume()
        if volume_node is None:
            return editor
        max_points = int(registration.get("max_points", 8000))
        iterations = int(registration.get("iterations", 50))
        tolerance = float(registration.get("tolerance", 1.0e-4))
        active_values = self._selected_icp_target_values() or _workflow_registration_active_values(model, registration)
        mask_like_node = self.maskSelector.currentNode() or volume_node
        reference_points = read_reference_points(
            reference,
            max_points=max_points,
            coordinate_system=self._current_workflow_reference_coordinate_system(),
        )
        if self._current_inputs_are_icp_aligned_reference_frame(reference):
            info = getattr(self, "_lastIcpAlignment", {})
            if isinstance(info, dict) and isinstance(info.get("reference_points"), np.ndarray):
                reference_points = info["reference_points"]
            resolved = resolve_reference_space_editor(
                editor,
                reference_points=reference_points,
                sample_points=reference_points,
                iterations=1,
                tolerance=tolerance,
                allow_scale=False,
                snap_planes=False,
                prealign_reference=False,
            )
            self._append_log("Resolved reference-space planes in the ICP-aligned reference frame.\n")
            return resolved
        sample_points = _sample_reference_points_from_mask_like(
            mask_like_node,
            volume_node,
            max_points=max_points,
            active_values=active_values,
        )
        if sample_points.size == 0:
            raise ValueError("Cannot replay workflow planes: current mask/image has no non-zero surface points.")
        reference_points, scaling_meta = scale_reference_points_preserving_pose(
            reference_points=reference_points,
            sample_points=sample_points,
            registration_config=registration,
        )
        resolved = resolve_reference_space_editor(
            editor,
            reference_points=reference_points,
            sample_points=sample_points,
            iterations=iterations,
            tolerance=tolerance,
            allow_scale=False,
            snap_planes=self._current_workflow_registration_flag("snap_planes", False),
            prealign_reference=_enabled_value(registration.get("prealign_reference_to_sample", False)),
        )
        status = resolved.get("registration", {})
        if isinstance(status, dict):
            status["reference_scaling"] = scaling_meta
            self._append_log(
                "Replayed reference-space planes with ICP "
                f"(iterations={int(status.get('iterations', 0))}, "
                f"mean_distance={float(status.get('mean_distance', 0.0)):.3g} mm).\n"
            )
        return resolved

    def _editor_needs_reference_resolution(self, editor):
        planes = (editor or {}).get("planes", []) if isinstance(editor, dict) else []
        if not isinstance(planes, list):
            return False
        for plane in planes:
            if not isinstance(plane, dict):
                continue
            if plane.get("reference_space", False) or plane.get("derive_from"):
                return True
        return False

    def reset_scene(self):
        signal_states = self._begin_input_node_update_suppression()
        try:
            try:
                slicer.util.setSliceViewerLayers(background=None, foreground=None, label=None)
            except Exception:
                pass
            for selector in (self.imageSelector, self.maskSelector):
                node = selector.currentNode()
                if node is not None and _is_generated_parosol_node(node):
                    selector.setCurrentNode(None)
            self.diskLabelSelector.setCurrentNode(None)
            self.bcLabelSelector.setCurrentNode(None)
            self.topDiskPlane = None
            self.bottomDiskPlane = None
            self.fixedPlane = None
            self.loadedPlane = None
            for row_data in getattr(self, "contactPlaneRows", []):
                try:
                    self.logic.remove_node(row_data.get("plane"))
                except Exception:
                    pass
            self.contactPlaneRows = []
            self.planeTable.setRowCount(0)
            self.loadTable.setRowCount(0)
            self._delete_bc_arrow_models()
            self.resultText.clear()
            removed = self.logic.clear_generated_nodes()
            self._append_log(
                f"Cleaned {removed} generated ParOSol scene node{'s' if removed != 1 else ''}.\n"
            )
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
        finally:
            self._end_input_node_update_suppression(signal_states)

    def _add_contact_plane(self, checked=False, *, name=None, axis="z", normal="-", bc="Loaded", value="1.0"):
        row = self._table_row_count()
        self.planeTable.insertRow(row)

        axis = str(axis).lower()
        normal = str(normal)
        bc = str(bc)
        name = name or f"Plane {row + 1}"
        normal_sign = 1 if normal == "+" else -1
        plane = self.logic.create_axis_plane(
            f"ParOSol_{name}_plane",
            self._volume(),
            axis=axis,
            normal_sign=normal_sign,
            bounds_node=self._bounds_node(),
        )

        row_data = {"plane": plane, "snap_guard": False}
        self.contactPlaneRows.append(row_data)
        try:
            row_data["observer"] = plane.AddObserver(
                vtk.vtkCommand.ModifiedEvent, self._on_contact_plane_modified
            )
        except Exception:
            row_data["observer"] = None

        self.planeTable.setItem(row, 0, qt.QTableWidgetItem(name))
        axis_combo = self._combo(["x", "y", "z"], axis)
        self.planeTable.setCellWidget(row, 1, axis_combo)
        normal_combo = self._combo(["+", "-"], normal)
        normal_combo.currentTextChanged.connect(
            lambda _text, widget=normal_combo: self._apply_plane_row_geometry(self._row_for_widget(widget))
        )
        self.planeTable.setCellWidget(row, 2, normal_combo)
        self.planeTable.setCellWidget(
            row,
            3,
            self._combo(
                ["Material disks", "Connective disk", "Bone surface"],
                self._widget_text(self.contactModelBox, "Material disks"),
            ),
        )
        bc_type = "Fixed" if bc == "Fixed" else "Dirichlet"
        self.planeTable.setCellWidget(row, 4, self._combo(["project", "intersect"], "project"))
        self.planeTable.setCellWidget(row, 5, self._combo(["None", "Fixed", "Dirichlet", "Neumann"], bc_type))
        self.planeTable.setCellWidget(
            row,
            6,
            self._combo(
                ["Plane normal", "World R/L", "World A/P", "World S/I", "Plane U", "Plane V", "Custom vector"],
                "Plane normal",
            ),
        )
        self.planeTable.setItem(row, 7, qt.QTableWidgetItem(""))
        self.planeTable.setItem(row, 8, qt.QTableWidgetItem(""))
        self.planeTable.setItem(row, 9, qt.QTableWidgetItem(""))
        self.planeTable.setItem(row, 10, qt.QTableWidgetItem(str(value)))
        self.planeTable.setCellWidget(row, 11, self._combo(["anatomy", "rectangle", "oval", "hex"], "anatomy"))
        self.planeTable.setCellWidget(row, 12, self._double_spin_cell(3.0, minimum=0.1, maximum=1000.0, step=0.5, suffix=" mm"))
        self.planeTable.setCellWidget(row, 13, self._double_spin_cell(2.0, minimum=0.0, maximum=1000.0, step=0.5, suffix=" mm"))
        self.planeTable.setItem(row, 14, qt.QTableWidgetItem("12.0"))
        self.planeTable.setCellWidget(row, 15, self._combo(["yes", "no"], "yes"))
        self.planeTable.setCellWidget(row, 16, self._target_label_combo(_first_int_text(self._current_workflow_disk_projection_values())))
        self.planeTable.setItem(row, 17, qt.QTableWidgetItem("3000.0"))
        self.planeTable.setItem(row, 18, qt.QTableWidgetItem("0.3"))
        axis_combo.currentTextChanged.connect(
            lambda _text, widget=axis_combo: self._apply_plane_row_geometry(self._row_for_widget(widget))
        )
        self._add_load_row_for_contact(row, name=name, bc_type=bc_type, value=value)
        try:
            self.planeTable.selectRow(row)
        except Exception:
            pass
        self._update_selected_plane_status()

    def _add_load_row_for_contact(self, contact_row, *, name, bc_type, value):
        self._ensure_load_table_label_column()
        row = self._load_row_count()
        self.loadTable.insertRow(row)
        self.loadTable.setItem(row, 0, qt.QTableWidgetItem(str(name)))
        mode_combo = self._combo(
            [
                "None",
                "Fixed",
                "Displacement",
                "Force",
                "Bending",
                "Bending symmetric",
                "Torsion",
                "Load history 3",
                "Load history 6",
            ],
            "Fixed" if bc_type == "Fixed" else "Displacement",
            dirty_callback=self._mark_workflow_replay_loads_dirty,
        )
        mode_combo.currentTextChanged.connect(
            lambda _text, widget=mode_combo: self._on_load_mode_changed(widget)
        )
        self.loadTable.setCellWidget(
            row,
            1,
            mode_combo,
        )
        self.loadTable.setCellWidget(
            row,
            2,
            self._combo(
                ["Plane normal", "World R/L", "World A/P", "World S/I", "Plane U", "Plane V", "Custom vector"],
                "Plane normal",
                dirty_callback=self._mark_workflow_replay_loads_dirty,
            ),
        )
        self.loadTable.setItem(row, 3, qt.QTableWidgetItem(""))
        self.loadTable.setItem(row, 4, qt.QTableWidgetItem(""))
        self.loadTable.setItem(row, 5, qt.QTableWidgetItem(""))
        self.loadTable.setItem(row, 6, qt.QTableWidgetItem(str(value)))
        self.loadTable.setCellWidget(row, 7, self._combo(["mm"], "mm"))
        self.loadTable.setCellWidget(row, LOAD_FIXED_DOFS_COLUMN, self._fixed_dofs_widget())
        self._update_load_row_units(row)
        try:
            self.contactPlaneRows[int(contact_row)]["load_row"] = row
        except Exception:
            pass
        self._set_load_table_nodeset_label(contact_row)

    def _fixed_dofs_widget(self, dofs=None):
        widget = qt.QWidget()
        layout = qt.QHBoxLayout(widget)
        try:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
        except Exception:
            pass
        selected = set(_valid_fixed_dofs(dofs) or ["x", "y", "z"])
        for axis in ("x", "y", "z"):
            checkbox = qt.QCheckBox(axis.upper())
            checkbox.checked = axis in selected
            checkbox.toolTip = f"Constrain model {axis.upper()} displacement DOF to zero."
            checkbox.toggled.connect(lambda _checked, owner=widget: self._on_fixed_dofs_changed(owner))
            layout.addWidget(checkbox)
        layout.addStretch(1)
        return widget

    def _fixed_dof_checkboxes(self, widget):
        if widget is None:
            return []
        try:
            return list(widget.findChildren(qt.QCheckBox))
        except Exception:
            return []

    def _set_load_fixed_dofs(self, row, dofs):
        if row is None or row < 0 or row >= self._load_row_count():
            return
        widget = self.loadTable.cellWidget(row, LOAD_FIXED_DOFS_COLUMN)
        if widget is None:
            widget = self._fixed_dofs_widget(dofs)
            self.loadTable.setCellWidget(row, LOAD_FIXED_DOFS_COLUMN, widget)
        selected = set(_valid_fixed_dofs(dofs) or ["x", "y", "z"])
        for checkbox in self._fixed_dof_checkboxes(widget):
            axis = str(self._qt_value(checkbox.text)).strip().lower()
            try:
                checkbox.blockSignals(True)
            except Exception:
                pass
            checkbox.checked = axis in selected
            try:
                checkbox.blockSignals(False)
            except Exception:
                pass
        self._update_fixed_dofs_widget_enabled(row)

    def _load_fixed_dofs(self, row):
        if row is None or row < 0 or row >= self._load_row_count():
            return None
        widget = self.loadTable.cellWidget(row, LOAD_FIXED_DOFS_COLUMN)
        selected = []
        for checkbox in self._fixed_dof_checkboxes(widget):
            axis = str(self._qt_value(checkbox.text)).strip().lower()
            if axis in {"x", "y", "z"} and bool(getattr(checkbox, "checked", False)):
                selected.append(axis)
        return _valid_fixed_dofs(selected)

    def _update_fixed_dofs_widget_enabled(self, row):
        if row is None or row < 0 or row >= self._load_row_count():
            return
        widget = self.loadTable.cellWidget(row, LOAD_FIXED_DOFS_COLUMN)
        if widget is None:
            widget = self._fixed_dofs_widget()
            self.loadTable.setCellWidget(row, LOAD_FIXED_DOFS_COLUMN, widget)
        enabled = self._widget_text(self.loadTable.cellWidget(row, 1), "Displacement") == "Fixed"
        try:
            widget.setEnabled(bool(enabled))
        except Exception:
            widget.enabled = bool(enabled)

    def _on_fixed_dofs_changed(self, _widget=None):
        try:
            self._mark_workflow_replay_loads_dirty()
        except Exception:
            pass

    def _combo(self, values, current, *, dirty_callback=None):
        combo = qt.QComboBox()
        combo.addItems(list(values))
        if current in values:
            combo.setCurrentText(current)
        callback = dirty_callback or self._mark_workflow_replay_editor_dirty
        combo.currentTextChanged.connect(lambda _text, callback=callback: callback())
        return combo

    def _double_spin_cell(self, value, *, minimum, maximum, step=1.0, suffix=""):
        spin = qt.QDoubleSpinBox()
        spin.minimum = float(minimum)
        spin.maximum = float(maximum)
        spin.decimals = 2
        spin.singleStep = float(step)
        spin.value = float(value)
        spin.suffix = suffix
        spin.valueChanged.connect(lambda _value: self._mark_workflow_replay_editor_dirty())
        return spin

    def _qt_value(self, value):
        return value() if callable(value) else value

    def _widget_text(self, widget, default=""):
        if widget is None:
            return default
        try:
            return str(self._qt_value(widget.currentText))
        except Exception:
            return default

    def _item_text(self, row, column, default=""):
        widget = self.planeTable.cellWidget(row, column)
        if widget is not None and hasattr(widget, "value"):
            try:
                return str(float(self._qt_value(widget.value)))
            except Exception:
                pass
        item = self.planeTable.item(row, column)
        if item is None:
            return default
        try:
            return str(self._qt_value(item.text))
        except Exception:
            return default

    def _table_row_count(self):
        try:
            return int(self._qt_value(self.planeTable.rowCount))
        except Exception:
            return 0

    def _contact_row_count(self):
        return self._table_row_count()

    def _load_row_count(self):
        try:
            return int(self._qt_value(self.loadTable.rowCount))
        except Exception:
            return 0

    def _ensure_load_table_label_column(self):
        if not hasattr(self, "loadTable") or self.loadTable is None:
            return
        try:
            column_count = int(self._qt_value(self.loadTable.columnCount))
        except Exception:
            return
        if column_count < LOAD_TABLE_COLUMN_COUNT:
            self.loadTable.setColumnCount(LOAD_TABLE_COLUMN_COUNT)
        try:
            fixed_header_item = self.loadTable.horizontalHeaderItem(LOAD_FIXED_DOFS_COLUMN)
            fixed_header_text = (
                str(self._qt_value(fixed_header_item.text)) if fixed_header_item is not None else ""
            )
        except Exception:
            fixed_header_text = ""
        if not fixed_header_text:
            self.loadTable.setHorizontalHeaderItem(
                LOAD_FIXED_DOFS_COLUMN,
                qt.QTableWidgetItem("Fixed DOFs"),
            )
        try:
            header_item = self.loadTable.horizontalHeaderItem(LOAD_NODESET_LABEL_COLUMN)
            header_text = str(self._qt_value(header_item.text)) if header_item is not None else ""
        except Exception:
            header_text = ""
        if not header_text:
            self.loadTable.setHorizontalHeaderItem(
                LOAD_NODESET_LABEL_COLUMN,
                qt.QTableWidgetItem("Label"),
            )

    def _contact_row_for_load_row(self, load_row):
        if load_row is None:
            return None
        try:
            load_row = int(load_row)
        except Exception:
            return None
        for row, row_data in enumerate(getattr(self, "contactPlaneRows", [])):
            try:
                if int(row_data.get("load_row")) == load_row:
                    return row
            except Exception:
                continue
        return load_row if 0 <= load_row < len(getattr(self, "contactPlaneRows", [])) else None

    def _set_load_table_nodeset_label(self, contact_row):
        if contact_row is None:
            return
        try:
            contact_row = int(contact_row)
        except Exception:
            return
        load_row = self._load_row_for_contact(contact_row)
        if load_row is None:
            return
        self._ensure_load_table_label_column()
        try:
            spec = self._contact_row_spec(contact_row)
            label = self._bc_preview_label(spec, contact_row)
        except Exception:
            label = ""
        item = qt.QTableWidgetItem("" if label in (None, "") else str(label))
        item.setToolTip("Contact-region label exported to ParOSol.")
        try:
            item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        except Exception:
            pass
        self.loadTable.setItem(load_row, LOAD_NODESET_LABEL_COLUMN, item)

    def _update_load_table_nodeset_labels(self):
        self._ensure_load_table_label_column()
        for row in range(len(getattr(self, "contactPlaneRows", []))):
            self._set_load_table_nodeset_label(row)

    def _row_for_widget(self, widget):
        try:
            columns = int(self._qt_value(self.planeTable.columnCount))
        except Exception:
            columns = 0
        for row in range(self._table_row_count()):
            for column in range(columns):
                if self.planeTable.cellWidget(row, column) is widget:
                    return row
        return None

    def _load_row_for_widget(self, widget):
        try:
            columns = int(self._qt_value(self.loadTable.columnCount))
        except Exception:
            columns = 0
        for row in range(self._load_row_count()):
            for column in range(columns):
                if self.loadTable.cellWidget(row, column) is widget:
                    return row
        return None

    def _on_load_mode_changed(self, widget):
        load_row = self._load_row_for_widget(widget)
        self._update_load_row_units(load_row)
        self._set_load_table_nodeset_label(self._contact_row_for_load_row(load_row))

    def _update_load_row_units(self, row):
        if row is None or row < 0 or row >= self._load_row_count():
            return
        mode = self._widget_text(self.loadTable.cellWidget(row, 1), "Displacement")
        previous = self._widget_text(self.loadTable.cellWidget(row, 7), "")
        units = {
            "None": [""],
            "Fixed": [""],
            "Displacement": ["mm", "%"],
            "Force": ["N"],
            "Bending": ["deg"],
            "Bending symmetric": ["deg"],
            "Torsion": ["deg"],
            "Load history 3": ["unit"],
            "Load history 6": ["unit"],
        }.get(mode, ["mm"])
        unit_combo = self.loadTable.cellWidget(row, 7)
        if unit_combo is None:
            unit_combo = qt.QComboBox()
            unit_combo.currentTextChanged.connect(
                lambda _text: self._mark_workflow_replay_loads_dirty()
            )
            self.loadTable.setCellWidget(row, 7, unit_combo)
        try:
            unit_combo.clear()
            unit_combo.addItems(units)
            unit_combo.setCurrentText(previous if previous in units else units[0])
        except Exception:
            pass
        self._update_fixed_dofs_widget_enabled(row)

    def _selected_contact_row(self):
        rows = self.planeTable.selectionModel().selectedRows()
        if rows:
            return int(rows[0].row())
        try:
            current = int(self._qt_value(self.planeTable.currentRow))
            return current if current >= 0 else None
        except Exception:
            return None

    def _selected_contact_plane(self):
        row = self._selected_contact_row()
        if row is None or row >= len(self.contactPlaneRows):
            return None
        return self.contactPlaneRows[row].get("plane")

    def _row_for_plane(self, plane):
        for row, row_data in enumerate(self.contactPlaneRows):
            if row_data.get("plane") is plane:
                return row
        return None

    def _on_contact_plane_modified(self, caller, _event):
        row = self._row_for_plane(caller)
        if row is None:
            return
        self._mark_workflow_replay_editor_dirty()
        self._snap_contact_plane_rotation(row)
        if row == self._selected_contact_row():
            self._update_selected_plane_status()

    def _snap_contact_plane_rotation(self, row):
        if row is None or row < 0 or row >= len(self.contactPlaneRows):
            return
        row_data = self.contactPlaneRows[row]
        if row_data.get("snap_guard"):
            return
        plane = row_data.get("plane")
        if plane is None:
            return
        row_data["snap_guard"] = True
        try:
            self.logic.snap_plane_rotation(plane, step_degrees=5.0)
        finally:
            row_data["snap_guard"] = False

    def _contact_row_spec(self, row):
        rows = getattr(self, "contactPlaneRows", [])
        row_data = rows[int(row)] if int(row) < len(rows) else {}
        axis = self._widget_text(self.planeTable.cellWidget(row, 1), "z")
        normal = self._widget_text(self.planeTable.cellWidget(row, 2), "-")
        contact = self._widget_text(self.planeTable.cellWidget(row, 3), "Material disks")
        if contact == "PMMA caps":
            contact = "Material disks"
        surface_mode = _projection_mode(
            self._widget_text(self.planeTable.cellWidget(row, 4), "project")
        )
        if contact == "Bone surface" and surface_mode == "project_bounded" and self._is_xtremect_profile():
            surface_mode = "intersect"
        load = self._load_row_spec(row)
        bc_type = load["bc_type"]
        direction = load["direction"]
        shape = self._widget_text(self.planeTable.cellWidget(row, 11), "anatomy")
        valid_bc_types = {
            "None",
            "Fixed",
            "Dirichlet",
            "Neumann",
            "Displacement",
            "Force",
            "Bending",
            "Bending symmetric",
            "Torsion",
            "Load history 3",
            "Load history 6",
        }
        valid_directions = {
            "normal",
            "x",
            "y",
            "z",
            "plane_u",
            "plane_v",
            "vector",
            "Plane normal",
            "World R/L",
            "World A/P",
            "World S/I",
            "Plane U",
            "Plane V",
            "Custom vector",
        }
        return {
            "name": self._item_text(row, 0, f"Plane {row + 1}"),
            "axis": axis.lower() if axis.lower() in {"x", "y", "z"} else "z",
            "normal": normal if normal in {"+", "-"} else "-",
            "contact": contact if contact in {"Material disks", "Connective disk", "Bone surface"} else "Material disks",
            "surface_mode": surface_mode if surface_mode in {"project_bounded", "project_global", "intersect"} else "project_bounded",
            "bc_type": bc_type if bc_type in valid_bc_types else "Dirichlet",
            "direction": direction if direction in valid_directions else "normal",
            "x": load["x"],
            "y": load["y"],
            "z": load["z"],
            "value": load["value"],
            "units": load["units"],
            "reference_nodeset": load.get("reference_nodeset"),
            "fixed_dofs": load.get("fixed_dofs") or row_data.get("fixed_dofs"),
            "shape": shape if shape in {"anatomy", "rectangle", "oval", "round", "square", "hex"} else "anatomy",
            "anatomy_constrained": bool(row_data.get("anatomy_constrained", False)),
            "thickness": float(self._item_text(row, 12, "3.0")),
            "intrusion": float(self._item_text(row, 13, "2.0")),
            "radius": float(self._item_text(row, 14, "12.0")),
            "use_plane_size": self._widget_text(self.planeTable.cellWidget(row, 15), "yes") == "yes",
            "disk_label": int(
                row_data.get(
                    "disk_label",
                    _generated_boundary_label_for_row(bc_type, row),
                )
            ),
            "nodeset_label": row_data.get("nodeset_label"),
            "disk_target_values": _combo_selected_int_tuple(self.planeTable.cellWidget(row, 16)),
            "disk_e": float(self._item_text(row, 17, "3000.0")),
            "disk_nu": float(self._item_text(row, 18, "0.3")),
        }

    def _applied_profile_key(self):
        profile = str(getattr(self, "_appliedProfileName", "") or self._widget_text(getattr(self, "profileBox", None), "")).strip()
        token = Path(profile).name if ("/" in profile or "\\" in profile) else profile
        return token.strip().lower()

    def _is_xtremect_profile(self):
        return self._applied_profile_key() in {
            "xtremecti",
            "xtremectii",
            "xtremecti_loadhistory_3",
            "xtremecti_loadhistory_6",
            "xtremectii_loadhistory_3",
            "xtremectii_loadhistory_6",
        }

    def _is_load_history_profile(self):
        return self._applied_profile_key() in {
            "load_history_3",
            "load_history_6",
            "load-history-3",
            "load-history-6",
            "xtremecti_loadhistory_3",
            "xtremecti_loadhistory_6",
            "xtremectii_loadhistory_3",
            "xtremectii_loadhistory_6",
        }

    def _selected_load_history_modes(self):
        modes = []
        for row in range(self._table_row_count()):
            try:
                spec = self._contact_row_spec(row)
            except Exception:
                continue
            mode = str(spec.get("bc_type", "")).strip()
            if mode.startswith("Load history"):
                modes.append(mode)
        return modes

    def _uses_load_history_bc(self):
        return bool(self._selected_load_history_modes())

    def _load_row_for_contact(self, contact_row):
        if contact_row is None or contact_row < 0 or contact_row >= len(self.contactPlaneRows):
            return None
        row = self.contactPlaneRows[int(contact_row)].get("load_row")
        if row is None or int(row) >= self._load_row_count():
            return int(contact_row) if int(contact_row) < self._load_row_count() else None
        return int(row)

    def _load_row_spec(self, contact_row):
        load_row = self._load_row_for_contact(contact_row)
        if load_row is None:
            return {
                "bc_type": "Displacement",
                "direction": "Plane normal",
                "x": "",
                "y": "",
                "z": "",
                "value": "1.0",
                "units": "mm",
            }
        mode = self._widget_text(self.loadTable.cellWidget(load_row, 1), "Displacement")
        direction = self._widget_text(self.loadTable.cellWidget(load_row, 2), "Plane normal")
        mode_map = {"Displacement": "Dirichlet", "Force": "Neumann"}
        return {
            "bc_type": mode_map.get(mode, mode),
            "direction": direction,
            "x": self._load_item_text(load_row, 3, ""),
            "y": self._load_item_text(load_row, 4, ""),
            "z": self._load_item_text(load_row, 5, ""),
            "value": self._load_item_text(load_row, 6, "0.0"),
            "units": self._widget_text(self.loadTable.cellWidget(load_row, 7), "mm"),
            "fixed_dofs": self._load_fixed_dofs(load_row),
            "reference_nodeset": self.contactPlaneRows[int(contact_row)].get(
                "reference_nodeset"
            )
            if contact_row is not None and int(contact_row) < len(self.contactPlaneRows)
            else None,
        }

    def _load_item_text(self, row, column, default=""):
        item = self.loadTable.item(row, column)
        if item is None:
            return default
        try:
            return str(self._qt_value(item.text))
        except Exception:
            return default

    def _interactive_load_settings(self):
        for row, row_data in enumerate(self.contactPlaneRows):
            spec = self._contact_row_spec(row)
            if spec["bc_type"] in {"None", "Fixed"}:
                continue
            if spec["bc_type"] in {"Bending", "Bending symmetric", "Torsion"}:
                raise ValueError(
                    f"{spec['bc_type']} requires interactive nodeset config export. "
                    "Run Create Regions, then export the model/config from the interactive editor."
                )
            plane = row_data.get("plane")
            direction = self._model_axis_vector_from_ras(
                self._bc_direction_vector(plane, spec)
            )
            raw_value = _load_value_number(spec)
            sign = -1.0 if raw_value < 0 else 1.0
            direction = tuple(sign * float(component) for component in direction)
            magnitude = abs(raw_value)
            loaded_label = 3 if spec["bc_type"] == "Neumann" else 2
            if spec["bc_type"] == "Neumann":
                return magnitude, 1.0, direction, loaded_label
            displacement_mm = magnitude
            self._append_log(
                f"{spec['name']}: prescribed displacement {raw_value:g} mm.\n"
            )
            return None, displacement_mm, direction, loaded_label
        return None, 1.0, self._model_axis_vector_from_ras((0.0, 0.0, -1.0)), 2

    def _interactive_nodeset_config(self, nodeset_path, disk_label_path=None):
        nodesets = {}
        load_case = {"type": "nodeset", "fixed": [], "prescribed": [], "loaded": []}
        for row, row_data in enumerate(self.contactPlaneRows):
            spec = self._contact_row_spec(row)
            if spec["bc_type"] == "None" or spec["contact"] == "Connective disk":
                continue
            name = _safe_identifier(spec["name"] or f"nodeset_{row + 1}")
            label = self._bc_preview_label(spec, row)
            if spec["contact"] == "Material disks" and disk_label_path is not None:
                nodeset_image = disk_label_path
                label = int(spec["disk_label"])
                selection = "outer_face_nodes"
            else:
                nodeset_image = nodeset_path
                selection = (
                    "interface_nodes"
                    if spec["contact"] == "Bone surface"
                    else "surface_nodes"
                )
            nodesets[name] = {
                "type": "label_image",
                "image": str(nodeset_image),
                "label": int(label),
                "selection": selection,
            }
            if spec["bc_type"] == "Fixed":
                fixed_dofs = _valid_fixed_dofs(spec.get("fixed_dofs"))
                if fixed_dofs is None:
                    fixed_dofs = ["x", "y", "z"]
                load_case["fixed"].append(
                    {"nodeset": name, "dofs": fixed_dofs, "value": 0.0}
                )
                continue
            if str(spec["bc_type"]).startswith("Load history"):
                continue
            plane = row_data.get("plane")
            direction = self._model_axis_vector_from_ras(
                self._bc_direction_vector(plane, spec)
            )
            raw_value = _load_value_number(spec)
            sign = -1.0 if raw_value < 0 else 1.0
            magnitude = abs(raw_value)
            if spec["bc_type"] == "Neumann":
                for axis, component in self._nonzero_components(
                    tuple(sign * float(value) for value in direction)
                ):
                    load_case["loaded"].append(
                        {
                            "nodeset": name,
                            "dof": axis,
                            "value": magnitude * float(component),
                            "units": "N",
                            "distribute": True,
                        }
                    )
                continue
            if spec["bc_type"] == "Bending":
                normal_axis, normal_sign = self._dominant_axis_with_sign(
                    self._model_axis_vector_from_ras(_plane_normal_world(plane))
                )
                load_case["prescribed"].append(
                    {
                        "nodeset": name,
                        "kind": "bending",
                        "mode": "linear",
                        "dof": normal_axis,
                        "value": raw_value * normal_sign,
                        "units": "deg",
                        "gradient_axis": self._dominant_axis(
                            self._model_axis_vector_from_ras(
                                self._bending_gradient_axis(plane, spec)
                            )
                        ),
                        "center": "centroid",
                    }
                )
                continue
            if spec["bc_type"] == "Bending symmetric":
                normal_axis, normal_sign = self._dominant_axis_with_sign(
                    self._model_axis_vector_from_ras(_plane_normal_world(plane))
                )
                load_case["prescribed"].append(
                    {
                        "nodeset": name,
                        "kind": "bending",
                        "mode": "symmetric",
                        "dof": normal_axis,
                        "value": raw_value * normal_sign,
                        "units": "deg",
                        "gradient_axis": self._dominant_axis(
                            self._model_axis_vector_from_ras(
                                self._bending_gradient_axis(plane, spec)
                            )
                        ),
                        "center": "centroid",
                        "neutral_fraction": 0.5,
                    }
                )
                continue
            if spec["bc_type"] == "Torsion":
                normal_axis, normal_sign = self._dominant_axis_with_sign(
                    self._model_axis_vector_from_ras(_plane_normal_world(plane))
                )
                load_case["prescribed"].append(
                    {
                        "nodeset": name,
                        "kind": "torsion",
                        "axis": normal_axis,
                        "value": raw_value * normal_sign,
                        "units": "deg",
                        "center": "centroid",
                    }
                )
                continue
            signed_direction = tuple(sign * float(value) for value in direction)
            displacement_value = self._displacement_nodeset_value(spec, magnitude)
            self._append_log(
                f"{spec['name']}: prescribed displacement {raw_value:g} {spec['units'] or 'mm'}.\n"
            )
            nonzero_components = self._nonzero_components(signed_direction)
            if self._uses_xtremect_axial_lateral_constraints(spec, nonzero_components):
                load_axis = nonzero_components[0][0]
                lateral_dofs = [axis for axis in ("x", "y", "z") if axis != load_axis]
                load_case["fixed"].append(
                    {"nodeset": name, "dofs": lateral_dofs, "value": 0.0}
                )
            for axis, component in nonzero_components:
                value = self._scale_displacement_nodeset_value(
                    displacement_value,
                    float(component),
                )
                reference_length_mm = self._nodeset_reference_length_mm(
                    name,
                    spec.get("reference_nodeset"),
                    axis,
                )
                load_case["prescribed"].append(
                    {
                        "nodeset": name,
                        "dof": axis,
                        "value": value,
                        "units": spec["units"] if str(spec["units"]).strip() == "%" else "mm",
                        **(
                            {"reference_length_mm": float(reference_length_mm)}
                            if reference_length_mm is not None
                            and isinstance(value, str)
                            and value.strip().endswith("%")
                            else {}
                        ),
                        **(
                            {
                                "reference_nodeset": _safe_identifier(
                                    str(spec["reference_nodeset"])
                                )
                            }
                            if spec.get("reference_nodeset")
                            else {}
                        ),
                    }
                )
        for key in ("fixed", "prescribed", "loaded"):
            if not load_case[key]:
                load_case.pop(key)
        return nodesets, load_case

    def _validate_interactive_nodeset_export_labels(
        self,
        nodeset_specs,
        nodeset_node,
        disk_node,
        *,
        nodeset_path,
        disk_label_path=None,
    ):
        path_to_node = {}
        if nodeset_node is not None and nodeset_path is not None:
            try:
                path_to_node[str(Path(nodeset_path).expanduser().resolve())] = nodeset_node
            except Exception:
                path_to_node[str(nodeset_path)] = nodeset_node
        if disk_node is not None and disk_label_path is not None:
            try:
                path_to_node[str(Path(disk_label_path).expanduser().resolve())] = disk_node
            except Exception:
                path_to_node[str(disk_label_path)] = disk_node

        missing = []
        synced_paths = set()
        for name, spec in (nodeset_specs or {}).items():
            if not isinstance(spec, dict):
                continue
            image = spec.get("image")
            if image is None:
                continue
            image_path = Path(str(image)).expanduser()
            try:
                resolved_image = str(image_path.resolve())
                node = path_to_node.get(resolved_image)
            except Exception:
                resolved_image = str(image)
                node = path_to_node.get(str(image))
            if node is None:
                continue
            try:
                label = int(spec.get("label", 0))
            except Exception:
                label = 0
            try:
                node_count = int(np.count_nonzero(slicer.util.arrayFromVolume(node) == label))
            except Exception:
                node_count = 0
            if (
                label > 0
                and node_count > 0
                and resolved_image not in synced_paths
                and hasattr(self, "logic")
            ):
                try:
                    self._rewrite_exported_labelmap_from_node(node, image_path)
                    synced_paths.add(resolved_image)
                    self._append_log(
                        f"Refreshed exported labelmap {image_path.name} from current Slicer node.\n"
                    )
                except Exception:
                    pass
            file_count = self._exported_label_count(image_path, label)
            if label <= 0 or file_count == 0:
                missing.append(f"{name} label {label} in {image_path.name}")

        if missing:
            raise ValueError(
                "Exported contact-region labelmap is missing required nodeset "
                f"label(s): {', '.join(missing)}. Recreate contact regions "
                "before running, or reload the SlicerParOSol module and apply the workflow again."
            )

    def _rewrite_exported_labelmap_from_node(self, node, image_path):
        image_path = Path(str(image_path)).expanduser()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        import SimpleITK as sitk

        array = np.asarray(slicer.util.arrayFromVolume(node), dtype=np.uint16)
        image = sitk.GetImageFromArray(array)
        copied_geometry = False
        if image_path.exists():
            try:
                reference = sitk.ReadImage(str(image_path))
                if tuple(reference.GetSize()) == tuple(image.GetSize()):
                    image.CopyInformation(reference)
                    copied_geometry = True
            except Exception:
                copied_geometry = False
        if not copied_geometry:
            self.logic.export_volume(node, image_path)
            try:
                reference = sitk.ReadImage(str(image_path))
                if tuple(reference.GetSize()) == tuple(image.GetSize()):
                    image.CopyInformation(reference)
                    copied_geometry = True
            except Exception:
                copied_geometry = False
        if not copied_geometry:
            raise ValueError(
                f"Could not refresh exported labelmap geometry for {image_path.name}."
            )
        sitk.WriteImage(image, str(image_path), True)
        return image_path

    def _exported_label_count(self, image, label):
        try:
            image_path = Path(str(image)).expanduser()
            if not image_path.exists():
                return 0
            import SimpleITK as sitk

            array = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
            return int(np.count_nonzero(np.asarray(array) == int(label)))
        except Exception:
            return 0

    def _exported_nonzero_labels(self, image):
        try:
            image_path = Path(str(image)).expanduser()
            if not image_path.exists():
                return []
            import SimpleITK as sitk

            array = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))))
            values = np.unique(array)
            return sorted(int(value) for value in values if int(value) != 0)
        except Exception:
            return []

    def _validate_exported_config_nodeset_files(self, config_path):
        import yaml

        config_path = Path(config_path).expanduser()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        nodesets = config.get("nodesets", {})
        if not isinstance(nodesets, dict):
            return
        missing = []
        for name, spec in nodesets.items():
            if not isinstance(spec, dict):
                continue
            if str(spec.get("type", "label_image")).strip().lower() != "label_image":
                continue
            image = spec.get("image")
            if not image:
                continue
            try:
                label = int(spec.get("label", 0))
            except Exception:
                label = 0
            image_path = Path(str(image)).expanduser()
            if not image_path.is_absolute():
                image_path = (config_path.parent / image_path).resolve()
            count = self._exported_label_count(image_path, label)
            if label > 0 and count == 0:
                present = self._exported_nonzero_labels(image_path)
                if present:
                    present_text = "present labels: " + ", ".join(str(value) for value in present[:12])
                    if len(present) > 12:
                        present_text += ", ..."
                else:
                    present_text = "file missing or contains no non-zero labels"
                missing.append(f"{name} label {label} in {image_path.name} ({present_text})")
        if missing:
            raise ValueError(
                "Exported ParOSol config references missing nodeset label(s): "
                + "; ".join(missing)
                + ". Recreate contact regions and rerun export."
            )

    def _uses_xtremect_axial_lateral_constraints(self, spec, nonzero_components):
        if not self._is_xtremect_profile():
            return False
        if str(spec.get("bc_type", "")).strip().lower() != "dirichlet":
            return False
        if len(nonzero_components) != 1:
            return False
        name = str(spec.get("name", "")).strip().lower()
        return name in {"top", "loaded", "superior"}

    def _displacement_nodeset_value(self, spec, magnitude):
        units = str(spec.get("units", "mm")).strip().lower()
        if units in {"%", "percent", "percentage"}:
            return f"{float(magnitude):g}%"
        return float(magnitude)

    def _scale_displacement_nodeset_value(self, value, component):
        if isinstance(value, str) and value.strip().endswith("%"):
            magnitude = float(value.strip()[:-1])
            return f"{magnitude * float(component):g}%"
        return float(value) * float(component)

    def _nonzero_components(self, direction):
        direction = _normalized(direction)
        pairs = [
            (axis, float(component))
            for axis, component in zip(("x", "y", "z"), direction)
            if abs(float(component)) > 1e-12
        ]
        return pairs or [("z", -1.0)]

    def _model_axis_vector_from_ras(self, ras_vector):
        ras_vector = _normalized(ras_vector)
        volume = self._volume()
        if volume is None:
            return tuple(ras_vector)
        ijk_to_ras = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(ijk_to_ras)
        axis_dirs = _ijk_axis_directions_ras(ijk_to_ras)
        model_vector = tuple(_dot(ras_vector, axis_dir) for axis_dir in axis_dirs)
        if _vector_length(model_vector) <= 1.0e-9:
            return tuple(ras_vector)
        return tuple(_normalized(model_vector))

    def _dominant_axis(self, vector):
        values = [abs(float(component)) for component in vector]
        return ("x", "y", "z")[int(np.argmax(values))]

    def _dominant_axis_with_sign(self, vector):
        index = int(np.argmax([abs(float(component)) for component in vector]))
        component = float(vector[index])
        sign = -1.0 if component < 0 else 1.0
        return ("x", "y", "z")[index], sign

    def _displacement_reference_length_mm(self, loaded_row, direction):
        loaded_plane = self.contactPlaneRows[loaded_row].get("plane")
        loaded_center = _plane_center(loaded_plane)
        unit_direction = _normalized(direction)
        candidates = []
        for row, row_data in enumerate(self.contactPlaneRows):
            if row == loaded_row:
                continue
            spec = self._contact_row_spec(row)
            if spec["bc_type"] != "Fixed":
                continue
            fixed_center = _plane_center(row_data.get("plane"))
            if loaded_center is None or fixed_center is None:
                continue
            separation = abs(_dot(_subtract(loaded_center, fixed_center), unit_direction))
            if separation > 1e-6:
                candidates.append(separation)
        if candidates:
            return float(max(candidates))
        return self._image_extent_along_direction_mm(unit_direction)

    def _nodeset_reference_length_mm(self, nodeset_name, reference_nodeset_name, axis):
        if not reference_nodeset_name:
            return None
        label_node = self.bcLabelSelector.currentNode() if hasattr(self, "bcLabelSelector") else None
        volume = self._volume()
        if label_node is None or volume is None:
            return None
        nodeset_label = None
        reference_label = None
        for row in range(len(self.contactPlaneRows)):
            spec = self._contact_row_spec(row)
            safe_name = _safe_identifier(spec["name"] or f"nodeset_{row + 1}")
            label = self._bc_preview_label(spec, row)
            if safe_name == str(nodeset_name):
                nodeset_label = int(label)
            if safe_name == _safe_identifier(str(reference_nodeset_name)):
                reference_label = int(label)
        if nodeset_label is None or reference_label is None:
            return None
        nodeset_center = self.logic.label_centroid_ras(label_node, nodeset_label, volume)
        reference_center = self.logic.label_centroid_ras(label_node, reference_label, volume)
        if nodeset_center is None or reference_center is None:
            return None
        axis_index = {"x": 0, "y": 1, "z": 2}.get(str(axis).strip().lower())
        if axis_index is None:
            return None
        ijk_to_ras = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(ijk_to_ras)
        axis_dir = _ijk_axis_directions_ras(ijk_to_ras)[axis_index]
        length = abs(_dot(_subtract(nodeset_center, reference_center), axis_dir))
        if length <= 1.0e-6:
            return None
        return float(length)

    def _image_extent_along_direction_mm(self, direction):
        volume = self._volume()
        if volume is None:
            return 1.0
        bounds = [0.0] * 6
        volume.GetRASBounds(bounds)
        extents = (
            abs(bounds[1] - bounds[0]),
            abs(bounds[3] - bounds[2]),
            abs(bounds[5] - bounds[4]),
        )
        extent = sum(float(extents[index]) * abs(float(direction[index])) for index in range(3))
        return max(float(extent), 1.0)

    def _bc_arrow_length_mm(self, volume=None):
        volume = volume or self._volume()
        if volume is None:
            return 10.0
        bounds = [0.0] * 6
        volume.GetRASBounds(bounds)
        max_extent = max(
            abs(bounds[1] - bounds[0]),
            abs(bounds[3] - bounds[2]),
            abs(bounds[5] - bounds[4]),
        )
        return max(5.0, 0.08 * float(max_extent))

    def _load_arrow_display_scale(self):
        try:
            return max(0.1, float(self.loadArrowScaleSpin.value))
        except Exception:
            return 1.0

    def _intersect_preview_arrow_length_mm(self):
        volume = self._volume()
        if volume is None:
            return 1.0
        try:
            spacing = [abs(float(value)) for value in volume.GetSpacing()]
        except Exception:
            spacing = [1.0, 1.0, 1.0]
        voxel = min([value for value in spacing if value > 0] or [1.0])
        bounds = [0.0] * 6
        volume.GetRASBounds(bounds)
        max_extent = max(
            abs(bounds[1] - bounds[0]),
            abs(bounds[3] - bounds[2]),
            abs(bounds[5] - bounds[4]),
        )
        return max(8.0 * voxel, min(6.0, 0.04 * float(max_extent)))

    def _delete_bc_arrow_models(self):
        for node in self.bcArrowNodes:
            self.logic.remove_node(node)
        self.bcArrowNodes = []
        for node in self.bcMarkerNodes:
            self.logic.remove_node(node)
        self.bcMarkerNodes = []
        for index in reversed(range(slicer.mrmlScene.GetNumberOfNodes())):
            node = slicer.mrmlScene.GetNthNode(index)
            if node is not None and str(node.GetName()).startswith("ParOSol_BC_arrow"):
                self.logic.remove_node(node)
            if node is not None and str(node.GetName()).startswith("ParOSol_BC_marker"):
                self.logic.remove_node(node)

    def _nodeset_label_debug_summary(self, nodeset_labelmap, label):
        if nodeset_labelmap is None:
            return "no nodeset labelmap"
        try:
            array = np.asarray(slicer.util.arrayFromVolume(nodeset_labelmap))
        except Exception as exc:
            return f"unreadable nodeset labelmap ({exc})"
        indices = np.argwhere(array == int(label))
        if indices.size == 0:
            return f"label {int(label)} count=0"
        mins = indices.min(axis=0)
        maxs = indices.max(axis=0)
        return (
            f"label {int(label)} count={indices.shape[0]} "
            f"zyx=({int(mins[0])}:{int(maxs[0])}, "
            f"{int(mins[1])}:{int(maxs[1])}, {int(mins[2])}:{int(maxs[2])})"
        )

    def _refresh_bc_arrow_models(self, nodeset_labelmap=None, reference_node=None):
        self._delete_bc_arrow_models()
        reference_node = reference_node or self._workflow_replay_preview_volume()
        arrow_scale = self._load_arrow_display_scale()
        arrow_length = self._bc_arrow_length_mm(reference_node) * arrow_scale
        if abs(arrow_scale - 1.0) > 1.0e-6:
            self._append_log(f"Load preview: load arrow scale={arrow_scale:g}x.\n")
        created = 0
        for row, row_data in enumerate(self.contactPlaneRows):
            plane = row_data.get("plane")
            spec = self._contact_row_spec(row)
            if spec["bc_type"] == "None" or spec["contact"] == "Connective disk":
                continue
            label = self._bc_preview_label(spec, row)
            self._append_log(
                f"{spec['name']}: BC mode={spec['bc_type']}, contact={spec['contact']}, "
                f"surface_mode={spec.get('surface_mode', 'project')}, preview_label={label}; "
                f"{self._nodeset_label_debug_summary(nodeset_labelmap, label)}.\n"
            )
            if spec["direction"] in {"Plane normal", "normal"}:
                normal = _plane_normal_world(plane)
                self._append_log(
                    f"{spec['name']}: load plane normal RAS=({normal[0]:.3g}, {normal[1]:.3g}, {normal[2]:.3g}).\n"
                )
            if spec["contact"] == "Bone surface" and spec.get("surface_mode") == "intersect":
                points = []
                center = None
                if nodeset_labelmap is not None:
                    points = self.logic.label_sample_points_ras(
                        nodeset_labelmap,
                        label,
                        reference_node,
                        max_points=384,
                    )
                    center = self.logic.label_centroid_ras(nodeset_labelmap, label, reference_node)
                if not points:
                    points = self._intersect_nodeset_points_ras(row, label, max_points=384)
                    center = _points_centroid(points)
            else:
                points = self.logic.label_sample_points_ras(
                    nodeset_labelmap, label, reference_node, max_points=384
                )
                center = self.logic.label_centroid_ras(nodeset_labelmap, label, reference_node)
            if nodeset_labelmap is not None and not points:
                self._append_log(
                    f"{spec['name']}: no boundary nodes found for current {spec['bc_type']} label. "
                    "Recreate contact regions or enlarge/reposition the plane.\n"
                )
                continue
            if center is None:
                center = _plane_center(plane)
            if center is None:
                continue
            if not points:
                points = [center]
            sampled_count = len(points)
            spec = dict(spec)
            spec["_preview_half_width"] = self._preview_half_width_along_gradient(plane, spec, points, center)
            if spec["bc_type"] == "Fixed":
                label = 1
                color, _opacity = _nodeset_label_color(label)
                marker = self.logic.create_point_markers(
                    f"ParOSol_BC_marker_{row}_Fixed",
                    points,
                    color,
                )
                if marker is not None:
                    self.bcMarkerNodes.append(marker)
                continue
            if str(spec["bc_type"]).startswith("Load history"):
                marker = self.logic.create_point_markers(
                    f"ParOSol_BC_marker_{row}_{spec['bc_type'].replace(' ', '_')}",
                    points,
                    (0.95, 0.72, 0.18),
                )
                if marker is not None:
                    self.bcMarkerNodes.append(marker)
                    created += len(points)
                self._append_log(
                    f"{spec['name']}: sampled {sampled_count} boundary points, "
                    f"showing {spec['bc_type']} driver markers.\n"
                )
                continue
            else:
                label = 3 if spec["bc_type"] == "Neumann" else 2
            color, _opacity = _nodeset_label_color(label)
            vectors = []
            for point_index, point in enumerate(points):
                point_direction = self._bc_direction_vector_for_point(plane, spec, point, center)
                if _load_value_number(spec) < 0 and spec["bc_type"] not in {"Bending", "Bending symmetric"}:
                    point_direction = tuple(-float(component) for component in point_direction)
                point_length = (
                    arrow_length
                    * self._bc_value_display_scale(spec)
                    * self._bc_vector_relative_scale(plane, spec, point, center)
                )
                if spec["contact"] == "Bone surface" and spec.get("surface_mode") == "intersect":
                    point_length = min(
                        point_length,
                        self._intersect_preview_arrow_length_mm() * arrow_scale,
                    )
                if _vector_length(point_direction) <= 1e-6 or point_length <= 1e-6:
                    continue
                if spec["contact"] == "Bone surface" and spec.get("surface_mode") == "intersect":
                    plane_normal = _plane_normal_world(plane)
                    if _dot(point_direction, plane_normal) >= 0:
                        arrow_start = _arrow_start_for_tip(point, point_direction, point_length)
                    else:
                        arrow_start = tuple(float(value) for value in point)
                else:
                    arrow_start = _arrow_start_for_tip(point, point_direction, point_length)
                vectors.append((arrow_start, point_direction, point_length))
            self._append_log(
                f"{spec['name']}: sampled {sampled_count} boundary points, built {len(vectors)} "
                f"{spec['bc_type']} vectors.\n"
            )
            glyph = self.logic.create_arrow_glyph_model(
                f"ParOSol_BC_arrow_samples_{row}_{spec['bc_type']}",
                vectors,
                color,
            )
            if glyph is not None:
                self.bcArrowNodes.append(glyph)
                created += len(vectors)
                bounds = [0.0] * 6
                try:
                    glyph.GetPolyData().GetBounds(bounds)
                    self._append_log(
                        f"{spec['name']}: arrow bounds RAS=({bounds[0]:.3g}:{bounds[1]:.3g}, "
                        f"{bounds[2]:.3g}:{bounds[3]:.3g}, {bounds[4]:.3g}:{bounds[5]:.3g}).\n"
                    )
                except Exception:
                    pass
        self._append_log(f"Updated BC vector markers: {created} sampled vectors.\n")

    def _bc_direction_vector_for_point(self, plane, spec, point, center):
        if spec["bc_type"] == "Bending":
            normal = self._bc_direction_vector(plane, {"direction": "Plane normal", "x": "", "y": "", "z": ""})
            gradient_axis = self._bending_gradient_axis(plane, spec)
            sign = -1.0 if _load_value_number(spec) < 0 else 1.0
            side = 1.0 if sign * _dot(_subtract(point, center), gradient_axis) >= 0 else -1.0
            return tuple(side * float(component) for component in normal)
        if spec["bc_type"] == "Bending symmetric":
            normal = self._bc_direction_vector(plane, {"direction": "Plane normal", "x": "", "y": "", "z": ""})
            value = self._symmetric_bending_value(plane, spec, point, center)
            side = 1.0 if value >= 0 else -1.0
            return tuple(side * float(component) for component in normal)
        if spec["bc_type"] == "Torsion":
            normal = self._bc_direction_vector(plane, {"direction": "Plane normal", "x": "", "y": "", "z": ""})
            radial = _subtract(point, center)
            tangent = _cross(normal, radial)
            if _vector_length(tangent) <= 1e-6:
                return (0.0, 0.0, 0.0)
            return tuple(_normalized(tangent))
        return self._bc_direction_vector(plane, spec)

    def _bc_vector_relative_scale(self, plane, spec, point, center):
        if spec["bc_type"] in {"Bending", "Bending symmetric"}:
            direction = self._bending_gradient_axis(plane, spec)
            rel = _subtract(point, center)
            signed_distance = _dot(rel, direction)
            scale = abs(signed_distance)
            if plane is not None:
                _center, _normal, _u, _v, half_u, half_v = _plane_geometry(
                    plane,
                    shape=spec["shape"],
                    radius_mm=spec["radius"],
                    square_width_mm=spec["radius"] * 2.0,
                    hex_radius_mm=spec["radius"],
                    use_plane_size=spec["use_plane_size"],
                )
                normalized = min(1.0, scale / max(half_u, half_v, 1.0))
            else:
                normalized = min(1.0, scale / 10.0)
            if spec["bc_type"] == "Bending symmetric":
                value = self._symmetric_bending_value(plane, spec, point, center)
                return max(0.2, min(1.0, abs(value)))
            return max(0.2, normalized)
        if spec["bc_type"] == "Torsion":
            radial = _vector_length(_subtract(point, center))
            if plane is not None:
                _center, _normal, _u, _v, half_u, half_v = _plane_geometry(
                    plane,
                    shape=spec["shape"],
                    radius_mm=spec["radius"],
                    square_width_mm=spec["radius"] * 2.0,
                    hex_radius_mm=spec["radius"],
                    use_plane_size=spec["use_plane_size"],
                )
                return max(0.2, min(1.0, radial / max(half_u, half_v, 1.0)))
            return max(0.2, min(1.0, radial / 10.0))
        return 1.0

    def _bc_value_display_scale(self, spec):
        try:
            magnitude = abs(_load_value_number(spec))
        except Exception:
            magnitude = 0.0
        if magnitude <= 0:
            return 0.0
        units = str(spec.get("units", "")).strip().lower()
        mode = str(spec.get("bc_type", "")).strip()
        if mode in {"Bending", "Bending symmetric", "Torsion"} or units == "deg":
            reference = 1.0
        elif mode == "Neumann" or units in {"n", "nmm"}:
            reference = 100.0
        else:
            reference = 1.0
        return max(0.25, min(3.0, magnitude / reference))

    def _preview_half_width_along_gradient(self, plane, spec, points, center):
        if spec["bc_type"] not in {"Bending", "Bending symmetric"}:
            return None
        gradient_axis = self._bending_gradient_axis(plane, spec)
        distances = [
            abs(_dot(_subtract(point, center), gradient_axis))
            for point in points
        ]
        if distances:
            half_width = max(float(max(distances)), 1.0e-6)
            if half_width > 1.0e-6:
                return half_width
        return None

    def _bending_gradient_axis(self, plane, spec):
        if spec.get("direction") in {"Plane normal", "normal"} and plane is not None:
            normal = self._bc_direction_vector(plane, {"direction": "Plane normal", "x": "", "y": "", "z": ""})
            u_axis, _v_axis = _plane_axes_from_plane(plane, normal)
            return tuple(u_axis)
        return self._bc_direction_vector(plane, spec)

    def _symmetric_bending_value(self, plane, spec, point, center):
        gradient_axis = self._bending_gradient_axis(plane, spec)
        signed_distance = _dot(_subtract(point, center), gradient_axis)
        half_width = spec.get("_preview_half_width")
        if half_width is None and plane is not None:
            _center, _normal, _u, _v, half_u, half_v = _plane_geometry(
                plane,
                shape=spec["shape"],
                radius_mm=spec["radius"],
                square_width_mm=spec["radius"] * 2.0,
                hex_radius_mm=spec["radius"],
                use_plane_size=spec["use_plane_size"],
            )
            half_width = max(half_u, half_v, 1.0)
        if half_width is None:
            half_width = 10.0
        relative = max(-1.0, min(1.0, signed_distance / max(half_width, 1.0)))
        neutral_fraction = 0.5
        sign = -1.0 if _load_value_number(spec) < 0 else 1.0
        return sign * (relative * relative - neutral_fraction)

    def _disk_materials(self):
        materials = {}
        for row in range(self._table_row_count()):
            spec = self._contact_row_spec(row)
            if spec["contact"] not in {"Material disks", "Connective disk"}:
                continue
            materials[int(spec["disk_label"])] = {
                "name": f"{spec['name']}_disk",
                "E": float(spec["disk_e"]),
                "nu": float(spec["disk_nu"]),
            }
        return materials

    def _bc_direction_vector(self, plane, spec):
        direction = str(spec.get("direction", "normal"))
        if direction in {"x", "World R/L"}:
            return (1.0, 0.0, 0.0)
        if direction in {"y", "World A/P"}:
            return (0.0, 1.0, 0.0)
        if direction in {"z", "World S/I"}:
            return (0.0, 0.0, 1.0)
        if direction in {"vector", "Custom vector"}:
            return _normalized(
                (
                    float(spec["x"] or 0.0),
                    float(spec["y"] or 0.0),
                    float(spec["z"] or 0.0),
                )
            )
        normal = _plane_normal_world(plane)
        if direction in {"plane_u", "plane_v", "Plane U", "Plane V"} and plane is not None:
            u_axis, v_axis = _plane_axes_from_plane(plane, normal)
            return tuple(u_axis if direction in {"plane_u", "Plane U"} else v_axis)
        return tuple(normal)

    def _apply_plane_row_geometry(self, row):
        if row is None or row < 0 or row >= len(self.contactPlaneRows):
            return
        plane = self.contactPlaneRows[row].get("plane")
        if plane is None:
            return
        spec = self._contact_row_spec(row)
        axis_index = {"x": 0, "y": 1, "z": 2}[spec["axis"]]
        normal_sign = 1 if spec["normal"] == "+" else -1
        normal = [0.0, 0.0, 0.0]
        normal[axis_index] = float(normal_sign)
        plane.SetNormal(normal)
        self._update_selected_plane_status()

    def _set_selected_plane_axis_normal(self, axis, normal):
        row = self._selected_contact_row()
        if row is None or row < 0:
            return
        axis_widget = self.planeTable.cellWidget(row, 1)
        normal_widget = self.planeTable.cellWidget(row, 2)
        if axis_widget is not None:
            axis_widget.setCurrentText(str(axis).lower())
        if normal_widget is not None:
            normal_widget.setCurrentText(str(normal))
        self._apply_plane_row_geometry(row)

    def _apply_selected_plane_normal_from_fields(self):
        try:
            vector = (
                float(self.planeNormalREdit.text),
                float(self.planeNormalAEdit.text),
                float(self.planeNormalSEdit.text),
            )
            self._set_selected_plane_normal_vector(vector)
        except Exception as exc:
            slicer.util.errorDisplay(f"Plane normal must be three numeric RAS values: {exc}")

    def _set_selected_plane_normal_vector(self, vector):
        plane = self._selected_contact_plane()
        if plane is None:
            return
        if _vector_length(vector) <= 1.0e-9:
            raise ValueError("Plane normal cannot be zero.")
        normal = _normalized(vector)
        plane.SetNormal(normal)
        row = self._selected_contact_row()
        if row is not None and 0 <= row < self._table_row_count():
            dominant_axis = int(np.argmax(np.abs(normal)))
            axis_widget = self.planeTable.cellWidget(row, 1)
            sign_widget = self.planeTable.cellWidget(row, 2)
            if axis_widget is not None:
                axis_widget.setCurrentText(("x", "y", "z")[dominant_axis])
            if sign_widget is not None:
                sign_widget.setCurrentText("+" if normal[dominant_axis] >= 0 else "-")
        self._update_selected_plane_status()

    def _delete_selected_contact_plane(self):
        row = self._selected_contact_row()
        if row is None or row >= len(self.contactPlaneRows):
            return
        load_row = self._load_row_for_contact(row)
        self.logic.remove_node(self.contactPlaneRows[row].get("plane"))
        self.contactPlaneRows.pop(row)
        self.planeTable.removeRow(row)
        if load_row is not None and load_row < self._load_row_count():
            self.loadTable.removeRow(load_row)
        for index, row_data in enumerate(self.contactPlaneRows):
            row_data["load_row"] = index if index < self._load_row_count() else None
        self._update_selected_plane_status()

    def _flip_selected_contact_plane(self):
        row = self._selected_contact_row()
        if row is None:
            return
        combo = self.planeTable.cellWidget(row, 2)
        if combo is None:
            return
        combo.setCurrentText("-" if self._widget_text(combo) == "+" else "+")
        self._apply_plane_row_geometry(row)

    def _move_selected_plane_in_plane(self, direction):
        plane = self._selected_disk_plane()
        if plane is None:
            return
        step = float(self.planeNudgeStep.value)
        if direction in {"left", "right"}:
            axis = "u"
            distance = -step if direction == "left" else step
        else:
            axis = "v"
            distance = step if direction == "up" else -step
        self.logic.move_plane_in_plane(plane, axis=axis, distance_mm=distance)
        row = self._selected_contact_row()
        name = self._contact_row_spec(row)["name"] if row is not None else "selected"
        self._update_selected_plane_status()
        self._append_log(f"Moved {name} {direction} by {step:g} mm.\n")

    def _rotate_selected_plane(self, axis, sign):
        plane = self._selected_disk_plane()
        if plane is None:
            return
        step = float(self.planeRotateStep.value) * float(sign)
        self.logic.rotate_plane(plane, axis=axis, angle_degrees=step)
        row = self._selected_contact_row()
        name = self._contact_row_spec(row)["name"] if row is not None else "selected"
        axis_label = {"u": "U", "v": "V", "n": "normal"}.get(str(axis), str(axis))
        self._update_selected_plane_status()
        self._append_log(f"Rotated {name} around {axis_label} by {step:g} deg.\n")

    def _update_selected_plane_status(self):
        row = self._selected_contact_row()
        plane = self._selected_contact_plane()
        if row is None or plane is None:
            self.planeStatusLabel.text = "Select a plane row."
            return
        spec = self._contact_row_spec(row)
        center = [0.0, 0.0, 0.0]
        normal = [0.0, 0.0, 1.0]
        plane.GetCenter(center)
        try:
            plane.GetNormalWorld(normal)
        except Exception:
            plane.GetNormal(normal)
        self.planeNormalREdit.setText(f"{float(normal[0]):.6g}")
        self.planeNormalAEdit.setText(f"{float(normal[1]):.6g}")
        self.planeNormalSEdit.setText(f"{float(normal[2]):.6g}")
        size = [0.0, 0.0]
        if hasattr(plane, "GetSize"):
            try:
                plane.GetSize(size)
            except Exception:
                pass
        self.planeStatusLabel.text = (
            f"{spec['name']} | {spec['bc_type']}\n"
            f"Normal RAS: {normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}"
        )
        self.planeStatusLabel.toolTip = (
            f"{spec['name']} | {spec['bc_type']} {spec['direction']}\n"
            f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\n"
            f"Normal: ({normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f})\n"
            f"Size: ({size[0]:.2f}, {size[1]:.2f})"
        )

    def _create_disk_plane(self, side):
        volume = self._volume()
        plane = self.logic.create_plane(
            f"ParOSol_{side}_disk_plane",
            volume,
            side=side,
            bounds_node=self._bounds_node(),
        )
        if side == "top":
            self.topDiskPlane = plane
        else:
            self.bottomDiskPlane = plane
        self._show_in_standard_slice_views(volume, label_node=None, reset_orientations=False)

    def _create_bc_plane(self, kind):
        volume = self._volume()
        side = "bottom" if kind == "fixed" else "top"
        plane = self.logic.create_plane(
            f"ParOSol_{kind}_bc_plane",
            volume,
            side=side,
            bounds_node=self._bounds_node(),
        )
        if kind == "fixed":
            self.fixedPlane = plane
        else:
            self.loadedPlane = plane

    def _selected_disk_plane(self):
        return self._selected_contact_plane()

    def _nudge_selected_plane(self, direction):
        plane = self._selected_disk_plane()
        if plane is None:
            return
        step = float(self.planeNudgeStep.value)
        distance = step if direction == "in" else -step
        self.logic.move_plane_along_normal(plane, distance)
        row = self._selected_contact_row()
        name = self._contact_row_spec(row)["name"] if row is not None else "selected"
        self._update_selected_plane_status()
        self._append_log(f"Moved {name} {direction} by {step:g} mm.\n")

    def _resize_selected_plane(self, direction):
        plane = self._selected_disk_plane()
        if plane is None:
            return
        step = max(float(self.planeNudgeStep.value), 0.1)
        factor = 1.0 + step / 10.0 if direction == "wider" else max(1.0 - step / 10.0, 0.1)
        self.logic.scale_plane_size(plane, factor)
        row = self._selected_contact_row()
        name = self._contact_row_spec(row)["name"] if row is not None else "selected"
        self._update_selected_plane_status()
        self._append_log(f"Made {name} {direction}.\n")

    def _nudge_disk_plane(self, side, direction):
        plane = self.topDiskPlane if side == "top" else self.bottomDiskPlane
        if plane is None:
            return
        step = float(self.planeNudgeStep.value)
        distance = step if direction == "in" else -step
        self.logic.move_plane_along_normal(plane, distance)
        self._append_log(f"Moved {side} plane {direction} by {step:g} mm.\n")

    def preview_disks(self):
        controller = getattr(self, "_boundaryPreviewController", None)
        if controller is not None and hasattr(controller, "preview_disks"):
            return controller.preview_disks()
        return self._preview_disks_impl()

    def preview_disks_and_next(self):
        self._advanceAfterContactRegions = True
        try:
            return self.preview_disks()
        finally:
            self._advanceAfterContactRegions = False

    def _preview_disks_impl(self):
        if not self._is_interactive_profile():
            slicer.util.errorDisplay("Select profile 'interactive_custom' to use disks/caps.")
            return
        if self._has_applied_workflow_replay_model():
            try:
                self._workflow_replay_boundary_preview(show_load_vectors=False)
                self._mark_stage_complete("boundary")
                if getattr(self, "_advanceAfterContactRegions", False):
                    self._advance_workflow_tab_after("boundary")
            except Exception as exc:
                slicer.util.errorDisplay(str(exc))
            return
        volume = self._volume()
        self._ensure_default_contact_planes()
        self.logic.remove_node(self.diskLabelSelector.currentNode())
        self.diskLabelSelector.setCurrentNode(None)
        self.logic.remove_node(self.bcLabelSelector.currentNode())
        self.bcLabelSelector.setCurrentNode(None)
        target_mask_node = self.maskSelector.currentNode()
        disk_target_values = self._current_workflow_disk_projection_values()
        try:
            target_values = (
                disk_target_values
                if disk_target_values is not None
                else (
                    None
                    if target_mask_node is not None
                    else self._validated_active_material_labels_for_preview()
                )
            )
        except ValueError as exc:
            slicer.util.errorDisplay(str(exc))
            return
        try:
            active_target = _target_mask_array(
                target_mask_node or volume,
                volume,
                active_values=target_values,
                fallback_to_nonzero=bool(disk_target_values),
            )
        except Exception:
            active_target = None
        if active_target is None or not np.any(active_target):
            target_text = (
                "selected mask/segment"
                if target_mask_node is not None
                else f"material labels {', '.join(str(value) for value in (target_values or ())) or 'none'}"
            )
            message = (
                "No active voxels found for boundary-condition generation from "
                f"{target_text}. Check the selected mask segment or apply the correct workflow profile labels."
            )
            self._append_log(message + "\n")
            slicer.util.warningDisplay(message)
            return
        contact_rows = []
        for row, row_data in enumerate(self.contactPlaneRows):
            spec = self._contact_row_spec(row)
            contact_rows.append(
                {
                    **spec,
                    "plane": row_data.get("plane"),
                    "row_index": row,
                    "preview_label": self._bc_preview_label(spec, row),
                }
            )
        try:
            padded_volume, padded_mask, did_pad = self.logic.pad_volume_for_projected_contacts(
                volume,
                target_mask_node,
                contact_rows,
                target_values=disk_target_values or target_values,
            )
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
            return
        if did_pad:
            old_dims = volume.GetImageData().GetDimensions()
            new_dims = padded_volume.GetImageData().GetDimensions()
            self.imageSelector.setCurrentNode(padded_volume)
            volume = padded_volume
            if padded_mask is not None:
                self.maskSelector.setCurrentNode(padded_mask)
                target_mask_node = padded_mask
            self._append_log(
                "Expanded modelling image to fit projected disks: "
                f"xyz {old_dims[0]}x{old_dims[1]}x{old_dims[2]} -> "
                f"{new_dims[0]}x{new_dims[1]}x{new_dims[2]}.\n"
            )

        try:
            disk_labelmap, nodesets = self.logic.generate_workflow_contact_labelmaps(
                volume,
                contact_rows,
                target_mask_node=target_mask_node,
                target_values=disk_target_values or target_values,
            )
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
            return

        label = disk_labelmap
        if label is not None:
            self.diskLabelSelector.setCurrentNode(label)
        self.bcLabelSelector.setCurrentNode(nodesets)
        self._show_mask_3d_preserving_mask_selection(
            self.maskSelector.currentNode(),
            volume,
            active_values=None,
        )
        if label is not None and int(np.count_nonzero(slicer.util.arrayFromVolume(label))) > 0:
            self.logic.labelmap_to_3d_segmentation(
                label,
                "ParOSol_contact_caps_3D",
                reference_node=volume,
                kind="disks",
            )
        else:
            self.logic.remove_named_node("ParOSol_contact_caps_3D")
        if int(np.count_nonzero(slicer.util.arrayFromVolume(nodesets))) > 0:
            self.logic.labelmap_to_3d_segmentation(
                nodesets,
                "ParOSol_boundary_conditions_3D",
                reference_node=volume,
                kind="nodesets",
            )
        else:
            self.logic.remove_named_node("ParOSol_boundary_conditions_3D")
        cap_count = 0
        if label is not None:
            cap_count = int(np.count_nonzero(slicer.util.arrayFromVolume(label)))
        nodeset_count = int(np.count_nonzero(slicer.util.arrayFromVolume(nodesets)))
        self._show_in_standard_slice_views(volume, label_node=None, reset_orientations=False)
        self._delete_bc_arrow_models()
        if label is not None and cap_count == 0:
            self._append_log("Created contact regions: no disk voxels were generated; check plane position, normal, intrusion, and image padding.\n")
        else:
            self._append_log(
                f"Created contact regions: disk voxels={cap_count}, contact-region voxels={nodeset_count}. "
                "Showing caps when present, otherwise contact regions.\n"
            )
        for row, row_data in enumerate(self.contactPlaneRows):
            if row_data.get("plane") is None:
                continue
            spec = self._contact_row_spec(row)
            if spec["bc_type"] != "None" and spec["contact"] != "Connective disk":
                label_value = self._bc_preview_label(spec, row)
                self._append_log(
                    f"{spec['name']}: generated {spec['bc_type']} contact={spec['contact']} "
                    f"surface_mode={spec.get('surface_mode', 'project')} "
                    f"{self._nodeset_label_debug_summary(nodesets, label_value)}.\n"
                )
        self._profileHasGeneratedBoundaryConditions = True
        self._mark_stage_complete("boundary")
        self._update_profile_mode()
        if getattr(self, "_advanceAfterContactRegions", False):
            self._advance_workflow_tab_after("boundary")

    def preview_loads(self):
        controller = getattr(self, "_loadPreviewController", None)
        if controller is not None and hasattr(controller, "preview_loads"):
            return controller.preview_loads()
        return self._preview_loads_impl()

    def preview_loads_and_next(self):
        self._advanceAfterLoads = True
        try:
            return self.preview_loads()
        finally:
            self._advanceAfterLoads = False

    def _preview_loads_impl(self):
        if not self._is_interactive_profile():
            slicer.util.errorDisplay("Select profile 'interactive_custom' to preview loads.")
            return
        nodesets = self.bcLabelSelector.currentNode()
        if nodesets is None:
            slicer.util.warningDisplay("Create contact regions first.")
            return
        try:
            if int(np.count_nonzero(slicer.util.arrayFromVolume(nodesets))) == 0:
                slicer.util.warningDisplay("The contact-region labelmap is empty. Run Create Regions first.")
                return
        except Exception:
            pass
        try:
            self._sync_nodeset_labels_to_current_loads(nodesets)
        except ValueError as exc:
            slicer.util.errorDisplay(str(exc))
            return
        context_mask_node = self.maskSelector.currentNode()
        volume = self._workflow_replay_preview_volume()
        if context_mask_node is not None and volume is not None:
            self._show_mask_3d_preserving_mask_selection(context_mask_node, volume)
        self._show_in_standard_slice_views(volume, label_node=None, reset_orientations=False)
        self.logic.remove_named_node("ParOSol_boundary_conditions_3D")
        self.logic.labelmap_to_3d_segmentation(
            nodesets,
            "ParOSol_boundary_conditions_3D",
            reference_node=volume,
            kind="nodesets",
        )
        self._refresh_bc_arrow_models(nodesets, reference_node=volume)
        self._profileHasGeneratedBoundaryConditions = True
        self._mark_stage_complete("loads")
        self._update_profile_mode()
        if getattr(self, "_advanceAfterLoads", False):
            self._advance_workflow_tab_after("loads")

    def _on_load_arrow_scale_changed(self, *_args):
        if not getattr(self, "bcArrowNodes", None):
            return
        try:
            nodesets = self.bcLabelSelector.currentNode()
        except Exception:
            nodesets = None
        try:
            reference_node = self._workflow_replay_preview_volume()
        except Exception:
            reference_node = None
        self._refresh_bc_arrow_models(nodesets, reference_node=reference_node)

    def _sync_nodeset_labels_to_current_loads(self, nodesets):
        if nodesets is None:
            return
        try:
            array = np.asarray(slicer.util.arrayFromVolume(nodesets)).copy()
        except Exception:
            return
        changed = False
        allowed_labels = set()
        for row in range(self._table_row_count()):
            spec = self._contact_row_spec(row)
            desired = (
                0
                if spec["bc_type"] == "None" or spec["contact"] == "Connective disk"
                else int(self._bc_preview_label(spec, row))
            )
            if desired > 0:
                allowed_labels.add(int(desired))
            row_offset = int(row) + 1
            row_labels = {
                100 + row_offset,
                200 + row_offset,
                300 + row_offset,
                400 + row_offset,
                _generated_boundary_label_for_row("Fixed", row),
                _generated_boundary_label_for_row("Dirichlet", row),
                _generated_boundary_label_for_row("Neumann", row),
                _generated_boundary_label_for_row("None", row),
            }
            existing = [label for label in row_labels if np.any(array == label)]
            if not existing and row == 0:
                existing = [
                    label for label in (1, 2, 3)
                    if np.any(array == label) and label != desired
                ]
            for old_label in existing:
                if int(old_label) == desired:
                    continue
                array[array == int(old_label)] = desired
                changed = True
        array, surface_changed = self._restrict_intersect_nodeset_array(array)
        changed = changed or surface_changed
        if allowed_labels:
            keep = np.isin(array, tuple(sorted(allowed_labels)))
            if np.any((array != 0) & ~keep):
                array = np.where(keep, array, 0)
                changed = True
        if changed:
            slicer.util.updateVolumeFromArray(nodesets, array.astype(np.uint16, copy=False))
            nodesets.Modified()
            self.logic.style_labelmap(nodesets, "nodesets")
            self._append_log("Updated boundary-condition labels from current load table.\n")

    def _refresh_material_disk_nodesets_from_disk_labels(
        self,
        nodesets,
        disk_labelmap,
        volume,
        *,
        target_mask_node=None,
        target_values=None,
    ):
        if nodesets is None or volume is None:
            return nodesets
        contact_rows = []
        for row, row_data in enumerate(self.contactPlaneRows):
            plane = row_data.get("plane")
            if plane is None:
                continue
            spec = self._contact_row_spec(row)
            contact_rows.append(
                {
                    **spec,
                    "plane": plane,
                    "row_index": row,
                    "preview_label": self._bc_preview_label(spec, row),
                }
            )
        if not contact_rows:
            return nodesets

        generated_disk_node = None
        refreshed_node = None
        try:
            generated_disk_node, refreshed_node = self.logic.generate_workflow_contact_labelmaps(
                volume,
                contact_rows,
                target_mask_node=target_mask_node,
                target_values=target_values,
            )
            if generated_disk_node is not None:
                if hasattr(self, "diskLabelSelector"):
                    previous_disk = self.diskLabelSelector.currentNode()
                    if previous_disk is not generated_disk_node:
                        self.logic.remove_node(previous_disk)
                    self.diskLabelSelector.setCurrentNode(generated_disk_node)
                else:
                    self.logic.remove_node(generated_disk_node)
                generated_disk_node = None
            self._append_log("Regenerated workflow contact labelmaps from current planes before export.\n")
            for row_spec in contact_rows:
                row = int(row_spec.get("row_index", 0))
                spec = self._contact_row_spec(row)
                if spec["bc_type"] == "None" or spec["contact"] == "Connective disk":
                    continue
                label = int(self._bc_preview_label(spec, row))
                count = int(np.count_nonzero(slicer.util.arrayFromVolume(refreshed_node) == label))
                self._append_log(
                    f"{spec['name']}: export {self._nodeset_label_debug_summary(refreshed_node, label)}.\n"
                )
                if count == 0:
                    raise ValueError(
                        f"{spec['name']}: nodeset label {label} is empty after regenerating workflow geometry."
                    )
            return refreshed_node
        finally:
            if generated_disk_node is not None:
                self.logic.remove_node(generated_disk_node)
            if refreshed_node is not None and refreshed_node is not nodesets:
                self.logic.group_node(refreshed_node, "Loads")

    def _restrict_intersect_nodeset_array(self, array):
        if self._volume() is None or array is None:
            return array, False
        rows = self._intersect_face_rows()
        if not rows:
            return array, False
        expected = self._intersect_nodeset_array(rows)
        if expected.shape != array.shape:
            return array, False
        labels = {int(row["label"]) for row in rows}
        changed = False
        filtered = np.asarray(array).copy()
        for label in labels:
            stale = (filtered == label) & (expected != label)
            if np.any(stale):
                filtered[stale] = 0
                changed = True
            missing = (expected == label) & (filtered != label)
            if np.any(missing):
                filtered[missing] = label
                changed = True
        return filtered, changed

    def _intersect_face_rows(self):
        rows = []
        for row, row_data in enumerate(self.contactPlaneRows):
            if row_data.get("plane") is None:
                continue
            spec = self._contact_row_spec(row)
            if (
                spec["contact"] == "Bone surface"
                and spec.get("surface_mode") == "intersect"
                and spec["bc_type"] != "None"
            ):
                rows.append(
                    {
                        "axis": spec["axis"],
                        "normal": spec["normal"],
                        "label": self._bc_preview_label(spec, row),
                        "row": row,
                        "plane": row_data.get("plane"),
                        "shape": spec.get("shape", "anatomy"),
                        "radius": spec.get("radius", 12.0),
                        "use_plane_size": spec.get("use_plane_size", True),
                    }
                )
        return rows

    def _intersect_nodeset_array(self, rows):
        volume = self._volume()
        if volume is None:
            return np.zeros((0, 0, 0), dtype=np.uint16)
        target_mask_node = self.maskSelector.currentNode()
        target_values = None if target_mask_node is not None else self._validated_active_material_labels_for_preview()
        return self.logic.fast_face_nodeset_array(
            volume,
            target_mask_node or volume,
            rows,
            active_values=target_values,
        )

    def _intersect_nodeset_points_ras(self, row, label, *, max_points):
        matching_rows = [
            item for item in self._intersect_face_rows()
            if int(item.get("row", -1)) == int(row) and int(item.get("label", 0)) == int(label)
        ]
        if not matching_rows:
            return []
        expected = self._intersect_nodeset_array(matching_rows)
        return _label_array_sample_points_ras(expected, int(label), self._volume(), max_points=max_points)

    def delete_disks(self):
        signal_states = self._begin_input_node_update_suppression()
        try:
            for row_data in self.contactPlaneRows:
                self.logic.remove_node(row_data.get("plane"))
            self.contactPlaneRows = []
            self.planeTable.setRowCount(0)
            self.loadTable.setRowCount(0)
            self.logic.remove_node(self.topDiskPlane)
            self.logic.remove_node(self.bottomDiskPlane)
            self.logic.remove_node(self.diskLabelSelector.currentNode())
            self.logic.remove_node(self.bcLabelSelector.currentNode())
            self.logic.remove_named_node("ParOSol_contact_caps_3D")
            self.logic.remove_named_node("ParOSol_boundary_conditions_3D")
            self._delete_bc_arrow_models()
            self.topDiskPlane = None
            self.bottomDiskPlane = None
            self.diskLabelSelector.setCurrentNode(None)
            self.bcLabelSelector.setCurrentNode(None)
            self._profileHasGeneratedBoundaryConditions = False
            self._update_profile_mode()
            self._show_in_standard_slice_views(self._volume(), label_node=None, reset_orientations=False)
            self._append_log("Deleted disk planes, disk labelmap, and auto contact regions.\n")
        finally:
            self._end_input_node_update_suppression(signal_states)

    def _bc_preview_label(self, spec, row=0):
        explicit = spec.get("nodeset_label")
        if explicit is not None:
            try:
                return int(explicit)
            except Exception:
                pass
        return _generated_boundary_label_for_row(spec.get("bc_type"), row)

    def preview_bc(self):
        if not self._is_interactive_profile():
            slicer.util.errorDisplay("Select profile 'interactive_custom' to edit contact regions.")
            return
        volume = self._volume()
        label = self.bcLabelSelector.currentNode()
        label = self.logic.generate_bc_labels(
            volume,
            label,
            self.fixedPlane,
            self.loadedPlane,
            thickness_mm=self.bcThickness.value,
        )
        self.bcLabelSelector.setCurrentNode(label)
        self._show_in_standard_slice_views(
            volume,
            label_node=label,
            label_opacity=0.5,
            reset_orientations=False,
        )
        self.logic.labelmap_to_3d_segmentation(
            label,
            "ParOSol_boundary_conditions_3D",
            reference_node=volume,
            kind="nodesets",
        )

    def delete_bc(self):
        signal_states = self._begin_input_node_update_suppression()
        try:
            self.logic.remove_node(self.fixedPlane)
            self.logic.remove_node(self.loadedPlane)
            self.logic.remove_node(self.bcLabelSelector.currentNode())
            self.logic.remove_named_node("ParOSol_boundary_conditions_3D")
            self._delete_bc_arrow_models()
            self.fixedPlane = None
            self.loadedPlane = None
            self.bcLabelSelector.setCurrentNode(None)
            self._show_in_standard_slice_views(self._volume(), label_node=None, reset_orientations=False)
            self._append_log("Deleted BC planes and BC labelmap.\n")
        finally:
            self._end_input_node_update_suppression(signal_states)

    def check_runtime(self):
        try:
            ok, report = self.logic.check_runtime(
                on_output=self._append_log,
                return_text=True,
            )
            self.runtimeStatusLabel.text = "Runtime OK" if ok else "Runtime check failed"
            if ok:
                dialog_text = (
                    "Runtime OK\n\n"
                    f"{_runtime_success_details(report)}\n\n"
                    f"Command: {self.logic.parosol_executable()}\n\n"
                    "ParOSol-py runtime is available."
                )
            else:
                dialog_text = (
                    "Runtime check failed\n\n"
                    "Use the Setup module to install ParOSol-py, then run the check again.\n\n"
                    + report
                )
            self._show_text_dialog(
                "ParOSol Runtime Check",
                dialog_text,
                minimum_width=640,
                minimum_height=260 if ok else 360,
            )
        except Exception as exc:
            self.runtimeStatusLabel.text = "Runtime check failed"
            slicer.util.errorDisplay(str(exc))

    def _show_text_dialog(self, title, text, *, minimum_width=700, minimum_height=420):
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(title)
        layout = qt.QVBoxLayout(dialog)
        text_box = qt.QPlainTextEdit()
        text_box.readOnly = True
        text_box.setPlainText(str(text))
        text_box.minimumWidth = int(minimum_width)
        text_box.minimumHeight = int(minimum_height)
        layout.addWidget(text_box)
        buttons = qt.QDialogButtonBox(qt.QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec_()

    def install_runtime_from_pypi(self):
        try:
            self.runtimeStatusLabel.text = "Installing from PyPI..."
            self.logic.install_pypi_into_slicer(on_output=self._append_log)
            self.runtimeStatusLabel.text = "Runtime OK"
        except Exception as exc:
            self.runtimeStatusLabel.text = "Install failed"
            slicer.util.errorDisplay(str(exc))

    def export_case(self, dry_run=False):
        output_dir = Path(self.outputDirectory.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(f"SlicerParOSol build: {SLICER_PAROSOL_BUILD}\n")
        source_paths = _parosol_source_checkout_import_paths()
        if source_paths:
            self._append_log(
                "ParOSol-py source: "
                + ", ".join(str(path) for path in source_paths)
                + "\n"
            )
        interactive = self._uses_generated_interactive_model()
        editor_active = self._is_interactive_profile()
        fast_profile = self._source_profile_for_fast_run()
        profile_for_run = (
            fast_profile
            if fast_profile and not interactive
            else self._widget_text(self.profileBox, "interactive_custom")
        )
        material_override = self._material_override() if editor_active else None
        disk_materials = self._disk_materials() if interactive else {}
        density_mode = bool(material_override and material_override.get("image_type") == "density")
        export_displacements = self._export_displacements_enabled()
        output_fields = self._selected_output_fields()
        postprocess_config = self._postprocess_config()
        has_caps = False
        if interactive and self.diskLabelSelector.currentNode() is not None:
            try:
                has_caps = bool(
                    np.count_nonzero(slicer.util.arrayFromVolume(self.diskLabelSelector.currentNode()))
                )
            except Exception:
                has_caps = True
        solver_volume = None
        image_node = self._volume()
        mask_node = self.maskSelector.currentNode()
        self._apply_workflow_label_map_to_segmentation(mask_node)
        disk_node = self.diskLabelSelector.currentNode() if interactive else None
        nodeset_node = self.bcLabelSelector.currentNode() if interactive else None
        try:
            if (
                interactive
                and self._has_applied_workflow_replay_model()
                and not self._uses_load_history_bc()
            ):
                workflow_image_path, workflow_mask_path = (
                    self._workflow_replay_source_input_paths_for_export(output_dir)
                )
                load_case_override = self._current_interactive_load_case_override()
                effective_mpi_processes = self._effective_mpi_processes()
                effective_mpi_launcher = (
                    self._selected_mpi_launcher() if effective_mpi_processes > 1 else ""
                )
                workflow_replay_config = self._interactive_workflow_replay_config_for_export(
                    image_path=workflow_image_path,
                    mask_path=workflow_mask_path,
                    output_dir=output_dir,
                    load_case_override=load_case_override,
                    nodeset_specs=None,
                    material_override=material_override,
                    mpi_processes=effective_mpi_processes,
                    mpi_launcher=effective_mpi_launcher,
                    tolerance=self._effective_solver_tolerance(),
                    export_displacements=export_displacements,
                    output_fields=output_fields,
                    postprocess_config=postprocess_config,
                )
                if workflow_replay_config is None:
                    raise ValueError("The applied workflow is not a workflow-replay model.")
                config_path = self.logic.write_config(
                    workflow_replay_config,
                    output_dir / "parosol_slicer_case.yaml",
                )
                self._validate_exported_config_nodeset_files(config_path)
                self._append_log(
                    "Export mode: workflow replay from the current interactive planes.\n"
                )
                self._append_log(f"Exported config: {config_path}\n")
                self._mark_stage_complete("export")
                if dry_run:
                    return config_path
                return config_path

            if (
                interactive
                and self._has_applied_workflow_replay_model()
                and self._uses_load_history_bc()
            ):
                workflow_image_path, workflow_mask_path = (
                    self._workflow_replay_source_input_paths_for_export(output_dir)
                )
                load_case_override = self._current_interactive_load_case_override()
                effective_mpi_processes = self._effective_mpi_processes()
                effective_mpi_launcher = (
                    self._selected_mpi_launcher() if effective_mpi_processes > 1 else ""
                )
                workflow_replay_config = self._interactive_workflow_replay_config_for_export(
                    image_path=workflow_image_path,
                    mask_path=workflow_mask_path,
                    output_dir=output_dir,
                    load_case_override=load_case_override,
                    nodeset_specs=None,
                    material_override=material_override,
                    mpi_processes=effective_mpi_processes,
                    mpi_launcher=effective_mpi_launcher,
                    tolerance=self._effective_solver_tolerance(),
                    export_displacements=export_displacements,
                    output_fields=output_fields,
                    postprocess_config=postprocess_config,
                )
                workflow_replay_batch_config = self._workflow_replay_load_history_config_for_export(
                    workflow_replay_config,
                    output_dir=output_dir,
                    load_case_override=load_case_override,
                )
                config_path = self.logic.write_config(
                    workflow_replay_batch_config,
                    output_dir / "parosol_slicer_case.yaml",
                )
                self._validate_exported_config_nodeset_files(config_path)
                self._append_log(
                    "Export mode: workflow replay load history batch from the current interactive planes.\n"
                )
                self._append_log(f"Exported config: {config_path}\n")
                self._mark_stage_complete("export")
                if dry_run:
                    return config_path
                return config_path

            if bool(self.resampleIsotropicCheckBox.checked):
                image_node, mask_node, disk_node, nodeset_node, resampled = (
                    self.logic.ensure_isotropic_inputs(
                        image_node,
                        target_spacing_mm=float(self.isotropicSpacingSpin.value),
                        spacing_tolerance_mm=float(self._resample_spacing_tolerance_mm),
                        spacing_tolerance_relative=float(self._resample_spacing_tolerance_relative),
                        canonicalize_within_tolerance=bool(self._resample_canonicalize_within_tolerance),
                        image_is_label=bool(
                            isinstance(material_override, dict)
                            and str(material_override.get("image_type", "")).strip().lower()
                            in {"material_labels", "labels", "segmentation"}
                        ),
                        mask_node=mask_node,
                        disk_labelmap=disk_node,
                        nodeset_labelmap=nodeset_node,
                    )
                )
                if resampled:
                    self.imageSelector.setCurrentNode(image_node)
                    if mask_node is not None:
                        self.maskSelector.setCurrentNode(mask_node)
                    if disk_node is not None:
                        self.diskLabelSelector.setCurrentNode(disk_node)
                    if nodeset_node is not None:
                        self.bcLabelSelector.setCurrentNode(nodeset_node)
                    self._show_mask_3d_preserving_mask_selection(mask_node, image_node)
                    self._append_log(
                        f"Resampled inputs to isotropic {float(self.isotropicSpacingSpin.value):g} mm spacing.\n"
                    )
            else:
                spacing = tuple(abs(float(value)) for value in image_node.GetSpacing())
                if not _is_isotropic_spacing(spacing):
                    spacing_text = ", ".join(f"{value:.6g}" for value in spacing)
                    raise ValueError(
                        f"ParOSol requires isotropic spacing, but the current spacing is ({spacing_text}) mm. Enable "
                        "'Resample to isotropic spacing before solve' or choose an isotropic input."
                    )
            if interactive:
                workflow_image_path = self.logic.export_volume(
                    image_node, output_dir / "slicer_input.nii.gz"
                )
                workflow_disk_label_path = (
                    self.logic.export_volume(disk_node, output_dir / "disk_labels.nii.gz")
                    if disk_node is not None
                    else None
                )
                workflow_mask_path = self.logic.export_mask_like(
                    mask_node,
                    image_node,
                    output_dir / "slicer_mask.nii.gz",
                )
                workflow_nodeset_path = (
                    self.logic.export_volume(nodeset_node, output_dir / "nodesets.nii.gz")
                    if nodeset_node is not None
                    else None
                )
                workflow_nodeset_specs = None
                if workflow_nodeset_path is not None:
                    workflow_nodeset_specs, load_case_override = self._interactive_nodeset_config(
                        workflow_nodeset_path,
                        disk_label_path=workflow_disk_label_path,
                    )
                else:
                    load_case_override = None
                effective_mpi_processes = self._effective_mpi_processes()
                effective_mpi_launcher = (
                    self._selected_mpi_launcher() if effective_mpi_processes > 1 else ""
                )
                workflow_replay_config = self._interactive_workflow_replay_config_for_export(
                    image_path=workflow_image_path,
                    mask_path=workflow_mask_path,
                    output_dir=output_dir,
                    load_case_override=load_case_override,
                    nodeset_specs=workflow_nodeset_specs,
                    material_override=material_override,
                    mpi_processes=effective_mpi_processes,
                    mpi_launcher=effective_mpi_launcher,
                    tolerance=self._effective_solver_tolerance(),
                    export_displacements=export_displacements,
                    output_fields=output_fields,
                    postprocess_config=postprocess_config,
                )
                if workflow_replay_config is not None and not self._uses_load_history_bc():
                    config_path = self.logic.write_config(
                        workflow_replay_config,
                        output_dir / "parosol_slicer_case.yaml",
                    )
                    self._validate_exported_config_nodeset_files(config_path)
                    self._append_log(
                        "Export mode: workflow replay from the applied Slicer workflow.\n"
                    )
                    self._append_log(f"Exported config: {config_path}\n")
                    self._mark_stage_complete("export")
                    if dry_run:
                        return config_path
                    return config_path
            if interactive and disk_node is not None and nodeset_node is not None:
                refresh_target_values = None
                try:
                    disk_target_values = self._current_workflow_disk_projection_values()
                    refresh_target_values = (
                        disk_target_values
                        if disk_target_values is not None
                        else (
                            None
                            if mask_node is not None
                            else self._validated_active_material_labels_for_preview()
                        )
                    )
                except Exception:
                    refresh_target_values = None
                export_seed_node = self.logic.create_labelmap_like(
                    image_node, "ParOSol_export_seed_nodesets"
                )
                refreshed_node = None
                try:
                    refreshed_node = self._refresh_material_disk_nodesets_from_disk_labels(
                        export_seed_node,
                        disk_node,
                        image_node,
                        target_mask_node=mask_node,
                        target_values=refresh_target_values,
                    )
                finally:
                    if refreshed_node is not export_seed_node:
                        self.logic.remove_node(export_seed_node)
                nodeset_node = refreshed_node
                self.bcLabelSelector.setCurrentNode(nodeset_node)
                disk_node = self.diskLabelSelector.currentNode()
                try:
                    has_caps = bool(
                        disk_node is not None
                        and np.count_nonzero(slicer.util.arrayFromVolume(disk_node))
                    )
                except Exception:
                    has_caps = disk_node is not None
            if density_mode and has_caps:
                solver_volume = self.logic.density_material_input_volume(
                    image_node,
                    disk_node,
                    material_override=material_override,
                    disk_materials=disk_materials,
                    cap_e_mpa=3000.0,
                    mask_node=mask_node,
                )
                disk_labels = []
                disk_materials = {}
                material_override = {
                    "image_type": "material_mpa",
                    "materials": {
                        "units": "MPa",
                        "nu": float(self.materialNuSpin.value),
                    },
                }
            else:
                solver_volume, disk_labels = self.logic.solver_input_volume(
                    image_node,
                    disk_node,
                    disk_material_value=None,
                    mask_node=mask_node,
                )
            if self._should_normalize_xtremect_material_labels(material_override):
                normalized_volume = self.logic.normalize_xtremect_material_labels(solver_volume)
                if normalized_volume is not solver_volume:
                    if solver_volume is not image_node:
                        self.logic.remove_node(solver_volume)
                    solver_volume = normalized_volume
                    self._append_log(
                        "Normalized XtremeCT material labels for Slicer AIM export: 99->100, 126->127.\n"
                    )
            image_path = self.logic.export_volume(solver_volume, output_dir / "slicer_input.nii.gz")
            disk_label_path = None
            if interactive and disk_node is not None:
                disk_label_path = self.logic.export_volume(disk_node, output_dir / "disk_labels.nii.gz")
            mask_path = self.logic.export_mask_like(
                mask_node,
                image_node,
                output_dir / "slicer_mask.nii.gz",
            )
            nodeset_path = (
                self.logic.export_volume(nodeset_node, output_dir / "nodesets.nii.gz")
                if interactive
                else None
            )
            if interactive:
                if nodeset_path is None:
                    raise ValueError("Run Create Regions before exporting an interactive model.")
                nodeset_specs, load_case_override = self._interactive_nodeset_config(
                    nodeset_path,
                    disk_label_path=disk_label_path,
                )
                self._validate_interactive_nodeset_export_labels(
                    nodeset_specs,
                    nodeset_node,
                    disk_node,
                    nodeset_path=nodeset_path,
                    disk_label_path=disk_label_path,
                )
            else:
                nodeset_specs = None
                load_case_override = None
            effective_mpi_processes = self._effective_mpi_processes()
            effective_mpi_launcher = (
                self._selected_mpi_launcher() if effective_mpi_processes > 1 else ""
            )
            config = self._batch_workflow_config_for_export(
                image_path=image_path,
                mask_path=mask_path,
                nodeset_path=nodeset_path,
                nodeset_specs=nodeset_specs,
                load_case_override=load_case_override,
                output_dir=output_dir,
                material_override=material_override,
                mpi_processes=effective_mpi_processes,
                mpi_launcher=effective_mpi_launcher,
                tolerance=self._effective_solver_tolerance(),
                export_displacements=export_displacements,
                output_fields=output_fields,
                postprocess_config=postprocess_config,
            )
            if config is None:
                if self._uses_load_history_bc():
                    raise ValueError(
                        "Cannot export load-history contact regions as a single ParOSol case. "
                        "Apply the workflow and create contact regions so Slicer can write a load-history batch."
                    )
                config = self.logic.build_config(
                    image_path=image_path,
                    mask_path=mask_path,
                    nodeset_path=nodeset_path,
                    profile=profile_for_run,
                    output_dir=output_dir,
                    force_n=None,
                    displacement_value=1.0,
                    direction_vector=(0.0, 0.0, -1.0),
                    loaded_label=2,
                    disk_labels=disk_labels,
                    disk_materials=disk_materials,
                    disk_e_mpa=3000.0,
                    disk_nu=float(self.materialNuSpin.value),
                    material_override=material_override,
                    nodeset_specs=nodeset_specs,
                    load_case_override=load_case_override,
                    mpi_processes=effective_mpi_processes,
                    mpi_launcher=effective_mpi_launcher,
                    tolerance=self._effective_solver_tolerance(),
                    export_displacements=export_displacements,
                    output_fields=output_fields,
                    postprocess_config=postprocess_config,
                )
            preprocessing = self._preprocessing_config()
            if preprocessing:
                config["preprocessing"] = preprocessing
            custom_preprocessing = self._custom_preprocessing_config()
            if custom_preprocessing:
                config["custom_preprocessing"] = custom_preprocessing
            config_path = self.logic.write_config(config, output_dir / "parosol_slicer_case.yaml")
            self._validate_exported_config_nodeset_files(config_path)
            self._append_log(f"Exported config: {config_path}\n")
            self._mark_stage_complete("export")
            if dry_run:
                return config_path
            return config_path
        finally:
            if solver_volume is not None and solver_volume is not image_node:
                self.logic.remove_node(solver_volume)

    def _workflow_replay_load_history_config_for_export(
        self,
        config,
        *,
        output_dir,
        load_case_override,
    ):
        if not isinstance(config, dict):
            raise ValueError("The applied workflow is not a workflow-replay model.")
        if not isinstance(load_case_override, dict):
            raise ValueError("Create contact regions before exporting a load-history profile.")
        output_dir = Path(output_dir)
        config = copy.deepcopy(config)
        case_name = output_dir.name or "slicer_case"
        case_cfg = config.setdefault("case", {})
        case_cfg["name"] = case_name
        case_cfg["work_dir"] = str(output_dir / case_name)

        batch_cfg = {
            "work_dir": str(output_dir),
            "summary": str(output_dir / "result.json"),
            "cases": self._load_history_batch_cases(load_case_override),
        }
        config["batch"] = batch_cfg
        config["postprocess"] = self._load_history_postprocess_config(
            output_dir,
            batch_cfg["cases"],
        )
        config["execution"] = {
            "interface": "slicer-workflow-replay-batch",
            "profile": self._appliedProfileName,
            "image": str(config.get("input", {}).get("image", "")),
            "mask": config.get("input", {}).get("mask"),
            "output_dir": str(output_dir),
        }
        return config

    def _current_interactive_load_case_override(self):
        _nodesets, load_case = self._interactive_nodeset_config(None)
        return load_case

    def _interactive_workflow_replay_config_for_export(
        self,
        *,
        image_path,
        mask_path,
        output_dir,
        load_case_override=None,
        nodeset_specs=None,
        material_override=None,
        mpi_processes=None,
        mpi_launcher=None,
        tolerance=None,
        export_displacements=False,
        output_fields=None,
        postprocess_config=None,
    ):
        source = getattr(self, "_appliedProfileConfig", None)
        if not isinstance(source, dict):
            return None
        if not self._has_applied_workflow_replay_model():
            return None
        source_model = source.get("model")
        if not isinstance(source_model, dict):
            return None

        config = copy.deepcopy(source)
        config.pop("batch", None)
        replay_nodesets = _workflow_replay_nodeset_specs(
            nodeset_specs if nodeset_specs is not None else config.get("nodesets", {})
        )
        if replay_nodesets:
            config["nodesets"] = replay_nodesets
        else:
            config.pop("nodesets", None)

        output_dir = Path(output_dir)
        case_name = output_dir.name or "slicer_case"
        case_cfg = config.setdefault("case", {})
        case_cfg["name"] = case_name
        case_cfg["work_dir"] = str(output_dir)

        input_cfg = config.setdefault("input", {})
        input_cfg["image"] = str(image_path)
        input_cfg.setdefault("spacing", "auto")
        input_cfg.setdefault("origin", "auto")
        if mask_path:
            input_cfg["mask"] = str(mask_path)
        else:
            input_cfg.pop("mask", None)

        _merge_workflow_replay_material_override(config, material_override)

        model_cfg = config.setdefault("model", {})
        model_cfg["density_image"] = str(image_path)
        if mask_path:
            model_cfg["mask_image"] = str(mask_path)
        else:
            model_cfg.pop("mask_image", None)
        replay_cfg = model_cfg.setdefault("workflow_replay", {})
        replay_cfg["enabled"] = True
        preprocessing = self._preprocessing_config(force=True)
        if preprocessing:
            config["preprocessing"] = preprocessing
        else:
            config.pop("preprocessing", None)
        custom_preprocessing = self._custom_preprocessing_config()
        if custom_preprocessing:
            config["custom_preprocessing"] = custom_preprocessing
        else:
            config.pop("custom_preprocessing", None)
        model_outputs = model_cfg.setdefault("outputs", {})
        model_dir = output_dir / "model"
        model_outputs["material_image"] = str(model_dir / "material.nii.gz")
        model_outputs["nodeset_image"] = str(model_dir / "nodesets.nii.gz")
        model_outputs["disk_label_image"] = str(model_dir / "disks.nii.gz")
        model_outputs["manifest"] = str(model_dir / "model.json")
        model_outputs["qc_image"] = str(model_dir / "qc.png")

        editor_dirty = self._workflow_replay_editor_dirty_for_export()
        contract_editor = getattr(self, "_workflowReplayContractEditor", None)
        stored_editor = getattr(self, "_workflowReplayResolvedEditor", None)
        if isinstance(contract_editor, dict) and contract_editor.get("planes") and not editor_dirty:
            editor = copy.deepcopy(contract_editor)
        elif isinstance(stored_editor, dict) and stored_editor.get("planes") and not editor_dirty:
            editor = copy.deepcopy(stored_editor)
        else:
            editor = self._editor_state_config()
        if not editor.get("planes"):
            source_editor = self._editor_from_active_workflow()
            if isinstance(source_editor, dict) and source_editor.get("planes"):
                editor = copy.deepcopy(source_editor)
        config["slicer_editor"] = _workflow_replay_editor_for_export_space(editor, replay_cfg)
        if load_case_override is not None:
            config["load_case"] = copy.deepcopy(load_case_override)

        output_cfg = config.setdefault("output", {})
        output_fields = list(
            output_fields
            or (["sed", "displacements"] if bool(export_displacements) else ["sed"])
        )
        output_cfg["fields"] = output_fields
        output_cfg["export_fields"] = True
        output_cfg["result"] = str(output_dir / "result.json")
        output_cfg["summary"] = output_cfg["result"]
        output_cfg["run_summary"] = str(output_dir / "summary.json")
        output_cfg["fields_dir"] = str(output_dir / "fields")
        output_cfg["visualize"] = True
        output_cfg["visualization"] = str(output_dir / "overview.png")

        solver_cfg = config.setdefault("solver", self.logic.default_solver_config())
        solver_cfg["outputs"] = output_fields
        if mpi_processes is not None:
            solver_cfg["mpi_processes"] = max(1, int(mpi_processes))
            if int(solver_cfg["mpi_processes"]) <= 1:
                solver_cfg["mpi_launcher"] = ""
            elif mpi_launcher:
                solver_cfg["mpi_launcher"] = str(mpi_launcher)
        if tolerance is not None:
            solver_cfg["tolerance"] = float(tolerance)
        if postprocess_config is not None:
            config["postprocess"] = copy.deepcopy(postprocess_config)
        return config

    def _should_normalize_xtremect_material_labels(self, material_override):
        if not isinstance(material_override, dict):
            return False
        if str(material_override.get("image_type", "")).strip().lower() not in {
            "material_labels",
            "labels",
            "segmentation",
        }:
            return False
        if self._is_xtremect_profile() or self._applied_profile_key() in {"load_history_3", "load_history_6"}:
            return True
        preset = self._widget_text(getattr(self, "materialPresetBox", None), "")
        return "xtremect" in str(preset).strip().lower()

    def _load_history_case_count(self):
        modes = self._selected_load_history_modes()
        if any(str(mode).endswith("6") for mode in modes):
            return 6
        key = self._applied_profile_key()
        if key.endswith("_6"):
            return 6
        return 3

    def _load_history_batch_cases(self, load_case_override):
        if not isinstance(load_case_override, dict):
            raise ValueError("Create contact regions before exporting a load-history profile.")
        fixed = copy.deepcopy(load_case_override.get("fixed", []))
        driver_nodeset = self._load_history_driver_nodeset(load_case_override)
        cases = [
            {
                "name_suffix": "compression_z",
                "load_case": {
                    "type": "nodeset",
                    "fixed": copy.deepcopy(fixed),
                    "prescribed": [
                        {"nodeset": driver_nodeset, "dof": "z", "value": "-1%", "units": "%"}
                    ],
                },
            },
            {
                "name_suffix": "shear_zx",
                "load_case": {
                    "type": "nodeset",
                    "fixed": copy.deepcopy(fixed),
                    "prescribed": [
                        {"nodeset": driver_nodeset, "dof": "x", "value": "1%", "units": "%"}
                    ],
                },
            },
            {
                "name_suffix": "shear_zy",
                "load_case": {
                    "type": "nodeset",
                    "fixed": copy.deepcopy(fixed),
                    "prescribed": [
                        {"nodeset": driver_nodeset, "dof": "y", "value": "1%", "units": "%"}
                    ],
                },
            },
        ]
        if self._load_history_case_count() >= 6:
            cases.extend(
                [
                    {
                        "name_suffix": "bending_x",
                        "load_case": {
                            "type": "nodeset",
                            "fixed": copy.deepcopy(fixed),
                            "prescribed": [
                                {
                                    "nodeset": driver_nodeset,
                                    "kind": "bending",
                                    "mode": "linear",
                                    "dof": "z",
                                    "value": "-1deg",
                                    "units": "deg",
                                    "gradient_axis": "y",
                                    "moment_axis": "x",
                                    "center": "centroid",
                                }
                            ],
                        },
                    },
                    {
                        "name_suffix": "bending_y",
                        "load_case": {
                            "type": "nodeset",
                            "fixed": copy.deepcopy(fixed),
                            "prescribed": [
                                {
                                    "nodeset": driver_nodeset,
                                    "kind": "bending",
                                    "mode": "linear",
                                    "dof": "z",
                                    "value": "-1deg",
                                    "units": "deg",
                                    "gradient_axis": "x",
                                    "moment_axis": "y",
                                    "center": "centroid",
                                }
                            ],
                        },
                    },
                    {
                        "name_suffix": "torsion_z",
                        "load_case": {
                            "type": "nodeset",
                            "fixed": copy.deepcopy(fixed),
                            "prescribed": [
                                {
                                    "nodeset": driver_nodeset,
                                    "kind": "torsion",
                                    "axis": "z",
                                    "value": "-1deg",
                                    "units": "deg",
                                    "center": "centroid",
                                }
                            ],
                        },
                    },
                ]
            )
        return cases

    def _load_history_driver_nodeset(self, load_case_override):
        for row in range(self._table_row_count()):
            spec = self._contact_row_spec(row)
            if str(spec.get("bc_type", "")).startswith("Load history"):
                return _safe_identifier(spec["name"] or f"nodeset_{row + 1}")
        for section in ("prescribed", "loaded"):
            entries = load_case_override.get(section, [])
            if entries:
                return str(entries[0].get("nodeset", "top"))
        raise ValueError("Load-history profiles need one non-fixed driver nodeset.")

    def _load_history_postprocess_config(self, output_dir, cases):
        case_names = [str(case["name_suffix"]) for case in cases]
        fields_dir = Path(output_dir) / "fields"
        return {
            "load_history": {
                "enabled": True,
                "method": "nnls",
                "fields": ["sed"],
                "cases": case_names,
                "summary": str(Path(output_dir) / "load_history_summary.json"),
                "output": str(fields_dir / "load_history_estimated_sed.nii.gz"),
                "final_rerun": {
                    "enabled": True,
                    "name_suffix": "load_history_final",
                    "fields": ["sed"],
                    "field": "sed",
                    "output": str(fields_dir / "load_history_final_sed.nii.gz"),
                },
            }
        }

    def _batch_workflow_config_for_export(
        self,
        *,
        image_path,
        mask_path,
        output_dir,
        nodeset_path=None,
        nodeset_specs=None,
        load_case_override=None,
        material_override=None,
        mpi_processes=None,
        mpi_launcher=None,
        tolerance=None,
        export_displacements=False,
        output_fields=None,
        postprocess_config=None,
    ):
        source = getattr(self, "_appliedProfileConfig", None)
        editor_load_history_requested = self._uses_load_history_bc()
        load_history_requested = self._is_load_history_profile() or editor_load_history_requested
        if not isinstance(source, dict):
            if not load_history_requested:
                return None
            profile_name = self._widget_text(getattr(self, "profileBox", None), "interactive_custom")
            profile_defaults = _profile_defaults(profile_name)
            solver_defaults = {}
            logic = getattr(self, "logic", None)
            if logic is not None and hasattr(logic, "default_solver_config"):
                solver_defaults = logic.default_solver_config()
            config = {
                "input": {"image_type": profile_defaults["image_type"]},
                "materials": copy.deepcopy(profile_defaults["materials"]),
                "solver": solver_defaults,
            }
        else:
            if not isinstance(source.get("batch"), dict) and not load_history_requested:
                return None
            config = copy.deepcopy(source)
            config.pop("slicer_editor", None)
            config.pop("workflow_template", None)
            config.pop("model", None)
            if load_history_requested:
                config.pop("load_case", None)

        case_name = Path(output_dir).name or "slicer_case"
        case_cfg = config.setdefault("case", {})
        case_cfg["name"] = case_name
        case_cfg["work_dir"] = str(Path(output_dir) / case_name)

        input_cfg = config.setdefault("input", {})
        input_cfg["image"] = str(image_path)
        input_cfg.setdefault("spacing", "auto")
        input_cfg.setdefault("origin", "auto")
        if mask_path:
            input_cfg["mask"] = str(mask_path)
        else:
            input_cfg.pop("mask", None)

        if material_override:
            if "image_type" in material_override:
                input_cfg["image_type"] = material_override["image_type"]
            if "materials" in material_override:
                config["materials"] = material_override["materials"]
            if "solver" in material_override:
                config["solver"] = _deep_merge_workflow_config(
                    config.get("solver", {}),
                    material_override["solver"],
                )
        if nodeset_specs:
            config["nodesets"] = copy.deepcopy(nodeset_specs)
        elif nodeset_path is not None:
            config["nodesets"] = {
                "top": {
                    "type": "label_image",
                    "image": str(nodeset_path),
                    "label": 201,
                    "selection": "interface_nodes",
                }
            }

        output_cfg = config.setdefault("output", {})
        output_fields = list(output_fields or (["sed", "displacements"] if bool(export_displacements) else ["sed"]))
        output_cfg["fields"] = output_fields
        output_cfg["export_fields"] = True
        output_cfg["result"] = str(Path(output_dir) / case_name / "result.json")
        output_cfg["summary"] = output_cfg["result"]
        output_cfg["run_summary"] = str(Path(output_dir) / case_name / "summary.json")
        output_cfg["fields_dir"] = str(Path(output_dir) / case_name / "fields")
        output_cfg["visualization"] = str(Path(output_dir) / case_name / "overview.png")

        batch_cfg = config.setdefault("batch", {})
        batch_cfg["work_dir"] = str(Path(output_dir))
        batch_cfg["summary"] = str(Path(output_dir) / "result.json")
        if load_history_requested:
            if editor_load_history_requested or isinstance(load_case_override, dict):
                batch_cfg["cases"] = self._load_history_batch_cases(load_case_override)
                config["postprocess"] = self._load_history_postprocess_config(
                    output_dir,
                    batch_cfg["cases"],
                )
            elif not batch_cfg.get("cases"):
                raise ValueError(
                    "Load-history batch export requires authored batch cases or created contact regions."
                )

        solver_cfg = config.setdefault("solver", {})
        if mpi_processes is not None:
            solver_cfg["mpi_processes"] = max(1, int(mpi_processes))
            if int(solver_cfg["mpi_processes"]) <= 1:
                solver_cfg["mpi_launcher"] = ""
            elif mpi_launcher:
                solver_cfg["mpi_launcher"] = str(mpi_launcher)
        if tolerance is not None:
            solver_cfg["tolerance"] = float(tolerance)
        solver_cfg["outputs"] = output_fields
        if postprocess_config is not None:
            if not load_history_requested:
                config["postprocess"] = copy.deepcopy(postprocess_config)

        config["execution"] = {
            "interface": "slicer-workflow-batch",
            "profile": getattr(self, "_appliedProfileName", None)
            or self._widget_text(getattr(self, "profileBox", None), "interactive_custom"),
            "image": str(image_path),
            "mask": None if mask_path is None else str(mask_path),
            "output_dir": str(output_dir),
        }
        return config

    def export_case_only(self):
        output_root = Path(self.outputDirectory.directory).expanduser().resolve()
        staging_dir = output_root / "_portable_bundle_staging"
        bundle_path = self._portable_bundle_path(None, output_root)
        selected = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save Portable ParOSol Bundle",
            str(bundle_path),
            "ParOSol bundles (*.parosol);;All files (*)",
        )
        if isinstance(selected, (list, tuple)):
            selected = selected[0]
        bundle_path = Path(str(selected)).expanduser() if selected else None
        if not bundle_path:
            return
        if not bundle_path.name.lower().endswith(".parosol"):
            bundle_path = bundle_path.with_suffix(".parosol")
        old_output = self.outputDirectory.directory
        success = False
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            self.outputDirectory.directory = str(staging_dir)
            config_path = self.export_case(dry_run=True)
            self.logic.export_parosol_input(config_path, on_output=self._append_log)
            if bundle_path.exists():
                bundle_path.unlink()
            bundle_path = self.logic.create_portable_bundle(
                config_path,
                bundle_path,
                on_output=self._append_log,
            )
            success = True
            self._show_text_dialog(
                "Portable Bundle Exported",
                (
                    f"Bundle:\n{bundle_path}\n\n"
                    "Copy this one file to the machine that should solve it, then run:\n"
                    f"parosol run {bundle_path.name} --output {bundle_path.stem}_results\n\n"
                    "The remote run writes result.json, summary.json, fields, logs, and "
                    "the ParOSol input H5 in the output folder."
                ),
                minimum_width=760,
                minimum_height=260,
            )
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
        finally:
            self.outputDirectory.directory = old_output
            if success:
                shutil.rmtree(staging_dir, ignore_errors=True)

    def _portable_bundle_path(self, config_path, output_root):
        case_name = Path(output_root).name or "parosol_case"
        try:
            import yaml

            config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            if isinstance(config, dict):
                configured_name = config.get("case", {}).get("name")
                if configured_name:
                    case_name = str(configured_name)
        except Exception:
            pass
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name).strip("._")
        if not safe_name:
            safe_name = "parosol_case"
        return Path(output_root) / f"{safe_name}.parosol"

    def _workflow_description_prompt_default(self):
        config = self._active_workflow_config()
        profile_name = self._widget_text(getattr(self, "profileBox", None), "interactive_custom")
        requirements = self._workflow_input_requirements(config, profile_name)
        lines = []
        summary = str(requirements.get("summary", "")).strip()
        if summary:
            lines.append(summary)
        for section, label in (
            ("image_labels", "Image labels"),
            ("mask_labels", "Mask labels"),
        ):
            labels = requirements.get(section, {})
            if isinstance(labels, dict) and labels:
                label_text = ", ".join(f"{key}={value}" for key, value in sorted(labels.items()))
                lines.append(f"{label}: {label_text}.")
        return "\n".join(lines)

    def _prompt_workflow_description(self):
        selected = qt.QInputDialog.getMultiLineText(
            slicer.util.mainWindow(),
            "Describe Workflow",
            "Describe this workflow in a few words, including required inputs.",
            self._workflow_description_prompt_default(),
        )
        if isinstance(selected, tuple):
            description, accepted = selected[0], bool(selected[1])
        else:
            description, accepted = selected, True
        if not accepted:
            return None
        return str(description or "").strip()

    def save_workflow_template(self):
        description = self._prompt_workflow_description()
        if description is None:
            return
        default_dir = USER_WORKFLOW_ROOT
        default_dir.mkdir(parents=True, exist_ok=True)
        selected = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save ParOSol Workflow",
            str(default_dir / "parosol_workflow.parosol-workflow"),
            "ParOSol workflows (*.parosol-workflow);;All files (*)",
        )
        if isinstance(selected, tuple):
            selected = selected[0]
        if not selected:
            return
        workflow_bundle = Path(selected).expanduser().resolve()
        if not workflow_bundle.name.lower().endswith(".parosol-workflow"):
            workflow_bundle = workflow_bundle.with_suffix(".parosol-workflow")
        staging_dir = workflow_bundle.parent / f".{workflow_bundle.stem}_staging"
        old_output = self.outputDirectory.directory
        success = False
        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            self.outputDirectory.directory = str(staging_dir)
            config_path = self.export_case(dry_run=True)
            workflow_path = staging_dir / "workflow.yaml"
            self._write_workflow_template(config_path, workflow_path, description=description)
            self._write_workflow_bundle(staging_dir, workflow_bundle)
            self._register_user_workflow_profile(workflow_bundle, description=description)
            success = True
            self._append_log(f"Saved workflow: {workflow_bundle}\n")
            self._show_text_dialog(
                "Workflow Saved",
                (
                    f"Workflow:\n{workflow_bundle}\n\n"
                    "Run from the command line with:\n"
                    "parosol NEW_SCAN.nii.gz --profile interactive_custom "
                    f"--template {workflow_bundle} --output OUT_DIR"
                ),
                minimum_width=720,
                minimum_height=220,
            )
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
        finally:
            self.outputDirectory.directory = old_output
            if success:
                shutil.rmtree(staging_dir, ignore_errors=True)

    def _register_user_workflow_profile(self, workflow_bundle, description=""):
        try:
            from bone_imaging_derivatives import register_profile_asset

            register_profile_asset(
                "parosol-fea",
                Path(workflow_bundle).stem.replace("_", " ").replace("-", " ").title(),
                workflow_bundle,
                kind="parosol-workflow",
                metadata={"description": str(description or "").strip()},
            )
        except Exception as exc:
            self._append_log(f"Saved workflow, but could not update shared profile registry: {exc}\n")

    def _write_workflow_bundle(self, template_dir, workflow_bundle):
        template_dir = Path(template_dir)
        workflow_bundle = Path(workflow_bundle)
        workflow_bundle.parent.mkdir(parents=True, exist_ok=True)
        files = [
            path
            for path in sorted(template_dir.rglob("*"))
            if path.is_file() and path.resolve() != workflow_bundle.resolve()
            and path.name not in WORKFLOW_BUNDLE_EXCLUDED_FILES
        ]
        manifest = {
            "format": "parosol-py-workflow",
            "version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "workflow.yaml",
            "files": [path.relative_to(template_dir).as_posix() for path in files],
        }
        if workflow_bundle.exists():
            workflow_bundle.unlink()
        with zipfile.ZipFile(workflow_bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
            for path in files:
                archive.write(path, path.relative_to(template_dir).as_posix())

    def _write_workflow_template(self, config_path, workflow_path, description=None):
        import yaml

        config_path = Path(config_path)
        workflow_path = Path(workflow_path)
        template_dir = workflow_path.parent
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Exported config is not a mapping: {config_path}")
        description_text = str(description or "").strip()
        material_override = self._material_override()
        if isinstance(material_override, dict):
            input_cfg = config.setdefault("input", {})
            if "image_type" in material_override:
                input_cfg["image_type"] = material_override["image_type"]
            if "materials" in material_override:
                config["materials"] = material_override["materials"]
            if "solver" in material_override:
                config["solver"] = _deep_merge_workflow_config(
                    config.get("solver", {}),
                    material_override["solver"],
                )
        workflow_meta = config.setdefault("workflow_template", {})
        config["slicer_editor"] = self._editor_state_config()
        if bool(getattr(self.icpRegistrationCheckBox, "checked", False)):
            for plane in config.get("slicer_editor", {}).get("planes", []):
                if isinstance(plane, dict):
                    plane["reference_space"] = True
        preprocessing = self._preprocessing_config(force=True)
        if preprocessing:
            config["preprocessing"] = preprocessing
        custom_preprocessing = self._custom_preprocessing_config()
        if custom_preprocessing:
            config["custom_preprocessing"] = custom_preprocessing
        else:
            config.pop("custom_preprocessing", None)
        custom_preprocessing = config.get("custom_preprocessing")
        if isinstance(custom_preprocessing, dict):
            self._bundle_custom_preprocessing_script(custom_preprocessing, template_dir)
        profile_name = self._widget_text(self.profileBox, "interactive_custom")
        model_cfg = config.setdefault("model", {})
        if not isinstance(model_cfg, dict):
            model_cfg = {}
            config["model"] = model_cfg
        registration_cfg = model_cfg.setdefault("registration", {})
        if not isinstance(registration_cfg, dict):
            registration_cfg = {}
            model_cfg["registration"] = registration_cfg
        if bool(getattr(self.icpRegistrationCheckBox, "checked", False)):
            selected_target_source = self._selected_icp_target_source()
            reference_points = self._write_current_icp_reference_points(
                template_dir,
                config=config,
                base_dir=config_path.parent,
            )
            icp_target = _first_int_text(self._selected_icp_target_values())
            registration_cfg.update(
                {
                    "enabled": True,
                    "method": registration_cfg.get("method", "lightweight_icp"),
                    "reference_points": reference_points,
                    "target_image": "workflow-reference",
                    "authored_target_image": selected_target_source,
                    "max_points": int(registration_cfg.get("max_points", 8000)),
                    "iterations": int(registration_cfg.get("iterations", 50)),
                    "source_landmark_mode": registration_cfg.get("source_landmark_mode", "linspace"),
                    "reference_landmark_mode": registration_cfg.get("reference_landmark_mode", "linspace"),
                }
            )
            if selected_target_source == "slicer-node":
                registration_cfg["reference_source_node"] = _node_reference_description(
                    self._selected_icp_target_node()
                )
            registration_cfg.pop("reference_authoring", None)
            registration_cfg.pop("self_reference", None)
            if icp_target:
                registration_cfg["target_label"] = int(icp_target)
                model_cfg.setdefault("targets", {})["registration"] = int(icp_target)
        else:
            registration_cfg["enabled"] = False
        input_cfg = config.setdefault("input", {})
        if isinstance(input_cfg, dict):
            input_cfg.pop("image", None)
            input_cfg.pop("mask", None)
        input_requirements = self._workflow_input_requirements(config, profile_name)
        if description_text:
            input_requirements["author_description"] = description_text
        config["input_requirements"] = input_requirements
        workflow_meta.update(
            {
                "version": 1,
                "created_by": "SlicerParOSol",
                "type": "single_case_fea",
                "profile": profile_name,
                "requires_reference": bool(registration_cfg.get("enabled", False)),
                "reference": {
                    "nodesets": "nodesets.nii.gz",
                    "disk_labels": "disk_labels.nii.gz",
                },
                "registration": {
                    "enabled": bool(registration_cfg.get("enabled", False)),
                    "method": "lightweight_icp",
                    "status": "reference_point_cloud_saved" if registration_cfg.get("enabled", False) else "disabled",
                },
                "notes": (
                    "This template preserves the authored workflow, preprocessing choices, "
                    "editable contact planes, load definitions, and optional ICP reference files."
                ),
            }
        )
        if description_text:
            workflow_meta["description"] = description_text
        self._relativize_workflow_paths(config, template_dir)
        workflow_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def _bundle_custom_preprocessing_script(self, custom_preprocessing, template_dir):
        options = _custom_preprocessing_options(custom_preprocessing)
        if options:
            custom_dir = Path(template_dir) / "custom_preprocessing"
            custom_dir.mkdir(parents=True, exist_ok=True)
            for option in options:
                self._bundle_custom_preprocessing_option(option, custom_dir, template_dir)
            return
        custom_dir = Path(template_dir) / "custom_preprocessing"
        custom_dir.mkdir(parents=True, exist_ok=True)
        self._bundle_custom_preprocessing_option(custom_preprocessing, custom_dir, template_dir)

    def _bundle_custom_preprocessing_option(self, custom_preprocessing, custom_dir, template_dir):
        script = custom_preprocessing.get("script") or custom_preprocessing.get("path")
        if not script:
            return
        source = Path(str(script)).expanduser()
        if not source.is_absolute():
            source = (Path(template_dir) / source).resolve()
        if not source.is_file():
            raise ValueError(f"Custom preprocessing script does not exist: {source}")
        target = Path(custom_dir) / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        custom_preprocessing["script"] = str(target)
        custom_preprocessing.pop("path", None)

    def _write_current_icp_reference_points(self, template_dir, *, config=None, base_dir=None):
        template_dir = Path(template_dir)
        reference_dir = template_dir / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        output_path = reference_dir / "slicer_reference_points.npz"
        source = self._selected_icp_target_source()
        points = None
        if source == "workflow-reference":
            reference = self._current_workflow_reference_points()
            if reference:
                active_config = self._active_workflow_config()
                model = active_config.get("model", {}) if isinstance(active_config, dict) else {}
                registration = model.get("registration", {}) if isinstance(model, dict) else {}
                max_points = int(registration.get("max_points", 8000)) if isinstance(registration, dict) else 8000
                points = read_reference_points(
                    reference,
                    max_points=max_points,
                    coordinate_system=self._current_workflow_reference_coordinate_system(),
                )
            else:
                source = "self"
        if source == "slicer-node":
            max_points = int(
                (config or {}).get("model", {}).get("registration", {}).get("max_points", 8000)
            ) if isinstance(config, dict) else 8000
            points = self._icp_reference_points_from_selected_target_node(max_points=max_points)
        if source == "self":
            points = self._icp_reference_points_for_workflow_save(config=config, base_dir=base_dir)
        if points is None:
            reference_node = self._volume()
            mask_like_node = self.maskSelector.currentNode() or reference_node
            points = _sample_reference_points_from_mask_like(
                mask_like_node,
                reference_node,
                max_points=8000,
                active_values=self._selected_icp_target_values(),
            )
        if points.size == 0:
            raise ValueError("Cannot save ICP reference: current mask/image has no non-zero surface points.")
        np.savez_compressed(output_path, points=np.asarray(points, dtype=np.float32))
        return output_path.relative_to(template_dir).as_posix()

    def _icp_reference_points_from_selected_target_node(self, *, max_points=8000):
        target_node = self._selected_icp_target_node()
        if target_node is None:
            raise ValueError("ICP target image is set to Slicer node, but no target node is selected.")
        reference_node = self._volume()
        if reference_node is None and _is_segmentation_node(target_node):
            raise ValueError("A current image is required to sample an ICP target segmentation.")
        sampling_reference = reference_node if _is_segmentation_node(target_node) else target_node
        points = _sample_reference_points_from_mask_like(
            target_node,
            sampling_reference,
            max_points=max_points,
            active_values=self._selected_icp_target_values(),
        )
        if points.size == 0:
            raise ValueError("Selected ICP target node has no non-zero target points.")
        return np.asarray(points, dtype=np.float32)

    def _cached_icp_reference_points_for_current_preview(self, expected_reference=None):
        info = getattr(self, "_lastIcpAlignment", None)
        if not isinstance(info, dict):
            return None
        points = info.get("reference_points")
        if not isinstance(points, np.ndarray) or points.size == 0:
            return None
        if (
            expected_reference
            and info.get("reference")
            and str(expected_reference) != str(info.get("reference"))
        ):
            return None
        reference_node = self._volume()
        if reference_node is None or reference_node.GetID() != info.get("image_id"):
            return None
        mask_node = self.maskSelector.currentNode() if hasattr(self, "maskSelector") else None
        stored_mask_id = str(info.get("mask_id", ""))
        if stored_mask_id and (mask_node is None or mask_node.GetID() != stored_mask_id):
            return None
        return np.asarray(points, dtype=np.float32)

    def _icp_reference_points_for_workflow_save(self, *, config=None, base_dir=None):
        cached = self._cached_self_reference_points_for_current_preview()
        if cached is not None:
            return cached
        if not isinstance(config, dict):
            return None
        model_cfg = copy.deepcopy(config.get("model", {}))
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        input_cfg = config.get("input", {})
        if isinstance(input_cfg, dict):
            if not model_cfg.get("density_image") and input_cfg.get("image"):
                model_cfg["density_image"] = str(input_cfg["image"])
            if not model_cfg.get("mask_image") and input_cfg.get("mask"):
                model_cfg["mask_image"] = str(input_cfg["mask"])
        if not model_cfg.get("density_image"):
            return None
        _prepare_parosol_py_runtime_import()
        from parosol_py.modeling.common import build_preprocessed_inputs_preview

        preview = build_preprocessed_inputs_preview(
            model_cfg,
            base_dir=Path(base_dir or "."),
            preprocessing_config=config.get("preprocessing"),
            custom_preprocessing_config=config.get("custom_preprocessing"),
        )
        return _sample_reference_points_from_parosol_preview_mask(
            preview.mask_zyx,
            spacing=preview.spacing,
            origin=preview.origin,
            max_points=int(model_cfg.get("registration", {}).get("max_points", 8000))
            if isinstance(model_cfg.get("registration", {}), dict)
            else 8000,
            active_values=self._selected_icp_target_values(),
        )

    def _cached_self_reference_points_for_current_preview(self):
        info = getattr(self, "_lastIcpAlignment", None)
        if not isinstance(info, dict) or not info.get("self_reference"):
            return None
        return self._cached_icp_reference_points_for_current_preview()

    def _editor_state_config(self):
        planes = []
        loads = []
        for row, row_data in enumerate(self.contactPlaneRows):
            spec = self._contact_row_spec(row)
            plane = row_data.get("plane")
            plane_config = {
                "name": spec["name"],
                "axis": spec["axis"],
                "normal": spec["normal"],
                "contact": spec["contact"],
                "surface_mode": spec.get("surface_mode", "project"),
                "bc_mode": _profile_bc_mode(spec["bc_type"]),
                "direction": spec["direction"],
                "shape": spec["shape"],
                "thickness_mm": float(spec["thickness"]),
                "intrusion_depth_mm": float(spec["intrusion"]),
                "use_plane_size": bool(spec["use_plane_size"]),
                "disk": {"E": float(spec["disk_e"]), "nu": float(spec["disk_nu"])},
            }
            if spec.get("anatomy_constrained"):
                plane_config["anatomy_constrained"] = True
            if spec.get("disk_target_values"):
                plane_config["disk_target"] = int(spec["disk_target_values"][0])
            fixed_dofs = _valid_fixed_dofs(spec.get("fixed_dofs"))
            if fixed_dofs is not None and fixed_dofs != ["x", "y", "z"]:
                plane_config["fixed_dofs"] = fixed_dofs
            if plane is not None:
                try:
                    center, normal, u_axis, v_axis, half_u, half_v = _plane_geometry(
                        plane,
                        shape=spec["shape"],
                        radius_mm=spec["radius"],
                        square_width_mm=spec["radius"] * 2.0,
                        hex_radius_mm=spec["radius"],
                        use_plane_size=spec["use_plane_size"],
                    )
                except Exception:
                    center = _plane_center(plane) or [0.0, 0.0, 0.0]
                    normal = _plane_normal_world(plane)
                    u_axis, v_axis = _plane_axes_from_plane(plane, _normalized(normal))
                    half_u, half_v = _plane_half_size(
                        plane,
                        shape=spec["shape"],
                        radius_mm=spec["radius"],
                        square_width_mm=spec["radius"] * 2.0,
                        hex_radius_mm=spec["radius"],
                        use_plane_size=spec["use_plane_size"],
                    )
                plane_config["center_ras"] = [float(value) for value in center]
                plane_config["normal_ras"] = [float(value) for value in normal]
                plane_config["u_axis_ras"] = [float(value) for value in u_axis]
                plane_config["v_axis_ras"] = [float(value) for value in v_axis]
                plane_config["size_mm"] = [float(half_u) * 2.0, float(half_v) * 2.0]
            planes.append(plane_config)
            load_config = {
                "nodeset": spec["name"],
                "mode": _profile_bc_mode(spec["bc_type"]),
                "direction": spec["direction"],
                "vector_ras": [
                    _float_or_zero(spec["x"]),
                    _float_or_zero(spec["y"]),
                    _float_or_zero(spec["z"]),
                ],
                "value": _float_or_zero(spec["value"]),
                "units": spec["units"],
            }
            if fixed_dofs is not None and fixed_dofs != ["x", "y", "z"]:
                load_config["fixed_dofs"] = fixed_dofs
            loads.append(load_config)
        return {"version": 1, "planes": planes, "loads": loads}

    def _relativize_workflow_paths(self, config, template_dir):
        def rel(path_text):
            if not path_text:
                return path_text
            path = Path(path_text).expanduser()
            try:
                return str(path.resolve().relative_to(template_dir.resolve()))
            except Exception:
                return str(path_text)

        input_cfg = config.get("input", {})
        if isinstance(input_cfg, dict):
            for key in ("image", "mask"):
                if key in input_cfg:
                    input_cfg[key] = rel(input_cfg[key])
        nodesets = config.get("nodesets", {})
        if isinstance(nodesets, dict):
            for spec in nodesets.values():
                if isinstance(spec, dict) and "image" in spec:
                    spec["image"] = rel(spec["image"])
        model = config.get("model", {})
        if isinstance(model, dict):
            for key in ("density_image", "mask_image"):
                if key in model:
                    model[key] = rel(model[key])
            registration = model.get("registration", {})
            if isinstance(registration, dict) and "reference_points" in registration:
                registration["reference_points"] = rel(registration["reference_points"])
        custom_preprocessing = config.get("custom_preprocessing", {})
        if isinstance(custom_preprocessing, dict):
            if "script" in custom_preprocessing:
                custom_preprocessing["script"] = rel(custom_preprocessing["script"])
            for option in _custom_preprocessing_options(custom_preprocessing):
                if "script" in option:
                    option["script"] = rel(option["script"])

    def run_case(self):
        return self._executionController.run_case()

    def _validate_inputs_for_selected_workflow(self):
        if self._volume() is None:
            message = "Select an image or material-label volume before running."
            self._set_run_status(message, error=True)
            raise ValueError(message)
        profile_text = self._widget_text(getattr(self, "profileBox", None), "interactive_custom")
        config = self._active_workflow_config()
        if self._workflow_requires_mask(config, profile_text) and self.maskSelector.currentNode() is None:
            message = (
                f"Workflow '{profile_text}' requires a mask or label segmentation. "
                "Select the segmentation/labelmap that contains the anatomy labels, then run again."
            )
            self._set_run_status(message, error=True)
            raise ValueError(message)
        self._update_input_readiness_status()

    def _prepare_selected_workflow_for_visible_run(self):
        profile_text = self._widget_text(getattr(self, "profileBox", None), "interactive_custom")
        if str(profile_text).strip().lower() != "interactive_custom":
            self._apply_profile_without_tab_advance()
        if self._is_interactive_profile() and self._volume() is not None:
            self._prepare_visible_workflow_stages_for_run()

    def _prepare_visible_workflow_stages_for_run(self):
        if not self._is_interactive_profile() or self._volume() is None:
            return
        state = self._stage_state()
        if state.anatomy_dirty or self._workflow_has_preprocessing_to_apply():
            self.preprocess_inputs()
            state = self._stage_state()
            if state.anatomy_dirty:
                raise RuntimeError("Prepare Image did not complete; run stopped before export.")

        if not getattr(self, "contactPlaneRows", []):
            editor = self._editor_from_applied_profile()
            if isinstance(editor, dict):
                resolved_editor = (
                    self._resolve_reference_space_editor_for_current_sample(editor)
                    if self._editor_needs_reference_resolution(editor)
                    else editor
                )
                resolved_editor = self._resolve_bbox_relative_editor_for_current_sample(resolved_editor)
                self._apply_profile_planes_and_loads(resolved_editor)

        nodesets = self.bcLabelSelector.currentNode()
        if state.boundary_dirty or nodesets is None:
            self.preview_disks()
            state = self._stage_state()
            nodesets = self.bcLabelSelector.currentNode()
            if state.boundary_dirty or nodesets is None:
                raise RuntimeError("Contact regions were not created; run stopped before export.")

        if nodesets is not None and (state.loads_dirty or not self._has_visible_load_preview()):
            self.preview_loads()
            state = self._stage_state()
            if state.loads_dirty or not self._has_visible_load_preview():
                raise RuntimeError("Load preview did not complete; run stopped before export.")

    def _has_visible_load_preview(self):
        for attr_name in ("bcArrowNodes", "bcMarkerNodes"):
            for node in getattr(self, attr_name, []) or []:
                try:
                    if node is not None and slicer.mrmlScene.IsNodePresent(node):
                        return True
                except Exception:
                    continue
        return False

    def _run_case_impl(self):
        try:
            self._validate_inputs_for_selected_workflow()
            self._start_run_status("Preparing workflow...")
            if self._fast_recipe_run_enabled():
                self._append_log("Fast recipe run: skipping visual stage preparation.\n")
            else:
                self._prepare_selected_workflow_for_visible_run()
            self._set_run_status("Exporting solver inputs...")
            config_path = self.export_case(dry_run=False)
            _remove_stale_result_fields(Path(self.outputDirectory.directory))
            self.runButton.enabled = False
            if hasattr(self, "quickRunButton"):
                self.quickRunButton.enabled = False
            self.stopButton.enabled = True
            if hasattr(self, "quickStopButton"):
                self.quickStopButton.enabled = True
            command = "batch" if _config_file_has_batch(config_path) else "run"
            self._set_run_status("Running ParOSol...")
            self.logic.run_parosol(
                [command, str(config_path)],
                on_output=self._append_log,
                on_finished=self._on_finished,
                cwd=config_path.parent,
            )
        except Exception as exc:
            self.runButton.enabled = True
            if hasattr(self, "quickRunButton"):
                self.quickRunButton.enabled = True
            self.stopButton.enabled = False
            if hasattr(self, "quickStopButton"):
                self.quickStopButton.enabled = False
            self._finish_run_status(f"Run failed: {exc}", error=True)
            slicer.util.errorDisplay(str(exc))

    def stop_case(self):
        self.logic.interrupt()

    def discover_fea_batch(self):
        root = str(getattr(self.batchDatasetRootSelector, "currentPath", "") or "").strip()
        if not root:
            slicer.util.errorDisplay("Select a dataset root before discovery.")
            return
        try:
            self._feaBatchCases = self.logic.discover_fea_batch_cases(
                root,
                subject_id=str(self.batchSubjectEdit.text or "").strip(),
                site=str(self.batchSiteEdit.text or "").strip(),
                session_id=str(self.batchSessionEdit.text or "").strip(),
            )
        except Exception as exc:
            self.batchStatusLabel.text = f"Discovery failed: {exc}"
            slicer.util.errorDisplay(str(exc))
            return
        self._populate_fea_batch_table()
        self._populate_fea_batch_role_selectors()
        self._refresh_fea_batch_readiness()
        self._append_log(
            f"[fea batch] discovered {len(self._feaBatchCases)} case(s) from {root}\n"
        )

    def _populate_fea_batch_table(self):
        table = self.batchTable
        table.setRowCount(len(self._feaBatchCases))
        for row, case in enumerate(self._feaBatchCases):
            self._set_fea_batch_action(row, "Run")
            model = self._fea_batch_model_artifact(case)
            values = [
                case.subject_id,
                case.site,
                case.session_id or "",
                Path(model.path).name if model is not None else "",
                "",
            ]
            for column, value in enumerate(values, start=1):
                item = qt.QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                table.setItem(row, column, item)
        try:
            table.resizeColumnsToContents()
        except Exception:
            pass

    def _set_fea_batch_action(self, row, label):
        button = qt.QPushButton(str(label))
        button.enabled = str(label) not in {"Queued", "Running", "Not implemented", "Missing input"}
        if str(label) == "Load":
            button.clicked.connect(lambda _checked=False, index=row: self._load_fea_batch_row_outputs(index))
        elif str(label) == "Run":
            button.clicked.connect(lambda _checked=False, index=row: self._queue_fea_batch_row(index))
        self.batchTable.setCellWidget(int(row), 0, button)

    def _set_fea_batch_status(self, row, status):
        row = int(row)
        row_count_value = getattr(self.batchTable, "rowCount", 0)
        row_count = int(row_count_value() if callable(row_count_value) else row_count_value)
        if 0 <= row < row_count:
            item = self.batchTable.item(row, 5)
            if item is None:
                item = qt.QTableWidgetItem("")
                item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                self.batchTable.setItem(row, 5, item)
            item.setText(str(status))

    def _fea_batch_model_artifact(self, case):
        selected = self._selected_fea_batch_roles()
        workflow = str(self.batchWorkflowBox.currentText or "").strip()
        roles = (selected.get("image"),) if selected.get("image") else role_options_for_workflow([case], workflow, "image")
        if not roles:
            roles = ("material_labelmap", "hom_ls_model", "model_labelmap", "labelmap")
        return case.first_artifact(role for role in roles if role)

    def _fea_batch_command_for_row(self, row):
        row = int(row)
        if row < 0 or row >= len(self._feaBatchCases):
            return None
        root = str(getattr(self.batchDatasetRootSelector, "currentPath", "") or "").strip()
        if not root:
            return None
        workflow = str(self.batchWorkflowBox.currentText or "").strip()
        commands = self.logic.build_fea_batch_commands(
            root,
            [self._feaBatchCases[row]],
            workflow=workflow,
            selected_roles=self._selected_fea_batch_roles(),
            dry_run=bool(self.batchDryRunCheckBox.checked),
        )
        return commands[0] if commands else None

    def _fea_batch_output_dir_for_row(self, row):
        command = self._fea_batch_command_for_row(row)
        if not command:
            return None
        context = parosol_command_derivative_context(command)
        output_dir = context.get("output_dir") if context else ""
        return Path(output_dir) if output_dir else None

    def _populate_fea_batch_role_selectors(self):
        signal_states = []
        for box in (self.batchImageRoleBox, self.batchMaskRoleBox):
            try:
                signal_states.append((box, box.blockSignals(True)))
            except Exception:
                signal_states.append((box, False))
            box.clear()
            box.addItem("Auto")
        workflow = str(self.batchWorkflowBox.currentText or "").strip()
        for role in role_options_for_workflow(self._feaBatchCases, workflow, "image"):
            self.batchImageRoleBox.addItem(role)
        for role in role_options_for_workflow(self._feaBatchCases, workflow, "mask"):
            self.batchMaskRoleBox.addItem(role)
        for box, was_blocked in signal_states:
            try:
                box.blockSignals(was_blocked)
            except Exception:
                pass

    def _on_fea_batch_workflow_changed(self):
        self._populate_fea_batch_role_selectors()
        self._refresh_fea_batch_readiness()

    def _selected_fea_batch_roles(self):
        selected = {}
        for group, box in (("image", self.batchImageRoleBox), ("mask", self.batchMaskRoleBox)):
            role = str(box.currentText or "").strip()
            if role and role != "Auto":
                selected[group] = role
        return selected

    def _selected_fea_batch_cases(self):
        selected = []
        for row, case in enumerate(self._feaBatchCases):
            item = self.batchTable.item(row, 0)
            checked = item is None or item.checkState() == qt.Qt.Checked
            if checked:
                selected.append(case)
        return selected

    def _refresh_fea_batch_readiness(self):
        cases = list(getattr(self, "_feaBatchCases", []) or [])
        if not cases:
            self.batchStatusLabel.text = "Discover a dataset to prepare FEA batch cases."
            return
        workflow = str(self.batchWorkflowBox.currentText or "").strip()
        supported, support_message = batch_profile_support_status(workflow)
        selected_roles = self._selected_fea_batch_roles()
        ready = 0
        missing_counts = {}
        for row, case in enumerate(cases):
            if not supported:
                self._set_fea_batch_action(row, "Not implemented")
                self._set_fea_batch_status(row, "Not implemented")
                continue
            ok, missing = case_readiness(case, workflow, selected_roles)
            output_dir = self._fea_batch_output_dir_for_row(row)
            has_results = bool(output_dir and (output_dir / "result.json").exists())
            if ok:
                ready += 1
                if has_results:
                    self._set_fea_batch_action(row, "Load")
                    self._set_fea_batch_status(row, "Done")
                else:
                    self._set_fea_batch_action(row, "Run")
                    self._set_fea_batch_status(row, "Ready")
            else:
                self._set_fea_batch_action(row, "Missing input")
                self._set_fea_batch_status(row, "Missing model labelmap" if "image" in missing else "Missing input")
            for group in missing:
                missing_counts[group] = missing_counts.get(group, 0) + 1
        if not supported:
            self.batchStatusLabel.text = support_message
            return
        if missing_counts:
            missing_text = ", ".join(f"{name}: {count}" for name, count in sorted(missing_counts.items()))
            self.batchStatusLabel.text = (
                f"{ready}/{len(cases)} case(s) ready for {workflow or 'selected workflow'}; missing {missing_text}."
            )
        else:
            self.batchStatusLabel.text = f"{ready}/{len(cases)} case(s) ready for {workflow or 'selected workflow'}."

    def run_fea_batch(self):
        if getattr(self.logic, "_proc", None) is not None:
            slicer.util.errorDisplay("FEA batch is already running.")
            return
        workflow = str(self.batchWorkflowBox.currentText or "").strip()
        supported, support_message = batch_profile_support_status(workflow)
        if not supported:
            slicer.util.errorDisplay(support_message)
            return
        rows = []
        for row, case in enumerate(getattr(self, "_feaBatchCases", []) or []):
            ok, _missing = case_readiness(case, workflow, self._selected_fea_batch_roles())
            if ok:
                rows.append(row)
        if not rows:
            slicer.util.errorDisplay("No discovered FEA cases have the material labelmap required by this workflow.")
            return
        self._feaBatchQueue = list(rows)
        for row in self._feaBatchQueue:
            self._set_fea_batch_action(row, "Queued")
            self._set_fea_batch_status(row, "Queued")
        self.batchRunButton.enabled = False
        self.batchStopButton.enabled = True
        self._append_log(f"[fea batch] queued {len(self._feaBatchQueue)} case(s)\n")
        self._start_next_fea_batch_job()

    def _queue_fea_batch_row(self, row):
        row = int(row)
        if row < 0 or row >= len(getattr(self, "_feaBatchCases", []) or []):
            return
        if row in getattr(self, "_feaBatchQueue", []):
            return
        case = self._feaBatchCases[row]
        ok, _missing = case_readiness(case, str(self.batchWorkflowBox.currentText or "").strip(), self._selected_fea_batch_roles())
        if not ok:
            slicer.util.errorDisplay("This FEA row is missing a model labelmap.")
            self._refresh_fea_batch_readiness()
            return
        self._feaBatchQueue.append(row)
        self._set_fea_batch_action(row, "Queued")
        self._set_fea_batch_status(row, "Queued")
        self.batchRunButton.enabled = False
        self.batchStopButton.enabled = True
        self._start_next_fea_batch_job()

    def _start_next_fea_batch_job(self):
        if getattr(self.logic, "_proc", None) is not None:
            return
        if not self._feaBatchQueue:
            self._feaBatchCurrent = None
            self.batchRunButton.enabled = True
            self.batchStopButton.enabled = False
            self.batchStatusLabel.text = "FEA batch idle."
            self._append_log("[fea batch] queue finished\n")
            return
        row = int(self._feaBatchQueue.pop(0))
        command = self._fea_batch_command_for_row(row)
        if not command:
            self._set_fea_batch_action(row, "Missing input")
            self._set_fea_batch_status(row, "Missing model labelmap")
            self._start_next_fea_batch_job()
            return
        self._feaBatchCurrent = {"row": row, "command": command}
        self._set_fea_batch_action(row, "Running")
        self._set_fea_batch_status(row, "Running")
        self.batchStatusLabel.text = f"Running FEA batch row {row + 1}/{len(self._feaBatchCases)}: {Path(command[0]).name}"
        self._append_log(f"[fea batch] running row {row + 1}: {' '.join(command)}\n")
        output_dir = parosol_command_derivative_context(command).get("output_dir")
        if output_dir:
            _remove_stale_result_fields(Path(output_dir))
        self.logic.run_parosol(
            command,
            on_output=self._append_log,
            on_finished=self._on_fea_batch_case_finished,
            cwd=Path(command[0]).parent,
        )

    def _on_fea_batch_case_finished(self, exit_code, interrupted):
        current = self._feaBatchCurrent or {}
        row = current.get("row")
        command = current.get("command")
        self._feaBatchCurrent = None
        if interrupted:
            if row is not None:
                self._set_fea_batch_action(row, "Run")
                self._set_fea_batch_status(row, "Canceled")
            self.batchRunButton.enabled = True
            self.batchStopButton.enabled = False
            self._feaBatchQueue = []
            self.batchStatusLabel.text = "FEA batch stopped."
            self._append_log("[fea batch] stopped\n")
            return
        if int(exit_code) != 0:
            if row is not None:
                self._set_fea_batch_action(row, "Run")
                self._set_fea_batch_status(row, f"Failed ({exit_code})")
            self.batchRunButton.enabled = True
            self.batchStopButton.enabled = False
            self._feaBatchQueue = []
            self.batchStatusLabel.text = f"FEA batch failed at row {int(row) + 1 if row is not None else '?'}."
            self._append_log(f"[fea batch] failed with exit code {exit_code}\n")
            return
        if command:
            self._write_completed_fea_batch_manifest(command)
            context = parosol_command_derivative_context(command)
            if row is not None and context.get("output_dir"):
                self._feaBatchRowOutputs[int(row)] = context["output_dir"]
        if row is not None:
            self._set_fea_batch_action(row, "Load")
            self._set_fea_batch_status(row, "Done")
        self._start_next_fea_batch_job()

    def stop_fea_batch(self):
        self._feaBatchQueue = []
        self.logic.interrupt()

    def _load_fea_batch_row_outputs(self, row):
        output_dir = self._feaBatchRowOutputs.get(int(row))
        if not output_dir:
            resolved = self._fea_batch_output_dir_for_row(row)
            output_dir = str(resolved) if resolved is not None else ""
        if not output_dir or not Path(output_dir).exists():
            slicer.util.errorDisplay("No result available for this FEA batch row.")
            return
        self.outputDirectory.directory = str(output_dir)
        self.load_results()
        self.batchStatusLabel.text = f"Loaded FEA results: {Path(output_dir).name}"

    def _write_completed_fea_batch_manifest(self, command):
        context = parosol_command_derivative_context(command)
        if not context:
            return
        try:
            manifest_path = _write_parosol_run_derivative_manifest(
                context["output_dir"],
                dataset_root=context["dataset_root"],
                subject_id=context["subject_id"],
                site=context["site"],
                session_id=context["session_id"],
            )
            self._append_log(f"[fea batch] FEA derivative manifest: {manifest_path}\n")
        except Exception as exc:
            self._append_log(f"[fea batch] Could not write FEA derivative manifest: {exc}\n")

    def _on_finished(self, exit_code, interrupted):
        self.runButton.enabled = True
        if hasattr(self, "quickRunButton"):
            self.quickRunButton.enabled = True
        self.stopButton.enabled = False
        if hasattr(self, "quickStopButton"):
            self.quickStopButton.enabled = False
        if interrupted:
            self._finish_run_status("Run stopped.", error=True)
        elif exit_code == 0:
            self._finish_run_status("Run finished.")
        else:
            self._finish_run_status(f"Run failed with exit code {exit_code}.", error=True)
        self._append_log(f"\n[process] finished exit_code={exit_code} interrupted={interrupted}\n")
        if exit_code == 0 and not interrupted:
            self.load_results()
            self._advance_workflow_tab_after("results")

    def load_results(self):
        output_dir = Path(self.outputDirectory.directory)
        result_path = output_dir / "result.json"
        data = None
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            self.resultText.setHtml(self._format_result_html(data))
        if self._export_displacements_enabled():
            try:
                self.logic.export_displacement_components_from_run(
                    output_dir,
                    on_output=self._append_log,
                )
            except Exception as exc:
                self._append_log(f"Could not export displacement components: {exc}\n")
        self._load_selected_result_fields(output_dir)
        overview_path = output_dir / "overview.png"
        if overview_path.exists():
            self._append_log(f"Overview: {overview_path}\n")
        if self._displacement_component_paths(output_dir):
            self._append_log(
                "Displacement components available; use Show Deformation Arrows to inspect displacement direction.\n"
            )
        context = self._fea_derivative_context(output_dir)
        if context:
            try:
                manifest_path = _write_parosol_run_derivative_manifest(output_dir, **context)
                self._append_log(f"FEA derivative manifest: {manifest_path}\n")
            except Exception as exc:
                self._append_log(f"Could not write FEA derivative manifest: {exc}\n")
        else:
            self._append_log(
                "FEA derivative manifest not written; set derivative dataset, subject, and site to publish this run.\n"
            )
        if isinstance(data, dict):
            self._show_load_history_resultants(data)

    def _fea_derivative_context(self, output_dir):
        dataset_root = str(getattr(self.derivativeDatasetRootSelector, "currentPath", "") or "").strip()
        subject_id = str(self.derivativeSubjectEdit.text or "").strip()
        site = str(self.derivativeSiteEdit.text or "").strip()
        session_id = str(self.derivativeSessionEdit.text or "").strip() or Path(output_dir).name
        if dataset_root and subject_id and site:
            return {
                "dataset_root": dataset_root,
                "subject_id": subject_id,
                "site": site,
                "session_id": session_id,
            }
        inferred_root = _infer_fea_dataset_root(output_dir)
        if inferred_root != Path(output_dir).expanduser().resolve():
            return {"dataset_root": inferred_root}
        return None

    def _show_load_history_resultants(self, data):
        load_history = (
            data.get("postprocess", {}).get("load_history", {})
            if isinstance(data.get("postprocess"), dict)
            else {}
        )
        if not isinstance(load_history, dict) or load_history.get("status") != "computed":
            return
        estimated = (
            load_history.get("results", {}).get("estimated_loads", [])
            if isinstance(load_history.get("results"), dict)
            else []
        )
        force = _sum_load_vectors(estimated, load_type="force", units="N")
        moment = _sum_load_vectors(estimated, load_type="moment", units="N*mm")
        force_center = self._load_history_resultant_center(prefer_nodeset=True)
        moment_center = self._volume_bounds_center()
        force_length = max(6.0, self._bc_arrow_length_mm() * 1.4)
        moment_length = max(12.0, self._volume_max_extent_mm() * 1.15)
        for name in (
            "ParOSol_load_history_resultant_force",
            "ParOSol_load_history_resultant_force_center",
            "ParOSol_load_history_resultant_moment",
            "ParOSol_load_history_resultant_moment_center",
        ):
            self.logic.remove_named_node(name)
        force_node = self._create_load_history_resultant_arrow(
            "ParOSol_load_history_resultant_force",
            force_center,
            force,
            force_length,
            (0.1, 0.55, 1.0),
        )
        if force_node is not None:
            self.logic.create_point_markers(
                "ParOSol_load_history_resultant_force_center",
                [force_center],
                (0.1, 0.55, 1.0),
                glyph_scale=3.5,
            )
            self._append_log(
                "Displayed load-history resultant force "
                f"at RAS={_format_point_tuple(force_center)}: "
                f"{_format_resultant_vector(force, 'N')}.\n"
            )
        if moment is not None:
            moment_node = self._create_load_history_resultant_arrow(
                "ParOSol_load_history_resultant_moment",
                moment_center,
                moment,
                moment_length,
                (0.95, 0.45, 0.1),
                as_axis=True,
            )
            if moment_node is not None:
                self.logic.create_point_markers(
                    "ParOSol_load_history_resultant_moment_center",
                    [moment_center],
                    (0.95, 0.45, 0.1),
                    glyph_scale=3.5,
                )
                self._append_log(
                    "Displayed load-history resultant moment axis "
                    f"through RAS={_format_point_tuple(moment_center)}: "
                    f"{_format_resultant_vector(moment, 'N*mm')}.\n"
                )

    def _create_load_history_resultant_arrow(self, name, center, vector, length, color, *, as_axis=False):
        if not isinstance(vector, dict):
            return None
        values = [float(vector.get(axis, 0.0) or 0.0) for axis in ("x", "y", "z")]
        magnitude = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(magnitude) or magnitude <= 0.0:
            return None
        direction = tuple(value / magnitude for value in values)
        if not as_axis:
            start = _arrow_start_for_tip(center, direction, length)
            return self.logic.create_arrow_model(name, start, direction, length, color)
        start = tuple(
                float(center[index]) - 0.5 * float(length) * float(direction[index])
                for index in range(3)
            )
        return self.logic.create_vector_line(
            name,
            start,
            direction,
            length,
            color,
            thickness=1.2,
            glyph_scale=2.5,
        )

    def _load_history_resultant_center(self, *, prefer_nodeset):
        for row, row_data in enumerate(self.contactPlaneRows):
            try:
                spec = self._contact_row_spec(row)
            except Exception:
                continue
            if str(spec.get("bc_type", "")).startswith("Load history"):
                center = _plane_center(row_data.get("plane"))
                if center is not None:
                    return center
                if prefer_nodeset:
                    label_node = self.bcLabelSelector.currentNode() if hasattr(self, "bcLabelSelector") else None
                    center = self.logic.label_centroid_ras(
                        label_node,
                        self._bc_preview_label(spec, row),
                        self._volume(),
                    )
                    if center is not None:
                        return center
        volume = self._volume()
        if volume is not None:
            return self._volume_bounds_center()
        return (0.0, 0.0, 0.0)

    def _volume_bounds_center(self):
        volume = self._volume()
        if volume is None:
            return (0.0, 0.0, 0.0)
        bounds = [0.0] * 6
        volume.GetRASBounds(bounds)
        return (
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        )

    def _volume_max_extent_mm(self):
        volume = self._volume()
        if volume is None:
            return 10.0
        bounds = [0.0] * 6
        volume.GetRASBounds(bounds)
        return max(
            abs(bounds[1] - bounds[0]),
            abs(bounds[3] - bounds[2]),
            abs(bounds[5] - bounds[4]),
            1.0,
        )

    def _load_selected_result_fields(self, output_dir):
        output_dir = Path(output_dir)
        exported_field_names = _result_exported_field_names(output_dir)
        if exported_field_names:
            fields_to_load = _result_fields_from_filenames(exported_field_names)
        else:
            fields_to_load = list(self._selected_output_fields())
            for extra_field in ("load_history_estimated_sed", "load_history_final_sed"):
                if (output_dir / "fields" / f"{extra_field}.nii.gz").exists() and extra_field not in fields_to_load:
                    fields_to_load.append(extra_field)
        scalar_fields = _scalar_result_fields()
        fields_to_load = [field for field in fields_to_load if field in scalar_fields]
        if any(
            field in fields_to_load
            for field in ("load_history_estimated_sed", "load_history_final_sed")
        ):
            fields_to_load = [field for field in fields_to_load if field != "sed"]
        preferred_load_history_node = None
        preferred_nonlinear_node = None
        for stale_field in _known_result_fields():
            self.logic.remove_named_node(_result_field_node_name(stale_field))
        for field in fields_to_load:
            if field == "displacements":
                continue
            path = output_dir / "fields" / f"{field}.nii.gz"
            if not path.exists():
                self._append_log(f"Selected output field not found: {path}\n")
                continue
            reference_node = self._volume()
            path_to_load = _restore_cropped_field_to_reference_grid(path, reference_node)
            display_name = _result_field_display_name(field)
            node = _load_volume_node(
                str(path_to_load),
                {"name": _result_field_node_name(field)},
            )
            if node is None:
                continue
            _copy_geometry_if_compatible(node, reference_node)
            _apply_result_scalar_display(node)
            if field in {"sed", "load_history_estimated_sed", "load_history_final_sed"}:
                finite_positive = _positive_finite_volume_values(node)
                if finite_positive.size:
                    _activate_parosol_result_volume(node)
                else:
                    message = (
                        "SED field is all zero. The solve produced displacement/reaction "
                        "outputs, but native ParOSol did not produce strain energy density "
                        "for this run. Pistoia postprocessing is therefore unavailable "
                        "unless the model is rerun with a smaller active domain or an explicit "
                        "SED reconstruction step."
                    )
                    self._append_log(message + "\n")
                    try:
                        slicer.util.warningDisplay(message)
                    except Exception:
                        pass
                if field == "load_history_estimated_sed":
                    preferred_load_history_node = node
                elif (
                    field == "load_history_final_sed"
                    and preferred_load_history_node is None
                ):
                    preferred_load_history_node = node
            elif field == "mechanical_work_density":
                preferred_nonlinear_node = node
            if Path(path_to_load) != path:
                self._append_log(f"Loaded {display_name} field on reference grid: {path_to_load}\n")
            else:
                self._append_log(f"Loaded {display_name} field: {path}\n")
        if preferred_nonlinear_node is not None:
            _activate_parosol_result_volume(preferred_nonlinear_node)
        elif preferred_load_history_node is not None:
            _activate_parosol_result_volume(preferred_load_history_node)

    def export_results_csv(self):
        output_dir = Path(self.outputDirectory.directory)
        result_path = output_dir / "result.json"
        if not result_path.exists():
            slicer.util.errorDisplay(f"No result.json found in {output_dir}")
            return
        data = json.loads(result_path.read_text(encoding="utf-8"))
        row = self._result_csv_row(data)
        csv_path = output_dir / "result.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        self._append_log(f"Exported result CSV: {csv_path}\n")
        slicer.util.infoDisplay(f"Exported result CSV:\n{csv_path}")

    def save_result_as(self):
        output_dir = Path(self.outputDirectory.directory).expanduser().resolve()
        result_path = output_dir / "result.json"
        if not result_path.exists():
            slicer.util.errorDisplay(f"No result.json found in {output_dir}")
            return
        default_target = output_dir.parent / (output_dir.name or "parosol_result")
        selected = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save ParOSol Results As",
            str(default_target),
            "Folders (*)",
        )
        if isinstance(selected, (list, tuple)):
            selected = selected[0]
        if not selected:
            return
        target_dir = Path(str(selected)).expanduser()
        if target_dir.exists() and not target_dir.is_dir():
            slicer.util.errorDisplay(f"Selected path is not a folder: {target_dir}")
            return
        manifest = _result_save_manifest(output_dir, target_dir.name)
        if not manifest:
            slicer.util.errorDisplay(f"No result files found in {output_dir}")
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = _copy_result_save_manifest(manifest, target_dir)
        self._append_log(f"Saved renamed result bundle: {target_dir} ({len(copied)} file(s))\n")
        slicer.util.infoDisplay(f"Saved renamed result bundle:\n{target_dir}")

    def _format_result_html(self, data):
        if not isinstance(data, dict):
            return f"<html><body><pre>{html.escape(str(data))}</pre></body></html>"
        if "cases" in data and isinstance(data.get("cases"), list):
            return self._format_batch_result_html(data)
        primary = self._primary_result_values(data)
        case_name = primary.get("case") or Path(self.outputDirectory.directory).name
        mechanics = data.get("mechanics", {}) if isinstance(data.get("mechanics"), dict) else {}
        failure = self._selected_failure_result(data)
        solver = data.get("solver", {}) if isinstance(data.get("solver"), dict) else {}
        quality = data.get("quality", {}) if isinstance(data.get("quality"), dict) else {}
        load_case = data.get("load_case", {}) if isinstance(data.get("load_case"), dict) else {}
        image = data.get("image", {}) if isinstance(data.get("image"), dict) else {}
        primary_rows = [
            ("Failure load", primary.get("failure_load")),
            ("Stiffness", primary.get("stiffness")),
            ("Linear reaction force", _format_generalized(mechanics.get("generalized_load"))),
        ]
        mechanics_rows = [
            ("Load case", _format_load_case(load_case)),
            ("Applied displacement", primary.get("applied_displacement")),
            ("Applied strain", _format_number(mechanics.get("applied_strain")) if mechanics.get("applied_strain") is not None else None),
            ("Reference length", f"{_format_number(mechanics.get('reference_length_mm'))} mm" if mechanics.get("reference_length_mm") is not None else None),
            ("Top nodes", _format_number(mechanics.get("top_node_count"))),
            ("Bottom nodes", _format_number(mechanics.get("bottom_node_count"))),
            ("Applied rotation", _format_number(mechanics.get("applied_rotation_degrees")) + " deg" if mechanics.get("applied_rotation_degrees") is not None else None),
            ("Status", mechanics.get("status")),
        ]
        failure_rows = [
            ("Selected", failure.get("label")),
            ("Method", failure.get("method")),
            ("Criterion", failure.get("criterion")),
            ("Status", failure.get("status")),
            ("Critical strain", _format_number(failure.get("critical_strain"))),
            ("Critical volume", f"{_format_number(failure.get('critical_volume_percent'))} %" if failure.get("critical_volume_percent") is not None else None),
            ("EES at critical volume", _format_number(failure.get("ees_at_critical_volume"))),
            ("Pistoia factor", _format_number(failure.get("factor"))),
            ("Deformation", _format_number(failure.get("deformation"))),
            ("Coefficient", _format_number(failure.get("coefficient"))),
            ("Height", f"{_format_number(failure.get('height_mm'))} mm" if failure.get("height_mm") is not None else None),
            ("Failure vector", _format_xyz(failure.get("failure_load"), "N") if isinstance(failure.get("failure_load"), dict) else None),
        ]
        solver_rows = [
            ("Iterations", _format_number(solver.get("iterations"))),
            ("Relative residual", _format_number(solver.get("relative_residual"))),
            ("Absolute residual", _format_number(solver.get("absolute_residual"))),
            ("Runtime", f"{_format_number(solver.get('runtime_seconds'))} s" if solver.get("runtime_seconds") is not None else None),
            ("Quality", quality.get("status")),
            ("Issues", ", ".join(str(item) for item in quality.get("issues", [])) if quality.get("issues") else "none"),
        ]
        nonlinear_rows = _nonlinear_result_values(data)
        nonlinear_html = (
            "<h3>Nonlinear</h3>"
            f"<table>{_html_table_rows(nonlinear_rows)}</table>"
            if nonlinear_rows
            else ""
        )
        image_rows = [
            ("Dimensions xyz", _format_sequence(image.get("dimensions_xyz"))),
            ("Spacing", _format_sequence(image.get("spacing"), suffix=" mm")),
        ]
        paths = []
        for label, path in (
            ("result.json", Path(self.outputDirectory.directory) / "result.json"),
            ("summary.json", Path(self.outputDirectory.directory) / "summary.json"),
        ):
            if path.exists():
                paths.append(f"<li><b>{html.escape(label)}</b>: {html.escape(str(path))}</li>")
        for field in self._selected_output_fields():
            if field == "displacements":
                continue
            path = Path(self.outputDirectory.directory) / "fields" / f"{field}.nii.gz"
            if path.exists():
                paths.append(
                    f"<li><b>{html.escape(_result_field_display_name(field))}</b>: "
                    f"{html.escape(str(path))}</li>"
                )
        path_html = "".join(paths)
        return (
            "<html><body>"
            "<style>"
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;}"
            "h2{margin:0 0 8px 0;} h3{margin:14px 0 5px 0;color:#333;}"
            "table{border-collapse:collapse;width:100%;}"
            "th{text-align:left;vertical-align:top;padding:4px 14px 4px 0;color:#555;width:38%;}"
            "td{padding:4px 0;vertical-align:top;}"
            ".primary{font-size:15px;} .muted{color:#666;} ul{margin-top:6px;}"
            "</style>"
            f"<h2>{html.escape(str(case_name))}</h2>"
            "<h3>Primary Results</h3>"
            f"<table class='primary'>{_html_table_rows(primary_rows)}</table>"
            "<h3>Mechanics</h3>"
            f"<table>{_html_table_rows(mechanics_rows)}</table>"
            "<h3>Failure Criterion</h3>"
            f"<table>{_html_table_rows(failure_rows)}</table>"
            "<h3>Solver / Quality</h3>"
            f"<table>{_html_table_rows(solver_rows)}</table>"
            f"{nonlinear_html}"
            "<h3>Image</h3>"
            f"<table>{_html_table_rows(image_rows)}</table>"
            "<h3 class='muted'>Output files</h3>"
            f"<ul>{path_html}</ul>"
            "</body></html>"
        )

    def _format_batch_result_html(self, data):
        cases = data.get("cases", [])
        load_history = (
            data.get("postprocess", {}).get("load_history", {})
            if isinstance(data.get("postprocess"), dict)
            else {}
        )
        load_history_html = ""
        final_html = ""
        if isinstance(load_history, dict) and load_history.get("status") == "computed":
            final_case = (
                load_history.get("final_rerun", {}).get("case", {})
                if isinstance(load_history.get("final_rerun"), dict)
                else {}
            )
            if isinstance(final_case, dict):
                final_rows = [
                    ("Failure load", _format_generalized(final_case.get("failure_generalized_load"))),
                    ("Stiffness", _format_generalized(final_case.get("generalized_stiffness"))),
                    ("Generalized load", _format_generalized(final_case.get("generalized_load"))),
                    ("Status", load_history.get("final_rerun", {}).get("status") if isinstance(load_history.get("final_rerun"), dict) else None),
                ]
                final_html = (
                    "<h3>Final Combined Rerun</h3>"
                    f"<table class='primary'>{_html_table_rows(final_rows)}</table>"
                )
            load_history_html = _format_load_history_html(load_history)
        rows = []
        for case in cases:
            if not isinstance(case, dict):
                continue
            rows.append(
                (
                    case.get("name") or case.get("case") or "",
                    _format_generalized(case.get("failure_generalized_load")),
                    _format_generalized(case.get("generalized_stiffness")),
                    _format_generalized(case.get("generalized_load")),
                )
            )
        table_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{html.escape(str(failure))}</td>"
            f"<td>{html.escape(str(stiffness))}</td>"
            f"<td>{html.escape(str(load))}</td>"
            "</tr>"
            for name, failure, stiffness, load in rows
        )
        return (
            "<html><body>"
            "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;}"
            "h2{margin:0 0 8px 0;} h3{margin:14px 0 5px 0;color:#333;}"
            "table{border-collapse:collapse;width:100%;} th,td{text-align:left;padding:4px 10px 4px 0;vertical-align:top;}"
            "th{color:#555;} .primary{font-size:15px;} .muted{color:#666;}</style>"
            "<h2>Batch Results</h2>"
            f"{final_html}"
            f"{load_history_html}"
            "<h3>Unit Cases</h3>"
            "<table><tr><th>Case</th><th>Failure load</th><th>Stiffness</th><th>Reaction</th></tr>"
            f"{table_rows}</table>"
            "</body></html>"
        )

    def _primary_result_values(self, data):
        case = _nested_get(data, ("case", "name"))
        failure = self._selected_failure_result(data)
        mechanics = data.get("mechanics", {}) if isinstance(data.get("mechanics"), dict) else {}
        solver = data.get("solver", {}) if isinstance(data.get("solver"), dict) else {}
        failure_value = _nested_get(failure, ("failure_generalized_load", "value"))
        failure_units = _nested_get(failure, ("failure_generalized_load", "units"), "N")
        failure_label = failure.get("label") or _nested_get(failure, ("failure_generalized_load", "name"), "load")
        stiffness_value = _nested_get(mechanics, ("generalized_stiffness", "value"))
        stiffness_units = _nested_get(mechanics, ("generalized_stiffness", "units"), "N/mm")
        stiffness_xyz = mechanics.get("stiffness")
        reaction = mechanics.get("reaction_force")
        applied = mechanics.get("applied_displacement")
        solver_text = None
        if solver:
            solver_text = (
                f"{_format_number(solver.get('iterations'))} iterations, "
                f"relative residual {_format_number(solver.get('relative_residual'))}, "
                f"{_format_number(solver.get('runtime_seconds'))} s"
            )
        return {
            "case": case,
            "failure_load": (
                f"{_format_number(failure_value)} {failure_units} ({failure_label})"
                if failure_value is not None
                else None
            ),
            "stiffness": (
                f"{_format_number(stiffness_value)} {stiffness_units}"
                if stiffness_value is not None
                else (_format_xyz(stiffness_xyz, "N/mm") if isinstance(stiffness_xyz, dict) else "not computed")
            ),
            "reaction_force": _format_xyz(reaction, "N") if isinstance(reaction, dict) else None,
            "applied_displacement": _format_xyz(applied, "mm") if isinstance(applied, dict) else None,
            "solver": solver_text,
        }

    def _selected_failure_result(self, data):
        failure = data.get("failure", {}) if isinstance(data.get("failure"), dict) else {}
        preset = self._widget_text(getattr(self, "failurePresetBox", None), "Pistoia EES 0.7% / 2%")
        if preset == "Kopperdahl/Crawford 0.68%":
            selected = failure.get("crawford_stiffness_height", {})
            if isinstance(selected, dict):
                selected = dict(selected)
                selected.setdefault("label", "Kopperdahl/Crawford 0.68%")
                return selected
        selected = dict(failure)
        if preset == "None":
            selected.setdefault("label", "None")
        else:
            selected.setdefault("label", "Pistoia EES 0.7% / 2%")
        return selected

    def _result_csv_row(self, data):
        failure = self._selected_failure_result(data)
        mechanics = data.get("mechanics", {}) if isinstance(data.get("mechanics"), dict) else {}
        solver = data.get("solver", {}) if isinstance(data.get("solver"), dict) else {}
        reaction = mechanics.get("reaction_force", {}) if isinstance(mechanics.get("reaction_force"), dict) else {}
        stiffness = mechanics.get("stiffness", {}) if isinstance(mechanics.get("stiffness"), dict) else {}
        failure_load = failure.get("failure_load", {}) if isinstance(failure.get("failure_load"), dict) else {}
        return {
            "case": _nested_get(data, ("case", "name"), ""),
            "profile": _nested_get(data, ("execution", "profile"), ""),
            "failure_load_generalized": _nested_get(failure, ("failure_generalized_load", "value"), ""),
            "failure_load_units": _nested_get(failure, ("failure_generalized_load", "units"), ""),
            "failure_load_x": failure_load.get("x", ""),
            "failure_load_y": failure_load.get("y", ""),
            "failure_load_z": failure_load.get("z", ""),
            "stiffness_generalized": _nested_get(mechanics, ("generalized_stiffness", "value"), ""),
            "stiffness_units": _nested_get(mechanics, ("generalized_stiffness", "units"), ""),
            "stiffness_x": stiffness.get("x", ""),
            "stiffness_y": stiffness.get("y", ""),
            "stiffness_z": stiffness.get("z", ""),
            "reaction_force_x": reaction.get("x", ""),
            "reaction_force_y": reaction.get("y", ""),
            "reaction_force_z": reaction.get("z", ""),
            "solver_iterations": solver.get("iterations", ""),
            "solver_relative_residual": solver.get("relative_residual", ""),
            "solver_runtime_seconds": solver.get("runtime_seconds", ""),
            "result_json": str(Path(self.outputDirectory.directory) / "result.json"),
        }

    def _displacement_component_paths(self, output_dir):
        output_dir = Path(output_dir)
        paths = {
            axis: output_dir / "fields" / f"displacement_{axis}.nii.gz"
            for axis in ("x", "y", "z")
        }
        return paths if all(path.exists() for path in paths.values()) else None

    def show_deformed_result(self):
        try:
            if not self._export_displacements_enabled():
                raise ValueError(
                    "Displacement fields were not requested for this run. Enable "
                    "'Displacements' in `Outputs` and rerun before showing deformation arrows."
                )
            output_dir = Path(self.outputDirectory.directory)
            self.logic.export_displacement_components_from_run(
                output_dir,
                on_output=self._append_log,
            )
            component_paths = self._displacement_component_paths(output_dir)
            if not component_paths:
                raise ValueError("Could not create displacement_x/y/z fields from this run.")
            reference_node, temporary_reference, use_material_mask = self._deformation_reference_node(
                output_dir,
                component_paths,
            )
            node, stats = self.logic.create_deformation_arrow_glyphs(
                reference_node,
                component_paths,
                scale=float(self.deformedScaleSpin.value),
                max_points=int(self.deformedMaxArrowsSpin.value),
                use_material_mask=use_material_mask,
                color_field_path=Path(output_dir) / "fields" / "sed.nii.gz",
                color_field_name="SED",
            )
            if temporary_reference is not None:
                self.logic.group_node(temporary_reference, "Results")
            if node is None:
                raise ValueError("Could not create deformation arrow preview")
            self._append_log(
                "Created deformation arrow preview: "
                f"{stats['count']} arrows, displacement magnitude "
                f"{stats['min_mm']:.4g}/{stats['median_mm']:.4g}/{stats['max_mm']:.4g} mm "
                f"(min/median/max), max displayed arrow {stats['max_display_mm']:.4g} mm, "
                f"colored by {stats['color_mode']} "
                f"({stats['color_min']:.4g}/{stats['color_median']:.4g}/{stats['color_max']:.4g}), "
                f"bins {stats['bin_counts']}.\n"
            )
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

    def _deformation_reference_node(self, output_dir, component_paths):
        displacement_probe = _load_volume_node(
            str(component_paths["x"]),
            {"name": "ParOSol_deformation_geometry_probe", "show": False},
        )
        try:
            displacement_shape = _node_array_shape(displacement_probe)
            current = self._volume()
            if _volume_grid_matches(current, displacement_probe):
                return current, None, True
            candidates = [
                (Path(output_dir) / "slicer_input.nii.gz", "saved solver input geometry", True),
                (Path(output_dir) / "fields" / "sed.nii.gz", "SED field geometry", False),
                (Path(component_paths["x"]), "displacement field geometry", False),
            ]
            for path, description, use_material_mask in candidates:
                if not path.exists():
                    continue
                node = _load_volume_node(
                    str(path),
                    {"name": "ParOSol_deformation_reference", "show": False},
                )
                if _volume_grid_matches(node, displacement_probe):
                    self._append_log(
                        f"Using {description} for deformation arrows.\n"
                    )
                    return node, node, use_material_mask
                self.logic.remove_node(node)
            raise ValueError(
                "Displacement fields do not match the current image and no matching "
                f"reference field was found. displacement_x shape {displacement_shape}; "
                f"current image shape {_node_array_shape(current)}."
            )
        finally:
            self.logic.remove_node(displacement_probe)

    def delete_deformed_result(self):
        signal_states = self._begin_input_node_update_suppression()
        try:
            self.logic.remove_named_node("ParOSol_deformed_model")
            self.logic.remove_named_node("ParOSol_deformation_arrows")
            for bin_index in range(5):
                self.logic.remove_named_node(f"ParOSol_deformation_arrows_bin_{bin_index + 1}")
            self.logic.remove_named_node("ParOSol_deformation_reference")
            self._append_log("Deleted deformation preview.\n")
        finally:
            self._end_input_node_update_suppression(signal_states)

    def delete_results(self):
        names = [
            "ParOSol_SED",
            "ParOSol_deformed_model",
            "ParOSol_deformation_arrows",
            "ParOSol_deformation_reference",
        ]
        names.extend(f"ParOSol_deformation_arrows_bin_{index + 1}" for index in range(5))
        removed = 0
        signal_states = self._begin_input_node_update_suppression()
        try:
            slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
            try:
                _clear_parosol_viewer_references()
                for name in names:
                    removed += int(self.logic.remove_named_node(name))
            finally:
                slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)
            _remove_empty_parosol_subject_hierarchy_folders()
        finally:
            self._end_input_node_update_suppression(signal_states)
        self.resultText.clear()
        self._append_log(f"Deleted {removed} ParOSol result scene node{'s' if removed != 1 else ''}.\n")

    def _append_log(self, text):
        self.logText.appendPlainText(str(text).rstrip("\n"))

    def _show_in_standard_slice_views(
        self,
        background_node,
        *,
        label_node=None,
        label_opacity=None,
        reset_orientations=True,
    ):
        layers = {"background": background_node, "label": label_node}
        if label_opacity is not None:
            layers["labelOpacity"] = float(label_opacity)
        try:
            slicer.util.setSliceViewerLayers(**layers)
        except Exception:
            pass
        if reset_orientations:
            _set_standard_slice_orientations()


def _set_standard_slice_orientations():
    orientations = {
        "Red": "Axial",
        "Yellow": "Sagittal",
        "Green": "Coronal",
    }
    try:
        nodes = slicer.util.getNodesByClass("vtkMRMLSliceNode")
    except Exception:
        nodes = []
    for node in nodes or []:
        name = str(node.GetName() or "")
        orientation = orientations.get(name)
        if not orientation:
            continue
        try:
            method = getattr(node, f"SetOrientationTo{orientation}", None)
            if method is not None:
                method()
            else:
                node.SetOrientation(orientation)
            node.Modified()
        except Exception:
            pass
    try:
        layout_manager = slicer.app.layoutManager()
    except Exception:
        layout_manager = None
    if layout_manager is None:
        return
    for view_name in orientations:
        try:
            widget = layout_manager.sliceWidget(view_name)
            logic = widget.sliceLogic() if widget is not None else None
            if logic is not None and hasattr(logic, "FitSliceToAll"):
                logic.FitSliceToAll()
        except Exception:
            pass


def _normalized(vector):
    length = math.sqrt(sum(float(v) * float(v) for v in vector))
    if length == 0:
        return [0.0, 0.0, 1.0]
    return [float(v) / length for v in vector]


def _default_plane_u_axis(normal):
    normal = _normalized(normal)
    x_axis = [1.0, 0.0, 0.0]
    if abs(_dot(normal, x_axis)) < 0.95:
        return x_axis
    return [0.0, 1.0, 0.0]


def _extent_along_axis(extent, axis):
    return sum(abs(float(axis[index])) * float(extent[index]) for index in range(3))


def _mask_bounds_ras(mask, reference_node):
    indices = np.argwhere(np.asarray(mask, dtype=bool))
    if indices.size == 0 or reference_node is None:
        return None
    ijk_to_ras = vtk.vtkMatrix4x4()
    reference_node.GetIJKToRASMatrix(ijk_to_ras)
    mins_zyx = indices.min(axis=0)
    maxs_zyx = indices.max(axis=0)
    corners = []
    for k in (int(mins_zyx[0]), int(maxs_zyx[0])):
        for j in (int(mins_zyx[1]), int(maxs_zyx[1])):
            for i in (int(mins_zyx[2]), int(maxs_zyx[2])):
                corners.append(ijk_to_ras.MultiplyPoint([i, j, k, 1.0])[:3])
    points = np.asarray(corners, dtype=float)
    return points.min(axis=0).tolist(), points.max(axis=0).tolist()


def _plane_center(plane):
    if plane is None:
        return None
    center = [0.0, 0.0, 0.0]
    try:
        plane.GetCenterWorld(center)
    except Exception:
        try:
            plane.GetCenter(center)
        except Exception:
            return None
    return tuple(float(value) for value in center)


def _looks_like_plane_node(plane):
    return plane is not None and (
        hasattr(plane, "GetCenter")
        or hasattr(plane, "GetCenterWorld")
    ) and (
        hasattr(plane, "GetNormal")
        or hasattr(plane, "GetNormalWorld")
    )


def _plane_normal_world(plane):
    normal = [0.0, 0.0, -1.0]
    if plane is not None:
        try:
            plane.GetNormalWorld(normal)
        except Exception:
            try:
                plane.GetNormal(normal)
            except Exception:
                pass
    return tuple(_normalized(normal))


def _snap_vector_to_angular_step(vector, step_degrees):
    unit = _normalized(vector)
    step = max(float(step_degrees), 0.1)
    polar = math.degrees(math.atan2(math.sqrt(unit[0] ** 2 + unit[1] ** 2), unit[2]))
    azimuth = math.degrees(math.atan2(unit[1], unit[0]))
    polar = round(polar / step) * step
    azimuth = round(azimuth / step) * step
    polar_radians = math.radians(polar)
    azimuth_radians = math.radians(azimuth)
    return _normalized(
        (
            math.sin(polar_radians) * math.cos(azimuth_radians),
            math.sin(polar_radians) * math.sin(azimuth_radians),
            math.cos(polar_radians),
        )
    )


def _set_display_color(display, color):
    red, green, blue = (float(color[0]), float(color[1]), float(color[2]))
    try:
        display.SetColor(red, green, blue)
    except TypeError:
        display.SetColor((red, green, blue))
    if hasattr(display, "SetSelectedColor"):
        try:
            display.SetSelectedColor(red, green, blue)
        except TypeError:
            display.SetSelectedColor((red, green, blue))


def _is_generated_parosol_node(node):
    name = str(node.GetName() or "")
    lower = name.lower()
    if name == "ParOSolFEA":
        return False
    if lower.startswith("parosol_"):
        return True
    if "_parosol_" in lower:
        return True
    if lower.startswith("slicerparosol"):
        return True
    return False


def _is_removable_generated_parosol_node(node):
    if node is None or not _is_generated_parosol_node(node):
        return False
    if _is_mrml_infrastructure_node(node):
        return False
    return True


def _is_mrml_infrastructure_node(node):
    return bool(
        node is not None
        and (
            node.IsA("vtkMRMLDisplayNode")
            or node.IsA("vtkMRMLStorageNode")
            or node.IsA("vtkMRMLColorNode")
            or node.IsA("vtkMRMLSubjectHierarchyNode")
        )
    )


def _prepare_node_for_removal(node):
    _detach_node_from_viewers(node)
    _detach_node_from_subject_hierarchy(node)
    _remove_volume_rendering_display_nodes(node)
    try:
        if hasattr(node, "GetNumberOfDisplayNodes"):
            for index in range(int(node.GetNumberOfDisplayNodes())):
                display = node.GetNthDisplayNode(index)
                _hide_display_node(display)
        elif hasattr(node, "GetDisplayNode"):
            _hide_display_node(node.GetDisplayNode())
    except Exception:
        pass
    try:
        if hasattr(node, "SetAndObserveStorageNodeID"):
            node.SetAndObserveStorageNodeID(None)
    except Exception:
        pass


def _clear_parosol_viewer_references():
    try:
        slicer.util.setSliceViewerLayers(background=None, foreground=None, label=None)
    except Exception:
        pass
    try:
        selection = slicer.app.applicationLogic().GetSelectionNode()
        for method_name in (
            "SetReferenceActiveVolumeID",
            "SetReferenceSecondaryVolumeID",
            "SetReferenceActiveLabelVolumeID",
        ):
            if hasattr(selection, method_name):
                getattr(selection, method_name)(None)
    except Exception:
        pass


def _detach_node_from_viewers(node):
    if node is None or not hasattr(node, "GetID"):
        return
    node_id = node.GetID()
    if not node_id:
        return
    try:
        for index in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSliceCompositeNode")):
            composite = slicer.mrmlScene.GetNthNodeByClass(index, "vtkMRMLSliceCompositeNode")
            for getter, setter in (
                ("GetBackgroundVolumeID", "SetBackgroundVolumeID"),
                ("GetForegroundVolumeID", "SetForegroundVolumeID"),
                ("GetLabelVolumeID", "SetLabelVolumeID"),
            ):
                if hasattr(composite, getter) and hasattr(composite, setter):
                    try:
                        if getattr(composite, getter)() == node_id:
                            getattr(composite, setter)(None)
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        selection = slicer.app.applicationLogic().GetSelectionNode()
        for getter, setter in (
            ("GetActiveVolumeID", "SetReferenceActiveVolumeID"),
            ("GetSecondaryVolumeID", "SetReferenceSecondaryVolumeID"),
            ("GetActiveLabelVolumeID", "SetReferenceActiveLabelVolumeID"),
        ):
            if hasattr(selection, getter) and hasattr(selection, setter):
                try:
                    if getattr(selection, getter)() == node_id:
                        getattr(selection, setter)(None)
                except Exception:
                    pass
    except Exception:
        pass


def _detach_node_from_subject_hierarchy(node):
    if node is None or not hasattr(node, "GetID"):
        return
    try:
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
            slicer.mrmlScene
        )
        if sh_node is None:
            return
        item_id = sh_node.GetItemByDataNode(node)
        if item_id:
            sh_node.SetItemParent(item_id, sh_node.GetSceneItemID())
    except Exception:
        pass


def _remove_empty_parosol_subject_hierarchy_folders():
    try:
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
            slicer.mrmlScene
        )
        if sh_node is None:
            return
        scene_item_id = sh_node.GetSceneItemID()
        child_ids = vtk.vtkIdList()
        sh_node.GetItemChildren(scene_item_id, child_ids, False)
        root_ids = [
            child_ids.GetId(index)
            for index in range(child_ids.GetNumberOfIds())
            if sh_node.GetItemName(child_ids.GetId(index)) == "SlicerParOSol"
        ]
        for root_id in root_ids:
            _remove_empty_subject_hierarchy_folder_tree(sh_node, root_id)
    except Exception:
        pass


def _remove_empty_subject_hierarchy_folder_tree(sh_node, item_id):
    child_ids = vtk.vtkIdList()
    try:
        sh_node.GetItemChildren(item_id, child_ids, False)
    except Exception:
        return False
    for index in range(child_ids.GetNumberOfIds()):
        _remove_empty_subject_hierarchy_folder_tree(sh_node, child_ids.GetId(index))

    child_ids = vtk.vtkIdList()
    try:
        sh_node.GetItemChildren(item_id, child_ids, False)
    except Exception:
        return False
    if child_ids.GetNumberOfIds() > 0:
        return False
    try:
        data_node = sh_node.GetItemDataNode(item_id)
    except Exception:
        data_node = None
    if data_node is not None:
        return False
    try:
        sh_node.RemoveItem(item_id)
        return True
    except Exception:
        return False


def _remove_volume_rendering_display_nodes(node):
    if node is None or not hasattr(node, "GetID"):
        return
    node_id = node.GetID()
    if not node_id:
        return
    display_nodes = []
    try:
        for index in range(slicer.mrmlScene.GetNumberOfNodes()):
            candidate = slicer.mrmlScene.GetNthNode(index)
            if candidate is None or not candidate.IsA("vtkMRMLVolumeRenderingDisplayNode"):
                continue
            referenced_id = None
            for method_name in ("GetVolumeNodeID", "GetAndObserveVolumeNodeID"):
                if hasattr(candidate, method_name):
                    try:
                        referenced_id = getattr(candidate, method_name)()
                    except Exception:
                        referenced_id = None
                    if referenced_id:
                        break
            if referenced_id == node_id:
                display_nodes.append(candidate)
    except Exception:
        return
    for display in display_nodes:
        try:
            _hide_display_node(display)
            if slicer.mrmlScene.IsNodePresent(display):
                slicer.mrmlScene.RemoveNode(display)
        except Exception:
            pass


def _hide_display_node(display):
    if display is None:
        return
    for method_name in ("SetAllSegmentsVisibility", "SetAllSegmentsVisibility3D"):
        if hasattr(display, method_name):
            try:
                getattr(display, method_name)(False)
            except Exception:
                pass
    try:
        node = display.GetDisplayableNode() if hasattr(display, "GetDisplayableNode") else None
        segmentation = node.GetSegmentation() if _is_segmentation_node(node) else None
        if segmentation is not None:
            for index in range(int(segmentation.GetNumberOfSegments())):
                segment_id = segmentation.GetNthSegmentID(index)
                for method_name in (
                    "SetSegmentVisibility",
                    "SetSegmentVisibility3D",
                    "SetSegmentVisibility2DFill",
                    "SetSegmentVisibility2DOutline",
                ):
                    if hasattr(display, method_name):
                        try:
                            getattr(display, method_name)(segment_id, False)
                        except Exception:
                            pass
    except Exception:
        pass
    if hasattr(display, "SetVisibility2D"):
        try:
            display.SetVisibility2D(False)
        except Exception:
            pass
    elif hasattr(display, "SetSliceIntersectionVisibility"):
        try:
            display.SetSliceIntersectionVisibility(False)
        except Exception:
            pass
    for method_name, args in (
        ("SetVisibility", (False,)),
        ("SetVisibility3D", (False,)),
    ):
        if hasattr(display, method_name):
            try:
                getattr(display, method_name)(*args)
            except Exception:
                pass


def _generated_node_removal_priority(node):
    if node is None:
        return 99
    if node.IsA("vtkMRMLDisplayNode") or node.IsA("vtkMRMLStorageNode"):
        return 80
    if node.IsA("vtkMRMLColorNode"):
        return 90
    if node.IsA("vtkMRMLSubjectHierarchyNode"):
        return 95
    return 10


def _load_volume_node(path, properties=None):
    properties = properties or {}
    try:
        loaded = slicer.util.loadVolume(str(path), properties)
    except TypeError:
        loaded = slicer.util.loadVolume(str(path), properties, returnNode=True)
    if isinstance(loaded, tuple):
        return loaded[1] if len(loaded) > 1 else None
    if isinstance(loaded, bool):
        name = properties.get("name")
        return slicer.mrmlScene.GetFirstNodeByName(str(name)) if loaded and name else None
    return loaded


def _load_label_volume_node(path, properties=None):
    properties = properties or {}
    try:
        loaded = slicer.util.loadLabelVolume(str(path), properties)
    except TypeError:
        loaded = slicer.util.loadLabelVolume(str(path), properties, returnNode=True)
    if isinstance(loaded, tuple):
        return loaded[1] if len(loaded) > 1 else None
    if isinstance(loaded, bool):
        name = properties.get("name")
        return slicer.mrmlScene.GetFirstNodeByName(str(name)) if loaded and name else None
    return loaded


def _filter_labelmap_to_values(label_node, allowed_values):
    if label_node is None or not allowed_values:
        return False
    try:
        allowed = {int(value) for value in allowed_values if int(value) > 0}
    except Exception:
        return False
    if not allowed:
        return False
    try:
        array = np.asarray(slicer.util.arrayFromVolume(label_node))
    except Exception:
        return False
    keep = np.isin(array, tuple(sorted(allowed)))
    stray = (array != 0) & ~keep
    if not np.any(stray):
        return False
    filtered = np.where(keep, array, 0).astype(array.dtype, copy=False)
    slicer.util.updateVolumeFromArray(label_node, filtered)
    label_node.Modified()
    return True


def _volume_file_shape(path):
    node = _load_volume_node(
        str(path),
        {"name": "ParOSol_shape_probe", "show": False},
    )
    try:
        return _node_array_shape(node)
    finally:
        if node is not None:
            try:
                slicer.mrmlScene.RemoveNode(node)
            except Exception:
                pass


def _copy_geometry_if_compatible(node, reference_node):
    if node is None or reference_node is None:
        return False
    node_image = node.GetImageData()
    ref_image = reference_node.GetImageData()
    if node_image is None or ref_image is None:
        return False
    if tuple(node_image.GetDimensions()) != tuple(ref_image.GetDimensions()):
        return False
    try:
        node.CopyOrientation(reference_node)
    except Exception:
        matrix = vtk.vtkMatrix4x4()
        reference_node.GetIJKToRASMatrix(matrix)
        node.SetIJKToRASMatrix(matrix)
    node.Modified()
    return True


def _reference_grid_image(reference_node):
    if reference_node is None:
        return None
    path_text = ""
    try:
        fd, path_text = tempfile.mkstemp(suffix=".nii.gz", prefix="parosol_reference_grid_")
        os.close(fd)
        path = Path(path_text)
        if not slicer.util.saveNode(reference_node, str(path)):
            return None
        return sitk.ReadImage(str(path))
    except Exception:
        return None
    finally:
        try:
            if path_text:
                Path(path_text).unlink(missing_ok=True)
        except Exception:
            pass


def _restore_cropped_field_to_reference_grid(field_path, reference_node):
    """Return a full-grid copy when a ParOSol field was exported as a tight crop."""
    if reference_node is None:
        return Path(field_path)
    try:
        reference_array = np.asarray(slicer.util.arrayFromVolume(reference_node))
        field_image = sitk.ReadImage(str(field_path))
        field_array = sitk.GetArrayFromImage(field_image)
    except Exception:
        return Path(field_path)
    if tuple(field_array.shape) == tuple(reference_array.shape):
        return Path(field_path)
    active = np.argwhere(reference_array != 0)
    if active.size == 0:
        return Path(field_path)
    lower = active.min(axis=0)
    upper = active.max(axis=0) + 1
    bbox_shape = tuple(int(v) for v in (upper - lower))
    if tuple(field_array.shape) != bbox_shape:
        return Path(field_path)

    restored = np.zeros(reference_array.shape, dtype=field_array.dtype)
    z0, y0, x0 = (int(v) for v in lower)
    z1, y1, x1 = (int(v) for v in upper)
    restored[z0:z1, y0:y1, x0:x1] = field_array
    restored_image = sitk.GetImageFromArray(restored)
    reference_image = _reference_grid_image(reference_node)
    if reference_image is not None and tuple(reference_image.GetSize()) == tuple(reversed(reference_array.shape)):
        restored_image.CopyInformation(reference_image)
    else:
        restored_image.SetSpacing(field_image.GetSpacing())
        restored_image.SetOrigin(field_image.GetOrigin())
        restored_image.SetDirection(field_image.GetDirection())

    out_dir = Path(field_path).parent / "reference_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(field_path).name
    sitk.WriteImage(restored_image, str(out_path))
    return out_path


def _decode_process_output(raw):
    try:
        if hasattr(raw, "data"):
            raw = raw.data()
        if isinstance(raw, str):
            return raw
        return bytes(raw).decode("utf-8", errors="replace")
    except Exception:
        return str(raw)


def _unique_paths(paths):
    seen = set()
    unique = []
    for path in paths:
        if not path:
            continue
        normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _filter_runtime_noise(text):
    if not text:
        return ""
    lines = str(text).splitlines()
    kept = []
    skip_until_blank = False
    for line in lines:
        stripped = line.strip()
        if skip_until_blank:
            if not stripped:
                skip_until_blank = False
            continue
        if "Possible incompatible factory load" in line:
            skip_until_blank = True
            continue
        if "itkObjectFactoryBase.cxx" in line:
            skip_until_blank = True
            continue
        if "Error ImageIO factory did not return an ImageIOBase: MRMLIDImageIO" in line:
            continue
        kept.append(line)
    if not kept:
        return ""
    suffix = "\n" if str(text).endswith("\n") else ""
    return "\n".join(kept) + suffix


def _runtime_report_value(report, key):
    prefix = str(key).strip()
    for line in str(report or "").splitlines():
        text = line.strip()
        if text.startswith(prefix + " "):
            value = text[len(prefix) :].strip()
            return value or None
        if text.startswith(prefix + ":"):
            value = text[len(prefix) + 1 :].strip()
            return value or None
    return None


def _runtime_success_details(report):
    rows = [f"SlicerParOSol build: {SLICER_PAROSOL_BUILD}"]
    for key, label in (
        ("parosol_py", "ParOSol-py version"),
        ("source", "ParOSol-py source"),
        ("native", "Native solver"),
        ("packaged MPI launcher", "Packaged MPI launcher"),
    ):
        value = _runtime_report_value(report, key)
        if value:
            rows.append(f"{label}: {value}")
    return "\n".join(rows)


def _subject_hierarchy_folder(sh_node, parent_id, name):
    child_ids = vtk.vtkIdList()
    try:
        sh_node.GetItemChildren(parent_id, child_ids, False)
        for index in range(child_ids.GetNumberOfIds()):
            child_id = child_ids.GetId(index)
            if sh_node.GetItemName(child_id) == name:
                return child_id
    except Exception:
        pass
    return sh_node.CreateFolderItem(parent_id, name)


def _parosol_folder_for_node_name(name):
    text = str(name).lower()
    if "arrow" in text or "marker" in text or "boundary_conditions_3d" in text:
        return "Loads"
    if "disk" in text or "cap" in text or "nodeset" in text or "contact" in text:
        return "Contact Regions"
    if "plane" in text:
        return "Planes"
    if "mask" in text:
        return "Inputs"
    return "Generated"


def _safe_identifier(text):
    import re

    value = re.sub(r"[^0-9A-Za-z_]+", "_", str(text).strip())
    value = re.sub(r"_+", "_", value).strip("_").lower()
    if not value:
        value = "nodeset"
    if value[0].isdigit():
        value = f"nodeset_{value}"
    return value


def _generated_boundary_label_for_row(bc_type, row):
    token = str(bc_type or "Dirichlet").strip()
    row_offset = int(row) + 1
    if token == "None":
        return GENERATED_INACTIVE_LABEL_BASE + row_offset - 1
    if token == "Fixed":
        return GENERATED_FIXED_LABEL_BASE + row_offset - 1
    if token == "Neumann":
        return GENERATED_NEUMANN_LABEL_BASE + row_offset - 1
    return GENERATED_DIRICHLET_LABEL_BASE + row_offset - 1


def _workflow_nodeset_label_map(config):
    nodesets = config.get("nodesets", {}) if isinstance(config, dict) else {}
    if not isinstance(nodesets, dict):
        return {}
    labels = {}
    for name, spec in nodesets.items():
        if not isinstance(spec, dict) or spec.get("label") is None:
            continue
        try:
            labels[_safe_identifier(name)] = int(spec["label"])
        except Exception:
            continue
    return labels


def _workflow_replay_nodeset_specs(nodesets):
    if not isinstance(nodesets, dict):
        return {}
    cleaned = {}
    for name, spec in nodesets.items():
        if not isinstance(spec, dict):
            continue
        cleaned_spec = copy.deepcopy(spec)
        cleaned_spec.pop("image", None)
        cleaned[str(name)] = cleaned_spec
    return cleaned


def _workflow_replay_editor_for_export_space(editor, replay_cfg):
    if not isinstance(editor, dict):
        return editor
    resolved = copy.deepcopy(editor)
    model_space = str((replay_cfg or {}).get("model_space", "reference")).strip().lower()
    if model_space != "reference":
        return resolved
    planes = editor.get("planes", [])
    if not isinstance(planes, list):
        return resolved
    reference_relative_spaces = {"model_bbox", "active_bbox", "image_bbox"}
    for plane in resolved.get("planes", []):
        if not isinstance(plane, dict):
            continue
        relative_to = str(plane.get("relative_to", "") or "").strip().lower()
        if relative_to.startswith("resolved_"):
            continue
        if bool(plane.get("reference_space", False)):
            continue
        if relative_to in reference_relative_spaces:
            continue
        if plane.get("center_fraction") is not None or plane.get("bbox_fraction_bounds") is not None:
            continue
        if plane.get("center_ras") is not None and plane.get("normal_ras") is not None:
            plane["reference_scaled"] = True
            plane["reference_space"] = False
    return resolved


def _merge_workflow_replay_material_override(config, material_override):
    if not isinstance(material_override, dict):
        return
    input_cfg = config.setdefault("input", {})
    if "image_type" in material_override:
        input_cfg["image_type"] = material_override["image_type"]
    materials = material_override.get("materials")
    if isinstance(materials, dict):
        base = config.get("materials", {})
        config["materials"] = _deep_merge_workflow_config(base, materials)
    solver = material_override.get("solver")
    if isinstance(solver, dict):
        config["solver"] = _deep_merge_workflow_config(config.get("solver", {}), solver)


def _deep_merge_workflow_config(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    target = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_merge_workflow_config(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _uniform_spatial_sample(points, *, max_points):
    if len(points) <= int(max_points):
        return list(points)
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        return list(points[: int(max_points)])
    selected = _grid_sample_indices(coordinates, max_points=int(max_points))
    if len(selected) < int(max_points):
        selected = _fill_with_farthest_indices(
            coordinates,
            selected,
            max_points=int(max_points),
        )
    return [tuple(float(value) for value in coordinates[index]) for index in selected]


def _label_array_sample_points_ras(array, label, reference_node, *, max_points):
    if array is None or reference_node is None:
        return []
    indices_zyx = np.argwhere(np.asarray(array) == int(label))
    if indices_zyx.size == 0:
        return []
    indices_zyx = _grid_sample_indices_zyx(indices_zyx, max_points=max(1, int(max_points)))
    ijk_to_ras = vtk.vtkMatrix4x4()
    reference_node.GetIJKToRASMatrix(ijk_to_ras)
    points = []
    for index in indices_zyx:
        ijk = (float(index[2]), float(index[1]), float(index[0]))
        points.append(tuple(float(value) for value in ijk_to_ras.MultiplyPoint([*ijk, 1.0])[:3]))
    return points


def _intersect_plane_nodeset_array(
    active,
    volume_node,
    plane,
    label,
    *,
    shape="anatomy",
    radius_mm=12.0,
    use_plane_size=True,
):
    active = np.asarray(active, dtype=bool)
    nodesets = np.zeros_like(active, dtype=np.uint16)
    if plane is None or volume_node is None or not np.any(active):
        return nodesets
    center, normal, u_axis, v_axis, half_u, half_v = _plane_geometry(
        plane,
        shape=shape,
        radius_mm=radius_mm,
        square_width_mm=float(radius_mm) * 2.0,
        hex_radius_mm=radius_mm,
        use_plane_size=use_plane_size,
    )
    indices_zyx = np.argwhere(active)
    if indices_zyx.size == 0:
        return nodesets
    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)
    origin = np.asarray(
        [ijk_to_ras.GetElement(axis, 3) for axis in range(3)],
        dtype=np.float64,
    )
    i_axis = np.asarray([ijk_to_ras.GetElement(axis, 0) for axis in range(3)], dtype=np.float64)
    j_axis = np.asarray([ijk_to_ras.GetElement(axis, 1) for axis in range(3)], dtype=np.float64)
    k_axis = np.asarray([ijk_to_ras.GetElement(axis, 2) for axis in range(3)], dtype=np.float64)
    ras = (
        origin
        + indices_zyx[:, 2:3] * i_axis
        + indices_zyx[:, 1:2] * j_axis
        + indices_zyx[:, 0:1] * k_axis
    )
    rel = ras - np.asarray(center, dtype=np.float64)
    distance = rel @ np.asarray(normal, dtype=np.float64)
    u = rel @ np.asarray(u_axis, dtype=np.float64)
    v = rel @ np.asarray(v_axis, dtype=np.float64)
    tolerance = _voxel_tolerance(volume_node)
    footprint_tolerance = max(float(tolerance), 1.0e-9)
    shape_name = str(shape).strip().lower()
    if shape_name in {"anatomy", "rectangle", "rectangular"}:
        inside = (np.abs(u) <= float(half_u) + footprint_tolerance) & (
            np.abs(v) <= float(half_v) + footprint_tolerance
        )
    elif shape_name == "square":
        half = min(float(half_u), float(half_v))
        inside = (np.abs(u) <= half + footprint_tolerance) & (
            np.abs(v) <= half + footprint_tolerance
        )
    elif shape_name == "hex":
        half = max(min(float(half_u), float(half_v)), 1.0e-9)
        uu = u / half
        vv = v / half
        hex_tolerance = footprint_tolerance / half
        inside = (
            (np.abs(uu) <= 1.0 + hex_tolerance)
            & (np.abs(0.5 * uu + 0.8660254 * vv) <= 1.0 + hex_tolerance)
            & (np.abs(0.5 * uu - 0.8660254 * vv) <= 1.0 + hex_tolerance)
        )
    elif shape_name in {"oval", "round"}:
        half_u_oval = max(float(half_u) + footprint_tolerance, 1.0e-9)
        half_v_oval = max(float(half_v) + footprint_tolerance, 1.0e-9)
        inside = ((u / half_u_oval) ** 2 + (v / half_v_oval) ** 2) <= 1.0
    else:
        inside = (np.abs(u) <= float(half_u) + footprint_tolerance) & (
            np.abs(v) <= float(half_v) + footprint_tolerance
        )
    selected = inside & (np.abs(distance) <= tolerance)
    if not np.any(selected) and (np.any(inside) or shape_name == "anatomy"):
        abs_distance = np.abs(distance)
        candidates = inside if np.any(inside) else np.ones(distance.shape, dtype=bool)
        nearest_distance = float(np.min(abs_distance[candidates]))
        snap_tolerance = 2.0 * float(tolerance)
        if nearest_distance <= snap_tolerance:
            slab_tolerance = max(0.25 * float(tolerance), 1.0e-9)
            selected = candidates & (abs_distance <= nearest_distance + slab_tolerance)
    if np.any(selected):
        selected_zyx = indices_zyx[selected]
        nodesets[selected_zyx[:, 0], selected_zyx[:, 1], selected_zyx[:, 2]] = int(label)
    return nodesets


def _points_centroid(points):
    if not points:
        return None
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        return None
    return tuple(float(value) for value in np.mean(values, axis=0))


def _grid_sample_indices(coordinates, *, max_points):
    mins = coordinates.min(axis=0)
    spans = np.maximum(coordinates.max(axis=0) - mins, 1e-6)
    normalized = (coordinates - mins) / spans
    active_axes = np.where(spans > 1e-6)[0]
    if active_axes.size == 0:
        return [0]
    grid_side = max(2, int(math.ceil(math.sqrt(max_points))))
    levels = np.linspace(0.0, 1.0, grid_side)
    selected = []
    used = set()

    center = np.full(3, 0.5, dtype=np.float64)
    center_index = int(np.argmin(np.sum((normalized - center) ** 2, axis=1)))
    selected.append(center_index)
    used.add(center_index)

    for anchored_axis in active_axes:
        for level_a in levels:
            for level_b in levels:
                target = center.copy()
                other_axes = [axis for axis in active_axes if axis != anchored_axis]
                if len(other_axes) >= 1:
                    target[other_axes[0]] = level_a
                if len(other_axes) >= 2:
                    target[other_axes[1]] = level_b
                distances = np.sum((normalized - target) ** 2, axis=1)
                for index in np.argsort(distances)[:8]:
                    index = int(index)
                    if index not in used:
                        selected.append(index)
                        used.add(index)
                        break
                if len(selected) >= max_points:
                    return selected[:max_points]
    return selected[:max_points]


def _fill_with_farthest_indices(coordinates, selected, *, max_points):
    mins = coordinates.min(axis=0)
    spans = np.maximum(coordinates.max(axis=0) - mins, 1e-6)
    normalized = (coordinates - mins) / spans
    if not selected:
        selected = [int(np.argmin(np.sum((normalized - 0.5) ** 2, axis=1)))]
    min_distances = np.min(
        np.stack(
            [np.sum((normalized - normalized[index]) ** 2, axis=1) for index in selected],
            axis=0,
        ),
        axis=0,
    )
    used = set(int(index) for index in selected)
    while len(selected) < int(max_points):
        next_index = int(np.argmax(min_distances))
        if next_index in used:
            break
        selected.append(next_index)
        used.add(next_index)
        distances = np.sum((normalized - normalized[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
    return selected[: int(max_points)]


def _grid_sample_indices_zyx(indices_zyx, *, max_points):
    indices = np.asarray(indices_zyx, dtype=np.float64)
    if indices.shape[0] <= int(max_points):
        return np.asarray(indices_zyx, dtype=np.int64)
    mins = indices.min(axis=0)
    maxs = indices.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    active_axes = np.where(spans > 1.0e-6)[0]
    if active_axes.size == 0:
        return np.asarray(indices_zyx[:1], dtype=np.int64)

    bins_per_axis = np.ones(3, dtype=np.int64)
    base_bins = max(
        2,
        int(math.ceil(int(max_points) ** (1.0 / max(len(active_axes), 1)))),
    )
    for axis in active_axes:
        relative = spans[axis] / max(float(np.max(spans[active_axes])), 1.0)
        bins_per_axis[axis] = max(2, int(round(base_bins * max(relative, 0.5))))

    normalized = (indices - mins) / spans
    raw_cells = np.floor(normalized * bins_per_axis).astype(np.int64)
    cells = np.minimum(raw_cells, bins_per_axis - 1)
    cell_candidates: dict[tuple[int, int, int], list[int]] = {}
    for point_index, cell in enumerate(cells):
        cell_candidates.setdefault(tuple(int(value) for value in cell), []).append(point_index)

    representatives = []
    for cell, point_indices in cell_candidates.items():
        cell_center = mins + ((np.asarray(cell, dtype=np.float64) + 0.5) / bins_per_axis) * spans
        local = indices[point_indices]
        distances = np.sum(((local - cell_center) / spans) ** 2, axis=1)
        best = int(point_indices[int(np.argmin(distances))])
        representatives.append(best)

    if len(representatives) <= int(max_points):
        ordered = sorted(
            representatives,
            key=lambda idx: tuple(float(value) for value in cells[idx]),
        )
        return np.asarray(indices[ordered], dtype=np.int64)

    rep_points = indices[representatives]
    selected_rep_positions = _evenly_spaced_indices(rep_points, max_points=int(max_points))
    selected = [representatives[position] for position in selected_rep_positions]
    return np.asarray(indices[selected], dtype=np.int64)


def _evenly_spaced_indices(points, *, max_points):
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.shape[0] <= int(max_points):
        return list(range(coordinates.shape[0]))
    mins = coordinates.min(axis=0)
    spans = np.maximum(coordinates.max(axis=0) - mins, 1.0e-6)
    normalized = (coordinates - mins) / spans
    selected = [int(np.argmin(np.sum((normalized - 0.5) ** 2, axis=1)))]
    min_distances = np.sum((normalized - normalized[selected[0]]) ** 2, axis=1)
    used = set(selected)
    while len(selected) < int(max_points):
        next_index = int(np.argmax(min_distances))
        if next_index in used:
            break
        selected.append(next_index)
        used.add(next_index)
        distances = np.sum((normalized - normalized[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
    return selected[: int(max_points)]


def _arrow_polydata(start, direction, length_mm):
    source = vtk.vtkArrowSource()
    source.SetTipLength(0.25)
    source.SetTipRadius(0.08)
    source.SetShaftRadius(0.025)
    x_axis = _normalized(direction)
    helper = (0.0, 0.0, 1.0) if abs(x_axis[2]) < 0.9 else (0.0, 1.0, 0.0)
    z_axis = _normalized(_cross(x_axis, helper))
    y_axis = _normalized(_cross(z_axis, x_axis))

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for row in range(3):
        matrix.SetElement(row, 0, x_axis[row] * float(length_mm))
        matrix.SetElement(row, 1, y_axis[row] * float(length_mm))
        matrix.SetElement(row, 2, z_axis[row] * float(length_mm))
        matrix.SetElement(row, 3, float(start[row]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(source.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(transform_filter.GetOutput())
    return polydata


def _arrow_start_for_tip(tip, direction, length_mm):
    unit = _normalized(direction)
    return tuple(
        float(tip[index]) - float(unit[index]) * float(length_mm)
        for index in range(3)
    )


def _ijk_axis_directions_ras(ijk_to_ras):
    directions = []
    for axis in range(3):
        column = tuple(float(ijk_to_ras.GetElement(row, axis)) for row in range(3))
        directions.append(_normalized(column))
    return directions


def _color_table(kind):
    name = f"ParOSol_{kind}_colors"
    existing = slicer.mrmlScene.GetFirstNodeByName(name)
    if existing is not None:
        if kind == "nodesets":
            if existing.GetNumberOfColors() < 512:
                existing.SetNumberOfColors(512)
            existing.SetColor(1, "fixed_dirichlet_xyz", 0.1, 0.35, 1.0, 0.9)
            existing.SetColor(2, "prescribed_displacement", 1.0, 0.55, 0.05, 0.9)
            existing.SetColor(3, "force_neumann", 1.0, 0.05, 0.08, 0.9)
        elif kind == "mask":
            if existing.GetNumberOfColors() < 512:
                existing.SetNumberOfColors(512)
            _seed_mask_color_table(existing)
        return existing
    color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", name)
    color_node.SetTypeToUser()
    if kind == "disks":
        color_node.SetNumberOfColors(203)
        color_node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
        color_node.SetColor(201, "top_cap", 0.72, 0.72, 0.72, 0.75)
        color_node.SetColor(202, "bottom_cap", 0.48, 0.48, 0.48, 0.75)
    elif kind == "nodesets":
        color_node.SetNumberOfColors(512)
        color_node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
        color_node.SetColor(1, "fixed_dirichlet_xyz", 0.1, 0.35, 1.0, 0.9)
        color_node.SetColor(2, "prescribed_displacement", 1.0, 0.55, 0.05, 0.9)
        color_node.SetColor(3, "force_neumann", 1.0, 0.05, 0.08, 0.9)
    else:
        color_node.SetNumberOfColors(512)
        _seed_mask_color_table(color_node)
    try:
        color_node.HideFromEditorsOn()
    except Exception:
        pass
    return color_node


def _seed_mask_color_table(color_node):
    color_node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
    for label, name, color, opacity in _known_mask_label_styles():
        if label >= color_node.GetNumberOfColors():
            color_node.SetNumberOfColors(label + 1)
        color_node.SetColor(label, name, color[0], color[1], color[2], opacity)


def _extend_label_color_table(color_node, labels, *, kind):
    labels = [int(label) for label in labels if int(label) >= 0]
    if not labels:
        return
    max_label = max(labels)
    if max_label >= color_node.GetNumberOfColors():
        color_node.SetNumberOfColors(max_label + 1)
    for label in labels:
        if label == 0:
            color_node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
            continue
        if kind == "nodesets":
            color, opacity = _nodeset_label_color(label)
            color_node.SetColor(label, f"nodeset_{label}", color[0], color[1], color[2], opacity)
        elif kind == "mask":
            name, color, opacity = _mask_label_style(label)
            color_node.SetColor(label, name, color[0], color[1], color[2], opacity)
        else:
            hue = ((label * 37) % 100) / 100.0
            red = 0.25 + 0.55 * hue
            green = 0.70 - 0.35 * hue
            blue = 0.35 + 0.45 * (1.0 - hue)
            color_node.SetColor(label, f"label_{label}", red, green, blue, 0.65)


def _known_mask_label_styles():
    return (
        (1, "bone_or_body", (0.42, 0.72, 0.95), 0.45),
        (2, "process_or_femur", (0.95, 0.62, 0.20), 0.45),
        (20, "vertebral_body", (0.42, 0.72, 0.95), 0.45),
        (48, "vertebral_process", (0.95, 0.62, 0.20), 0.45),
        (100, "trabecular_bone", (0.78, 0.86, 0.48), 0.45),
        (127, "cortical_bone", (0.88, 0.72, 0.42), 0.45),
    )


def _mask_label_style(label):
    value = int(label or 0)
    for known_label, name, color, opacity in _known_mask_label_styles():
        if value == known_label:
            return name, color, opacity
    hue = ((value * 37) % 100) / 100.0
    red = 0.25 + 0.55 * hue
    green = 0.70 - 0.35 * hue
    blue = 0.35 + 0.45 * (1.0 - hue)
    return f"label_{value}", (red, green, blue), 0.45


def _result_jet_color_table():
    name = "ParOSol_Result_JET"
    existing = slicer.mrmlScene.GetFirstNodeByName(name)
    if existing is not None:
        return existing
    color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", name)
    color_node.SetTypeToUser()
    color_node.SetNumberOfColors(256)
    color_node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
    for index in range(1, 256):
        value = index / 255.0
        red = min(max(1.5 - abs(4.0 * value - 3.0), 0.0), 1.0)
        green = min(max(1.5 - abs(4.0 * value - 2.0), 0.0), 1.0)
        blue = min(max(1.5 - abs(4.0 * value - 1.0), 0.0), 1.0)
        color_node.SetColor(index, f"jet_{index}", red, green, blue, 1.0)
    color_node.HideFromEditorsOn()
    return color_node


def _sed_jet_color_table():
    return _result_jet_color_table()


def _apply_result_scalar_display(volume_node):
    volume_node.CreateDefaultDisplayNodes()
    display = volume_node.GetDisplayNode()
    if display is None:
        return
    color_node = _result_jet_color_table()
    if color_node is not None:
        display.SetAndObserveColorNodeID(color_node.GetID())
    finite_positive = _positive_finite_volume_values(volume_node)
    if finite_positive.size:
        lower, upper, maximum = _sed_display_range(finite_positive)
        if np.isfinite(upper) and upper > lower:
            display.AutoWindowLevelOff()
            display.SetWindowLevelMinMax(lower, upper)
        display.ApplyThresholdOn()
        display.SetLowerThreshold(lower)
        display.SetUpperThreshold(maximum)
    display.SetInterpolate(0)


def _apply_sed_display(volume_node):
    _apply_result_scalar_display(volume_node)
    finite_positive = _positive_finite_volume_values(volume_node)
    _apply_sed_volume_rendering(volume_node, finite_positive)


def _apply_material_preview_display(volume_node):
    volume_node.CreateDefaultDisplayNodes()
    display = volume_node.GetDisplayNode()
    if display is None:
        return
    color_node = _sed_jet_color_table()
    if color_node is not None:
        display.SetAndObserveColorNodeID(color_node.GetID())
    finite_positive = _positive_finite_volume_values(volume_node)
    if finite_positive.size:
        lower, upper, maximum = _sed_display_range(finite_positive)
        if np.isfinite(upper) and upper > lower:
            display.AutoWindowLevelOff()
            display.SetWindowLevelMinMax(lower, upper)
        display.ApplyThresholdOn()
        display.SetLowerThreshold(lower)
        display.SetUpperThreshold(maximum)
    display.SetInterpolate(0)


def _positive_finite_volume_values(volume_node):
    try:
        array = np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=float)
        return array[np.isfinite(array) & (array > 0.0)]
    except Exception:
        return np.asarray([], dtype=float)


def _sed_display_range(finite_positive):
    lower = float(max(np.percentile(finite_positive, 0.1), np.finfo(float).eps))
    upper = float(np.percentile(finite_positive, 99.5))
    if not np.isfinite(upper) or upper <= lower:
        upper = float(np.max(finite_positive))
    maximum = float(np.max(finite_positive))
    return lower, upper, maximum


def _apply_sed_volume_rendering(volume_node, finite_positive):
    if not finite_positive.size:
        return
    try:
        volume_rendering_module = getattr(slicer.modules, "volumerendering", None)
        if volume_rendering_module is None:
            return
        logic = volume_rendering_module.logic()
        display_node = logic.CreateDefaultVolumeRenderingNodes(volume_node)
        if display_node is None:
            return
        display_node.SetVisibility(True)
        if hasattr(display_node, "SetFollowVolumeDisplayNode"):
            display_node.SetFollowVolumeDisplayNode(False)
        property_node = display_node.GetVolumePropertyNode()
        if property_node is None:
            return
        volume_property = property_node.GetVolumeProperty()
        lower, upper, _maximum = _sed_display_range(finite_positive)
        p50 = float(np.percentile(finite_positive, 50.0))
        p90 = float(np.percentile(finite_positive, 90.0))
        p99 = float(np.percentile(finite_positive, 99.0))

        color = vtk.vtkColorTransferFunction()
        for value, red, green, blue in _jet_transfer_points(lower, upper):
            color.AddRGBPoint(float(value), red, green, blue)

        opacity = vtk.vtkPiecewiseFunction()
        opacity.AddPoint(0.0, 0.0)
        opacity.AddPoint(lower, 0.0)
        opacity.AddPoint(max(lower, p50), 0.01)
        opacity.AddPoint(max(lower, p90), 0.08)
        opacity.AddPoint(max(lower, p99), 0.22)
        opacity.AddPoint(max(upper, p99), 0.35)

        volume_property.SetColor(color)
        volume_property.SetScalarOpacity(opacity)
        volume_property.ShadeOff()
        volume_property.SetInterpolationTypeToLinear()
        property_node.Modified()
        display_node.Modified()
    except Exception:
        return


def _jet_transfer_points(lower, upper):
    if upper <= lower:
        upper = lower + 1.0
    values = []
    for index in range(6):
        t = index / 5.0
        value = float(lower) + t * (float(upper) - float(lower))
        red = min(max(1.5 - abs(4.0 * t - 3.0), 0.0), 1.0)
        green = min(max(1.5 - abs(4.0 * t - 2.0), 0.0), 1.0)
        blue = min(max(1.5 - abs(4.0 * t - 1.0), 0.0), 1.0)
        values.append((value, red, green, blue))
    return values


def _segmentations_logic():
    try:
        return slicer.modules.segmentations.logic()
    except Exception:
        return slicer.util.getModuleLogic("Segmentations")


def _style_segmentation_3d(segmentation_node, *, kind):
    if segmentation_node is None:
        return
    segmentation_node.CreateDefaultDisplayNodes()
    _ensure_closed_surface_representation(segmentation_node)
    display = segmentation_node.GetDisplayNode()
    if display is None:
        return
    try:
        display.SetVisibility(True)
        display.SetVisibility3D(True)
        display.SetAllSegmentsVisibility(True)
        display.SetAllSegmentsVisibility3D(True)
        display.SetAllSegmentsOpacity3D(0.45 if kind == "mask" else 0.8)
        display.SetAllSegmentsOpacity2DFill(0.45 if kind == "mask" else 0.7)
        display.SetAllSegmentsOpacity2DOutline(1.0)
    except Exception:
        pass

    segmentation = segmentation_node.GetSegmentation()
    for index in range(segmentation.GetNumberOfSegments()):
        segment_id = segmentation.GetNthSegmentID(index)
        segment = segmentation.GetSegment(segment_id)
        label = _segment_label_value(segment)
        if kind == "nodesets":
            color, opacity = _nodeset_label_color(label)
        elif kind == "disks":
            color, opacity = (0.62, 0.62, 0.62), 0.7
        else:
            _name, color, opacity = _mask_label_style(label)
        try:
            segment.SetColor(color)
        except Exception:
            pass
        try:
            display.SetSegmentOverrideColor(segment_id, color)
            display.SetSegmentOpacity3D(segment_id, opacity)
            display.SetSegmentOpacity2DFill(segment_id, opacity)
            display.SetSegmentVisibility3D(segment_id, True)
            display.SetSegmentVisibility2DFill(segment_id, True)
            display.SetSegmentVisibility2DOutline(segment_id, True)
        except Exception:
            pass


def _ensure_closed_surface_representation(segmentation_node):
    if segmentation_node is None:
        return
    try:
        segmentation = segmentation_node.GetSegmentation()
        if segmentation is not None and segmentation.ContainsRepresentation("Closed surface"):
            return
    except Exception:
        pass
    try:
        segmentation_node.CreateClosedSurfaceRepresentation()
    except Exception:
        pass


def _activate_parosol_result_volume(node):
    if node is None:
        return
    try:
        node_id = node.GetID()
    except Exception:
        return
    if not node_id:
        return

    try:
        selection = slicer.app.applicationLogic().GetSelectionNode()
        for setter in ("SetReferenceActiveVolumeID", "SetActiveVolumeID"):
            if hasattr(selection, setter):
                try:
                    getattr(selection, setter)(node_id)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        for index in range(
            slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSliceCompositeNode")
        ):
            composite = slicer.mrmlScene.GetNthNodeByClass(
                index,
                "vtkMRMLSliceCompositeNode",
            )
            if hasattr(composite, "SetBackgroundVolumeID"):
                composite.SetBackgroundVolumeID(node_id)
    except Exception:
        pass

    try:
        if not node.GetNumberOfDisplayNodes():
            return
        for index in range(int(node.GetNumberOfDisplayNodes())):
            display = node.GetNthDisplayNode(index)
            _show_display_node(display)
    except Exception:
        pass

def _segment_label_value(segment):
    if segment is None:
        return None
    name = segment.GetName()
    if not name:
        return None
    lower_name = str(name).strip().lower()
    if "fixed" in lower_name:
        return 1
    if "displacement" in lower_name or "dirichlet" in lower_name:
        return 2
    if "force" in lower_name or "neumann" in lower_name:
        return 3
    if "body" in lower_name or "centrum" in lower_name:
        return 20
    if "process" in lower_name or "posterior" in lower_name or "arch" in lower_name:
        return 48
    match = None
    try:
        import re

        match = re.search(r"(\d+)$", lower_name)
    except Exception:
        match = None
    if match:
        return int(match.group(1))
    return None


def _nodeset_label_color(label):
    value = int(label or 0)
    if value == 1 or 100 <= value < 200:
        return (0.1, 0.35, 1.0), 0.9
    if value == 3 or 300 <= value < 400:
        return (1.0, 0.05, 0.08), 0.9
    if 400 <= value < 500:
        return (0.55, 0.35, 0.85), 0.7
    return (1.0, 0.55, 0.05), 0.9


def _density_to_e_mpa(density, e_config):
    density = np.asarray(density, dtype=np.float64)
    equation = str(e_config.get("equation", "linear")).strip().lower()
    if equation == "power":
        coefficient = float(e_config.get("coefficient", 1.0))
        exponent = float(e_config.get("exponent", 1.0))
        reference = float(e_config.get("reference_density", e_config.get("reference", 1.0)))
        if reference == 0:
            raise ValueError("Density reference must be non-zero")
        return coefficient * np.power(np.maximum(density, 0.0) / reference, exponent)
    if equation in {"mulder", "mulder2007", "mulder_2007", "framework_mulder", "framework_mulder2007"}:
        slope = float(e_config.get("slope", e_config.get("a", 25.0)))
        intercept = float(e_config.get("intercept", e_config.get("b", -5830.0)))
        return slope * density + intercept
    if equation == "polynomial":
        coefficients = e_config.get("coefficients", [0.0, 1.0])
        youngs = np.zeros(density.shape, dtype=np.float64)
        for power, coefficient in enumerate(coefficients):
            youngs += float(coefficient) * np.power(density, power)
        return np.maximum(youngs, 0.0)
    slope = float(e_config.get("slope", 1.0))
    intercept = float(e_config.get("intercept", 0.0))
    return np.maximum(slope * density + intercept, 0.0)


def _ogo_binned_density_values(density, *, active, number_bins):
    density = np.asarray(density, dtype=np.float64)
    active_nonzero = np.asarray(active, dtype=bool) & (density != 0)
    n_bins = int(number_bins)
    if n_bins <= 0:
        raise ValueError("number bins must be positive")
    values = density[active_nonzero]
    if values.size == 0:
        raise ValueError(
            "Cannot bin material because no active non-zero density voxels were found."
        )
    bin_edges = np.linspace(float(values.min()), float(values.max()), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_ids = np.digitize(density, bin_edges, right=False)
    bin_ids = np.clip(bin_ids, 1, n_bins)
    binned = np.zeros(density.shape, dtype=np.float64)
    binned[active_nonzero] = bin_centers[bin_ids[active_nonzero].astype(np.int64) - 1]
    return binned, bin_edges, bin_centers


def _density_floor_config_value(*configs):
    for config in configs:
        if not isinstance(config, dict):
            continue
        for key in ("minimum_e_mpa", "floor_e_mpa", "floor_mpa", "floor"):
            value = config.get(key)
            if value is not None:
                return float(value)
    return None


def _plane_axes(normal):
    helper = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.8 else [0.0, 1.0, 0.0]
    u = _cross(normal, helper)
    u = _normalized(u)
    v = _cross(normal, u)
    return u, _normalized(v)


def _plane_geometry(
    plane,
    *,
    shape,
    radius_mm,
    square_width_mm,
    hex_radius_mm,
    use_plane_size,
):
    corner_geometry = _plane_corner_geometry(
        plane,
        shape=shape,
        radius_mm=radius_mm,
        square_width_mm=square_width_mm,
        hex_radius_mm=hex_radius_mm,
        use_plane_size=use_plane_size,
    )
    if corner_geometry and use_plane_size:
        return corner_geometry

    center = _plane_center(plane) or [0.0, 0.0, 0.0]
    normal = _plane_normal_world(plane)
    u_axis, v_axis = _plane_axes_from_plane(plane, normal)
    half_u, half_v = _plane_half_size(
        plane,
        shape=shape,
        radius_mm=radius_mm,
        square_width_mm=square_width_mm,
        hex_radius_mm=hex_radius_mm,
        use_plane_size=use_plane_size,
    )
    return center, normal, u_axis, v_axis, half_u, half_v


def _workflow_plane_spec_from_slicer_row(row_spec, plane):
    shape = str(row_spec.get("shape", "anatomy")).strip() or "anatomy"
    radius = float(row_spec.get("radius", row_spec.get("radius_mm", 12.0)))
    square_width = float(
        row_spec.get("square_width", row_spec.get("square_width_mm", radius * 2.0))
    )
    hex_radius = float(row_spec.get("hex_radius", row_spec.get("hex_radius_mm", radius)))
    center, normal, u_axis, v_axis, half_u, half_v = _plane_geometry(
        plane,
        shape=shape,
        radius_mm=radius,
        square_width_mm=square_width,
        hex_radius_mm=hex_radius,
        use_plane_size=bool(row_spec.get("use_plane_size", True)),
    )
    contact = str(row_spec.get("contact", "Material disks")).strip()
    if contact == "PMMA caps":
        contact = "Material disks"
    return {
        "name": str(row_spec.get("name") or "Plane"),
        "axis": str(row_spec.get("axis", "z")).strip().lower(),
        "normal": str(row_spec.get("normal", "-")).strip(),
        "contact": contact,
        "surface_mode": _projection_mode(row_spec.get("surface_mode", "project_bounded")),
        "bc_mode": str(row_spec.get("bc_type", "None")).strip(),
        "direction": str(row_spec.get("direction", "Plane normal")).strip(),
        "shape": shape,
        "anatomy_constrained": bool(row_spec.get("anatomy_constrained", False)),
        "thickness_mm": float(row_spec.get("thickness", row_spec.get("thickness_mm", 3.0))),
        "intrusion_depth_mm": float(
            row_spec.get("intrusion", row_spec.get("intrusion_depth_mm", 2.0))
        ),
        "use_plane_size": True,
        "center_ras": [float(value) for value in center],
        "normal_ras": [float(value) for value in normal],
        "u_axis_ras": [float(value) for value in u_axis],
        "v_axis_ras": [float(value) for value in v_axis],
        "size_mm": [float(half_u) * 2.0, float(half_v) * 2.0],
    }


def _project_vector_onto_plane(vector, normal):
    normal = _normalized(normal)
    return [
        float(vector[index]) - _dot(vector, normal) * float(normal[index])
        for index in range(3)
    ]


def _plane_corner_geometry(
    plane,
    *,
    shape,
    radius_mm,
    square_width_mm,
    hex_radius_mm,
    use_plane_size,
):
    if not hasattr(plane, "GetPlaneCornerPointsWorld"):
        return None
    points = vtk.vtkPoints()
    try:
        plane.GetPlaneCornerPointsWorld(points)
    except Exception:
        return None
    if points.GetNumberOfPoints() < 4:
        return None

    corners = [points.GetPoint(index) for index in range(4)]
    plane_normal = _plane_normal_world(plane)
    edge_u = _subtract(corners[3], corners[0])
    edge_v = _subtract(corners[1], corners[0])
    edge_u = _project_vector_onto_plane(edge_u, plane_normal)
    edge_v = _project_vector_onto_plane(edge_v, plane_normal)
    half_u = _vector_length(edge_u) / 2.0
    half_v = _vector_length(edge_v) / 2.0

    fallback_u, fallback_v = _plane_axes_from_plane(plane, plane_normal)
    if half_u > 1.0e-6:
        u_axis = _normalized(edge_u)
    else:
        u_axis = fallback_u
    if half_v > 1.0e-6:
        v_candidate = _project_vector_onto_plane(edge_v, plane_normal)
        v_candidate = _project_vector_onto_plane(v_candidate, u_axis)
        v_axis = (
            _normalized(v_candidate)
            if _vector_length(v_candidate) > 1.0e-6
            else _normalized(_cross(plane_normal, u_axis))
        )
    else:
        v_axis = fallback_v
    v_axis = _project_vector_onto_plane(v_axis, plane_normal)
    v_axis = _project_vector_onto_plane(v_axis, u_axis)
    v_axis = (
        _normalized(v_axis)
        if _vector_length(v_axis) > 1.0e-6
        else _normalized(_cross(plane_normal, u_axis))
    )
    u_axis = _normalized(_cross(v_axis, plane_normal))

    u_coordinates = [_dot(corner, u_axis) for corner in corners]
    v_coordinates = [_dot(corner, v_axis) for corner in corners]
    n_coordinates = [_dot(corner, plane_normal) for corner in corners]
    min_u, max_u = min(u_coordinates), max(u_coordinates)
    min_v, max_v = min(v_coordinates), max(v_coordinates)
    center = [
        float(u_axis[index]) * 0.5 * (min_u + max_u)
        + float(v_axis[index]) * 0.5 * (min_v + max_v)
        + float(plane_normal[index]) * (sum(n_coordinates) / float(len(n_coordinates)))
        for index in range(3)
    ]
    half_u = 0.5 * (max_u - min_u)
    half_v = 0.5 * (max_v - min_v)

    fallback_half_u, fallback_half_v = _plane_half_size(
        plane,
        shape=shape,
        radius_mm=radius_mm,
        square_width_mm=square_width_mm,
        hex_radius_mm=hex_radius_mm,
        use_plane_size=use_plane_size,
    )
    if half_u <= 1.0e-6:
        half_u = fallback_half_u
    if half_v <= 1.0e-6:
        half_v = fallback_half_v
    if half_u <= 0 or half_v <= 0:
        return None
    return center, plane_normal, u_axis, v_axis, half_u, half_v


def _plane_axes_from_plane(plane, normal):
    if hasattr(plane, "GetAxes"):
        u = [0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0]
        w = [0.0, 0.0, 0.0]
        try:
            plane.GetAxes(u, v, w)
            if _vector_length(u) > 0 and _vector_length(v) > 0:
                return _normalized(u), _normalized(v)
        except Exception:
            pass
    return _plane_axes(normal)


def _set_plane_axes_world(plane, u_axis, v_axis, normal):
    u_axis, v_axis, normal = _right_handed_plane_axes_for_slicer(
        u_axis, v_axis, normal
    )
    if hasattr(plane, "SetAxesWorld"):
        try:
            plane.SetAxesWorld(u_axis, v_axis, normal)
            return
        except Exception:
            pass
    if hasattr(plane, "SetAxes"):
        try:
            plane.SetAxes(u_axis, v_axis, normal)
            return
        except Exception:
            pass
    if hasattr(plane, "SetNormalWorld"):
        try:
            plane.SetNormalWorld(normal)
            return
        except Exception:
            pass
    plane.SetNormal(normal)


def _right_handed_plane_axes_for_slicer(u_axis, v_axis, normal):
    """Return axes accepted by vtkMRMLMarkupsPlaneNode.

    Workflow recipes may store an in-plane basis whose handedness is convenient
    for geometry calculations. Slicer markups require X cross Y to point along
    Z, and reject the update otherwise. Preserve the requested normal and first
    in-plane axis, then derive the second in-plane axis from them.
    """
    normal = _normalized(normal)
    u_axis = _project_vector_onto_plane(u_axis, normal)
    if _vector_length(u_axis) <= 1.0e-6:
        u_axis = _project_vector_onto_plane(_cross(v_axis, normal), normal)
    if _vector_length(u_axis) <= 1.0e-6:
        u_axis, _v_axis = _plane_axes(normal)
    u_axis = _normalized(u_axis)
    v_axis = _normalized(_cross(normal, u_axis))
    return u_axis, v_axis, normal


def _plane_half_size(
    plane,
    *,
    shape,
    radius_mm,
    square_width_mm,
    hex_radius_mm,
    use_plane_size,
):
    if use_plane_size and hasattr(plane, "GetSizeWorld"):
        size = [0.0, 0.0]
        try:
            plane.GetSizeWorld(size)
            half_u = abs(float(size[0])) / 2.0
            half_v = abs(float(size[1])) / 2.0
            if half_u > 0 and half_v > 0:
                return half_u, half_v
        except Exception:
            pass
    if use_plane_size and hasattr(plane, "GetSize"):
        size = [0.0, 0.0]
        try:
            plane.GetSize(size)
            half_u = abs(float(size[0])) / 2.0
            half_v = abs(float(size[1])) / 2.0
            if half_u > 0 and half_v > 0:
                return half_u, half_v
        except Exception:
            pass
    shape = str(shape).strip().lower()
    if shape == "square":
        half = float(square_width_mm) / 2.0
        return half, half
    if shape == "hex":
        half = float(hex_radius_mm)
        return half, half
    radius = float(radius_mm)
    return radius, radius


def _vector_length(vector):
    return math.sqrt(sum(float(v) * float(v) for v in vector))


def _subtract(a, b):
    return [float(a[index]) - float(b[index]) for index in range(3)]


def _workflow_editor_pose_equivalent(current_editor, stored_editor, *, tolerance=1.0e-4):
    if not isinstance(current_editor, dict) or not isinstance(stored_editor, dict):
        return False
    current_planes = current_editor.get("planes", [])
    stored_planes = stored_editor.get("planes", [])
    if not isinstance(current_planes, list) or not isinstance(stored_planes, list):
        return False
    if len(current_planes) != len(stored_planes):
        return False
    stored_by_name = {
        str(plane.get("name", "")): plane
        for plane in stored_planes
        if isinstance(plane, dict)
    }
    for current_plane in current_planes:
        if not isinstance(current_plane, dict):
            return False
        name = str(current_plane.get("name", ""))
        stored_plane = stored_by_name.get(name)
        if not isinstance(stored_plane, dict):
            return False
        for key in ("center_ras", "normal_ras", "u_axis_ras", "v_axis_ras", "size_mm"):
            if not _numeric_sequence_close(
                current_plane.get(key),
                stored_plane.get(key),
                tolerance=tolerance,
            ):
                return False
    return True


def _numeric_sequence_close(left, right, *, tolerance):
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return left is None and right is None
    if len(left) != len(right):
        return False
    try:
        return all(
            abs(float(a) - float(b)) <= float(tolerance)
            for a, b in zip(left, right, strict=True)
        )
    except Exception:
        return False


def _dot(a, b):
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _rotate_vector_about_axis(vector, axis, angle_radians):
    axis = _normalized(axis)
    vector = tuple(float(value) for value in vector)
    cos_angle = math.cos(float(angle_radians))
    sin_angle = math.sin(float(angle_radians))
    cross = _cross(axis, vector)
    dot = _dot(axis, vector)
    return tuple(
        vector[index] * cos_angle
        + cross[index] * sin_angle
        + axis[index] * dot * (1.0 - cos_angle)
        for index in range(3)
    )


def _style_interactive_plane(plane):
    if plane is None:
        return
    plane.CreateDefaultDisplayNodes()
    display = plane.GetDisplayNode()
    if display is None:
        return
    for method, args in (
        ("SetHandlesInteractive", (True,)),
        ("SetVisibility", (True,)),
        ("SetTranslationHandleVisibility", (True,)),
        ("SetRotationHandleVisibility", (True,)),
        ("SetScaleHandleVisibility", (True,)),
        ("SetNormalVisibility", (True,)),
        ("SetVisibility2D", (True,)),
        ("SetVisibility3D", (True,)),
        ("SetInteractionHandleScale", (1.5,)),
        ("SetOpacity", (0.35,)),
        ("SetFillVisibility", (True,)),
        ("SetOutlineVisibility", (True,)),
    ):
        if hasattr(display, method):
            try:
                getattr(display, method)(*args)
            except Exception:
                pass


def _configure_plane_table_visibility(table):
    table.setColumnHidden(1, True)
    table.setColumnHidden(2, True)
    for column in range(5, 11):
        table.setColumnHidden(column, True)
    table.setColumnHidden(12, False)
    table.setColumnHidden(13, False)


def _configure_resizable_table(table):
    try:
        table.setMinimumWidth(0)
    except Exception:
        pass
    try:
        table.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
    except Exception:
        pass
    try:
        table.setSizeAdjustPolicy(qt.QAbstractScrollArea.AdjustIgnored)
    except Exception:
        pass
    try:
        table.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    except Exception:
        pass


def _inside_shape(shape, u, v, *, half_u_mm, half_v_mm):
    shape = str(shape).strip().lower()
    half_u = max(float(half_u_mm), 1e-9)
    half_v = max(float(half_v_mm), 1e-9)
    if shape in {"anatomy", "rectangle", "rectangular"}:
        return abs(u) <= half_u and abs(v) <= half_v
    if shape == "square":
        half = min(half_u, half_v)
        return abs(u) <= half and abs(v) <= half
    if shape == "hex":
        half = min(half_u, half_v)
        uu = u / half
        vv = v / half
        return abs(uu) <= 1.0 and abs(0.5 * uu + 0.8660254 * vv) <= 1.0 and abs(0.5 * uu - 0.8660254 * vv) <= 1.0
    if shape in {"oval", "round"}:
        return (float(u) / half_u) ** 2 + (float(v) / half_v) ** 2 <= 1.0
    return abs(u) <= half_u and abs(v) <= half_v


def _inside_shape_array(shape, u, v, *, half_u_mm, half_v_mm):
    shape = str(shape).strip().lower()
    half_u = max(float(half_u_mm), 1e-9)
    half_v = max(float(half_v_mm), 1e-9)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if shape in {"anatomy", "rectangle", "rectangular"}:
        return (np.abs(u) <= half_u) & (np.abs(v) <= half_v)
    if shape == "square":
        half = min(half_u, half_v)
        return (np.abs(u) <= half) & (np.abs(v) <= half)
    if shape == "hex":
        half = min(half_u, half_v)
        uu = u / half
        vv = v / half
        return (
            (np.abs(uu) <= 1.0)
            & (np.abs(0.5 * uu + 0.8660254 * vv) <= 1.0)
            & (np.abs(0.5 * uu - 0.8660254 * vv) <= 1.0)
        )
    if shape in {"oval", "round"}:
        return (u / half_u) ** 2 + (v / half_v) ** 2 <= 1.0
    return (np.abs(u) <= half_u) & (np.abs(v) <= half_v)


def _uses_anatomy_constraint(shape, *, anatomy_constrained=False):
    if str(shape).strip().lower() == "anatomy":
        return True
    return _enabled_value(anatomy_constrained)


def _target_mask_array(node, reference_node=None, *, active_values=None, fallback_to_nonzero=False):
    if node is None:
        return None
    try:
        array = np.asarray(_array_from_mask_like(node, reference_node))
    except Exception:
        return None
    if active_values is not None:
        values = tuple(int(value) for value in active_values)
        if values:
            active = np.isin(array, values)
            if np.any(active) or not bool(fallback_to_nonzero):
                return active
    return array != 0


def _target_values_tuple(active_values):
    if active_values is None:
        return None
    if isinstance(active_values, str):
        parts = [
            item.strip()
            for item in active_values.replace(";", ",").split(",")
            if item.strip()
        ]
        return tuple(int(item) for item in parts) or None
    try:
        return tuple(int(value) for value in active_values) or None
    except TypeError:
        return (int(active_values),)


def _node_csv_attribute(node, attribute):
    if node is None or not hasattr(node, "GetAttribute"):
        return []
    text = node.GetAttribute(attribute) or ""
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _selected_label_values_for_node(node):
    values = []
    for item in _node_csv_attribute(node, LABEL_SELECTION_VALUES_ATTRIBUTE):
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(values) or None


def _label_values_from_node(node, reference_node=None):
    if node is None:
        return []
    if _is_segmentation_node(node):
        labels = []
        try:
            segmentation = node.GetSegmentation()
            for segment_id in _segmentation_segment_ids(node):
                segment = segmentation.GetSegment(segment_id)
                labels.append(_segment_label_value(segment))
        except Exception:
            return []
        return sorted({int(value) for value in labels if int(value) != 0})
    try:
        array = np.asarray(_array_from_mask_like(node, reference_node, apply_selection=False))
    except Exception:
        return []
    values = []
    for value in np.unique(array):
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer != 0 and abs(float(value) - integer) <= 1.0e-6:
            values.append(integer)
        if len(values) >= 200:
            break
    return values


def _combo_data(widget):
    if widget is None:
        return None
    try:
        return widget.itemData(widget.currentIndex)
    except Exception:
        try:
            return widget.currentData()
        except Exception:
            return None


def _combo_count(widget):
    try:
        count = widget.count
        return int(count() if callable(count) else count)
    except Exception:
        return 0


def _list_widget_count(widget):
    try:
        count = widget.count
        return int(count() if callable(count) else count)
    except Exception:
        return 0


def _set_combo_data(widget, value):
    if widget is None:
        return
    text = "" if value is None else str(value)
    for index in range(_combo_count(widget)):
        try:
            if str(widget.itemData(index)) == text:
                widget.setCurrentIndex(index)
                return
        except Exception:
            pass
    if text:
        try:
            widget.addItem(f"Label {text}", text)
            widget.setCurrentIndex(_combo_count(widget) - 1)
        except Exception:
            pass


def _combo_selected_int_tuple(widget):
    value = _combo_data(widget)
    if value is None or str(value).strip() == "":
        return None
    try:
        return (int(value),)
    except (TypeError, ValueError):
        return None


def _registration_target_image_source(registration):
    if not isinstance(registration, dict):
        return "workflow-reference"
    source = str(registration.get("target_image", "") or "").strip().lower()
    if source in {"workflow-reference", "self", "slicer-node"}:
        return source
    if registration.get("self_reference") or registration.get("reference_authoring"):
        return "self"
    if _enabled_value(registration.get("enabled", False)) and not registration.get("reference_points"):
        return "self"
    return "workflow-reference"


def _node_reference_description(node):
    if node is None:
        return "Slicer node"
    try:
        name = node.GetName()
    except Exception:
        name = ""
    try:
        node_id = node.GetID()
    except Exception:
        node_id = ""
    if name and node_id:
        return f"Slicer node {name} ({node_id})"
    return f"Slicer node {name or node_id or 'unknown'}"


def _first_int_text(values):
    if not values:
        return ""
    try:
        return str(int(tuple(values)[0]))
    except (TypeError, ValueError, IndexError):
        return ""


def _promote_material_label_array_for_values(array, values):
    material = np.asarray(array)
    if not np.issubdtype(material.dtype, np.integer):
        return material
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return material
    info = np.iinfo(material.dtype)
    if int(values.min()) >= int(info.min) and int(values.max()) <= int(info.max):
        return material
    if int(values.min()) >= np.iinfo(np.int16).min and int(values.max()) <= np.iinfo(np.int16).max:
        return material.astype(np.int16, copy=False)
    return material.astype(np.int32, copy=False)


def _aspect_ratio_zyx(value):
    if isinstance(value, dict):
        raw = value.get("ratio", value.get("ratios", value.get("aspect_ratio")))
        if raw is None:
            raw = value
    else:
        raw = value
    if isinstance(raw, dict):
        ordered = [raw.get("z"), raw.get("y"), raw.get("x")]
    elif isinstance(raw, str):
        text = raw.strip().strip("[]()")
        ordered = [part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()]
    else:
        ordered = list(raw) if isinstance(raw, (list, tuple)) else []
    if len(ordered) != 3:
        raise ValueError("Aspect-ratio crop needs three z/y/x values, for example 1.2, 1, none.")

    parsed = []
    for item in ordered:
        if item is None:
            parsed.append(None)
            continue
        token = str(item).strip().lower()
        if token in {"", "none", "null", "auto"}:
            parsed.append(None)
            continue
        value_float = float(item)
        if value_float <= 0:
            raise ValueError("Aspect-ratio crop values must be positive or none.")
        parsed.append(float(value_float))
    return parsed[0], parsed[1], parsed[2]


def _format_aspect_ratio_zyx(ratio_zyx):
    parts = []
    for value in _aspect_ratio_zyx(ratio_zyx):
        if value is None:
            parts.append("none")
        else:
            parts.append(f"{float(value):g}")
    return ", ".join(parts)


def _bbox_fraction_bounds(value):
    if value is None:
        return None
    if isinstance(value, dict):
        raw = [value.get(axis) for axis in ("x", "y", "z")]
    else:
        raw = list(value) if isinstance(value, (list, tuple)) else []
    if len(raw) != 3 or any(item is None for item in raw):
        raise ValueError("bbox_fraction_bounds must define x, y, and z bounds")
    bounds = []
    for item in raw:
        values = list(item) if isinstance(item, (list, tuple)) else [item, item]
        if len(values) != 2:
            raise ValueError("each bbox_fraction_bounds axis must contain min and max")
        bounds.append([float(values[0]), float(values[1])])
    return np.asarray(bounds, dtype=np.float64)


def _bbox_fraction_bounds_metadata(value):
    bounds = _bbox_fraction_bounds(value)
    if bounds is None:
        return {}
    return {
        axis: [float(bounds[index, 0]), float(bounds[index, 1])]
        for index, axis in enumerate(("x", "y", "z"))
    }


def _bbox_ratio(value):
    if isinstance(value, dict):
        raw = value.get("ratio", value.get("ratios", value.get("bbox_ratio")))
        if raw is None:
            raw = value
    else:
        raw = value
    if isinstance(raw, dict):
        ordered = [
            raw.get("reference", raw.get("first")),
            raw.get("constrained", raw.get("second", raw.get("cropped"))),
            raw.get("free", raw.get("third")),
        ]
        if all(item is None for item in ordered):
            ratio_zyx = _aspect_ratio_zyx(raw)
            return ratio_zyx[1], ratio_zyx[0], ratio_zyx[2]
    elif isinstance(raw, str):
        text = raw.strip().strip("[]()")
        ordered = [part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()]
    else:
        ordered = list(raw) if isinstance(raw, (list, tuple)) else []
    if len(ordered) != 3:
        raise ValueError("BBox-ratio crop needs three values, for example 1, 1.2, none.")

    parsed = []
    for item in ordered:
        if item is None:
            parsed.append(None)
            continue
        token = str(item).strip().lower()
        if token in {"", "none", "null", "auto"}:
            parsed.append(None)
            continue
        value_float = float(item)
        if value_float <= 0:
            raise ValueError("BBox-ratio crop values must be positive or none.")
        parsed.append(float(value_float))
    return parsed[0], parsed[1], parsed[2]


def _bbox_ratio_to_zyx(value):
    reference, constrained, free = _bbox_ratio(value)
    return constrained, reference, free


def _format_bbox_ratio(bbox_ratio):
    parts = []
    for value in _bbox_ratio(bbox_ratio):
        if value is None:
            parts.append("none")
        else:
            parts.append(f"{float(value):g}")
    return ", ".join(parts)


def _bbox_crop_from(value):
    if isinstance(value, dict):
        raw = value.get("crop_from", value.get("bbox_crop_from", value))
    else:
        raw = value
    if isinstance(raw, dict):
        ordered = [
            raw.get("reference", raw.get("first")),
            raw.get("constrained", raw.get("second", raw.get("cropped"))),
            raw.get("free", raw.get("third")),
        ]
        if all(item is None for item in ordered):
            ordered = [raw.get("z"), raw.get("y"), raw.get("x")]
            if any(item is not None for item in ordered):
                return tuple(_crop_from_value(item) for item in ordered)
    elif isinstance(raw, str):
        text = raw.strip().strip("[]()")
        ordered = [part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()]
    else:
        ordered = list(raw) if isinstance(raw, (list, tuple)) else []
    if not ordered:
        return None, None, None
    if len(ordered) != 3:
        raise ValueError("BBox crop-from needs three values, for example center, min, center.")
    return tuple(_crop_from_value(item) for item in ordered)


def _bbox_crop_from_to_zyx(value):
    reference, constrained, free = _bbox_crop_from(value)
    return constrained, reference, free


def _crop_from_value(value):
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"", "none", "null", "auto", "center", "centre"}:
        return None
    if token in {"min", "low", "lo", "start"}:
        return "min"
    if token in {"max", "high", "hi", "end"}:
        return "max"
    raise ValueError("BBox crop-from values must be min, max, center, or none.")


def _format_bbox_crop_from(value):
    parts = []
    for item in _bbox_crop_from(value):
        parts.append("center" if item is None else str(item))
    return ", ".join(parts)


def _aspect_ratio_crop_slices_zyx(
    mask_array,
    ratio_zyx,
    *,
    spacing_xyz,
    crop_from_zyx=None,
):
    mask_array = np.asarray(mask_array)
    active = mask_array != 0
    if not np.any(active):
        raise ValueError("Mask is empty; cannot crop to an aspect ratio.")

    ratio_zyx = _aspect_ratio_zyx(ratio_zyx)
    numeric_axes = [axis for axis, value in enumerate(ratio_zyx) if value is not None]
    if not numeric_axes:
        full_slices = tuple(slice(0, int(mask_array.shape[axis])) for axis in range(3))
        return full_slices, (0, 0, 0), {
            "source_slices_zyx": tuple((sl.start, sl.stop) for sl in full_slices),
            "bbox_zyx": tuple((0, int(mask_array.shape[axis])) for axis in range(3)),
        }

    reference_axes = [
        axis for axis in numeric_axes if np.isclose(float(ratio_zyx[axis]), 1.0)
    ]
    if not reference_axes:
        raise ValueError("Aspect-ratio crop needs one preserved axis with ratio 1.")
    coords = np.argwhere(active)
    lo_zyx = coords.min(axis=0).astype(np.int64)
    hi_zyx = (coords.max(axis=0) + 1).astype(np.int64)
    size_zyx = hi_zyx - lo_zyx
    spacing_xyz = [abs(float(value)) for value in spacing_xyz]
    spacing_zyx = np.asarray(
        [
            spacing_xyz[2] if len(spacing_xyz) > 2 and spacing_xyz[2] > 0 else 1.0,
            spacing_xyz[1] if len(spacing_xyz) > 1 and spacing_xyz[1] > 0 else 1.0,
            spacing_xyz[0] if spacing_xyz and spacing_xyz[0] > 0 else 1.0,
        ],
        dtype=np.float64,
    )
    physical_size_zyx = size_zyx.astype(np.float64) * spacing_zyx
    reference_axis = int(
        min(reference_axes, key=lambda axis: float(physical_size_zyx[axis]))
    )
    reference_length_mm = float(size_zyx[reference_axis]) * float(spacing_zyx[reference_axis])
    crop_from_zyx = crop_from_zyx or (None, None, None)
    warnings = []

    out_lo = lo_zyx.copy()
    out_hi = hi_zyx.copy()
    for axis, axis_ratio in enumerate(ratio_zyx):
        if axis_ratio is None:
            continue
        target_mm = reference_length_mm * float(axis_ratio)
        requested_voxels = max(1, int(round(target_mm / float(spacing_zyx[axis]))))
        available_voxels = int(size_zyx[axis])
        if requested_voxels > available_voxels:
            axis_name = ("z", "y", "x")[axis]
            warnings.append(
                "bbox_ratio cannot reach requested bbox_ratio on "
                f"{axis_name} axis: requested {target_mm:g} mm "
                f"({requested_voxels} voxels) exceeds foreground extent "
                f"{float(physical_size_zyx[axis]):g} mm ({available_voxels} voxels); "
                f"using the full available {axis_name} extent."
            )
        target_voxels = min(available_voxels, requested_voxels)
        crop_from = crop_from_zyx[axis]
        if crop_from == "min":
            start = int(hi_zyx[axis]) - target_voxels
        elif crop_from == "max":
            start = int(lo_zyx[axis])
        else:
            center = 0.5 * (float(lo_zyx[axis]) + float(hi_zyx[axis]))
            start = int(round(center - 0.5 * float(target_voxels)))
        start = max(int(lo_zyx[axis]), min(start, int(hi_zyx[axis]) - target_voxels))
        out_lo[axis] = start
        out_hi[axis] = start + target_voxels

    slices = tuple(slice(int(out_lo[axis]), int(out_hi[axis])) for axis in range(3))
    offset_ijk = (int(out_lo[2]), int(out_lo[1]), int(out_lo[0]))
    return slices, offset_ijk, {
        "source_slices_zyx": tuple((sl.start, sl.stop) for sl in slices),
        "bbox_zyx": tuple((int(lo_zyx[axis]), int(hi_zyx[axis])) for axis in range(3)),
        "aspect_ratio_warnings": tuple(warnings),
    }


def _copy_cropped_geometry(source_node, cropped_node, offset_ijk):
    ijk_to_ras = vtk.vtkMatrix4x4()
    source_node.GetIJKToRASMatrix(ijk_to_ras)
    cropped_ijk_to_ras = vtk.vtkMatrix4x4()
    cropped_ijk_to_ras.DeepCopy(ijk_to_ras)
    shifted_origin = ijk_to_ras.MultiplyPoint(
        [
            float(offset_ijk[0]),
            float(offset_ijk[1]),
            float(offset_ijk[2]),
            1.0,
        ]
    )[:3]
    for row in range(3):
        cropped_ijk_to_ras.SetElement(row, 3, float(shifted_origin[row]))
    cropped_node.SetIJKToRASMatrix(cropped_ijk_to_ras)
    cropped_node.Modified()


def _crop_margin_zyx(volume_node, *, margin_voxels, padding_mm):
    if padding_mm is None:
        margin = max(0, int(margin_voxels))
        return np.asarray((margin, margin, margin), dtype=np.int64)
    spacing_xyz = [abs(float(value)) for value in volume_node.GetSpacing()]
    spacing_zyx = np.asarray(
        [
            spacing_xyz[2] if spacing_xyz[2] > 0 else 1.0,
            spacing_xyz[1] if spacing_xyz[1] > 0 else 1.0,
            spacing_xyz[0] if spacing_xyz[0] > 0 else 1.0,
        ],
        dtype=np.float64,
    )
    margin_from_mm = np.ceil(float(padding_mm) / spacing_zyx).astype(np.int64)
    return np.maximum(margin_from_mm, max(0, int(margin_voxels)))


def _padded_crop_array(array, requested_mins, requested_maxs):
    requested_mins = np.asarray(requested_mins, dtype=np.int64)
    requested_maxs = np.asarray(requested_maxs, dtype=np.int64)
    output_shape = tuple(int(max(1, requested_maxs[axis] - requested_mins[axis])) for axis in range(3))
    cropped = np.zeros(output_shape, dtype=array.dtype)

    source_mins = np.maximum(requested_mins, 0)
    source_maxs = np.minimum(requested_maxs, np.asarray(array.shape, dtype=np.int64))
    if np.any(source_maxs <= source_mins):
        empty = tuple(slice(0, 0) for _axis in range(3))
        return cropped, empty, empty

    target_mins = source_mins - requested_mins
    target_maxs = target_mins + (source_maxs - source_mins)
    source_slices = tuple(slice(int(source_mins[axis]), int(source_maxs[axis])) for axis in range(3))
    target_slices = tuple(slice(int(target_mins[axis]), int(target_maxs[axis])) for axis in range(3))
    cropped[target_slices] = array[source_slices]
    return cropped, source_slices, target_slices


def _is_isotropic_spacing(spacing, *, tolerance=1.0e-6, relative_tolerance=1.0e-3):
    values = [abs(float(value)) for value in spacing if abs(float(value)) > 0]
    if not values:
        return True
    span = max(values) - min(values)
    scale = max(min(values), 1.0e-12)
    return span <= max(float(tolerance), float(relative_tolerance) * scale)


def _resample_mask_like_node(mask_node, source_reference, name, *, target_spacing_mm, reference_node):
    if _is_segmentation_node(mask_node):
        label_node = _segmentation_to_labelmap_node(mask_node, source_reference)
        try:
            return _resample_volume_node(
                label_node,
                name,
                target_spacing_mm=target_spacing_mm,
                label=True,
                reference_node=reference_node,
            )
        finally:
            if label_node is not None and slicer.mrmlScene.IsNodePresent(label_node):
                slicer.mrmlScene.RemoveNode(label_node)
    return _resample_volume_node(
        mask_node,
        name,
        target_spacing_mm=target_spacing_mm,
        label=True,
        reference_node=reference_node,
    )


def _volume_node_from_parosol_ras_array(array_zyx, name, spacing, origin, *, label):
    _remove_named_scene_node(str(name))
    node_class = "vtkMRMLLabelMapVolumeNode" if label else "vtkMRMLScalarVolumeNode"
    output_node = slicer.mrmlScene.AddNewNodeByClass(node_class, str(name))
    array = np.asarray(array_zyx)
    if label:
        array = np.asarray(np.rint(array), dtype=np.uint16)
    else:
        array = np.asarray(array, dtype=np.float32)
    slicer.util.updateVolumeFromArray(output_node, array)
    output_node.SetIJKToRASMatrix(_ijk_to_ras_from_origin_spacing(origin, spacing))
    output_node.CreateDefaultDisplayNodes()
    output_node.Modified()
    return output_node


def _canonicalize_volume_node_to_parosol_ras_grid(source_node, name, *, label):
    if source_node is None:
        return None
    if _volume_node_has_parosol_ras_grid(source_node):
        return source_node
    image = _sitk_image_from_slicer_volume_node(source_node)
    oriented = sitk.DICOMOrient(image, "RAS")
    _remove_named_scene_node(str(name))
    node_class = "vtkMRMLLabelMapVolumeNode" if label else "vtkMRMLScalarVolumeNode"
    output_node = slicer.mrmlScene.AddNewNodeByClass(node_class, str(name))
    array = sitk.GetArrayFromImage(oriented)
    if label:
        array = np.asarray(np.rint(array), dtype=np.uint16)
    slicer.util.updateVolumeFromArray(output_node, array)
    _set_slicer_volume_geometry_from_sitk_image(output_node, oriented)
    output_node.CreateDefaultDisplayNodes()
    output_node.Modified()
    return output_node


def _sitk_image_from_slicer_volume_node(node):
    if node is None or not hasattr(node, "GetImageData") or node.GetImageData() is None:
        raise ValueError("Volume node has no image data.")
    image = sitk.GetImageFromArray(np.asarray(slicer.util.arrayFromVolume(node)))
    ijk_to_ras = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(ijk_to_ras)
    matrix = np.asarray(
        [[ijk_to_ras.GetElement(row, column) for column in range(4)] for row in range(4)],
        dtype=float,
    )
    axes_ras = matrix[:3, :3]
    spacing = np.linalg.norm(axes_ras, axis=0)
    spacing = np.where(spacing > 1.0e-12, spacing, 1.0)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    axes_lps = ras_to_lps @ axes_ras
    direction_lps = axes_lps @ np.diag(1.0 / spacing)
    image.SetSpacing(tuple(float(value) for value in spacing))
    image.SetOrigin(tuple(float(value) for value in (ras_to_lps @ matrix[:3, 3])))
    image.SetDirection(tuple(float(value) for value in direction_lps.reshape(-1)))
    return image


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


def _volume_node_has_parosol_ras_grid(node):
    if node is None or _is_segmentation_node(node) or not hasattr(node, "GetImageData"):
        return False
    if node.GetImageData() is None:
        return False
    matrix = vtk.vtkMatrix4x4()
    try:
        node.GetIJKToRASMatrix(matrix)
    except Exception:
        return False
    axes = np.asarray(
        [[matrix.GetElement(row, column) for column in range(3)] for row in range(3)],
        dtype=float,
    )
    spacing = np.linalg.norm(axes, axis=0)
    if np.any(spacing <= 1.0e-12):
        return False
    direction = axes @ np.diag(1.0 / spacing)
    return bool(np.allclose(direction, np.eye(3), atol=1.0e-6))


def _resample_volume_node(source_node, name, *, target_spacing_mm, label, reference_node=None):
    image_data = source_node.GetImageData()
    if image_data is None:
        raise ValueError(f"Node has no image data: {source_node.GetName()}")
    dims = image_data.GetDimensions()
    old_spacing = tuple(abs(float(value)) for value in source_node.GetSpacing())
    new_spacing = (float(target_spacing_mm),) * 3
    output_dims = [
        max(1, int(math.ceil(float(dims[index]) * old_spacing[index] / new_spacing[index])))
        for index in range(3)
    ]

    old_ijk_to_ras = vtk.vtkMatrix4x4()
    source_node.GetIJKToRASMatrix(old_ijk_to_ras)
    new_ijk_to_ras = vtk.vtkMatrix4x4()
    _scaled_ijk_to_ras(old_ijk_to_ras, old_spacing, new_spacing, new_ijk_to_ras)

    reslice_axes = vtk.vtkMatrix4x4()
    inverse_old = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(old_ijk_to_ras, inverse_old)
    vtk.vtkMatrix4x4.Multiply4x4(inverse_old, new_ijk_to_ras, reslice_axes)

    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(image_data)
    reslice.SetResliceAxes(reslice_axes)
    reslice.SetOutputExtent(0, output_dims[0] - 1, 0, output_dims[1] - 1, 0, output_dims[2] - 1)
    reslice.SetOutputSpacing(1.0, 1.0, 1.0)
    reslice.SetOutputOrigin(0.0, 0.0, 0.0)
    if label:
        reslice.SetInterpolationModeToNearestNeighbor()
    else:
        reslice.SetInterpolationModeToLinear()
    reslice.Update()

    node_class = "vtkMRMLLabelMapVolumeNode" if label else "vtkMRMLScalarVolumeNode"
    output_node = slicer.mrmlScene.AddNewNodeByClass(node_class, name)
    output_node.SetAndObserveImageData(reslice.GetOutput())
    output_node.SetIJKToRASMatrix(new_ijk_to_ras)
    if reference_node is not None:
        output_node.CreateDefaultDisplayNodes()
    output_node.Modified()
    return output_node


def _ijk_to_ras_from_origin_spacing(origin, spacing):
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for axis in range(3):
        matrix.SetElement(axis, axis, float(spacing[axis]))
        matrix.SetElement(axis, 3, float(origin[axis]))
    return matrix

def _smooth_volume_node(source_node, name, *, sigma_mm, label=False):
    if source_node is None:
        raise ValueError("Smooth requires an input volume.")
    image_data = source_node.GetImageData()
    if image_data is None:
        raise ValueError(f"Node has no image data: {source_node.GetName()}")
    _remove_named_scene_node(name)
    spacing = tuple(max(abs(float(value)), 1.0e-12) for value in source_node.GetSpacing())
    sigma_vox = tuple(float(sigma_mm) / spacing[index] for index in range(3))
    smooth = vtk.vtkImageGaussianSmooth()
    smooth.SetInputData(image_data)
    smooth.SetStandardDeviations(*sigma_vox)
    smooth.SetRadiusFactors(2.0, 2.0, 2.0)
    smooth.Update()
    node_class = "vtkMRMLLabelMapVolumeNode" if label else "vtkMRMLScalarVolumeNode"
    output_node = slicer.mrmlScene.AddNewNodeByClass(node_class, name)
    output_node.SetAndObserveImageData(smooth.GetOutput())
    try:
        output_node.CopyOrientation(source_node)
    except Exception:
        matrix = vtk.vtkMatrix4x4()
        source_node.GetIJKToRASMatrix(matrix)
        output_node.SetIJKToRASMatrix(matrix)
    output_node.CreateDefaultDisplayNodes()
    output_node.Modified()
    return output_node


def _largest_connected_component_mask(foreground):
    foreground = np.asarray(foreground, dtype=bool)
    if foreground.ndim != 3:
        raise ValueError("Largest connected component expects a 3D mask.")
    try:
        return _largest_connected_component_mask_numpy(foreground)
    except MemoryError:
        raise


def _largest_connected_component_mask_numpy(foreground):
    foreground = np.asarray(foreground, dtype=bool)
    if not np.any(foreground):
        return np.zeros_like(foreground, dtype=bool)
    visited = np.zeros_like(foreground, dtype=bool)
    best_component = []
    shape = foreground.shape
    starts = np.argwhere(foreground)
    for start in starts:
        z, y, x = (int(start[0]), int(start[1]), int(start[2]))
        if visited[z, y, x]:
            continue
        stack = [(z, y, x)]
        visited[z, y, x] = True
        component = []
        while stack:
            cz, cy, cx = stack.pop()
            component.append((cz, cy, cx))
            for nz, ny, nx in (
                (cz - 1, cy, cx),
                (cz + 1, cy, cx),
                (cz, cy - 1, cx),
                (cz, cy + 1, cx),
                (cz, cy, cx - 1),
                (cz, cy, cx + 1),
            ):
                if (
                    0 <= nz < shape[0]
                    and 0 <= ny < shape[1]
                    and 0 <= nx < shape[2]
                    and foreground[nz, ny, nx]
                    and not visited[nz, ny, nx]
                ):
                    visited[nz, ny, nx] = True
                    stack.append((nz, ny, nx))
        if len(component) > len(best_component):
            best_component = component
    output = np.zeros_like(foreground, dtype=bool)
    if best_component:
        coords = np.asarray(best_component, dtype=np.int64)
        output[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return output


def _workflow_target_active_values(model, target):
    if not isinstance(model, dict):
        return None
    if isinstance(target, (int, float)) and abs(float(target) - int(target)) <= 1.0e-6:
        return (int(target),)
    target = str(target).strip().lower()
    if not target:
        return None
    try:
        return (int(target),)
    except (TypeError, ValueError):
        pass
    labels = model.get("labels", {})
    if not isinstance(labels, dict):
        return None

    def label_value(*keys):
        for key in keys:
            value = labels.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    if target in {"vertebral_body", "body"}:
        value = label_value("body", "vertebral_body")
        return (value,) if value is not None else None
    if target in {"vertebral_process", "process", "posterior_elements"}:
        value = label_value("process", "vertebral_process")
        return (value,) if value is not None else None
    if target in {"vertebra", "spine", "all", "body_and_process", "vertebral_body_and_process"}:
        values = [
            value
            for value in (
                label_value("body", "vertebral_body"),
                label_value("process", "vertebral_process"),
            )
            if value is not None
        ]
        return tuple(values) or None
    return None


def _workflow_registration_active_values(model, registration):
    if not isinstance(model, dict) or not isinstance(registration, dict):
        return None
    targets = model.get("targets", {})
    target = None
    if isinstance(targets, dict):
        target = targets.get("registration")
    if target is None:
        target = registration.get("target", registration.get("target_label", ""))
    return _workflow_target_active_values(model, target)


def _workflow_disk_projection_active_values(model):
    if not isinstance(model, dict):
        return None
    targets = model.get("targets", {})
    target = None
    if isinstance(targets, dict):
        target = targets.get("disk_projection", targets.get("disk"))
    geometry = model.get("geometry", {})
    disk = geometry.get("disk", {}) if isinstance(geometry, dict) else {}
    if target is None and isinstance(disk, dict):
        target = disk.get("target", disk.get("target_label"))
    values = _workflow_target_active_values(model, target)
    if values is not None:
        return values
    if isinstance(disk, dict) and disk.get("target_label") is not None:
        try:
            return (int(disk.get("target_label")),)
        except (TypeError, ValueError):
            return None
    return None


def _sample_reference_points_from_mask_like(
    mask_like_node,
    reference_node,
    *,
    max_points=8000,
    active_values=None,
):
    if mask_like_node is None or reference_node is None:
        return np.zeros((0, 3), dtype=np.float32)
    array = np.asarray(_array_from_mask_like(mask_like_node, reference_node))
    active = None
    if active_values is not None:
        values = tuple(int(value) for value in active_values)
        if values:
            active = np.isin(array, values)
            if not np.any(active) and np.any(array != 0):
                active = None
    if active is None:
        active = array != 0
    if active.ndim != 3 or not np.any(active):
        return np.zeros((0, 3), dtype=np.float32)
    surface = _surface_mask_6_connected(active)
    indices_zyx = np.argwhere(surface)
    if indices_zyx.size == 0:
        indices_zyx = np.argwhere(active)
    sampled = _grid_sample_indices_zyx(indices_zyx, max_points=max(1, int(max_points)))
    ijk_to_ras = vtk.vtkMatrix4x4()
    reference_node.GetIJKToRASMatrix(ijk_to_ras)
    points = np.zeros((sampled.shape[0], 3), dtype=np.float32)
    for point_index, index in enumerate(sampled):
        ijk = (float(index[2]), float(index[1]), float(index[0]))
        points[point_index, :] = ijk_to_ras.MultiplyPoint([*ijk, 1.0])[:3]
    return points


def _sample_reference_points_from_parosol_preview_mask(
    mask_zyx,
    *,
    spacing,
    origin,
    max_points=8000,
    active_values=None,
):
    array = np.asarray(mask_zyx)
    active = None
    if active_values is not None:
        values = tuple(int(value) for value in active_values)
        if values:
            active = np.isin(array, values)
            if not np.any(active) and np.any(array != 0):
                active = None
    if active is None:
        active = array != 0
    if active.ndim != 3 or not np.any(active):
        return np.zeros((0, 3), dtype=np.float32)
    surface = _surface_mask_6_connected(active)
    indices_zyx = np.argwhere(surface)
    if indices_zyx.size == 0:
        indices_zyx = np.argwhere(active)
    sampled = _grid_sample_indices_zyx(indices_zyx, max_points=max(1, int(max_points)))
    spacing_xyz = np.asarray(spacing, dtype=np.float64)
    origin_xyz = np.asarray(origin, dtype=np.float64)
    points = origin_xyz + sampled[:, [2, 1, 0]].astype(np.float64) * spacing_xyz
    return np.asarray(points, dtype=np.float32)


def _self_reference_registration_metadata(registration_cfg, *, point_count, transform=None):
    cfg = registration_cfg if isinstance(registration_cfg, dict) else {}
    transform = transform or {
        "rotation": np.eye(3, dtype=float),
        "translation": np.zeros(3, dtype=float),
        "iterations": 0,
        "mean_distance": 0.0,
    }
    return {
        "enabled": True,
        "reference_authoring": True,
        "self_reference": True,
        "target_image": "self",
        "applied_to_model_grid": False,
        "method": str(cfg.get("method", "lightweight_icp")),
        "iterations": int(transform.get("iterations", 0)),
        "mean_distance": float(transform.get("mean_distance", 0.0)),
        "rotation": np.asarray(transform.get("rotation", np.eye(3)), dtype=float).tolist(),
        "translation": np.asarray(transform.get("translation", np.zeros(3)), dtype=float).tolist(),
        "point_count": int(point_count),
    }


def _foreground_extent_points_from_mask_like(mask_like_node, reference_node):
    if mask_like_node is None or reference_node is None:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        array = np.asarray(_array_from_mask_like(mask_like_node, reference_node))
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    indices = np.argwhere(array != 0)
    if indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mins_zyx = indices.min(axis=0)
    maxs_zyx = indices.max(axis=0)
    ijk_to_ras = vtk.vtkMatrix4x4()
    reference_node.GetIJKToRASMatrix(ijk_to_ras)
    points = []
    for k in (int(mins_zyx[0]), int(maxs_zyx[0])):
        for j in (int(mins_zyx[1]), int(maxs_zyx[1])):
            for i in (int(mins_zyx[2]), int(maxs_zyx[2])):
                points.append(ijk_to_ras.MultiplyPoint([float(i), float(j), float(k), 1.0])[:3])
    return np.asarray(points, dtype=np.float32)


def _surface_mask_6_connected(active):
    active = np.asarray(active, dtype=bool)
    padded = np.pad(active, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return active & ~interior


def _remove_named_scene_node(name):
    try:
        node = slicer.util.getNode(str(name))
    except Exception:
        return
    try:
        if node is not None and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)
    except Exception:
        pass


def _scaled_ijk_to_ras(old_ijk_to_ras, old_spacing, new_spacing, output_matrix):
    output_matrix.DeepCopy(old_ijk_to_ras)
    for axis in range(3):
        scale = float(new_spacing[axis]) / max(float(old_spacing[axis]), 1.0e-12)
        for row in range(3):
            output_matrix.SetElement(row, axis, old_ijk_to_ras.GetElement(row, axis) * scale)


def _is_segmentation_node(node):
    return node is not None and node.IsA("vtkMRMLSegmentationNode")


def _cached_preview_segmentation(name, signature):
    try:
        node = slicer.util.getNode(str(name))
    except Exception:
        return None
    if not _is_segmentation_node(node):
        return None
    if str(node.GetAttribute(PREVIEW_SEGMENT_SIGNATURE_ATTRIBUTE) or "") != str(signature):
        return None
    return node


def _labelmap_preview_in_reference_geometry(label_node, reference_node=None, *, name):
    if label_node is None or reference_node is None:
        return label_node
    try:
        if not label_node.IsA("vtkMRMLLabelMapVolumeNode"):
            return label_node
    except Exception:
        return label_node
    label_image = label_node.GetImageData() if hasattr(label_node, "GetImageData") else None
    reference_image = (
        reference_node.GetImageData() if hasattr(reference_node, "GetImageData") else None
    )
    if label_image is None or reference_image is None:
        return label_node
    if tuple(label_image.GetDimensions()) != tuple(reference_image.GetDimensions()):
        return label_node
    if _volume_geometry_signature(label_node) == _volume_geometry_signature(reference_node):
        return label_node
    preview = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode",
        str(name),
    )
    try:
        slicer.util.updateVolumeFromArray(
            preview,
            np.asarray(slicer.util.arrayFromVolume(label_node)).astype(np.uint16, copy=False),
        )
        preview.CopyOrientation(reference_node)
        preview.CreateDefaultDisplayNodes()
        preview.Modified()
        return preview
    except Exception:
        try:
            if slicer.mrmlScene.IsNodePresent(preview):
                slicer.mrmlScene.RemoveNode(preview)
        except Exception:
            pass
        return label_node


def _integer_labelmap_for_segmentation_import(label_node):
    if label_node is None:
        return label_node
    try:
        array = np.asarray(slicer.util.arrayFromVolume(label_node))
    except Exception:
        return label_node
    if np.issubdtype(array.dtype, np.integer):
        return label_node
    preview = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode",
        f"{label_node.GetName()}_integer_import",
    )
    try:
        integer_array = np.asarray(np.rint(array), dtype=np.int64)
        integer_array = np.clip(integer_array, 0, np.iinfo(np.uint16).max).astype(
            np.uint16,
            copy=False,
        )
        slicer.util.updateVolumeFromArray(preview, integer_array)
        preview.CopyOrientation(label_node)
        preview.CreateDefaultDisplayNodes()
        preview.Modified()
        return preview
    except Exception:
        try:
            if slicer.mrmlScene.IsNodePresent(preview):
                slicer.mrmlScene.RemoveNode(preview)
        except Exception:
            pass
        raise


def _preview_segmentation_signature(label_node, reference_node=None, *, kind="mask"):
    return json.dumps(
        {
            "kind": str(kind),
            "label": _volume_node_signature(label_node),
            "reference": _volume_geometry_signature(reference_node or label_node),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _volume_node_signature(node):
    if node is None:
        return None
    image = node.GetImageData() if hasattr(node, "GetImageData") else None
    scalars = None
    if image is not None:
        try:
            scalars = image.GetPointData().GetScalars()
        except Exception:
            scalars = None
    return {
        "id": str(node.GetID() or ""),
        "geometry": _volume_geometry_signature(node),
        "image_mtime": int(image.GetMTime()) if image is not None else 0,
        "scalar_mtime": int(scalars.GetMTime()) if scalars is not None else 0,
    }


def _volume_geometry_signature(node):
    if node is None:
        return None
    image = node.GetImageData() if hasattr(node, "GetImageData") else None
    dims = tuple(int(value) for value in image.GetDimensions()) if image is not None else ()
    matrix = vtk.vtkMatrix4x4()
    try:
        node.GetIJKToRASMatrix(matrix)
        ijk_to_ras = [
            round(float(matrix.GetElement(row, column)), 8)
            for row in range(4)
            for column in range(4)
        ]
    except Exception:
        ijk_to_ras = []
    return {"dims": dims, "ijk_to_ras": ijk_to_ras}


def _volume_grid_matches(first, second, *, tolerance=1.0e-6):
    if first is None or second is None:
        return False
    if _node_array_shape(first) != _node_array_shape(second):
        return False
    try:
        first_matrix = _volume_ijk_to_ras_array(first)
        second_matrix = _volume_ijk_to_ras_array(second)
    except Exception:
        return False
    return bool(np.allclose(first_matrix, second_matrix, atol=float(tolerance), rtol=0.0))


def _segmentation_segment_ids(segmentation_node):
    if segmentation_node is None:
        return []
    try:
        segmentation = segmentation_node.GetSegmentation()
        return [
            str(segmentation.GetNthSegmentID(index))
            for index in range(int(segmentation.GetNumberOfSegments()))
        ]
    except Exception:
        return []


def _segmentation_selected_segment_ids(segmentation_node):
    segment_ids = _segmentation_segment_ids(segmentation_node)
    selected_ids = _node_csv_attribute(segmentation_node, SEGMENT_SELECTION_IDS_ATTRIBUTE)
    if selected_ids:
        return [segment_id for segment_id in selected_ids if segment_id in segment_ids]
    selected_id = segmentation_node.GetAttribute(SEGMENT_SELECTION_ATTRIBUTE)
    if selected_id and selected_id not in {SEGMENT_SELECTION_ALL, SEGMENT_SELECTION_SUBSET}:
        return [selected_id] if selected_id in segment_ids else []
    return segment_ids


def _export_segments_to_labelmap_node(segmentation_node, label_node, reference_node=None):
    segment_ids = _segmentation_selected_segment_ids(segmentation_node)
    if not segment_ids:
        return
    segment_ids_array = vtk.vtkStringArray()
    for segment_id in segment_ids:
        segment_ids_array.InsertNextValue(str(segment_id))
    try:
        if reference_node is not None:
            exported = _segmentations_logic().ExportSegmentsToLabelmapNode(
                segmentation_node,
                segment_ids_array,
                label_node,
                reference_node,
            )
        else:
            exported = _segmentations_logic().ExportSegmentsToLabelmapNode(
                segmentation_node,
                segment_ids_array,
                label_node,
            )
        if exported is not False:
            _preserve_exported_segment_label_values(
                segmentation_node, label_node, segment_ids
            )
            return
    except Exception:
        pass
    try:
        segmentation_node.CreateDefaultDisplayNodes()
    except Exception:
        pass
    display_node = segmentation_node.GetDisplayNode()
    previous_visibility = {}
    if display_node is not None:
        for segment_id in _segmentation_segment_ids(segmentation_node):
            try:
                previous_visibility[segment_id] = bool(display_node.GetSegmentVisibility(segment_id))
                display_node.SetSegmentVisibility(segment_id, segment_id in segment_ids)
            except Exception:
                pass
    try:
        if reference_node is not None:
            _segmentations_logic().ExportVisibleSegmentsToLabelmapNode(
                segmentation_node,
                label_node,
                reference_node,
            )
        else:
            _segmentations_logic().ExportVisibleSegmentsToLabelmapNode(
                segmentation_node,
                label_node,
            )
    finally:
        if display_node is not None:
            for segment_id, visible in previous_visibility.items():
                try:
                    display_node.SetSegmentVisibility(segment_id, visible)
                except Exception:
                    pass
    _preserve_exported_segment_label_values(segmentation_node, label_node, segment_ids)


def _preserve_exported_segment_label_values(segmentation_node, label_node, segment_ids):
    if segmentation_node is None or label_node is None:
        return
    try:
        array = np.asarray(slicer.util.arrayFromVolume(label_node))
    except Exception:
        return
    if array.size == 0:
        return
    try:
        segmentation = segmentation_node.GetSegmentation()
    except Exception:
        segmentation = None
    selected_values = _selected_label_values_for_node(segmentation_node) or ()
    explicit_map = _segment_label_value_map(segmentation_node)
    remap = {}
    for index, segment_id in enumerate(segment_ids, start=1):
        exported_label = index
        target_label = explicit_map.get(str(segment_id))
        if index <= len(selected_values):
            target_label = selected_values[index - 1]
        if target_label is None and segmentation is not None:
            try:
                target_label = _segment_label_value(
                    segmentation.GetSegment(str(segment_id))
                )
            except Exception:
                target_label = None
        try:
            target_label = None if target_label is None else int(target_label)
        except (TypeError, ValueError):
            target_label = None
        if target_label is not None and target_label > 0 and target_label != exported_label:
            remap[exported_label] = target_label
    if not remap:
        return
    remapped = np.asarray(array).copy()
    for source, target in remap.items():
        remapped[array == int(source)] = int(target)
    slicer.util.updateVolumeFromArray(
        label_node, remapped.astype(np.uint16, copy=False)
    )
    label_node.Modified()


def _segment_label_value_map(node):
    values = {}
    text = node.GetAttribute(SEGMENT_LABEL_VALUE_MAP_ATTRIBUTE) if node is not None and hasattr(node, "GetAttribute") else ""
    for item in str(text or "").split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        try:
            values[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return values


def _segmentation_to_labelmap_node(segmentation_node, reference_node=None):
    if segmentation_node is None:
        return None
    label_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode",
        f"{segmentation_node.GetName()}_labelmap",
    )
    _export_segments_to_labelmap_node(segmentation_node, label_node, reference_node)
    return label_node


def _array_from_mask_like(node, reference_node=None, *, apply_selection=True):
    if _is_segmentation_node(node):
        label_node = _segmentation_to_labelmap_node(node, reference_node)
        try:
            return np.asarray(slicer.util.arrayFromVolume(label_node)).copy()
        finally:
            if label_node is not None and slicer.mrmlScene.IsNodePresent(label_node):
                slicer.mrmlScene.RemoveNode(label_node)
    array = slicer.util.arrayFromVolume(node)
    if apply_selection:
        values = _selected_label_values_for_node(node)
        if values:
            return np.where(np.isin(np.asarray(array), values), array, 0)
    return array


def _voxel_tolerance(volume_node):
    try:
        spacing = [abs(float(value)) for value in volume_node.GetSpacing()]
    except Exception:
        spacing = [1.0, 1.0, 1.0]
    positive = [value for value in spacing if value > 0]
    return 0.75 * min(positive or [1.0])


def _mark_flat_cap_face_nodeset(
    nodeset_array,
    disk_array,
    plane,
    disk_labelmap,
    volume_node,
    *,
    disk_label,
    nodeset_label,
):
    mask = np.asarray(disk_array) == int(disk_label)
    if not np.any(mask):
        return
    center = [0.0, 0.0, 0.0]
    normal = [0.0, 0.0, 1.0]
    plane.GetCenter(center)
    try:
        plane.GetNormalWorld(normal)
    except Exception:
        plane.GetNormal(normal)
    normal = _normalized(normal)
    ijk_to_ras = vtk.vtkMatrix4x4()
    disk_labelmap.GetIJKToRASMatrix(ijk_to_ras)
    tolerance = _voxel_tolerance(volume_node)

    distances = []
    indices = np.argwhere(mask)
    for k, j, i in indices:
        ras = ijk_to_ras.MultiplyPoint([int(i), int(j), int(k), 1.0])[:3]
        distances.append(_dot(_subtract(ras, center), normal))
    if not distances:
        return
    outer_distance = min(distances)
    for (k, j, i), distance in zip(indices, distances):
        if abs(float(distance) - outer_distance) <= tolerance:
            nodeset_array[int(k), int(j), int(i)] = int(nodeset_label)


def _surface_index_map(mask, *, side):
    if mask is None or mask.ndim != 3:
        return None
    any_foreground = np.any(mask, axis=0)
    if not np.any(any_foreground):
        return None
    if str(side).strip().lower() == "top":
        reversed_index = np.argmax(mask[::-1, :, :], axis=0)
        surface = mask.shape[0] - 1 - reversed_index
    else:
        surface = np.argmax(mask, axis=0)
    surface = surface.astype(np.int32, copy=False)
    surface[~any_foreground] = -1
    return surface


def _projected_surface_index_map(
    mask,
    *,
    ijk_to_ras,
    plane_center,
    plane_normal,
    min_distance=None,
    max_distance=None,
):
    if mask is None or mask.ndim != 3:
        return None
    foreground = np.argwhere(mask)
    if foreground.size == 0:
        return None

    axis = _dominant_ijk_axis(ijk_to_ras, plane_normal)
    if axis == 0:
        surface = np.full((mask.shape[0], mask.shape[1]), -1, dtype=np.int32)
    elif axis == 1:
        surface = np.full((mask.shape[0], mask.shape[2]), -1, dtype=np.int32)
    else:
        surface = np.full((mask.shape[1], mask.shape[2]), -1, dtype=np.int32)
    best_signed = np.full(surface.shape, np.inf, dtype=np.float64)
    for k, j, i in foreground:
        key = _surface_key(axis, int(i), int(j), int(k))
        ras = ijk_to_ras.MultiplyPoint([int(i), int(j), int(k), 1.0])[:3]
        signed = _dot(_subtract(ras, plane_center), plane_normal)
        if signed < -1e-6:
            continue
        if min_distance is not None and signed < float(min_distance):
            continue
        if max_distance is not None and signed > float(max_distance):
            continue
        if signed < best_signed[key]:
            best_signed[key] = signed
            surface[key] = int((i, j, k)[axis])
    return {"axis": axis, "surface": surface}


def _bounded_projected_surface_context(
    mask,
    *,
    ijk_to_ras,
    plane_center,
    plane_normal,
    u_axis,
    v_axis,
    shape,
    half_u,
    half_v,
    thickness_mm,
    intrusion_depth_mm,
    volume_node=None,
    lateral_margin_mm=None,
    max_search_depth_mm=None,
):
    if mask is None or mask.ndim != 3:
        return None, None
    tolerance = _voxel_tolerance(volume_node) if volume_node is not None else 0.75
    lateral_margin = (
        float(lateral_margin_mm)
        if lateral_margin_mm is not None
        else max(float(tolerance), 1.0e-6)
    )
    max_depth = _bounded_projection_search_depth(
        mask,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        plane_normal=plane_normal,
        u_axis=u_axis,
        v_axis=v_axis,
        shape=shape,
        half_u=half_u,
        half_v=half_v,
        thickness_mm=thickness_mm,
        intrusion_depth_mm=intrusion_depth_mm,
        lateral_margin_mm=lateral_margin,
        max_search_depth_mm=max_search_depth_mm,
        tolerance=tolerance,
    )
    surface_index = _local_projected_surface_index_map(
        mask,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        plane_normal=plane_normal,
        u_axis=u_axis,
        v_axis=v_axis,
        shape=shape,
        half_u=half_u,
        half_v=half_v,
        lateral_margin_mm=lateral_margin,
        min_distance=0.0,
        max_distance=max_depth,
    )
    distances = _surface_index_distances(
        surface_index,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        plane_normal=plane_normal,
    )
    if not distances:
        return None, None
    surface_distance = _robust_projected_surface_distance(distances)
    footprint_min_distance, footprint_max_distance = _projected_cap_footprint_distance_range(
        surface_distance,
        intrusion_depth_mm=intrusion_depth_mm,
    )
    filtered = _filter_surface_index_by_distance(
        surface_index,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        plane_normal=plane_normal,
        min_distance=footprint_min_distance - tolerance,
        max_distance=footprint_max_distance + tolerance,
    )
    return surface_distance, filtered


def _projected_cap_distance_range(surface_distance, *, thickness_mm, intrusion_depth_mm):
    intrusion = max(float(intrusion_depth_mm), 0.0)
    thickness = max(float(thickness_mm), 0.0)
    if surface_distance is None:
        inner_distance = intrusion
    else:
        inner_distance = float(surface_distance) + intrusion
    outer_distance = inner_distance - thickness
    return outer_distance, inner_distance


def _projected_cap_footprint_distance_range(surface_distance, *, intrusion_depth_mm):
    intrusion = max(float(intrusion_depth_mm), 0.0)
    if surface_distance is None:
        return 0.0, intrusion
    surface = float(surface_distance)
    return 0.0, surface + intrusion


def _try_fill_axis_aligned_projected_disk(
    disk_array,
    target_mask,
    output_labelmap,
    volume_node,
    plane,
    *,
    label,
    shape,
    anatomy_constrained=False,
    thickness_mm,
    intrusion_depth_mm,
    radius_mm,
    square_width_mm,
    hex_radius_mm,
    use_plane_size,
    cap_mode,
):
    if disk_array is None or target_mask is None or target_mask.ndim != 3:
        return False
    if str(cap_mode).strip().lower() != "projected_cap":
        return False
    geometry = _plane_geometry(
        plane,
        shape=shape,
        radius_mm=radius_mm,
        square_width_mm=square_width_mm,
        hex_radius_mm=hex_radius_mm,
        use_plane_size=use_plane_size,
    )
    if geometry is None:
        return False
    center, normal, u_axis, v_axis, half_u, half_v = geometry
    ijk_to_ras = vtk.vtkMatrix4x4()
    output_labelmap.GetIJKToRASMatrix(ijk_to_ras)
    ras_to_ijk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(ijk_to_ras, ras_to_ijk)
    axis = _axis_aligned_ijk_axis(ijk_to_ras, normal)
    if axis is None:
        return False
    tolerance = _voxel_tolerance(volume_node)
    surface_distance, surface_map = _axis_aligned_projected_surface_distance_map(
        target_mask,
        ijk_to_ras=ijk_to_ras,
        ras_to_ijk=ras_to_ijk,
        plane_center=center,
        plane_normal=normal,
        u_axis=u_axis,
        v_axis=v_axis,
        shape=shape,
        half_u=half_u,
        half_v=half_v,
        thickness_mm=thickness_mm,
        intrusion_depth_mm=intrusion_depth_mm,
        lateral_margin_mm=tolerance,
        tolerance=tolerance,
        axis=axis,
    )
    if surface_distance is None or surface_map is None:
        return False
    use_anatomy_constraint = _uses_anatomy_constraint(
        shape,
        anatomy_constrained=anatomy_constrained,
    )

    cap_outer_distance, cap_inner_distance = _projected_cap_distance_range(
        surface_distance,
        thickness_mm=thickness_mm,
        intrusion_depth_mm=intrusion_depth_mm,
    )
    footprint_min_distance, footprint_max_distance = _projected_cap_footprint_distance_range(
        surface_distance,
        intrusion_depth_mm=intrusion_depth_mm,
    )
    distance_min, distance_max = _disk_distance_range(
        "projected_cap",
        first_surface_distance=surface_distance,
        opposite_surface_distance=None,
        cap_outer_distance=cap_outer_distance,
        cap_inner_distance=cap_inner_distance,
        thickness_mm=thickness_mm,
        intrusion_depth_mm=intrusion_depth_mm,
        tolerance=tolerance,
    )
    bounds = _projection_ijk_bounds(
        target_mask.shape,
        ras_to_ijk=ras_to_ijk,
        plane_center=center,
        plane_normal=normal,
        u_axis=u_axis,
        v_axis=v_axis,
        half_u=half_u + tolerance,
        half_v=half_v + tolerance,
        min_distance=distance_min,
        max_distance=distance_max,
    )
    if bounds is None:
        return True
    i0, i1, j0, j1, k0, k1 = bounds
    array_axis = _array_axis_from_ijk_axis(axis)
    normal_coords = _normal_coordinates_for_bounds(bounds, array_axis)
    normal_distances = _normal_coordinate_distances(
        normal_coords,
        axis,
        ijk_to_ras=ijk_to_ras,
        ras_to_ijk=ras_to_ijk,
        plane_center=center,
        plane_normal=normal,
    )
    slab_mask = (
        (normal_distances >= float(distance_min))
        & (normal_distances <= float(distance_max))
    )
    if not np.any(slab_mask):
        return True

    column_slices = _surface_map_slices_for_bounds(axis, i0, i1, j0, j1, k0, k1)
    surface_submap = surface_map[column_slices]
    inside_columns = _axis_aligned_footprint_mask(
        axis,
        column_slices,
        ijk_to_ras=ijk_to_ras,
        ras_to_ijk=ras_to_ijk,
        plane_center=center,
        u_axis=u_axis,
        v_axis=v_axis,
        shape=shape,
        half_u=half_u,
        half_v=half_v,
    )
    if use_anatomy_constraint:
        columns = (
            np.isfinite(surface_submap)
            & inside_columns
            & (surface_submap >= footprint_min_distance - tolerance)
            & (surface_submap <= footprint_max_distance + tolerance)
        )
    else:
        columns = inside_columns
    if not np.any(columns):
        return True

    sub_slices = (slice(k0, k1 + 1), slice(j0, j1 + 1), slice(i0, i1 + 1))
    disk_oriented = np.moveaxis(disk_array[sub_slices], array_axis, 0)
    target_oriented = np.moveaxis(np.asarray(target_mask[sub_slices]), array_axis, 0)
    for normal_index in np.flatnonzero(slab_mask):
        writable = columns & ~target_oriented[int(normal_index)].astype(bool)
        if np.any(writable):
            disk_oriented[int(normal_index)][writable] = int(label)
    return True


def _axis_aligned_projected_surface_distance_map(
    mask,
    *,
    ijk_to_ras,
    ras_to_ijk,
    plane_center,
    plane_normal,
    u_axis,
    v_axis,
    shape,
    half_u,
    half_v,
    thickness_mm,
    intrusion_depth_mm,
    lateral_margin_mm,
    tolerance,
    axis,
):
    max_depth = _bounded_projection_search_depth(
        mask,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        plane_normal=plane_normal,
        u_axis=u_axis,
        v_axis=v_axis,
        shape=shape,
        half_u=half_u,
        half_v=half_v,
        thickness_mm=thickness_mm,
        intrusion_depth_mm=intrusion_depth_mm,
        lateral_margin_mm=lateral_margin_mm,
        max_search_depth_mm=None,
        tolerance=tolerance,
    )
    bounds = _projection_ijk_bounds(
        mask.shape,
        ras_to_ijk=ras_to_ijk,
        plane_center=plane_center,
        plane_normal=plane_normal,
        u_axis=u_axis,
        v_axis=v_axis,
        half_u=float(half_u) + float(lateral_margin_mm),
        half_v=float(half_v) + float(lateral_margin_mm),
        min_distance=0.0,
        max_distance=max_depth,
    )
    if bounds is None:
        return None, None
    i0, i1, j0, j1, k0, k1 = bounds
    array_axis = _array_axis_from_ijk_axis(axis)
    sub_slices = (slice(k0, k1 + 1), slice(j0, j1 + 1), slice(i0, i1 + 1))
    oriented = np.moveaxis(np.asarray(mask[sub_slices]), array_axis, 0)
    normal_coords = _normal_coordinates_for_bounds(bounds, array_axis)
    normal_distances = _normal_coordinate_distances(
        normal_coords,
        axis,
        ijk_to_ras=ijk_to_ras,
        ras_to_ijk=ras_to_ijk,
        plane_center=plane_center,
        plane_normal=plane_normal,
    )
    valid_normal = (
        (normal_distances >= -float(tolerance))
        & (normal_distances <= float(max_depth) + float(tolerance))
    )
    if not np.any(valid_normal):
        return None, None
    ordered = np.flatnonzero(valid_normal)
    ordered = ordered[np.argsort(normal_distances[ordered])]
    search = oriented[ordered]
    hits = search.any(axis=0)
    if not np.any(hits):
        return None, None
    first_index = np.argmax(search, axis=0)
    sorted_distances = normal_distances[ordered]
    surface_distances = sorted_distances[first_index]
    surface_normal_coords = normal_coords[ordered[first_index]]
    i_grid, j_grid, k_grid = _axis_aligned_column_ijk_grids(
        axis,
        surface_normal_coords,
        i0=i0,
        i1=i1,
        j0=j0,
        j1=j1,
        k0=k0,
        k1=k1,
    )
    u, v = _axis_aligned_column_uv(
        i_grid,
        j_grid,
        k_grid,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        u_axis=u_axis,
        v_axis=v_axis,
    )
    inside = _inside_shape_array(
        shape,
        u,
        v,
        half_u_mm=float(half_u) + float(lateral_margin_mm),
        half_v_mm=float(half_v) + float(lateral_margin_mm),
    )
    hits = hits & inside
    if not np.any(hits):
        return None, None
    surface_map = np.full(_surface_map_shape(mask.shape, axis), np.nan, dtype=np.float64)
    surface_slices = _surface_map_slices_for_bounds(axis, i0, i1, j0, j1, k0, k1)
    surface_view = surface_map[surface_slices]
    surface_view[hits] = surface_distances[hits]
    values = surface_view[np.isfinite(surface_view)]
    if values.size == 0:
        return None, None
    return _robust_projected_surface_distance(values), surface_map


def _axis_aligned_ijk_axis(ijk_to_ras, plane_normal, *, tolerance=1.0e-4):
    ras_to_ijk_direction = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(ijk_to_ras, ras_to_ijk_direction)
    direction = np.asarray(
        ras_to_ijk_direction.MultiplyPoint(
            [float(plane_normal[0]), float(plane_normal[1]), float(plane_normal[2]), 0.0]
        )[:3],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        return None
    direction /= length
    axis = int(np.argmax(np.abs(direction)))
    off_axis = np.delete(np.abs(direction), axis)
    if off_axis.size and float(np.max(off_axis)) > float(tolerance):
        return None
    return axis


def _array_axis_from_ijk_axis(axis):
    return {0: 2, 1: 1, 2: 0}[int(axis)]


def _normal_coordinates_for_bounds(bounds, array_axis):
    i0, i1, j0, j1, k0, k1 = bounds
    if int(array_axis) == 0:
        return np.arange(k0, k1 + 1, dtype=np.float64)
    if int(array_axis) == 1:
        return np.arange(j0, j1 + 1, dtype=np.float64)
    return np.arange(i0, i1 + 1, dtype=np.float64)


def _normal_coordinate_distances(
    normal_coords,
    axis,
    *,
    ijk_to_ras,
    ras_to_ijk,
    plane_center,
    plane_normal,
):
    center_ijk = ras_to_ijk.MultiplyPoint(
        [float(plane_center[0]), float(plane_center[1]), float(plane_center[2]), 1.0]
    )[:3]
    coords = np.asarray(normal_coords, dtype=np.float64)
    i = np.full(coords.shape, float(center_ijk[0]), dtype=np.float64)
    j = np.full(coords.shape, float(center_ijk[1]), dtype=np.float64)
    k = np.full(coords.shape, float(center_ijk[2]), dtype=np.float64)
    if int(axis) == 0:
        i = coords
    elif int(axis) == 1:
        j = coords
    else:
        k = coords
    ras_x, ras_y, ras_z = _ras_arrays_from_ijk(ijk_to_ras, i, j, k)
    return (
        (ras_x - float(plane_center[0])) * float(plane_normal[0])
        + (ras_y - float(plane_center[1])) * float(plane_normal[1])
        + (ras_z - float(plane_center[2])) * float(plane_normal[2])
    )


def _axis_aligned_column_ijk_grids(axis, normal_coords, *, i0, i1, j0, j1, k0, k1):
    if int(axis) == 0:
        k_values = np.arange(k0, k1 + 1, dtype=np.float64)
        j_values = np.arange(j0, j1 + 1, dtype=np.float64)
        k_grid, j_grid = np.meshgrid(k_values, j_values, indexing="ij")
        return np.asarray(normal_coords, dtype=np.float64), j_grid, k_grid
    if int(axis) == 1:
        k_values = np.arange(k0, k1 + 1, dtype=np.float64)
        i_values = np.arange(i0, i1 + 1, dtype=np.float64)
        k_grid, i_grid = np.meshgrid(k_values, i_values, indexing="ij")
        return i_grid, np.asarray(normal_coords, dtype=np.float64), k_grid
    j_values = np.arange(j0, j1 + 1, dtype=np.float64)
    i_values = np.arange(i0, i1 + 1, dtype=np.float64)
    j_grid, i_grid = np.meshgrid(j_values, i_values, indexing="ij")
    return i_grid, j_grid, np.asarray(normal_coords, dtype=np.float64)


def _surface_map_shape(mask_shape, axis):
    if int(axis) == 0:
        return int(mask_shape[0]), int(mask_shape[1])
    if int(axis) == 1:
        return int(mask_shape[0]), int(mask_shape[2])
    return int(mask_shape[1]), int(mask_shape[2])


def _surface_map_slices_for_bounds(axis, i0, i1, j0, j1, k0, k1):
    if int(axis) == 0:
        return slice(k0, k1 + 1), slice(j0, j1 + 1)
    if int(axis) == 1:
        return slice(k0, k1 + 1), slice(i0, i1 + 1)
    return slice(j0, j1 + 1), slice(i0, i1 + 1)


def _axis_aligned_footprint_mask(
    axis,
    column_slices,
    *,
    ijk_to_ras,
    ras_to_ijk,
    plane_center,
    u_axis,
    v_axis,
    shape,
    half_u,
    half_v,
):
    center_ijk = ras_to_ijk.MultiplyPoint(
        [float(plane_center[0]), float(plane_center[1]), float(plane_center[2]), 1.0]
    )[:3]
    first = np.arange(column_slices[0].start, column_slices[0].stop, dtype=np.float64)
    second = np.arange(column_slices[1].start, column_slices[1].stop, dtype=np.float64)
    first_grid, second_grid = np.meshgrid(first, second, indexing="ij")
    if int(axis) == 0:
        i_grid = np.full(first_grid.shape, float(center_ijk[0]), dtype=np.float64)
        j_grid = second_grid
        k_grid = first_grid
    elif int(axis) == 1:
        i_grid = second_grid
        j_grid = np.full(first_grid.shape, float(center_ijk[1]), dtype=np.float64)
        k_grid = first_grid
    else:
        i_grid = second_grid
        j_grid = first_grid
        k_grid = np.full(first_grid.shape, float(center_ijk[2]), dtype=np.float64)
    u, v = _axis_aligned_column_uv(
        i_grid,
        j_grid,
        k_grid,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        u_axis=u_axis,
        v_axis=v_axis,
    )
    return _inside_shape_array(shape, u, v, half_u_mm=half_u, half_v_mm=half_v)


def _axis_aligned_column_uv(
    i_grid,
    j_grid,
    k_grid,
    *,
    ijk_to_ras,
    plane_center,
    u_axis,
    v_axis,
):
    ras_x, ras_y, ras_z = _ras_arrays_from_ijk(ijk_to_ras, i_grid, j_grid, k_grid)
    rel_x = ras_x - float(plane_center[0])
    rel_y = ras_y - float(plane_center[1])
    rel_z = ras_z - float(plane_center[2])
    u = rel_x * float(u_axis[0]) + rel_y * float(u_axis[1]) + rel_z * float(u_axis[2])
    v = rel_x * float(v_axis[0]) + rel_y * float(v_axis[1]) + rel_z * float(v_axis[2])
    return u, v


def _ras_arrays_from_ijk(ijk_to_ras, i, j, k):
    return (
        float(ijk_to_ras.GetElement(0, 0)) * i
        + float(ijk_to_ras.GetElement(0, 1)) * j
        + float(ijk_to_ras.GetElement(0, 2)) * k
        + float(ijk_to_ras.GetElement(0, 3)),
        float(ijk_to_ras.GetElement(1, 0)) * i
        + float(ijk_to_ras.GetElement(1, 1)) * j
        + float(ijk_to_ras.GetElement(1, 2)) * k
        + float(ijk_to_ras.GetElement(1, 3)),
        float(ijk_to_ras.GetElement(2, 0)) * i
        + float(ijk_to_ras.GetElement(2, 1)) * j
        + float(ijk_to_ras.GetElement(2, 2)) * k
        + float(ijk_to_ras.GetElement(2, 3)),
    )


def _local_projected_surface_index_map(
    mask,
    *,
    ijk_to_ras,
    plane_center,
    plane_normal,
    u_axis,
    v_axis,
    shape,
    half_u,
    half_v,
    lateral_margin_mm=0.0,
    min_distance=None,
    max_distance=None,
):
    if mask is None or mask.ndim != 3:
        return None
    ras_to_ijk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(ijk_to_ras, ras_to_ijk)
    distance_min = 0.0 if min_distance is None else float(min_distance)
    distance_max = _positive_volume_extent_along_normal(
        mask.shape,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        plane_normal=plane_normal,
    )
    if max_distance is not None:
        distance_max = min(distance_max, float(max_distance))
    if distance_max < distance_min:
        return None

    bounds = _projection_ijk_bounds(
        mask.shape,
        ras_to_ijk=ras_to_ijk,
        plane_center=plane_center,
        plane_normal=plane_normal,
        u_axis=u_axis,
        v_axis=v_axis,
        half_u=float(half_u) + float(lateral_margin_mm),
        half_v=float(half_v) + float(lateral_margin_mm),
        min_distance=distance_min,
        max_distance=distance_max,
    )
    if bounds is None:
        return None
    i0, i1, j0, j1, k0, k1 = bounds
    submask = np.asarray(mask[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1])
    foreground = np.argwhere(submask)
    if foreground.size == 0:
        return None

    axis = _dominant_ijk_axis(ijk_to_ras, plane_normal)
    if axis == 0:
        surface = np.full((mask.shape[0], mask.shape[1]), -1, dtype=np.int32)
    elif axis == 1:
        surface = np.full((mask.shape[0], mask.shape[2]), -1, dtype=np.int32)
    else:
        surface = np.full((mask.shape[1], mask.shape[2]), -1, dtype=np.int32)
    best_signed = np.full(surface.shape, np.inf, dtype=np.float64)
    for local_k, local_j, local_i in foreground:
        k = int(k0 + local_k)
        j = int(j0 + local_j)
        i = int(i0 + local_i)
        ras = ijk_to_ras.MultiplyPoint([i, j, k, 1.0])[:3]
        rel = _subtract(ras, plane_center)
        signed = _dot(rel, plane_normal)
        if signed < distance_min - 1.0e-6:
            continue
        if signed > distance_max + 1.0e-6:
            continue
        u = _dot(rel, u_axis)
        v = _dot(rel, v_axis)
        if not _inside_shape(
            shape,
            u,
            v,
            half_u_mm=float(half_u) + float(lateral_margin_mm),
            half_v_mm=float(half_v) + float(lateral_margin_mm),
        ):
            continue
        key = _surface_key(axis, i, j, k)
        if signed < best_signed[key]:
            best_signed[key] = signed
            surface[key] = int((i, j, k)[axis])
    return {"axis": axis, "surface": surface}


def _bounded_projection_search_depth(
    mask,
    *,
    ijk_to_ras,
    plane_center,
    plane_normal,
    u_axis,
    v_axis,
    shape,
    half_u,
    half_v,
    thickness_mm,
    intrusion_depth_mm,
    lateral_margin_mm,
    max_search_depth_mm,
    tolerance,
):
    min_depth = max(
        float(thickness_mm) + float(intrusion_depth_mm) + 2.0 * float(tolerance),
        float(tolerance),
    )
    extent = _positive_volume_extent_along_normal(
        mask.shape,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        plane_normal=plane_normal,
    )
    configured_max = float(max_search_depth_mm) if max_search_depth_mm is not None else 30.0
    max_depth = max(min_depth, min(float(configured_max), max(float(extent), min_depth)))
    hits = _probe_surface_distances(
        mask,
        ijk_to_ras=ijk_to_ras,
        plane_center=plane_center,
        plane_normal=plane_normal,
        u_axis=u_axis,
        v_axis=v_axis,
        shape=shape,
        half_u=half_u,
        half_v=half_v,
        lateral_margin_mm=lateral_margin_mm,
        max_distance=max_depth,
        step_mm=max(float(tolerance) * 0.5, 1.0e-3),
    )
    if len(hits) >= 5:
        hits_array = np.asarray(hits, dtype=np.float64)
        spread = float(np.percentile(hits_array, 75) - np.percentile(hits_array, 25))
        stable_spread = max(
            float(thickness_mm) + float(intrusion_depth_mm) + 2.0 * float(tolerance),
            3.0 * float(tolerance),
        )
        if spread <= stable_spread:
            estimated = (
                float(np.percentile(hits_array, 25))
                + float(thickness_mm)
                + float(intrusion_depth_mm)
                + 2.0 * float(tolerance)
            )
            return max(min_depth, min(max_depth, estimated))
    return max_depth


def _probe_surface_distances(
    mask,
    *,
    ijk_to_ras,
    plane_center,
    plane_normal,
    u_axis,
    v_axis,
    shape,
    half_u,
    half_v,
    lateral_margin_mm,
    max_distance,
    step_mm,
):
    ras_to_ijk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(ijk_to_ras, ras_to_ijk)
    hits = []
    for u, v in _projection_probe_offsets(
        shape,
        half_u=float(half_u),
        half_v=float(half_v),
        inset_mm=float(lateral_margin_mm),
    ):
        probe = [
            float(plane_center[index])
            + float(u_axis[index]) * float(u)
            + float(v_axis[index]) * float(v)
            for index in range(3)
        ]
        distance = 0.0
        while distance <= float(max_distance) + 1.0e-6:
            ras = [
                probe[index] + float(plane_normal[index]) * distance
                for index in range(3)
            ]
            ijk = ras_to_ijk.MultiplyPoint([ras[0], ras[1], ras[2], 1.0])[:3]
            i, j, k = (int(round(float(value))) for value in ijk)
            if (
                0 <= k < mask.shape[0]
                and 0 <= j < mask.shape[1]
                and 0 <= i < mask.shape[2]
                and bool(mask[k, j, i])
            ):
                hits.append(float(distance))
                break
            distance += float(step_mm)
    return hits


def _projection_probe_offsets(shape, *, half_u, half_v, inset_mm):
    half_u = max(float(half_u) - float(inset_mm), 0.0)
    half_v = max(float(half_v) - float(inset_mm), 0.0)
    raw_offsets = [(0.0, 0.0)]
    raw_offsets.extend(
        [
            (-0.65 * half_u, 0.0),
            (0.65 * half_u, 0.0),
            (0.0, -0.65 * half_v),
            (0.0, 0.65 * half_v),
        ]
    )
    for u_sign in (-1.0, 1.0):
        for v_sign in (-1.0, 1.0):
            raw_offsets.append((0.45 * u_sign * half_u, 0.45 * v_sign * half_v))
    for u_sign in (-1.0, 1.0):
        for v_sign in (-1.0, 1.0):
            raw_offsets.append((0.85 * u_sign * half_u, 0.85 * v_sign * half_v))
    return [
        (u, v)
        for u, v in raw_offsets
        if _inside_shape(shape, u, v, half_u_mm=max(half_u, 1.0e-9), half_v_mm=max(half_v, 1.0e-9))
    ]


def _projection_ijk_bounds(
    mask_shape,
    *,
    ras_to_ijk,
    plane_center,
    plane_normal,
    u_axis,
    v_axis,
    half_u,
    half_v,
    min_distance,
    max_distance,
):
    raw_bounds = _projection_ijk_bounds_unclamped(
        ras_to_ijk=ras_to_ijk,
        plane_center=plane_center,
        plane_normal=plane_normal,
        u_axis=u_axis,
        v_axis=v_axis,
        half_u=half_u,
        half_v=half_v,
        min_distance=min_distance,
        max_distance=max_distance,
    )
    if raw_bounds is None:
        return None
    i0, i1, j0, j1, k0, k1 = raw_bounds
    i0 = max(0, i0)
    i1 = min(int(mask_shape[2]) - 1, i1)
    j0 = max(0, j0)
    j1 = min(int(mask_shape[1]) - 1, j1)
    k0 = max(0, k0)
    k1 = min(int(mask_shape[0]) - 1, k1)
    if i1 < i0 or j1 < j0 or k1 < k0:
        return None
    return i0, i1, j0, j1, k0, k1


def _projection_ijk_bounds_unclamped(
    *,
    ras_to_ijk,
    plane_center,
    plane_normal,
    u_axis,
    v_axis,
    half_u,
    half_v,
    min_distance,
    max_distance,
):
    corners = []
    for distance in (float(min_distance), float(max_distance)):
        for u_sign in (-1.0, 1.0):
            for v_sign in (-1.0, 1.0):
                ras = [
                    float(plane_center[index])
                    + float(plane_normal[index]) * distance
                    + float(u_axis[index]) * u_sign * float(half_u)
                    + float(v_axis[index]) * v_sign * float(half_v)
                    for index in range(3)
                ]
                corners.append(ras_to_ijk.MultiplyPoint([ras[0], ras[1], ras[2], 1.0])[:3])
    if not corners:
        return None
    ijk = np.asarray(corners, dtype=np.float64)
    i0 = int(math.floor(float(np.min(ijk[:, 0])))) - 2
    i1 = int(math.ceil(float(np.max(ijk[:, 0])))) + 2
    j0 = int(math.floor(float(np.min(ijk[:, 1])))) - 2
    j1 = int(math.ceil(float(np.max(ijk[:, 1])))) + 2
    k0 = int(math.floor(float(np.min(ijk[:, 2])))) - 2
    k1 = int(math.ceil(float(np.max(ijk[:, 2])))) + 2
    return i0, i1, j0, j1, k0, k1


def _positive_volume_extent_along_normal(mask_shape, *, ijk_to_ras, plane_center, plane_normal):
    max_distance = 0.0
    for k in (0, int(mask_shape[0]) - 1):
        for j in (0, int(mask_shape[1]) - 1):
            for i in (0, int(mask_shape[2]) - 1):
                ras = ijk_to_ras.MultiplyPoint([i, j, k, 1.0])[:3]
                distance = _dot(_subtract(ras, plane_center), plane_normal)
                max_distance = max(max_distance, float(distance))
    return max_distance


def _surface_index_distances(surface_index, *, ijk_to_ras, plane_center, plane_normal):
    if surface_index is None:
        return []
    distances = []
    for i, j, k in _surface_map_indices(surface_index):
        ras = ijk_to_ras.MultiplyPoint([int(i), int(j), int(k), 1.0])[:3]
        distances.append(float(_dot(_subtract(ras, plane_center), plane_normal)))
    return distances


def _robust_projected_surface_distance(distances):
    values = np.asarray(list(distances), dtype=np.float64)
    if values.size == 0:
        return None
    if values.size >= 20:
        return float(np.percentile(values, 5))
    return float(np.min(values))


def _filter_surface_index_by_distance(
    surface_index,
    *,
    ijk_to_ras,
    plane_center,
    plane_normal,
    min_distance,
    max_distance,
):
    if surface_index is None:
        return None
    filtered = {
        "axis": int(surface_index["axis"]),
        "surface": np.array(surface_index["surface"], copy=True),
    }
    for first in range(filtered["surface"].shape[0]):
        for second in range(filtered["surface"].shape[1]):
            value = int(filtered["surface"][first, second])
            if value < 0:
                continue
            axis = int(filtered["axis"])
            if axis == 0:
                i, j, k = value, second, first
            elif axis == 1:
                i, j, k = second, value, first
            else:
                i, j, k = second, first, value
            ras = ijk_to_ras.MultiplyPoint([int(i), int(j), int(k), 1.0])[:3]
            distance = _dot(_subtract(ras, plane_center), plane_normal)
            if distance < float(min_distance) or distance > float(max_distance):
                filtered["surface"][first, second] = -1
    if not np.any(filtered["surface"] >= 0):
        return None
    return filtered


def _disk_distance_range(
    cap_mode,
    *,
    first_surface_distance,
    opposite_surface_distance,
    cap_outer_distance,
    cap_inner_distance,
    thickness_mm,
    intrusion_depth_mm,
    tolerance,
):
    mode = str(cap_mode).strip().lower()
    if mode == "projected_cap":
        return (
            float(cap_outer_distance) - float(tolerance),
            float(cap_inner_distance) + float(tolerance),
        )
    if mode == "connective_disk" and first_surface_distance is not None and opposite_surface_distance is not None:
        lower_distance = -float(opposite_surface_distance) - float(intrusion_depth_mm)
        upper_distance = float(first_surface_distance) + float(intrusion_depth_mm)
        return (
            min(lower_distance, upper_distance) - float(tolerance),
            max(lower_distance, upper_distance) + float(tolerance),
        )
    half = float(thickness_mm) / 2.0
    return -half - float(tolerance), half + float(tolerance)


def _required_projected_contact_padding_ijk(
    volume_node,
    target_node,
    rows,
    *,
    target_values=None,
):
    image = volume_node.GetImageData()
    if image is None:
        return np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    dims = image.GetDimensions()
    target_mask = _target_mask_array(
        target_node or volume_node,
        volume_node,
        active_values=target_values,
        fallback_to_nonzero=bool(target_values),
    )
    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)
    ras_to_ijk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(ijk_to_ras, ras_to_ijk)
    tolerance = _voxel_tolerance(volume_node)
    pad_before = np.zeros(3, dtype=np.int64)
    pad_after = np.zeros(3, dtype=np.int64)

    for row in rows or ():
        if not _row_needs_projected_contact_padding(row):
            continue
        plane = row.get("plane")
        if not _looks_like_plane_node(plane):
            continue
        shape = row.get("shape", "anatomy")
        radius = float(row.get("radius", 12.0))
        thickness = float(row.get("thickness", row.get("thickness_mm", 3.0)))
        intrusion = float(row.get("intrusion", row.get("intrusion_depth_mm", 2.0)))
        center, normal, u_axis, v_axis, half_u, half_v = _plane_geometry(
            plane,
            shape=shape,
            radius_mm=radius,
            square_width_mm=radius * 2.0,
            hex_radius_mm=radius,
            use_plane_size=bool(row.get("use_plane_size", True)),
        )
        mode = _projection_mode(row.get("surface_mode", "project"))
        if mode == "project_global":
            first_surface_distance = _first_projected_surface_distance(
                target_mask,
                ijk_to_ras=ijk_to_ras,
                plane_center=center,
                plane_normal=normal,
                u_axis=u_axis,
                v_axis=v_axis,
                shape=shape,
                half_u=half_u,
                half_v=half_v,
            )
            opposite_surface_distance = _first_projected_surface_distance(
                target_mask,
                ijk_to_ras=ijk_to_ras,
                plane_center=center,
                plane_normal=tuple(-float(value) for value in normal),
                u_axis=u_axis,
                v_axis=v_axis,
                shape=shape,
                half_u=half_u,
                half_v=half_v,
            )
        else:
            first_surface_distance = None
            opposite_surface_distance = None
            axis = _axis_aligned_ijk_axis(ijk_to_ras, normal)
            if axis is not None:
                first_surface_distance, _surface_map = _axis_aligned_projected_surface_distance_map(
                    target_mask,
                    ijk_to_ras=ijk_to_ras,
                    ras_to_ijk=ras_to_ijk,
                    plane_center=center,
                    plane_normal=normal,
                    u_axis=u_axis,
                    v_axis=v_axis,
                    shape=shape,
                    half_u=half_u,
                    half_v=half_v,
                    thickness_mm=thickness,
                    intrusion_depth_mm=intrusion,
                    lateral_margin_mm=tolerance,
                    tolerance=tolerance,
                    axis=axis,
                )
                opposite_surface_distance, _opposite_map = _axis_aligned_projected_surface_distance_map(
                    target_mask,
                    ijk_to_ras=ijk_to_ras,
                    ras_to_ijk=ras_to_ijk,
                    plane_center=center,
                    plane_normal=tuple(-float(value) for value in normal),
                    u_axis=u_axis,
                    v_axis=v_axis,
                    shape=shape,
                    half_u=half_u,
                    half_v=half_v,
                    thickness_mm=thickness,
                    intrusion_depth_mm=intrusion,
                    lateral_margin_mm=tolerance,
                    tolerance=tolerance,
                    axis=axis,
                )
            if axis is None or (first_surface_distance is None and opposite_surface_distance is None):
                first_surface_distance, _surface = _bounded_projected_surface_context(
                    target_mask,
                    ijk_to_ras=ijk_to_ras,
                    plane_center=center,
                    plane_normal=normal,
                    u_axis=u_axis,
                    v_axis=v_axis,
                    shape=shape,
                    half_u=half_u,
                    half_v=half_v,
                    thickness_mm=thickness,
                    intrusion_depth_mm=intrusion,
                    volume_node=volume_node,
                )
                opposite_surface_distance, _opposite = _bounded_projected_surface_context(
                    target_mask,
                    ijk_to_ras=ijk_to_ras,
                    plane_center=center,
                    plane_normal=tuple(-float(value) for value in normal),
                    u_axis=u_axis,
                    v_axis=v_axis,
                    shape=shape,
                    half_u=half_u,
                    half_v=half_v,
                    thickness_mm=thickness,
                    intrusion_depth_mm=intrusion,
                    volume_node=volume_node,
                )

        cap_outer_distance, cap_inner_distance = _projected_cap_distance_range(
            first_surface_distance,
            thickness_mm=thickness,
            intrusion_depth_mm=intrusion,
        )
        distance_min, distance_max = _disk_distance_range(
            "connective_disk" if str(row.get("contact", "")).strip().lower() == "connective disk" else "projected_cap",
            first_surface_distance=first_surface_distance,
            opposite_surface_distance=opposite_surface_distance,
            cap_outer_distance=cap_outer_distance,
            cap_inner_distance=cap_inner_distance,
            thickness_mm=thickness,
            intrusion_depth_mm=intrusion,
            tolerance=tolerance,
        )
        bounds = _projection_ijk_bounds_unclamped(
            ras_to_ijk=ras_to_ijk,
            plane_center=center,
            plane_normal=normal,
            u_axis=u_axis,
            v_axis=v_axis,
            half_u=half_u + tolerance,
            half_v=half_v + tolerance,
            min_distance=distance_min,
            max_distance=distance_max,
        )
        if bounds is None:
            continue
        i0, i1, j0, j1, k0, k1 = bounds
        pad_before = np.maximum(
            pad_before,
            np.asarray(
                [max(0, -int(i0)), max(0, -int(j0)), max(0, -int(k0))],
                dtype=np.int64,
            ),
        )
        pad_after = np.maximum(
            pad_after,
            np.asarray(
                [
                    max(0, int(i1) - (int(dims[0]) - 1)),
                    max(0, int(j1) - (int(dims[1]) - 1)),
                    max(0, int(k1) - (int(dims[2]) - 1)),
                ],
                dtype=np.int64,
            ),
        )
    return pad_before, pad_after


def _row_needs_projected_contact_padding(row):
    contact = str(row.get("contact", "")).strip().lower()
    if contact not in {"material disks", "connective disk", "pmma caps"}:
        return False
    return _projection_mode(row.get("surface_mode", "project")) != "intersect"


def _pad_volume_node(
    source_node,
    name,
    pad_before_ijk,
    pad_after_ijk,
    *,
    label=False,
    source_array=None,
):
    array = (
        np.asarray(source_array)
        if source_array is not None
        else np.asarray(slicer.util.arrayFromVolume(source_node))
    )
    before = np.asarray(pad_before_ijk, dtype=np.int64)
    after = np.asarray(pad_after_ijk, dtype=np.int64)
    pad_width = (
        (int(before[2]), int(after[2])),
        (int(before[1]), int(after[1])),
        (int(before[0]), int(after[0])),
    )
    padded = np.pad(array, pad_width, mode="constant", constant_values=0)
    node_class = (
        "vtkMRMLLabelMapVolumeNode"
        if bool(label) or source_node.IsA("vtkMRMLLabelMapVolumeNode")
        else "vtkMRMLScalarVolumeNode"
    )
    node = slicer.mrmlScene.AddNewNodeByClass(node_class, name)
    slicer.util.updateVolumeFromArray(node, padded)
    _copy_padded_geometry(source_node, node, before)
    node.CreateDefaultDisplayNodes()
    node.Modified()
    return node


def _copy_padded_geometry(source_node, padded_node, pad_before_ijk):
    before = np.asarray(pad_before_ijk, dtype=np.int64)
    _copy_cropped_geometry(
        source_node,
        padded_node,
        (-int(before[0]), -int(before[1]), -int(before[2])),
    )


def _projection_mode(value):
    mode = str(value or "project").strip().lower().replace("-", "_")
    if mode in {"project", "bounded", "project_bounded", "bounded_project"}:
        return "project_bounded"
    if mode in {"project_global", "global", "legacy_project", "legacy_global"}:
        return "project_global"
    if mode == "intersect":
        return "intersect"
    return "project_bounded"


def _surface_mode_ui_text(value):
    return "intersect" if _projection_mode(value) == "intersect" else "project"


def _dominant_ijk_axis(ijk_to_ras, plane_normal):
    ras_to_ijk_direction = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(ijk_to_ras, ras_to_ijk_direction)
    direction = ras_to_ijk_direction.MultiplyPoint(
        [float(plane_normal[0]), float(plane_normal[1]), float(plane_normal[2]), 0.0]
    )[:3]
    return int(np.argmax(np.abs(direction)))


def _surface_key(axis, i, j, k):
    if axis == 0:
        return int(k), int(j)
    if axis == 1:
        return int(k), int(i)
    return int(j), int(i)


def _surface_ijk_for_voxel(surface_index, i, j, k):
    if surface_index is None:
        return None
    axis = int(surface_index["axis"])
    surface = surface_index["surface"]
    key = _surface_key(axis, i, j, k)
    if key[0] < 0 or key[1] < 0 or key[0] >= surface.shape[0] or key[1] >= surface.shape[1]:
        return None
    value = int(surface[key])
    if value < 0:
        return None
    if axis == 0:
        return value, int(j), int(k)
    if axis == 1:
        return int(i), value, int(k)
    return int(i), int(j), value


def _closest_surface_distance(surface_index, *, ijk_to_ras, plane_center, plane_normal):
    if surface_index is None:
        return None
    closest = None
    for i, j, k in _surface_map_indices(surface_index):
        ras = ijk_to_ras.MultiplyPoint([int(i), int(j), int(k), 1.0])[:3]
        distance = _dot(_subtract(ras, plane_center), plane_normal)
        if closest is None or distance < closest:
            closest = distance
    return closest


def _first_projected_surface_distance(
    mask,
    *,
    ijk_to_ras,
    plane_center,
    plane_normal,
    u_axis,
    v_axis,
    shape,
    half_u,
    half_v,
):
    if mask is None or mask.ndim != 3:
        return None
    closest = None
    for k, j, i in np.argwhere(mask):
        ras = ijk_to_ras.MultiplyPoint([int(i), int(j), int(k), 1.0])[:3]
        rel = _subtract(ras, plane_center)
        distance = _dot(rel, plane_normal)
        if distance < -1e-6:
            continue
        u = _dot(rel, u_axis)
        v = _dot(rel, v_axis)
        if not _inside_shape(shape, u, v, half_u_mm=half_u, half_v_mm=half_v):
            continue
        if closest is None or distance < closest:
            closest = distance
    return closest


def _surface_map_indices(surface_index):
    if surface_index is None:
        return
    axis = int(surface_index["axis"])
    surface = surface_index["surface"]
    for first in range(surface.shape[0]):
        for second in range(surface.shape[1]):
            value = int(surface[first, second])
            if value < 0:
                continue
            if axis == 0:
                yield value, second, first
            elif axis == 1:
                yield second, value, first
            else:
                yield second, first, value


def _mask_value(mask, k, j, i):
    if mask is None:
        return False
    if (
        k < 0
        or j < 0
        or i < 0
        or k >= mask.shape[0]
        or j >= mask.shape[1]
        or i >= mask.shape[2]
    ):
        return False
    return bool(mask[k, j, i])


def _parse_vector(text):
    parts = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Load direction vector must contain three comma-separated values")
    return tuple(parts)


def _nested_get(mapping, path, default=None):
    value = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _first_present(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _dict_value(mapping, key):
    return mapping.get(key) if isinstance(mapping, dict) else None


def _nonlinear_result_values(data):
    if not isinstance(data, dict):
        return []
    nonlinear = data.get("nonlinear", {}) if isinstance(data.get("nonlinear"), dict) else {}
    solver = data.get("solver", {}) if isinstance(data.get("solver"), dict) else {}
    solver_nonlinear = (
        solver.get("nonlinear", {}) if isinstance(solver.get("nonlinear"), dict) else {}
    )
    materials = data.get("materials", {}) if isinstance(data.get("materials"), dict) else {}
    material_nonlinear = (
        materials.get("nonlinear", {}) if isinstance(materials.get("nonlinear"), dict) else {}
    )
    if not nonlinear and not solver_nonlinear and not material_nonlinear:
        return []
    return [
        (
            "Material",
            _first_present(
                _dict_value(nonlinear, "material"),
                _dict_value(nonlinear, "material_type"),
                _dict_value(material_nonlinear, "material"),
                _dict_value(material_nonlinear, "type"),
                "nonlinear density",
            ),
        ),
        (
            "Preset",
            _first_present(_dict_value(nonlinear, "preset"), _dict_value(material_nonlinear, "preset")),
        ),
        (
            "Plastic iterations",
            _format_number(
                _first_present(
                    _dict_value(nonlinear, "plastic_iterations"),
                    _dict_value(solver_nonlinear, "plastic_iterations"),
                )
            ),
        ),
        (
            "Yielded elements",
            _format_number(
                _first_present(
                    _dict_value(nonlinear, "yielded_last"),
                    _dict_value(nonlinear, "yielded_elements"),
                    _dict_value(nonlinear, "yielded_count"),
                    _dict_value(solver_nonlinear, "yielded_last"),
                    _dict_value(solver_nonlinear, "yielded_elements"),
                )
            ),
        ),
        (
            "Plastic convergence",
            _format_number(
                _first_present(
                    _dict_value(nonlinear, "plastic_convergence_last"),
                    _dict_value(nonlinear, "plastic_change"),
                    _dict_value(solver_nonlinear, "plastic_convergence_last"),
                    _dict_value(solver_nonlinear, "plastic_change"),
                )
            ),
        ),
        (
            "Convergence tolerance",
            _format_number(
                _first_present(
                    _dict_value(nonlinear, "plastic_tolerance"),
                    _dict_value(nonlinear, "convergence_tolerance"),
                    _dict_value(solver_nonlinear, "plastic_tolerance"),
                    _dict_value(solver_nonlinear, "convergence_tolerance"),
                )
            ),
        ),
        (
            "Max plastic iterations",
            _format_number(
                _first_present(
                    _dict_value(nonlinear, "max_plastic_iterations"),
                    _dict_value(nonlinear, "maximum_plastic_iterations"),
                    _dict_value(solver_nonlinear, "max_plastic_iterations"),
                    _dict_value(solver_nonlinear, "maximum_plastic_iterations"),
                )
            ),
        ),
        ("Status", _first_present(_dict_value(nonlinear, "status"), _dict_value(solver_nonlinear, "status"))),
    ]


def _format_number(value):
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.3g}"
    if abs(number) >= 1:
        return f"{number:.4g}"
    return f"{number:.4g}"


def _format_scientific_number(value):
    try:
        mantissa_text, exponent_text = f"{float(value):.6e}".split("e")
    except Exception:
        return str(value)
    mantissa_text = mantissa_text.rstrip("0").rstrip(".")
    exponent = int(exponent_text)
    return f"{mantissa_text}e{exponent}"


def _nice_spacing_default_mm(value):
    value = abs(float(value))
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    if value < 0.1:
        return round(value, 3)
    if value < 1.0:
        return round(value, 2)
    return round(value, 1)


def _format_xyz(values, units):
    if not isinstance(values, dict):
        return None
    parts = []
    for axis in ("x", "y", "z"):
        value = values.get(axis)
        if value is not None:
            parts.append(f"{axis}={_format_number(value)}")
    if not parts:
        return None
    return ", ".join(parts) + f" {units}"


def _format_generalized(values):
    if not isinstance(values, dict):
        return None
    value = values.get("value")
    if value is None:
        return None
    units = values.get("units", "")
    name = values.get("name") or values.get("component")
    text = f"{_format_number(value)}"
    if units:
        text += f" {units}"
    if name:
        text += f" ({name})"
    return text


def _format_load_history_html(load_history):
    results = load_history.get("results", {}) if isinstance(load_history.get("results"), dict) else {}
    details = load_history.get("details", {}) if isinstance(load_history.get("details"), dict) else {}
    estimated = results.get("estimated_loads", [])
    force_vector = _sum_load_vectors(estimated, load_type="force", units="N")
    moment_vector = _sum_load_vectors(estimated, load_type="moment", units="N*mm")
    rows = [
        ("Method", load_history.get("method") or "nnls"),
        ("Cases", ", ".join(str(item) for item in load_history.get("cases", [])) if load_history.get("cases") else None),
        ("Force resultant", _format_resultant_vector(force_vector, "N")),
        ("Moment resultant", _format_resultant_vector(moment_vector, "N*mm")),
        ("Fit residual", _format_number(details.get("residual"))),
        ("Estimated SED", load_history.get("output")),
    ]
    individual_rows = []
    if isinstance(estimated, list):
        for entry in estimated:
            if not isinstance(entry, dict):
                continue
            vector_text = _format_xyz(entry.get("vector"), entry.get("units", ""))
            active = _load_history_entry_active(entry)
            individual_rows.append(
                "<tr>"
                f"<td>{html.escape(str(entry.get('case', '')))}</td>"
                f"<td>{html.escape(str(entry.get('load_type', '')))}</td>"
                f"<td>{html.escape(_format_number(entry.get('value')))}</td>"
                f"<td>{html.escape(str(entry.get('units') or ''))}</td>"
                f"<td>{'yes' if active else 'no'}</td>"
                f"<td>{html.escape(vector_text or '')}</td>"
                "</tr>"
            )
    individual_html = ""
    if individual_rows:
        individual_html = (
            "<h4>Estimated Load Contributions</h4>"
            "<table><tr><th>Case</th><th>Type</th><th>Contribution</th><th>Units</th><th>Active</th><th>Vector</th></tr>"
            f"{''.join(individual_rows)}</table>"
        )
    return (
        "<h3>Load Estimation</h3>"
        f"<table>{_html_table_rows(rows)}</table>"
        f"{individual_html}"
    )


def _load_history_entry_active(entry):
    vector = entry.get("vector") if isinstance(entry, dict) else None
    if isinstance(vector, dict):
        magnitude = math.sqrt(
            sum(float(vector.get(axis, 0.0) or 0.0) ** 2 for axis in ("x", "y", "z"))
        )
        return math.isfinite(magnitude) and magnitude > 1e-9
    try:
        value = abs(float(entry.get("value", 0.0)))
    except Exception:
        return False
    return math.isfinite(value) and value > 1e-9


def _sum_load_vectors(entries, *, load_type, units):
    if not isinstance(entries, list):
        return None
    total = {"x": 0.0, "y": 0.0, "z": 0.0}
    found = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("load_type", "")).strip().lower() != load_type:
            continue
        if str(entry.get("units", "")).strip() != units:
            continue
        vector = entry.get("vector")
        if not isinstance(vector, dict):
            continue
        for axis in ("x", "y", "z"):
            value = vector.get(axis)
            if value is not None:
                total[axis] += float(value)
                found = True
    return total if found else None


def _format_resultant_vector(vector, units):
    if not isinstance(vector, dict):
        return None
    values = [float(vector.get(axis, 0.0) or 0.0) for axis in ("x", "y", "z")]
    magnitude = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        return None
    axis_text = ", ".join(
        f"{axis}={_format_number(value / magnitude)}"
        for axis, value in zip(("x", "y", "z"), values)
    )
    return f"{_format_xyz(vector, units)}; magnitude={_format_number(magnitude)} {units}; axis=({axis_text})"


def _format_sequence(values, suffix=""):
    if not isinstance(values, (list, tuple)):
        return None
    text = ", ".join(_format_number(value) for value in values)
    return text + suffix


def _format_load_case(values):
    if not isinstance(values, dict):
        return None
    parts = []
    load_type = str(values.get("type", "")).strip().lower()
    keys = ["type", "axis", "force", "moment", "rotation_degrees"]
    if load_type not in {"nodeset", "custom"}:
        keys.insert(2, "strain")
    for key in keys:
        value = values.get(key)
        if value is not None:
            parts.append(f"{key}={_format_number(value)}")
    return ", ".join(parts) if parts else None


def _result_field_display_name(field):
    names = {
        "sed": "SED",
        "load_history_estimated_sed": "Load-history estimated SED",
        "load_history_final_sed": "Load-history final SED",
        "effective_strain": "Effective strain",
        "von_mises": "Von Mises",
        "strain": "Strain tensor",
        "stress": "Stress tensor",
        "plastic_strain": "Plastic strain tensor",
        "plastic_strain_magnitude": "Plastic strain magnitude",
        "plastic_dissipation": "Plastic dissipation",
        "mechanical_work_density": "Mechanical work density",
    }
    return names.get(str(field), str(field).replace("_", " ").title())


def _result_field_node_name(field):
    names = {
        "sed": "ParOSol_SED",
        "load_history_estimated_sed": "ParOSol_LoadHistory_Estimated_SED",
        "load_history_final_sed": "ParOSol_LoadHistory_Final_SED",
        "effective_strain": "ParOSol_Effective_strain",
        "von_mises": "ParOSol_Von_Mises",
        "strain": "ParOSol_Strain_tensor",
        "stress": "ParOSol_Stress_tensor",
        "plastic_strain": "ParOSol_Plastic_strain_tensor",
        "plastic_strain_magnitude": "ParOSol_Plastic_strain_magnitude",
        "plastic_dissipation": "ParOSol_Plastic_dissipation",
        "mechanical_work_density": "ParOSol_Mechanical_work_density",
    }
    return names.get(str(field), f"ParOSol_{_result_field_display_name(field).replace(' ', '_')}")


def _known_result_fields():
    return (
        "sed",
        "load_history_estimated_sed",
        "load_history_final_sed",
        "effective_strain",
        "von_mises",
        "strain",
        "stress",
        "plastic_strain",
        "plastic_strain_magnitude",
        "plastic_dissipation",
        "mechanical_work_density",
    )


def _scalar_result_fields():
    return (
        "sed",
        "load_history_estimated_sed",
        "load_history_final_sed",
        "effective_strain",
        "von_mises",
        "plastic_strain_magnitude",
        "plastic_dissipation",
        "mechanical_work_density",
    )


def _remove_stale_result_fields(output_dir):
    fields_dir = Path(output_dir) / "fields"
    if not fields_dir.is_dir():
        return 0
    removed = 0
    for path in fields_dir.glob("*.nii.gz"):
        try:
            path.unlink()
        except OSError:
            continue
        removed += 1
    return removed


def _result_exported_field_names(output_dir):
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    if not result_path.exists():
        return set()
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    exported = data.get("outputs", {}).get("exported", {})
    if not isinstance(exported, dict):
        return set()
    names = set()
    for value in exported.values():
        path = Path(str(value))
        if path.name.endswith(".nii.gz") and "fields" in path.parts:
            names.add(path.name)
    return names


def _result_fields_from_filenames(field_names):
    fields = []
    for name in sorted(str(item) for item in field_names):
        if not name.endswith(".nii.gz"):
            continue
        field = name[:-7]
        if field.startswith("displacement_"):
            field = "displacements"
        if field not in _scalar_result_fields() and field != "displacements":
            continue
        if field not in fields:
            fields.append(field)
    preferred_order = {
        "sed": 0,
        "load_history_estimated_sed": 1,
        "load_history_final_sed": 2,
        "effective_strain": 3,
        "von_mises": 4,
        "plastic_strain_magnitude": 8,
        "plastic_dissipation": 9,
        "mechanical_work_density": 10,
        "displacements": 11,
    }
    return sorted(fields, key=lambda field: (preferred_order.get(field, 99), str(field)))


def _result_save_manifest(output_dir, case_prefix):
    output_dir = Path(output_dir)
    prefix = _safe_result_prefix(case_prefix)
    manifest = []
    fixed_files = (
        ("result.json", "result.json"),
        ("summary.json", "summary.json"),
        ("parosol_slicer_case.yaml", "parosol_slicer_case.yaml"),
        ("slicer_input.nii.gz", "input.nii.gz"),
        ("slicer_mask.nii.gz", "input_mask.nii.gz"),
        ("disk_labels.nii.gz", "disk_labels.nii.gz"),
        ("nodesets.nii.gz", "nodesets.nii.gz"),
        ("parosol_input.h5", "parosol_input.h5"),
        ("overview.png", "overview.png"),
        ("result.csv", "result.csv"),
    )
    used_destinations = set()

    def add(source, destination_suffix):
        source = Path(source)
        if not source.exists() or not source.is_file():
            return
        destination_name = f"{prefix}_{destination_suffix}"
        if destination_name in used_destinations:
            return
        used_destinations.add(destination_name)
        manifest.append((source, destination_name))

    for relative, destination_suffix in fixed_files:
        add(output_dir / relative, destination_suffix)

    fields_dir = output_dir / "fields"
    if fields_dir.is_dir():
        requested_field_names = _result_requested_field_names(output_dir)
        for field_name in sorted(requested_field_names):
            add(fields_dir / field_name, field_name)
    return manifest


def _result_requested_field_names(output_dir):
    output_dir = Path(output_dir)
    names = set()
    result_path = output_dir / "result.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            exported = data.get("outputs", {}).get("exported", {})
            if isinstance(exported, dict):
                for value in exported.values():
                    path = Path(str(value))
                    if path.name.endswith(".nii.gz") and "fields" in path.parts:
                        names.add(path.name)
        except Exception:
            pass
    config_path = output_dir / "parosol_slicer_case.yaml"
    if config_path.exists():
        try:
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            requested = []
            if isinstance(config, dict):
                output = config.get("output", {})
                solver = config.get("solver", {})
                postprocess = config.get("postprocess", {})
                if isinstance(output, dict) and isinstance(output.get("fields"), list):
                    requested.extend(output.get("fields", []))
                if isinstance(solver, dict) and isinstance(solver.get("outputs"), list):
                    requested.extend(solver.get("outputs", []))
                load_history = postprocess.get("load_history", {}) if isinstance(postprocess, dict) else {}
                if isinstance(load_history, dict):
                    for key in ("output",):
                        value = load_history.get(key)
                        if value:
                            requested.append(value)
                    final_rerun = load_history.get("final_rerun", {})
                    if isinstance(final_rerun, dict) and final_rerun.get("output"):
                        requested.append(final_rerun.get("output"))
            for field in requested:
                field = str(field).strip()
                if not field:
                    continue
                if field == "displacements":
                    names.update(f"displacement_{axis}.nii.gz" for axis in ("x", "y", "z"))
                elif field.endswith(".nii.gz"):
                    names.add(Path(field).name)
                else:
                    names.add(f"{field}.nii.gz")
        except Exception:
            pass
    if not names and (output_dir / "fields" / "sed.nii.gz").exists():
        names.add("sed.nii.gz")
    return names


def _copy_result_save_manifest(manifest, target_dir):
    target_dir = Path(target_dir)
    copied = []
    for source, destination_name in manifest:
        source = Path(source)
        destination = target_dir / str(destination_name)
        if source.resolve() == destination.resolve():
            continue
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _safe_result_prefix(value):
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "parosol_result")).strip("._")
    return prefix or "parosol_result"


def _html_table_rows(rows):
    return "\n".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
        if value not in (None, "")
    )


def _workflow_template_type(config):
    if not isinstance(config, dict):
        return ""
    template = config.get("workflow_template", {})
    if not isinstance(template, dict):
        return ""
    return str(template.get("type", "")).strip().lower()


def _resolve_workflow_relative_paths(config, base_dir):
    config = copy.deepcopy(config)
    base_dir = Path(base_dir)

    def resolve(path_text):
        if not path_text:
            return path_text
        path = Path(str(path_text)).expanduser()
        if path.is_absolute():
            return str(path)
        return str((base_dir / path).resolve())

    input_cfg = config.get("input", {})
    if isinstance(input_cfg, dict):
        for key in ("image", "mask"):
            if key in input_cfg:
                input_cfg[key] = resolve(input_cfg[key])
    nodesets = config.get("nodesets", {})
    if isinstance(nodesets, dict):
        for spec in nodesets.values():
            if isinstance(spec, dict) and "image" in spec:
                spec["image"] = resolve(spec["image"])
    model = config.get("model", {})
    if isinstance(model, dict):
        for key in ("density_image", "mask_image"):
            if key in model:
                model[key] = resolve(model[key])
        registration = model.get("registration", {})
        if isinstance(registration, dict) and "reference_points" in registration:
            registration["reference_points"] = resolve(registration["reference_points"])
        replay = model.get("workflow_replay", {})
        if isinstance(replay, dict):
            for key in ("reference_points", "editor_reference_points", "disk_labels", "nodesets"):
                if key in replay:
                    replay[key] = resolve(replay[key])
    custom_preprocessing = config.get("custom_preprocessing", {})
    if isinstance(custom_preprocessing, dict):
        if "script" in custom_preprocessing:
            custom_preprocessing["script"] = resolve(custom_preprocessing["script"])
        for option in _custom_preprocessing_options(custom_preprocessing):
            if "script" in option:
                option["script"] = resolve(option["script"])
    return config


def _config_file_has_batch(path):
    try:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(data.get("batch"), dict)


def _config_file_batch_summary(path):
    path = Path(path)
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return path.parent / "result.json"
    if not isinstance(data, dict) or not isinstance(data.get("batch"), dict):
        return path.parent / "result.json"
    summary = data["batch"].get("summary") or "result.json"
    summary_path = Path(str(summary)).expanduser()
    if not summary_path.is_absolute():
        summary_path = path.parent / summary_path
    return summary_path


def _profile_defaults(profile):
    key = str(profile).strip().lower()
    if key == "xtremecti":
        return _label_profile(E=6829.0)
    if key == "xtremectii":
        return _label_profile(E=8748.0)
    if key == "proximal_femur_sideways_fall":
        cfg = _label_profile(E=10000.0)
        cfg["materials"]["labels"][2] = {
            "name": "proximal_femur",
            "E": 10000.0,
            "nu": 0.3,
        }
        cfg["load_case"] = {"type": "body_weight", "axis": "y", "force": -1.0}
        return cfg
    if key == "vertebra":
        cfg = _label_profile(E=10000.0)
        cfg["materials"]["labels"][20] = {
            "name": "vertebral_body",
            "E": 10000.0,
            "nu": 0.3,
        }
        cfg["materials"]["labels"][48] = {
            "name": "vertebral_process",
            "E": 10000.0,
            "nu": 0.3,
        }
        cfg["load_case"] = {"type": "constrained_axial", "axis": "z", "strain": -0.01}
        return cfg
    return _label_profile(E=8748.0)


def _enabled_value(value):
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def _resample_density_interpolation_value(value):
    text = str(value or "linear").strip().lower()
    aliases = {
        "b-spline": "bspline",
        "b_spline": "bspline",
        "cubic": "bspline",
    }
    text = aliases.get(text, text)
    return text if text in {"linear", "bspline", "nearest"} else "linear"


def _label_profile(E):
    return {
        "image_type": "material_labels",
        "materials": {
            "units": "MPa",
            "labels": {
                100: {"name": "trabecular_bone", "E": float(E), "nu": 0.3},
                127: {"name": "cortical_bone", "E": float(E), "nu": 0.3},
            },
        },
        "load_case": {"type": "constrained_axial", "axis": "z", "strain": -0.01},
    }


def _postprocess_preset_config(preset):
    key = str(preset).strip().lower()
    postprocess = {
        "fields": {
            "mask_to_segmentation": True,
        }
    }
    if key in {"none", "off", "disabled"}:
        postprocess["pistoia"] = {
            "criterion": "none",
            "critical_strain": None,
            "critical_volume_percent": None,
        }
        postprocess["failure_load"] = False
    elif key in {"kopperdahl", "crawford", "crawford_0.68", "walle", "walle_0.2", "linear_0.2"}:
        postprocess["pistoia"] = {
            "criterion": "none",
            "critical_strain": None,
            "critical_volume_percent": None,
        }
        postprocess["failure_load"] = {
            "linear_deformation": 0.0068,
            "crawford_coefficient": 0.0068,
            "preferred": "crawford_stiffness_height",
            "label": "Kopperdahl/Crawford 0.68%",
        }
    else:
        postprocess["pistoia"] = {
            "criterion": "pistoia",
            "critical_strain": 0.007,
            "critical_volume_percent": 2.0,
        }
        postprocess["failure_load"] = False
    return postprocess


def _slicer_profile_template(profile):
    key = str(profile).strip().lower()
    if key in {"xtremecti", "xtremectii"}:
        cfg = _profile_defaults(key)
        cfg["preprocessing"] = {
            "crop_to_bb": {"enabled": True, "margin_mm": 0.0},
            "largest_cc": True,
            "smooth": {"enabled": False, "density": False, "labels": False},
            "resample_isotropic": {
                "enabled": True,
                "mode": "fixed",
                "target_spacing_mm": 0.0820 if key == "xtremecti" else 0.0607,
                "spacing_tolerance_mm": 0.001,
                "canonicalize_within_tolerance": True,
            },
        }
        cfg["solver"] = {"tolerance": 1.0e-4, "outputs": ["sed"]}
        cfg["output"] = {
            "fields": ["sed"],
            "export_fields": True,
            "visualization_field": "sed",
        }
        cfg["postprocess"] = _postprocess_preset_config("pistoia")
        cfg["slicer_editor"] = _axial_editor_template(
            axis="z",
            contact="Bone surface",
            surface_mode="intersect",
            displacement_value=1.0,
            displacement_units="%",
        )
        return cfg
    if key == "vertebra":
        cfg = _profile_defaults("vertebra")
        cfg.update(
            {
                "materials": {
                    "units": "MPa",
                    "density": {
                        "E": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
                        "nu": 0.3,
                        "active_threshold": 0.0,
                    },
                    "pmma": {"E": 2500.0, "nu": 0.3},
                },
                "preprocessing": {
                    "largest_cc": True,
                    "smooth": {"enabled": True, "sigma_mm": 1.0, "density": True, "labels": True},
                    "crop_to_bb": {"enabled": True, "margin_voxels": 8},
                    "resample_isotropic": {
                        "mode": "auto",
                        "target_spacing_mm": 1.0,
                        "density_interpolation": "bspline",
                    },
                },
                "model": {
                    "type": "spine_compression",
                    "labels": {"body": 20, "process": 48},
                    "registration": {
                        "enabled": True,
                        "method": "lightweight_icp",
                        "target": "vertebral_body",
                    },
                },
                "solver": {"outputs": ["sed"]},
                "output": {
                    "fields": ["sed"],
                    "export_fields": True,
                    "visualization_field": "sed",
                },
                "postprocess": _postprocess_preset_config("kopperdahl"),
            }
        )
        cfg["slicer_editor"] = _axial_editor_template(
            axis="z",
            contact="Material disks",
            top_name="Superior disk",
            bottom_name="Inferior disk",
            displacement_value=-0.2,
            disk_e=2500.0,
            disk_nu=0.3,
            thickness_mm=3.0,
            intrusion_depth_mm=6.0,
            outside_fraction=0.25,
            size_fraction=1.6,
        )
        return cfg
    if key in {"proximal_femur", "proximal_femur_sideways_fall"}:
        cfg = _profile_defaults("proximal_femur_sideways_fall")
        cfg.update(
            {
                "materials": {
                    "units": "MPa",
                    "density": {
                        "E": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
                        "nu": 0.3,
                        "active_threshold": 0.0,
                    },
                    "pmma": {"E": 2500.0, "nu": 0.3},
                },
                "preprocessing": {
                    "largest_cc": True,
                    "smooth": {"enabled": True, "sigma_mm": 1.0, "density": True, "labels": True},
                    "crop_to_bb": {"enabled": True, "margin_voxels": 8},
                    "resample_isotropic": {
                        "mode": "auto",
                        "target_spacing_mm": 1.0,
                        "density_interpolation": "bspline",
                    },
                },
                "model": {
                    "type": "proximal_femur_sideways_fall",
                    "labels": {"femur": 2},
                    "side": "left",
                    "registration": {"enabled": False, "method": "lightweight_icp"},
                },
            }
        )
        cfg["slicer_editor"] = _axial_editor_template(
            axis="y",
            contact="Material disks",
            top_name="Impact disk",
            bottom_name="Support disk",
            displacement_value=1.0,
            disk_e=2500.0,
            disk_nu=0.3,
            thickness_mm=3.0,
            intrusion_depth_mm=6.0,
        )
        return cfg
    if key == "interactive_custom":
        return {"slicer_editor": {"planes": [], "loads": []}, **_profile_defaults("xtremectii")}
    raise ValueError(f"Unknown profile or profile file: {profile}")


def _axial_editor_template(
    *,
    axis,
    contact,
    surface_mode="project",
    top_name="Top",
    bottom_name="Bottom",
    displacement_value=1.0,
    displacement_units="mm",
    disk_e=3000.0,
    disk_nu=0.3,
    thickness_mm=3.0,
    intrusion_depth_mm=2.0,
    outside_fraction=0.0,
    size_fraction=None,
):
    top_plane = {
        "name": top_name,
        "axis": axis,
        "normal": "-",
        "contact": contact,
        "surface_mode": surface_mode,
        "bc_mode": "Displacement",
        "direction": "Plane normal",
        "shape": "anatomy",
        "thickness_mm": thickness_mm,
        "intrusion_depth_mm": intrusion_depth_mm,
        "disk": {"E": disk_e, "nu": disk_nu},
    }
    bottom_plane = {
        "name": bottom_name,
        "axis": axis,
        "normal": "+",
        "contact": contact,
        "surface_mode": surface_mode,
        "bc_mode": "Fixed",
        "direction": "Plane normal",
        "shape": "anatomy",
        "thickness_mm": thickness_mm,
        "intrusion_depth_mm": intrusion_depth_mm,
        "disk": {"E": disk_e, "nu": disk_nu},
    }
    outside_fraction = max(float(outside_fraction), 0.0)
    if outside_fraction > 0.0:
        size_fraction_values = [float(size_fraction or 1.0), float(size_fraction or 1.0)]
        top_plane.update(
            {
                "relative_to": "model_bbox",
                "center_fraction": _axis_center_fraction(axis, 1.0 + outside_fraction),
                "size_fraction": size_fraction_values,
            }
        )
        bottom_plane.update(
            {
                "relative_to": "model_bbox",
                "center_fraction": _axis_center_fraction(axis, -outside_fraction),
                "size_fraction": size_fraction_values,
            }
        )
    return {
        "version": 1,
        "planes": [top_plane, bottom_plane],
        "loads": [
            {
                "nodeset": top_name,
                "mode": "Displacement",
                "direction": "Plane normal",
                "value": displacement_value,
                "units": displacement_units,
            },
            {"nodeset": bottom_name, "mode": "Fixed", "value": 0.0, "units": ""},
        ],
    }


def _axis_center_fraction(axis, axis_fraction):
    values = [0.5, 0.5, 0.5]
    index = {"x": 0, "y": 1, "z": 2}.get(str(axis).strip().lower(), 2)
    values[index] = float(axis_fraction)
    return values


def _editor_state_from_config(config):
    if not isinstance(config, dict):
        return {"planes": [], "loads": []}
    load_case = config.get("load_case", {}) if isinstance(config.get("load_case"), dict) else {}
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    model_type = str(model.get("type", load_case.get("type", ""))).strip().lower()
    if model_type in {"spine_compression", "vertebra", "vertebra_compression"}:
        disk = model.get("geometry", {}).get("disk", {}) if isinstance(model.get("geometry"), dict) else {}
        return _axial_editor_template(
            axis=str(load_case.get("axis", model.get("geometry", {}).get("axis", "z"))),
            contact="Material disks",
            top_name="Superior disk",
            bottom_name="Inferior disk",
            displacement_value=float(load_case.get("displacement", -0.2)),
            disk_e=float(config.get("materials", {}).get("pmma", {}).get("E", 2500.0)),
            disk_nu=float(config.get("materials", {}).get("pmma", {}).get("nu", 0.3)),
            thickness_mm=float(disk.get("thickness_mm", 3.0)),
            intrusion_depth_mm=float(disk.get("intrusion_depth_mm", 6.0)),
            outside_fraction=float(disk.get("outside_fraction", 0.25)),
            size_fraction=float(disk.get("size_fraction", 1.6)),
        )
    if model_type in {"proximal_femur", "proximal_femur_sideways_fall", "sideways_fall"}:
        geometry = model.get("geometry", {}) if isinstance(model.get("geometry"), dict) else {}
        cap = geometry.get("cap", {}) if isinstance(geometry.get("cap"), dict) else {}
        return _axial_editor_template(
            axis=str(geometry.get("cap_axis", load_case.get("axis", "y"))),
            contact="Material disks",
            top_name="Impact disk",
            bottom_name="Support disk",
            displacement_value=float(load_case.get("displacement", 1.0)),
            disk_e=float(config.get("materials", {}).get("pmma", {}).get("E", 2500.0)),
            disk_nu=float(config.get("materials", {}).get("pmma", {}).get("nu", 0.3)),
            thickness_mm=float(cap.get("thickness_mm", 3.0)),
            intrusion_depth_mm=float(cap.get("intrusion_depth_mm", 6.0)),
        )
    axis = str(load_case.get("axis", "z"))
    return _axial_editor_template(
        axis=axis,
        contact="Bone surface",
        surface_mode="intersect",
        displacement_value=1.0,
    )


def _custom_preprocessing_script_value(custom_preprocessing):
    if isinstance(custom_preprocessing, dict):
        value = custom_preprocessing.get("script", custom_preprocessing.get("path"))
    else:
        value = custom_preprocessing
    text = str(value or "").strip()
    return text or None


def _custom_preprocessing_label_value(custom_preprocessing):
    if not isinstance(custom_preprocessing, dict):
        return None
    for key in ("name", "label", "preset"):
        text = str(custom_preprocessing.get(key, "") or "").strip()
        if text:
            return text
    return None


def _custom_preprocessing_options(custom_preprocessing):
    if not isinstance(custom_preprocessing, dict):
        return []
    options = custom_preprocessing.get("options", custom_preprocessing.get("scripts", []))
    if not isinstance(options, (list, tuple)):
        return []
    return [option for option in options if isinstance(option, dict)]


def _custom_preprocessing_option_id(option):
    for key in ("id", "name", "label", "preset", "script", "path"):
        text = str(option.get(key, "") or "").strip() if isinstance(option, dict) else ""
        if text:
            return text
    return "custom_preprocessing"


def _custom_preprocessing_function_value(custom_preprocessing):
    if not isinstance(custom_preprocessing, dict):
        return None
    text = str(custom_preprocessing.get("function", "") or "").strip()
    return text or None


def _custom_preprocessing_identifier(value):
    token = re.sub(r"\W+", "_", str(value or "").strip().lower()).strip("_")
    if not token:
        token = "custom_preprocessing"
    if token[0].isdigit():
        token = f"preprocess_{token}"
    return token


def _custom_preprocessing_scaffold(function_name):
    return CUSTOM_PREPROCESSING_SCAFFOLD_TEMPLATE.replace(
        "__FUNCTION_NAME__",
        _custom_preprocessing_identifier(function_name),
    )


def _profile_bc_mode(value):
    token = str(value or "Displacement").strip().lower()
    if token in {"none", "no load", "material only"}:
        return "None"
    if token in {"fixed", "fix"}:
        return "Fixed"
    if token in {"force", "neumann"}:
        return "Force"
    if token == "bending":
        return "Bending"
    if token == "bending symmetric":
        return "Bending symmetric"
    if token == "torsion":
        return "Torsion"
    if token in {"load history 3", "load_history_3", "load-history-3"}:
        return "Load history 3"
    if token in {"load history 6", "load_history_6", "load-history-6"}:
        return "Load history 6"
    return "Displacement"


def _valid_fixed_dofs(value):
    if value is None:
        return None
    if isinstance(value, str):
        raw_values = re.split(r"[\s,;]+", value.strip())
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        return None
    dofs = []
    for raw in raw_values:
        token = str(raw).strip().lower()
        if token in {"x", "y", "z"} and token not in dofs:
            dofs.append(token)
    return dofs or None


def _float_or_zero(value):
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _load_value_number(spec):
    raw = spec.get("value", 0.0) if isinstance(spec, dict) else spec
    if raw in (None, ""):
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().lower()
    units = str(spec.get("units", "") if isinstance(spec, dict) else "").strip().lower()
    for suffix in ("percent", "percentage", "%", "degrees", "degree", "deg", "radians", "radian", "rad", "mm", "nmm", "n"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    if not text:
        return 0.0
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if match is None:
        return 0.0
    value = float(match.group(0))
    if units in {"percent", "percentage"}:
        return value
    return value


def _default_fea_derivative_root(dataset_root, *, subject_id=None, site=None, session_id=None):
    """Return the shared derivative folder for a ParOSol-backed FEA run."""
    root = Path(dataset_root).expanduser().resolve()
    parts = [root / "derivatives" / "FEA"]
    if subject_id:
        subject = str(subject_id).removeprefix("sub-")
        parts.append(Path(f"sub-{subject}"))
    if site:
        site_name = str(site).removeprefix("site-")
        parts.append(Path(f"site-{site_name}"))
    if session_id:
        session = str(session_id)
        if session and not session.startswith("ses-"):
            session = f"ses-{session}"
        parts.append(Path(session))
    return Path(*parts)


def _fea_derivative_record(*, role, path, subject_id="", site="", session_id="", space="native", source="", metadata=None):
    from bone_imaging_derivatives import DerivativeRecord

    subject = str(subject_id or "unknown").removeprefix("sub-") or "unknown"
    site_name = str(site or "unknown").removeprefix("site-") or "unknown"
    session = str(session_id or "run").removeprefix("ses-") or "run"
    inputs = (str(source),) if source else ()
    return DerivativeRecord(
        derivative="FEA",
        role=str(role),
        subject_id=subject,
        site=site_name,
        session_id=session,
        stack_index=None,
        space=str(space or "native"),
        path=Path(path),
        source="generated",
        inputs=inputs,
        metadata={"backend": "parosol", **dict(metadata or {})},
    )


def _write_fea_derivative_manifest(
    manifest_path,
    *,
    dataset_root,
    records,
    workflow="ParOSolFEA",
    metadata=None,
):
    """Write a compact manifest for FEA outputs generated by this Slicer module."""
    from bone_imaging_derivatives import DerivativeManifest
    from bone_imaging_derivatives import read_manifest as read_shared_manifest
    from bone_imaging_derivatives import write_manifest as write_shared_manifest

    output_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    incoming_records = tuple(records or ())
    if not incoming_records:
        return output_path
    existing_records = ()
    if output_path.exists():
        try:
            existing = read_shared_manifest(output_path)
            existing_records = tuple(existing.records)
        except Exception:
            existing_records = ()
    merged_by_id = {record.record_id: record for record in existing_records}
    for record in incoming_records:
        merged_by_id[record.record_id] = record
    manifest = DerivativeManifest.create(
        "FEA",
        Path(dataset_root).expanduser().resolve(),
        {"name": str(workflow), "version": str(MODULE_VERSION)},
        tuple(merged_by_id.values()),
    )
    write_shared_manifest(manifest, output_path)
    return output_path


def _infer_fea_dataset_root(run_dir):
    resolved = Path(run_dir).expanduser().resolve()
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part == "derivatives" and index + 1 < len(parts) and parts[index + 1] == "FEA":
            return Path(*parts[:index])
    return resolved


def _write_parosol_run_derivative_manifest(output_dir, *, dataset_root=None, subject_id="", site="", session_id=""):
    """Describe standard ParOSol run outputs in the shared FEA derivative format."""
    run_dir = Path(output_dir).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve() if dataset_root else _infer_fea_dataset_root(run_dir)
    candidates = (
        ("solver_config", run_dir / "parosol_slicer_case.yaml"),
        ("solver_input", run_dir / "parosol_input.h5"),
        ("material_image", run_dir / "slicer_input.nii.gz"),
        ("boundary_conditions", run_dir / "slicer_mask.nii.gz"),
        ("sed_map", run_dir / "fields" / "sed.nii.gz"),
        ("summary_table", run_dir / "summary.json"),
        ("diagnostic_log", run_dir / "parosol_stdout.log"),
        ("diagnostic_log", run_dir / "parosol_stderr.log"),
    )
    records = [
        _fea_derivative_record(
            role=role,
            path=path,
            subject_id=subject_id,
            site=site,
            session_id=session_id,
            source=str(run_dir),
        )
        for role, path in candidates
        if path.exists()
    ]
    manifest_path = _default_fea_derivative_root(dataset) / "manifest.json"
    return _write_fea_derivative_manifest(
        manifest_path,
        dataset_root=dataset,
        records=records,
        metadata={"run_dir": str(run_dir)},
    )
