from __future__ import annotations

import importlib
import json
from importlib import metadata
import os
from pathlib import Path
import shutil
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
PLATE_ROD_LOCAL_REPO = TOOLBOX_ROOT.parent / "bone-plate-rod-thinning"


def _use_local_core_repo():
    return os.environ.get("PLATE_ROD_INSTALL_LOCAL") == "1" and PLATE_ROD_LOCAL_REPO.exists()


def _remove_local_core_repo_from_sys_path():
    local_repo = str(PLATE_ROD_LOCAL_REPO)
    sys.path[:] = [path for path in sys.path if str(path) != local_repo]

from SlicerBoneImagingToolboxLib.slicer_pip import slicer_pip_install, slicer_python_executable  # noqa: E402

from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


def run_plate_rod_batch(*args, **kwargs):
    from plate_rod_thinning.batch import run_plate_rod_batch as _run_plate_rod_batch

    return _run_plate_rod_batch(*args, **kwargs)


MODULE_VERSION = "0.1.3"
PLATE_ROD_CITATION = (
    "Walle M, Yeritsyan D, Abbasian M, Oftadeh R, Müller R, Nazarian A. "
    "A graph model to describe the network connectivity of trabecular plates and rods. "
    "Front Bioeng Biotechnol. 2024 May 6;12:1384280. "
    "doi: 10.3389/fbioe.2024.1384280. PMID: 38770275; PMCID: PMC11103010."
)
SEGMENT_NAME_HINTS = {
    "bone segmentation": ("Bone segmentation", "bone", "seg"),
    "trabecular compartment mask": ("Trabecular mask", "trabecular", "trab"),
}
LABEL_COLORS = (
    ("Plate", (0.0, 0.45, 1.0)),
    ("Rod", (1.0, 0.05, 0.02)),
    ("Junction", (0.9, 0.1, 0.25)),
)
TOPOLOGY_LABEL_COLORS = (
    ("Surface endpoint", (0.35, 0.75, 1.0)),
    ("Surface inner", (0.0, 0.45, 1.0)),
    ("Surface-surface junction", (0.55, 0.15, 0.95)),
    ("Surface-curve junction", (0.9, 0.1, 0.25)),
    ("Arc endpoint", (1.0, 0.8, 0.15)),
    ("Arc inner", (1.0, 0.55, 0.0)),
    ("Arc-arc junction", (0.75, 0.05, 0.05)),
    ("Isolated", (0.65, 0.65, 0.65)),
)
_PLATE_ROD_PROCESS_SCRIPT = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main(job_json_path: str) -> int:
    job_path = Path(job_json_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("local_repo") and str(job["local_repo"]) not in sys.path:
        sys.path.insert(0, str(job["local_repo"]))

    from plate_rod_thinning import PlateRodParameters, plate_rod_analysis

    print("[plate-rod] reading masks", flush=True)
    bone_image = sitk.ReadImage(str(job["bone_path"]), sitk.sitkUInt8)
    trab_image = sitk.ReadImage(str(job["trab_path"]), sitk.sitkUInt8)
    common_region_path = str(job.get("common_region_path") or "")
    if common_region_path:
        common_image = sitk.ReadImage(common_region_path, sitk.sitkUInt8)
        if common_image.GetSize() != bone_image.GetSize():
            raise ValueError("Common scan region mask must match bone and trabecular mask geometry.")
        bone_image = sitk.Cast((bone_image > 0) & (common_image > 0), sitk.sitkUInt8)
        trab_image = sitk.Cast((trab_image > 0) & (common_image > 0), sitk.sitkUInt8)
    bone = sitk.GetArrayFromImage(bone_image) > 0
    trab = sitk.GetArrayFromImage(trab_image) > 0
    if bone.shape != trab.shape:
        raise ValueError("Bone segmentation and trabecular compartment mask must have matching geometry.")

    print("[plate-rod] running thinning and morphometry", flush=True)
    parameters = PlateRodParameters(
        slenderness=int(job["slenderness"]),
        max_iterations=int(job["max_iterations"]),
        voxel_spacing_mm=tuple(float(value) for value in bone_image.GetSpacing()),
        min_plate_voxels=int(job["min_plate_voxels"]),
        min_rod_voxels=int(job["min_rod_voxels"]),
    )
    result = plate_rod_analysis(bone & trab, analysis_mask=trab, parameters=parameters)

    print("[plate-rod] writing outputs", flush=True)
    maps = {
        "Skeleton topology labels": result.topology_labels,
        "Full-thickness labels": result.full_thickness_labels,
        "Component labels": result.component_labels,
    }
    output_paths = {}
    for role, array in maps.items():
        path = Path(job["output_paths"][role])
        image = sitk.GetImageFromArray(np.asarray(array))
        image.CopyInformation(bone_image)
        sitk.WriteImage(image, str(path))
        output_paths[role] = str(path)

    summary_path = Path(job["summary_path"])
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    print("[plate-rod] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
"""


class PlateRodMorphometryHRpQCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Plate/Rod Morphometry"
        parent.categories = ["Bone Imaging.HR-pQCT"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Compute trabecular plate/rod skeleton labels, full-thickness labels, and morphometry summaries.\n"
            f"Module version: {MODULE_VERSION}\n\n"
            f"Citation: {PLATE_ROD_CITATION}"
        )
        parent.acknowledgementText = (
            "This module wraps the separate plate_rod_thinning core package. "
            f"Please cite: {PLATE_ROD_CITATION}"
        )


class PlateRodMorphometryHRpQCTLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._proc = None
        self._lastCoreInstallMessage = ""

    def run_batch_workflow(self, dataset_root, *, use_common_region=True, force=False, progress=None):
        """Run folder mode through the package batch API, not Slicer logic."""
        return run_plate_rod_batch(
            Path(dataset_root),
            use_common_region=bool(use_common_region),
            force=bool(force),
            progress=progress,
        )

    def run_folder_batch(self, dataset_root, *, use_common_region=True, force=False, progress=None):
        """Slicer folder-mode action boundary for package batch execution."""
        return self.run_batch_workflow(
            dataset_root,
            use_common_region=use_common_region,
            force=force,
            progress=progress,
        )

    @staticmethod
    def folder_batch_command(dataset_root, *, subject_id="", site="", use_common_region=True, force=False):
        command = ["-m", "plate_rod_thinning.cli", "run-batch", str(Path(dataset_root).expanduser().resolve())]
        if subject_id:
            command.extend(["--subject", str(subject_id)])
        if site:
            command.extend(["--site", str(site)])
        if not use_common_region:
            command.append("--no-common-region")
        if force:
            command.append("--force")
        return command

    def run_folder_batch_job(self, dataset_root, *, subject_id="", site="", use_common_region=True, force=False,
                             on_output=None, on_finished=None):
        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.MergedChannels)
        proc.readyRead.connect(lambda: on_output and on_output(bytes(proc.readAll()).decode("utf-8", errors="replace")))
        proc.finished.connect(lambda code, status: on_finished and on_finished(code, status))
        proc.start(slicer_python_executable(slicer.app.applicationFilePath()), self.folder_batch_command(
            dataset_root, subject_id=subject_id, site=site, use_common_region=use_common_region, force=force,
        ))
        return proc

    def core_runtime_status(self):
        try:
            version = metadata.version("plate-rod-thinning")
        except metadata.PackageNotFoundError:
            return False, "Plate/Rod core is not installed in Slicer Python."
        except Exception as exc:
            return False, f"Plate/Rod core package status could not be checked: {exc}"

        try:
            package = importlib.import_module("plate_rod_thinning")
            importlib.import_module("plate_rod_thinning._c_backend")
        except Exception as exc:
            package_path = ""
            try:
                package_path = f"\nImported package path: {importlib.import_module('plate_rod_thinning').__file__}"
            except Exception:
                pass
            return False, f"Plate/Rod core installed without compiled backend ({version}): {exc}{package_path}"
        metal_status = "Metal availability unknown"
        try:
            from plate_rod_thinning import metal_backend

            metal_status = "Metal available" if metal_backend.status().available else "Metal unavailable"
        except Exception as exc:
            metal_status = f"Metal status unavailable: {exc}"
        package_path = getattr(package, "__file__", "")
        path_detail = f" from {package_path}" if package_path else ""
        return True, f"Plate/Rod core available ({version}, compiled backend; {metal_status}){path_detail}."

    def is_core_available(self):
        return self.core_runtime_status()[0]

    def install_or_update_core(self):
        slicer_pip_install("numpy scipy")
        _remove_local_core_repo_from_sys_path()
        slicer_pip_install(
            "--upgrade --force-reinstall --prefer-binary "
            "--only-binary :all: --no-deps plate-rod-thinning>=0.1.3"
        )
        importlib.invalidate_caches()
        _remove_local_core_repo_from_sys_path()
        for name in list(sys.modules):
            if name == "plate_rod_thinning" or name.startswith("plate_rod_thinning."):
                sys.modules.pop(name, None)
        available, message = self.core_runtime_status()
        self._lastCoreInstallMessage = message
        if not available:
            raise RuntimeError(
                f"{message}\n"
                "Use Slicer's bundled Python 3.12 on macOS x86_64 or arm64, where the published "
                "plate-rod-thinning wheel includes the compiled backend."
            )
        return message

    def prepare_plate_rod_job(
        self,
        bone_segmentation_node,
        trabecular_mask_node,
        *,
        bone_segment_id=None,
        trabecular_segment_id=None,
        common_region_node=None,
        common_region_segment_id=None,
        slenderness=0,
        max_iterations=200,
        min_plate_voxels=0,
        min_rod_voxels=0,
        use_metal=True,
        output_prefix="",
    ):
        reference_node = self._first_available_reference_node(bone_segmentation_node, trabecular_mask_node, common_region_node)
        temporary_reference_node = None
        if reference_node is None and bone_segmentation_node is not None and bone_segmentation_node.IsA("vtkMRMLSegmentationNode"):
            temporary_reference_node = self._segmentation_reference_node(
                bone_segmentation_node,
                [("bone segmentation", bone_segment_id)],
            )
            reference_node = temporary_reference_node

        job_dir = Path(tempfile.mkdtemp(prefix="hrpqct_plate_rod_job_"))
        prefix = str(output_prefix or "PlateRod").strip() or "PlateRod"
        try:
            bone_image = self._volume_to_sitk_uint8(
                bone_segmentation_node,
                "bone segmentation",
                selected_segment_id=bone_segment_id,
                reference_node=reference_node,
            )
            trab_image = self._volume_to_sitk_uint8(
                trabecular_mask_node,
                "trabecular compartment mask",
                selected_segment_id=trabecular_segment_id,
                reference_node=reference_node,
            )
            common_region_path = None
            if common_region_node is not None:
                from SlicerBoneImagingToolboxLib.masks import clip_mask_to_region

                common_region = self._volume_to_sitk_uint8(
                    common_region_node,
                    "common scan region mask",
                    selected_segment_id=common_region_segment_id,
                    reference_node=reference_node,
                )
                bone_image = clip_mask_to_region(bone_image, common_region)
                trab_image = clip_mask_to_region(trab_image, common_region)
                common_region_path = job_dir / "common_region.nrrd"
                sitk.WriteImage(common_region, str(common_region_path))
            bone_path = job_dir / "bone.nrrd"
            trab_path = job_dir / "trab.nrrd"
            sitk.WriteImage(bone_image, str(bone_path))
            sitk.WriteImage(trab_image, str(trab_path))

            output_paths = {
                "Skeleton topology labels": str(job_dir / "skeleton_topology_labels.nrrd"),
                "Full-thickness labels": str(job_dir / "full_thickness_labels.nrrd"),
                "Component labels": str(job_dir / "component_labels.nrrd"),
            }
            job = {
                "job_dir": str(job_dir),
                "job_json_path": str(job_dir / "job.json"),
                "bone_path": str(bone_path),
                "trab_path": str(trab_path),
                "common_region_path": str(common_region_path) if common_region_path else "",
                "common_region_node_id": common_region_node.GetID() if common_region_node is not None else "",
                "common_region_node_name": common_region_node.GetName() if common_region_node is not None else "",
                "output_paths": output_paths,
                "summary_path": str(job_dir / "summary.json"),
                "prefix": prefix,
                "slenderness": int(slenderness),
                "max_iterations": int(max_iterations),
                "min_plate_voxels": int(min_plate_voxels),
                "min_rod_voxels": int(min_rod_voxels),
                "use_metal": bool(use_metal),
                "reference_node": reference_node or bone_segmentation_node,
                "temporary_reference_node": temporary_reference_node,
                "local_repo": str(PLATE_ROD_LOCAL_REPO) if _use_local_core_repo() else "",
            }
            Path(job["job_json_path"]).write_text(
                json.dumps({key: value for key, value in job.items() if key not in {"reference_node", "temporary_reference_node"}}),
                encoding="utf-8",
            )
            return job
        except Exception:
            if temporary_reference_node is not None:
                slicer.mrmlScene.RemoveNode(temporary_reference_node)
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def run_plate_rod_job(self, job, *, on_output=None, on_finished=None):
        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.SeparateChannels)
        env = qt.QProcessEnvironment.systemEnvironment()
        if env.contains("ITK_AUTOLOAD_PATH"):
            env.remove("ITK_AUTOLOAD_PATH")
        if env.contains("SITK_AUTOLOAD_PATH"):
            env.remove("SITK_AUTOLOAD_PATH")
        env.insert("ITK_AUTOLOAD_PATH", "")
        env.insert("SITK_AUTOLOAD_PATH", "")
        if bool(job.get("use_metal", True)):
            env.insert("PLATE_ROD_USE_METAL_FULL", "1")
        else:
            env.insert("PLATE_ROD_USE_METAL_FULL", "0")
        proc.setProcessEnvironment(env)

        def _decode(raw):
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
                    data = str(raw).encode("utf-8", errors="replace")
            return data.decode("utf-8", errors="replace")

        def _read_stdout():
            text = _decode(proc.readAllStandardOutput())
            if text and on_output:
                on_output(text)

        def _read_stderr():
            text = _decode(proc.readAllStandardError())
            if text and on_output:
                on_output(text)

        def _finished(*signal_args):
            self._proc = None
            if len(signal_args) >= 2:
                exit_code = int(signal_args[0])
                exit_status = signal_args[1]
            elif len(signal_args) == 1:
                exit_code = int(signal_args[0])
                exit_status = 0
            else:
                exit_code = int(proc.exitCode())
                exit_status = proc.exitStatus()
            if on_finished:
                on_finished(exit_code, exit_status)

        proc.readyReadStandardOutput.connect(_read_stdout)
        proc.readyReadStandardError.connect(_read_stderr)
        proc.finished.connect(_finished)

        python_exe = shutil.which("PythonSlicer") or shutil.which("python") or shutil.which("python3")
        if python_exe is None:
            raise RuntimeError("Could not find Python executable in Slicer environment")
        proc.start(python_exe, ["-c", _PLATE_ROD_PROCESS_SCRIPT, str(job["job_json_path"])])
        if not proc.waitForStarted(3000):
            raise RuntimeError("Failed to start plate/rod process")
        self._proc = proc

    def load_plate_rod_job_outputs(self, job):
        nodes = {}
        for map_role, path in job["output_paths"].items():
            image = sitk.ReadImage(str(path))
            node = self._sitk_to_labelmap_volume(
                image,
                f"{job['prefix']}_{map_role.lower().replace(' ', '_').replace('-', '')}",
                job["reference_node"],
                map_role,
                slenderness=int(job["slenderness"]),
            )
            if map_role != "Component labels":
                set_labelmap_display_colors(node, map_role)
            if job.get("common_region_node_id"):
                node.SetAttribute("BoneImaging.PlateRod.CommonRegionNode", str(job.get("common_region_node_id", "")))
                node.SetAttribute("BoneImaging.PlateRod.CommonRegionName", str(job.get("common_region_node_name", "")))
            nodes[map_role] = node
        summary = json.loads(Path(job["summary_path"]).read_text(encoding="utf-8"))
        table_node = self._create_summary_table(summary, f"{job['prefix']}_summary")
        if job.get("common_region_node_id"):
            table_node.SetAttribute("BoneImaging.PlateRod.CommonRegionNode", str(job.get("common_region_node_id", "")))
            table_node.SetAttribute("BoneImaging.PlateRod.CommonRegionName", str(job.get("common_region_node_name", "")))
        show_full_thickness_labels_in_3d(nodes.get("Full-thickness labels"))
        return {"nodes": nodes, "table": table_node, "summary": summary}

    def cleanup_plate_rod_job(self, job):
        if not job:
            return
        temporary_reference_node = job.get("temporary_reference_node")
        if temporary_reference_node is not None:
            slicer.mrmlScene.RemoveNode(temporary_reference_node)
        shutil.rmtree(job.get("job_dir", ""), ignore_errors=True)

    def compute_plate_rod_morphometry(
        self,
        bone_segmentation_node,
        trabecular_mask_node,
        *,
        bone_segment_id=None,
        trabecular_segment_id=None,
        common_region_node=None,
        common_region_segment_id=None,
        slenderness=0,
        max_iterations=200,
        min_plate_voxels=0,
        min_rod_voxels=0,
        output_prefix="",
    ):
        from plate_rod_thinning import PlateRodParameters, plate_rod_analysis

        reference_node = self._first_available_reference_node(bone_segmentation_node, trabecular_mask_node, common_region_node)
        temporary_reference_node = None
        if reference_node is None and bone_segmentation_node is not None and bone_segmentation_node.IsA("vtkMRMLSegmentationNode"):
            temporary_reference_node = self._segmentation_reference_node(
                bone_segmentation_node,
                [("bone segmentation", bone_segment_id)],
            )
            reference_node = temporary_reference_node
        try:
            bone_image = self._volume_to_sitk_uint8(
                bone_segmentation_node,
                "bone segmentation",
                selected_segment_id=bone_segment_id,
                reference_node=reference_node,
            )
            trab_image = self._volume_to_sitk_uint8(
                trabecular_mask_node,
                "trabecular compartment mask",
                selected_segment_id=trabecular_segment_id,
                reference_node=reference_node,
            )
            if common_region_node is not None:
                from SlicerBoneImagingToolboxLib.masks import clip_mask_to_region

                common_region = self._volume_to_sitk_uint8(
                    common_region_node,
                    "common scan region mask",
                    selected_segment_id=common_region_segment_id,
                    reference_node=reference_node,
                )
                bone_image = clip_mask_to_region(bone_image, common_region)
                trab_image = clip_mask_to_region(trab_image, common_region)
            bone = sitk.GetArrayFromImage(bone_image) > 0
            trab = sitk.GetArrayFromImage(trab_image) > 0
            if bone.shape != trab.shape:
                raise ValueError("Bone segmentation and trabecular compartment mask must have matching geometry.")
            trabecular_bone = bone & trab
            parameters = PlateRodParameters(
                slenderness=int(slenderness),
                max_iterations=int(max_iterations),
                voxel_spacing_mm=tuple(float(value) for value in bone_image.GetSpacing()),
                min_plate_voxels=int(min_plate_voxels),
                min_rod_voxels=int(min_rod_voxels),
            )
            core_result = plate_rod_analysis(trabecular_bone, analysis_mask=trab, parameters=parameters)
            prefix = str(output_prefix or "PlateRod").strip() or "PlateRod"
            maps = {
                "Skeleton topology labels": core_result.topology_labels,
                "Full-thickness labels": core_result.full_thickness_labels,
                "Component labels": core_result.component_labels,
            }
            nodes = {}
            for map_role, array in maps.items():
                image = self._array_to_sitk_like(array, bone_image)
                node = self._sitk_to_labelmap_volume(
                    image,
                    f"{prefix}_{map_role.lower().replace(' ', '_').replace('-', '')}",
                    reference_node or bone_segmentation_node,
                    map_role,
                    slenderness=int(slenderness),
                )
                if map_role != "Component labels":
                    set_labelmap_display_colors(node, map_role)
                nodes[map_role] = node
            table_node = self._create_summary_table(core_result.summary, f"{prefix}_summary")
            if common_region_node is not None:
                table_node.SetAttribute("BoneImaging.PlateRod.CommonRegionNode", common_region_node.GetID())
                table_node.SetAttribute("BoneImaging.PlateRod.CommonRegionName", common_region_node.GetName())
            show_full_thickness_labels_in_3d(nodes.get("Full-thickness labels"))
            return {"nodes": nodes, "table": table_node, "summary": core_result.summary}
        finally:
            if temporary_reference_node is not None:
                slicer.mrmlScene.RemoveNode(temporary_reference_node)

    def _volume_to_sitk_uint8(self, volume_node, role, selected_segment_id=None, reference_node=None):
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
            with tempfile.TemporaryDirectory(prefix="hrpqct_plate_rod_in_") as temp_dir:
                path = Path(temp_dir) / f"{role.replace(' ', '_')}.nrrd"
                if not slicer.util.saveNode(volume_node, str(path)):
                    raise RuntimeError(f"Could not save selected {role} for plate/rod processing.")
                return sitk.ReadImage(str(path), sitk.sitkUInt8)
        finally:
            if temporary_node is not None:
                slicer.mrmlScene.RemoveNode(temporary_node)

    def _segment_id_for_role(self, segmentation_node, role, selected_segment_id=None):
        segmentation = segmentation_node.GetSegmentation()
        if selected_segment_id:
            segment = segmentation.GetSegment(str(selected_segment_id))
            if segment is None:
                raise ValueError(f"Selected segment ID {selected_segment_id} was not found in {segmentation_node.GetName()}.")
            return str(selected_segment_id)
        if segmentation.GetNumberOfSegments() == 1:
            return segmentation.GetNthSegmentID(0)
        hints = tuple(hint.lower() for hint in SEGMENT_NAME_HINTS.get(str(role), (str(role),)))
        for index in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(index)
            segment = segmentation.GetSegment(segment_id)
            name = str(segment.GetName() if segment is not None else "").strip().lower()
            if name in hints or any(hint in name for hint in hints):
                return segment_id
        raise ValueError(f"Could not find a {role} segment in {segmentation_node.GetName()}.")

    def _segmentation_reference_node(self, segmentation_node, roles_and_segment_ids):
        segment_ids = vtk.vtkStringArray()
        for role, selected_segment_id in roles_and_segment_ids:
            segment_ids.InsertNextValue(self._segment_id_for_role(segmentation_node, role, selected_segment_id))
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{segmentation_node.GetName()}_plate_rod_reference_geometry",
        )
        try:
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentation_node, segment_ids, labelmap_node)
            return labelmap_node
        except Exception:
            slicer.mrmlScene.RemoveNode(labelmap_node)
            raise

    def _segmentation_node_to_labelmap(self, segmentation_node, role, *, selected_segment_id=None, reference_node=None):
        segment_id = self._segment_id_for_role(segmentation_node, role, selected_segment_id)
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{segmentation_node.GetName()}_{role.replace(' ', '_')}",
        )
        segment_ids = vtk.vtkStringArray()
        segment_ids.InsertNextValue(segment_id)
        try:
            export_args = [segmentation_node, segment_ids, labelmap_node]
            extent_mode = getattr(slicer.vtkSlicerSegmentationsModuleLogic, "EXTENT_REFERENCE_GEOMETRY", None)
            if reference_node is not None:
                export_args.append(reference_node)
            if reference_node is not None and extent_mode is not None:
                export_args.append(extent_mode)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(*export_args)
            return labelmap_node
        except Exception:
            slicer.mrmlScene.RemoveNode(labelmap_node)
            raise

    def _first_available_reference_node(self, *nodes):
        for node in nodes:
            if node is not None and not node.IsA("vtkMRMLSegmentationNode"):
                return node
        return None

    def _array_to_sitk_like(self, array, reference_image):
        image = sitk.GetImageFromArray(np.asarray(array))
        image.CopyInformation(reference_image)
        return image

    def _sitk_to_labelmap_volume(self, image, name, reference_node, map_role, *, slenderness):
        with tempfile.TemporaryDirectory(prefix="hrpqct_plate_rod_out_") as temp_dir:
            path = Path(temp_dir) / f"{name}.nrrd"
            sitk.WriteImage(image, str(path))
            loaded = slicer.util.loadLabelVolume(str(path), {"name": name})
        if isinstance(loaded, tuple):
            success, volume_node = loaded
        else:
            success, volume_node = bool(loaded), loaded
        if not success or volume_node is None:
            raise RuntimeError(f"Could not load generated plate/rod map: {name}")
        if hasattr(reference_node, "CopyOrientation"):
            volume_node.CopyOrientation(reference_node)
        volume_node.SetAttribute("BoneImaging.PlateRod.Engine", "plate_rod_thinning")
        volume_node.SetAttribute("BoneImaging.PlateRod.MapRole", map_role)
        volume_node.SetAttribute("BoneImaging.PlateRod.Slenderness", str(int(slenderness)))
        return volume_node

    def _create_summary_table(self, summary, name):
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        table_node.SetAttribute("BoneImaging.PlateRod.Engine", "plate_rod_thinning")
        metric_column = vtk.vtkStringArray()
        metric_column.SetName("Metric")
        value_column = vtk.vtkStringArray()
        value_column.SetName("Value")
        for key in sorted(summary):
            value = summary[key]
            metric_column.InsertNextValue(str(key))
            if isinstance(value, float):
                value = f"{value:.8g}"
            value_column.InsertNextValue(str(value))
        table_node.GetTable().AddColumn(metric_column)
        table_node.GetTable().AddColumn(value_column)
        table_node.Modified()
        return table_node


def set_labelmap_display_colors(volume_node, map_role=""):
    display_node = volume_node.GetDisplayNode() if volume_node is not None else None
    if display_node is None:
        return
    labels = TOPOLOGY_LABEL_COLORS if map_role == "Skeleton topology labels" else LABEL_COLORS
    color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", f"{volume_node.GetName()}_colors")
    color_node.SetTypeToUser()
    color_node.SetNumberOfColors(len(labels) + 1)
    color_node.SetColor(0, "Background", 0.0, 0.0, 0.0, 0.0)
    for index, (name, color) in enumerate(labels, start=1):
        color_node.SetColor(index, name, color[0], color[1], color[2], 1.0)
    display_node.SetAndObserveColorNodeID(color_node.GetID())


def show_full_thickness_labels_in_3d(labelmap_node):
    if labelmap_node is None:
        return None
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode",
        f"{labelmap_node.GetName()}_3D",
    )
    segmentation_node.CreateDefaultDisplayNodes()
    if hasattr(segmentation_node, "SetReferenceImageGeometryParameterFromVolumeNode"):
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(labelmap_node)
    try:
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap_node, segmentation_node)
    except Exception:
        slicer.mrmlScene.RemoveNode(segmentation_node)
        raise

    segmentation = segmentation_node.GetSegmentation()
    for label_index, (name, color) in enumerate(LABEL_COLORS):
        if label_index >= segmentation.GetNumberOfSegments():
            break
        segment_id = segmentation.GetNthSegmentID(label_index)
        segment = segmentation.GetSegment(segment_id)
        if segment is None:
            continue
        segment.SetName(name)
        segment.SetColor(color[0], color[1], color[2])

    display_node = segmentation_node.GetDisplayNode()
    if display_node is not None:
        display_node.SetVisibility(True)
        display_node.SetVisibility3D(True)
        display_node.SetVisibility2DFill(False)
        display_node.SetVisibility2DOutline(True)
        display_node.SetOpacity3D(0.65)
    segmentation_node.CreateClosedSurfaceRepresentation()
    try:
        slicer.util.resetThreeDViews()
    except Exception:
        pass
    return segmentation_node


class PlateRodMorphometryHRpQCTWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = PlateRodMorphometryHRpQCTLogic()
        self._lastTable = None
        self._activePlateRodJob = None

        parameters_collapsible = ctk.ctkCollapsibleButton()
        parameters_collapsible.text = "Parameters"
        self.layout.addWidget(parameters_collapsible)
        form_layout = qt.QFormLayout(parameters_collapsible)

        self.boneSegmentationSelector = slicer.qMRMLNodeComboBox()
        self.boneSegmentationSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLSegmentationNode"]
        self.boneSegmentationSelector.selectNodeUponCreation = False
        self.boneSegmentationSelector.addEnabled = False
        self.boneSegmentationSelector.removeEnabled = False
        self.boneSegmentationSelector.noneEnabled = True
        self.boneSegmentationSelector.setMRMLScene(slicer.mrmlScene)
        self.boneSegmentationSelector.currentNodeChanged.connect(self._refresh_bone_segment_selector)

        self.boneSegmentSelector = self._segment_combo()
        form_layout.addRow("Bone segmentation", self._segmentation_input_row(self.boneSegmentationSelector, self.boneSegmentSelector))

        self.trabecularMaskSelector = slicer.qMRMLNodeComboBox()
        self.trabecularMaskSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLSegmentationNode"]
        self.trabecularMaskSelector.selectNodeUponCreation = False
        self.trabecularMaskSelector.addEnabled = False
        self.trabecularMaskSelector.removeEnabled = False
        self.trabecularMaskSelector.noneEnabled = True
        self.trabecularMaskSelector.setMRMLScene(slicer.mrmlScene)
        self.trabecularMaskSelector.currentNodeChanged.connect(self._refresh_trabecular_segment_selector)

        self.trabecularSegmentSelector = self._segment_combo()
        form_layout.addRow("Trabecular compartment mask", self._segmentation_input_row(self.trabecularMaskSelector, self.trabecularSegmentSelector))

        self.commonRegionMaskSelector = slicer.qMRMLNodeComboBox()
        self.commonRegionMaskSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLSegmentationNode"]
        self.commonRegionMaskSelector.selectNodeUponCreation = False
        self.commonRegionMaskSelector.addEnabled = False
        self.commonRegionMaskSelector.removeEnabled = False
        self.commonRegionMaskSelector.noneEnabled = True
        self.commonRegionMaskSelector.setMRMLScene(slicer.mrmlScene)
        self.commonRegionMaskSelector.currentNodeChanged.connect(self._refresh_common_region_segment_selector)

        self.commonRegionSegmentSelector = self._segment_combo()
        form_layout.addRow("Common scan region mask", self._segmentation_input_row(self.commonRegionMaskSelector, self.commonRegionSegmentSelector))

        self.slendernessSpinBox = qt.QSpinBox()
        self.slendernessSpinBox.minimum = 0
        self.slendernessSpinBox.maximum = 10
        self.slendernessSpinBox.value = 0
        form_layout.addRow("Slenderness", self.slendernessSpinBox)

        self.maxIterationsSpinBox = qt.QSpinBox()
        self.maxIterationsSpinBox.minimum = 1
        self.maxIterationsSpinBox.maximum = 10000
        self.maxIterationsSpinBox.value = 200
        form_layout.addRow("Max thinning iterations", self.maxIterationsSpinBox)

        self.minPlateSpinBox = qt.QSpinBox()
        self.minPlateSpinBox.minimum = 0
        self.minPlateSpinBox.maximum = 1000000
        self.minPlateSpinBox.value = 0
        form_layout.addRow("Minimum plate voxels", self.minPlateSpinBox)

        self.minRodSpinBox = qt.QSpinBox()
        self.minRodSpinBox.minimum = 0
        self.minRodSpinBox.maximum = 1000000
        self.minRodSpinBox.value = 0
        form_layout.addRow("Minimum rod voxels", self.minRodSpinBox)

        self.useMetalCheckBox = qt.QCheckBox()
        self.useMetalCheckBox.text = "Use Metal acceleration on macOS"
        self.useMetalCheckBox.checked = True
        self.useMetalCheckBox.toolTip = "Use the Metal skeletonization backend when available; falls back to the compiled backend otherwise."
        form_layout.addRow("Acceleration", self.useMetalCheckBox)

        self.outputPrefixEdit = qt.QLineEdit()
        self.outputPrefixEdit.text = "PlateRod"
        form_layout.addRow("Output prefix", self.outputPrefixEdit)

        self.runButton = qt.QPushButton("Run plate/rod morphometry")
        self.runButton.clicked.connect(self._run_plate_rod_morphometry)
        form_layout.addRow(self.runButton)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.visible = False
        self.progressBar.textVisible = False
        form_layout.addRow("Progress", self.progressBar)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        form_layout.addRow("Status", self.statusLabel)

        self.installCoreButton = qt.QPushButton("Install / update compiled plate-rod core")
        self.installCoreButton.toolTip = "Install the published plate-rod-thinning wheel and verify the compiled backend."
        self.installCoreButton.clicked.connect(self._install_or_update_core)
        form_layout.addRow(self.installCoreButton)
        self._refresh_core_status()

        batch = ctk.ctkCollapsibleButton()
        batch.text = "Folder Batch"
        self.layout.addWidget(batch)
        batch_form = qt.QFormLayout(batch)
        self.folderDatasetRootEdit = qt.QLineEdit()
        self.folderSubjectEdit = qt.QLineEdit()
        self.folderSiteEdit = qt.QLineEdit()
        self.folderUseCommonRegionCheck = qt.QCheckBox()
        self.folderUseCommonRegionCheck.checked = True
        self.folderForceCheck = qt.QCheckBox()
        self.folderRunButton = qt.QPushButton("Run folder batch")
        self.folderBatchStatus = qt.QLabel()
        self.folderRunButton.clicked.connect(self._run_folder_batch)
        batch_form.addRow("Dataset root", self.folderDatasetRootEdit)
        batch_form.addRow("Subject", self.folderSubjectEdit)
        batch_form.addRow("Site", self.folderSiteEdit)
        batch_form.addRow("Use common region", self.folderUseCommonRegionCheck)
        batch_form.addRow("Force recompute", self.folderForceCheck)
        batch_form.addRow(self.folderRunButton)
        batch_form.addRow(self.folderBatchStatus)

        self.layout.addStretch(1)

    def _run_folder_batch(self):
        self.folderRunButton.enabled = False
        self.folderBatchStatus.text = "Folder batch is running in the background."
        self.logic.run_folder_batch_job(
            self.folderDatasetRootEdit.text,
            subject_id=self.folderSubjectEdit.text,
            site=self.folderSiteEdit.text,
            use_common_region=bool(self.folderUseCommonRegionCheck.checked),
            force=bool(self.folderForceCheck.checked),
            on_finished=self._on_folder_batch_finished,
        )

    def _on_folder_batch_finished(self, exit_code, _exit_status):
        self.folderRunButton.enabled = True
        self.folderBatchStatus.text = f"Folder batch finished with exit code {int(exit_code)}."

    def _segmentation_input_row(self, node_selector, segment_selector):
        row = qt.QWidget()
        row_layout = qt.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(node_selector, 2)
        row_layout.addWidget(segment_selector, 1)
        return row

    def _segment_combo(self):
        combo = qt.QComboBox()
        combo.addItem("Auto", "")
        combo.enabled = False
        combo.toolTip = "Segment to use when the selected node is a Slicer segmentation."
        return combo

    def _refresh_bone_segment_selector(self, node=None):
        self._refresh_segment_selector(self.boneSegmentSelector, node, "bone segmentation")

    def _refresh_trabecular_segment_selector(self, node=None):
        self._refresh_segment_selector(self.trabecularSegmentSelector, node, "trabecular compartment mask")

    def _refresh_common_region_segment_selector(self, node=None):
        self._refresh_segment_selector(self.commonRegionSegmentSelector, node, "common scan region mask")

    def _refresh_segment_selector(self, selector, node, role):
        selector.blockSignals(True)
        selector.clear()
        selector.addItem("Auto", "")
        is_segmentation = bool(node is not None and node.IsA("vtkMRMLSegmentationNode"))
        selector.enabled = is_segmentation
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
                label = segment_name
                if auto_id and segment_id == auto_id:
                    label = f"{segment_name} (auto)"
                selector.addItem(label, segment_id)
        selector.blockSignals(False)

    def _selected_segment_id(self, selector):
        selected_segment_id = str(selector.currentData or "").strip()
        return selected_segment_id or None

    def _refresh_core_status(self):
        _, message = self.logic.core_runtime_status()
        self.statusLabel.text = message

    def _install_or_update_core(self):
        with slicer.util.tryWithErrorDisplay("Failed to install/update plate-rod core.", waitCursor=True):
            self.runButton.enabled = False
            self.installCoreButton.enabled = False
            completed = False
            try:
                self._set_progress(True, "Installing/updating compiled plate-rod core...")
                message = self.logic.install_or_update_core()
                self._set_progress(False, message)
                completed = True
            finally:
                if not completed:
                    self._set_progress(False, "Plate/rod core install/update stopped.")
                self.runButton.enabled = True
                self.installCoreButton.enabled = True

    def _set_progress(self, running, message):
        self.progressBar.visible = bool(running)
        self.statusLabel.text = str(message)
        slicer.app.processEvents()

    def _show_process_output(self, text):
        message = str(text or "").strip()
        lowered = message.lower()
        if "reading masks" in lowered:
            self._set_progress(True, "Reading selected masks...")
        elif "running thinning" in lowered:
            self._set_progress(True, "Running thinning and morphometry...")
        elif "writing outputs" in lowered:
            self._set_progress(True, "Creating output labelmaps...")

    def _run_plate_rod_morphometry(self):
        with slicer.util.tryWithErrorDisplay("Failed to run plate/rod morphometry.", waitCursor=True):
            if self.logic._proc is not None:
                raise RuntimeError("Plate/rod morphometry is already running.")
            self.runButton.enabled = False
            self.installCoreButton.enabled = False
            try:
                self._set_progress(True, "Reading selected masks...")
                self._activePlateRodJob = self.logic.prepare_plate_rod_job(
                    self.boneSegmentationSelector.currentNode(),
                    self.trabecularMaskSelector.currentNode(),
                    bone_segment_id=self._selected_segment_id(self.boneSegmentSelector),
                    trabecular_segment_id=self._selected_segment_id(self.trabecularSegmentSelector),
                    common_region_node=self.commonRegionMaskSelector.currentNode(),
                    common_region_segment_id=self._selected_segment_id(self.commonRegionSegmentSelector),
                    slenderness=int(self.slendernessSpinBox.value),
                    max_iterations=int(self.maxIterationsSpinBox.value),
                    min_plate_voxels=int(self.minPlateSpinBox.value),
                    min_rod_voxels=int(self.minRodSpinBox.value),
                    use_metal=bool(self.useMetalCheckBox.checked),
                    output_prefix=self.outputPrefixEdit.text,
                )
                self._set_progress(True, "Running thinning and morphometry...")
                self.logic.run_plate_rod_job(
                    self._activePlateRodJob,
                    on_output=self._show_process_output,
                    on_finished=self._on_plate_rod_process_finished,
                )
            except Exception:
                self.logic.cleanup_plate_rod_job(self._activePlateRodJob)
                self._activePlateRodJob = None
                self.runButton.enabled = True
                self.installCoreButton.enabled = True
                self._set_progress(False, "Plate/rod morphometry stopped.")
                raise

    def _on_plate_rod_process_finished(self, exit_code, exit_status):
        try:
            if int(exit_code) != 0:
                self._set_progress(False, f"Plate/rod morphometry failed with exit code {int(exit_code)}.")
                slicer.util.errorDisplay("Plate/rod morphometry failed. Check the Slicer log for process output.")
                return
            self._set_progress(True, "Creating output labelmaps...")
            result = self.logic.load_plate_rod_job_outputs(self._activePlateRodJob)
            self._lastTable = result["table"]
            slicer.util.selectModule("Tables")
            self._set_progress(False, "Plate/rod morphometry complete.")
        finally:
            self.logic.cleanup_plate_rod_job(self._activePlateRodJob)
            self._activePlateRodJob = None
            self.runButton.enabled = True
            self.installCoreButton.enabled = True


class PlateRodMorphometryHRpQCTTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("Plate/Rod Morphometry module smoke test passed.")
