from __future__ import annotations

import csv
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
MICROARCHITECTURE_LOCAL_REPO = TOOLBOX_ROOT.parent / "bone-microarchitecture"
MICROARCHITECTURE_LOCAL_SRC = MICROARCHITECTURE_LOCAL_REPO / "src"
if MICROARCHITECTURE_LOCAL_SRC.exists() and str(MICROARCHITECTURE_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(MICROARCHITECTURE_LOCAL_SRC))
TIMELAPSED_LOCAL_SRC = TOOLBOX_ROOT.parent / "TimelapsedHRpQCT" / "src"
if TIMELAPSED_LOCAL_SRC.exists() and str(TIMELAPSED_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(TIMELAPSED_LOCAL_SRC))
SEGMENTATION_MODULE_DIR = TOOLBOX_ROOT / "HRpQCTTools" / "SegmentationHRpQCT"
if str(SEGMENTATION_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_MODULE_DIR))
SCANCO_IO_DIR = TOOLBOX_ROOT / "IOTools" / "ScancoIO"
if str(SCANCO_IO_DIR) not in sys.path:
    sys.path.insert(0, str(SCANCO_IO_DIR))

from SlicerBoneImagingToolboxLib.slicer_pip import slicer_pip_install  # noqa: E402
from SlicerBoneImagingToolboxLib.segmentation_methods import (  # noqa: E402
    BONE_SEGMENTATION_METHODS,
    ENDOSTEAL_CONTOUR_METHODS,
    PERIOSTEAL_CONTOUR_METHODS,
)
from SlicerBoneImagingToolboxLib.slicer_update_ui import run_toolbox_update_dialog  # noqa: E402

from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


MODULE_VERSION = "0.1.0"
REGISTERED_MICROARCHITECTURE_DIR_NAME = "RegisteredMicroarchitecture"
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


class BoneMicroarchitecture(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Bone Microarchitecture"
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


class BoneMicroarchitectureLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._proc = None

    def is_registered_series_running(self):
        return self._proc is not None

    def run_registered_series_job(self, job_path, on_output=None, on_finished=None):
        if self._proc is not None:
            raise RuntimeError("A registered microarchitecture process is already running")

        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.MergedChannels)
        env = qt.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        for key in ("ITK_AUTOLOAD_PATH", "SITK_AUTOLOAD_PATH"):
            if env.contains(key):
                env.remove(key)
            env.insert(key, "")
        proc.setProcessEnvironment(env)

        def _read_output():
            raw = proc.readAll()
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
            text = data.decode("utf-8", errors="replace")
            if on_output and text:
                filtered_lines = []
                for line in text.splitlines(keepends=True):
                    if "MRMLIDImageIO" in line:
                        continue
                    if "ImageIO factory did not return an ImageIOBase" in line:
                        continue
                    filtered_lines.append(line)
                filtered = "".join(filtered_lines)
                if filtered:
                    on_output(filtered)

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

        proc.readyRead.connect(_read_output)
        proc.finished.connect(_finished)

        executable_dir = Path(sys.executable).resolve().parent if sys.executable else None
        sibling_python_slicer = executable_dir / "PythonSlicer" if executable_dir else None
        python_exe = (
            shutil.which("PythonSlicer")
            or (str(sibling_python_slicer) if sibling_python_slicer and sibling_python_slicer.exists() else "")
            or sys.executable
            or shutil.which("python3")
        )
        if not python_exe:
            raise RuntimeError("Could not find Python executable in Slicer environment")

        script_path = Path(__file__).resolve()
        args = [str(script_path), "--registered-series-job", str(job_path)]
        if on_output:
            on_output(f"[registered-process] launching: {python_exe} {' '.join(args)}\n")
        proc.start(python_exe, args)
        if not proc.waitForStarted(3000):
            raise RuntimeError("Failed to start registered microarchitecture process")
        self._proc = proc
        if on_output:
            try:
                on_output(f"[registered-process] started (pid={int(proc.processId())})\n")
            except Exception:
                on_output("[registered-process] started\n")

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

    def timelapsed_runtime_status(self):
        try:
            importlib.import_module("timelapsedhrpqct.dataset.discovery")
            importlib.import_module("timelapsedhrpqct.config.models")
        except Exception as exc:
            return False, f"Timelapsed HR-pQCT discovery is not available: {exc}"
        return True, "Timelapsed HR-pQCT discovery is available."

    def registered_microarchitecture_root(self, dataset_root, output_root=""):
        output_text = str(output_root or "").strip()
        if output_text:
            return Path(output_text).expanduser()
        dataset_root = Path(str(dataset_root)).expanduser()
        if dataset_root.name == REGISTERED_MICROARCHITECTURE_DIR_NAME:
            return dataset_root
        return dataset_root / REGISTERED_MICROARCHITECTURE_DIR_NAME

    def registered_subject_site_dir(self, output_root, subject_id, site):
        return Path(output_root) / f"sub-{subject_id}" / f"site-{site or 'unknown'}"

    def registered_session_output_dir(self, output_root, row):
        return (
            self.registered_subject_site_dir(output_root, row["subject_id"], row["site"])
            / "native_space"
            / f"ses-{row['session_id']}"
            / "microarchitecture"
        )

    def registered_session_masks_dir(self, output_root, row):
        return (
            self.registered_subject_site_dir(output_root, row["subject_id"], row["site"])
            / "native_space"
            / f"ses-{row['session_id']}"
            / "masks"
        )

    def registered_session_mask_path(self, output_root, row, role):
        site = row.get("site") or "unknown"
        filename = f"sub-{row['subject_id']}_ses-{row['session_id']}_site-{site}_mask-{role}.nii.gz"
        return self.registered_session_masks_dir(output_root, row) / filename

    def discover_registered_series(self, dataset_root, *, subject_filter="", site_filter=""):
        from timelapsedhrpqct.config.models import DiscoveryConfig
        from timelapsedhrpqct.dataset.discovery import discover_raw_sessions

        from timelapsedhrpqct.utils.session_ids import session_sort_key

        sessions = discover_raw_sessions(
            Path(str(dataset_root)).expanduser(),
            DiscoveryConfig(),
            canonicalize_sessions=True,
        )
        subject_filter = str(subject_filter or "").strip()
        site_filter = str(site_filter or "").strip().lower()
        rows = []
        for session in sorted(
            sessions,
            key=lambda item: (item.subject_id, item.site or "", session_sort_key(item.session_id)),
        ):
            if subject_filter and session.subject_id != subject_filter:
                continue
            if site_filter and str(session.site or "").lower() != site_filter:
                continue
            masks = dict(session.raw_mask_paths or {})
            row = {
                "subject_id": str(session.subject_id),
                "site": str(session.site or ""),
                "session_id": str(session.session_id),
                "image_path": str(session.raw_image_path),
                "seg_path": str(session.raw_seg_path or ""),
                "full_path": str(masks.get("full") or ""),
                "trab_path": str(masks.get("trab") or ""),
                "cort_path": str(masks.get("cort") or ""),
                "stack_index": int(session.stack_index or 1),
            }
            missing = [
                label
                for label, value in (
                    ("image", row["image_path"]),
                    ("bone seg", row["seg_path"]),
                    ("full", row["full_path"]),
                    ("trab", row["trab_path"]),
                    ("cort", row["cort_path"]),
                )
                if not value
            ]
            row["status"] = "Ready" if not missing else f"Missing {', '.join(missing)}"
            rows.append(row)
        return rows

    def sequential_registration_pairs(self, rows):
        try:
            from timelapsedhrpqct.utils.session_ids import session_sort_key
        except Exception:
            session_sort_key = lambda value: str(value)

        groups = {}
        for row in rows:
            groups.setdefault((row["subject_id"], row["site"], row.get("stack_index", 1)), []).append(row)
        pairs = []
        for (subject_id, site, stack_index), group_rows in sorted(groups.items()):
            ordered = sorted(group_rows, key=lambda row: session_sort_key(row["session_id"]))
            for fixed, moving in zip(ordered, ordered[1:]):
                pairs.append(
                    {
                        "subject_id": subject_id,
                        "site": site,
                        "stack_index": stack_index,
                        "fixed_session": fixed["session_id"],
                        "moving_session": moving["session_id"],
                    }
                )
        return pairs

    def write_registered_series_manifest(self, dataset_root, output_root, rows):
        root = self.registered_microarchitecture_root(dataset_root, output_root)
        root.mkdir(parents=True, exist_ok=True)
        for row in rows:
            subject_site_dir = self.registered_subject_site_dir(root, row["subject_id"], row["site"])
            for relative_dir in (
                "registration/adjacent",
                "registration/composed",
                "common_space/masks_from_each_session",
                "common_space/common_masks",
                f"native_space/ses-{row['session_id']}/masks",
                f"native_space/ses-{row['session_id']}/microarchitecture/maps",
            ):
                (subject_site_dir / relative_dir).mkdir(parents=True, exist_ok=True)
        manifest = {
            "workflow": REGISTERED_MICROARCHITECTURE_DIR_NAME,
            "dataset_root": str(Path(str(dataset_root)).expanduser()),
            "output_root": str(root),
            "registration_strategy": "sequential_adjacent_then_composed",
            "measurement_space": "native_image_space_common_region",
            "sessions": rows,
            "sequential_registration_pairs": self.sequential_registration_pairs(rows),
        }
        manifest_path = root / "registered_microarchitecture_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def prepare_registered_series_workspace(
        self,
        dataset_root,
        output_root,
        rows,
        *,
        segmentation_method="seg_gauss",
        periosteal_contour_method="standard",
        endosteal_contour_method="standard",
        progress_callback=None,
    ):
        root = self.registered_microarchitecture_root(dataset_root, output_root)
        self.write_registered_series_manifest(dataset_root, root, rows)
        prepared_rows = []
        generated = []
        self._registered_progress(progress_callback, f"[registered] preparing {len(rows)} session(s)")
        for row in rows:
            self._registered_progress(
                progress_callback,
                f"[registered] preparing sub-{row['subject_id']} ses-{row['session_id']}",
            )
            prepared = dict(row)
            generated.extend(
                self._complete_registered_series_masks(
                    root,
                    prepared,
                    segmentation_method=segmentation_method,
                    periosteal_contour_method=periosteal_contour_method,
                    endosteal_contour_method=endosteal_contour_method,
                )
            )
            missing = self._registered_row_missing(prepared)
            prepared["status"] = "Ready" if not missing else f"Missing {', '.join(missing)}"
            prepared_rows.append(prepared)
        self._mark_incomplete_registered_groups(prepared_rows)
        generated.extend(self._build_registered_common_regions(root, prepared_rows, progress_callback=progress_callback))
        manifest_path = self.write_registered_series_manifest(dataset_root, root, prepared_rows)
        return {"manifest": str(manifest_path), "rows": prepared_rows, "generated": generated}

    def _registered_progress(self, progress_callback, message):
        if progress_callback is not None:
            progress_callback(str(message))

    def _registered_row_missing(self, row):
        return [
            label
            for label, value in (
                ("image", row.get("image_path")),
                ("bone seg", row.get("seg_path")),
                ("full", row.get("full_path")),
                ("trab", row.get("trab_path")),
                ("cort", row.get("cort_path")),
            )
            if not value or not Path(str(value)).expanduser().exists()
        ]

    def _write_registered_mask(self, output_root, row, role, image):
        path = self.registered_session_mask_path(output_root, row, role)
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.Cast(image > 0, sitk.sitkUInt8), str(path))
        row[self._registered_mask_key(role)] = str(path)
        return str(path)

    def _registered_mask_key(self, role):
        return f"{role}_path" if role != "seg" else "seg_path"

    def _read_optional_registered_mask(self, row, key, role):
        path = str(row.get(key) or "").strip()
        if not path:
            return None
        if not Path(path).expanduser().exists():
            return None
        return self._read_registered_series_image(path, role=role)

    def _complete_registered_series_masks(
        self,
        output_root,
        row,
        *,
        segmentation_method,
        periosteal_contour_method,
        endosteal_contour_method,
    ):
        generated = []
        if not row.get("image_path"):
            return generated
        image = self._read_registered_series_image(row["image_path"], role="image")

        resolved_compartments, provenance = self._resolve_registered_compartment_masks(row, image)
        for role, mask in resolved_compartments.items():
            key = self._registered_mask_key(role)
            original = self._read_optional_registered_mask(row, key, role)
            source = provenance.get(role, "provided")
            needs_workspace_copy = (
                source.startswith("derived_from_")
                or original is None
                or not self._same_image_geometry(original, image)
            )
            if needs_workspace_copy:
                path = self._write_registered_mask(output_root, row, role, mask)
                generated.append({"session": row["session_id"], "role": role, "path": path, "source": source})

        missing_compartments = [role for role in ("full", "trab", "cort") if role in self._registered_row_missing(row)]
        if missing_compartments and periosteal_contour_method != "none":
            contour_masks = self._generate_registered_contours(
                image,
                row,
                segmentation_method=segmentation_method,
                periosteal_contour_method=periosteal_contour_method,
                endosteal_contour_method=endosteal_contour_method,
            )
            for role in missing_compartments:
                if role in contour_masks:
                    path = self._write_registered_mask(output_root, row, role, contour_masks[role])
                    generated.append({"session": row["session_id"], "role": role, "path": path, "source": "generated"})

        if "bone seg" in self._registered_row_missing(row) and segmentation_method != "none":
            full = self._read_optional_registered_mask(row, "full_path", "full")
            trab = self._read_optional_registered_mask(row, "trab_path", "trab")
            cort = self._read_optional_registered_mask(row, "cort_path", "cort")
            if full is not None and trab is not None and cort is not None:
                seg = self._generate_registered_bone_segmentation(
                    image,
                    full,
                    trab,
                    cort,
                    segmentation_method=segmentation_method,
                )
                path = self._write_registered_mask(output_root, row, "seg", seg)
                generated.append({"session": row["session_id"], "role": "seg", "path": path, "source": "generated"})
        return generated

    def _same_image_geometry(self, first, second):
        return (
            first.GetSize() == second.GetSize()
            and first.GetSpacing() == second.GetSpacing()
            and first.GetOrigin() == second.GetOrigin()
            and first.GetDirection() == second.GetDirection()
        )

    def _resolve_registered_compartment_masks(self, row, image):
        from timelapsedhrpqct.processing.masks import resolve_masks

        provided = {
            role: mask
            for role, mask in {
                "full": self._read_optional_registered_mask(row, "full_path", "full"),
                "trab": self._read_optional_registered_mask(row, "trab_path", "trab"),
                "cort": self._read_optional_registered_mask(row, "cort_path", "cort"),
            }.items()
            if mask is not None
        }
        if len(provided) < 2:
            return {}, {}
        resolved, provenance = resolve_masks(image=image, provided_masks=provided, desired_roles=["full", "trab", "cort"])
        return resolved, provenance

    def _derive_registered_compartment_masks(self, row, image):
        resolved, provenance = self._resolve_registered_compartment_masks(row, image)
        return {
            role: mask
            for role, mask in resolved.items()
            if provenance.get(role, "").startswith("derived_from_")
        }

    def _segmentation_defaults(self, segmentation_method):
        from SegmentationHRpQCT import METHOD_PRESETS

        defaults = dict(METHOD_PRESETS.get(str(segmentation_method), METHOD_PRESETS["seg_gauss"]))
        defaults["method"] = "seg_gauss" if str(segmentation_method) == "none" else str(segmentation_method)
        defaults["enabled"] = str(segmentation_method) != "none"
        return defaults

    def _site_defaults(self, row):
        from SegmentationHRpQCT import SITE_PRESETS

        site = str(row.get("site") or "").lower()
        if site not in SITE_PRESETS:
            site = "radius"
        return site, SITE_PRESETS[site]

    def _generate_registered_contours(
        self,
        image,
        row,
        *,
        segmentation_method,
        periosteal_contour_method,
        endosteal_contour_method,
    ):
        from timelapsedhrpqct.processing.contour_generation import (
            ContourGenerationParams,
            InnerContourParams,
            OuterContourParams,
            SegmentationParams,
            generate_masks_from_image,
        )

        site, site_defaults = self._site_defaults(row)
        inner = dict(site_defaults["inner"])
        inner["site"] = site
        inner["contour_method"] = str(endosteal_contour_method)
        outer = dict(site_defaults["outer"])
        outer["contour_method"] = str(periosteal_contour_method)
        params = ContourGenerationParams(
            outer=OuterContourParams(**outer),
            inner=InnerContourParams(**inner),
            segmentation=SegmentationParams(**self._segmentation_defaults(segmentation_method)),
        )
        generated = generate_masks_from_image(image, params, verbose=False)
        return {"full": generated.full, "trab": generated.trab, "cort": generated.cort, "seg": generated.seg}

    def _generate_registered_bone_segmentation(self, image, full_mask, trab_mask, cort_mask, *, segmentation_method):
        from timelapsedhrpqct.processing.contour_generation import (
            SegmentationParams,
            _segment_bone_xyz,
            numpy_xyz_to_sitk_binary,
            sitk_to_numpy_xyz,
        )

        params = SegmentationParams(**self._segmentation_defaults(segmentation_method))
        seg_xyz = _segment_bone_xyz(
            image_xyz=sitk_to_numpy_xyz(image),
            full_mask_xyz=sitk_to_numpy_xyz(full_mask) > 0,
            trab_mask_xyz=sitk_to_numpy_xyz(trab_mask) > 0,
            cort_mask_xyz=sitk_to_numpy_xyz(cort_mask) > 0,
            params=params,
            spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        )
        return numpy_xyz_to_sitk_binary(seg_xyz, image)

    def _registered_group_key(self, row):
        return (row["subject_id"], row["site"], row.get("stack_index", 1))

    def _mark_incomplete_registered_groups(self, rows):
        groups = {}
        for row in rows:
            groups.setdefault(self._registered_group_key(row), []).append(row)
        for group_rows in groups.values():
            if len(group_rows) < 2:
                continue
            if any(row.get("status") != "Ready" for row in group_rows):
                for row in group_rows:
                    if row.get("status") == "Ready":
                        row["status"] = "Missing common region (group has incomplete timepoints)"

    def _registered_common_mask_path(self, output_root, row, role):
        subject_site_dir = self.registered_subject_site_dir(output_root, row["subject_id"], row["site"])
        return subject_site_dir / "common_space" / "common_masks" / (
            f"sub-{row['subject_id']}_site-{row['site']}_stack-{int(row.get('stack_index', 1)):02d}_mask-{role}_common.nii.gz"
        )

    def _registered_common_session_mask_path(self, output_root, row, role):
        return self.registered_session_masks_dir(output_root, row) / (
            f"sub-{row['subject_id']}_ses-{row['session_id']}_site-{row['site']}_mask-{role}_native_common.nii.gz"
        )

    def _registered_transform_path(self, output_root, row, baseline_row):
        subject_site_dir = self.registered_subject_site_dir(output_root, row["subject_id"], row["site"])
        return subject_site_dir / "registration" / "composed" / (
            f"sub-{row['subject_id']}_site-{row['site']}_stack-{int(row.get('stack_index', 1)):02d}_"
            f"from-ses-{row['session_id']}_to-ses-{baseline_row['session_id']}.tfm"
        )

    def _registered_pairwise_transform_path(self, output_root, moving_row, fixed_row):
        subject_site_dir = self.registered_subject_site_dir(output_root, moving_row["subject_id"], moving_row["site"])
        return subject_site_dir / "registration" / "pairwise" / (
            f"sub-{moving_row['subject_id']}_site-{moving_row['site']}_"
            f"stack-{int(moving_row.get('stack_index', 1)):02d}_"
            f"from-ses-{moving_row['session_id']}_to-ses-{fixed_row['session_id']}.tfm"
        )

    def _resample_registered_mask(self, mask, reference, transform):
        return sitk.Cast(
            sitk.Resample(
                sitk.Cast(mask > 0, sitk.sitkUInt8),
                reference,
                transform,
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
            > 0,
            sitk.sitkUInt8,
        )

    def _register_to_baseline(self, fixed_image, moving_image, fixed_mask, moving_mask):
        from timelapsedhrpqct.processing.registration import RegistrationSettings, register_images

        return register_images(
            fixed_image=sitk.Cast(fixed_image, sitk.sitkFloat32),
            moving_image=sitk.Cast(moving_image, sitk.sitkFloat32),
            fixed_mask=sitk.Cast(fixed_mask > 0, sitk.sitkUInt8),
            moving_mask=sitk.Cast(moving_mask > 0, sitk.sitkUInt8),
            settings=RegistrationSettings(),
        )

    def _build_registered_common_regions(self, output_root, rows, *, progress_callback=None):
        from timelapsedhrpqct.utils.session_ids import session_sort_key
        from timelapsedhrpqct.processing.transform_chain import (
            PairwiseTransform,
            compose_sequential_to_baseline,
            flatten_transform,
        )

        generated = []
        groups = {}
        for row in rows:
            if row.get("status") == "Ready":
                groups.setdefault(self._registered_group_key(row), []).append(row)
        for _group_key, group_rows in groups.items():
            if len(group_rows) < 2:
                continue
            ordered = sorted(group_rows, key=lambda row: session_sort_key(row["session_id"]))
            baseline = ordered[0]
            self._registered_progress(
                progress_callback,
                f"[registered] building common region for sub-{baseline['subject_id']} "
                f"site-{baseline['site']} stack-{int(baseline.get('stack_index', 1)):02d} "
                f"from {len(ordered)} timepoint(s)",
            )
            baseline_image = self._read_registered_series_image(baseline["image_path"], role="image")
            pairwise = []
            previous = baseline
            previous_image = baseline_image
            previous_full = self._read_registered_series_image(previous["full_path"], role="full")
            for row in ordered[1:]:
                self._registered_progress(
                    progress_callback,
                    f"[registered] registering sub-{row['subject_id']} ses-{row['session_id']} "
                    f"to ses-{previous['session_id']}",
                )
                image = self._read_registered_series_image(row["image_path"], role="image")
                full = self._read_registered_series_image(row["full_path"], role="full")
                result = self._register_to_baseline(
                    fixed_image=previous_image,
                    moving_image=image,
                    fixed_mask=previous_full,
                    moving_mask=full,
                )
                pairwise_transform = flatten_transform(result.transform)
                pairwise_path = self._registered_pairwise_transform_path(output_root, row, previous)
                pairwise_path.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteTransform(pairwise_transform, str(pairwise_path))
                generated.append(
                    {
                        "session": row["session_id"],
                        "role": "transform_pairwise",
                        "path": str(pairwise_path),
                        "source": "registration",
                    }
                )
                pairwise.append(PairwiseTransform(session_id=row["session_id"], transform=pairwise_transform))
                previous = row
                previous_image = image
                previous_full = full

            identity = sitk.Transform(3, sitk.sitkIdentity)
            baseline_transforms = compose_sequential_to_baseline(
                pairwise_transforms=pairwise,
                baseline_session_id=baseline["session_id"],
                dimension=3,
            )
            transforms = {item.session_id: item.transform for item in baseline_transforms}
            transforms[baseline["session_id"]] = transforms.get(baseline["session_id"], identity)
            for row in ordered:
                transform_path = self._registered_transform_path(output_root, row, baseline)
                transform_path.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteTransform(flatten_transform(transforms[row["session_id"]]), str(transform_path))
                generated.append(
                    {
                        "session": row["session_id"],
                        "role": "transform_composed",
                        "path": str(transform_path),
                        "source": "registration",
                    }
                )

            common_by_role = {}
            for row in ordered:
                self._registered_progress(
                    progress_callback,
                    f"[registered] resampling masks for sub-{row['subject_id']} ses-{row['session_id']} into common space",
                )
                transform = transforms[row["session_id"]]
                for role in ("full", "trab", "cort"):
                    mask = self._read_registered_series_image(row[f"{role}_path"], role=role)
                    mask_common = self._resample_registered_mask(mask, baseline_image, transform)
                    if role not in common_by_role:
                        common_by_role[role] = sitk.Cast(mask_common > 0, sitk.sitkUInt8)
                    else:
                        common_by_role[role] = sitk.Cast((common_by_role[role] > 0) & (mask_common > 0), sitk.sitkUInt8)

            for role, common_mask in common_by_role.items():
                self._registered_progress(progress_callback, f"[registered] writing {role} common-space mask")
                common_path = self._registered_common_mask_path(output_root, baseline, role)
                common_path.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(common_mask, str(common_path))
                generated.append({"session": baseline["session_id"], "role": f"{role}_common", "path": str(common_path), "source": "common_space"})

            for row in ordered:
                self._registered_progress(
                    progress_callback,
                    f"[registered] returning common masks to native space for sub-{row['subject_id']} ses-{row['session_id']}",
                )
                image = self._read_registered_series_image(row["image_path"], role="image")
                transform = transforms[row["session_id"]]
                inverse = transform.GetInverse()
                common_paths = {}
                for role, common_mask in common_by_role.items():
                    native_common = self._resample_registered_mask(common_mask, image, inverse)
                    path = self._registered_common_session_mask_path(output_root, row, role)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sitk.WriteImage(native_common, str(path))
                    common_paths[role] = str(path)
                    generated.append({"session": row["session_id"], "role": f"{role}_native_common", "path": str(path), "source": "native_common"})
                row["full_path"] = common_paths["full"]
                row["trab_path"] = common_paths["trab"]
                row["cort_path"] = common_paths["cort"]
                row["measurement_space"] = "native_image_space_common_region"
        return generated

    def _read_registered_series_image(self, path, *, role):
        path = Path(str(path)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{role} file does not exist: {path}")
        if ".aim" in path.name.lower():
            from ScancoIOLib import aim_io

            scaling = "density" if role == "image" else "native"
            image, _metadata = aim_io.read_image(path, scaling=scaling)
            if role != "image":
                image = sitk.Cast(image > 0, sitk.sitkUInt8)
            return image
        pixel_type = sitk.sitkFloat32 if role == "image" else sitk.sitkUInt8
        image = sitk.ReadImage(str(path), pixel_type)
        if role != "image":
            image = sitk.Cast(image > 0, sitk.sitkUInt8)
        return image

    def run_registered_series_microarchitecture(
        self,
        dataset_root,
        output_root,
        rows,
        *,
        thickness_method="hildebrand",
        thickness_backend="auto",
    ):
        from bone_microarchitecture import compute_microarchitecture
        from bone_microarchitecture.results import SUMMARY_COLUMNS, measurement_rows, write_measurement_csv

        root = self.registered_microarchitecture_root(dataset_root, output_root)
        self.write_registered_series_manifest(dataset_root, root, rows)
        long_rows = []
        written = []
        skipped_rows = []
        for row in rows:
            if row.get("status") != "Ready":
                skipped_rows.append(
                    {
                        "subject_id": row.get("subject_id", ""),
                        "site": row.get("site", ""),
                        "session_id": row.get("session_id", ""),
                        "status": row.get("status", "Not ready"),
                    }
                )
                continue
            image = self._read_registered_series_image(row["image_path"], role="image")
            bone_seg = self._read_registered_series_image(row["seg_path"], role="bone seg")
            full_mask = self._read_registered_series_image(row["full_path"], role="full")
            trab_mask = self._read_registered_series_image(row["trab_path"], role="trab")
            cort_mask = self._read_registered_series_image(row["cort_path"], role="cort")
            sizes = {item.GetSize() for item in (image, bone_seg, full_mask, trab_mask, cort_mask)}
            if len(sizes) != 1:
                raise ValueError(
                    f"Registered series inputs for sub-{row['subject_id']} ses-{row['session_id']} "
                    "must have matching sizes."
                )

            result = compute_microarchitecture(
                bone_mask=sitk.GetArrayFromImage(bone_seg),
                periosteal_mask=sitk.GetArrayFromImage(full_mask),
                trabecular_mask=sitk.GetArrayFromImage(trab_mask),
                cortical_mask=sitk.GetArrayFromImage(cort_mask),
                grayscale=sitk.GetArrayFromImage(image),
                spacing=tuple(reversed(tuple(trab_mask.GetSpacing()))),
                thickness_method=str(thickness_method),
                thickness_backend=str(thickness_backend),
            )
            session_dir = self.registered_session_output_dir(root, row)
            maps_dir = session_dir / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            csv_path = session_dir / "measurements.csv"
            write_measurement_csv(csv_path, result.measurements, result.maps)
            for map_role, array in result.maps.items():
                map_image = self._array_to_sitk_like(array, trab_mask)
                sitk.WriteImage(map_image, str(maps_dir / f"{map_role.replace('.', '')}.nii.gz"))
            for summary_row in measurement_rows(result.measurements, result.maps):
                long_row = {
                    "Subject": row["subject_id"],
                    "Site": row["site"],
                    "Session": row["session_id"],
                }
                long_row.update(summary_row)
                long_rows.append(long_row)
            written.append(str(csv_path))

        if not written:
            skipped = len(rows)
            raise RuntimeError(
                "No registered series measurements were run. "
                f"{skipped} session(s) were skipped because required image, segmentation, full, trabecular, "
                "or cortical masks were missing. Run discovery, generate the missing masks, then run measurements."
            )

        long_path = root / "microarchitecture_long.csv"
        with long_path.open("w", newline="", encoding="utf-8") as stream:
            fieldnames = ["Subject", "Site", "Session", *SUMMARY_COLUMNS]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(long_rows)
        return {
            "manifest": str(root / "registered_microarchitecture_manifest.json"),
            "long_csv": str(long_path),
            "session_csvs": written,
            "skipped_rows": skipped_rows,
        }

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


class BoneMicroarchitectureWidget(ScriptedLoadableModuleWidget):
    def _tip(self, widget, text):
        widget.toolTip = str(text)
        return widget

    def setup(self):
        super().setup()
        self.logic = BoneMicroarchitectureLogic()
        self._lastMetrics = None
        self._lastMaps = None
        self._lastTableNode = None
        self._lastRegisteredRows = []
        self._allRegisteredRows = []

        self.modeTabs = qt.QTabWidget()
        self.layout.addWidget(self.modeTabs)

        single_tab = qt.QWidget()
        single_layout = qt.QVBoxLayout(single_tab)

        box = ctk.ctkCollapsibleButton()
        box.text = "Bone Microarchitecture"
        single_layout.addWidget(box)
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
        single_layout.addWidget(calibration)
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
        single_layout.addWidget(thickness_settings)
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
        single_layout.addWidget(self.logText)
        single_layout.addStretch(1)

        series_tab = qt.QWidget()
        series_layout = qt.QVBoxLayout(series_tab)
        self._setup_registered_series_tab(series_layout)

        self.modeTabs.addTab(single_tab, "Single Scan")
        self.modeTabs.addTab(series_tab, "Registered Series")
        self.layout.addStretch(1)
        self._update_dependency_ui()

    def _setup_registered_series_tab(self, layout):
        box = ctk.ctkCollapsibleButton()
        box.text = "RegisteredMicroarchitecture"
        layout.addWidget(box)
        form = qt.QFormLayout(box)

        self.seriesDatasetRootEdit = qt.QLineEdit()
        self.seriesOutputRootEdit = qt.QLineEdit()
        self.seriesSubjectCombo = qt.QComboBox()
        self.seriesSiteCombo = qt.QComboBox()
        self.seriesSubjectCombo.addItem("All subjects", "")
        self.seriesSiteCombo.addItem("All sites", "")
        self._tip(
            self.seriesDatasetRootEdit,
            "Dataset root discovered with Timelapsed HR-pQCT filename and header conventions.",
        )
        self._tip(
            self.seriesOutputRootEdit,
            "Output folder. Defaults to RegisteredMicroarchitecture under the dataset root.",
        )
        self.seriesDatasetRootEdit.textChanged.connect(self._update_registered_output_default)

        dataset_row = qt.QWidget()
        dataset_layout = qt.QHBoxLayout(dataset_row)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.addWidget(self.seriesDatasetRootEdit, 1)
        browse_dataset = qt.QPushButton("Browse")
        browse_dataset.clicked.connect(self._browse_series_dataset_root)
        dataset_layout.addWidget(browse_dataset)
        form.addRow("Dataset root", dataset_row)

        output_row = qt.QWidget()
        output_layout = qt.QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.seriesOutputRootEdit, 1)
        browse_output = qt.QPushButton("Browse")
        browse_output.clicked.connect(self._browse_series_output_root)
        output_layout.addWidget(browse_output)
        form.addRow("Output root", output_row)

        self.discoverSeriesButton = qt.QPushButton("Discover series")
        self.discoverSeriesButton.clicked.connect(self._discover_registered_series)
        form.addRow(self.discoverSeriesButton)

        self.seriesSubjectCombo.currentIndexChanged.connect(self._refresh_registered_series_table)
        self.seriesSiteCombo.currentIndexChanged.connect(self._refresh_registered_series_table)
        form.addRow("Subject", self.seriesSubjectCombo)
        form.addRow("Site", self.seriesSiteCombo)

        missing_masks_row = qt.QWidget()
        missing_masks_layout = qt.QHBoxLayout(missing_masks_row)
        missing_masks_layout.setContentsMargins(0, 0, 0, 0)
        self.seriesSegmentationMethodCombo = qt.QComboBox()
        for value, descriptor in BONE_SEGMENTATION_METHODS.items():
            self.seriesSegmentationMethodCombo.addItem(descriptor.label, value)
        self.seriesPeriostealContourCombo = qt.QComboBox()
        for value, descriptor in PERIOSTEAL_CONTOUR_METHODS.items():
            self.seriesPeriostealContourCombo.addItem(descriptor.label, value)
        self.seriesEndostealContourCombo = qt.QComboBox()
        for value, descriptor in ENDOSTEAL_CONTOUR_METHODS.items():
            self.seriesEndostealContourCombo.addItem(descriptor.label, value)
        missing_masks_layout.addWidget(qt.QLabel("Bone"))
        missing_masks_layout.addWidget(self.seriesSegmentationMethodCombo)
        missing_masks_layout.addWidget(qt.QLabel("Peri"))
        missing_masks_layout.addWidget(self.seriesPeriostealContourCombo)
        missing_masks_layout.addWidget(qt.QLabel("Endo"))
        missing_masks_layout.addWidget(self.seriesEndostealContourCombo)
        self._tip(
            missing_masks_row,
            "Methods used by Prepare to create missing segmentation, full, trabecular, or cortical masks. "
            "Masks that can be derived from existing compartments are derived first.",
        )
        form.addRow("Missing masks", missing_masks_row)

        settings_row = qt.QWidget()
        settings_layout = qt.QHBoxLayout(settings_row)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self.seriesThicknessMethodCombo = qt.QComboBox()
        for label, value in [("Exact sphere fitting", "hildebrand"), ("Bounded EDT", "edt")]:
            self.seriesThicknessMethodCombo.addItem(label, value)
        self.seriesThicknessBackendCombo = qt.QComboBox()
        for label, value in [("Auto", "auto"), ("CPU", "cpu"), ("Apple MPS (macOS)", "mps"), ("OpenCL GPU", "opencl")]:
            self.seriesThicknessBackendCombo.addItem(label, value)
        settings_layout.addWidget(qt.QLabel("Method"))
        settings_layout.addWidget(self.seriesThicknessMethodCombo)
        settings_layout.addWidget(qt.QLabel("Backend"))
        settings_layout.addWidget(self.seriesThicknessBackendCombo)
        form.addRow("Thickness", settings_row)

        self.runRegisteredSeriesButton = qt.QPushButton("Run")
        self.runRegisteredSeriesButton.clicked.connect(self._run_registered_series)
        self._tip(
            self.runRegisteredSeriesButton,
            "Prepare the registered workspace, derive or generate missing masks when possible, then run measurements.",
        )
        form.addRow(self.runRegisteredSeriesButton)

        self.seriesStatusLabel = qt.QLabel()
        self.seriesStatusLabel.wordWrap = True
        form.addRow("Status", self.seriesStatusLabel)

        self.seriesTable = qt.QTableWidget()
        self.seriesTable.setColumnCount(9)
        self.seriesTable.setHorizontalHeaderLabels(
            ["Subject", "Site", "Session", "Image", "Bone seg", "Full", "Trab", "Cort", "Status"]
        )
        self.seriesTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.seriesTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.seriesTable.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.seriesTable)
        self.seriesLogText = qt.QTextEdit()
        self.seriesLogText.readOnly = True
        self.seriesLogText.minimumHeight = 120
        self.seriesLogText.placeholderText = "Registered series log"
        layout.addWidget(self.seriesLogText)
        layout.addStretch(1)

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

    def _series_log(self, message):
        self.seriesLogText.append(str(message).rstrip())
        self.seriesLogText.ensureCursorVisible()

    def _set_registered_series_running(self, running):
        self.runRegisteredSeriesButton.enabled = not running and self.logic.is_core_available()
        self.discoverSeriesButton.enabled = not running
        self.seriesDatasetRootEdit.enabled = not running
        self.seriesOutputRootEdit.enabled = not running
        self.seriesSubjectCombo.enabled = not running
        self.seriesSiteCombo.enabled = not running
        self.seriesSegmentationMethodCombo.enabled = not running
        self.seriesPeriostealContourCombo.enabled = not running
        self.seriesEndostealContourCombo.enabled = not running
        self.seriesThicknessMethodCombo.enabled = not running
        self.seriesThicknessBackendCombo.enabled = not running

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
        self.runRegisteredSeriesButton.enabled = available

    def _browse_series_dataset_root(self):
        path = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            "Select dataset root",
            self.seriesDatasetRootEdit.text,
        )
        if path:
            self.seriesDatasetRootEdit.text = str(path)

    def _browse_series_output_root(self):
        path = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            "Select RegisteredMicroarchitecture output root",
            self.seriesOutputRootEdit.text,
        )
        if path:
            self.seriesOutputRootEdit.text = str(path)

    def _update_registered_output_default(self):
        dataset_root = str(self.seriesDatasetRootEdit.text or "").strip()
        if not dataset_root:
            return
        current_output = str(self.seriesOutputRootEdit.text or "").strip()
        if current_output:
            return
        self.seriesOutputRootEdit.text = str(self.logic.registered_microarchitecture_root(dataset_root))

    def _discover_registered_series(self):
        dataset_root = str(self.seriesDatasetRootEdit.text or "").strip()
        if not dataset_root:
            slicer.util.errorDisplay("Select a dataset root before discovery.")
            return
        try:
            rows = self._with_wait_cursor(
                lambda: self.logic.discover_registered_series(dataset_root)
            )
        except Exception as exc:
            slicer.util.errorDisplay(f"Registered series discovery failed:\n{exc}")
            self._series_log(f"[registered] discovery failed: {exc}")
            return
        self._allRegisteredRows = list(rows)
        self._populate_registered_series_filters(rows)
        self._refresh_registered_series_table()
        rows = self._lastRegisteredRows
        ready = sum(1 for row in rows if row.get("status") == "Ready")
        self.seriesStatusLabel.text = (
            f"Discovered {len(self._allRegisteredRows)} session(s); showing {len(rows)}, {ready} ready."
        )
        self._series_log(
            f"[registered] discovered {len(self._allRegisteredRows)} session(s), showing {len(rows)}, {ready} ready."
        )

    def _populate_registered_series_filters(self, rows):
        current_subject = str(self.seriesSubjectCombo.currentData or "")
        current_site = str(self.seriesSiteCombo.currentData or "")
        self.seriesSubjectCombo.blockSignals(True)
        self.seriesSiteCombo.blockSignals(True)
        self.seriesSubjectCombo.clear()
        self.seriesSiteCombo.clear()
        self.seriesSubjectCombo.addItem("All subjects", "")
        self.seriesSiteCombo.addItem("All sites", "")
        for subject_id in sorted({str(row.get("subject_id") or "") for row in rows if row.get("subject_id")}):
            self.seriesSubjectCombo.addItem(f"sub-{subject_id}", subject_id)
        for site in sorted({str(row.get("site") or "") for row in rows if row.get("site")}):
            self.seriesSiteCombo.addItem(site, site)
        subject_index = self.seriesSubjectCombo.findData(current_subject)
        site_index = self.seriesSiteCombo.findData(current_site)
        self.seriesSubjectCombo.setCurrentIndex(subject_index if subject_index >= 0 else 0)
        self.seriesSiteCombo.setCurrentIndex(site_index if site_index >= 0 else 0)
        self.seriesSubjectCombo.blockSignals(False)
        self.seriesSiteCombo.blockSignals(False)

    def _filtered_registered_rows(self):
        subject = str(self.seriesSubjectCombo.currentData or "")
        site = str(self.seriesSiteCombo.currentData or "")
        rows = []
        for row in self._allRegisteredRows:
            if subject and str(row.get("subject_id") or "") != subject:
                continue
            if site and str(row.get("site") or "") != site:
                continue
            rows.append(row)
        return rows

    def _refresh_registered_series_table(self):
        self._lastRegisteredRows = self._filtered_registered_rows()
        self._populate_registered_series_table(self._lastRegisteredRows)
        if self._allRegisteredRows:
            ready = sum(1 for row in self._lastRegisteredRows if row.get("status") == "Ready")
            self.seriesStatusLabel.text = (
                f"Showing {len(self._lastRegisteredRows)} of {len(self._allRegisteredRows)} discovered session(s); "
                f"{ready} ready."
            )

    def _populate_registered_series_table(self, rows):
        self.seriesTable.setRowCount(len(rows))
        columns = [
            "subject_id",
            "site",
            "session_id",
            "image_path",
            "seg_path",
            "full_path",
            "trab_path",
            "cort_path",
            "status",
        ]
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(columns):
                value = str(row.get(key) or "")
                display_value = Path(value).name if key.endswith("_path") and value else value
                item = qt.QTableWidgetItem(display_value)
                item.setToolTip(value)
                self.seriesTable.setItem(row_index, column_index, item)
        self.seriesTable.resizeColumnsToContents()

    def _prepare_registered_series(self):
        if not self._lastRegisteredRows:
            self._discover_registered_series()
        if not self._lastRegisteredRows:
            return None
        try:
            prepared = self._with_wait_cursor(
                lambda: self.logic.prepare_registered_series_workspace(
                    self.seriesDatasetRootEdit.text,
                    self.seriesOutputRootEdit.text,
                    self._lastRegisteredRows,
                    segmentation_method=str(self.seriesSegmentationMethodCombo.currentData),
                    periosteal_contour_method=str(self.seriesPeriostealContourCombo.currentData),
                    endosteal_contour_method=str(self.seriesEndostealContourCombo.currentData),
                )
            )
        except Exception as exc:
            slicer.util.errorDisplay(f"RegisteredMicroarchitecture workspace preparation failed:\n{exc}")
            self._series_log(f"[registered] workspace preparation failed: {exc}")
            return None
        self._allRegisteredRows = list(prepared["rows"])
        self._populate_registered_series_filters(self._allRegisteredRows)
        self._refresh_registered_series_table()
        generated_count = len(prepared.get("generated", []))
        self.seriesStatusLabel.text = (
            f"Prepared RegisteredMicroarchitecture workspace: {prepared['manifest']} "
            f"({generated_count} mask file(s) generated or derived)."
        )
        self._series_log(f"[registered] wrote manifest: {prepared['manifest']}")
        for item in prepared.get("generated", []):
            self._series_log(f"[registered] {item['source']} {item['role']}: {item['path']}")
        return prepared

    def _write_registered_series_job(self):
        dataset_root = str(self.seriesDatasetRootEdit.text or "").strip()
        if not dataset_root:
            raise ValueError("Select a dataset root before running registered microarchitecture.")
        output_root = str(self.seriesOutputRootEdit.text or "").strip()
        root = self.logic.registered_microarchitecture_root(dataset_root, output_root)
        job_dir = root / "slicer_run_configs"
        job_dir.mkdir(parents=True, exist_ok=True)
        fd, job_path = tempfile.mkstemp(prefix="registered_microarchitecture_", suffix=".json", dir=str(job_dir))
        os.close(fd)
        result_path = Path(job_path).with_suffix(".result.json")
        job = {
            "dataset_root": dataset_root,
            "output_root": output_root,
            "rows": self._lastRegisteredRows,
            "segmentation_method": str(self.seriesSegmentationMethodCombo.currentData),
            "periosteal_contour_method": str(self.seriesPeriostealContourCombo.currentData),
            "endosteal_contour_method": str(self.seriesEndostealContourCombo.currentData),
            "thickness_method": str(self.seriesThicknessMethodCombo.currentData),
            "thickness_backend": str(self.seriesThicknessBackendCombo.currentData),
            "result_path": str(result_path),
        }
        Path(job_path).write_text(json.dumps(job, indent=2), encoding="utf-8")
        return Path(job_path), result_path

    def _run_registered_series(self):
        if not self._lastRegisteredRows:
            self._discover_registered_series()
        if not self._lastRegisteredRows:
            return
        try:
            job_path, result_path = self._write_registered_series_job()
            self._registeredSeriesResultPath = result_path
            self._set_registered_series_running(True)
            self.seriesStatusLabel.text = "Registered microarchitecture is running in the background."
            self.logic.run_registered_series_job(
                job_path,
                on_output=self._series_log,
                on_finished=self._on_registered_series_finished,
            )
        except Exception as exc:
            self._set_registered_series_running(False)
            slicer.util.errorDisplay(f"Registered series microarchitecture failed:\n{exc}")
            self._series_log(f"[registered] measurements failed: {exc}")
            return

    def _on_registered_series_finished(self, exit_code, exit_status):
        self._set_registered_series_running(False)
        self._series_log(f"[registered-process] finished with exit code {exit_code}")
        result_path = Path(str(getattr(self, "_registeredSeriesResultPath", "") or ""))
        result = {}
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._series_log(f"[registered] could not read worker result: {exc}")
        if int(exit_code) != 0 or result.get("error"):
            message = result.get("error") or "Registered microarchitecture worker failed."
            self.seriesStatusLabel.text = "Registered microarchitecture failed."
            slicer.util.errorDisplay(f"Registered series microarchitecture failed:\n{message}")
            return

        prepared = result.get("prepared", {})
        outputs = result.get("outputs", {})
        if prepared.get("rows"):
            self._allRegisteredRows = list(prepared["rows"])
            self._populate_registered_series_filters(self._allRegisteredRows)
            self._refresh_registered_series_table()
        measured_count = len(outputs.get("session_csvs", []))
        skipped_count = len(outputs.get("skipped_rows", []))
        self.seriesStatusLabel.text = (
            f"Wrote registered series measurements for {measured_count} session(s); "
            f"skipped {skipped_count}. Output: {outputs['long_csv']}"
        )
        if prepared.get("manifest"):
            self._series_log(f"[registered] wrote manifest: {prepared['manifest']}")
        for item in prepared.get("generated", []):
            self._series_log(f"[registered] {item['source']} {item['role']}: {item['path']}")
        self._series_log(f"[registered] wrote manifest: {outputs['manifest']}")
        self._series_log(f"[registered] wrote long table: {outputs['long_csv']}")
        for path in outputs.get("session_csvs", []):
            self._series_log(f"[registered] wrote session table: {path}")
        for row in outputs.get("skipped_rows", []):
            self._series_log(
                f"[registered] skipped sub-{row['subject_id']} ses-{row['session_id']}: {row['status']}"
            )

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


def _write_registered_series_worker_result(path, payload):
    Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
    Path(str(path)).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_registered_series_worker(job_path):
    job_path = Path(str(job_path)).expanduser()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_path = Path(str(job["result_path"])).expanduser()
    logic = BoneMicroarchitectureLogic()
    def print_progress(message):
        print(message, flush=True)

    try:
        print(f"[registered] worker job: {job_path}", flush=True)
        print("[registered] preparing workspace, masks, registration, and common regions ...", flush=True)
        prepared = logic.prepare_registered_series_workspace(
            job["dataset_root"],
            job["output_root"],
            job["rows"],
            segmentation_method=job["segmentation_method"],
            periosteal_contour_method=job["periosteal_contour_method"],
            endosteal_contour_method=job["endosteal_contour_method"],
            progress_callback=print_progress,
        )
        print(f"[registered] wrote manifest: {prepared['manifest']}", flush=True)
        for item in prepared.get("generated", []):
            print(f"[registered] {item['source']} {item['role']}: {item['path']}", flush=True)
        print("[registered] running native-space common-region measurements ...", flush=True)
        outputs = logic.run_registered_series_microarchitecture(
            job["dataset_root"],
            job["output_root"],
            prepared["rows"],
            thickness_method=job["thickness_method"],
            thickness_backend=job["thickness_backend"],
        )
        measured_count = len(outputs.get("session_csvs", []))
        skipped_count = len(outputs.get("skipped_rows", []))
        print(f"[registered] measured {measured_count} session(s), skipped {skipped_count}.", flush=True)
        print(f"[registered] wrote long table: {outputs['long_csv']}", flush=True)
        for path in outputs.get("session_csvs", []):
            print(f"[registered] wrote session table: {path}", flush=True)
        for row in outputs.get("skipped_rows", []):
            print(f"[registered] skipped sub-{row['subject_id']} ses-{row['session_id']}: {row['status']}", flush=True)
        _write_registered_series_worker_result(result_path, {"prepared": prepared, "outputs": outputs})
        return 0
    except Exception as exc:
        print(f"[registered] failed: {exc}", flush=True)
        _write_registered_series_worker_result(result_path, {"error": str(exc)})
        return 1


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--registered-series-job":
        return _run_registered_series_worker(argv[1])
    return 0


class BoneMicroarchitectureTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.test_BoneMicroarchitecture1()

    def test_BoneMicroarchitecture1(self):
        self.delayDisplay("Bone Microarchitecture smoke test")
        logic = BoneMicroarchitectureLogic()
        if not logic.is_core_available():
            self.skipTest("Microarchitecture core is not installed")
        self.assertTrue(logic.is_core_available())


if __name__ == "__main__":
    raise SystemExit(_main())
