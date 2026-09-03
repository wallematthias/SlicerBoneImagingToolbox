from __future__ import annotations

import csv
import importlib
import json
from importlib import metadata
import os
from pathlib import Path
import re
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

from SlicerBoneImagingToolboxLib.slicer_pip import slicer_pip_install, slicer_python_executable  # noqa: E402
from SlicerBoneImagingToolboxLib.segmentation_methods import (  # noqa: E402
    BONE_SEGMENTATION_METHODS,
    ENDOSTEAL_CONTOUR_METHODS,
    PERIOSTEAL_CONTOUR_METHODS,
)

from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


def run_microarchitecture_batch(*args, **kwargs):
    from bone_microarchitecture.batch import run_microarchitecture_batch as _run_microarchitecture_batch

    return _run_microarchitecture_batch(*args, **kwargs)


MODULE_VERSION = "0.1.0"
REGISTERED_MICROARCHITECTURE_DIR_NAME = "registered"
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
        parent.title = "Microarchitecture"
        parent.categories = ["Bone Imaging.Microstructural Analysis"]
        parent.icon = qt.QIcon(str(Path(__file__).with_name("Resources") / "Icons" / "BoneMicroarchitecture.png"))
        parent.index = 70
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Compute bone microarchitecture measurements, including Tt.BMD, from Slicer masks.\n"
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = (
            "Author: Matthias Walle. "
            "This module wraps a lightweight Python microarchitecture core for Slicer."
        )


class BoneMicroarchitectureLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._proc = None

    def is_registered_series_running(self):
        return self._proc is not None

    def run_batch_workflow(self, dataset_root, *, use_common_region=True, force=False, progress=None):
        """Run folder mode through the package batch API, not Slicer logic."""
        return run_microarchitecture_batch(
            dataset_root,
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
    def folder_batch_command(dataset_root, *, subject_id="", site="", session_id="", use_common_region=True, force=False,
                             thickness_method="hildebrand", thickness_backend="auto"):
        command = ["-m", "bone_microarchitecture.cli", "run-batch", str(Path(dataset_root).expanduser().resolve())]
        if subject_id:
            command.extend(["--subject", str(subject_id)])
        if site:
            command.extend(["--site", str(site)])
        if session_id:
            command.extend(["--session", str(session_id)])
        if not use_common_region:
            command.append("--no-common-region")
        if force:
            command.append("--force")
        command.extend(["--thickness-method", str(thickness_method), "--thickness-backend", str(thickness_backend)])
        return command

    def run_folder_batch_job(self, dataset_root, *, subject_id="", site="", session_id="", use_common_region=True, force=False,
                             thickness_method="hildebrand", thickness_backend="auto", on_output=None, on_finished=None):
        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.MergedChannels)

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
                on_output(text)

        def _finished(*signal_args):
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
        proc.start(slicer_python_executable(slicer.app.applicationFilePath()), self.folder_batch_command(
            dataset_root, subject_id=subject_id, site=site, session_id=session_id, use_common_region=use_common_region, force=force,
            thickness_method=thickness_method, thickness_backend=thickness_backend,
        ))
        return proc

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
        slicer_pip_install("--upgrade --prefer-binary numpy>=2.0,<3.0 scipy>=1.18,<2.0")
        if sys.platform == "darwin":
            slicer_pip_install("--upgrade --prefer-binary pyobjc-framework-Metal>=10")
        else:
            slicer_pip_install("--upgrade --prefer-binary pyopencl>=2024.1")
        if MICROARCHITECTURE_LOCAL_REPO.exists():
            slicer_pip_install(f"--no-deps -e {MICROARCHITECTURE_LOCAL_REPO}")
        else:
            slicer_pip_install("--upgrade --prefer-binary bone-microarchitecture>=0.2.3")
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
            root = Path(output_text).expanduser()
        else:
            root = Path(str(dataset_root)).expanduser()
        return self._derivative_family_root(root, "Microarchitecture")

    def _derivative_family_root(self, root, family):
        root = Path(str(root)).expanduser()
        if root.name in {REGISTERED_MICROARCHITECTURE_DIR_NAME, "RegisteredMicroarchitecture"}:
            return root.parent
        if root.name == family:
            return root
        if root.name == "derivatives":
            return root / family
        return root / "derivatives" / family

    @staticmethod
    def _voi_token(site):
        return re.sub(r"[^A-Za-z0-9]+", "", str(site or "").strip()).lower() or "unknown"

    def registered_subject_dir(self, output_root, subject_id):
        return Path(output_root) / f"sub-{subject_id}"

    def registered_session_output_dir(self, output_root, row):
        return (
            self.registered_subject_dir(output_root, row["subject_id"])
            / f"ses-{row['session_id']}"
            / "xct"
            / "measurements"
        )

    def discover_registered_series(self, dataset_root, *, subject_filter="", site_filter=""):
        from timelapsedhrpqct.config.models import DiscoveryConfig
        from timelapsedhrpqct.dataset.discovery import discover_raw_sessions

        from timelapsedhrpqct.utils.session_ids import session_sort_key

        sessions = discover_raw_sessions(
            Path(str(dataset_root)).expanduser(),
            DiscoveryConfig(),
            canonicalize_sessions=False,
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

    def discover_independent_cases(self, dataset_root, *, subject_filter="", site_filter=""):
        rows = self.discover_registered_series(
            dataset_root,
            subject_filter=subject_filter,
            site_filter=site_filter,
        )
        ready_by_key = {}
        try:
            from bone_microarchitecture import batch as microarchitecture_batch

            for case in microarchitecture_batch._discover_cases(Path(str(dataset_root)).expanduser()):
                record = case["bone_segmentation"]
                key = (
                    str(record.subject_id),
                    str(record.site or ""),
                    self._microarchitecture_session_key(record.session_id),
                    int(record.stack_index or 1),
                )
                ready_by_key[key] = {
                    "subject_id": str(record.subject_id),
                    "site": str(record.site or ""),
                    "session_id": str(record.session_id or ""),
                    "image_path": str(case["transformed_image"].path),
                    "seg_path": str(case["bone_segmentation"].path),
                    "full_path": str(case["periosteal_mask"].path),
                    "trab_path": str(case["trabecular_mask"].path),
                    "cort_path": str(case.get("cortical_mask").path) if case.get("cortical_mask") else "",
                    "stack_index": int(record.stack_index or 1),
                    "status": "Ready",
                }
        except Exception:
            ready_by_key = {}

        merged = {}
        for row in rows:
            key = (
                str(row.get("subject_id", "")),
                self._canonical_microarchitecture_site(row.get("site", "")),
                self._microarchitecture_session_key(row.get("session_id", "")),
                int(row.get("stack_index", 1)),
            )
            merged[key] = ready_by_key.get(key, row)
        for key, row in ready_by_key.items():
            merged.setdefault(key, row)
        return list(merged.values())

    @staticmethod
    def _canonical_microarchitecture_site(site):
        normalized = str(site or "").strip().lower()
        return {
            "rl": "radiusleft",
            "rr": "radiusright",
            "tl": "tibialeft",
            "tr": "tibiaright",
        }.get(normalized, normalized)

    @staticmethod
    def _microarchitecture_session_key(session_id):
        value = str(session_id or "").strip().upper()
        if value.startswith("SES-"):
            value = value[4:]
        if value.startswith("Y") and value[1:].isdigit():
            value = value[1:]
        return value.lstrip("0") or "0"

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
            (self.registered_session_output_dir(root, row) / "maps").mkdir(parents=True, exist_ok=True)
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
            missing = self._registered_row_missing(prepared)
            prepared["status"] = "Ready" if not missing else f"Missing {', '.join(missing)}"
            prepared_rows.append(prepared)
        self._mark_incomplete_registered_groups(prepared_rows)
        generated.extend(
            self._build_registered_common_regions(
                root,
                prepared_rows,
                dataset_root=dataset_root,
                progress_callback=progress_callback,
            )
        )
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

    def _shared_common_region_root(self, dataset_root):
        return self._derivative_family_root(Path(str(dataset_root)).expanduser(), "CommonRegion")

    def _shared_common_region_subject_site_dir(self, dataset_root, row):
        return (
            self._shared_common_region_root(dataset_root)
            / f"sub-{row['subject_id']}"
            / "xct"
        )

    def _shared_common_region_common_mask_path(self, dataset_root, row):
        return self._shared_common_region_subject_site_dir(dataset_root, row) / "masks" / (
            f"sub-{row['subject_id']}_voi-{self._voi_token(row['site'])}_stack-{int(row.get('stack_index', 1)):02d}_mask-scan-region_common.nii.gz"
        )

    def _shared_common_region_native_mask_path(self, dataset_root, row):
        return self._shared_common_region_root(dataset_root) / f"sub-{row['subject_id']}" / f"ses-{row['session_id']}" / "xct" / "masks" / (
            f"sub-{row['subject_id']}_ses-{row['session_id']}_voi-{self._voi_token(row['site'])}_mask-scan-region_native_common.nii.gz"
        )

    def _read_existing_registered_transform(self, path):
        path = Path(str(path)).expanduser()
        if not path.exists():
            return None
        return sitk.ReadTransform(str(path))

    def _shared_timelapsed_pairwise_transform_path(self, dataset_root, moving_row, fixed_row):
        root = Path(str(dataset_root)).expanduser()
        path = (
            root
            / "derivatives"
            / "Registration"
            / f"sub-{moving_row['subject_id']}"
            / f"ses-{moving_row['session_id']}"
            / "xct"
            / "pairwise"
            / (
                f"sub-{moving_row['subject_id']}_ses-{moving_row['session_id']}_voi-{self._voi_token(moving_row['site'])}_"
                f"stack-{int(moving_row.get('stack_index', 1)):02d}_"
                f"from-ses-{moving_row['session_id']}_to-ses-{fixed_row['session_id']}_pairwise.tfm"
            )
        )
        if path.exists():
            return path
        from timelapsedhrpqct.dataset.derivative_paths import existing_derivative_path, timelapse_pairwise_transform_path

        fallback = timelapse_pairwise_transform_path(
            root,
            moving_row["subject_id"],
            moving_row["site"],
            int(moving_row.get("stack_index", 1)),
            moving_row["session_id"],
            fixed_row["session_id"],
        )
        return existing_derivative_path(fallback)

    def _shared_timelapsed_baseline_transform_path(self, dataset_root, row, baseline_row):
        root = Path(str(dataset_root)).expanduser()
        path = (
            root
            / "derivatives"
            / "Registration"
            / f"sub-{row['subject_id']}"
            / f"ses-{row['session_id']}"
            / "xct"
            / "baseline"
            / (
                f"sub-{row['subject_id']}_ses-{row['session_id']}_voi-{self._voi_token(row['site'])}_"
                f"stack-{int(row.get('stack_index', 1)):02d}_"
                f"from-ses-{row['session_id']}_to-ses-{baseline_row['session_id']}_baseline.tfm"
            )
        )
        if path.exists():
            return path
        from timelapsedhrpqct.dataset.derivative_paths import existing_derivative_path, timelapse_baseline_transform_path

        fallback = timelapse_baseline_transform_path(
            root,
            row["subject_id"],
            row["site"],
            int(row.get("stack_index", 1)),
            row["session_id"],
            baseline_row["session_id"],
        )
        return existing_derivative_path(fallback)

    def _resample_registered_mask(self, mask, reference, transform):
        from SlicerBoneImagingToolboxLib.masks import resample_mask

        return resample_mask(mask, reference, transform)

    def _registered_scan_region(self, image):
        from SlicerBoneImagingToolboxLib.masks import scan_region_mask

        return scan_region_mask(image)

    def _clip_registered_mask_to_scan_region(self, mask, scan_region):
        from SlicerBoneImagingToolboxLib.masks import clip_mask_to_region

        return clip_mask_to_region(mask, scan_region)

    def _register_to_baseline(self, fixed_image, moving_image, fixed_mask, moving_mask):
        from SlicerBoneImagingToolboxLib.registration import register_image_pair

        return register_image_pair(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
        )

    def _build_registered_common_regions(self, output_root, rows, *, dataset_root=None, progress_callback=None):
        from SlicerBoneImagingToolboxLib.common_region import CommonRegionSession, build_common_scan_region
        from SlicerBoneImagingToolboxLib.derivatives import DerivativeManifest, DerivativeRecord, write_manifest
        from timelapsedhrpqct.utils.session_ids import session_sort_key
        from timelapsedhrpqct.processing.transform_chain import (
            PairwiseTransform,
            compose_sequential_to_baseline,
            flatten_transform,
        )

        generated = []
        dataset_root = Path(str(dataset_root)).expanduser() if dataset_root is not None else Path(str(output_root)).expanduser()
        common_records = []
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
                shared_pairwise_path = self._shared_timelapsed_pairwise_transform_path(dataset_root, row, previous)
                pairwise_transform = self._read_existing_registered_transform(shared_pairwise_path)
                if pairwise_transform is not None:
                    pairwise_transform = flatten_transform(pairwise_transform)
                    pairwise_path = shared_pairwise_path
                    source = "reused_registration"
                    self._registered_progress(
                        progress_callback,
                        f"[registered] reusing registration sub-{row['subject_id']} ses-{row['session_id']} "
                        f"to ses-{previous['session_id']}",
                    )
                    image = None
                    full = None
                else:
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
                    pairwise_path = self._shared_timelapsed_pairwise_transform_path(dataset_root, row, previous)
                    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
                    sitk.WriteTransform(pairwise_transform, str(pairwise_path))
                    source = "registration"
                generated.append(
                    {
                        "session": row["session_id"],
                        "role": "transform_pairwise",
                        "path": str(pairwise_path),
                        "source": source,
                    }
                )
                pairwise.append(PairwiseTransform(session_id=row["session_id"], transform=pairwise_transform))
                previous = row
                previous_image = image if image is not None else self._read_registered_series_image(row["image_path"], role="image")
                previous_full = full if full is not None else self._read_registered_series_image(row["full_path"], role="full")

            identity = sitk.Transform(3, sitk.sitkIdentity)
            baseline_transforms = compose_sequential_to_baseline(
                pairwise_transforms=pairwise,
                baseline_session_id=baseline["session_id"],
                dimension=3,
            )
            transforms = {item.session_id: item.transform for item in baseline_transforms}
            transforms[baseline["session_id"]] = transforms.get(baseline["session_id"], identity)
            for row in ordered:
                shared_baseline_path = self._shared_timelapsed_baseline_transform_path(dataset_root, row, baseline)
                shared_baseline_transform = self._read_existing_registered_transform(shared_baseline_path)
                if shared_baseline_transform is not None:
                    transforms[row["session_id"]] = flatten_transform(shared_baseline_transform)
                    transform_path = shared_baseline_path
                    source = "reused_registration"
                else:
                    transform_path = self._shared_timelapsed_baseline_transform_path(dataset_root, row, baseline)
                    transform_path.parent.mkdir(parents=True, exist_ok=True)
                    sitk.WriteTransform(flatten_transform(transforms[row["session_id"]]), str(transform_path))
                    source = "registration"
                generated.append(
                    {
                        "session": row["session_id"],
                        "role": "transform_composed",
                        "path": str(transform_path),
                        "source": source,
                    }
                )

            common_region_sessions = []
            for row in ordered:
                self._registered_progress(
                    progress_callback,
                    f"[registered] resampling scan region for sub-{row['subject_id']} ses-{row['session_id']} into common space",
                )
                transform = transforms[row["session_id"]]
                image = self._read_registered_series_image(row["image_path"], role="image")
                common_region_sessions.append(
                    CommonRegionSession(
                        subject_id=row["subject_id"],
                        site=row["site"],
                        session_id=row["session_id"],
                        stack_index=int(row.get("stack_index", 1)),
                        image=image,
                        transform_to_reference=transform,
                    )
                )
            common_region = build_common_scan_region(
                common_region_sessions,
                reference_session_id=baseline["session_id"],
            )
            common_scan_region = common_region.common_mask

            self._registered_progress(progress_callback, "[registered] writing scan-region common-space mask")
            shared_common_path = self._shared_common_region_common_mask_path(dataset_root, baseline)
            shared_common_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(common_scan_region, str(shared_common_path))
            common_records.append(
                DerivativeRecord(
                    derivative="CommonRegion",
                    role="scan_region_common",
                    subject_id=str(baseline["subject_id"]),
                    site=str(baseline["site"]),
                    session_id=str(baseline["session_id"]),
                    stack_index=int(baseline.get("stack_index", 1)),
                    space="reference",
                    path=str(shared_common_path),
                    source="generated",
                    metadata={"reference_session_id": str(baseline["session_id"]), "producer": "BoneMicroarchitecture"},
                )
            )
            generated.append(
                {
                    "session": baseline["session_id"],
                    "role": "scan_region_common",
                    "path": str(shared_common_path),
                    "source": "CommonRegion",
                }
            )

            for row in ordered:
                self._registered_progress(
                    progress_callback,
                    f"[registered] returning common scan region to native space for sub-{row['subject_id']} ses-{row['session_id']}",
                )
                native_common = common_region.native_masks[row["session_id"]]
                shared_native_path = self._shared_common_region_native_mask_path(dataset_root, row)
                shared_native_path.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(native_common, str(shared_native_path))
                row["native_common_scan_region_path"] = str(shared_native_path)
                common_records.append(
                    DerivativeRecord(
                        derivative="CommonRegion",
                        role="scan_region_native_common",
                        subject_id=str(row["subject_id"]),
                        site=str(row["site"]),
                        session_id=str(row["session_id"]),
                        stack_index=int(row.get("stack_index", 1)),
                        space="native",
                        path=str(shared_native_path),
                        source="generated",
                        metadata={"reference_session_id": str(baseline["session_id"]), "producer": "BoneMicroarchitecture"},
                    )
                )
                generated.append(
                    {
                        "session": row["session_id"],
                        "role": "scan_region_native_common",
                        "path": str(shared_native_path),
                        "source": "CommonRegion",
                    }
                )
                row["measurement_space"] = "native_image_space_common_region"
        if common_records:
            common_root = self._shared_common_region_root(dataset_root)
            write_manifest(
                common_root / "manifest.json",
                DerivativeManifest(
                    workflow="CommonRegion",
                    version=MODULE_VERSION,
                    dataset_root=str(dataset_root),
                    records=common_records,
                    metadata={"module": "BoneMicroarchitecture", "producer": "registered_microarchitecture"},
                ),
            )
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
        if role == "image":
            from SlicerBoneImagingToolboxLib.image_io import read_image

            return read_image(path)
        from SlicerBoneImagingToolboxLib.image_io import read_mask

        return read_mask(path)

    def run_registered_series_microarchitecture(
        self,
        dataset_root,
        output_root,
        rows,
        *,
        thickness_method="hildebrand",
        thickness_backend="auto",
        progress_callback=None,
    ):
        from bone_microarchitecture import compute_microarchitecture, default_thickness_backend
        from bone_microarchitecture.results import SUMMARY_COLUMNS, measurement_rows, write_measurement_csv

        root = self.registered_microarchitecture_root(dataset_root, output_root)
        self.write_registered_series_manifest(dataset_root, root, rows)
        requested_thickness_backend = str(thickness_backend or "auto").strip().lower()
        resolved_thickness_backend = (
            default_thickness_backend()
            if requested_thickness_backend == "auto"
            else requested_thickness_backend
        )
        requested_thickness_method = str(thickness_method or "hildebrand").strip().lower()
        self._registered_progress(
            progress_callback,
            f"[registered] thickness: {requested_thickness_method} "
            f"backend: {requested_thickness_backend} -> {resolved_thickness_backend}",
        )
        long_rows = []
        written = []
        skipped_rows = []
        for row in rows:
            if row.get("status") != "Ready":
                self._registered_progress(
                    progress_callback,
                    f"[registered] skipping sub-{row.get('subject_id', '')} "
                    f"ses-{row.get('session_id', '')}: {row.get('status', 'Not ready')}",
                )
                skipped_rows.append(
                    {
                        "subject_id": row.get("subject_id", ""),
                        "site": row.get("site", ""),
                        "session_id": row.get("session_id", ""),
                        "status": row.get("status", "Not ready"),
                    }
                )
                continue
            self._registered_progress(
                progress_callback,
                f"[registered] measuring sub-{row['subject_id']} ses-{row['session_id']}",
            )
            image = self._read_registered_series_image(row["image_path"], role="image")
            bone_seg = self._read_registered_series_image(row["seg_path"], role="bone seg")
            full_mask = self._read_registered_series_image(row["full_path"], role="full")
            trab_mask = self._read_registered_series_image(row["trab_path"], role="trab")
            cort_mask = self._read_registered_series_image(row["cort_path"], role="cort")
            scan_region = None
            if row.get("native_common_scan_region_path"):
                scan_region = self._read_registered_series_image(
                    row["native_common_scan_region_path"],
                    role="scan region",
                )
                bone_seg = self._clip_registered_mask_to_scan_region(bone_seg, scan_region)
                full_mask = self._clip_registered_mask_to_scan_region(full_mask, scan_region)
                trab_mask = self._clip_registered_mask_to_scan_region(trab_mask, scan_region)
                cort_mask = self._clip_registered_mask_to_scan_region(cort_mask, scan_region)
            size_items = [image, bone_seg, full_mask, trab_mask, cort_mask]
            if scan_region is not None:
                size_items.append(scan_region)
            sizes = {item.GetSize() for item in size_items}
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
                thickness_method=requested_thickness_method,
                thickness_backend=resolved_thickness_backend,
            )
            session_dir = self.registered_session_output_dir(root, row)
            maps_dir = session_dir / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"sub-{row['subject_id']}_ses-{row['session_id']}_voi-{self._voi_token(row['site'])}"
            csv_path = session_dir / f"{prefix}_measurements.csv"
            write_measurement_csv(csv_path, result.measurements, result.maps)
            for map_role, array in result.maps.items():
                map_image = self._array_to_sitk_like(array, trab_mask)
                sitk.WriteImage(map_image, str(maps_dir / f"{prefix}_map-{map_role.lower().replace('.', '-')}.nii.gz"))
            for summary_row in measurement_rows(result.measurements, result.maps):
                long_row = {
                    "Subject": row["subject_id"],
                    "Site": row["site"],
                    "Session": row["session_id"],
                }
                long_row.update(summary_row)
                long_rows.append(long_row)
            written.append(str(csv_path))
            self._registered_progress(
                progress_callback,
                f"[registered] finished measuring sub-{row['subject_id']} ses-{row['session_id']}",
            )

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
        self._style_microarchitecture_map(volume_node, map_role)
        return volume_node

    def _style_microarchitecture_map(self, volume_node, map_role):
        if volume_node is None:
            return
        try:
            volume_node.CreateDefaultDisplayNodes()
            display_node = volume_node.GetDisplayNode()
        except Exception:
            display_node = None
        if display_node is None:
            return
        try:
            display_node.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
        except Exception:
            pass
        for method_name in ("AutoWindowLevelOn", "AutoThresholdOff"):
            method = getattr(display_node, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    def _map_role_from_path(self, path):
        stem = str(Path(path).name)
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        else:
            stem = Path(stem).stem
        marker = "_map-"
        if marker in stem:
            token = stem.rsplit(marker, 1)[-1]
        else:
            token = stem
        return token.replace("-", ".")

    @staticmethod
    def _microarchitecture_map_folder_name(path, group=None):
        path = Path(str(path))
        path_text = str(path).replace("\\", "/").lower()
        suffix = "_xct_registered_microstructure" if "/registered_measurements/" in path_text else "_xct_microstructure"
        match = re.search(r"(?i)(sub-[^_]+)_(ses-[^_]+)_voi-([^_]+)", path.name)
        if match:
            return f"{match.group(1)}_{match.group(2)}_voi-{match.group(3)}{suffix}"
        if group:
            subject = str(group.get("subject") or "microarchitecture")
            session = str(group.get("session") or "session")
            site = str(group.get("site") or group.get("voi") or "voi")
            if not subject.startswith("sub-"):
                subject = f"sub-{subject}"
            if not session.startswith("ses-"):
                session = f"ses-{session}"
            if not site.startswith("voi-"):
                site = f"voi-{site}"
            return f"{subject}_{session}_{site}{suffix}"
        return f"microarchitecture{suffix}"

    @staticmethod
    def _put_node_in_subject_hierarchy_folder(node, folder_name):
        try:
            if node is None or not folder_name:
                return False
            sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            if sh_node is None:
                return False
            scene_item = sh_node.GetSceneItemID()
            folder_item = sh_node.GetItemChildWithName(scene_item, str(folder_name))
            if not folder_item:
                folder_item = sh_node.CreateFolderItem(scene_item, str(folder_name))
            try:
                slicer.app.processEvents()
            except Exception:
                pass
            node_item = sh_node.GetItemByDataNode(node)
            if node_item:
                sh_node.SetItemParent(node_item, folder_item)
                return True
            return False
        except Exception:
            return False

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
        common_region_node=None,
        common_region_segment_id=None,
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
            common_region_node,
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

            common_region = None
            if common_region_node is not None:
                from SlicerBoneImagingToolboxLib.masks import clip_mask_to_region

                common_region = self._volume_to_sitk_uint8(
                    common_region_node,
                    "analysis mask",
                    selected_segment_id=common_region_segment_id,
                    reference_node=reference_node,
                )
                if common_region.GetSize() != trab_seg.GetSize():
                    raise ValueError("Analysis mask size must match the selected masks.")
                bone_seg = clip_mask_to_region(bone_seg, common_region)
                peri_mask = clip_mask_to_region(peri_mask, common_region)
                trab_seg = clip_mask_to_region(trab_seg, common_region)
                if cort_mask is not None:
                    cort_mask = clip_mask_to_region(cort_mask, common_region)

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
            if common_region_node is not None:
                table_node.SetAttribute("BoneImaging.Microarchitecture.AnalysisMaskNode", common_region_node.GetID())
                table_node.SetAttribute("BoneImaging.Microarchitecture.AnalysisMaskName", common_region_node.GetName())
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
        self._lastTableBaseName = "microarchitecture_measurements"
        self._lastMeasurementColumns = []
        self._lastMeasurementRows = []
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
        self.commonRegionMaskSelector, self.commonRegionSegmentCombo = self._mask_selector_row(
            form,
            "Analysis mask",
            "analysis mask",
            "Optional mask used to constrain all selected microarchitecture analysis regions.",
        )

        calibration = ctk.ctkCollapsibleButton()
        calibration.text = "Grayscale Calibration"
        calibration.collapsed = True
        single_layout.addWidget(calibration)
        calibration_form = qt.QFormLayout(calibration)

        self.imageUnitsCombo = qt.QComboBox()
        for label, value in [
            ("Auto / already calibrated", "bmd"),
            ("HU", "hu"),
            ("Scanco native", "scanco"),
            ("Attenuation", "attenuation"),
        ]:
            self.imageUnitsCombo.addItem(label, value)
        self._tip(
            self.imageUnitsCombo,
            "Advanced override for BMD conversion. AIM images are read with AIMIO density scaling when source metadata is available.",
        )
        self.muScalingSpin = self._double_spin(1, 100000, 1, 8192)
        self.muWaterSpin = self._double_spin(0, 10, 4, 0.2409)
        self.rescaleSlopeSpin = self._double_spin(-100000, 100000, 4, 1603.51904)
        self.rescaleInterceptSpin = self._double_spin(-100000, 100000, 4, -391.209015)
        self._tip(self.muScalingSpin, "mu_scaling value for Scanco native to BMD conversion.")
        self._tip(self.muWaterSpin, "mu_water value for HU to BMD conversion.")
        self._tip(self.rescaleSlopeSpin, "rescale_slope value for BMD conversion.")
        self._tip(self.rescaleInterceptSpin, "rescale_intercept value for BMD conversion.")
        calibration_form.addRow("Calibration source", self.imageUnitsCombo)
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

        self.modeTabs.addTab(single_tab, "Scene")
        self.layout.addStretch(1)
        self._update_dependency_ui()

    def _run_folder_batch(self):
        if not self._folderBatchGroups:
            self._discover_folder_batch_groups()
        for row_index in range(len(self._folderBatchGroups)):
            self._queue_folder_batch_row(row_index)

    def _on_folder_batch_finished(self, exit_code, _exit_status):
        self.folderRunButton.enabled = True
        self.folderBatchStatus.text = f"Microarchitecture batch finished with exit code {int(exit_code)}."

    def _update_folder_registered_options(self, *args):
        del args
        registered = bool(getattr(self, "folderRegisteredCheck", None) and self.folderRegisteredCheck.checked)
        if hasattr(self, "folderRegisteredWorkflowCombo"):
            self.folderRegisteredWorkflowCombo.enabled = registered
        self._configure_folder_batch_table_for_mode()

    def _configure_folder_batch_table_for_mode(self):
        if bool(getattr(self, "folderRegisteredCheck", None) and self.folderRegisteredCheck.checked):
            headers = ["Action", "Subject", "Site", "Sessions", "Status"]
        else:
            headers = ["Action", "Image", "Subject", "Site", "Session", "Status"]
        self.folderBatchTable.setColumnCount(len(headers))
        self.folderBatchTable.setHorizontalHeaderLabels(headers)

    def _is_folder_batch_image_path(self, path):
        name = path.name.lower()
        if any(token in name for token in ("_mask-", "_seg", "_map", "manifest", "measurements", "slicer_run_config")):
            return False
        if name.endswith((".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")):
            return True
        upper = path.name.upper()
        return upper.startswith("ISQ") or ".AIM" in upper

    def _browse_folder_dataset_root(self):
        path = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            "Select dataset root",
            self.folderDatasetRootEdit.text,
        )
        if path:
            self.folderDatasetRootEdit.text = str(path)

    def _discover_folder_batch_groups(self):
        root_text = str(self.folderDatasetRootEdit.text or "").strip()
        if not root_text:
            self.folderBatchStatus.text = "Select a dataset root before discovery."
            return
        root = Path(root_text).expanduser()
        subject_filter = ""
        site_filter = ""
        self._configure_folder_batch_table_for_mode()
        groups = {}
        registered = bool(self.folderRegisteredCheck.checked)
        try:
            if registered:
                rows = self.logic.discover_registered_series(root, subject_filter=subject_filter, site_filter=site_filter)
                for row in rows:
                    key = (str(row.get("subject_id", "")), str(row.get("site", "")), int(row.get("stack_index", 1)))
                    group = groups.setdefault(
                        key,
                        {
                            "subject": str(row.get("subject_id", "")),
                            "site": str(row.get("site", "")),
                            "stack_index": int(row.get("stack_index", 1)),
                            "sessions": set(),
                            "rows": [],
                            "status": "Ready",
                            "mode": "registered",
                        },
                    )
                    group["sessions"].add(str(row.get("session_id", "")))
                    group["rows"].append(row)
                    if row.get("status") != "Ready":
                        group["status"] = "Missing inputs"
            else:
                rows = self.logic.discover_independent_cases(root, subject_filter=subject_filter, site_filter=site_filter)
                for row_index, row in enumerate(rows):
                    key = (str(row.get("subject_id", "")), str(row.get("site", "")), str(row.get("session_id", "")), int(row.get("stack_index", 1)), row_index)
                    groups[key] = {
                        "image": str(row.get("image_path", "")),
                        "subject": str(row.get("subject_id", "")),
                        "site": str(row.get("site", "")),
                        "session": str(row.get("session_id", "")),
                        "stack_index": int(row.get("stack_index", 1)),
                        "rows": [row],
                        "status": str(row.get("status", "Discovered")),
                        "mode": "independent",
                    }
        except Exception:
            for path in root.iterdir():
                if not path.is_file() or not self._is_folder_batch_image_path(path):
                    continue
                name = path.name
                if "sub-" not in name or "site-" not in name:
                    continue
                subject = ""
                site = ""
                for part in path.parts:
                    if part.startswith("sub-"):
                        subject = part[4:]
                    elif part.startswith("site-"):
                        site = part[5:]
                if subject_filter and subject != subject_filter:
                    continue
                if site_filter and site != site_filter:
                    continue
                if subject or site:
                    key = (subject, site, 1)
                    groups.setdefault(
                        key,
                        {"image": str(path), "subject": subject, "site": site, "session": "", "stack_index": 1, "sessions": set(), "rows": [], "status": "Discovered", "mode": "independent"},
                    )
        self._folderBatchGroups = sorted(groups.values(), key=lambda item: (item["subject"], item["site"], item["stack_index"]))
        self.folderBatchTable.setRowCount(len(self._folderBatchGroups))
        for row_index, group in enumerate(self._folderBatchGroups):
            result_path = self._folder_result_path_for_group(root, group)
            if result_path is not None:
                group["result_path"] = str(result_path)
                group["status"] = "Done"
            if registered:
                sessions = ", ".join(sorted(s for s in group.get("sessions", set()) if s))
                values = [group.get("subject", ""), group.get("site", ""), sessions, group.get("status", "Discovered")]
            else:
                image_name = Path(str(group.get("image", ""))).name if group.get("image") else ""
                values = [image_name, group.get("subject", ""), group.get("site", ""), group.get("session", ""), group.get("status", "Discovered")]
            for column, value in enumerate(values, start=1):
                item = qt.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                self.folderBatchTable.setItem(row_index, column, item)
            if group.get("result_path"):
                self._set_folder_group_action(row_index, "Load")
            else:
                self._set_folder_group_action(row_index, "Run")
        try:
            self.folderBatchTable.resizeColumnsToContents()
        except Exception:
            pass
        mode = "registered series" if registered else "independent case"
        self.folderBatchStatus.text = f"Discovered {len(self._folderBatchGroups)} {mode} group(s)."

    def _set_folder_group_status(self, row_index, status):
        if row_index is None:
            self.folderBatchStatus.text = str(status)
            return
        if 0 <= int(row_index) < self._table_count(self.folderBatchTable, "rowCount"):
            item = qt.QTableWidgetItem(str(status))
            item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
            status_column = self._table_count(self.folderBatchTable, "columnCount") - 1
            self.folderBatchTable.setItem(int(row_index), status_column, item)

    @staticmethod
    def _table_count(table, attribute):
        value = getattr(table, attribute)
        return int(value() if callable(value) else value)

    def _set_folder_group_action(self, row_index, action):
        if isinstance(action, qt.QPushButton):
            button = action
        else:
            button = qt.QPushButton(str(action))
        text = button.text
        label = str(text() if callable(text) else text)
        if label == "Load":
            button.clicked.connect(lambda _checked=False, index=row_index: self._load_folder_batch_outputs(index))
        elif label in {"Running", "Queued"}:
            button.enabled = False
        else:
            button.clicked.connect(lambda _checked=False, index=row_index: self._queue_folder_batch_row(index))
        self.folderBatchTable.setCellWidget(int(row_index), 0, button)

    def _folder_result_path_for_group(self, root, group, existing_only=True):
        group = dict(group or {})
        if group.get("mode") == "registered":
            result_path = str(group.get("result_path") or "").strip()
            return Path(result_path).expanduser() if result_path else None
        subject = str(group.get("subject") or "unknown").strip() or "unknown"
        site = str(group.get("site") or "unknown").strip() or "unknown"
        session = str(group.get("session") or "unknown").strip() or "unknown"
        root = Path(str(root)).expanduser()
        candidates = []
        for session_part in self._folder_result_session_parts(session):
            prefix = f"sub-{subject}_{session_part}_voi-{BoneMicroarchitectureLogic._voi_token(site)}"
            for family_root in (
                root / "derivatives" / "Microarchitecture",
                root / "Microarchitecture",
            ):
                candidates.append(
                    family_root
                    / f"sub-{subject}"
                    / session_part
                    / "xct"
                    / "measurements"
                    / f"{prefix}_measurements.csv"
                )
        existing = next((path for path in candidates if path.exists()), None)
        if existing is not None or existing_only:
            return existing
        return candidates[0]

    def _folder_result_session_parts(self, session):
        value = str(session or "unknown").strip() or "unknown"
        bare = value[4:] if value.lower().startswith("ses-") else value
        variants = []
        for candidate in (bare, bare.upper(), f"Y{bare}", f"Y{bare.lstrip('0') or '0'}"):
            session_part = candidate if str(candidate).lower().startswith("ses-") else f"ses-{candidate}"
            if session_part not in variants:
                variants.append(session_part)
        return variants

    def _queue_folder_batch_row(self, row_index):
        if row_index is None:
            return
        group = self._folderBatchGroups[int(row_index)]
        if group.get("result_path") and Path(str(group.get("result_path"))).expanduser().exists():
            self._set_folder_group_action(row_index, "Load")
            self._set_folder_group_status(row_index, "Done")
            return
        if bool(self.folderRegisteredCheck.checked):
            if len(group.get("rows", [])) < 2 or group.get("status") != "Ready":
                self._set_folder_group_status(row_index, "Missing inputs")
                self.folderBatchStatus.text = "Registered series needs at least two ready sessions with image, segmentation, full, trab, and cort masks."
                return
            queued = {"mode": "registered", "row_index": int(row_index), "group": group}
        else:
            if group.get("status") != "Ready":
                self._set_folder_group_status(row_index, str(group.get("status") or "Missing masks"))
                self.folderBatchStatus.text = "Microarchitecture batch row is missing required masks. Prepare segmentation, full/periosteal, and trabecular masks before running."
                return
            queued = {"mode": "independent", "row_index": int(row_index), "group": group}
        if not any(job.get("row_index") == int(row_index) for job in self._folderBatchQueue):
            self._folderBatchQueue.append(queued)
            self._set_folder_group_status(row_index, "Queued")
            self._set_folder_group_action(row_index, "Running" if self._folderBatchCurrent and self._folderBatchCurrent.get("row_index") == int(row_index) else "Queued")
        self._start_next_folder_batch_job()

    def _start_next_folder_batch_job(self):
        if self._folderBatchCurrent is not None or not self._folderBatchQueue:
            return
        job = self._folderBatchQueue.pop(0)
        self._folderBatchCurrent = job
        row_index = job["row_index"]
        self._set_folder_group_status(row_index, "Running")
        self._set_folder_group_action(row_index, "Running")
        self.folderRunButton.enabled = False
        try:
            dataset_root = str(self.folderDatasetRootEdit.text or "").strip()
            output_root = dataset_root
            if job.get("mode") == "independent":
                job["result_path"] = self._folder_result_path_for_group(dataset_root, job["group"], existing_only=False)
                self._folderBatchProcess = self.logic.run_folder_batch_job(
                    dataset_root,
                    subject_id=str(job["group"].get("subject", "")),
                    site=str(job["group"].get("site", "")),
                    session_id=str(job["group"].get("session", "")),
                    use_common_region=True,
                    force=not bool(self.folderSkipExistingCheck.checked),
                    thickness_method=str(self.folderThicknessMethodCombo.currentData),
                    thickness_backend=str(self.folderThicknessBackendCombo.currentData),
                    on_output=self._series_log,
                    on_finished=self._on_folder_batch_job_finished,
                )
            else:
                job_path, result_path = self._write_registered_series_job_for_rows(
                    job["group"]["rows"],
                    dataset_root=dataset_root,
                    output_root=output_root,
                )
                job["result_path"] = result_path
                self._folderBatchProcess = self.logic.run_registered_series_job(
                    job_path,
                    on_output=self._series_log,
                    on_finished=self._on_folder_batch_job_finished,
                )
        except Exception as exc:
            self._set_folder_group_status(row_index, "Failed")
            self._set_folder_group_action(row_index, "Run")
            self._folderBatchCurrent = None
            self.folderBatchStatus.text = f"Registered batch row failed to start: {exc}"
            self._start_next_folder_batch_job()

    def _on_folder_batch_job_finished(self, exit_code, exit_status):
        del exit_status
        if self._folderBatchCurrent is None:
            self._folderBatchProcess = None
            self.folderRunButton.enabled = True
            self.folderBatchStatus.text = f"Microarchitecture batch finished with exit code {int(exit_code)}."
            return
        row_index = self._folderBatchCurrent.get("row_index")
        self._set_folder_group_status(row_index, "Done" if int(exit_code) == 0 else "Failed")
        if int(exit_code) == 0:
            result_path = self._folderBatchCurrent.get("result_path")
            if result_path is None or not Path(str(result_path)).expanduser().exists():
                result_path = self._folder_result_path_for_group(
                    self.folderDatasetRootEdit.text,
                    self._folderBatchCurrent["group"],
                    existing_only=True,
                ) or self._folder_result_path_for_group(
                    self.folderDatasetRootEdit.text,
                    self._folderBatchCurrent["group"],
                    existing_only=False,
                )
            self._folderBatchGroups[int(row_index)]["result_path"] = str(result_path or "")
            self._set_folder_group_action(row_index, "Load")
        else:
            self._set_folder_group_action(row_index, "Run")
        self._folderBatchCurrent = None
        self._folderBatchProcess = None
        if self._folderBatchQueue:
            self.folderBatchStatus.text = f"Microarchitecture batch running; {len(self._folderBatchQueue)} queued."
            self._start_next_folder_batch_job()
        else:
            self.folderRunButton.enabled = True
            self.folderBatchStatus.text = "Microarchitecture batch queue finished."

    def _load_folder_batch_outputs(self, row_index):
        if row_index is None or not (0 <= int(row_index) < len(self._folderBatchGroups)):
            return
        group = self._folderBatchGroups[int(row_index)]
        result_path = Path(str(group.get("result_path") or "")).expanduser()
        if not result_path.exists():
            result_path = self._folder_result_path_for_group(self.folderDatasetRootEdit.text, group, existing_only=True)
            if result_path is None or not result_path.exists():
                self.folderBatchStatus.text = "No saved result is available for this batch row."
                return
            group["result_path"] = str(result_path)
        try:
            if result_path.suffix.lower() == ".csv":
                outputs = {"long_csv": str(result_path), "session_csvs": [str(result_path)]}
            else:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                outputs = dict(result.get("outputs", {}) or {})
            long_csv = str(outputs.get("long_csv") or "")
            loaded_table = None
            if long_csv and Path(long_csv).exists():
                loaded_table = self._measurement_table_from_csv(long_csv, Path(long_csv).stem)
                self._lastTableNode = loaded_table
                self.exportCsvButton.enabled = True
                self._show_measurement_table(loaded_table)
            loaded_maps = 0
            for csv_path in outputs.get("session_csvs", []) or []:
                maps_dir = Path(str(csv_path)).expanduser().parent.parent / "maps"
                if not maps_dir.exists():
                    continue
                for map_path in sorted(maps_dir.glob("*.nii.gz")):
                    name = f"{group.get('subject', 'microarchitecture')}_{Path(csv_path).parent.parent.name}_{map_path.name[:-7]}"
                    try:
                        loaded = slicer.util.loadVolume(str(map_path), {"name": name})
                    except TypeError:
                        loaded = slicer.util.loadVolume(str(map_path))
                    if isinstance(loaded, tuple):
                        success, node = loaded
                    else:
                        success, node = bool(loaded), loaded
                    if success:
                        try:
                            node.SetName(name)
                        except Exception:
                            pass
                        self._style_microarchitecture_map(node, self._map_role_from_path(map_path))
                        self._put_node_in_subject_hierarchy_folder(
                            node,
                            self._microarchitecture_map_folder_name(map_path, group),
                        )
                        loaded_maps += 1
            self.folderBatchStatus.text = f"Loaded registered measurements and {loaded_maps} map volume(s)."
        except Exception as exc:
            self.folderBatchStatus.text = f"Could not load registered batch outputs: {exc}"

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
        log_widget = getattr(self, "seriesLogText", None) or getattr(self, "folderBatchLogText", None)
        if log_widget is not None:
            log_widget.append(str(message).rstrip())
            log_widget.ensureCursorVisible()

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

    def _write_registered_series_job_for_rows(self, rows, *, dataset_root=None, output_root=None):
        dataset_root = str(dataset_root or "").strip()
        if not dataset_root:
            raise ValueError("Select a dataset root before running registered microarchitecture.")
        output_root = str(output_root or "").strip()
        root = self.logic.registered_microarchitecture_root(dataset_root, output_root)
        job_dir = root / "slicer_run_configs"
        job_dir.mkdir(parents=True, exist_ok=True)
        fd, job_path = tempfile.mkstemp(prefix="registered_microarchitecture_", suffix=".json", dir=str(job_dir))
        os.close(fd)
        result_path = Path(job_path).with_suffix(".result.json")
        job = {
            "dataset_root": dataset_root,
            "output_root": output_root,
            "rows": rows,
            "thickness_method": str(
                getattr(self, "folderThicknessMethodCombo", self.thicknessMethodCombo).currentData
            ),
            "thickness_backend": str(
                getattr(self, "folderThicknessBackendCombo", self.thicknessBackendCombo).currentData
            ),
            "common_region_only": str(getattr(self, "folderRegisteredWorkflowCombo", None).currentData) == "common_region_only"
            if getattr(self, "folderRegisteredWorkflowCombo", None) is not None
            else False,
            "result_path": str(result_path),
        }
        Path(job_path).write_text(json.dumps(job, indent=2), encoding="utf-8")
        return Path(job_path), result_path

    def _install_core(self):
        try:
            self._with_wait_cursor(self.logic.install_or_update_core)
        except Exception as exc:
            slicer.util.errorDisplay(f"Microarchitecture core installation failed:\n{exc}")
            self._log(f"[setup] microarchitecture core installation failed: {exc}")
            return
        self._log("[setup] microarchitecture core installed or updated.")
        self._update_dependency_ui()

    def _show_measurement_table(self, table_node):
        try:
            slicer.util.selectModule("Tables")
            tables_widget = slicer.modules.tables.widgetRepresentation()
            tables_widget.setCurrentTableNode(table_node)
        except Exception as exc:
            self._log(f"[microarchitecture] table created but could not switch to Tables module: {exc}")

    def _cache_measurement_rows(self, columns, rows):
        self._lastMeasurementColumns = [str(column) for column in columns]
        self._lastMeasurementRows = [dict(row) for row in rows]

    def _cache_measurement_rows_from_table(self, table_node):
        table = table_node.GetTable()
        columns = [table.GetColumnName(index) for index in range(table.GetNumberOfColumns())]
        rows = []
        for row_index in range(table.GetNumberOfRows()):
            row = {}
            for column_index, column_name in enumerate(columns):
                value = table.GetValue(row_index, column_index)
                row[column_name] = value.ToString() if hasattr(value, "ToString") else str(value)
            rows.append(row)
        self._cache_measurement_rows(columns, rows)

    def _measurement_table_from_rows(self, rows, name):
        if self._lastTableNode is not None:
            try:
                slicer.mrmlScene.RemoveNode(self._lastTableNode)
            except Exception:
                pass
            self._lastTableNode = None
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        table_node.SetAttribute("BoneImaging.Microarchitecture.Engine", "bone_microarchitecture")
        for column_name in self._lastMeasurementColumns:
            column = vtk.vtkStringArray()
            column.SetName(column_name)
            for row in rows:
                column.InsertNextValue(str(row.get(column_name, "")))
            table_node.GetTable().AddColumn(column)
        table_node.Modified()
        return table_node

    def _measurement_table_from_csv(self, path, name=None):
        with open(path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = [dict(row) for row in reader]
            columns = list(reader.fieldnames or [])
        self._cache_measurement_rows(columns, rows)
        self._lastTableBaseName = str(name or Path(path).stem)
        return self._measurement_table_from_rows(rows, self._lastTableBaseName)

    def _export_measurements_csv(self):
        if not self._lastMetrics and not self._lastMeasurementRows:
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
                    common_region_node=self.commonRegionMaskSelector.currentNode(),
                    common_region_segment_id=self._selected_segment_id(self.commonRegionSegmentCombo),
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
        self._lastTableBaseName = table_node.GetName()
        from bone_microarchitecture.results import SUMMARY_COLUMNS, measurement_rows

        self._cache_measurement_rows(SUMMARY_COLUMNS, measurement_rows(metrics, maps))
        self.exportCsvButton.enabled = True
        self._show_measurement_table(table_node)
        self._log(f"[microarchitecture] wrote table: {table_node.GetName()}")
        for key, node in output_nodes.items():
            if key != "table":
                self._log(f"[microarchitecture] wrote {key}: {node.GetName()}")

        for row in self._lastMeasurementRows:
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
        print("[registered] preparing workspace, registration, and common regions ...", flush=True)
        prepared = logic.prepare_registered_series_workspace(
            job["dataset_root"],
            job["output_root"],
            job["rows"],
            progress_callback=print_progress,
        )
        print(f"[registered] wrote manifest: {prepared['manifest']}", flush=True)
        for item in prepared.get("generated", []):
            print(f"[registered] {item['source']} {item['role']}: {item['path']}", flush=True)
        if bool(job.get("common_region_only")):
            print("[registered] common-region-only workflow complete; measurements skipped.", flush=True)
            _write_registered_series_worker_result(result_path, {"prepared": prepared, "outputs": {}})
            return 0
        print("[registered] running native-space common-region measurements ...", flush=True)
        outputs = logic.run_registered_series_microarchitecture(
            job["dataset_root"],
            job["output_root"],
            prepared["rows"],
            thickness_method=job["thickness_method"],
            thickness_backend=job["thickness_backend"],
            progress_callback=print_progress,
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
