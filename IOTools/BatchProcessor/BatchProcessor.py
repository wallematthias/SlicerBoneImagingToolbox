from __future__ import annotations

import csv
from dataclasses import replace
import json
import numpy as np
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import ctk
import SimpleITK as sitk
import qt
import slicer
import vtk

TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

def _local_repo_path(*parts):
    """Resolve sibling core repositories from normal checkouts or worktrees."""
    relative = Path(*parts)
    for base in (TOOLBOX_ROOT.parent, TOOLBOX_ROOT.parent.parent):
        candidate = base / relative
        if candidate.exists():
            return candidate
    return TOOLBOX_ROOT.parent / relative


DERIVATIVES_LOCAL_SRC = _local_repo_path("bone-imaging-derivatives", "src")
if DERIVATIVES_LOCAL_SRC.exists() and str(DERIVATIVES_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(DERIVATIVES_LOCAL_SRC))

SCANCO_IO_MODULE_DIR = TOOLBOX_ROOT / "IOTools" / "ScancoIO"
if SCANCO_IO_MODULE_DIR.exists() and str(SCANCO_IO_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SCANCO_IO_MODULE_DIR))

try:
    from bone_imaging_derivatives import (  # noqa: E402
        BatchArtifact,
        CaseKey,
        DerivativeManifest,
        DerivativeRecord,
        discover_derivative_artifacts,
        discover_manifests,
        discover_raw_xct_images,
        manifest_path,
        list_profiles,
        preferred_contours,
        prerequisite_status,
        read_manifest,
        record_output_path,
        write_manifest,
    )
except Exception as exc:
    _DERIVATIVES_IMPORT_ERROR = exc
    BatchArtifact = CaseKey = DerivativeManifest = DerivativeRecord = object

    def _missing_derivatives_runtime(*_args, **_kwargs):
        raise RuntimeError(
            "The Bone Imaging Derivative Contract runtime package is not installed. "
            "Open Bone Imaging > Setup and install/update runtime packages."
        ) from _DERIVATIVES_IMPORT_ERROR

    discover_derivative_artifacts = _missing_derivatives_runtime
    discover_manifests = _missing_derivatives_runtime
    discover_raw_xct_images = _missing_derivatives_runtime
    manifest_path = _missing_derivatives_runtime
    list_profiles = _missing_derivatives_runtime
    preferred_contours = _missing_derivatives_runtime
    prerequisite_status = _missing_derivatives_runtime
    read_manifest = _missing_derivatives_runtime
    record_output_path = _missing_derivatives_runtime
    write_manifest = _missing_derivatives_runtime
from SlicerBoneImagingToolboxLib.fea_batch import (  # noqa: E402
    build_parosol_case_commands,
    case_readiness,
    discover_fea_batch_cases,
    parosol_command_derivative_context,
    workflow_role_requirements,
)
from SlicerBoneImagingToolboxLib.remote_batch import (  # noqa: E402
    BACKEND_ENV_VAR as SLICER_BONE_BATCH_BACKEND,
    CONFIG_ENV_VAR as SLICER_BONE_BATCH_REMOTE_CONFIG,
    SshSlurmBatchBackend,
    load_remote_batch_config,
)
from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


MODULE_VERSION = "0.1.0"
TOOL_PROFILES = {
    "bone_contouring": (
        ("XtremeCT I", "XtremeCTI", False),
        ("XtremeCT II", "XtremeCTII", False),
        ("XtremeCT II - Geodesic", "XtremeCTII-Geodesic", False),
        ("XtremeCT II - LH", "XtremeCTII-LH", False),
    ),
    "mask_label_algebra": (
        ("Standard", "standard", False),
    ),
    "microarchitecture": (
        ("XtremeCT II", "xtremectii", False),
        ("XtremeCT II - registered", "xtremectii-registered", True),
    ),
    "timelapse": (
        ("Standard", "standard", True),
        ("ETH-UofC", "eth-uofc", True),
        ("Shriners", "shriners", True),
    ),
    "plate_rod": (
        ("Standard", "standard", False),
        ("Standard - registered", "standard-registered", True),
    ),
    "fea": (
        ("XtremeCT I", "XtremeCTI", False),
        ("XtremeCT II", "XtremeCTII", False),
        ("Load history 3", "load_history_3", False),
        ("Load history 6", "load_history_6", False),
    ),
    "mechanoregulation": (
        ("XtremeCT I", "XtremeCTI", False),
        ("XtremeCT II", "XtremeCTII", False),
        ("Load history 3", "load_history_3", False),
        ("Load history 6", "load_history_6", False),
    ),
}
_REGISTRATION_FAMILY_PRIORITY = {"ImportedRegistration": 0, "Registration": 1}
_SEGMENT_COLORS = {
    "full": (0.2, 0.8, 0.25),
    "trab": (0.0, 0.75, 1.0),
    "cort": (1.0, 0.55, 0.1),
    "seg": (1.0, 0.95, 0.3),
    "fea-materials": (0.92, 0.62, 0.15),
    "common_region": (0.72, 0.42, 1.0),
    "resorption": (1.0, 0.05, 0.70),
    "formation": (1.0, 0.48, 0.0),
}
_MICROARCHITECTURE_MEASUREMENT_NAME = re.compile(
    r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_voi-(?P<voi>[^_]+)"
    r"(?:_stack-(?P<stack>\d+))?_measurements\.csv$",
    re.IGNORECASE,
)
_REGISTRATION_TRANSFORM_NAME = re.compile(
    r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_voi-(?P<voi>[^_]+)"
    r"(?:_stack-(?P<stack>\d+))?_.*\.tfm$",
    re.IGNORECASE,
)
_TIMELAPSE_PAIRWISE_TABLE_NAME = re.compile(
    r"^sub-(?P<subject>[^_]+)_voi-(?P<voi>[^_]+)_pairwise_remodelling\.csv$",
    re.IGNORECASE,
)
_TIMELAPSE_REMODELLING_NAME = re.compile(
    r"^sub-(?P<subject>[^_]+)_voi-(?P<voi>[^_]+)_desc-(?P<roi>.+?)_t0-(?P<t0>[^_]+)_t1-(?P<t1>[^_]+)_.*_remodelling\.(?:nii\.gz|nii|nrrd|nhdr|mha|mhd|aim)$",
    re.IGNORECASE,
)
_SUPPRESSED_PROCESS_OUTPUT_MARKERS = (
    "Possible incompatible factory load:",
    "Error ImageIO factory did not return an ImageIOBase: MRMLIDImageIO",
    "Running itk version :",
    "Loaded factory version:",
    "Loading factory:",
    "itkObjectFactoryBase.cxx",
    "libMRMLIDIOPlugin.dylib",
)


class BatchProcessor(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Batch Processor"
        parent.categories = ["Bone Imaging.I/O"]
        parent.index = 35
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Inspect a normalized bone imaging dataset and route rows to toolbox batch workflows.\n"
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = "Author: Matthias Walle. Part of the Bone Imaging Toolbox for 3D Slicer."


class BatchProcessorLogic(ScriptedLoadableModuleLogic):
    _CLI_COMMANDS = {
        "bone_contouring": ("bone_contouring.cli", "run-batch"),
        "mask_label_algebra": ("bone_contouring.cli", "mask-label-algebra"),
        "microarchitecture": ("bone_microarchitecture.cli", "run-batch"),
        "plate_rod": ("plate_rod_thinning.cli", "run-batch"),
        "timelapse": ("timelapsedhrpqct.cli", "run"),
        "fea": ("parosol_py.cli", ""),
        "mechanoregulation": ("bonemechreg.cli", "run"),
    }

    @staticmethod
    def normalized_dataset_status(dataset_root):
        """Return whether ``dataset_root`` follows the normalized sub/ses/xct layout."""
        root = Path(str(dataset_root or "")).expanduser()
        if root.name == "derivatives":
            root = root.parent
        if not root.exists():
            return False, "Select an existing dataset root."
        subject_dirs = sorted(path for path in root.glob("sub-*") if path.is_dir())
        if not subject_dirs:
            return False, "Dataset is not normalized yet. Use Dataset Naming Helper first."
        image_count = 0
        for subject_dir in subject_dirs:
            for session_dir in sorted(subject_dir.glob("ses-*")):
                xct_dir = session_dir / "xct"
                if not xct_dir.is_dir():
                    continue
                image_count += sum(1 for path in xct_dir.iterdir() if BatchProcessorLogic._is_xct_image(path))
        if image_count == 0:
            return False, "No modality images were found under sub-*/ses-*/xct."
        return True, f"Normalized dataset with {len(subject_dirs)} subject(s) and {image_count} modality image(s)."

    @staticmethod
    def _is_xct_image(path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name.lower()
        return name.endswith((".aim", ".isq", ".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".nhdr"))

    def discover_rows(self, dataset_root, *, tool: str, profile: str, registered: bool):
        """Return display rows from shared artifact discovery."""
        root = Path(str(dataset_root or "")).expanduser()
        if root.name == "derivatives":
            root = root.parent
        tool = str(tool or "")
        registered = self.profile_requests_registration(tool, profile) or bool(registered and tool in {"microarchitecture", "plate_rod", "timelapse"})
        if tool == "fea":
            ok, message = self.normalized_dataset_status(root)
            if not ok:
                return [], message
            return self._fea_table_rows(root, profile)
        if tool == "mechanoregulation":
            return self._mechanoregulation_table_rows(root, profile)
        ok, message = self.normalized_dataset_status(root)
        if not ok:
            return [], message
        if tool == "mask_label_algebra":
            return self._mask_label_algebra_table_rows(root)
        images = discover_raw_xct_images(root)
        contour_artifacts = (
            *discover_derivative_artifacts(root, "IPLContours"),
            *discover_derivative_artifacts(root, "ImportedContours"),
            *discover_derivative_artifacts(root, "BoneContours"),
        )
        registration_records = self._discover_registration_records(root)
        common_region_records = self._discover_common_region_records(root)
        existing_outputs = self._discover_existing_outputs(root, self._output_family_for_tool(tool))
        rows = self._table_rows_for_tool(
            images,
            contour_artifacts,
            registration_records,
            common_region_records,
            self._existing_outputs_for_profile(tool, registered, existing_outputs),
            tool=tool,
            profile=profile,
            registered=registered,
        )
        return rows, f"Discovered {len(rows)} row(s)."

    @staticmethod
    def profiles_for_tool(tool: str):
        return TOOL_PROFILES.get(str(tool or ""), (("Standard", "standard", False),))

    @staticmethod
    def profile_requests_registration(tool: str, profile: str) -> bool:
        if str(tool or "") == "timelapse":
            return True
        profile = str(profile or "")
        return any(value == profile and registered for _label, value, registered in BatchProcessorLogic.profiles_for_tool(tool))

    @staticmethod
    def profile_groups_all_stacks(tool: str, profile: str) -> bool:
        if str(tool or "") != "timelapse":
            return False
        return str(profile or "").strip().lower() in {"multistack", "ped-fx"}

    @staticmethod
    def _output_family_for_tool(tool: str) -> str:
        return {
            "bone_contouring": "BoneContours",
            "mask_label_algebra": "BoneContours",
            "microarchitecture": "Microarchitecture",
            "timelapse": "Timelapse",
            "plate_rod": "PlateRodMorphometry",
            "fea": "FEA",
            "mechanoregulation": "Mechanoregulation",
        }.get(str(tool or ""), "Microarchitecture")

    @staticmethod
    def _required_roles_for_tool(tool: str, profile: str) -> tuple[str, ...]:
        del profile
        if tool in {"bone_contouring", "mask_label_algebra"}:
            return ()
        if tool == "timelapse":
            return ("segmentation", "full", "trab", "cort")
        if tool == "plate_rod":
            return ("segmentation", "trab")
        if tool == "fea":
            return ()
        return ("segmentation", "full", "trab", "cort")

    @staticmethod
    def _fea_table_rows(root: Path, profile: str):
        cases = discover_fea_batch_cases(root)
        rows = []
        supported_roles = workflow_role_requirements(profile).get("image")
        preferred_roles = tuple(supported_roles.preferred_roles) if supported_roles else ("material_labelmap",)
        for case in cases:
            source = case.first_artifact(preferred_roles)
            if source is None:
                continue
            ok, missing = case_readiness(case, profile)
            status = "Ready" if ok else "Missing HOM_LS"
            action = "Run" if ok else "Missing"
            output_paths = BatchProcessorLogic._fea_output_paths_for_case(root, case, profile)
            if ok and output_paths:
                status = "Done"
                action = "Load"
            row = {
                "action": action,
                "subject": case.subject_id,
                "session": case.session_id,
                "session_value": case.session_id,
                "voi": case.site,
                "voi_value": case.site,
                "registered": False,
                "status": status if not missing else "Missing HOM_LS",
                "input": f"source={Path(source.path).name}" if source is not None else "source=missing HOM_LS",
                "image_path": str(source.path) if source is not None else "",
                "fea_case": case,
                "profile": str(profile or "").strip(),
            }
            if output_paths:
                row["output_paths"] = [str(path) for path in output_paths]
            rows.append(row)
        return rows, f"Discovered {len(rows)} FEA source row(s)."

    @staticmethod
    def _fea_command_for_row(dataset_root, profile: str, row: dict, force: bool = False) -> list[str]:
        del force
        case = row.get("fea_case")
        if case is None:
            source = str(row.get("image_path") or "").strip()
            if not source:
                raise ValueError("FEA row is missing a HOM_LS/material label map.")
            case = SimpleNamespace(
                subject_id=str(row.get("subject") or ""),
                site=str(row.get("voi_value", row.get("voi")) or ""),
                session_id=str(row.get("session_value", row.get("session")) or ""),
                first_artifact=lambda roles: SimpleNamespace(path=source),
            )
        commands = build_parosol_case_commands(
            dataset_root,
            [case],
            workflow=str(profile or "").strip(),
            selected_roles=None,
            dry_run=False,
        )
        if not commands:
            raise ValueError("FEA row is missing a HOM_LS/material label map.")
        return ["-m", "parosol_py.cli", *commands[0]]

    @staticmethod
    def _mechanoregulation_core():
        local_src = _local_repo_path("BoneMechanoregulation")
        if local_src.exists() and str(local_src) not in sys.path:
            sys.path.insert(0, str(local_src))
        from bonemechreg.timelapse import available_case_rois, case_outputs, discover_timelapse_cases

        return discover_timelapse_cases, case_outputs, available_case_rois

    @staticmethod
    def _mechanoregulation_table_rows(root: Path, profile: str):
        discover_timelapse_cases, case_outputs, available_case_rois = BatchProcessorLogic._mechanoregulation_core()
        cases = discover_timelapse_cases(root, sed_profile=str(profile or "").strip())
        rows = []
        for case in cases:
            sed_path = Path(case.baseline_sed_path) if case.baseline_sed_path else None
            sed_ready = sed_path is not None and sed_path.exists()
            output_paths = BatchProcessorLogic._mechanoregulation_output_paths_for_case(case, case_outputs, available_case_rois)
            action = "Run" if sed_ready else "Missing"
            status = "Ready" if sed_ready else "Missing SED"
            if sed_ready and output_paths:
                action = "Load"
                status = "Done"
            session_pair = BatchProcessorLogic._mechanoregulation_session_pair(case)
            row = {
                "action": action,
                "subject": str(case.subject_id).removeprefix("sub-"),
                "session": session_pair,
                "session_value": session_pair,
                "voi": str(case.site or BatchProcessorLogic._mechanoregulation_site_from_case(case)).lower(),
                "voi_value": str(case.site or BatchProcessorLogic._mechanoregulation_site_from_case(case)).lower(),
                "registered": False,
                "status": status,
                "input": BatchProcessorLogic._mechanoregulation_input_text(case, sed_path, profile, available_case_rois),
                "image_path": str(case.remodelling_image_path),
                "sed_path": str(sed_path) if sed_path is not None and sed_path.exists() else "",
                "mechanoregulation_case_id": str(case.case_id),
                "profile": str(profile or "").strip(),
            }
            if output_paths:
                row["output_paths"] = [str(path) for path in output_paths]
            rows.append(row)
        return rows, f"Discovered {len(rows)} mechanoregulation row(s)."

    @staticmethod
    def _mechanoregulation_input_text(case, sed_path: Path | None, profile: str, available_case_rois=None) -> str:
        parts = [f"remodelling={Path(case.remodelling_image_path).name}"]
        if sed_path is not None and sed_path.exists():
            parts.append(f"sed={sed_path.name}")
        else:
            parts.append(f"sed=missing {str(profile or '').strip()} baseline SED")
        try:
            roi_paths = available_case_rois(case) if callable(available_case_rois) else {}
        except Exception:
            roi_paths = {}
        for roi, path in roi_paths.items():
            label = str(roi or "roi").strip().lower()
            if path is not None and Path(path).exists():
                parts.append(f"{label}={Path(path).name}")
            else:
                parts.append(f"{label}=whole remodelling grid")
        return "\n".join(parts)

    @staticmethod
    def _mechanoregulation_session_pair(case) -> str:
        t0 = str(getattr(case, "baseline_session_id", "") or "").strip()
        t1 = str(getattr(case, "followup_session_id", "") or "").strip()
        if not t0 or not t1:
            match = re.search(r"_t0-(?P<t0>[^_]+)_t1-(?P<t1>[^_]+)_", Path(case.remodelling_image_path).name)
            if match:
                t0 = t0 or match.group("t0")
                t1 = t1 or match.group("t1")
        return f"{t0}-{t1}" if t0 or t1 else str(case.case_id)

    @staticmethod
    def _mechanoregulation_site_from_case(case) -> str:
        match = re.search(r"(?:^|_)voi-(?P<site>[^_]+)", Path(case.remodelling_image_path).name, flags=re.IGNORECASE)
        return match.group("site") if match else ""

    @staticmethod
    def _mechanoregulation_output_paths_for_case(case, case_outputs, available_case_rois) -> list[Path]:
        expected_paths: list[Path] = []
        for roi in available_case_rois(case):
            outputs = case_outputs(case, roi=roi)
            for key in ("csv", "curves", "schulte_curves", "summary"):
                expected_paths.append(Path(outputs[key]))
        if not expected_paths or not all(path.exists() for path in expected_paths):
            return []
        return expected_paths

    @staticmethod
    def _mechanoregulation_command_for_row(dataset_root, profile: str, row: dict, force: bool = False) -> list[str]:
        args = [
            "-m",
            "bonemechreg.cli",
            "run",
            str(BatchProcessorLogic._dataset_root(dataset_root)),
            "--profile",
            str(profile or "").strip(),
            "--case-id",
            str(row.get("mechanoregulation_case_id") or row.get("image_path") or "").strip(),
            "--verbose",
        ]
        if force:
            args.append("--reanalyze")
        return args

    @staticmethod
    def publish_fea_batch_outputs(dataset_root, row: dict, profile: str) -> list[Path]:
        """Publish a completed ParOSol run into stable FEA derivative artifacts."""
        root = BatchProcessorLogic._dataset_root(dataset_root)
        case = BatchProcessorLogic._fea_case_from_row(row)
        if case is None:
            return []
        run_dir = BatchProcessorLogic._fea_run_dir_for_case(root, case, profile)
        if run_dir is None or not run_dir.exists():
            return []
        result_json = run_dir / "result.json"
        result_data = BatchProcessorLogic._read_json_file(result_json)
        sed_source = BatchProcessorLogic._fea_sed_source(run_dir, result_data)
        map_path, table_path = BatchProcessorLogic._fea_canonical_output_paths(root, case, profile)
        published: list[Path] = []
        inputs = BatchProcessorLogic._fea_publish_inputs(case, result_json)
        if sed_source is not None and sed_source.exists():
            map_path.parent.mkdir(parents=True, exist_ok=True)
            if sed_source.resolve() != map_path.resolve():
                shutil.copy2(sed_source, map_path)
            published.append(map_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        BatchProcessorLogic._write_fea_summary_csv(table_path, case, profile, result_data)
        published.append(table_path)
        BatchProcessorLogic._write_fea_manifest(root, case, profile, published, inputs, result_data)
        return published

    @staticmethod
    def _fea_case_from_row(row: dict):
        case = row.get("fea_case")
        if case is not None:
            return case
        source = str(row.get("image_path") or "").strip()
        if not source:
            return None
        return SimpleNamespace(
            subject_id=str(row.get("subject") or ""),
            site=str(row.get("voi_value", row.get("voi")) or ""),
            session_id=str(row.get("session_value", row.get("session")) or ""),
            first_artifact=lambda roles: SimpleNamespace(path=source),
        )

    @staticmethod
    def _fea_run_dir_for_case(root: Path, case, profile: str) -> Path | None:
        commands = build_parosol_case_commands(root, [case], workflow=str(profile or "").strip())
        if not commands:
            return None
        context = parosol_command_derivative_context(commands[0])
        output_dir_text = context.get("output_dir") if context else ""
        return Path(output_dir_text) if output_dir_text else None

    @staticmethod
    def _fea_canonical_output_paths(root: Path, case, profile: str) -> tuple[Path, Path]:
        subject = str(case.subject_id or "").strip()
        session = str(case.session_id or "").strip()
        site = str(case.site or "").strip()
        profile_slug = BatchProcessorLogic._filename_token(profile)
        stem = f"sub-{subject}_ses-{session}_voi-{site}_desc-{profile_slug}"
        base = record_output_path(root, "FEA", subject, site, f"ses-{session}")
        return (
            base / "maps" / f"{stem}_map-sed.nii.gz",
            base / "measurements" / f"{stem}_fea.csv",
        )

    @staticmethod
    def _filename_token(value) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-") or "parosol"

    @staticmethod
    def _read_json_file(path: Path) -> dict:
        try:
            if Path(path).exists():
                return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    @staticmethod
    def _fea_sed_source(run_dir: Path, result_data: dict) -> Path | None:
        exported = result_data.get("outputs", {}).get("exported", {}) if isinstance(result_data, dict) else {}
        load_history = result_data.get("postprocess", {}).get("load_history", {}) if isinstance(result_data, dict) else {}
        final_rerun = load_history.get("final_rerun", {}) if isinstance(load_history, dict) else {}
        for candidate in (
            exported.get("sed"),
            final_rerun.get("output") if isinstance(final_rerun, dict) else None,
            load_history.get("output") if isinstance(load_history, dict) else None,
            run_dir / "fields" / "sed.nii.gz",
            run_dir / "sed.nii.gz",
        ):
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.exists() and path.name.lower().endswith((".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")):
                return path
        return None

    @staticmethod
    def _fea_publish_inputs(case, result_json: Path) -> tuple[str, ...]:
        inputs = []
        try:
            source = case.first_artifact(("material_labelmap", "hom_ls_model", "model_labelmap", "labelmap"))
        except Exception:
            source = None
        if source is not None and getattr(source, "path", None):
            inputs.append(str(source.path))
        if result_json.exists():
            inputs.append(str(result_json))
        return tuple(inputs)

    @staticmethod
    def _write_fea_summary_csv(path: Path, case, profile: str, result_data: dict) -> None:
        headers = [
            "Sample",
            "Profile",
            "Stiffness (N/mm)",
            "Failure load (N)",
        ]
        is_load_history = str(profile or "").strip().lower().startswith("load_history")
        if is_load_history:
            headers.extend(["Scale factors", "Input load amplitudes", "Estimated loads"])
        row = {
            "Sample": f"sub-{case.subject_id} ses-{case.session_id} voi-{case.site}",
            "Profile": str(profile or ""),
            "Stiffness (N/mm)": BatchProcessorLogic._fea_value(
                result_data,
                ("mechanics", "generalized_stiffness", "value"),
                ("mechanics", "stiffness", "z"),
            ),
            "Failure load (N)": BatchProcessorLogic._fea_value(
                result_data,
                ("failure", "failure_generalized_load", "value"),
                ("failure", "failure_load", "z"),
            ),
        }
        load_history = BatchProcessorLogic._fea_load_history(result_data)
        details = load_history.get("details", {}) if isinstance(load_history, dict) else {}
        results = load_history.get("results", {}) if isinstance(load_history, dict) else {}
        if is_load_history:
            row["Scale factors"] = BatchProcessorLogic._join_fea_values(details.get("scaling_factors"))
            row["Input load amplitudes"] = BatchProcessorLogic._join_fea_values(details.get("input_load_amplitudes"))
            row["Estimated loads"] = BatchProcessorLogic._join_fea_values(
                [
                    item.get("value")
                    for item in (results.get("estimated_loads") or [])
                    if isinstance(item, dict) and item.get("value") is not None
                ]
            )
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _fea_load_history(data: dict) -> dict:
        if not isinstance(data, dict):
            return {}
        load_history = data.get("postprocess", {}).get("load_history", {})
        return load_history if isinstance(load_history, dict) else {}

    @staticmethod
    def _join_fea_values(values) -> str:
        if values in (None, ""):
            return ""
        if isinstance(values, (str, int, float)):
            return str(values)
        try:
            return ";".join(str(value) for value in values)
        except TypeError:
            return str(values)

    @staticmethod
    def _fea_value(data: dict, *paths: tuple[str, ...]):
        for keys in paths:
            value = data
            for key in keys:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if value is not None:
                return value
        return ""

    @staticmethod
    def _write_fea_manifest(root: Path, case, profile: str, paths: list[Path], inputs: tuple[str, ...], result_data: dict) -> None:
        records = []
        for path in paths:
            is_table = path.suffix.lower() == ".csv"
            records.append(
                DerivativeRecord(
                    "FEA",
                    "summary_table" if is_table else "sed_map",
                    str(case.subject_id),
                    str(case.site),
                    str(case.session_id),
                    None,
                    "table" if is_table else "native",
                    path,
                    "generated",
                    inputs=inputs,
                    metadata={
                        "profile": str(profile or ""),
                        "stiffness_n_per_mm": BatchProcessorLogic._fea_value(
                            result_data,
                            ("mechanics", "generalized_stiffness", "value"),
                            ("mechanics", "stiffness", "z"),
                        ),
                        "failure_load_n": BatchProcessorLogic._fea_value(
                            result_data,
                            ("failure", "failure_generalized_load", "value"),
                            ("failure", "failure_load", "z"),
                        ),
                        "load_history_scale_factors": BatchProcessorLogic._join_fea_values(
                            BatchProcessorLogic._fea_load_history(result_data).get("details", {}).get("scaling_factors")
                        ),
                        "load_history_input_load_amplitudes": BatchProcessorLogic._join_fea_values(
                            BatchProcessorLogic._fea_load_history(result_data).get("details", {}).get("input_load_amplitudes")
                        ),
                    },
                    content_type="table" if is_table else "image",
                    software={"name": "SlicerBoneImagingToolbox", "version": MODULE_VERSION},
                )
            )
        path = manifest_path(root, "FEA")
        existing = []
        if path.exists():
            try:
                existing = list(read_manifest(path).records)
            except Exception:
                existing = []
        replacement_keys = {
            (record.role, record.subject_id, record.site, record.session_id, record.path.name)
            for record in records
        }
        kept = [
            record
            for record in existing
            if (record.role, record.subject_id, record.site, record.session_id, record.path.name) not in replacement_keys
        ]
        manifest = DerivativeManifest.create(
            "FEA",
            root,
            {"name": "SlicerBoneImagingToolbox", "version": MODULE_VERSION},
            records=tuple(kept + records),
        )
        write_manifest(manifest, path)

    @staticmethod
    def _role_summary(contours):
        order = ("segmentation", "full", "trab", "cort")
        found = [role for role in order if role in contours]
        found.extend(sorted(role for role in contours if role not in order))
        return ", ".join(found)

    @staticmethod
    def _status_text(result):
        if result is None or result.status == "ready":
            return "Ready"
        if result.status == "loadable":
            return "Done"
        if result.status == "missing":
            return f"Missing {', '.join(result.missing_roles)}"
        if result.status == "review":
            review = ", ".join(result.review_roles) if result.review_roles else "inputs"
            return f"Review {review}"
        return str(result.status).title()

    @staticmethod
    def _action_for_status(result):
        if result is not None and result.status == "loadable":
            return "Load"
        if result is not None and result.status in {"missing", "review"}:
            return result.status.title()
        return "Run"

    @staticmethod
    def _voi_text(key):
        if key.stack_index is None:
            return key.voi
        return f"{key.voi} stack-{int(key.stack_index):02d}"

    @staticmethod
    def _registration_key(record):
        return record.subject_id, record.session_id, BatchProcessorLogic._lookup_site(record.site), record.stack_index

    @staticmethod
    def _case_lookup_key(key):
        return key.subject_id, key.session_id, BatchProcessorLogic._lookup_site(key.voi), key.stack_index

    @staticmethod
    def _lookup_site(site):
        return re.sub(r"[^a-z0-9]+", "", str(site or "").strip().lower())

    @staticmethod
    def _registration_records_by_key(registration_records):
        grouped = {}
        for record in registration_records:
            if record.role not in {"transform_pairwise", "transform_to_reference"}:
                continue
            key = BatchProcessorLogic._registration_key(record)
            grouped.setdefault(key, []).append(record)
        return {key: BatchProcessorLogic._preferred_registration_records(records) for key, records in grouped.items()}

    @staticmethod
    def _registered_images_by_key(registration_records):
        grouped = {}
        for record in registration_records:
            if record.role not in {"transformed_image", "source_image_view"}:
                continue
            if hasattr(record, "path") and not Path(record.path).exists():
                continue
            key = BatchProcessorLogic._registration_key(record)
            grouped.setdefault(key, []).append(record)
        return grouped

    @staticmethod
    def _common_region_records_by_key(common_region_records):
        grouped = {}
        for record in common_region_records:
            if record.role != "scan_region_native_common":
                continue
            if hasattr(record, "path") and not Path(record.path).exists():
                continue
            key = BatchProcessorLogic._registration_key(record)
            grouped.setdefault(key, []).append(record)
        return grouped

    @staticmethod
    def _discover_registration_records(root):
        records_by_path = {}
        for family in ("ImportedRegistration", "Registration"):
            for artifact in discover_derivative_artifacts(root, family):
                if artifact.role not in {"transform_pairwise", "transform_to_reference"}:
                    continue
                records_by_path.setdefault(
                    artifact.path.resolve(),
                    SimpleNamespace(
                        role=artifact.role,
                        subject_id=artifact.key.subject_id,
                        session_id=artifact.key.session_id,
                        site=artifact.key.voi,
                        stack_index=artifact.key.stack_index,
                        path=artifact.path,
                        derivative=family,
                    ),
                )
            for path in sorted((Path(root) / "derivatives" / family).glob("sub-*/ses-*/xct/*/*.tfm")):
                match = _REGISTRATION_TRANSFORM_NAME.match(path.name)
                if match is None:
                    continue
                stack = match.group("stack")
                role = "transform_pairwise" if path.parent.name == "pairwise" else "transform_to_reference"
                records_by_path.setdefault(
                    path.resolve(),
                    SimpleNamespace(
                        role=role,
                        subject_id=match.group("subject"),
                        session_id=match.group("session"),
                        site=match.group("voi").lower(),
                        stack_index=int(stack) if stack else None,
                        path=path,
                        derivative=family,
                    )
                )
        return tuple(records_by_path.values())

    @staticmethod
    def _preferred_registration_records(records):
        by_role = {}
        for record in sorted(
            records,
            key=lambda record: (
                _REGISTRATION_FAMILY_PRIORITY.get(getattr(record, "derivative", ""), 2),
                str(getattr(record, "path", "")),
            ),
        ):
            by_role.setdefault(record.role, record)
        return list(by_role.values())

    @staticmethod
    def _discover_common_region_records(root):
        return tuple(
            record
            for manifest in discover_manifests(root)
            if manifest.derivative_family == "CommonRegion"
            for record in manifest.records
            if record.role == "scan_region_native_common"
        )

    @staticmethod
    def _discover_existing_outputs(root, derivative_family):
        outputs = list(discover_derivative_artifacts(root, derivative_family))
        if derivative_family == "Timelapse":
            for path in sorted((Path(root) / "derivatives" / "Timelapse").glob("sub-*/xct/analysis/*_pairwise_remodelling.csv")):
                match = _TIMELAPSE_PAIRWISE_TABLE_NAME.match(path.name)
                if match is None:
                    continue
                outputs.append(
                    BatchArtifact(
                        path,
                        CaseKey(match.group("subject"), "__series__", match.group("voi").lower(), None),
                        "pairwise_remodelling_table",
                        "Timelapse",
                    )
                )
            for path in sorted((Path(root) / "derivatives" / "Timelapse").glob("sub-*/xct/analysis/visualize/*_remodelling.*")):
                match = _TIMELAPSE_REMODELLING_NAME.match(path.name)
                if match is None:
                    continue
                outputs.append(
                    BatchArtifact(
                        path,
                        CaseKey(match.group("subject"), match.group("t0"), match.group("voi").lower(), None),
                        "remodelling_image",
                        "Timelapse",
                    )
                )
            return tuple(outputs)
        if derivative_family != "Microarchitecture":
            return tuple(outputs)
        micro_root = Path(root) / "derivatives" / "Microarchitecture"
        measurement_patterns = (
            "sub-*/ses-*/xct/measurements/*_measurements.csv",
            "sub-*/ses-*/xct/registered_measurements/*_measurements.csv",
        )
        for pattern in measurement_patterns:
            for path in sorted(micro_root.glob(pattern)):
                match = _MICROARCHITECTURE_MEASUREMENT_NAME.match(path.name)
                if match is None:
                    continue
                stack = match.group("stack")
                path_text = str(path).replace("\\", "/").lower()
                outputs.append(
                    BatchArtifact(
                        path,
                        CaseKey(
                            match.group("subject"),
                            match.group("session"),
                            match.group("voi").lower(),
                            int(stack) if stack else None,
                        ),
                        "measurements_table",
                        "Microarchitecture",
                        metadata={"use_common_region": "/registered_measurements/" in path_text},
                    )
                )
        map_patterns = (
            "sub-*/ses-*/xct/maps/*_map-*.nii.gz",
        )
        for pattern in map_patterns:
            for path in sorted(micro_root.glob(pattern)):
                match = _MICROARCHITECTURE_MEASUREMENT_NAME.match(path.name.split("_map-", 1)[0] + "_measurements.csv")
                if match is None:
                    continue
                stack = match.group("stack")
                path_text = str(path).replace("\\", "/").lower()
                outputs.append(
                    BatchArtifact(
                        path,
                        CaseKey(
                            match.group("subject"),
                            match.group("session"),
                            match.group("voi").lower(),
                            int(stack) if stack else None,
                        ),
                        "microarchitecture_map",
                        "Microarchitecture",
                        metadata={"use_common_region": False},
                    )
                )
        return tuple(outputs)

    @staticmethod
    def _existing_outputs_for_profile(tool: str, registered: bool, existing_outputs):
        if str(tool or "") not in {"microarchitecture", "plate_rod"}:
            return tuple(existing_outputs)
        filtered = []
        for artifact in existing_outputs:
            if BatchProcessorLogic._is_map_output_for_tool(tool, artifact):
                filtered.append(artifact)
                continue
            path_text = str(artifact.path).replace("\\", "/").lower()
            common_region = bool(artifact.metadata.get("use_common_region"))
            path_registered = "/registered_measurements/" in path_text
            if registered and (common_region or path_registered):
                filtered.append(artifact)
            elif not registered and not common_region and not path_registered:
                filtered.append(artifact)
        return tuple(filtered)

    @staticmethod
    def _is_microarchitecture_map_output(artifact) -> bool:
        role = str(getattr(artifact, "role", "") or "")
        return role == "microarchitecture_map" or role.endswith("_map")

    @staticmethod
    def _is_plate_rod_map_output(artifact) -> bool:
        return str(getattr(artifact, "role", "") or "") in {"plate_rod_label_map", "skeleton_map"}

    @staticmethod
    def _is_map_output_for_tool(tool: str, artifact) -> bool:
        if str(tool or "") == "microarchitecture":
            return BatchProcessorLogic._is_microarchitecture_map_output(artifact)
        if str(tool or "") == "plate_rod":
            return BatchProcessorLogic._is_plate_rod_map_output(artifact)
        return False

    @staticmethod
    def _is_measurement_output_for_tool(tool: str, artifact) -> bool:
        role = str(getattr(artifact, "role", "") or "")
        if str(tool or "") == "plate_rod":
            return role == "plate_rod_measurements_table"
        return role == "measurements_table"

    def rediscover_row_output_paths(self, dataset_root, tool: str, row: dict) -> list[str]:
        """Return output paths currently available on disk for one displayed row."""
        root = self._dataset_root(dataset_root)
        if tool == "timelapse":
            return self._timelapse_output_paths_for_row(root, row)
        if tool == "mask_label_algebra":
            return [str(path) for path in self._imported_contour_paths_for_row(root, row)]
        if tool == "fea":
            return [str(path) for path in self._fea_output_paths_for_row(root, row)]
        if tool == "mechanoregulation":
            return [str(path) for path in self._mechanoregulation_output_paths_for_row(root, row)]
        key = self._row_case_key(row)
        if key is None:
            return []
        outputs = self._discover_existing_outputs(root, self._output_family_for_tool(tool))
        outputs = self._existing_outputs_for_profile(tool, bool(row.get("registered")), outputs)
        return [str(artifact.path) for artifact in outputs if artifact.key == key]

    @staticmethod
    def _imported_contour_paths_for_row(root: Path, row: dict) -> list[Path]:
        key = BatchProcessorLogic._row_case_key(row)
        if key is None:
            return []
        lookup_key = BatchProcessorLogic._case_lookup_key(key)
        records = []
        for artifact in discover_derivative_artifacts(root, "ImportedContours"):
            if BatchProcessorLogic._case_lookup_key(artifact.key) != lookup_key:
                continue
            if str(getattr(artifact, "content_type", "") or "").lower() not in {"", "mask", "label"}:
                continue
            if not Path(artifact.path).exists():
                continue
            records.append(artifact)
        return sorted({Path(artifact.path) for artifact in records}, key=lambda path: path.name)

    def common_region_paths_for_row(self, dataset_root, row: dict) -> list[str]:
        """Return native common-region masks matching one registered row."""
        root = self._dataset_root(dataset_root)
        key = self._row_case_key(row)
        if key is None:
            return []
        lookup_key = self._case_lookup_key(key)
        records = self._common_region_records_by_key(self._discover_common_region_records(root))
        return [
            str(record.path)
            for record in records.get(lookup_key, [])
            if Path(record.path).exists()
        ]

    @staticmethod
    def _fea_output_paths_for_row(root: Path, row: dict) -> list[Path]:
        profile = str(row.get("profile") or "").strip()
        if not profile:
            return []
        case = BatchProcessorLogic._fea_case_from_row(row)
        if case is None:
            return []
        return BatchProcessorLogic._fea_output_paths_for_case(root, case, profile)

    @staticmethod
    def _mechanoregulation_output_paths_for_row(root: Path, row: dict) -> list[Path]:
        case_id = str(row.get("mechanoregulation_case_id") or "").strip()
        subject = str(row.get("subject") or "").strip().removeprefix("sub-")
        if not case_id or not subject:
            return []
        base = record_output_path(root, "Mechanoregulation", subject, str(row.get("voi_value", row.get("voi")) or ""), "runs", case_id)
        expected = []
        for pattern in (
            "*_mechanoregulation_summary.csv",
            "*_conditional_curves.png",
            "*_schulte_binned_curves.png",
            "*_mechanoregulation_summary.json",
        ):
            matches = sorted(base.glob(pattern))
            if not matches:
                return []
            expected.extend(matches)
        return expected

    @staticmethod
    def _fea_output_paths_for_case(root: Path, case, profile: str) -> list[Path]:
        map_path, table_path = BatchProcessorLogic._fea_canonical_output_paths(root, case, profile)
        paths: list[Path] = []
        if map_path.exists():
            paths.append(map_path)
        if table_path.exists():
            paths.append(table_path)
        if paths:
            return paths
        output_dir = BatchProcessorLogic._fea_run_dir_for_case(root, case, profile)
        if output_dir is None or not output_dir.exists():
            return []
        for name in ("result.json", "summary.json", "overview.png"):
            path = output_dir / name
            if path.exists():
                paths.append(path)
        fields_dir = output_dir / "fields"
        if fields_dir.exists():
            for pattern in ("*.nii.gz", "*.nii", "*.nrrd", "*.nhdr", "*.mha", "*.mhd"):
                paths.extend(sorted(fields_dir.glob(pattern)))
        return paths

    @staticmethod
    def _timelapse_output_paths_for_row(root: Path, row: dict) -> list[str]:
        subject = str(row.get("subject") or "").strip()
        voi = str(row.get("voi_value", row.get("voi")) or "").strip().lower()
        if not subject or not voi:
            return []
        analysis_dir = Path(root) / "derivatives" / "Timelapse" / f"sub-{subject}" / "xct" / "analysis"
        if not analysis_dir.exists():
            return []
        paths: list[Path] = []
        table_path = analysis_dir / f"sub-{subject}_voi-{voi}_pairwise_remodelling.csv"
        if table_path.exists():
            paths.append(table_path)
        visualize_dir = analysis_dir / "visualize"
        for pattern in ("*.nii.gz", "*.nii", "*.nrrd", "*.nhdr", "*.mha", "*.mhd", "*.AIM", "*.aim"):
            for path in sorted(visualize_dir.glob(pattern)):
                if f"sub-{subject}_voi-{voi}_" in path.name and "_remodelling" in path.name:
                    paths.append(path)
        return [str(path) for path in paths]

    @staticmethod
    def _row_case_key(row: dict):
        subject = str(row.get("subject") or "").strip()
        session = str(row.get("session_value", row.get("session")) or "").strip()
        voi = str(row.get("voi_value", row.get("voi")) or "").strip().lower()
        if not subject or not session or not voi:
            return None
        stack = row.get("stack_index", row.get("stack"))
        try:
            stack_index = int(stack) if stack not in (None, "") else None
        except (TypeError, ValueError):
            stack_index = None
        return CaseKey(subject, session, voi, stack_index)

    @staticmethod
    def _output_artifacts_by_key(existing_outputs):
        grouped = {}
        for artifact in existing_outputs:
            grouped.setdefault(artifact.key, []).append(artifact)
        return grouped

    @staticmethod
    def _input_text(
        image,
        contours,
        registrations,
        common_regions=(),
        *,
        include_registrations: bool = False,
        include_image: bool = True,
        roles: tuple[str, ...] | None = None,
    ):
        aliases = (
            ("seg", "segmentation"),
            ("full", "full"),
            ("trab", "trab"),
            ("cort", "cort"),
        )
        allowed_roles = set(roles) if roles is not None else {role for _label, role in aliases}
        parts = [image.path.name] if include_image else []
        mask_parts = []
        for label, role in aliases:
            if role in allowed_roles and role in contours:
                mask_parts.append(f"{label}={contours[role].path.name}")
        if mask_parts:
            parts.extend(mask_parts)
        for common in sorted(common_regions, key=lambda record: str(record.path)):
            parts.append(f"common={common.path.name}")
        if include_registrations:
            for registration in sorted(registrations, key=lambda record: (record.role, str(record.path))):
                parts.append(f"registration={registration.path.name}")
        return "\n".join(parts)

    @staticmethod
    def _mask_label_algebra_core():
        local_src = _local_repo_path("bone-contouring", "src")
        if local_src.exists() and str(local_src) not in sys.path:
            sys.path.insert(0, str(local_src))
        from bone_contouring.algebra import discover_mask_label_algebra_batch

        return discover_mask_label_algebra_batch

    @staticmethod
    def _mask_label_algebra_table_rows(root: Path):
        discover_mask_label_algebra_batch = BatchProcessorLogic._mask_label_algebra_core()
        rows = []
        for algebra_row in discover_mask_label_algebra_batch(root):
            key = algebra_row.image.key
            status = {
                "ready": "Ready",
                "loadable": "Done",
                "missing": "Missing contour inputs",
            }.get(algebra_row.status, str(algebra_row.status).title())
            action = {
                "ready": "Run",
                "loadable": "Load",
                "missing": "Missing",
            }.get(algebra_row.status, "Run")
            inputs = [algebra_row.image.path.name]
            for role, artifact in sorted(algebra_row.contours.items()):
                label = "seg" if role == "segmentation" else role
                inputs.append(f"{label}={artifact.path.name}")
            if algebra_row.derivable_roles:
                inputs.append(f"derive={', '.join(algebra_row.derivable_roles)}")
            output_paths = [str(path) for path in BatchProcessorLogic._imported_contour_paths_for_row(
                root,
                {
                    "subject": key.subject_id,
                    "session_value": key.session_id,
                    "voi_value": key.voi,
                    "stack_index": key.stack_index,
                },
            )]
            row = {
                "action": action,
                "subject": key.subject_id,
                "session": key.session_id,
                "session_value": key.session_id,
                "voi": BatchProcessorLogic._voi_text(key),
                "voi_value": key.voi,
                "registered": False,
                "image_path": str(algebra_row.image.path),
                "input": "\n".join(inputs),
                "status": status,
            }
            if output_paths:
                row["output_paths"] = output_paths
            rows.append(row)
        return rows, f"Discovered {len(rows)} Mask And Label Algebra row(s)."

    def command_for_row(self, dataset_root, *, tool: str, profile: str, row: dict, force: bool = False) -> list[str]:
        """Return a core-package CLI command for one batch row."""
        tool_key = str(tool)
        if tool_key == "fea":
            return self._fea_command_for_row(self._dataset_root(dataset_root), profile, row, force=force)
        if tool_key == "mechanoregulation":
            return self._mechanoregulation_command_for_row(self._dataset_root(dataset_root), profile, row, force=force)
        if tool_key not in self._CLI_COMMANDS:
            raise ValueError(f"{tool_key} does not expose a one-row batch processor command yet.")
        module, command = self._CLI_COMMANDS[tool_key]
        args = ["-m", module, command, str(self._dataset_root(dataset_root))]
        if row.get("subject"):
            args.extend(["--subject", str(row["subject"])])
        session_value = row.get("session_value", row.get("session"))
        if int(row.get("action_row_span") or 0) > 1:
            session_value = ""
        if session_value:
            args.extend(["--session", str(session_value)])
        voi_value = row.get("voi_value", row.get("voi"))
        if voi_value:
            if tool in {"bone_contouring", "mask_label_algebra"}:
                args.extend(["--voi", str(voi_value)])
            else:
                args.extend(["--site", str(voi_value)])
        profile_value = str(profile or "").strip()
        if tool == "bone_contouring" and profile_value:
            args.extend(["--profile", profile_value])
        if tool == "timelapse" and profile_value:
            args.extend(["--profile", profile_value])
        if tool in {"microarchitecture", "plate_rod"}:
            if self.profile_requests_registration(tool, profile_value):
                args.append("--require-common-region")
            else:
                args.append("--no-common-region")
        if force:
            args.append("--force")
        return args

    def _table_rows_for_tool(self, image_records, contour_artifacts, registration_records, common_region_records, existing_outputs, *, tool: str, profile: str, registered: bool):
        image_records = list(image_records)
        required_roles = self._required_roles_for_tool(tool, profile)
        registrations_by_key = self._registration_records_by_key(registration_records)
        common_regions_by_key = self._common_region_records_by_key(common_region_records)
        outputs_by_key = self._output_artifacts_by_key(existing_outputs)
        status_outputs = [
            artifact for artifact in existing_outputs
            if not self._is_map_output_for_tool(tool, artifact)
        ]
        status_by_key = {}
        for image in image_records:
            contours = preferred_contours(contour_artifacts, image.key)
            status_by_key[image.key] = prerequisite_status(
                image,
                contours,
                required_roles=required_roles,
                existing_outputs=status_outputs,
            )
        if registered:
            grouped = {}
            for record in image_records:
                result = status_by_key.get(record.key)
                contours = preferred_contours(contour_artifacts, record.key)
                group_stack = None if self.profile_groups_all_stacks(tool, profile) else record.key.stack_index
                key = (record.key.subject_id, record.key.voi, group_stack)
                group = grouped.setdefault(
                    key,
                    {
                        "action": "Run",
                        "subject": record.key.subject_id,
                        "voi": self._voi_text(record.key),
                        "voi_value": record.key.voi,
                        "sessions": [],
                        "status": "Ready",
                        "missing": set(),
                        "review": set(),
                        "rows": [],
                    },
                )
                group["sessions"].append(str(record.key.session_id))
                if result is not None and result.status == "missing":
                    group["missing"].update(result.missing_roles)
                    group["action"] = "Missing"
                elif result is not None and result.status == "review":
                    group["review"].update(result.review_roles)
                    group["action"] = "Review"
                elif result is not None and result.status == "loadable" and group["action"] == "Run":
                    group["action"] = "Load"
                registrations = registrations_by_key.get(
                    self._case_lookup_key(record.key),
                    [],
                )
                common_regions = common_regions_by_key.get(
                    self._case_lookup_key(record.key),
                    [],
                )
                row = {
                    "subject": record.key.subject_id,
                    "session": record.key.session_id,
                    "session_value": record.key.session_id,
                    "voi": self._voi_text(record.key),
                    "voi_value": record.key.voi,
                    "registered": True,
                    "image_path": str(record.path),
                    "status": self._status_text(result),
                    "input": self._input_text(
                        record,
                        contours,
                        registrations,
                        common_regions,
                        include_registrations=True,
                        include_image=tool != "plate_rod",
                        roles=required_roles,
                    ),
                }
                if tool in {"microarchitecture", "plate_rod"} and not common_regions:
                    group["missing"].add("common region")
                    group["action"] = "Missing"
                    row["status"] = "Missing common region"
                row_outputs = outputs_by_key.get(record.key, [])
                has_measurements_output = any(
                    self._is_measurement_output_for_tool(tool, artifact)
                    for artifact in row_outputs
                )
                output_paths = [str(artifact.path) for artifact in row_outputs]
                if output_paths and (tool not in {"microarchitecture", "plate_rod"} or has_measurements_output):
                    row["output_paths"] = output_paths
                row["has_measurements_output"] = has_measurements_output
                group["rows"].append(row)
            rows = []
            for group_key, group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or 0)):
                group_rows = sorted(group["rows"], key=lambda row: (len(str(row["session"])), str(row["session"])))
                if tool in {"microarchitecture", "plate_rod"}:
                    group_output_paths = sorted(
                        {
                            path
                            for row in group_rows
                            for path in row.get("output_paths", [])
                        }
                    )
                    if group_output_paths and all(row.get("has_measurements_output") for row in group_rows):
                        group["action"] = "Load"
                        if group_rows:
                            group_rows[0]["output_paths"] = group_output_paths
                elif tool == "timelapse":
                    group_output_paths = sorted(
                        {
                            str(artifact.path)
                            for artifact in existing_outputs
                            if artifact.key.subject_id == group["subject"]
                            and artifact.key.voi == group["voi_value"]
                            and artifact.derivative == "Timelapse"
                        }
                    )
                    if group_output_paths:
                        group["action"] = "Load"
                        if group_rows:
                            group_rows[0]["output_paths"] = group_output_paths
                if len(group_rows) < 2:
                    group["action"] = "Missing"
                    for row in group_rows:
                        row["status"] = "Missing longitudinal series"
                span = len(group_rows)
                group_id = "|".join(str(value or "") for value in group_key)
                for index, row in enumerate(group_rows):
                    row["group_id"] = group_id
                    row["action"] = group["action"] if index == 0 else ""
                    row["action_row_span"] = span if index == 0 else 0
                    row.pop("has_measurements_output", None)
                    rows.append(row)
            return rows
        rows = []
        for record in sorted(image_records, key=lambda item: (item.key.subject_id, item.key.voi, str(item.key.session_id), item.key.stack_index or 0)):
            result = status_by_key.get(record.key)
            contours = preferred_contours(contour_artifacts, record.key)
            status = self._status_text(result)
            action = self._action_for_status(result)
            registrations = registrations_by_key.get(
                self._case_lookup_key(record.key),
                [],
            )
            row = {
                "action": action,
                "subject": record.key.subject_id,
                "session": record.key.session_id,
                "session_value": record.key.session_id,
                "voi": self._voi_text(record.key),
                "voi_value": record.key.voi,
                "registered": False,
                "image_path": str(record.path),
                "input": self._input_text(
                    record,
                    contours,
                    registrations,
                    include_registrations=False,
                    include_image=tool != "plate_rod",
                    roles=required_roles,
                ),
                "status": status,
            }
            row_outputs = outputs_by_key.get(record.key, [])
            has_measurements_output = any(self._is_measurement_output_for_tool(tool, artifact) for artifact in row_outputs)
            output_paths = [str(artifact.path) for artifact in row_outputs]
            if output_paths and (tool not in {"microarchitecture", "plate_rod"} or has_measurements_output):
                row["output_paths"] = output_paths
            rows.append(row)
        return rows

    @staticmethod
    def _dataset_root(dataset_root):
        root = Path(str(dataset_root or "")).expanduser()
        return root.parent if root.name == "derivatives" else root


class BatchProcessorWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = BatchProcessorLogic()
        self._batchRows = []
        self._batchQueue = []
        self._batchProcess = None
        self._batchProcessOutput = {}
        self._batchRunningRow = None
        self._batchCancelled = False
        self._remoteJobs = {}
        self._serverBackendEnabled = os.environ.get(SLICER_BONE_BATCH_BACKEND, "").strip().lower() in {
            "server",
            "ssh",
            "slurm",
            "arc",
        }
        self._remotePollTimer = qt.QTimer()
        self._remotePollTimer.setInterval(15000)
        self._remotePollTimer.timeout.connect(self._poll_remote_jobs)

        discovery_box = ctk.ctkCollapsibleButton()
        discovery_box.text = "Discovery"
        discovery_box.collapsed = False
        self.layout.addWidget(discovery_box)
        discovery_form = qt.QFormLayout(discovery_box)

        root_row = qt.QWidget()
        root_layout = qt.QHBoxLayout(root_row)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.datasetRootEdit = qt.QLineEdit()
        self.browseDatasetButton = qt.QPushButton("...")
        self.browseDatasetButton.toolTip = "Select the normalized dataset root."
        self.browseDatasetButton.clicked.connect(self._browse_dataset_root)
        root_layout.addWidget(self.datasetRootEdit)
        root_layout.addWidget(self.browseDatasetButton)
        self.datasetRootLabel = qt.QLabel("Dataset root")
        discovery_form.addRow(self.datasetRootLabel, root_row)

        self.serverRootEdit = qt.QLineEdit()
        self.serverRootEdit.placeholderText = "Remote normalized dataset root on the server"
        self.serverRootEdit.visible = False
        self.serverRootLabel = qt.QLabel("Server directory")
        self.serverRootLabel.visible = False
        discovery_form.addRow(self.serverRootLabel, self.serverRootEdit)

        self.analyzeButton = qt.QPushButton("Analyze")
        self.analyzeButton.clicked.connect(self._analyze_dataset)
        discovery_form.addRow("", self.analyzeButton)

        workflow_box = ctk.ctkCollapsibleButton()
        workflow_box.text = "Workflow"
        workflow_box.collapsed = False
        self.layout.addWidget(workflow_box)
        workflow_form = qt.QFormLayout(workflow_box)

        self.toolCombo = qt.QComboBox()
        for label, value in (
            ("Bone Contouring", "bone_contouring"),
            ("Mask And Label Algebra", "mask_label_algebra"),
            ("Microarchitecture", "microarchitecture"),
            ("Timelapsed Remodelling", "timelapse"),
            ("Plate/Rod Morphometry", "plate_rod"),
            ("ParOsol-FEA", "fea"),
            ("Mechanoregulation", "mechanoregulation"),
        ):
            self.toolCombo.addItem(label, value)
        self.toolCombo.currentIndexChanged.connect(self._on_tool_changed)
        workflow_form.addRow("Tool", self.toolCombo)

        self.profileCombo = qt.QComboBox()
        self.profileCombo.currentIndexChanged.connect(self._on_profile_changed)
        workflow_form.addRow("Profile", self.profileCombo)

        self.profileHintLabel = qt.QLabel()
        self.profileHintLabel.wordWrap = True
        self.profileHintLabel.visible = False
        workflow_form.addRow("", self.profileHintLabel)

        self.skipExistingCheck = qt.QCheckBox()
        self.skipExistingCheck.checked = True
        self.skipExistingCheck.toggled.connect(self._on_skip_existing_toggled)
        workflow_form.addRow("Skip existing", self.skipExistingCheck)

        self.backendCombo = qt.QComboBox()
        self.backendCombo.addItem("Local", "local")
        self.backendCombo.addItem("Server", "server")
        self.backendCombo.setItemData(1, "Server backends are configured in private adapters.", qt.Qt.ToolTipRole)
        self.backendCombo.currentIndexChanged.connect(self._on_backend_changed)
        if self._serverBackendEnabled:
            previous = self.backendCombo.blockSignals(True)
            try:
                self.backendCombo.setCurrentIndex(1)
            finally:
                self.backendCombo.blockSignals(previous)
        self.backendLabel = qt.QLabel("Execution backend")
        self.backendLabel.visible = self._serverBackendEnabled
        self.backendCombo.visible = self._serverBackendEnabled
        workflow_form.addRow(self.backendLabel, self.backendCombo)
        self._apply_backend_visibility()

        self.statusLabel = qt.QLabel("Select a normalized dataset root.")
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)

        self.table = qt.QTableWidget()
        self.table.minimumHeight = 180
        self.table.cellDoubleClicked.connect(self._on_table_cell_double_clicked)
        self.layout.addWidget(self.table)
        self.table.horizontalHeader().setStretchLastSection(True)
        self._populate_profile_combo()
        self._update_table_headers()

        self.runAllButton = qt.QPushButton("Run all")
        self.runAllButton.enabled = False
        self.runAllButton.clicked.connect(self._queue_all_rows)
        self.layout.addWidget(self.runAllButton)

        self.cancelBatchButton = qt.QPushButton("Cancel")
        self.cancelBatchButton.enabled = False
        self.cancelBatchButton.clicked.connect(self._cancel_batch)
        self.layout.addWidget(self.cancelBatchButton)

        self.batchLog = qt.QTextEdit()
        self.batchLog.readOnly = True
        self.batchLog.minimumHeight = 120
        self.layout.addWidget(self.batchLog)
        self.layout.addStretch(1)

    def _browse_dataset_root(self):
        path = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            "Select local processing directory" if self._selected_backend_key() == "server" else "Select normalized dataset root",
            self.datasetRootEdit.text,
        )
        if path:
            self.datasetRootEdit.text = str(path)

    def _apply_backend_visibility(self):
        server_selected = self._selected_backend_key() == "server" and self._serverBackendEnabled
        if hasattr(self, "datasetRootLabel"):
            self.datasetRootLabel.text = "Local processing directory" if server_selected else "Dataset root"
        if hasattr(self, "browseDatasetButton"):
            self.browseDatasetButton.toolTip = (
                "Select the local mirror used for server output load-back."
                if server_selected
                else "Select the normalized dataset root."
            )
        if hasattr(self, "serverRootEdit"):
            self.serverRootEdit.visible = server_selected
        if hasattr(self, "serverRootLabel"):
            self.serverRootLabel.visible = server_selected
        if server_selected:
            self._populate_remote_defaults()

    def _populate_remote_defaults(self):
        try:
            config = load_remote_batch_config()
        except Exception:
            return
        if hasattr(self, "serverRootEdit") and not str(self.serverRootEdit.text or "").strip():
            self.serverRootEdit.text = str(config.remote_root)
        if hasattr(self, "datasetRootEdit") and not str(self.datasetRootEdit.text or "").strip() and config.local_root:
            self.datasetRootEdit.text = str(Path(config.local_root).expanduser())

    def _update_table_headers(self, *args):
        del args
        headers = ["Action", "Subject", "Session", "VOI", "Status", "Input"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

    def _profiles_for_tool(self, tool):
        profiles = list(self.logic.profiles_for_tool(tool))
        if str(tool or "") == "timelapse":
            try:
                from timelapsedhrpqct.config.profiles import list_config_profiles

                discovered = []
                for value in list_config_profiles():
                    discovered.append((self._timelapse_profile_label(value), value, True))
                if discovered:
                    profiles = sorted(discovered, key=lambda item: item[0].lower())
            except Exception:
                pass
        if str(tool or "") == "bone_contouring":
            try:
                existing = {str(value) for _label, value, _registered in profiles}
                for record in list_profiles("bone-contouring"):
                    if record.name not in existing:
                        profiles.append((record.name, record.name, False))
            except Exception:
                pass
        return tuple(profiles)

    @staticmethod
    def _timelapse_profile_label(value):
        labels = {
            "standard": "Standard",
            "eth-uofc": "ETH-UofC",
            "shriners": "Shriners",
            "ucsf": "UCSF",
            "xct1-standard": "XtremeCT I Standard",
            "ped-fx": "Pediatric Fracture",
            "multistack": "Multistack",
        }
        key = str(value or "").strip()
        if key in labels:
            return labels[key]
        return re.sub(r"[-_]+", " ", key).strip().title() or "Profile"

    def _populate_profile_combo(self):
        if not hasattr(self, "profileCombo"):
            return
        tool = self._selected_tool_key()
        previous = self.profileCombo.blockSignals(True)
        try:
            self.profileCombo.clear()
            for label, value, _registered in self._profiles_for_tool(tool):
                self.profileCombo.addItem(label, value)
        finally:
            self.profileCombo.blockSignals(previous)
        self._update_profile_hint()

    def _is_selected_profile_shipped(self):
        tool = self._selected_tool_key()
        profile = str(self.profileCombo.currentData or "")
        return any(str(value) == profile for _label, value, _registered in self.logic.profiles_for_tool(tool))

    def _update_profile_hint(self):
        if not hasattr(self, "profileHintLabel"):
            return
        hint = (
            "For Bone Contouring, radius/tibia/knee settings are selected automatically from each row's VOI."
        )
        show_hint = self._selected_tool_key() == "bone_contouring" and self._is_selected_profile_shipped()
        self.profileHintLabel.text = hint
        self.profileCombo.toolTip = hint if show_hint else ""
        self.profileHintLabel.visible = show_hint

    def _on_tool_changed(self, *args):
        del args
        self._populate_profile_combo()
        self._update_profile_hint()
        if self._has_active_batch():
            self._append_log("[batch] Tool/profile change will apply after the active queue finishes.")
            return
        self._update_table_headers()
        if str(self.datasetRootEdit.text or "").strip():
            self._analyze_dataset()

    def _on_profile_changed(self, *args):
        del args
        self._update_profile_hint()
        if self._has_active_batch():
            self._append_log("[batch] Tool/profile change will apply after the active queue finishes.")
            return
        self._update_table_headers()
        if str(self.datasetRootEdit.text or "").strip():
            self._analyze_dataset()

    def _on_skip_existing_toggled(self, *args):
        del args
        if str(self.datasetRootEdit.text or "").strip():
            self._analyze_dataset()

    def _on_backend_changed(self, *args):
        del args
        self._apply_backend_visibility()
        if self._has_active_batch():
            self._append_log("[batch] Backend change will apply after the active queue finishes.")
            return
        server_text = str(self.serverRootEdit.text or "").strip() if hasattr(self, "serverRootEdit") else ""
        if str(self.datasetRootEdit.text or "").strip() or server_text:
            self._analyze_dataset()

    def _selected_tool_key(self):
        return str(getattr(self.toolCombo, "currentData", "") or "")

    def _selected_backend_key(self):
        if not getattr(self, "_serverBackendEnabled", False):
            return "local"
        return str(getattr(self.backendCombo, "currentData", "") or "local")

    def _current_local_dataset_root(self):
        return str(self.datasetRootEdit.text or "").strip()

    def _current_remote_dataset_root(self):
        if not hasattr(self, "serverRootEdit"):
            return ""
        return str(self.serverRootEdit.text or "").strip()

    def _remote_backend(self, *, local_root=None, remote_root=None):
        try:
            config = load_remote_batch_config()
            local_text = str(local_root or "").strip()
            remote_text = str(remote_root or "").strip()
            if local_text or remote_text:
                config = replace(
                    config,
                    local_root=local_text or config.local_root,
                    remote_root=(remote_text or config.remote_root).rstrip("/"),
                )
            return SshSlurmBatchBackend(config)
        except Exception as exc:
            self._append_log(f"[batch] remote backend unavailable: {exc}")
            return None

    def _registered_table_mode(self):
        return self.logic.profile_requests_registration(
            self._selected_tool_key(),
            str(self.profileCombo.currentData),
        )

    def _analyze_dataset(self):
        if not hasattr(self, "statusLabel") or not hasattr(self, "table"):
            return
        root = self._current_local_dataset_root()
        if self._selected_backend_key() == "server":
            rows, message = self._discover_remote_rows(
                root,
                self._current_remote_dataset_root(),
                self._selected_tool_key(),
                str(self.profileCombo.currentData),
                self._registered_table_mode(),
            )
            self.statusLabel.text = message
            self._populate_rows(rows)
            return
        if self._selected_tool_key() == "mechanoregulation":
            ok, message = Path(root).expanduser().exists(), "Select an existing dataset root."
        else:
            ok, message = self.logic.normalized_dataset_status(root)
        if not ok:
            self.statusLabel.text = f"{message} Open Dataset Naming Helper to normalize loose filenames."
            self.table.setRowCount(0)
            return
        rows, message = self.logic.discover_rows(
            root,
            tool=self._selected_tool_key(),
            profile=str(self.profileCombo.currentData),
            registered=self._registered_table_mode(),
        )
        self.statusLabel.text = message
        self._populate_rows(rows)

    def _discover_remote_rows(self, local_root, remote_root, tool, profile, registered):
        backend = self._remote_backend(local_root=local_root, remote_root=remote_root)
        if backend is None:
            return [], f"Set {SLICER_BONE_BATCH_REMOTE_CONFIG} to a private ARC/SLURM backend config."
        if not str(backend.config.remote_root or "").strip():
            return [], "Enter a server directory."
        try:
            result = subprocess.run(
                backend.discover_argv(
                    families=(
                        "IPLContours",
                        "ImportedContours",
                        "BoneContours",
                        "ImportedRegistration",
                        "Registration",
                        "CommonRegion",
                        self.logic._output_family_for_tool(tool),
                    )
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            payload = json.loads(result.stdout)
        except Exception as exc:
            return [], f"Remote discovery failed: {exc}"
        if not bool(payload.get("normalized", {}).get("ok")):
            return [], str(payload.get("normalized", {}).get("message") or "Remote dataset is not normalized.")
        image_records = [self._artifact_from_remote_payload(item) for item in payload.get("raw_images", [])]
        derivative_records = [self._artifact_from_remote_payload(item) for item in payload.get("derivatives", [])]
        contour_artifacts = [
            artifact for artifact in derivative_records
            if artifact.derivative in {"IPLContours", "ImportedContours", "BoneContours"}
        ]
        registration_records = [
            SimpleNamespace(
                role=artifact.role,
                subject_id=artifact.key.subject_id,
                session_id=artifact.key.session_id,
                site=artifact.key.voi,
                stack_index=artifact.key.stack_index,
                path=artifact.path,
                derivative=artifact.derivative,
            )
            for artifact in derivative_records
            if artifact.derivative in {"ImportedRegistration", "Registration"}
        ]
        common_region_records = [
            SimpleNamespace(
                role=artifact.role,
                subject_id=artifact.key.subject_id,
                session_id=artifact.key.session_id,
                site=artifact.key.voi,
                stack_index=artifact.key.stack_index,
                path=artifact.path,
                derivative=artifact.derivative,
            )
            for artifact in derivative_records
            if artifact.derivative == "CommonRegion"
        ]
        existing_outputs = [
            artifact for artifact in derivative_records
            if artifact.derivative == self.logic._output_family_for_tool(tool)
        ]
        rows = self.logic._table_rows_for_tool(
            image_records,
            contour_artifacts,
            registration_records,
            common_region_records,
            self.logic._existing_outputs_for_profile(tool, registered, existing_outputs),
            tool=tool,
            profile=profile,
            registered=registered,
        )
        for row in rows:
            row["remote_backend"] = backend.config.name
        return rows, f"Remote {backend.config.name}: discovered {len(rows)} row(s) in {backend.config.remote_root}."

    @staticmethod
    def _artifact_from_remote_payload(item):
        key = item.get("key", {})
        return BatchArtifact(
            Path(str(item.get("path") or "")),
            CaseKey(
                str(key.get("subject_id") or ""),
                str(key.get("session_id") or ""),
                str(key.get("voi") or ""),
                key.get("stack_index"),
            ),
            str(item.get("role") or ""),
            str(item.get("family") or "") or None,
            str(item.get("source") or "remote"),
            item.get("metadata") or {},
        )

    def _populate_rows(self, rows):
        self._batchRows = list(rows)
        self._apply_remote_job_overrides()
        self.table.clearSpans()
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            action = self._effective_row_action(row)
            span = int(row.get("action_row_span") or 0)
            if span > 1:
                self.table.setSpan(row_index, 0, span, 1)
            if action:
                button = qt.QPushButton(action)
                button.enabled = action in {"Run", "Load"}
                button.clicked.connect(lambda _checked=False, index=row_index: self._on_row_action(index))
                self.table.setCellWidget(row_index, 0, button)
            else:
                item = qt.QTableWidgetItem("")
                item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                self.table.setItem(row_index, 0, item)
            values = [row["subject"], row["session"], row["voi"], row["status"], row["input"]]
            for column, value in enumerate(values, start=1):
                item = qt.QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                item.setToolTip(str(value))
                self.table.setItem(row_index, column, item)
        self.runAllButton.enabled = bool(rows)
        try:
            self.table.resizeColumnsToContents()
        except Exception:
            pass

    def _on_table_cell_double_clicked(self, row, column):
        headers = ["Action", "Subject", "Session", "VOI", "Status", "Input"]
        if int(column) != headers.index("Input"):
            return
        item = self.table.item(int(row), int(column))
        if item is None:
            return
        self.table.resizeRowToContents(row)
        self.table.resizeColumnToContents(column)

    def _append_log(self, text):
        if not text:
            return
        if hasattr(self, "batchLog"):
            self.batchLog.append(str(text).rstrip())

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

    def _effective_row_action(self, row):
        action = str(row.get("action") or "")
        if action == "Load" and not bool(self.skipExistingCheck.checked):
            return "Run"
        return action

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

    def _process_environment(self):
        environment = qt.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        for key in ("ITK_AUTOLOAD_PATH", "SITK_AUTOLOAD_PATH"):
            if environment.contains(key):
                environment.remove(key)
            environment.insert(key, "")
        python_paths = [str(TOOLBOX_ROOT), str(DERIVATIVES_LOCAL_SRC)]
        for repo_name in (
            "bone-contouring/src",
            "bone-microarchitecture/src",
            "bone-plate-rod-thinning",
            "Timelapsed" + "HRpQCT/src",
            "parosol-py/src",
            "BoneMechanoregulation",
        ):
            candidate = _local_repo_path(*Path(repo_name).parts)
            if candidate.exists():
                python_paths.append(str(candidate))
        existing = str(environment.value("PYTHONPATH") or "")
        if existing:
            python_paths.append(existing)
        environment.insert("PYTHONPATH", os.pathsep.join(path for path in python_paths if path))
        if self._selected_tool_key() == "plate_rod" and sys.platform == "darwin":
            environment.insert("PLATE_ROD_USE_METAL", "1")
            environment.insert("PLATE_ROD_USE_METAL_FULL", "1")
        return environment

    def _subprocess_args(self, args):
        args = list(args)
        if len(args) < 2 or args[0] != "-m":
            return args
        module = str(args[1])
        local_sources = {
            "bone_contouring.cli": _local_repo_path("bone-contouring", "src"),
            "bone_microarchitecture.cli": _local_repo_path("bone-microarchitecture", "src"),
            "timelapsedhrpqct.cli": _local_repo_path("Timelapsed" + "HRpQCT", "src"),
            "plate_rod_thinning.cli": _local_repo_path("bone-plate-rod-thinning"),
            "parosol_py.cli": _local_repo_path("parosol-py", "src"),
            "bonemechreg.cli": _local_repo_path("BoneMechanoregulation"),
        }
        local_src = local_sources.get(module)
        if local_src is None or not local_src.exists():
            return args
        bootstrap = (
            "import sys; "
            f"sys.path.insert(0, {str(local_src)!r}); "
            f"from {module} import main; "
            "raise SystemExit(main())"
        )
        return ["-c", bootstrap] + args[2:]

    def _on_row_action(self, row_index):
        if row_index < 0 or row_index >= len(self._batchRows):
            self._append_log("[batch] Selected row is no longer available. Click Analyze again.")
            return
        row = self._batchRows[row_index]
        action = self._effective_row_action(row)
        if action == "Run":
            self._queue_row(row_index)
            return
        if action == "Load":
            self._load_row_outputs(row_index)
            return
        self._append_log(f"[batch] Row is not runnable: {row.get('status', action)}")

    def _queue_row(self, row_index):
        if row_index in self._queued_row_indices():
            return
        self._batchQueue.append(self._batch_job_for_row(row_index))
        self._set_row_action(row_index, "Queued")
        self._set_row_status(row_index, "Queued")
        self.cancelBatchButton.enabled = True
        self._append_log(f"[batch] queued row {int(row_index) + 1}")
        self._start_next_batch_job()

    def _queue_all_rows(self):
        queued = 0
        for row_index, row in enumerate(self._batchRows):
            if self._effective_row_action(row) != "Run":
                continue
            if row_index in self._queued_row_indices():
                continue
            self._batchQueue.append(self._batch_job_for_row(row_index))
            self._set_row_action(row_index, "Queued")
            self._set_row_status(row_index, "Queued")
            queued += 1
        self.cancelBatchButton.enabled = bool(self._batchQueue or self._batchProcess is not None)
        self._append_log(f"[batch] queued {queued} row(s)")
        self._start_next_batch_job()

    def _has_active_batch(self):
        return bool(self._batchProcess is not None or self._batchQueue)

    def _queued_row_indices(self):
        indices = []
        for job in self._batchQueue:
            if isinstance(job, dict):
                indices.append(int(job.get("row_index", -1)))
            else:
                indices.append(int(job))
        return set(indices)

    def _batch_job_for_row(self, row_index):
        row_index = int(row_index)
        return {
            "row_index": row_index,
            "tool": self._selected_tool_key(),
            "profile": str(self.profileCombo.currentData or ""),
            "force": not bool(self.skipExistingCheck.checked),
            "backend": self._selected_backend_key(),
            "local_root": self._current_local_dataset_root(),
            "remote_root": self._current_remote_dataset_root(),
            "row": dict(self._batchRows[row_index]),
        }

    def _set_row_status(self, row_index, status):
        if 0 <= int(row_index) < len(self._batchRows):
            self._batchRows[int(row_index)]["status"] = str(status)
        status_column = 4
        item = self.table.item(int(row_index), status_column)
        if item is not None:
            item.setText(str(status))
            item.setToolTip(str(status))

    def _set_row_action(self, row_index, action):
        row_index = int(row_index)
        if 0 <= row_index < len(self._batchRows):
            self._batchRows[row_index]["action"] = str(action)
        button = qt.QPushButton(str(action))
        button.enabled = str(action) in {"Run", "Load"}
        button.clicked.connect(lambda _checked=False, index=row_index: self._on_row_action(index))
        self.table.setCellWidget(row_index, 0, button)

    def _start_next_batch_job(self):
        if self._batchProcess is not None:
            return
        if not self._batchQueue:
            self._batchRunningRow = None
            self.runAllButton.enabled = bool(self._batchRows)
            self.cancelBatchButton.enabled = False
            if str(self.datasetRootEdit.text or "").strip() and not self._remoteJobs:
                self._analyze_dataset()
            return
        job = self._batchQueue.pop(0)
        if isinstance(job, dict):
            row_index = int(job.get("row_index", -1))
        else:
            row_index = int(job)
            job = self._batch_job_for_row(row_index)
        if row_index < 0 or row_index >= len(self._batchRows):
            self._start_next_batch_job()
            return
        try:
            dataset_root = str(job.get("local_root") or self.datasetRootEdit.text)
            args = self.logic.command_for_row(
                dataset_root,
                tool=str(job.get("tool") or ""),
                profile=str(job.get("profile") or ""),
                row=dict(job.get("row") or {}),
                force=bool(job.get("force")),
            )
        except Exception as exc:
            self._set_row_status(row_index, f"Error: {exc}")
            self._append_log(f"[batch] {exc}")
            self._start_next_batch_job()
            return
        self.runAllButton.enabled = False
        self.cancelBatchButton.enabled = True
        self._batchRunningRow = row_index
        self._batchCancelled = False
        self._set_row_status(row_index, "Running")
        self._set_row_action(row_index, "Running")
        backend_key = str(job.get("backend") or "local")
        remote_backend = self._remote_backend(
            local_root=job.get("local_root"),
            remote_root=job.get("remote_root"),
        ) if backend_key == "server" else None
        if backend_key == "server" and remote_backend is not None:
            remote_args = remote_backend.config.remote_args(args, dataset_root=job.get("local_root"))
            job_name = f"bone-{job.get('tool')}-{row_index + 1}"
            submit_argv = remote_backend.submit_argv(
                remote_args,
                job_name=job_name,
            )
            job["remote_job_name"] = job_name
            process_program = submit_argv[0]
            process_args = submit_argv[1:]
            self._append_log(f"[batch] submitting remote {remote_backend.config.name}: {' '.join(submit_argv)}")
        else:
            process_program = self._python_slicer_executable()
            process_args = self._subprocess_args(args)
            self._append_log(f"[batch] launching: {process_program} {' '.join(process_args)}")
        process = qt.QProcess()
        process.setProcessChannelMode(qt.QProcess.MergedChannels)
        if backend_key != "server":
            process.setProcessEnvironment(self._process_environment())
        process.readyRead.connect(lambda process=process: self._append_process_output(process))
        process.finished.connect(
            lambda *signal_args, row_index=row_index, process=process, job=job: self._batch_process_finished(
                row_index,
                process,
                job,
                *signal_args,
            )
        )
        self._batchProcess = process
        self._batchProcessOutput[id(process)] = ""
        process.start(process_program, process_args)
        if not process.waitForStarted(1000):
            self._batchProcess = None
            self._set_row_status(row_index, "Error")
            self._append_log("[batch] could not start process")
            self._start_next_batch_job()

    def _cancel_batch(self):
        self._batchCancelled = True
        queued = list(self._batchQueue)
        self._batchQueue = []
        for job in queued:
            row_index = int(job.get("row_index", -1)) if isinstance(job, dict) else int(job)
            self._set_row_status(row_index, "Cancelled")
            self._set_row_action(row_index, "Run")
        process = self._batchProcess
        for remote_job in list(self._remoteJobs.values()):
            if not isinstance(remote_job, dict) or not remote_job.get("job_id"):
                continue
            backend = self._remote_backend(
                local_root=remote_job.get("local_root"),
                remote_root=remote_job.get("remote_root"),
            )
            if backend is None:
                continue
            try:
                subprocess.run(backend.cancel_argv(str(remote_job["job_id"])), capture_output=True, text=True, timeout=30)
            except Exception as exc:
                self._append_log(f"[batch] remote cancel failed: {exc}")
        self._remoteJobs = {}
        self._stop_remote_polling_if_idle()
        if process is not None:
            self._append_log("[batch] cancelling running job")
            process.terminate()
            try:
                if not process.waitForFinished(3000):
                    process.kill()
            except Exception:
                pass
        else:
            self.cancelBatchButton.enabled = False
            self.runAllButton.enabled = bool(self._batchRows)

    def _append_process_output(self, process):
        text = self._clean_process_output(self._qbytearray_to_text(process.readAll()))
        if text:
            key = id(process)
            self._batchProcessOutput[key] = (self._batchProcessOutput.get(key, "") + "\n" + text).strip()
            self._append_log(text)

    @staticmethod
    def _clean_process_output(text: str) -> str:
        lines = []
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            if stripped.startswith("itk version "):
                continue
            if any(marker in stripped for marker in _SUPPRESSED_PROCESS_OUTPUT_MARKERS):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _batch_process_finished(self, row_index, process, job=None, *signal_args):
        self._append_process_output(process)
        exit_code = int(signal_args[0]) if signal_args else int(process.exitCode())
        process_output = self._batchProcessOutput.pop(id(process), "")
        self._batchProcess = None
        job = dict(job or {})
        tool_key = str(job.get("tool") or self._selected_tool_key())
        profile = str(job.get("profile") or self.profileCombo.currentData or "")
        if self._batchCancelled:
            self._set_row_status(row_index, "Cancelled")
            self._set_row_action(row_index, "Run")
            self._append_log("[batch] cancelled")
        elif exit_code == 0:
            if str(job.get("backend") or "local") == "server":
                self._mark_remote_job_submitted(row_index, job, process_output)
                self._start_next_batch_job()
                return
            if tool_key == "fea" and 0 <= row_index < len(self._batchRows):
                published = self.logic.publish_fea_batch_outputs(
                    str(job.get("local_root") or self.datasetRootEdit.text),
                    dict(job.get("row") or self._batchRows[row_index]),
                    profile,
                )
                if published:
                    self._append_log(f"[batch] published {len(published)} FEA artifact(s)")
            self._refresh_row_output_paths(row_index, tool_key=tool_key)
            self._set_row_status(row_index, "Done")
            self._set_row_action(row_index, "Load")
            self._append_log("[batch] finished")
        else:
            self._set_row_status(row_index, f"Error {exit_code}")
            self._set_row_action(row_index, "Run")
            self._append_log(f"[batch] failed with exit code {exit_code}")
        self._start_next_batch_job()

    def _mark_remote_job_submitted(self, row_index, job, process_output):
        backend = self._remote_backend(local_root=job.get("local_root"), remote_root=job.get("remote_root"))
        job_id = backend.parse_job_id(process_output) if backend is not None else None
        if not job_id:
            self._set_row_status(row_index, "Error: no SLURM job id")
            self._set_row_action(row_index, "Run")
            self._append_log("[batch] remote submit finished but no SLURM job id was returned")
            return
        row_key = self._row_remote_key(row_index, job)
        self._remoteJobs[row_key] = {
            "job_id": str(job_id),
            "row_index": int(row_index),
            "tool": str(job.get("tool") or ""),
            "profile": str(job.get("profile") or ""),
            "local_root": str(job.get("local_root") or ""),
            "remote_root": str(job.get("remote_root") or ""),
            "remote_job_name": str(job.get("remote_job_name") or ""),
            "state": "SUBMITTED",
        }
        self._set_row_status(row_index, f"Submitted {job_id}")
        self._set_row_action(row_index, "Submitted")
        self._append_log(f"[batch] submitted remote SLURM job {job_id}")
        self._start_remote_polling()

    def _start_remote_polling(self):
        if self._remoteJobs and not self._remotePollTimer.isActive():
            self._remotePollTimer.start()

    def _stop_remote_polling_if_idle(self):
        if not self._remoteJobs and self._remotePollTimer.isActive():
            self._remotePollTimer.stop()

    def _poll_remote_jobs(self):
        if not self._remoteJobs:
            self._stop_remote_polling_if_idle()
            return
        for row_key, remote_job in list(self._remoteJobs.items()):
            backend = self._remote_backend(
                local_root=remote_job.get("local_root"),
                remote_root=remote_job.get("remote_root"),
            )
            if backend is None:
                continue
            job_id = str(remote_job.get("job_id") or "")
            row_index = int(remote_job.get("row_index", -1))
            try:
                result = subprocess.run(backend.status_argv(job_id), capture_output=True, text=True, timeout=30)
                state_text = (result.stdout or result.stderr or "").strip()
            except Exception as exc:
                self._append_log(f"[batch] could not poll remote job {job_id}: {exc}")
                continue
            status, terminal = self._remote_job_terminal_state(state_text)
            remote_job["state"] = status
            if row_index >= 0 and row_index < len(self._batchRows):
                self._set_row_status(row_index, f"{status} {job_id}".strip())
            if not terminal:
                continue
            self._remoteJobs.pop(row_key, None)
            if status == "Done":
                if row_index >= 0 and row_index < len(self._batchRows):
                    self._set_row_status(row_index, "Done")
                    self._set_row_action(row_index, "Load")
                self._append_log(f"[batch] remote job {job_id} completed")
            else:
                if row_index >= 0 and row_index < len(self._batchRows):
                    self._set_row_action(row_index, "Run")
                self._append_log(f"[batch] remote job {job_id} ended: {state_text or status}")
                log_name = str(remote_job.get("remote_job_name") or "")
                if log_name:
                    try:
                        log_result = subprocess.run(backend.log_argv(log_name), capture_output=True, text=True, timeout=30)
                        log_text = self._clean_process_output((log_result.stdout or log_result.stderr or "").strip())
                        if log_text:
                            self._append_log(log_text)
                    except Exception:
                        pass
        self._stop_remote_polling_if_idle()

    @staticmethod
    def _remote_job_terminal_state(state_text):
        text = str(state_text or "").strip().upper()
        if not text:
            return "Submitted", False
        first = text.splitlines()[0].split("|")[0].strip()
        if first in {"PENDING", "CONFIGURING"}:
            return "Queued", False
        if first in {"RUNNING", "COMPLETING"}:
            return "Running", False
        if first.startswith("COMPLETED"):
            return "Done", True
        if "OUT_OF_MEMORY" in first or "OOM" in first:
            return "OOM", True
        if first in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"}:
            return first.title(), True
        return first.title(), True

    def _row_remote_key(self, row_index, job=None):
        row = dict((job or {}).get("row") or (self._batchRows[int(row_index)] if 0 <= int(row_index) < len(self._batchRows) else {}))
        return "|".join(
            str(value or "")
            for value in (
                (job or {}).get("tool") or self._selected_tool_key(),
                (job or {}).get("profile") or self.profileCombo.currentData,
                row.get("subject"),
                row.get("session"),
                row.get("voi_value", row.get("voi")),
                row.get("stack_index"),
            )
        )

    def _apply_remote_job_overrides(self):
        for row_index, row in enumerate(self._batchRows):
            key = self._row_remote_key(row_index, {"row": row, "tool": self._selected_tool_key(), "profile": self.profileCombo.currentData})
            remote_job = self._remoteJobs.get(key)
            if not remote_job:
                continue
            state = str(remote_job.get("state") or "SUBMITTED").title()
            job_id = str(remote_job.get("job_id") or "")
            row["status"] = f"{state} {job_id}".strip()
            row["action"] = "Submitted"

    def _sync_remote_outputs_for_tool(self, tool_key, *, local_root=None, remote_root=None):
        backend = self._remote_backend(local_root=local_root, remote_root=remote_root)
        if backend is None:
            return
        family = self.logic._output_family_for_tool(tool_key)
        try:
            argv = backend.sync_output_argv(family)
            self._append_log(f"[batch] syncing remote outputs: {' '.join(argv)}")
            result = subprocess.run(argv, capture_output=True, text=True, timeout=300)
            if result.stdout.strip():
                self._append_log(result.stdout)
            if result.stderr.strip():
                self._append_log(result.stderr)
            if result.returncode != 0:
                self._append_log(f"[batch] remote output sync failed with exit code {result.returncode}")
        except Exception as exc:
            self._append_log(f"[batch] remote output sync failed: {exc}")

    def _refresh_row_output_paths(self, row_index, *, tool_key=None):
        if row_index < 0 or row_index >= len(self._batchRows):
            return []
        tool_key = str(tool_key or self._selected_tool_key())
        row = self._batchRows[row_index]
        dataset_root = self._current_local_dataset_root()
        paths = []
        if (
            tool_key in {"microarchitecture", "plate_rod"}
            and bool(row.get("registered"))
            and int(row.get("action_row_span") or 0) > 1
        ):
            for offset in range(int(row.get("action_row_span") or 0)):
                child_index = row_index + offset
                if child_index >= len(self._batchRows):
                    break
                paths.extend(
                    self.logic.rediscover_row_output_paths(
                        dataset_root,
                        tool_key,
                        self._batchRows[child_index],
                    )
                )
        else:
            paths = self.logic.rediscover_row_output_paths(
                dataset_root,
                tool_key,
                row,
            )
        paths = [str(path) for path in self._deduplicated_paths(Path(path) for path in paths)]
        if paths:
            row["output_paths"] = paths
        return paths

    def _load_row_outputs(self, row_index):
        if row_index < 0 or row_index >= len(self._batchRows):
            self._append_log("[batch] Selected row is no longer available. Click Analyze again.")
            return
        if self._selected_backend_key() == "server":
            self._sync_remote_outputs_for_tool(
                self._selected_tool_key(),
                local_root=self._current_local_dataset_root(),
                remote_root=self._current_remote_dataset_root(),
            )
        row = self._batchRows[row_index]
        output_paths = [Path(path) for path in row.get("output_paths", []) if str(path)]
        if not output_paths:
            output_paths = [Path(path) for path in self._refresh_row_output_paths(row_index)]
        output_paths = self._deduplicated_paths(output_paths)
        if not output_paths:
            self._append_log(
                "[batch] No output paths were discovered for this row yet. "
                f"Click Analyze again or load from the detailed {self.toolCombo.currentText} module."
            )
            return
        if self._selected_tool_key() in {"bone_contouring", "mask_label_algebra"}:
            self._load_bone_contour_outputs_as_segmentation(row, output_paths)
            return
        if self._selected_tool_key() == "timelapse":
            self._load_timelapse_outputs(row, output_paths)
            return
        if self._selected_tool_key() == "fea":
            self._load_fea_outputs(row, output_paths)
            return
        if self._selected_tool_key() == "mechanoregulation":
            self._load_mechanoregulation_outputs(row, output_paths)
            return
        loaded = 0
        for path in output_paths:
            if not path.exists():
                continue
            try:
                is_table = path.suffix.lower() == ".csv"
                if path.suffix.lower() == ".csv":
                    self._remove_existing_node_named(path.stem)
                    result = slicer.util.loadTable(str(path))
                elif self._selected_tool_key() == "plate_rod" and path.suffix.lower() == ".npy":
                    self._remove_existing_node_named(path.stem)
                    result = self._load_plate_rod_npy_map(path)
                elif path.name.lower().endswith((".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")):
                    self._remove_existing_node_named(path.stem)
                    result = slicer.util.loadVolume(str(path), {"name": path.stem})
                else:
                    self._append_log(f"[batch] Skipping unsupported output: {path.name}")
                    continue
                if isinstance(result, tuple):
                    success = bool(result[0])
                    node = result[1] if len(result) > 1 else None
                else:
                    success = bool(result)
                    node = result
                if success:
                    if is_table and node is not None:
                        self._show_table_node(node)
                        if self._selected_tool_key() == "plate_rod":
                            self._put_node_in_subject_hierarchy_folder(node, self._plate_rod_output_folder_name(path, row))
                    elif self._selected_tool_key() == "microarchitecture" and node is not None:
                        self._style_microarchitecture_volume(node, path)
                        folder_name = self._microarchitecture_map_folder_name(path, row)
                        self._put_node_in_subject_hierarchy_folder(node, folder_name)
                    elif self._selected_tool_key() == "plate_rod" and node is not None:
                        self._style_plate_rod_volume(node, path)
                        self._put_node_in_subject_hierarchy_folder(node, self._plate_rod_output_folder_name(path, row))
                    loaded += 1
            except Exception as exc:
                self._append_log(f"[batch] Could not load {path.name}: {exc}")
        if self._selected_tool_key() in {"microarchitecture", "plate_rod"} and bool(row.get("registered")):
            self._load_registered_common_region_overlays(row_index)
        self._append_log(f"[batch] loaded {loaded} output file(s)")

    def _load_fea_outputs(self, row, output_paths):
        loaded = 0
        folder_name = self._fea_output_folder_name(row)
        loadable_suffixes = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")
        for path in output_paths:
            if not path.exists():
                continue
            try:
                self._remove_existing_node_named(path.stem)
                is_table = path.suffix.lower() == ".csv"
                if is_table:
                    result = self._table_node_from_csv(path, path.stem)
                    self._show_table_node(result)
                elif path.name.lower().endswith(loadable_suffixes):
                    result = slicer.util.loadVolume(str(path), {"name": path.stem})
                else:
                    continue
                if isinstance(result, tuple):
                    success = bool(result[0])
                    node = result[1] if len(result) > 1 else None
                else:
                    success = bool(result)
                    node = result
                if success:
                    if node is not None:
                        if not is_table:
                            self._style_fea_volume(node, path)
                        self._put_node_in_subject_hierarchy_folder(node, folder_name)
                    loaded += 1
            except Exception as exc:
                self._append_log(f"[batch] Could not load {path.name}: {exc}")
        self._append_log(f"[batch] loaded ParOsol-FEA output(s): {loaded}")

    def _load_mechanoregulation_outputs(self, row, output_paths):
        loaded = 0
        folder_name = self._mechanoregulation_folder_name(row)
        loaded += self._load_mechanoregulation_context_volumes(row, folder_name)
        summary_paths = [
            Path(path)
            for path in output_paths
            if Path(path).exists() and Path(path).suffix.lower() == ".csv"
        ]
        if summary_paths:
            try:
                compact_path = self._write_mechanoregulation_summary_table_csv(summary_paths)
                table_name = f"{Path(summary_paths[0]).stem}_summary"
                self._remove_existing_node_named(table_name)
                node = self._table_node_from_csv(compact_path, table_name)
                self._show_table_node(node)
                self._put_node_in_subject_hierarchy_folder(node, folder_name)
                loaded += 1
            except Exception as exc:
                self._append_log(f"[batch] Could not load compact mechanoregulation table: {exc}")
        for path in output_paths:
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".csv":
                    continue
                elif path.name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self._append_log(f"[batch] Mechanoregulation curve: {path}")
                    loaded += 1
            except Exception as exc:
                self._append_log(f"[batch] Could not load {path.name}: {exc}")
        self._append_log(f"[batch] loaded mechanoregulation output(s): {loaded}")

    def _write_mechanoregulation_summary_table_csv(self, summary_paths):
        output_dir = Path(tempfile.gettempdir()) / "SlicerBoneImagingToolbox" / "BatchProcessor"
        output_dir.mkdir(parents=True, exist_ok=True)
        if not summary_paths:
            raise ValueError("No mechanoregulation summary CSV paths were provided")
        first_stem = Path(summary_paths[0]).stem
        output_path = output_dir / f"{first_stem}_compact.csv"
        rows = []
        for summary_path in summary_paths:
            with Path(summary_path).open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                source_row = next(reader, {})
            roi = self._mechanoregulation_summary_roi(Path(summary_path), source_row)
            rows.extend(self._mechanoregulation_compact_rows(roi, source_row))
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["ROI", "Metric", "Unit", "Low conf", "Median", "High conf"])
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    @staticmethod
    def _mechanoregulation_summary_roi(summary_path: Path, source_row: dict) -> str:
        value = str(source_row.get("roi") or source_row.get("compartment") or "").strip()
        if value:
            return value
        match = re.search(r"_roi-([^_]+)", summary_path.stem)
        return match.group(1) if match else "ROI"

    @classmethod
    def _mechanoregulation_compact_rows(cls, roi: str, source_row: dict) -> list[dict]:
        def first_value(*names):
            for name in names:
                value = source_row.get(name)
                if value not in (None, ""):
                    return value
            return ""

        return [
            cls._mechanoregulation_compact_row(roi, "CCR", "fraction", "", first_value("CCR", "ccr"), ""),
            cls._mechanoregulation_compact_row(
                roi,
                "Lazy min",
                "% normalized SED",
                "",
                first_value("CCR_low_threshold", "lazy_min", "lazy_zone_low", "sed_lazy_min"),
                "",
            ),
            cls._mechanoregulation_compact_row(
                roi,
                "Lazy max",
                "% normalized SED",
                "",
                first_value("CCR_high_threshold", "lazy_max", "lazy_zone_high", "sed_lazy_max"),
                "",
            ),
            cls._mechanoregulation_compact_row(
                roi,
                "ORR",
                "% per 1% SED decrease",
                first_value("OR_R_CI_low", "ORR_low_conf", "orr_low_conf"),
                first_value("OR_R", "ORR", "orr"),
                first_value("OR_R_CI_high", "ORR_high_conf", "orr_high_conf"),
            ),
            cls._mechanoregulation_compact_row(
                roi,
                "ORF",
                "% per 1% SED increase",
                first_value("OR_F_CI_low", "ORF_low_conf", "orf_low_conf"),
                first_value("OR_F", "ORF", "orf"),
                first_value("OR_F_CI_high", "ORF_high_conf", "orf_high_conf"),
            ),
        ]

    @classmethod
    def _mechanoregulation_compact_row(cls, roi: str, metric: str, unit: str, low, median, high) -> dict:
        return {
            "ROI": str(roi),
            "Metric": metric,
            "Unit": unit,
            "Low conf": cls._format_compact_table_value(low),
            "Median": cls._format_compact_table_value(median),
            "High conf": cls._format_compact_table_value(high),
        }

    @staticmethod
    def _format_compact_table_value(value) -> str:
        text = str(value).strip()
        if not text:
            return ""
        try:
            number = float(text)
        except ValueError:
            return text
        if not np.isfinite(number):
            return ""
        return f"{number:.4g}"

    def _load_mechanoregulation_context_volumes(self, row, folder_name):
        loaded = 0
        sed_node = None
        context_paths = [("sed", str(row.get("sed_path") or "").strip())]
        for role, path_text in context_paths:
            if not path_text:
                continue
            path = Path(path_text)
            if not path.exists():
                continue
            try:
                self._remove_existing_node_named(path.stem)
                if role == "remodelling" and hasattr(slicer.util, "loadLabelVolume"):
                    result = slicer.util.loadLabelVolume(str(path), {"name": path.stem})
                else:
                    result = slicer.util.loadVolume(str(path), {"name": path.stem})
                if isinstance(result, tuple):
                    success = bool(result[0])
                    node = result[1] if len(result) > 1 else None
                else:
                    success = bool(result)
                    node = result
                if not success or node is None:
                    continue
                if role == "sed":
                    self._style_fea_volume(node, path)
                    sed_node = node
                self._put_node_in_subject_hierarchy_folder(node, folder_name)
                loaded += 1
            except Exception as exc:
                self._append_log(f"[batch] Could not load mechanoregulation {role} context {path.name}: {exc}")
        remodelling_path = Path(str(row.get("image_path") or "").strip())
        if remodelling_path.exists():
            loaded += self._load_mechanoregulation_remodelling_as_segmentation(
                row,
                remodelling_path,
                folder_name,
                sed_node,
            )
        return loaded

    def _load_mechanoregulation_remodelling_as_segmentation(self, row, path: Path, folder_name: str, sed_node):
        if "_remodelling" not in path.name.lower():
            return 0
        node_name = f"{path.stem}_formation-resorption"
        self._remove_existing_node_named(path.stem)
        self._remove_existing_node_named(node_name)
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", node_name)
        segmentation_node.CreateDefaultDisplayNodes()
        temporary_nodes = []
        try:
            if sed_node is not None:
                segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(sed_node)
            remodelling_image = sitk.ReadImage(str(path))
            remodelling_array = sitk.GetArrayFromImage(remodelling_image)
            masks = (
                ("resorption", "Resorption", np.isin(remodelling_array, (1,))),
                ("formation", "Formation", np.isin(remodelling_array, (3, 4))),
            )
            for role, segment_name, mask in masks:
                if not np.any(mask):
                    continue
                label_node = self._load_binary_event_labelmap(mask, remodelling_image, path, role, sed_node)
                temporary_nodes.append(label_node)
                if sed_node is None and segmentation_node.GetSegmentation().GetNumberOfSegments() == 0:
                    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(label_node)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label_node, segmentation_node)
                self._name_last_segment(segmentation_node, segment_name, role)
            if segmentation_node.GetSegmentation().GetNumberOfSegments() < 1:
                slicer.mrmlScene.RemoveNode(segmentation_node)
                return 0
            segmentation_node.SetAttribute("BoneImaging.Mechanoregulation.RemodellingSource", str(path))
            if sed_node is not None:
                segmentation_node.SetAttribute("BoneImaging.Mechanoregulation.SEDNodeID", sed_node.GetID())
                segmentation_node.SetAttribute("BoneImaging.Mechanoregulation.SEDNodeName", sed_node.GetName())
                try:
                    slicer.util.setSliceViewerLayers(background=sed_node, fit=False)
                except Exception:
                    pass
            self._put_node_in_subject_hierarchy_folder(segmentation_node, folder_name)
            self._center_slices_on_node(segmentation_node)
            self._append_log("[batch] loaded mechanoregulation formation/resorption segmentation")
            return 1
        except Exception as exc:
            try:
                slicer.mrmlScene.RemoveNode(segmentation_node)
            except Exception:
                pass
            self._append_log(f"[batch] Could not load mechanoregulation remodelling segmentation {path.name}: {exc}")
            return 0
        finally:
            for node in temporary_nodes:
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass

    def _load_binary_event_labelmap(self, mask, reference_image, source_path: Path, role: str, reference_node):
        with tempfile.TemporaryDirectory(prefix="hrpqct_mechreg_events_") as temp_dir:
            temp_path = Path(temp_dir) / f"{source_path.stem}_{role}.nrrd"
            label_image = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8), isVector=False)
            label_image.CopyInformation(reference_image)
            sitk.WriteImage(label_image, str(temp_path))
            loaded = slicer.util.loadLabelVolume(str(temp_path), {"name": f"{source_path.stem}_{role}_label"})
        if isinstance(loaded, tuple):
            success, label_node = loaded
        else:
            success, label_node = bool(loaded), loaded
        if not success or label_node is None:
            raise RuntimeError(f"Could not load mechanoregulation event labelmap: {source_path.name}")
        if reference_node is not None:
            label_node.CopyOrientation(reference_node)
        return label_node

    def _load_timelapse_outputs(self, row, output_paths):
        loaded = 0
        folder_name = self._timelapse_folder_name(row)
        for path in output_paths:
            if not path.exists():
                continue
            try:
                self._remove_existing_node_named(path.stem)
                if path.suffix.lower() == ".csv":
                    result = self._load_timelapse_summary_table(path, row)
                elif path.name.lower().endswith((".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")):
                    result = slicer.util.loadVolume(str(path), {"name": path.stem})
                else:
                    continue
                if isinstance(result, tuple):
                    success = bool(result[0])
                    node = result[1] if len(result) > 1 else None
                else:
                    success = bool(result)
                    node = result
                if success:
                    if node is not None:
                        if path.suffix.lower() != ".csv":
                            self._style_timelapse_remodelling_volume(node, path)
                        self._put_node_in_subject_hierarchy_folder(node, folder_name)
                    loaded += 1
            except Exception as exc:
                self._append_log(f"[batch] Could not load {path.name}: {exc}")
        self._append_log(f"[batch] loaded Timelapse outputs: {loaded} file(s)")

    def _load_timelapse_summary_table(self, csv_path: Path, row):
        summary_path = self._write_timelapse_summary_csv(csv_path, row)
        name = f"{Path(csv_path).stem}_summary"
        self._remove_existing_node_named(name)
        table_node = self._table_node_from_csv(summary_path, name)
        self._show_table_node(table_node)
        return table_node

    def _show_table_node(self, table_node):
        try:
            layout_manager = slicer.app.layoutManager()
            layout_with_table = slicer.modules.tables.logic().GetLayoutWithTable(layout_manager.layout)
            layout_manager.setLayout(layout_with_table)
            slicer.app.applicationLogic().GetSelectionNode().SetActiveTableID(table_node.GetID())
            slicer.app.applicationLogic().PropagateTableSelection()
        except Exception as exc:
            self._append_log(f"[batch] could not show table view: {exc}")

    @staticmethod
    def _table_node_from_csv(csv_path: Path, name: str):
        with Path(csv_path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = [dict(row) for row in reader]
            columns = list(reader.fieldnames or [])
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        for column_name in columns:
            column = vtk.vtkStringArray()
            column.SetName(column_name)
            for row in rows:
                column.InsertNextValue(str(row.get(column_name, "")))
            table_node.AddColumn(column)
        try:
            table_node.SetUseColumnTitleAsColumnHeader(True)
            table_node.Modified()
            table_node.GetTable().Modified()
        except Exception:
            pass
        return table_node

    @staticmethod
    def _write_timelapse_summary_csv(csv_path: Path, row) -> Path:
        headers = ["Sample", "Pair", "Profile", "ROI", "FV/BV", "RV/BV", "AV/BV", "NV/BV"]
        output_dir = Path(tempfile.gettempdir()) / "SlicerBoneImagingToolbox" / "BatchProcessor" / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(csv_path).stem}_summary.csv"
        with Path(csv_path).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            with output_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=headers)
                writer.writeheader()
                for input_row in reader:
                    formation = BatchProcessorWidget._csv_float(input_row.get("formation_frac_bv0"))
                    resorption = BatchProcessorWidget._csv_float(input_row.get("resorption_frac_bv0"))
                    writer.writerow(
                        {
                            "Sample": BatchProcessorWidget._timelapse_sample_text(input_row, row),
                            "Pair": BatchProcessorWidget._timelapse_pair_text(input_row),
                            "Profile": input_row.get("profile") or "",
                            "ROI": input_row.get("compartment") or "",
                            "FV/BV": BatchProcessorWidget._format_fraction(formation),
                            "RV/BV": BatchProcessorWidget._format_fraction(resorption),
                            "AV/BV": BatchProcessorWidget._format_fraction(
                                None if formation is None or resorption is None else formation + resorption
                            ),
                            "NV/BV": BatchProcessorWidget._format_fraction(
                                None if formation is None or resorption is None else formation - resorption
                            ),
                        }
                    )
        return output_path

    @staticmethod
    def _timelapse_sample_text(input_row, fallback_row):
        subject = str(input_row.get("subject_id") or fallback_row.get("subject") or "").strip()
        site = str(input_row.get("site") or fallback_row.get("voi_value", fallback_row.get("voi")) or "").strip()
        parts = []
        if subject:
            parts.append(f"sub-{subject}")
        if site:
            parts.append(f"voi-{site}")
        return " ".join(parts)

    @staticmethod
    def _timelapse_pair_text(input_row):
        t0 = str(input_row.get("t0") or input_row.get("session_t0_original") or "").strip()
        t1 = str(input_row.get("t1") or input_row.get("session_t1_original") or "").strip()
        return f"{t0}-{t1}" if t0 or t1 else ""

    @staticmethod
    def _csv_float(value):
        try:
            text = str(value).strip()
            return float(text) if text else None
        except Exception:
            return None

    @staticmethod
    def _format_fraction(value):
        if value is None:
            return ""
        return f"{float(value):.6g}"

    @staticmethod
    def _timelapse_folder_name(row):
        subject = str(row.get("subject") or "unknown")
        voi = str(row.get("voi_value", row.get("voi")) or "unknown")
        return f"sub-{subject}_voi-{voi}_timelapse-remodelling"

    @staticmethod
    def _style_timelapse_remodelling_volume(node, path):
        if node is None or "_remodelling" not in Path(path).name.lower():
            return
        try:
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
        except Exception:
            display_node = None
        if display_node is None:
            return
        color_node = BatchProcessorWidget._remodelling_color_node()
        color_node_id = color_node.GetID() if color_node is not None else "vtkMRMLColorTableNodeFileLabels.txt"
        try:
            display_node.SetAndObserveColorNodeID(color_node_id)
            display_node.SetVisibility(True)
            if hasattr(display_node, "SetOpacity"):
                display_node.SetOpacity(1.0)
            if hasattr(display_node, "SetInterpolate"):
                display_node.SetInterpolate(False)
            elif hasattr(display_node, "InterpolateOff"):
                display_node.InterpolateOff()
            if hasattr(display_node, "AutoWindowLevelOff"):
                display_node.AutoWindowLevelOff()
            if hasattr(display_node, "SetWindowLevel"):
                display_node.SetWindowLevel(5.0, 2.5)
            elif hasattr(display_node, "SetWindow") and hasattr(display_node, "SetLevel"):
                display_node.SetWindow(5.0)
                display_node.SetLevel(2.5)
        except Exception:
            pass

    @staticmethod
    def _remodelling_color_node():
        name = "Timelapse_RemodellingColors"
        try:
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None:
                return existing
            color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", name)
            if hasattr(color_node, "SetTypeToUser"):
                color_node.SetTypeToUser()
            if hasattr(color_node, "SetNumberOfColors"):
                color_node.SetNumberOfColors(6)
            colors = {
                0: ("background", 0.0, 0.0, 0.0, 0.0),
                1: ("resorption", 1.00, 0.05, 0.70, 1.0),
                2: ("quiescent", 0.62, 0.62, 0.62, 0.32),
                3: ("formation", 1.00, 0.48, 0.00, 1.0),
                4: ("formation", 1.00, 0.48, 0.00, 1.0),
                5: ("quiescent", 0.62, 0.62, 0.62, 0.32),
            }
            for value, color in colors.items():
                color_node.SetColor(
                    int(value),
                    color[0],
                    float(color[1]),
                    float(color[2]),
                    float(color[3]),
                    float(color[4]),
                )
            if hasattr(color_node, "HideFromEditorsOn"):
                color_node.HideFromEditorsOn()
            return color_node
        except Exception:
            return None

    def _deduplicated_paths(self, paths):
        seen = set()
        output = []
        for path in paths:
            path = Path(path)
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            output.append(path)
        return output

    @staticmethod
    def _map_role_from_path(path: Path) -> str:
        stem = path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem
        if "_map-" not in stem:
            return ""
        return stem.rsplit("_map-", 1)[-1].replace("-", ".")

    @staticmethod
    def _load_plate_rod_npy_map(path: Path):
        array = np.load(str(path), allow_pickle=False)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", Path(path).stem)
        slicer.util.updateVolumeFromArray(node, np.asarray(array))
        node.CreateDefaultDisplayNodes()
        node.SetAttribute("BoneImaging.PlateRod.Engine", "plate_rod_thinning")
        node.SetAttribute("BoneImaging.PlateRod.MapRole", "skeleton" if "skeleton" in path.name.lower() else "plate_rod")
        return node

    @staticmethod
    def _style_plate_rod_volume(node, path):
        if node is None:
            return
        try:
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
        except Exception:
            display_node = None
        if display_node is None:
            return
        try:
            color_node = BatchProcessorWidget._plate_rod_color_node()
            color_node_id = color_node.GetID() if color_node is not None else "vtkMRMLColorTableNodeFileLabels.txt"
            display_node.SetAndObserveColorNodeID(color_node_id)
            display_node.SetVisibility(True)
            if hasattr(display_node, "SetInterpolate"):
                display_node.SetInterpolate(False)
            elif hasattr(display_node, "InterpolateOff"):
                display_node.InterpolateOff()
        except Exception:
            pass

    @staticmethod
    def _plate_rod_color_node():
        name = "PlateRodMorphometry_Colors"
        try:
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None:
                return existing
            color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", name)
            if hasattr(color_node, "SetTypeToUser"):
                color_node.SetTypeToUser()
            if hasattr(color_node, "SetNumberOfColors"):
                color_node.SetNumberOfColors(4)
            colors = {
                0: ("background", 0.0, 0.0, 0.0, 0.0),
                1: ("plate", 0.10, 0.45, 1.00, 1.0),
                2: ("rod", 1.00, 0.10, 0.08, 1.0),
                3: ("junction", 1.00, 0.72, 0.05, 1.0),
            }
            for value, color in colors.items():
                color_node.SetColor(
                    int(value),
                    color[0],
                    float(color[1]),
                    float(color[2]),
                    float(color[3]),
                    float(color[4]),
                )
            if hasattr(color_node, "HideFromEditorsOn"):
                color_node.HideFromEditorsOn()
            return color_node
        except Exception:
            return None

    def _style_microarchitecture_volume(self, node, path):
        if node is None or "_map-" not in Path(path).name.lower():
            return
        map_role = self._map_role_from_path(Path(path))
        try:
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
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

    @staticmethod
    def _style_fea_volume(node, path):
        if node is None:
            return
        if "_map-sed" not in Path(path).name.lower() and Path(path).name.lower() != "sed.nii.gz":
            return
        try:
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
        except Exception:
            display_node = None
        if display_node is None:
            return
        color_node = BatchProcessorWidget._fea_sed_color_node()
        if color_node is not None:
            try:
                display_node.SetAndObserveColorNodeID(color_node.GetID())
            except Exception:
                pass
        for method_name in ("AutoWindowLevelOn", "AutoThresholdOff"):
            method = getattr(display_node, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    @staticmethod
    def _fea_sed_color_node():
        name = "BatchProcessor_FEA_SED_JET"
        try:
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None:
                return existing
            color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", name)
            color_node.SetTypeToUser()
            color_node.SetNumberOfColors(256)
            color_node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
            for index in range(1, 256):
                t = index / 255.0
                red = min(max(1.5 - abs(4.0 * t - 3.0), 0.0), 1.0)
                green = min(max(1.5 - abs(4.0 * t - 2.0), 0.0), 1.0)
                blue = min(max(1.5 - abs(4.0 * t - 1.0), 0.0), 1.0)
                color_node.SetColor(index, f"sed_{index}", red, green, blue, 1.0)
            try:
                color_node.HideFromEditorsOn()
            except Exception:
                pass
            return color_node
        except Exception:
            return None

    @staticmethod
    def _microarchitecture_map_folder_name(path, row):
        path = Path(path)
        text = path.name
        path_text = str(path).replace("\\", "/").lower()
        suffix = "_xct_registered_microstructure" if "/registered_measurements/" in path_text else "_xct_microstructure"
        match = re.search(
            r"(?i)(sub-[^_]+)_([^_]*ses-[^_]+)_voi-([^_]+)",
            text,
        )
        if match:
            return f"{match.group(1)}_{match.group(2)}_voi-{match.group(3)}{suffix}"
        subject = str(row.get("subject") or "unknown")
        session = str(row.get("session") or "unknown")
        voi = str(row.get("site") or row.get("voi") or "unknown")
        return f"sub-{subject}_ses-{session}_voi-{voi}{suffix}"

    @staticmethod
    def _plate_rod_output_folder_name(path, row):
        path = Path(path)
        path_text = str(path).replace("\\", "/").lower()
        suffix = "_xct_registered_plate-rod" if "/registered_measurements/" in path_text else "_xct_plate-rod"
        match = re.search(
            r"(?i)(sub-[^_]+)_([^_]*ses-[^_]+)_voi-([^_]+)",
            path.name,
        )
        if match:
            return f"{match.group(1)}_{match.group(2)}_voi-{match.group(3)}{suffix}"
        subject = str(row.get("subject") or "unknown")
        session = str(row.get("session") or "unknown")
        voi = str(row.get("site") or row.get("voi") or "unknown")
        return f"sub-{subject}_ses-{session}_voi-{voi}{suffix}"

    @staticmethod
    def _fea_output_folder_name(row):
        subject = str(row.get("subject") or "unknown")
        session = str(row.get("session") or "unknown")
        voi = str(row.get("voi_value", row.get("voi")) or "unknown")
        profile = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(row.get("profile") or "parosol-fea")).strip("-")
        return f"sub-{subject}_ses-{session}_voi-{voi}_{profile}_parosol-fea"

    @staticmethod
    def _mechanoregulation_folder_name(row):
        subject = str(row.get("subject") or "unknown")
        session = str(row.get("session") or "unknown")
        voi = str(row.get("voi_value", row.get("voi")) or "unknown")
        profile = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(row.get("profile") or "mechanoregulation")).strip("-")
        return f"sub-{subject}_pair-{session}_voi-{voi}_{profile}_mechanoregulation"

    def _registered_group_rows(self, row_index):
        if row_index < 0 or row_index >= len(self._batchRows):
            return []
        row = self._batchRows[row_index]
        span = int(row.get("action_row_span") or 1)
        group_id = row.get("group_id")
        rows = []
        for offset in range(max(1, span)):
            child_index = row_index + offset
            if child_index >= len(self._batchRows):
                break
            child = self._batchRows[child_index]
            if offset == 0 or (group_id and child.get("group_id") == group_id):
                rows.append(child)
        return rows

    def _load_registered_common_region_overlays(self, row_index):
        rows = self._registered_group_rows(row_index)
        if not rows:
            return
        loaded = 0
        for row in rows:
            paths = [
                Path(path)
                for path in self.logic.common_region_paths_for_row(self.datasetRootEdit.text, row)
            ]
            paths = self._deduplicated_paths(paths)
            if not paths:
                continue
            loaded += self._load_common_region_outputs_as_segmentation(row, paths)
        if loaded:
            self._append_log(f"[batch] loaded {loaded} common-region segmentation(s)")

    def _load_common_region_outputs_as_segmentation(self, row, output_paths):
        mask_paths = [Path(path) for path in output_paths if Path(path).exists()]
        if not mask_paths:
            return 0
        subject = str(row.get("subject") or "unknown")
        session = str(row.get("session_value", row.get("session")) or "unknown")
        voi = str(row.get("voi_value", row.get("voi")) or "unknown")
        node_name = f"sub-{subject}_ses-{session}_voi-{voi}_common-region"
        self._remove_existing_node_named(node_name)
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            node_name,
        )
        segmentation_node.CreateDefaultDisplayNodes()
        reference_node = self._ensure_loaded_source_volume(row.get("image_path"))
        temporary_nodes = []
        try:
            if reference_node is not None:
                segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_node)
            for path in sorted(mask_paths):
                label_node = self._load_mask_as_labelmap(path, "common_region", reference_node)
                temporary_nodes.append(label_node)
                if reference_node is None:
                    reference_node = label_node
                    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_node)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label_node, segmentation_node)
                self._name_last_segment(segmentation_node, f"Common region ses-{session}", "common_region")
            segmentation_node.SetAttribute("BoneImaging.MaskRoles", "common_region")
            segmentation_node.SetAttribute("BoneImaging.CommonRegion", "scan_region_native_common")
            self._put_node_in_subject_hierarchy_folder(
                segmentation_node,
                self._registered_common_region_folder_name(row),
            )
            self._center_slices_on_node(segmentation_node)
            return 1
        except Exception as exc:
            try:
                slicer.mrmlScene.RemoveNode(segmentation_node)
            except Exception:
                pass
            self._append_log(f"[batch] Could not load CommonRegion segmentation: {exc}")
            return 0
        finally:
            for node in temporary_nodes:
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass

    def _registered_common_region_folder_name(self, row):
        subject = str(row.get("subject") or "unknown")
        session = str(row.get("session_value", row.get("session")) or "unknown")
        voi = str(row.get("voi_value", row.get("voi")) or "unknown")
        if self._selected_tool_key() == "plate_rod":
            return f"sub-{subject}_ses-{session}_voi-{voi}_xct_registered_plate-rod_common-region"
        return f"sub-{subject}_ses-{session}_voi-{voi}_xct_registered_microstructure_common-region"

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

    @staticmethod
    def _remove_existing_node_named(name):
        try:
            matches = slicer.mrmlScene.GetNodesByName(str(name))
            matches.UnRegister(None)
            for index in range(matches.GetNumberOfItems()):
                node = matches.GetItemAsObject(index)
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
        except Exception:
            pass

    def _load_bone_contour_outputs_as_segmentation(self, row, output_paths):
        mask_paths = [Path(path) for path in output_paths if self._is_bone_contour_segmentation_output(Path(path))]
        if not mask_paths:
            self._append_log("[batch] No BoneContours mask/label outputs were discovered for this row.")
            return
        source_name = Path(str(row.get("image_path") or mask_paths[0])).stem
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            f"{source_name}_contours",
        )
        segmentation_node.CreateDefaultDisplayNodes()
        reference_node = self._ensure_loaded_source_volume(row.get("image_path"))
        loaded_roles = []
        temporary_nodes = []
        try:
            if reference_node is not None:
                segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_node)
            for path in sorted(mask_paths, key=self._mask_role_sort_key):
                if not path.exists():
                    continue
                role = self._mask_role_from_path(path)
                label_node = self._load_mask_as_labelmap(path, role, reference_node)
                temporary_nodes.append(label_node)
                if reference_node is None:
                    reference_node = label_node
                    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_node)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label_node, segmentation_node)
                self._name_last_segment(segmentation_node, self._segment_name_for_role(role), role)
                loaded_roles.append(role)
            if loaded_roles:
                segmentation_node.SetAttribute("BoneImaging.MaskRoles", ",".join(loaded_roles))
                if reference_node is not None:
                    slicer.util.setSliceViewerLayers(background=reference_node, fit=False)
                self._center_slices_on_node(segmentation_node)
                self._append_log(f"[batch] loaded BoneContours segmentation with masks: {', '.join(loaded_roles)}")
            else:
                slicer.mrmlScene.RemoveNode(segmentation_node)
                self._append_log("[batch] No readable BoneContours mask outputs were found.")
        except Exception as exc:
            try:
                slicer.mrmlScene.RemoveNode(segmentation_node)
            except Exception:
                pass
            self._append_log(f"[batch] Could not load BoneContours segmentation: {exc}")
        finally:
            for node in temporary_nodes:
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass

    @staticmethod
    def _is_mask_output(path: Path) -> bool:
        name = path.name.lower()
        return "_mask" in name and name.endswith((".aim", ".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd"))

    @staticmethod
    def _is_label_output(path: Path) -> bool:
        name = path.name.lower()
        return "_label" in name and name.endswith((".aim", ".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd"))

    @staticmethod
    def _is_bone_contour_segmentation_output(path: Path) -> bool:
        role = BatchProcessorWidget._mask_role_from_path(path)
        if BatchProcessorWidget._is_fea_material_role(role):
            return False
        return BatchProcessorWidget._is_mask_output(path) or BatchProcessorWidget._is_label_output(path)

    @staticmethod
    def _is_fea_material_role(role: str) -> bool:
        role_text = str(role or "").lower().replace("-", "_")
        return any(token in role_text for token in ("fea_material", "material", "hom_ls", "model_label"))

    @staticmethod
    def _mask_role_from_path(path: Path) -> str:
        match = re.search(r"_desc-([^_]+)_(?:mask|label)", path.name, re.IGNORECASE)
        return match.group(1).lower() if match else "mask"

    @staticmethod
    def _mask_role_sort_key(path: Path):
        order = {"full": 0, "trab": 1, "cort": 2, "seg": 3}
        role = BatchProcessorWidget._mask_role_from_path(path)
        return order.get(role, 99), path.name

    @staticmethod
    def _segment_name_for_role(role: str) -> str:
        return {
            "full": "Full mask",
            "trab": "Trabecular mask",
            "cort": "Cortical mask",
            "seg": "Bone segmentation",
        }.get(str(role), str(role).replace("_", " ").title())

    def _load_mask_as_labelmap(self, path: Path, role: str, reference_node):
        with tempfile.TemporaryDirectory(prefix="hrpqct_batch_mask_") as temp_dir:
            temp_path = Path(temp_dir) / f"{path.stem}_{role}.nrrd"
            mask_image = self._read_mask_image(path)
            if self._is_label_output(path):
                sitk.WriteImage(sitk.Cast(mask_image, sitk.sitkUInt8), str(temp_path))
            else:
                sitk.WriteImage(sitk.Cast(mask_image > 0, sitk.sitkUInt8), str(temp_path))
            loaded = slicer.util.loadLabelVolume(str(temp_path), {"name": f"{path.stem}_{role}_label"})
        if isinstance(loaded, tuple):
            success, label_node = loaded
        else:
            success, label_node = bool(loaded), loaded
        if not success or label_node is None:
            raise RuntimeError(f"Could not load mask labelmap: {path.name}")
        if reference_node is not None:
            label_node.CopyOrientation(reference_node)
        return label_node

    @staticmethod
    def _read_mask_image(path: Path):
        if path.name.lower().endswith(".aim") or re.search(r"\.aim;\d+$", path.name, re.IGNORECASE):
            from ScancoIOLib import aim_io

            image, metadata = aim_io.read_aim(path, scaling="native")
            del metadata
            return image
        return sitk.ReadImage(str(path))

    @staticmethod
    def _name_last_segment(segmentation_node, segment_name: str, role: str) -> None:
        segmentation = segmentation_node.GetSegmentation()
        if segmentation.GetNumberOfSegments() < 1:
            return
        segment_id = segmentation.GetNthSegmentID(segmentation.GetNumberOfSegments() - 1)
        segment = segmentation.GetSegment(segment_id)
        if segment is None:
            return
        segment.SetName(segment_name)
        color = _SEGMENT_COLORS.get(str(role))
        if color is not None:
            segment.SetColor(color[0], color[1], color[2])
        if hasattr(segment, "SetTag"):
            segment.SetTag("HRpQCT.Role", role)

    @staticmethod
    def _find_loaded_source_volume(image_path):
        if not image_path:
            return None
        target = Path(str(image_path))
        target_name = target.name
        target_stem = target.stem
        try:
            nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        except Exception:
            return None
        for node in nodes:
            candidates = [
                node.GetName() if hasattr(node, "GetName") else "",
                node.GetAttribute("HRpQCT.AIMSourcePath") if hasattr(node, "GetAttribute") else "",
            ]
            storage = node.GetStorageNode() if hasattr(node, "GetStorageNode") else None
            if storage is not None and hasattr(storage, "GetFileName"):
                candidates.append(storage.GetFileName())
            for candidate in candidates:
                if not candidate:
                    continue
                candidate_path = Path(str(candidate))
                if candidate_path == target or candidate_path.name == target_name or candidate_path.stem == target_stem:
                    return node
        return None

    def _ensure_loaded_source_volume(self, image_path):
        existing = self._find_loaded_source_volume(image_path)
        if existing is not None or not image_path:
            return existing
        image_path = Path(str(image_path))
        if not image_path.exists():
            return None
        try:
            from ScancoIO import ScancoIOLogic

            return ScancoIOLogic().import_image(image_path, scaling="density", load_as="volume")
        except Exception as exc:
            self._append_log(f"[batch] Could not load source image through Scanco I/O: {exc}")
            return None

    @staticmethod
    def _center_slices_on_node(node):
        try:
            bounds = [0.0] * 6
            node.GetRASBounds(bounds)
            cx = 0.5 * (bounds[0] + bounds[1])
            cy = 0.5 * (bounds[2] + bounds[3])
            cz = 0.5 * (bounds[4] + bounds[5])
            slicer.modules.markups.logic().JumpSlicesToLocation(cx, cy, cz, True)
        except Exception:
            pass


class BatchProcessorTest(ScriptedLoadableModuleTest):
    def runTest(self):
        pass
