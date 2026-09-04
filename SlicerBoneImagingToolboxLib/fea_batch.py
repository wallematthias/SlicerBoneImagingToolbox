from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk

from .derivatives import (
    DerivativeRecord,
    discover_manifests,
    normalize_role,
    normalize_session_id,
    normalize_site,
    normalize_subject_id,
)
from .derivatives import discover_shared_artifacts

try:
    from bone_imaging_derivatives import record_output_path  # type: ignore
except Exception as exc:
    _DERIVATIVES_IMPORT_ERROR = exc

    def record_output_path(*_args, **_kwargs):
        raise RuntimeError(
            "The Bone Imaging Derivative Contract runtime package is not installed. "
            "Open Bone Imaging > Setup and install/update runtime packages."
        ) from _DERIVATIVES_IMPORT_ERROR


IMAGE_SUFFIXES = (".aim", ".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".nhdr")
DERIVATIVES_DIR_NAME = "derivatives"
LABELMAP_BATCH_PROFILES = frozenset({"xtremecti", "xtremectii", "load_history_3", "load_history_6"})


@dataclass(frozen=True)
class FEAArtifact:
    role: str
    path: str
    source: str
    derivative: str = ""
    space: str = "native"


@dataclass(frozen=True)
class FEABatchCase:
    subject_id: str
    site: str
    session_id: str
    artifacts: tuple[FEAArtifact, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject_id, self.site, self.session_id)

    def artifact_options(self, group: str) -> list[str]:
        roles = _role_group_members(group)
        return [artifact.path for artifact in self.artifacts if artifact.role in roles]

    def first_artifact(self, roles: Iterable[str]) -> FEAArtifact | None:
        role_order = tuple(str(role) for role in roles)
        for role in role_order:
            for artifact in self.artifacts:
                if artifact.role == role:
                    return artifact
        return None


@dataclass(frozen=True)
class FEAWorkflowRoleRequirement:
    name: str
    required: bool
    preferred_roles: tuple[str, ...]


def discover_fea_batch_cases(
    dataset_root: str | Path,
    *,
    subject_id: str = "",
    site: str = "",
    session_id: str = "",
) -> list[FEABatchCase]:
    """Discover FEA-ready artifacts and group them by subject, site, and session."""
    root = Path(dataset_root).expanduser().resolve()
    artifacts_by_key: dict[tuple[str, str, str], list[FEAArtifact]] = {}

    def add_artifact(key: tuple[str, str, str], artifact: FEAArtifact) -> None:
        requested_subject = normalize_subject_id(subject_id) or ""
        requested_site = normalize_site(site) or ""
        requested_session = normalize_session_id(session_id) or ""
        if requested_subject and key[0] != requested_subject:
            return
        if requested_site and key[1] != requested_site:
            return
        if requested_session and key[2] != requested_session:
            return
        artifacts_by_key.setdefault(key, [])
        if artifact.path not in {existing.path for existing in artifacts_by_key[key]}:
            artifacts_by_key[key].append(artifact)

    for artifact in discover_shared_artifacts(root, include_derivatives=True).records:
        key = (
            normalize_subject_id(_clean_token(artifact.subject_id)) or "",
            normalize_site(_clean_token(artifact.site)) or "",
            normalize_session_id(_clean_token(artifact.session_id)) or "",
        )
        if not key[0] or not key[1]:
            continue
        derivative = _artifact_derivative_family(artifact.path, root)
        role = _normalize_artifact_role(artifact.role, str(artifact.path), derivative=derivative)
        if role == "field_map":
            continue
        if artifact.kind == "image" and artifact.role == "map" and role not in {
            "material_labelmap",
            "hom_ls_model",
            "model_labelmap",
            "calibrated_image",
            "density_image",
        }:
            continue
        add_artifact(
            key,
            FEAArtifact(
                role=role,
                path=str(Path(artifact.path).expanduser()),
                source=artifact.metadata.get("source", "artifact") if isinstance(artifact.metadata, dict) else "artifact",
                derivative=derivative,
            ),
        )

    for record in _iter_manifest_records(root):
        key = (
            normalize_subject_id(_clean_token(record.subject_id)) or "",
            normalize_site(_clean_token(record.site)) or "",
            normalize_session_id(_clean_token(record.session_id)) or "",
        )
        if not key[0] or not key[1]:
            continue
        role = _normalize_artifact_role(record.role, record.path, derivative=record.derivative)
        add_artifact(
            key,
            FEAArtifact(
                role=role,
                path=str(Path(record.path).expanduser()),
                source=record.source or "manifest",
                derivative=record.derivative,
                space=record.space or "native",
            ),
        )

    for path in _iter_dataset_image_paths(root):
        tokens = _tokens_from_path(path, root)
        if not tokens["subject_id"] or not tokens["site"]:
            continue
        key = (tokens["subject_id"], tokens["site"], tokens["session_id"])
        add_artifact(
            key,
            FEAArtifact(
                role=_normalize_artifact_role("", str(path)),
                path=str(path),
                source="dataset",
            ),
        )

    cases = [
        FEABatchCase(
            subject_id=key[0],
            site=key[1],
            session_id=key[2],
            artifacts=tuple(_sort_artifacts(artifacts)),
        )
        for key, artifacts in artifacts_by_key.items()
    ]
    return sorted(cases, key=lambda case: case.key)


def workflow_role_requirements(workflow: str) -> dict[str, FEAWorkflowRoleRequirement]:
    """Return the artifact roles a workflow can consume, independent of folder layout."""
    key = str(workflow or "").strip().lower()
    if key in LABELMAP_BATCH_PROFILES:
        return {
            "image": FEAWorkflowRoleRequirement(
                "image",
                True,
                ("material_labelmap", "hom_ls_model", "model_labelmap", "labelmap"),
            ),
            "mask": FEAWorkflowRoleRequirement(
                "mask",
                False,
                ("mask_full", "mask", "common_region_mask", "mask_cort", "mask_trab"),
            ),
        }
    if "spine" in key or "vertebra" in key:
        return {
            "image": FEAWorkflowRoleRequirement(
                "image",
                True,
                ("calibrated_image", "density_image", "image", "raw_image"),
            ),
            "mask": FEAWorkflowRoleRequirement(
                "mask",
                False,
                ("vertebra_mask", "mask_full", "mask", "common_region_mask"),
            ),
        }
    return {
        "image": FEAWorkflowRoleRequirement(
            "image",
            True,
            ("calibrated_image", "density_image", "material_labelmap", "image", "raw_image"),
        ),
        "mask": FEAWorkflowRoleRequirement(
            "mask",
            False,
            ("mask_full", "mask", "common_region_mask", "vertebra_mask", "mask_cort", "mask_trab"),
        ),
    }


def batch_profile_support_status(workflow: str) -> tuple[bool, str]:
    """Return whether the batch discovery contract is implemented for a ParOSol profile."""
    key = str(workflow or "").strip().lower()
    if key in LABELMAP_BATCH_PROFILES:
        return True, ""
    return False, "Batch discovery for this profile is not implemented yet."


def build_parosol_case_commands(
    dataset_root: str | Path,
    cases: Iterable[FEABatchCase],
    *,
    workflow: str,
    selected_roles: dict[str, str] | None = None,
    dry_run: bool = False,
) -> list[list[str]]:
    """Build one ParOSol shortcut command argument list for each ready batch case."""
    root = str(Path(dataset_root).expanduser().resolve())
    selected_roles = selected_roles or {}
    supported, _message = batch_profile_support_status(workflow)
    if not supported:
        return []
    requirements = workflow_role_requirements(workflow)
    commands: list[list[str]] = []
    for case in cases:
        case_id = _case_name(case, workflow)
        image_roles = (selected_roles.get("image"),) if selected_roles.get("image") else requirements["image"].preferred_roles
        image = case.first_artifact(role for role in image_roles if role)
        if image is None:
            continue
        args = [
            image.path,
            "--profile",
            str(workflow),
        ]
        mask_requirement = requirements.get("mask")
        mask_roles = (
            (selected_roles.get("mask"),)
            if selected_roles.get("mask")
            else (() if mask_requirement is None else mask_requirement.preferred_roles)
        )
        mask = case.first_artifact(role for role in mask_roles if role)
        if mask is not None:
            args.extend(["--mask", mask.path])
        elif mask_requirement is not None and mask_requirement.required:
            continue
        args.extend(
            [
                "--dataset-root",
                root,
                "--subject",
                case.subject_id,
                "--site",
                case.site,
                "--name",
                case_id,
            ]
        )
        if dry_run:
            args.append("--dry-run")
        commands.append(args)
    return commands


def case_readiness(
    case: FEABatchCase,
    workflow: str,
    selected_roles: dict[str, str] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Report whether a case has the artifacts required by a workflow."""
    selected_roles = selected_roles or {}
    missing = []
    for group, requirement in workflow_role_requirements(workflow).items():
        roles = (selected_roles.get(group),) if selected_roles.get(group) else requirement.preferred_roles
        has_existing = case.first_artifact(role for role in roles if role) is not None
        if requirement.required and not has_existing:
            missing.append(group)
    return not missing, tuple(missing)


def discovered_role_options(cases: Iterable[FEABatchCase], group: str) -> list[str]:
    roles = _role_group_members(group)
    found = {
        artifact.role
        for case in cases
        for artifact in case.artifacts
        if artifact.role in roles
    }
    return [role for role in roles if role in found]


def role_options_for_workflow(
    cases: Iterable[FEABatchCase],
    workflow: str,
    group: str,
) -> list[str]:
    """Return discovered artifact roles that the selected workflow can consume."""
    cases = tuple(cases)
    requirement = workflow_role_requirements(workflow).get(group)
    if requirement is None:
        return []
    discovered = set(discovered_role_options(cases, group))
    return [role for role in requirement.preferred_roles if role in discovered]


def parosol_command_derivative_context(command: Iterable[str]) -> dict[str, str]:
    """Recover derivative metadata from a ParOSol shortcut command."""
    args = [str(item) for item in command]
    dataset_root = _option_value(args, "--dataset-root")
    subject_id = _option_value(args, "--subject")
    site = _option_value(args, "--site")
    case_id = _option_value(args, "--name") or _case_stem_from_command(args)
    if not dataset_root or not subject_id or not site:
        return {}
    session_id = _first_token(case_id, r"(?:^|[_/\-])ses-?([A-Za-z0-9]+)")
    output_dir = record_output_path(
        Path(dataset_root).expanduser().resolve(),
        "FEA",
        _clean_token(subject_id),
        _clean_token(site),
        "runs",
        case_id,
    )
    return {
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "subject_id": _clean_token(subject_id),
        "site": _clean_token(site),
        "session_id": session_id,
        "case_id": case_id,
        "output_dir": str(output_dir),
    }


def _iter_manifest_records(root: Path) -> Iterable[DerivativeRecord]:
    derivatives_root = root / DERIVATIVES_DIR_NAME
    for manifest in discover_manifests(derivatives_root):
        yield from manifest.records


def _iter_dataset_image_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if DERIVATIVES_DIR_NAME in path.relative_to(root).parts:
            continue
        if _has_image_suffix(path):
            yield path


def _has_image_suffix(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in IMAGE_SUFFIXES)


def _tokens_from_path(path: Path, root: Path) -> dict[str, str]:
    text = "/".join(path.relative_to(root).parts)
    return {
        "subject_id": normalize_subject_id(_first_token(text, r"(?:^|[_/\-])sub-?([A-Za-z0-9]+)")) or "",
        "site": normalize_site(
            _first_token(text, r"(?:^|[_/\-])site-?([A-Za-z0-9]+)")
            or _first_token(text, r"(?:^|[_/\-])voi-?([A-Za-z0-9]+)")
        )
        or "",
        "session_id": normalize_session_id(_first_token(text, r"(?:^|[_/\-])ses-?([A-Za-z0-9]+)")) or "",
    }


def _artifact_derivative_family(path: Path, root: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return ""
    lowered = [part.lower() for part in parts]
    if DERIVATIVES_DIR_NAME in lowered:
        index = lowered.index(DERIVATIVES_DIR_NAME)
        if index + 1 < len(parts):
            return parts[index + 1]
    return ""


def _first_token(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return _clean_token(match.group(1)) if match else ""


def _clean_token(value: object) -> str:
    text = str(value or "").strip()
    for prefix in ("sub-", "site-", "ses-"):
        if text.lower().startswith(prefix):
            return text[len(prefix) :]
    return text


def _normalize_artifact_role(role: str, path: str, *, derivative: str = "") -> str:
    role_text = str(role or "").lower()
    shared_role = normalize_role(role_text)
    derivative_text = str(derivative or "").lower()
    name_text = Path(path).name.lower()
    text = f"{role_text} {name_text} {derivative_text}"
    path_text = str(path).lower()
    if derivative_text in {"microarchitecture", "platerodmorphometry", "plate_rod_morphometry"}:
        return "field_map" if _has_image_suffix(Path(path)) else "table"
    if shared_role == "cort" or "mask-cort" in text or "mask_cort" in text or role_text in {"cort", "cortical", "mask_cort"}:
        return "mask_cort"
    if shared_role == "trab" or "mask-trab" in text or "mask_trab" in text or role_text in {"trab", "trabecular", "mask_trab"}:
        return "mask_trab"
    if (
        shared_role == "full"
        or
        "mask-full" in text
        or "mask_full" in text
        or role_text in {"full", "periosteal", "mask_full"}
        or ("full" in text and "mask" in text)
    ):
        return "mask_full"
    if "common_regions" in path_text or ("common" in text and ("region" in text or "scan" in text)):
        return "common_region_mask"
    if "mask" in role_text or (("mask" in name_text) and "seg" not in name_text):
        return "mask"
    if (
        "hom_ls" in text
        or "hom-ls" in text
        or "modellabel" in text
        or "model_label" in text
        or "model-label" in text
        or "material_label" in text
        or "material-label" in text
        or "material" in text
    ):
        return "material_labelmap"
    if "label" in text:
        return "labelmap"
    if "seg" in text:
        return "segmentation"
    if "remodelling" in text or "remodeling" in text or "/visualize/" in path_text or "/fields/" in path_text:
        return "field_map"
    if "calibrated" in text or "calibration" in text or "bmd" in text or "density" in text:
        return "calibrated_image"
    if re.search(r"(?i)\.aim(?:;\d+)?$", Path(path).name):
        return "raw_image"
    return "image"


def _role_group_members(group: str) -> tuple[str, ...]:
    if group == "image":
        return ("calibrated_image", "density_image", "material_labelmap", "hom_ls_model", "model_labelmap", "labelmap", "segmentation", "image", "raw_image")
    if group == "mask":
        return ("vertebra_mask", "mask_full", "mask", "common_region_mask", "mask_cort", "mask_trab")
    return (group,)


def _sort_artifacts(artifacts: Iterable[FEAArtifact]) -> list[FEAArtifact]:
    order = {
        role: index
        for index, role in enumerate(
            (
                "calibrated_image",
                "density_image",
                "material_labelmap",
                "hom_ls_model",
                "model_labelmap",
                "labelmap",
                "segmentation",
                "image",
                "raw_image",
                "vertebra_mask",
                "mask_full",
                "mask",
                "common_region_mask",
                "mask_cort",
                "mask_trab",
            )
        )
    }
    return sorted(artifacts, key=lambda artifact: (order.get(artifact.role, 99), artifact.path))


def _case_name(case: FEABatchCase, workflow: str) -> str:
    parts = [f"sub-{case.subject_id}"]
    if case.session_id:
        parts.append(f"ses-{case.session_id}")
    parts.append(f"site-{case.site}")
    parts.append(_safe_token(workflow))
    return "_".join(parts)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_.-]+", "-", str(value).strip())
    return token.strip("-") or "workflow"


def _option_value(args: list[str], name: str) -> str:
    try:
        index = args.index(name)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    return str(args[index + 1])


def _case_stem_from_command(args: list[str]) -> str:
    if not args:
        return "parosol_case"
    path = Path(args[0])
    name = path.name
    for suffix in IMAGE_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _workflow_can_generate_material_labelmap(workflow: str, selected_roles: dict[str, str]) -> bool:
    return False


def _can_generate_material_labelmap(case: FEABatchCase) -> bool:
    has_segmentation = case.first_artifact(("segmentation",)) is not None
    has_trab = case.first_artifact(("mask_trab",)) is not None
    has_cort = case.first_artifact(("mask_cort",)) is not None
    has_full = case.first_artifact(("mask_full",)) is not None
    return (
        has_segmentation
        and ((has_trab and has_cort) or (has_full and (has_trab or has_cort)))
    )


def _generate_material_labelmap_artifact(
    dataset_root: Path,
    case: FEABatchCase,
    *,
    case_id: str,
    trab_label: int = 100,
    cort_label: int = 127,
) -> FEAArtifact | None:
    seg = case.first_artifact(("segmentation",))
    trab, cort = _resolve_trab_cort_artifacts(case, generate=True)
    if seg is None or trab is None or cort is None:
        return None
    output_path = (
        dataset_root
        / DERIVATIVES_DIR_NAME
        / "FEA"
        / f"sub-{_clean_token(case.subject_id)}"
        / f"site-{_clean_token(case.site)}"
        / "runs"
        / case_id
        / "model_labels.nii.gz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seg_image = sitk.ReadImage(seg.path)
    trab_image = sitk.ReadImage(trab.path)
    cort_image = sitk.ReadImage(cort.path)
    _assert_matching_image_geometry(
        (("segmentation", seg_image), ("trabecular mask", trab_image), ("cortical mask", cort_image)),
        operation="generate XtremeCT material labelmap",
    )
    seg_array = sitk.GetArrayFromImage(seg_image) > 0
    trab_array = sitk.GetArrayFromImage(trab_image) > 0
    cort_array = sitk.GetArrayFromImage(cort_image) > 0
    material = np.zeros(seg_array.shape, dtype=np.uint8)
    material[seg_array & trab_array] = int(trab_label)
    material[seg_array & cort_array] = int(cort_label)
    output_image = sitk.GetImageFromArray(material)
    output_image.CopyInformation(seg_image)
    sitk.WriteImage(output_image, str(output_path))
    return FEAArtifact(
        role="generated_material_labelmap",
        path=str(output_path),
        source="generated",
        derivative="FEA",
        space="native",
    )


def _resolve_trab_cort_artifacts(case: FEABatchCase, *, generate: bool = False) -> tuple[FEAArtifact | None, FEAArtifact | None]:
    trab = case.first_artifact(("mask_trab",))
    cort = case.first_artifact(("mask_cort",))
    if trab is not None and cort is not None:
        return trab, cort
    full = case.first_artifact(("mask_full",))
    if full is None or not generate:
        return trab, cort
    generated = _derived_compartment_artifact(case, full=full, trab=trab, cort=cort)
    if trab is None and generated is not None and generated.role == "mask_trab":
        trab = generated
    if cort is None and generated is not None and generated.role == "mask_cort":
        cort = generated
    return trab, cort


def _derived_compartment_artifact(
    case: FEABatchCase,
    *,
    full: FEAArtifact,
    trab: FEAArtifact | None,
    cort: FEAArtifact | None,
) -> FEAArtifact | None:
    if trab is None and cort is not None:
        role = "mask_trab"
        output_name = _derived_compartment_filename(case, "trab")
        subtract = cort
    elif cort is None and trab is not None:
        role = "mask_cort"
        output_name = _derived_compartment_filename(case, "cort")
        subtract = trab
    else:
        return None
    output_path = Path(full.path).parent / output_name
    full_image = sitk.ReadImage(full.path)
    subtract_image = sitk.ReadImage(subtract.path)
    _assert_matching_image_geometry(
        (("full mask", full_image), (f"{subtract.role} mask", subtract_image)),
        operation="derive missing compartment mask",
    )
    full_array = sitk.GetArrayFromImage(full_image) > 0
    subtract_array = sitk.GetArrayFromImage(subtract_image) > 0
    derived = (full_array & ~subtract_array).astype(np.uint8)
    output_image = sitk.GetImageFromArray(derived)
    output_image.CopyInformation(full_image)
    sitk.WriteImage(output_image, str(output_path))
    return FEAArtifact(role=role, path=str(output_path), source="generated", derivative="FEA", space="native")


def _derived_compartment_filename(case: FEABatchCase, compartment: str) -> str:
    return (
        f"sub-{_clean_token(case.subject_id)}_"
        f"ses-{_clean_token(case.session_id)}_"
        f"site-{_clean_token(case.site)}_"
        f"derived_mask-{compartment}.nii.gz"
    )


def _assert_matching_image_geometry(images: Iterable[tuple[str, sitk.Image]], *, operation: str) -> None:
    items = tuple(images)
    if not items:
        return
    reference_name, reference = items[0]
    for name, image in items[1:]:
        if (
            tuple(reference.GetSize()) != tuple(image.GetSize())
            or not np.allclose(reference.GetSpacing(), image.GetSpacing())
            or not np.allclose(reference.GetOrigin(), image.GetOrigin())
            or not np.allclose(reference.GetDirection(), image.GetDirection())
        ):
            raise ValueError(
                f"Cannot {operation} because {reference_name} and {name} image geometry do not match."
            )
