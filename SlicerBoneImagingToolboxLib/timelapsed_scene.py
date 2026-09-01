"""Pure planning helpers for Timelapsed Slicer scene runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TimelapsedSceneNodeCandidate:
    node_id: str
    name: str
    node_class: str
    attributes: dict[str, str] | None = None


@dataclass(frozen=True)
class TimelapsedSceneTimepoint:
    session_id: str
    image_node_id: str
    reg_mask_node_id: str = ""
    full_mask_node_id: str = ""
    trab_mask_node_id: str = ""
    cort_mask_node_id: str = ""
    seg_mask_node_id: str = ""
    reg_mask_segment_id: str = ""
    full_mask_segment_id: str = ""
    trab_mask_segment_id: str = ""
    cort_mask_segment_id: str = ""
    seg_mask_segment_id: str = ""
    transform_node_id: str = ""
    reg_mask_policy: str = "none"
    full_mask_policy: str = "none"
    trab_mask_policy: str = "none"
    cort_mask_policy: str = "none"
    seg_mask_policy: str = "none"
    image_path: Path | None = None
    reg_mask_path: Path | None = None
    full_mask_path: Path | None = None
    trab_mask_path: Path | None = None
    cort_mask_path: Path | None = None
    seg_mask_path: Path | None = None
    transform_path: Path | None = None


@dataclass(frozen=True)
class TimelapsedSceneRoiSelection:
    role: str
    node_ids: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()
    policies: tuple[str, ...] = ()
    paths: tuple[Path | None, ...] = ()


@dataclass(frozen=True)
class TimelapsedScenePlan:
    results_root: Path
    subject_id: str
    site: str
    run_id: str
    input_root: Path
    output_root: Path
    timepoints: tuple[TimelapsedSceneTimepoint, ...]
    rois: tuple[TimelapsedSceneRoiSelection, ...] = ()


@dataclass(frozen=True)
class TimelapsedSceneDiscovery:
    subject_id: str
    site: str
    timepoints: tuple[TimelapsedSceneTimepoint, ...]
    image_count: int = 0
    mask_count: int = 0
    matched_mask_count: int = 0


def discover_timelapsed_scene_timepoints(
    candidates: Iterable[TimelapsedSceneNodeCandidate],
) -> TimelapsedSceneDiscovery:
    """Group loaded Slicer node candidates into Timelapsed scene timepoints."""
    groups: dict[str, dict[str, str]] = {}
    subjects: list[str] = []
    sites: list[str] = []
    scalar_candidates: list[TimelapsedSceneNodeCandidate] = []
    mask_count = 0
    for candidate in candidates:
        if _is_generated_timelapsed_display_artifact(candidate):
            continue
        roles = _infer_scene_roles(candidate)
        role = roles[0] if roles else ""
        if role == "image":
            scalar_candidates.append(candidate)
        elif role and role != "transform":
            mask_count += 1
        session_id = _infer_scene_session(candidate)
        if not roles or not session_id:
            continue
        subject = _infer_scene_token(candidate, "subject", "sub")
        site = _infer_scene_token(candidate, "site", "site")
        if subject:
            subjects.append(subject)
        if site:
            sites.append(site)
        group = groups.setdefault(session_id, {})
        for candidate_role in roles:
            if candidate_role not in group:
                group[candidate_role] = candidate.node_id

    timepoints: list[TimelapsedSceneTimepoint] = []
    matched_mask_count = 0
    for session_id in sorted(groups, key=_scene_session_sort_key):
        group = groups[session_id]
        image_node_id = group.get("image", "")
        if not image_node_id:
            continue
        matched_mask_count += sum(1 for role in ("full", "trab", "cort", "seg") if group.get(role))
        timepoints.append(
            TimelapsedSceneTimepoint(
                session_id=session_id,
                image_node_id=image_node_id,
                reg_mask_node_id=group.get("regmask", "") or group.get("full", ""),
                full_mask_node_id=group.get("full", ""),
                trab_mask_node_id=group.get("trab", ""),
                cort_mask_node_id=group.get("cort", ""),
                seg_mask_node_id=group.get("seg", ""),
                transform_node_id=group.get("transform", ""),
                reg_mask_policy="node" if (group.get("regmask") or group.get("full")) else "none",
                full_mask_policy="node" if group.get("full") else "none",
                trab_mask_policy="node" if group.get("trab") else "none",
                cort_mask_policy="node" if group.get("cort") else "none",
                seg_mask_policy="node" if group.get("seg") else "none",
            )
        )
    if not timepoints and scalar_candidates:
        timepoints = [
            TimelapsedSceneTimepoint(session_id=str(index), image_node_id=candidate.node_id)
            for index, candidate in enumerate(scalar_candidates, start=1)
        ]
    return TimelapsedSceneDiscovery(
        subject_id=_most_common(subjects),
        site=_most_common(sites),
        timepoints=tuple(timepoints),
        image_count=len(scalar_candidates),
        mask_count=mask_count,
        matched_mask_count=matched_mask_count,
    )


def build_timelapsed_scene_plan(
    results_root: str | Path,
    subject_id: str,
    site: str,
    timepoints: Iterable[TimelapsedSceneTimepoint],
    run_id: str,
    rois: Iterable[TimelapsedSceneRoiSelection] | None = None,
) -> TimelapsedScenePlan:
    """Plan deterministic scene-run paths without interacting with Slicer."""
    clean_subject = _clean_token(subject_id, "sub") or "SceneSubject"
    clean_site = _clean_token(site, "site") or "scene"
    clean_run_id = _safe_token(run_id)

    selected_timepoints = tuple(timepoints)
    if len(selected_timepoints) < 2 or any(not timepoint.image_node_id.strip() for timepoint in selected_timepoints):
        raise ValueError("Timelapsed scene runs require at least two timepoints with image node ids")
    session_ids = [_clean_token(timepoint.session_id, "ses") for timepoint in selected_timepoints]
    duplicate_session_ids = sorted({session_id for session_id in session_ids if session_ids.count(session_id) > 1})
    if duplicate_session_ids:
        raise ValueError(f"Duplicate Timelapsed scene session_id: {duplicate_session_ids[0]}")

    root = Path(results_root)
    scene_root = root / "derivatives" / "Timelapsed" / "scene_runs" / clean_run_id
    input_root = scene_root / "input"
    planned_timepoints = tuple(
        _plan_timepoint(input_root, clean_subject, clean_site, timepoint)
        for timepoint in selected_timepoints
    )
    selected_rois = tuple(rois) if rois is not None else _legacy_rois_from_timepoints(selected_timepoints)
    planned_rois = tuple(
        _plan_roi(input_root, clean_subject, clean_site, planned_timepoints, roi)
        for roi in selected_rois
    )
    return TimelapsedScenePlan(
        results_root=root,
        subject_id=clean_subject,
        site=clean_site,
        run_id=clean_run_id,
        input_root=input_root,
        output_root=scene_root / "output",
        timepoints=planned_timepoints,
        rois=planned_rois,
    )


def timelapsed_scene_run_args(
    plan: TimelapsedScenePlan,
    *,
    mode: str,
    config_path: str | Path,
    generate_missing_masks: bool = False,
) -> list[str]:
    """Return the existing Timelapsed CLI arguments for a scene plan."""
    args = [
        "run",
        str(plan.input_root),
        "--output-root",
        str(plan.output_root),
        "--mode",
        mode,
        "--allow-scene-images",
        "--config",
        str(config_path),
    ]
    if not generate_missing_masks:
        args.append("--skip-mask-generation")
    return args


def scene_segment_matches_role(segment_name: str, segment_role: str, requested_role: str) -> bool:
    """Return whether a readable Slicer segment name/tag matches a Timelapsed role."""
    requested = _normalize_scene_role_token(requested_role)
    if not requested:
        return False
    aliases = _scene_role_aliases(requested)
    tokens = {
        _normalize_scene_role_token(segment_name),
        _normalize_scene_role_token(segment_role),
    }
    tokens = {token for token in tokens if token}
    return bool(tokens & aliases)


def _scene_role_aliases(role: str) -> set[str]:
    role = _normalize_scene_role_token(role)
    aliases = {role, f"mask_{role}", f"{role}_mask"}
    if role in {"full", "full_roi"}:
        aliases.update({"full", "full_mask", "periosteal", "periosteal_contour", "full_roi", "support", "support_roi"})
    elif role in {"trab", "trab_roi"}:
        aliases.update({"trab", "trab_mask", "trabecular", "trabecular_mask", "trabecular_roi", "trab_roi"})
    elif role in {"cort", "cort_roi"}:
        aliases.update({"cort", "cort_mask", "cortical", "cortical_mask", "cortical_roi", "cort_roi"})
    elif role in {"seg", "segmentation"}:
        aliases.update({"bone", "bone_seg", "bone_segmentation", "segmentation", "bone_mask"})
    elif role in {"regmask", "registration_roi"}:
        aliases.update({"reg", "registration_mask", "registration_roi", "reg_mask", "full", "full_mask", "support_roi"})
    return aliases


def _normalize_scene_role_token(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())).strip("_")


def _plan_timepoint(
    input_root: Path,
    subject_id: str,
    site: str,
    timepoint: TimelapsedSceneTimepoint,
) -> TimelapsedSceneTimepoint:
    session_id = _clean_token(timepoint.session_id, "ses")
    if not session_id:
        raise ValueError("Each Timelapsed scene timepoint requires a session_id")
    directory = input_root / f"sub-{subject_id}" / f"site-{site}" / "native_space" / f"ses-{session_id}"
    stem = f"sub-{subject_id}_ses-{session_id}_site-{site}"
    return replace(
        timepoint,
        session_id=session_id,
        image_path=directory / f"{stem}_image.nii.gz",
        reg_mask_path=_mask_path_for_policy(
            directory, stem, "regmask", timepoint.reg_mask_node_id, timepoint.reg_mask_policy
        ),
        full_mask_path=_mask_path_for_policy(
            directory, stem, "mask-full", timepoint.full_mask_node_id, timepoint.full_mask_policy
        ),
        trab_mask_path=_mask_path_for_policy(
            directory, stem, "mask-trab", timepoint.trab_mask_node_id, timepoint.trab_mask_policy
        ),
        cort_mask_path=_mask_path_for_policy(
            directory, stem, "mask-cort", timepoint.cort_mask_node_id, timepoint.cort_mask_policy
        ),
        seg_mask_path=_mask_path_for_policy(
            directory, stem, "mask-seg", timepoint.seg_mask_node_id, timepoint.seg_mask_policy
        ),
        transform_path=_optional_path(directory, stem, "transform", timepoint.transform_node_id, suffix_ext=".tfm"),
    )


def _legacy_rois_from_timepoints(
    timepoints: tuple[TimelapsedSceneTimepoint, ...],
) -> tuple[TimelapsedSceneRoiSelection, ...]:
    rois: list[TimelapsedSceneRoiSelection] = []
    for role, node_attr, segment_attr, policy_attr in (
        ("full", "full_mask_node_id", "full_mask_segment_id", "full_mask_policy"),
        ("trab", "trab_mask_node_id", "trab_mask_segment_id", "trab_mask_policy"),
        ("cort", "cort_mask_node_id", "cort_mask_segment_id", "cort_mask_policy"),
    ):
        node_ids = tuple(str(getattr(timepoint, node_attr, "") or "") for timepoint in timepoints)
        segment_ids = tuple(str(getattr(timepoint, segment_attr, "") or "") for timepoint in timepoints)
        policies = tuple(str(getattr(timepoint, policy_attr, "none") or "none") for timepoint in timepoints)
        if any(node_ids):
            rois.append(
                TimelapsedSceneRoiSelection(
                    role=role,
                    node_ids=node_ids,
                    segment_ids=segment_ids,
                    policies=policies,
                )
            )
    return tuple(rois)


def _plan_roi(
    input_root: Path,
    subject_id: str,
    site: str,
    timepoints: tuple[TimelapsedSceneTimepoint, ...],
    roi: TimelapsedSceneRoiSelection,
) -> TimelapsedSceneRoiSelection:
    role = _clean_roi_role(roi.role)
    node_ids = _pad_tuple(roi.node_ids, len(timepoints), "")
    segment_ids = _pad_tuple(roi.segment_ids, len(timepoints), "")
    policies = _pad_tuple(roi.policies, len(timepoints), "none")
    paths: list[Path | None] = []
    for index, timepoint in enumerate(timepoints):
        directory = timepoint.image_path.parent if timepoint.image_path is not None else (
            input_root / f"sub-{subject_id}" / f"site-{site}" / "native_space" / f"ses-{timepoint.session_id}"
        )
        stem = f"sub-{subject_id}_ses-{timepoint.session_id}_site-{site}"
        paths.append(_mask_path_for_policy(directory, stem, f"mask-{role}", node_ids[index], policies[index]))
    return replace(
        roi,
        role=role,
        node_ids=node_ids,
        segment_ids=segment_ids,
        policies=policies,
        paths=tuple(paths),
    )


def _pad_tuple(values: tuple[str, ...], length: int, fill: str) -> tuple[str, ...]:
    padded = [str(value or fill) for value in values[:length]]
    padded.extend([fill] * (length - len(padded)))
    return tuple(padded)


def _clean_roi_role(role: str) -> str:
    value = _safe_token(role).lower()
    if value in {"", "roi"}:
        return "roi1"
    if value in {"full", "trab", "cort", "full_roi", "trab_roi", "cort_roi", "regmask"} or value.startswith("roi"):
        return value
    return f"roi_{value}"


def _optional_path(directory: Path, stem: str, suffix: str, node_id: str, *, suffix_ext: str = ".nii.gz") -> Path | None:
    return directory / f"{stem}_{suffix}{suffix_ext}" if node_id.strip() else None


def _mask_path_for_policy(
    directory: Path,
    stem: str,
    suffix: str,
    node_id: str,
    policy: str,
) -> Path | None:
    normalized = str(policy or "").strip().lower()
    if node_id.strip():
        return directory / f"{stem}_{suffix}.nii.gz"
    return None


def _clean_token(value: str, prefix: str) -> str:
    text = str(value or "").strip()
    if text.lower().startswith(f"{prefix}-"):
        text = text[len(prefix) + 1 :]
    return _safe_token(text)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_.-]+", "-", str(value).strip())
    return token.strip("-")


def _is_generated_timelapsed_display_artifact(candidate: TimelapsedSceneNodeCandidate) -> bool:
    attributes = candidate.attributes or {}
    is_transform_node = "Transform" in str(candidate.node_class or "")
    if is_transform_node:
        return False
    if attributes.get("BoneImaging.MaskRoles") or attributes.get("BoneContouring.Role"):
        return False
    if attributes.get("TimelapsedHRpQCT.GeneratedMask") == "1":
        return True
    storage_path = str(attributes.get("StorageFileName", "") or "").replace("\\", "/")
    if "/TimelapsedScene/derivatives/Timelapsed/scene_runs/" in storage_path and "/output/" in storage_path:
        return True
    if attributes.get("TimelapsedHRpQCT.RemodellingFull") == "1":
        return True
    if attributes.get("TimelapsedHRpQCT.SliceReference") == "1":
        return True
    name = str(candidate.name or "").lower()
    return "remodelling" in name and (
        name.endswith("_full")
        or "slice_reference" in name
        or "segmentation" in name
    )


def _infer_scene_roles(candidate: TimelapsedSceneNodeCandidate) -> list[str]:
    attributes = candidate.attributes or {}
    role_text = str(
        attributes.get("BoneImaging.MaskRoles")
        or attributes.get("BoneContouring.Role")
        or ""
    )
    if role_text:
        roles = []
        for token in re.split(r"[,;\s]+", role_text):
            role = _normalize_scene_role(token)
            if role and role not in roles:
                roles.append(role)
        if roles:
            return roles
    role = _infer_scene_role(candidate)
    return [role] if role else []


def _normalize_scene_role(role: str) -> str:
    role_lower = str(role or "").strip().lower().replace("-", "_")
    if role_lower in {"full", "full_mask", "mask_full", "periosteal"}:
        return "full"
    if role_lower in {"trab", "trabecular", "trab_mask", "mask_trab"}:
        return "trab"
    if role_lower in {"cort", "cortical", "cort_mask", "mask_cort"}:
        return "cort"
    if role_lower in {"seg", "bone_seg", "bone_segmentation", "segmentation", "mask_seg"}:
        return "seg"
    if role_lower in {"reg", "regmask", "registration_mask", "mask_reg"}:
        return "regmask"
    if re.fullmatch(r"roi[0-9a-z]*", role_lower):
        return role_lower
    return ""


def _infer_scene_role(candidate: TimelapsedSceneNodeCandidate) -> str:
    attributes = candidate.attributes or {}
    role_text = " ".join(
        str(value)
        for key, value in attributes.items()
        if key.lower().endswith("role") or "role" in key.lower()
    ).lower()
    text = f"{_candidate_search_text(candidate)} {role_text}".lower()
    node_class = str(candidate.node_class or "")
    is_scalar = "ScalarVolume" in node_class
    is_mask_node = "LabelMapVolume" in node_class or "Segmentation" in node_class
    if "Transform" in node_class:
        storage_path = str((candidate.attributes or {}).get("StorageFileName", "") or "")
        if storage_path and Path(storage_path).suffix.lower() not in {"", ".tfm"}:
            return ""
        return "transform"
    if is_mask_node:
        if any(token in text for token in ("mask-full", "full-mask", "periosteal", "peri", "mask_full")):
            return "full"
        if any(token in text for token in ("mask-trab", "trabecular", "trab", "mask_trab")):
            return "trab"
        if any(token in text for token in ("mask-cort", "cortical", "cort", "mask_cort")):
            return "cort"
        if any(token in text for token in ("mask-seg", "bone-seg", "bone segmentation", "segmentation", "mask_seg")):
            return "seg"
    if is_scalar and not any(token in text for token in ("mask", "segmentation", "label")):
        return "image"
    return ""


def _infer_scene_session(candidate: TimelapsedSceneNodeCandidate) -> str:
    attributes = candidate.attributes or {}
    for key in ("session", "session_id", "HRpQCT.Session", "BoneImaging.Session"):
        value = attributes.get(key)
        if value:
            return _clean_token(value, "ses")
    text = _candidate_search_text(candidate)
    patterns = (
        r"(?i)(?:^|[^A-Za-z0-9])ses[-_]?([A-Za-z0-9.]+)",
        r"(?i)(?:^|[^A-Za-z0-9])session[-_]?([A-Za-z0-9.]+)",
        r"(?i)(?:^|[^A-Za-z0-9])Y([0-9]+)(?:[^A-Za-z0-9]|$)",
        r"(?i)(?:^|[^A-Za-z0-9])T([0-9]+)(?:[^A-Za-z0-9]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _safe_token(match.group(1))
    lowered = text.lower()
    if "baseline" in lowered:
        return "1"
    if "followup" in lowered or "follow-up" in lowered:
        return "2"
    return ""


def _infer_scene_token(candidate: TimelapsedSceneNodeCandidate, attribute_name: str, prefix: str) -> str:
    attributes = candidate.attributes or {}
    for key in (attribute_name, f"{attribute_name}_id", f"HRpQCT.{attribute_name.title()}"):
        value = attributes.get(key)
        if value:
            return _clean_token(value, prefix)
    text = _candidate_search_text(candidate)
    if attribute_name == "subject":
        subject_match = re.search(r"(?i)(?:^|[^A-Za-z0-9])sub[-_]?(.+?)_ses[-_]?", text)
        if subject_match:
            return _safe_token(subject_match.group(1))
    if attribute_name == "site":
        site_match = re.search(
            r"(?i)(?:^|[^A-Za-z0-9])site[-_]?(.+?)(?:_(?:image|mask[-_](?:full|trab|cort|seg)|seg|transform)|$)",
            text,
        )
        if site_match:
            return _safe_token(site_match.group(1))
    pattern = rf"(?i)(?:^|[^A-Za-z0-9]){re.escape(prefix)}[-_]?([A-Za-z0-9_.-]+)"
    match = re.search(pattern, text)
    if match:
        value = match.group(1)
        value = re.sub(r"(?i)[_-](?:image|mask[-_](?:full|trab|cort|seg)|seg|transform)$", "", value)
        return _safe_token(value)
    study_match = re.search(
        r"(?i)^([A-Za-z][A-Za-z0-9]+)_([0-9]+)_([A-Za-z]+)_Y[0-9]+(?:[^A-Za-z0-9]|$)",
        str(candidate.name or ""),
    )
    if not study_match:
        return ""
    if attribute_name == "subject":
        return _safe_token(f"{study_match.group(1)}_{study_match.group(2)}")
    if attribute_name == "site":
        return _safe_token(study_match.group(3))
    return ""


def _candidate_search_text(candidate: TimelapsedSceneNodeCandidate) -> str:
    attributes = candidate.attributes or {}
    path_text = " ".join(
        str(attributes.get(key, "") or "")
        for key in ("StorageFileName", "FileName", "filename", "path")
    )
    return f"{candidate.name or ''} {Path(path_text).name if path_text else ''} {path_text}"


def _scene_session_sort_key(session_id: str) -> tuple[int, int | str]:
    text = str(session_id)
    return (0, int(text)) if text.isdigit() else (1, text.lower())


def _most_common(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
