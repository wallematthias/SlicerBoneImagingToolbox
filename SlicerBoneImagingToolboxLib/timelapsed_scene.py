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
    full_mask_node_id: str = ""
    trab_mask_node_id: str = ""
    cort_mask_node_id: str = ""
    seg_mask_node_id: str = ""
    transform_node_id: str = ""
    full_mask_policy: str = "generate"
    trab_mask_policy: str = "generate"
    cort_mask_policy: str = "generate"
    seg_mask_policy: str = "generate"
    image_path: Path | None = None
    full_mask_path: Path | None = None
    trab_mask_path: Path | None = None
    cort_mask_path: Path | None = None
    seg_mask_path: Path | None = None
    transform_path: Path | None = None


@dataclass(frozen=True)
class TimelapsedScenePlan:
    results_root: Path
    subject_id: str
    site: str
    run_id: str
    input_root: Path
    output_root: Path
    timepoints: tuple[TimelapsedSceneTimepoint, ...]


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
        role = _infer_scene_role(candidate)
        if role == "image":
            scalar_candidates.append(candidate)
        elif role and role != "transform":
            mask_count += 1
        session_id = _infer_scene_session(candidate)
        if not role or not session_id:
            continue
        subject = _infer_scene_token(candidate, "subject", "sub")
        site = _infer_scene_token(candidate, "site", "site")
        if subject:
            subjects.append(subject)
        if site:
            sites.append(site)
        group = groups.setdefault(session_id, {})
        if role not in group:
            group[role] = candidate.node_id

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
                full_mask_node_id=group.get("full", ""),
                trab_mask_node_id=group.get("trab", ""),
                cort_mask_node_id=group.get("cort", ""),
                seg_mask_node_id=group.get("seg", ""),
                transform_node_id=group.get("transform", ""),
                full_mask_policy="node" if group.get("full") else "generate",
                trab_mask_policy="node" if group.get("trab") else "generate",
                cort_mask_policy="node" if group.get("cort") else "generate",
                seg_mask_policy="node" if group.get("seg") else "generate",
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
    return TimelapsedScenePlan(
        results_root=root,
        subject_id=clean_subject,
        site=clean_site,
        run_id=clean_run_id,
        input_root=input_root,
        output_root=scene_root / "output",
        timepoints=planned_timepoints,
    )


def timelapsed_scene_run_args(
    plan: TimelapsedScenePlan,
    *,
    mode: str,
    config_path: str | Path,
    generate_missing_masks: bool = True,
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
    if node_id.strip() or normalized == "generate":
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


def _infer_scene_role(candidate: TimelapsedSceneNodeCandidate) -> str:
    attributes = candidate.attributes or {}
    role_text = " ".join(
        str(value)
        for key, value in attributes.items()
        if key.lower().endswith("role") or "role" in key.lower()
    ).lower()
    text = f"{candidate.name} {role_text}".lower()
    node_class = str(candidate.node_class or "")
    is_scalar = "ScalarVolume" in node_class
    is_mask_node = "LabelMapVolume" in node_class or "Segmentation" in node_class
    if "Transform" in node_class:
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
    text = str(candidate.name or "")
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
    pattern = rf"(?i)(?:^|[^A-Za-z0-9]){re.escape(prefix)}[-_]?([A-Za-z0-9.]+)"
    match = re.search(pattern, str(candidate.name or ""))
    if match:
        return _safe_token(match.group(1))
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
