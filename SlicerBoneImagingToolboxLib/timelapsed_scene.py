"""Pure planning helpers for Timelapsed Slicer scene runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TimelapsedSceneTimepoint:
    session_id: str
    image_node_id: str
    full_mask_node_id: str = ""
    trab_mask_node_id: str = ""
    cort_mask_node_id: str = ""
    seg_mask_node_id: str = ""
    image_path: Path | None = None
    full_mask_path: Path | None = None
    trab_mask_path: Path | None = None
    cort_mask_path: Path | None = None
    seg_mask_path: Path | None = None


@dataclass(frozen=True)
class TimelapsedScenePlan:
    results_root: Path
    subject_id: str
    site: str
    run_id: str
    input_root: Path
    output_root: Path
    timepoints: tuple[TimelapsedSceneTimepoint, ...]


def build_timelapsed_scene_plan(
    results_root: str | Path,
    subject_id: str,
    site: str,
    timepoints: Iterable[TimelapsedSceneTimepoint],
    run_id: str,
) -> TimelapsedScenePlan:
    """Plan deterministic scene-run paths without interacting with Slicer."""
    clean_subject = _clean_token(subject_id, "sub")
    clean_site = _clean_token(site, "site")
    clean_run_id = _safe_token(run_id)
    if not clean_subject or not clean_site:
        raise ValueError("subject_id and site are required")

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
) -> list[str]:
    """Return the existing Timelapsed CLI arguments for a scene plan."""
    return [
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
        full_mask_path=_optional_path(directory, stem, "mask-full", timepoint.full_mask_node_id),
        trab_mask_path=_optional_path(directory, stem, "mask-trab", timepoint.trab_mask_node_id),
        cort_mask_path=_optional_path(directory, stem, "mask-cort", timepoint.cort_mask_node_id),
        seg_mask_path=_optional_path(directory, stem, "mask-seg", timepoint.seg_mask_node_id),
    )


def _optional_path(directory: Path, stem: str, suffix: str, node_id: str) -> Path | None:
    return directory / f"{stem}_{suffix}.nii.gz" if node_id.strip() else None


def _clean_token(value: str, prefix: str) -> str:
    text = str(value or "").strip()
    if text.lower().startswith(f"{prefix}-"):
        text = text[len(prefix) + 1 :]
    return _safe_token(text)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_.-]+", "-", str(value).strip())
    return token.strip("-")
