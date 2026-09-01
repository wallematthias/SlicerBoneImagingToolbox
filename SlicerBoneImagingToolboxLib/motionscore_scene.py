"""Pure planning helpers for MotionScore Slicer scene runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionScoreScenePlan:
    results_root: Path
    run_root: Path
    input_root: Path
    output_root: Path
    volume_npz_path: Path
    scan_id: str
    subject_id: str
    site: str
    session_id: str
    volume_node_id: str


def build_motionscore_scene_plan(
    *,
    results_root: Path,
    scan_id: str,
    subject_id: str = "scene",
    site: str = "scene",
    session_id: str = "scene",
    volume_node_id: str,
    run_id: str,
) -> MotionScoreScenePlan:
    clean_scan = _safe_token(scan_id)
    clean_subject = _clean_token(subject_id, "sub")
    clean_site = _clean_token(site, "site")
    clean_session = _clean_token(session_id, "ses")
    clean_run = _safe_token(run_id)
    if not clean_scan or not clean_subject or not clean_site or not clean_session:
        raise ValueError("scan_id, subject_id, site, and session_id are required")
    if not str(volume_node_id or "").strip():
        raise ValueError("volume_node_id is required")

    results_root = Path(results_root)
    run_root = results_root / "derivatives" / "MotionScore" / "scene_runs" / clean_run
    input_root = run_root / "input"
    volume_npz_path = input_root / (
        f"sub-{clean_subject}_ses-{clean_session}_site-{clean_site}_scan-{clean_scan}_volume.npz"
    )

    return MotionScoreScenePlan(
        results_root=results_root,
        run_root=run_root,
        input_root=input_root,
        output_root=run_root,
        volume_npz_path=volume_npz_path,
        scan_id=clean_scan,
        subject_id=clean_subject,
        site=clean_site,
        session_id=clean_session,
        volume_node_id=volume_node_id,
    )


def motionscore_scene_runner_args(
    plan: MotionScoreScenePlan,
    *,
    model_root: Path,
    model_id: str,
    manual_only: bool,
    confidence_threshold: int,
    slice_step: int,
    device: str,
) -> list[str]:
    args = [
        str(plan.volume_npz_path),
        "--confidence-threshold",
        str(confidence_threshold),
        "--output-root",
        str(plan.output_root),
    ]
    if manual_only:
        args.append("--manual-only")
    else:
        args.extend(
            [
                "--model-root",
                str(model_root),
                "--model-id",
                model_id,
                "--slice-step",
                str(slice_step),
            ]
        )
        if device.lower() != "auto":
            args.extend(["--device", device])
    args.extend(["--scan-id", plan.scan_id])
    args.extend(["--subject-id", plan.subject_id])
    args.extend(["--site", plan.site])
    args.extend(["--session-id", plan.session_id])
    return args


def motionscore_scene_predict_args(
    plan: MotionScoreScenePlan,
    *,
    model_root: Path,
    model_id: str,
    manual_only: bool,
    confidence_threshold: int,
    slice_step: int,
    device: str,
) -> list[str]:
    """Backward-compatible alias for scene-runner arguments."""
    return motionscore_scene_runner_args(
        plan,
        model_root=model_root,
        model_id=model_id,
        manual_only=manual_only,
        confidence_threshold=confidence_threshold,
        slice_step=slice_step,
        device=device,
    )


def _clean_token(value: str, prefix: str) -> str:
    text = str(value or "").strip()
    if text.lower().startswith(f"{prefix}-"):
        text = text[len(prefix) + 1 :]
    return _safe_token(text)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_.-]+", "-", str(value or "").strip())
    return token.strip("-")
