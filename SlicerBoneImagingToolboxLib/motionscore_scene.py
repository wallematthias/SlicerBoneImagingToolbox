"""Pure planning helpers for MotionScore Slicer scene runs."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionScoreScenePlan:
    results_root: Path
    run_root: Path
    input_root: Path
    output_root: Path
    image_path: Path
    scan_id: str
    subject_id: str
    site: str
    session_id: str
    volume_node_id: str


def build_motionscore_scene_plan(
    *,
    results_root: Path,
    scan_id: str,
    subject_id: str,
    site: str,
    session_id: str,
    volume_node_id: str,
    run_id: str,
) -> MotionScoreScenePlan:
    results_root = Path(results_root)
    run_root = results_root / "derivatives" / "MotionScore" / "scene_runs" / run_id
    input_root = run_root / "input"
    image_path = input_root / (
        f"sub-{subject_id}_{session_id}_site-{site}_scan-{scan_id}_image.nii.gz"
    )

    return MotionScoreScenePlan(
        results_root=results_root,
        run_root=run_root,
        input_root=input_root,
        output_root=run_root,
        image_path=image_path,
        scan_id=scan_id,
        subject_id=subject_id,
        site=site,
        session_id=session_id,
        volume_node_id=volume_node_id,
    )


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
    args = [
        "predict",
        str(plan.input_root),
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
    return args
