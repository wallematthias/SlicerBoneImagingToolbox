from pathlib import Path

import pytest

from SlicerBoneImagingToolboxLib.timelapsed_scene import (
    TimelapsedSceneTimepoint,
    build_timelapsed_scene_plan,
    timelapsed_scene_run_args,
)


def test_timelapsed_module_exposes_scene_and_batch_ui() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "self.timelapsedModeTabs" in source
    assert "Scene" in source
    assert "Batch" in source
    assert "def _on_run_scene_pipeline" in source
    assert "build_timelapsed_scene_plan" in source


def test_timelapsed_scene_plan_paths(tmp_path: Path) -> None:
    plan = build_timelapsed_scene_plan(
        results_root=tmp_path,
        subject_id="SAMPLE001",
        site="tibia",
        timepoints=[
            TimelapsedSceneTimepoint(session_id="ses-1", image_node_id="v1", full_mask_node_id="f1"),
            TimelapsedSceneTimepoint(session_id="ses-2", image_node_id="v2", full_mask_node_id="f2"),
        ],
        run_id="scene-test",
    )

    assert plan.input_root == tmp_path / "derivatives" / "Timelapsed" / "scene_runs" / "scene-test" / "input"
    assert plan.timepoints[0].image_path.name == "sub-SAMPLE001_ses-1_site-tibia_image.nii.gz"
    assert plan.timepoints[1].full_mask_path.name == "sub-SAMPLE001_ses-2_site-tibia_mask-full.nii.gz"


def test_timelapsed_scene_run_args_include_existing_pipeline_options(tmp_path: Path) -> None:
    plan = build_timelapsed_scene_plan(
        results_root=tmp_path,
        subject_id="SAMPLE001",
        site="radius",
        timepoints=[
            TimelapsedSceneTimepoint(session_id="ses-1", image_node_id="v1"),
            TimelapsedSceneTimepoint(session_id="ses-2", image_node_id="v2"),
        ],
        run_id="abc",
    )

    args = timelapsed_scene_run_args(plan, mode="regular", config_path=Path("/tmp/config.toml"))

    assert args[:2] == ["run", str(plan.input_root)]
    assert "--output-root" in args
    assert str(plan.output_root) in args
    assert "--config" in args


def test_timelapsed_scene_plan_requires_two_image_timepoints(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two timepoints with image node ids"):
        build_timelapsed_scene_plan(
            results_root=tmp_path,
            subject_id="SAMPLE001",
            site="tibia",
            timepoints=[
                TimelapsedSceneTimepoint(session_id="ses-1", image_node_id="v1"),
                TimelapsedSceneTimepoint(session_id="ses-2", image_node_id=""),
            ],
            run_id="scene-test",
        )
