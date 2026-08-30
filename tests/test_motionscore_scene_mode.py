from SlicerBoneImagingToolboxLib.motionscore_scene import (
    build_motionscore_scene_plan,
    motionscore_scene_predict_args,
)


def test_motionscore_scene_plan_uses_derivative_scene_folder(tmp_path):
    plan = build_motionscore_scene_plan(
        results_root=tmp_path,
        scan_id="scan-1",
        subject_id="SAMPLE001",
        site="tibia",
        session_id="ses-1",
        volume_node_id="node-1",
        run_id="scene-test",
    )

    assert plan.input_root.name == "input"
    assert "scene_runs" in str(plan.input_root)
    assert plan.image_path.name == "sub-SAMPLE001_ses-1_site-tibia_scan-scan-1_image.nii.gz"


def test_motionscore_scene_predict_args_can_run_manual_only(tmp_path):
    plan = build_motionscore_scene_plan(
        results_root=tmp_path,
        scan_id="scan-1",
        subject_id="SAMPLE001",
        site="radius",
        session_id="ses-1",
        volume_node_id="node-1",
        run_id="scene-test",
    )

    args = motionscore_scene_predict_args(
        plan,
        model_root=tmp_path / "models",
        model_id="base-v1",
        manual_only=True,
        confidence_threshold=75,
        slice_step=1,
        device="auto",
    )

    assert args[:2] == ["predict", str(plan.input_root)]
    assert "--manual-only" in args
    assert "--output-root" in args
