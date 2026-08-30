from SlicerBoneImagingToolboxLib.motionscore_scene import (
    build_motionscore_scene_plan,
    motionscore_scene_predict_args,
    motionscore_scene_runner_args,
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
    assert plan.volume_npz_path.name == "sub-SAMPLE001_ses-1_site-tibia_scan-scan-1_volume.npz"


def test_motionscore_scene_runner_args_can_run_manual_only(tmp_path):
    plan = build_motionscore_scene_plan(
        results_root=tmp_path,
        scan_id="scan-1",
        subject_id="SAMPLE001",
        site="radius",
        session_id="ses-1",
        volume_node_id="node-1",
        run_id="scene-test",
    )

    args = motionscore_scene_runner_args(
        plan,
        model_root=tmp_path / "models",
        model_id="base-v1",
        manual_only=True,
        confidence_threshold=75,
        slice_step=1,
        device="auto",
    )

    assert args[0] == str(plan.volume_npz_path)
    assert "--manual-only" in args
    assert "--output-root" in args
    assert "--scan-id" in args


def test_motionscore_scene_predict_args_aliases_runner_args(tmp_path):
    plan = build_motionscore_scene_plan(
        results_root=tmp_path,
        scan_id="scan-1",
        subject_id="SAMPLE001",
        site="radius",
        session_id="ses-1",
        volume_node_id="node-1",
        run_id="scene-test",
    )

    common = {
        "model_root": tmp_path / "models",
        "model_id": "base-v1",
        "manual_only": False,
        "confidence_threshold": 75,
        "slice_step": 2,
        "device": "mps",
    }

    assert motionscore_scene_predict_args(plan, **common) == motionscore_scene_runner_args(plan, **common)
