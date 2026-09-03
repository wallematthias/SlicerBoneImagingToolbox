from pathlib import Path

from SlicerBoneImagingToolboxLib.motionscore_scene import (
    build_motionscore_scene_plan,
    motionscore_scene_predict_args,
    motionscore_scene_runner_args,
)


def test_motionscore_module_exposes_scene_and_batch_modes():
    source = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "MotionScoreHRpQCT"
        / "MotionScoreHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "self.motionScoreModeTabs" in source
    assert "Scene" in source
    assert "Batch" in source
    assert "def onRunScenePredict" in source
    assert "build_motionscore_scene_plan" in source
    assert 'self.motionScoreModeTabs.addTab(self.batchModePage, "Batch")' in source
    assert 'self.motionScoreModeTabs.addTab(self.sceneModePage, "Scene")' in source
    assert source.index('self.motionScoreModeTabs.addTab(self.batchModePage, "Batch")') < source.index(
        'self.motionScoreModeTabs.addTab(self.sceneModePage, "Scene")'
    )


def test_motionscore_scene_is_lightweight_prediction_only():
    source = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "MotionScoreHRpQCT"
        / "MotionScoreHRpQCT.py"
    ).read_text(encoding="utf-8")
    scene_setup = source.split("    def _setup_scene_mode(self):", 1)[1].split("\n    def ", 1)[0]
    scene_run = source.split("    def onRunScenePredict(self):", 1)[1].split("\n    def ", 1)[0]

    assert "sceneScanIdEdit" not in scene_setup
    assert "sceneSubjectIdEdit" not in scene_setup
    assert "sceneSiteEdit" not in scene_setup
    assert "sceneSessionIdEdit" not in scene_setup
    assert "sceneResultsRootEdit" not in scene_setup
    assert "sceneReviewerEdit" not in scene_setup
    assert "sceneRunModeCombo" not in scene_setup
    assert "AI Assisted" not in scene_setup
    assert "Reviewer" not in scene_setup
    assert 'self.sceneRunButton = qt.QPushButton("Run")' in scene_setup
    assert "self.sceneResultLabel" in scene_setup
    assert "self.sceneProfileLabel" in scene_setup
    assert "self._default_scene_results_root()" in scene_run
    assert "manual_only=False" in scene_run
    assert "self._on_scene_prediction_finished" in scene_run


def test_motionscore_scene_setup_does_not_capture_cleanup():
    source = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "MotionScoreHRpQCT"
        / "MotionScoreHRpQCT.py"
    ).read_text(encoding="utf-8")
    cleanup_start = source.index("    def cleanup(self):")
    cleanup_end = source.index("    def _setup_scene_mode(self):", cleanup_start)
    cleanup = source[cleanup_start:cleanup_end]

    assert "self._preload_executor.shutdown" in cleanup


def test_motionscore_scene_plan_uses_derivative_scene_folder(tmp_path):
    plan = build_motionscore_scene_plan(
        results_root=tmp_path,
        scan_id="scan-1",
        volume_node_id="node-1",
        run_id="scene-test",
    )

    assert plan.input_root.name == "input"
    assert "scene_runs" in str(plan.input_root)
    assert plan.volume_npz_path.name == "sub-scene_ses-scene_site-scene_scan-scan-1_volume.npz"


def test_motionscore_scene_runner_args_can_run_manual_only(tmp_path):
    plan = build_motionscore_scene_plan(
        results_root=tmp_path,
        scan_id="scan-1",
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


def test_motionscore_scene_subprocess_can_import_toolbox_runner():
    source = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "MotionScoreHRpQCT"
        / "MotionScoreHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert 'env.insert("PYTHONPATH"' in source
    assert "str(_TOOLBOX_ROOT)" in source


def test_motionscore_subprocess_prefers_local_core_checkout_when_present():
    source = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "MotionScoreHRpQCT"
        / "MotionScoreHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert 'MOTIONSCORE_LOCAL_SRC = _active_repositories_root(_TOOLBOX_ROOT) / "MotionScoreCNN"' in source
    assert "MOTIONSCORE_LOCAL_SRC.exists()" in source
    assert "pythonpath_parts.append(str(MOTIONSCORE_LOCAL_SRC))" in source
    assert "sys.path.insert(0" in source
    assert 'full_args = ["-c", command] + list(args)' in source
    assert f"from {{module_name}} import main" in source


def test_motionscore_scene_review_loader_accepts_npz_scene_inputs():
    source = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "MotionScoreHRpQCT"
        / "MotionScoreHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert 'raw.suffix.lower() == ".npz"' in source
    assert '"volume_xyz" not in data' in source
    assert 'data["volume_xyz"]' in source
    assert "volume_xyz.transpose(2, 1, 0).copy()" in source
