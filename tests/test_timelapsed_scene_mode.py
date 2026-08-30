from pathlib import Path

import pytest

from SlicerBoneImagingToolboxLib.timelapsed_scene import (
    TimelapsedSceneNodeCandidate,
    TimelapsedSceneTimepoint,
    build_timelapsed_scene_plan,
    discover_timelapsed_scene_timepoints,
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
    assert "Discover Loaded Timepoints" in source
    assert "Append to table" in source
    assert "Initial transform" in source
    assert "Scene Results" in source
    assert "self.sceneResultsTable" in source
    assert "_load_scene_results_table" in source
    assert "_scene_processed_subject_site" in source
    assert "candidate.parent.name == \"derivatives\"" in source
    assert "return candidate.parent.parent" in source
    assert "_seed_scene_transform_registry" in source
    assert "upsert_transform_registry_record" in source
    assert "TransformRegistryRecord" in source
    assert "using external transform registry" not in source
    assert "FV/BV" in source
    assert "RV/BV" in source
    assert "AV/BV" in source
    assert "NV/BV" in source
    assert "pairwise_remodelling_csv_path" in source
    assert "vtkMRMLTransformNode" in source
    assert "transform_node_id" in source
    assert "_load_scene_run_outputs" in source
    assert "_load_scene_run_masks" in source
    assert "_load_scene_mask_labelmap" in source
    assert "iter_imported_stack_records" in source
    assert "iter_fused_session_records" in source
    assert "_adopt_scene_run_as_current_dataset" in source
    assert "_set_path_without_immediate_reset" in source
    assert "current dataset set to scene run" in source
    assert "remodelling image (full)" in source
    assert "_last_scene_plan" in source
    assert "loadTransform" in source
    assert "mask-full" in source
    assert "mask-trab" in source
    assert "mask-cort" in source
    assert "mask-seg" in source
    assert "self.sceneProfileCombo" in source
    assert "self.sceneMaskPolicyCombo" not in source
    assert "Missing masks" not in source
    assert "def _scene_mask_selector" in source
    assert "def _scene_selected_mask_node_id" in source
    assert "def _scene_mask_generation_requested" in source
    assert 'addItem("Generate", "__generate__")' in source
    assert 'addItem("None", "__none__")' in source
    assert "self._scene_mask_generation_requested()" in source
    assert "identifiersBox" not in source
    assert "sceneSubjectEdit" not in source
    assert "sceneSiteEdit" not in source
    assert "sceneAnalysisThresholdSlider" not in source
    assert "sceneMaskLowerSlider" not in source
    assert "Analysis Options" in source
    assert "Advanced Settings" in source
    assert "self.layout.addWidget(analysisSectionBox)" in source
    assert "self.layout.addWidget(settingsBox)" in source
    assert "self.layout.addWidget(statusBox)" in source
    assert "self.layout.addWidget(self.logText)" in source
    assert 'self.sceneStatusLabel.text = "Preparing scene run..."' in source
    assert "except Exception as exc" in source
    assert "self._resize_scene_timepoint_table()" in source
    assert "ScrollBarAsNeeded" in source
    assert "setMaximumHeight(min(430" in source
    assert "layout.setContentsMargins(0, 0, 0, 0)" in source
    assert 'env.insert("PYTHONPATH", os.environ["PYTHONPATH"])' in source
    assert "_resolve_local_pipeline_paths" in source
    assert 'base / "TimelapsedHRpQCT"' in source
    assert "timelapsedhrpqct.cli import main" in source
    assert 'MIN_PIPELINE_VERSION = "2.0.39"' in source
    assert "Move up" in source
    assert "Move down" in source
    assert "discover_timelapsed_scene_timepoints" in source
    assert "self._scene_generate_missing_masks()" in source
    assert 'importlib.import_module("SlicerBoneImagingToolboxLib.timelapsed_scene")' in source
    assert "importlib.reload(_timelapsed_scene)" in source


def test_timelapsed_scene_loads_pairwise_results_table() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert 'setHorizontalHeaderLabels(["Pair", "Mask", "FV/BV", "RV/BV", "AV/BV", "NV/BV"])' in source
    assert "formation_frac_bv0" in source
    assert "resorption_frac_bv0" in source
    assert "AV_BV" in source
    assert "NV_BV" in source
    assert "Loaded scene results table" in source
    assert "_scene_processed_subject_site(plan)" in source
    assert "_seed_scene_transform_registry(plan)" in source
    assert "_select_first_scene_remodelling_output" in source
    assert "_refresh_pair_metrics_for_current_selection()" in source


def test_timelapsed_scene_mask_policy_is_per_table_cell() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "self._scene_mask_selector()" in source
    assert 'selector.addItem("Generate", "__generate__")' in source
    assert 'selector.addItem("None", "__none__")' in source
    assert 'value in {"__generate__", "__none__"}' in source
    assert 'str(selector.currentData or "") == "__generate__"' in source
    assert "sceneMaskPolicyCombo" not in source


def test_timelapsed_scene_plan_paths(tmp_path: Path) -> None:
    plan = build_timelapsed_scene_plan(
        results_root=tmp_path,
        subject_id="SAMPLE001",
        site="tibia",
        timepoints=[
            TimelapsedSceneTimepoint(session_id="ses-1", image_node_id="v1", full_mask_node_id="f1"),
            TimelapsedSceneTimepoint(
                session_id="ses-2",
                image_node_id="v2",
                full_mask_node_id="f2",
                transform_node_id="t2",
            ),
        ],
        run_id="scene-test",
    )

    assert plan.input_root == tmp_path / "derivatives" / "Timelapsed" / "scene_runs" / "scene-test" / "input"
    assert plan.timepoints[0].image_path.name == "sub-SAMPLE001_ses-1_site-tibia_image.nii.gz"
    assert plan.timepoints[1].full_mask_path.name == "sub-SAMPLE001_ses-2_site-tibia_mask-full.nii.gz"
    assert plan.timepoints[1].transform_path.name == "sub-SAMPLE001_ses-2_site-tibia_transform.tfm"


def test_timelapsed_scene_plan_defaults_identifiers_for_loaded_scene_runs(tmp_path: Path) -> None:
    plan = build_timelapsed_scene_plan(
        results_root=tmp_path,
        subject_id="",
        site="",
        timepoints=[
            TimelapsedSceneTimepoint(session_id="1", image_node_id="v1"),
            TimelapsedSceneTimepoint(session_id="2", image_node_id="v2"),
        ],
        run_id="abc",
    )

    assert plan.subject_id == "SceneSubject"
    assert plan.site == "scene"
    assert plan.timepoints[0].image_path.name == "sub-SceneSubject_ses-1_site-scene_image.nii.gz"


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
    assert "--allow-scene-images" in args
    assert "--config" in args


def test_timelapsed_scene_run_args_can_skip_mask_generation(tmp_path: Path) -> None:
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

    args = timelapsed_scene_run_args(
        plan,
        mode="regular",
        config_path=Path("/tmp/config.toml"),
        generate_missing_masks=False,
    )

    assert "--skip-mask-generation" in args


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


def test_timelapsed_scene_plan_rejects_duplicate_session_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate Timelapsed scene session_id"):
        build_timelapsed_scene_plan(
            results_root=tmp_path,
            subject_id="SAMPLE001",
            site="tibia",
            timepoints=[
                TimelapsedSceneTimepoint(session_id="ses-1", image_node_id="v1"),
                TimelapsedSceneTimepoint(session_id="1", image_node_id="v2"),
            ],
            run_id="scene-test",
        )


def test_timelapsed_scene_export_converts_segmentation_nodes_against_reference_geometry() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert 'node.IsA("vtkMRMLSegmentationNode")' in source
    assert "ExportAllSegmentsToLabelmapNode" in source
    assert "EXTENT_REFERENCE_GEOMETRY" in source
    assert "reference_node_id=timepoint.image_node_id" in source


def test_timelapsed_scene_discovery_groups_loaded_images_and_optional_masks() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate("img1", "sub-SAMPLE001_ses-1_site-tibia_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("img2", "sub-SAMPLE001_ses-2_site-tibia_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("full1", "sub-SAMPLE001_ses-1_site-tibia_mask-full", "vtkMRMLLabelMapVolumeNode"),
            TimelapsedSceneNodeCandidate("trab1", "sub-SAMPLE001_ses-1_site-tibia_mask-trab", "vtkMRMLSegmentationNode"),
            TimelapsedSceneNodeCandidate("cort2", "sub-SAMPLE001_ses-2_site-tibia_mask-cort", "vtkMRMLLabelMapVolumeNode"),
            TimelapsedSceneNodeCandidate("seg2", "sub-SAMPLE001_ses-2_site-tibia_mask-seg", "vtkMRMLLabelMapVolumeNode"),
            TimelapsedSceneNodeCandidate("tfm2", "sub-SAMPLE001_ses-2_site-tibia_transform", "vtkMRMLTransformNode"),
        ]
    )

    assert discovery.subject_id == "SAMPLE001"
    assert discovery.site == "tibia"
    assert [timepoint.session_id for timepoint in discovery.timepoints] == ["1", "2"]
    assert discovery.timepoints[0].image_node_id == "img1"
    assert discovery.timepoints[0].full_mask_node_id == "full1"
    assert discovery.timepoints[0].trab_mask_node_id == "trab1"
    assert discovery.timepoints[0].cort_mask_node_id == ""
    assert discovery.timepoints[1].image_node_id == "img2"
    assert discovery.timepoints[1].cort_mask_node_id == "cort2"
    assert discovery.timepoints[1].seg_mask_node_id == "seg2"
    assert discovery.timepoints[1].transform_node_id == "tfm2"


def test_timelapsed_scene_discovery_ignores_masks_without_matching_loaded_image() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate("img1", "SAMPLE001_T1_tibia", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("mask2", "SAMPLE001_T2_tibia_mask-full", "vtkMRMLLabelMapVolumeNode"),
        ]
    )

    assert len(discovery.timepoints) == 1
    assert discovery.timepoints[0].session_id == "1"
    assert discovery.timepoints[0].full_mask_node_id == ""


def test_timelapsed_scene_discovery_supports_strambo_year_aim_names() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate("y00", "STRAMBO_0001_RL_Y00.AIM", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("y04", "STRAMBO_0001_RL_Y04.AIM;1", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("y08", "STRAMBO_0001_RL_Y08.AIM;1", "vtkMRMLScalarVolumeNode"),
        ]
    )

    assert discovery.subject_id == "STRAMBO_0001"
    assert discovery.site == "RL"
    assert [timepoint.session_id for timepoint in discovery.timepoints] == ["00", "04", "08"]
    assert [timepoint.image_node_id for timepoint in discovery.timepoints] == ["y00", "y04", "y08"]


def test_timelapsed_scene_discovery_falls_back_to_loaded_scalar_order() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate("first", "Loaded scan A", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("mask", "Unmatched full mask", "vtkMRMLLabelMapVolumeNode"),
            TimelapsedSceneNodeCandidate("second", "Completely random image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("third", "Another volume", "vtkMRMLScalarVolumeNode"),
        ]
    )

    assert [timepoint.session_id for timepoint in discovery.timepoints] == ["1", "2", "3"]
    assert [timepoint.image_node_id for timepoint in discovery.timepoints] == ["first", "second", "third"]
    assert all(timepoint.full_mask_node_id == "" for timepoint in discovery.timepoints)


def test_timelapsed_scene_discovery_reports_summary_counts() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate("img1", "sub-SAMPLE001_ses-1_site-tibia_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("img2", "sub-SAMPLE001_ses-2_site-tibia_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("full1", "sub-SAMPLE001_ses-1_site-tibia_mask-full", "vtkMRMLLabelMapVolumeNode"),
        ]
    )

    assert discovery.image_count == 2
    assert discovery.mask_count == 1
    assert discovery.matched_mask_count == 1
