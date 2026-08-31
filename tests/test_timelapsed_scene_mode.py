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
    assert "def _style_primary_run_button" in source
    assert 'self.sceneRunButton = qt.QPushButton("Run")' in source
    assert "self._style_primary_run_button(self.runTimelapseBtn)" in source
    assert "self._style_primary_run_button(self.sceneRunButton)" in source
    assert "button.setMinimumHeight(34)" in source
    assert "QPushButton { background:#1f6feb; color:white;" in source
    assert "actions.addWidget(self.sceneRunButton)" not in source
    assert "layout.addWidget(self.sceneRunButton)" in source
    assert "self.sceneStageItems = {}" in source
    assert "self.sceneStageRows = {}" in source
    assert "rowWidget.setFixedHeight(22)" in source
    assert "stage_label.setMinimumWidth(72)" in source
    assert "status_label.setMinimumWidth(84)" in source
    assert "self.sceneStageItems[key] = status_label" in source
    assert "self.sceneStageItems[stage_key].setText" in source
    assert "self.sceneStageItems[stage_key].setForeground" not in source
    assert "self.sceneStageLabels" not in source
    assert "self.sceneStageTable" not in source
    assert 'qt.QGroupBox("Scene Results")' not in source
    assert "self.sceneResultsTable" not in source
    assert "_load_scene_results_table" in source
    assert "_load_scene_results_table_node" in source
    assert "_show_scene_results_table_node" in source
    assert "vtkMRMLTableNode" in source
    assert "SetActiveTableID" in source
    assert "PropagateTableSelection" in source
    assert "GetLayoutWithTable" in source
    assert "currentChanged.connect(self._on_timelapsed_mode_changed)" in source
    assert "def _on_timelapsed_mode_changed" in source
    assert "self.runAnalysisBtn.visible = not scene_mode" in source
    assert "self.statusBox = statusBox" in source
    assert "self.statusBox.visible = not scene_mode" in source
    assert "self.sceneStatusBox.visible = scene_mode" in source
    assert "_clear_loaded_review_nodes()" in source
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
    assert "Only native/imported masks are loaded back for scene rediscovery." in source
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
    assert "analysisSectionBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "settingsBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "self.layout.addWidget(analysisSectionBox)" in source
    assert "self.layout.addWidget(settingsBox)" in source
    assert "self.layout.addWidget(statusBox)" in source
    assert "self.layout.addWidget(self.logText)" in source
    assert "self.logText.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "self.layout.addStretch(1)" in source
    assert 'self.sceneStatusLabel.text = "Preparing scene run..."' in source
    assert 'self._set_stage_status("dataset", "done")' in source
    assert 'self._set_stage_status("parse", "done")' in source
    assert 'for stage in ("masks", "registration", "analysis")' in source
    assert 'order = ["dataset", "parse", "masks", "registration", "analysis"]' in source
    assert 'self._set_scene_stage_message(text)' in source
    assert "def _set_scene_stage_message" in source
    assert "except Exception as exc" in source
    assert "self._resize_scene_timepoint_table()" in source
    assert "_scene_timepoint_visible_rows" in source
    assert "min(row_count, 8)" in source
    assert "max(2, min(row_count, 8))" in source
    assert "ScrollBarAsNeeded" in source
    assert "setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "viewport().update()" in source
    assert "layout().activate()" in source
    assert "setMaximumHeight(max(440, min(760, height + 350)))" in source


def test_timelapsed_batch_tab_uses_uncapped_height() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    resize_body = source.split("    def _resize_timelapsed_mode_tabs", 1)[1].split("\n    def ", 1)[0]
    assert "self.timelapsedModeTabs.setMaximumHeight(max(440, min(760, height + 350)))" in resize_body
    assert "self.timelapsedModeTabs.setMaximumHeight(16777215)" in resize_body
    assert "self.timelapsedModeTabs.setMaximumHeight(520)" not in resize_body
    assert "self.timelapsedModeTabs.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "setMaximumHeight(max(520" not in source
    assert "scene_results_table_path" in source
    assert "layout.setContentsMargins(0, 0, 0, 0)" in source


def test_timelapsed_batch_series_summary_is_parented_and_collapsed() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert 'analysisSectionBox.text = "Analysis Options"' in source
    assert "analysisSectionBox.collapsed = True" in source
    assert "analysisSectionLayout.addWidget(self.seriesSummaryBox)" in source
    assert "self.seriesSummaryBox.visible = False" in source
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


def test_timelapsed_batch_current_comparison_uses_table() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert 'metricsBox = qt.QGroupBox("Current Comparison")' in source
    assert "self.currentComparisonTable = qt.QTableWidget()" in source
    assert '["Mask", "FV/BV", "RV/BV", "AV/BV", "NV/BV"]' in source
    assert "metricsLayout.addWidget(self.currentComparisonTable)" in source
    assert "def _update_current_comparison_table" in source
    metric_rows_body = source.split("    def _set_pair_metric_rows", 1)[1].split("\n    def ", 1)[0]
    assert "self._update_current_comparison_table(normalized_rows)" in metric_rows_body
    assert "first_row = normalized_rows[0]" not in metric_rows_body
    assert "metricsLayout.addWidget(self.analysisFormationFractionLabel)" not in source


def test_timelapsed_batch_manual_fallback_supports_strambo_aim_names() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _strip_manual_aim_suffix" in source
    assert r"(?i)\.aim(?:;\d+)?$" in source
    assert r"(?i)\\.aim(?:;\\d+)?$" not in source
    assert "def _manual_metadata_from_filename" in source
    assert "STRAMBO_0003_TR_Y04.AIM" in source
    assert "tibia_right" in source
    assert 'subject_id=metadata.get("subject_id", "MANUAL")' in source
    assert 'session_id=metadata.get("session_id", f"T{idx}")' in source
    assert "def _prefer_manual_aim_candidate" in source
    assert "existing = entry.get(\"image\")" in source
    assert "entry[\"image\"] = self._prefer_manual_aim_candidate(existing, path)" in source
    assert "grouped[str(path.stem)" not in source


def test_timelapsed_scene_loads_pairwise_results_table() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "TimelapsedHRpQCT Scene Results" in source
    assert "formation_frac_bv0" in source
    assert "resorption_frac_bv0" in source
    assert "AV_BV" in source
    assert "NV_BV" in source
    assert "Loaded scene results table" in source
    assert "_scene_processed_subject_site(plan)" in source
    assert "_seed_scene_transform_registry(plan)" in source
    assert "_select_first_scene_remodelling_output" in source


def test_scene_remodelling_loadback_uses_interactive_display_mask() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _display_valid_mask_for_preview_inputs" in source
    valid_mask_lookup = source.split("    def _get_valid_mask_for_source", 1)[1].split("\n    def ", 1)[0]
    assert "self._display_valid_mask_for_preview_inputs(preview_inputs)" in valid_mask_lookup
    interactive_update = source.split("    def _on_apply_interactive_remodelling", 1)[1].split("\n    def ", 1)[0]
    assert "valid_mask = self._display_valid_mask_for_preview_inputs(preview_inputs)" in interactive_update
    assert 'str(preview_inputs.get("context", {}).get("compartment", "")) == "full"' not in interactive_update
    assert "_refresh_pair_metrics_for_current_selection()" in source


def test_scene_results_table_uses_loaded_preview_metrics() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _scene_result_rows_from_loaded_remodelling" in source
    load_table = source.split("    def _load_scene_results_table(self", 1)[1].split("\n    def ", 1)[0]
    assert "rows = self._scene_result_rows_from_loaded_remodelling()" in load_table
    assert "rows_source = \"current scene display\"" in load_table
    apply_update = source.split("    def _on_apply_interactive_remodelling", 1)[1].split("\n    def ", 1)[0]
    assert "self._refresh_scene_results_table_from_loaded_remodelling()" in apply_update


def test_scene_results_table_refresh_does_not_change_remodelling_selection() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    result_rows = source.split("    def _scene_result_rows_from_loaded_remodelling", 1)[1].split("\n    def ", 1)[0]
    assert "_refresh_remodelling_full_selector()" not in result_rows
    assert "sorted(source_paths, key=self._remodelling_source_sort_key)" in result_rows
    assert "def _selected_remodelling_source_path" in source
    refresh_selector = source.split("    def _refresh_remodelling_full_selector", 1)[1].split("\n    def ", 1)[0]
    assert "selected_source_path = self._selected_remodelling_source_path()" in refresh_selector
    assert "self._set_remodelling_selector_by_source_path(selected_source_path)" in refresh_selector


def test_pair_metrics_use_same_display_mask_as_remodelling_image() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    refresh_metrics = source.split("    def _refresh_pair_metrics_for_current_selection", 1)[1].split("\n    def ", 1)[0]
    assert "valid_mask=self._display_valid_mask_for_preview_inputs(preview_inputs)" in refresh_metrics
    assert 'valid_mask=preview_inputs["valid_mask"]' not in refresh_metrics


def test_remodelling_selection_drives_visible_label_layer() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "currentIndexChanged.connect(self._on_remodelling_selection_changed)" in source
    assert "def _activate_remodelling_display_for_current_selection" in source
    assert "slicer.util.setSliceViewerLayers(label=node, fit=False)" in source
    assert "display.SetVisibility(other_node is node)" in source
    assert "activate_display=False" in source
    select_first = source.split("    def _select_first_scene_remodelling_output", 1)[1].split("\n    def ", 1)[0]
    assert "self.remodellingFullSegCombo.setCurrentIndex(0)" in select_first
    assert "self._activate_remodelling_display_for_current_selection()" in select_first


def test_scene_table_updates_existing_columns_without_rebuilding() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    table_node = source.split("    def _load_scene_results_table_node", 1)[1].split("\n    def ", 1)[0]
    assert "existing_headers == headers" in table_node
    assert "column.SetNumberOfValues(len(rows))" in table_node
    assert "table_node.RemoveAllColumns()" in table_node


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


def test_timelapsed_scene_discovery_ignores_loaded_timelapsed_outputs() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate("img1", "sub-SAMPLE001_ses-1_site-tibia_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("img2", "sub-SAMPLE001_ses-2_site-tibia_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate(
                "remodel",
                "sub-SAMPLE001_site-tibia_comp-full_t0-1_t1-2_thr-225_cluster-12_remodelling_segmentation_full",
                "vtkMRMLSegmentationNode",
                {"TimelapsedHRpQCT.RemodellingFull": "1"},
            ),
            TimelapsedSceneNodeCandidate(
                "reference",
                "sub-SAMPLE001_site-tibia_comp-full_t0-1_t1-2_thr-225_cluster-12_remodelling_segmentation_slice_reference",
                "vtkMRMLScalarVolumeNode",
                {"TimelapsedHRpQCT.SliceReference": "1"},
            ),
        ]
    )

    assert discovery.image_count == 2
    assert discovery.mask_count == 0
    assert discovery.matched_mask_count == 0
    assert [timepoint.image_node_id for timepoint in discovery.timepoints] == ["img1", "img2"]
