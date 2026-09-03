import re
import textwrap
from pathlib import Path

import pytest

from SlicerBoneImagingToolboxLib.timelapsed_scene import (
    TimelapsedSceneNodeCandidate,
    TimelapsedSceneRoiSelection,
    TimelapsedSceneTimepoint,
    build_timelapsed_scene_plan,
    discover_timelapsed_scene_timepoints,
    scene_segment_matches_role,
    timelapsed_scene_run_args,
)


def _timelapsed_widget_method(method_name: str):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    method_source = source.split(f"    def {method_name}", 1)[1].split("\n    def ", 1)[0]
    namespace = {"Path": Path, "re": re}
    exec(f"def {method_name}" + textwrap.dedent(method_source), namespace)
    return namespace[method_name]


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
    assert "self.sceneRegistrationTable" in source
    assert "self.sceneRoiTable" in source
    assert '"Registration ROI"' in source
    assert '"Segmentation"' in source
    assert '"Full/support ROI"' not in source
    assert '"Analysis ROI 1"' not in source
    assert '"Analysis ROI 2"' not in source
    assert '"Use"' in source
    assert '"Role"' in source
    assert '"Status"' in source
    assert "def _style_primary_run_button" in source
    assert 'self.sceneRunButton = qt.QPushButton("Run")' in source
    assert 'self.sceneInterruptButton = qt.QPushButton("✕ Cancel")' in source
    assert 'self.sceneExportCsvButton = qt.QPushButton("Export CSV")' in source
    assert 'self.sceneClearLoadedButton = qt.QPushButton("Clear loaded")' in source
    assert 'self.clearLoadedResultsBtn = qt.QPushButton("Clear loaded")' in source
    assert "self.sceneInterruptButton.clicked.connect(self._on_cancel_run)" in source
    assert "self.sceneExportCsvButton.clicked.connect(self._on_export_scene_comparison_csv)" in source
    assert "self.sceneClearLoadedButton.clicked.connect(self._on_clear_loaded_timelapsed_results)" in source
    assert "self.clearLoadedResultsBtn.clicked.connect(self._on_clear_loaded_timelapsed_results)" in source
    assert "self._style_primary_run_button(self.runTimelapseBtn)" in source
    assert "self._style_primary_run_button(self.sceneRunButton)" in source
    assert "button.setMinimumHeight(34)" in source
    assert "QPushButton { background:#1f6feb; color:white;" in source
    assert "actions.addWidget(self.sceneRunButton)" not in source
    assert "sceneActionLayout.addWidget(self.sceneRunButton)" in source
    assert "sceneComparisonLayout.addWidget(self.sceneExportCsvButton)" in source
    assert "sceneSecondaryActionLayout.addWidget(self.sceneInterruptButton)" in source
    assert "sceneSecondaryActionLayout.addWidget(self.sceneClearLoadedButton)" in source
    assert "def _set_widget_enabled_safe" in source
    assert 'self._set_widget_enabled_safe(getattr(self, "sceneInterruptButton", None), running)' in source
    assert "def _on_export_scene_comparison_csv" in source
    assert 'default_export_filename("timelapsed_scene_comparisons")' in source
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
    assert "prefer_saved=False" in source
    assert "prefer_saved=True" in source
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
    assert "def _on_clear_loaded_timelapsed_results" in source
    assert "gc.collect()" in source
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
    assert "def _read_seg_array_for_preview" in source
    assert "direct_seg = _read_nonempty_seg(seg_path)" in source
    assert "if np.any(seg_arr):" in source
    assert 'metadata_path = getattr(session, "metadata_path", None)' in source
    assert "iter_imported_stack_records" in source
    assert "iter_fused_session_records" in source
    assert "Only native/imported masks are loaded back for scene rediscovery." in source
    assert "_adopt_scene_run_as_current_dataset" in source
    assert "_set_path_without_immediate_reset" in source
    assert "current dataset set to scene run" in source
    assert "remodelling image" in source
    assert "_last_scene_plan" in source
    assert "loadTransform" in source
    assert "vtkMRMLLinearTransformNode" in source
    assert "def _scene_transform_is_supported_initial_transform" in source
    assert "def _remove_scene_run_nonlinear_transform_nodes" in source
    assert "regmask" in source
    assert "_scene_roi_selections()" in source
    assert "roi.role" in source
    assert "mask-seg" in source
    assert "self.sceneProfileCombo" in source
    assert "self.sceneMaskPolicyCombo" not in source
    assert "Missing masks" not in source
    assert "def _scene_mask_selector" in source
    assert "def _scene_selected_mask_node_id" in source
    assert "def _scene_selected_mask_policy" in source
    assert "def _scene_settings_override" in source
    assert 'masks_cfg["roles"] = self._scene_requested_mask_roles()' in source
    assert 'masks_cfg["generate_segmentation"] = False' in source
    assert 'inner_cfg["contour_method"] = "none"' in source
    assert 'analysis_cfg["compartments"] = self._scene_analysis_compartments()' in source
    assert 'self._add_scene_combo_item(selector, "None", "__none__"' in source
    assert 'addItem("Generate", "__generate__")' not in source
    assert "self._scene_mask_generation_requested()" not in source
    assert "identifiersBox" not in source
    assert "sceneSubjectEdit" not in source
    assert "sceneSiteEdit" not in source
    assert "sceneAnalysisThresholdSlider" not in source
    assert "sceneMaskLowerSlider" not in source
    assert "Analysis Options" in source
    assert "Advanced Settings" in source
    assert "Discovery / Import" in source
    assert "self.maskSigma" in source
    assert "self.maskContourSigma" in source
    assert "self.maskLaplaceThreshold" in source
    assert "self.maskLaplaceLowPass" in source
    assert "self.maskLaplaceHighPass" in source
    assert "self.maskLaplaceEpsilon" in source
    assert "self.maskOuterKernel" in source
    assert "self.maskOuterOpen" in source
    assert "def _on_periosteal_contour_method_changed" in source
    assert '"generate": False' in source
    assert 'masks_cfg["overwrite"] = False' in source
    assert "TimelapsedHRpQCT.GeneratedMask" in source
    assert 'qt.QGroupBox("Interactive Preview")' not in source
    assert '"Auto update"' not in source
    assert '"Update remodelling image"' not in source
    assert "analysisSectionBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "settingsBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "self.batchLayout.addWidget(analysisSectionBox)" in source
    assert "self.sceneLayout.insertWidget(2, self.analysisSectionBox)" in source
    assert "self.batchLayout.insertWidget(2, self.analysisSectionBox)" in source
    assert "self.layout.addWidget(analysisSectionBox)" not in source
    assert "self.layout.addWidget(settingsBox)" in source
    assert "self.layout.addWidget(statusBox)" in source
    assert "self.layout.addWidget(self.logText)" in source
    assert "self.logText.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "self.layout.addStretch(1)" in source
    assert '"quiescent": 2' in source
    assert '"demineralisation": 2' in source
    assert '"mineralisation": 2' in source
    assert 'label_map.update({"demineralisation": 0, "quiescent": 0, "mineralisation": 0})' in source
    assert 'label_map.update({"demineralisation": 2, "quiescent": 2, "mineralisation": 2})' in source
    assert 'binary_cfg["enabled"] = True' not in source
    assert 'self.sceneStatusLabel.text = "Preparing scene run..."' in source
    assert 'self._set_stage_status("dataset", "done")' in source
    assert 'self._set_stage_status("parse", "done")' in source
    assert 'for stage in ("registration", "analysis")' in source
    assert 'order = ["dataset", "parse", "registration", "analysis"]' in source
    assert '("masks", "Masks")' not in source
    assert '("masks", "Masks / ROIs")' not in source
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
    assert "self.timelapsedModeTabs.setMaximumHeight(16777215)" in source


def test_timelapsed_batch_tab_uses_uncapped_height() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    resize_body = source.split("    def _resize_timelapsed_mode_tabs", 1)[1].split("\n    def ", 1)[0]
    assert "self.timelapsedModeTabs.setMaximumHeight(16777215)" in resize_body
    assert "self.timelapsedModeTabs.setMaximumHeight(520)" not in resize_body
    assert "self.timelapsedModeTabs.setMaximumHeight(520)" not in source
    assert "max(440, min(760" not in resize_body
    assert "self.timelapsedModeTabs.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "setMaximumHeight(max(520" not in source
    assert "scene_results_table_path" in source
    assert "layout.setContentsMargins(0, 0, 0, 0)" in source


def test_timelapsed_batch_cohort_summary_is_not_in_analysis_options() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert 'analysisSectionBox.text = "Analysis Options"' in source
    assert "analysisSectionBox.collapsed = False" in source
    assert 'qt.QGroupBox("Series Summary")' not in source
    assert "seriesSummaryExportBtn" not in source
    assert "analysisSectionLayout.addWidget(self.seriesSummaryBox)" not in source
    assert 'env.insert("PYTHONPATH", os.environ["PYTHONPATH"])' in source
    assert "_resolve_local_pipeline_paths" in source
    assert 'base / "TimelapsedHRpQCT"' in source
    assert 'selected / "derivatives" / "Timelapse"' in source
    assert 'if selected.name == "derivatives":' in source
    assert 'return selected / "Timelapse"' in source
    assert "timelapsedhrpqct.cli import main" in source
    assert 'MIN_PIPELINE_VERSION = "2.0.43"' in source
    assert "Move up" in source
    assert "Move down" in source
    assert "discover_timelapsed_scene_timepoints" in source
    assert "generate_missing_masks=False" in source
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


def test_timelapsed_batch_custom_profile_exposes_analysis_options_without_cli_profile() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert 'combo.addItem("Custom", "__custom__")' in source
    assert "def _selected_profile_is_custom" in source
    profile_args = source.split("    def _profile_cli_args", 1)[1].split("\n    def ", 1)[0]
    assert "if self._selected_profile_is_custom():" in profile_args
    assert "return []" in profile_args
    apply_profile = source.split("    def _on_apply_study_profile", 1)[1].split("\n    def ", 1)[0]
    assert "if self._selected_profile_is_custom():" in apply_profile
    assert "self._update_batch_analysis_options_visibility()" in apply_profile
    settings_override = source.split("    def _settings_override", 1)[1].split("\n    def ", 1)[0]
    assert "if self._selected_profile_is_custom() or bool(force_analysis_controls):" in settings_override
    assert 'settings["analysis"] = self._analysis_config_from_controls(pair_mode)' in settings_override
    mode_changed = source.split("    def _on_timelapsed_mode_changed", 1)[1].split("\n    def ", 1)[0]
    assert "self._update_batch_analysis_options_visibility()" in mode_changed
    visibility = source.split("    def _update_batch_analysis_options_visibility", 1)[1].split("\n    def ", 1)[0]
    assert "scene_mode = self._timelapsed_scene_mode_selected()" in visibility
    assert "custom = self._selected_profile_is_custom()" in visibility
    assert "self.analysisSectionBox.visible = scene_mode or custom" in visibility


def test_timelapsed_profile_display_order_matches_public_profiles() -> None:
    reporting_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCTLib"
        / "Reporting.py"
    )
    reporting = reporting_path.read_text(encoding="utf-8")

    assert '"eth-uofc"' in reporting
    assert '"multistack"' in reporting
    assert '"ped-fx"' in reporting
    assert '"shriners"' in reporting
    assert '"standard"' in reporting
    assert '"ucsf"' in reporting
    assert '"xct1-standard"' in reporting
    assert '"single-stack"' not in reporting
    assert '"low-memory"' not in reporting


def test_timelapsed_batch_remodelling_loader_accepts_any_roi_and_prefers_full() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert '"remodelling image", "transformed", "raw"' in source
    refresh = source.split("    def _refresh_remodelling_comparison_list", 1)[1].split("\n    def ", 1)[0]
    assert 'self.loadTypeCombo.currentText == "remodelling image"' in refresh
    assert 'str(ctx.get("compartment", "")).strip().lower() != "full"' not in refresh
    assert "remodelling_sort_key" in refresh
    assert '0 if str(ctx.get("compartment", "")).strip().lower() == "full" else 1' in refresh
    assert 'label = f"{ctx[\'t0\']} -> {ctx[\'t1\']} ({self._scene_display_compartment_name(ctx.get(\'compartment\', \'full\'))})"' in refresh
    load_selected = source.split("    def _on_load_selected", 1)[1].split("\n    def ", 1)[0]
    assert 'is_remodelling_load = data_type == "remodelling image"' in load_selected
    assert "loaded_remodelling_source_path = None" in load_selected
    assert "loaded_remodelling_source_path = str(Path(p).resolve())" in load_selected
    assert "self._set_remodelling_selector_by_source_path(loaded_remodelling_source_path)" in load_selected
    assert "ctx = self._parse_remodelling_source_context(loaded_remodelling_source_path)" in load_selected
    assert "saved_rows = self._saved_pair_metric_rows_for_context(ctx)" in load_selected
    assert "self._set_pair_metric_rows(saved_rows)" in load_selected
    assert "loaded as remodelling segmentation." in load_selected
    assert 'findText("remodelling image")' in source


def test_remodelling_source_parser_accepts_current_voi_desc_filenames() -> None:
    parse_context = _timelapsed_widget_method("_parse_remodelling_source_context")
    filename = (
        "/tmp/derivatives/Timelapse/sub-001/xct/analysis/visualize/"
        "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-5_remodelling.nii.gz"
    )

    ctx = parse_context(object(), filename)

    assert ctx is not None
    assert ctx["subject_id"] == "001"
    assert ctx["site"] == "radiusleft"
    assert ctx["compartment"] == "roi_union"
    assert ctx["t0"] == "001"
    assert ctx["t1"] == "002"
    assert ctx["threshold"] == 225.0
    assert ctx["cluster"] == 5


def test_timelapsed_batch_loaded_remodelling_populates_current_comparison_from_saved_csv() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _saved_pair_metric_rows_for_context" in source
    saved_rows = source.split("    def _saved_pair_metric_rows_for_context", 1)[1].split("\n    def ", 1)[0]
    assert "pairwise_remodelling_csv_path(imported, subject_id, site)" in saved_rows
    assert 'str(row.get("t0") or "").strip() != t0' in saved_rows
    assert 'str(row.get("t1") or "").strip() != t1' in saved_rows
    assert 'self._csv_float_or_nan(row.get("formation_frac_bv0"))' in saved_rows
    assert 'self._csv_float_or_nan(row.get("resorption_frac_bv0"))' in saved_rows
    assert 'float(row.get("formation_frac_bv0", "nan"))' not in saved_rows
    assert "def _csv_float_or_nan" in source
    refresh = source.split("    def _refresh_pair_metrics_for_current_selection", 1)[1].split("\n    def ", 1)[0]
    assert "saved_rows = self._saved_pair_metric_rows_for_context(ctx)" in refresh
    assert "if self._metric_rows_have_finite_fractions(saved_rows):" in refresh
    assert "self._set_pair_metric_rows(saved_rows)" in refresh
    assert "def _metric_rows_have_finite_fractions" in source


def test_timelapsed_analysis_options_are_positioned_below_active_profile() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "self.sceneLayout = layout" in source
    assert "self.sceneProfileBox = sceneProfileBox" in source
    assert "def _place_analysis_options_for_mode" in source
    assert "self.sceneLayout.insertWidget(2, self.analysisSectionBox)" in source
    assert "self.batchLayout.insertWidget(2, self.analysisSectionBox)" in source
    assert "self._place_analysis_options_for_mode()" in source.split(
        "    def _on_timelapsed_mode_changed", 1
    )[1].split("\n    def ", 1)[0]
    assert "self.layout.addWidget(analysisSectionBox)" not in source


def test_timelapsed_scene_layout_follows_batch_workflow_order() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    scene_ui = source.split("    def _build_scene_ui", 1)[1].split("\n    def ", 1)[0]

    assert 'sceneRegistrationBox = qt.QGroupBox("Timepoints")' in scene_ui
    assert 'sceneProfileBox = qt.QGroupBox("Study Profile")' in scene_ui
    assert "sceneRegistrationLayout.addWidget(sceneDiscoveryRow)" in scene_ui
    assert "sceneRegistrationLayout.addWidget(self.sceneRegistrationTable)" in scene_ui
    assert "sceneRegistrationLayout.addWidget(sceneRoiBox)" in scene_ui
    assert "sceneRegistrationLayout.addWidget(sceneWorkspaceRow)" in scene_ui
    assert "layout.addWidget(sceneRegistrationBox)" in scene_ui
    assert "layout.addWidget(sceneProfileBox)" in scene_ui
    assert scene_ui.index('layout.addWidget(sceneRegistrationBox)') < scene_ui.index('layout.addWidget(sceneProfileBox)')
    assert scene_ui.index("sceneRegistrationLayout.addWidget(self.sceneRegistrationTable)") < scene_ui.index("sceneRegistrationLayout.addWidget(sceneRoiBox)")
    assert scene_ui.index("sceneRegistrationLayout.addWidget(sceneRoiBox)") < scene_ui.index("layout.addWidget(sceneRegistrationBox)")
    assert scene_ui.index('layout.addWidget(sceneProfileBox)') < scene_ui.index('layout.addWidget(sceneActionBox)')
    assert "sceneActionLayout.addLayout(actions)" not in scene_ui
    assert "layout.addWidget(sceneWorkspaceBox)" not in scene_ui


def test_timelapsed_ui_uses_compact_vertical_spacing() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    build_ui = source.split("    def _build_ui", 1)[1].split("\n    def ", 1)[0]
    scene_ui = source.split("    def _build_scene_ui", 1)[1].split("\n    def ", 1)[0]
    assert "self.batchLayout.setContentsMargins(0, 0, 0, 0)" in build_ui
    assert "self.batchLayout.setSpacing(4)" in build_ui
    assert "form.setVerticalSpacing(4)" in build_ui
    assert "quickForm.setContentsMargins(6, 8, 6, 6)" in build_ui
    assert "analysisSectionLayout.setContentsMargins(6, 6, 6, 4)" in build_ui
    assert "settingsLayout.setContentsMargins(6, 6, 6, 4)" in build_ui
    assert "settingsLayout.setSpacing(6)" in build_ui
    assert "parseLayout.setContentsMargins(6, 6, 6, 4)" in build_ui
    assert "self.parseTable.setMinimumHeight(128)" in build_ui
    assert "self.logText.setMinimumHeight(140)" in build_ui
    assert "self.logText.setMaximumHeight(200)" in build_ui
    assert "layout.setSpacing(4)" in scene_ui
    assert "form.setContentsMargins(6, 8, 6, 6)" in scene_ui
    assert "sceneRegistrationLayout.setContentsMargins(6, 8, 6, 6)" in scene_ui
    assert "sceneActionLayout.setContentsMargins(6, 8, 6, 6)" in scene_ui


def test_timelapsed_scene_role_mapping_contracts_to_full_width_summary() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    scene_ui = source.split("    def _build_scene_ui", 1)[1].split("\n    def ", 1)[0]
    visibility = source.split("    def _update_scene_role_mapping_visibility", 1)[1].split("\n    def ", 1)[0]

    assert "sceneRoiBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in scene_ui
    assert "self.sceneRoiTable.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in scene_ui
    assert "self.sceneRoiTable.setMinimumWidth(420)" in scene_ui
    assert "sceneRoiLayout.addSpacing(6)" in scene_ui
    assert "roi_extra_padding = 8 if table is self.sceneRoiTable else 12" in source
    assert "height = header_height + visible_rows * row_height + roi_extra_padding" in source
    assert "header.setSectionResizeMode(1, qt.QHeaderView.Stretch)" in visibility
    assert "header.setSectionResizeMode(status_column, qt.QHeaderView.Fixed)" in visibility
    assert "header.setSectionResizeMode(1, qt.QHeaderView.Fixed)" in visibility


def test_timelapsed_scene_mask_group_selection_populates_matching_roles() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    selector = source.split("    def _scene_segmentation_node_selector", 1)[1].split("\n    def ", 1)[0]
    changed = source.split("    def _on_scene_mask_source_changed", 1)[1].split("\n    def ", 1)[0]
    assign = source.split("    def _apply_scene_detected_roles_for_timepoint", 1)[1].split("\n    def ", 1)[0]

    assert "lambda _node=None, selector=selector: self._on_scene_mask_source_changed(selector)" in selector
    assert "timepoint_index = self._scene_timepoint_index_for_mask_source_selector(selector)" in changed
    assert "self._apply_scene_detected_roles_for_timepoint(timepoint_index, source_node)" in changed
    assert 'for role in ("registration_roi", "segmentation", "roi1", "roi2", "roi3"):' in assign
    assert 'lookup_role = "full" if self._normalize_scene_role_name(role) == "registration_roi" else role' in assign
    assert "segment_id = self._scene_segment_id_for_node_role(source_node_id, lookup_role)" in assign
    assert "self._set_scene_mask_row_node(role_row, column, source_node_id, self.sceneRoiTable, role=role, segment_id=segment_id)" in assign
    assert 'self._set_scene_mask_row_policy(role_row, column, "node", self.sceneRoiTable)' in assign


def test_timelapsed_scene_auto_detect_roles_reapplies_existing_rows() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    auto_detect = source.split("    def _auto_detect_scene_rois", 1)[1].split("\n    def ", 1)[0]
    populate = source.split("    def _populate_scene_roi_rows_from_timepoints", 1)[1].split("\n    def ", 1)[0]

    assert "discovery = discover_timelapsed_scene_timepoints(self._scene_node_candidates())" in auto_detect
    assert "timepoints = list(discovery.timepoints)" in auto_detect
    assert "self._populate_scene_roi_rows_from_timepoints(timepoints, reapply_existing=True)" in auto_detect
    assert "def _populate_scene_roi_rows_from_timepoints(self, timepoints, reapply_existing=False):" in source
    assert "if nodes_by_session and (role not in existing_roles or reapply_existing):" in populate
    assert "role_row = self._scene_role_row_index(role)" in populate
    assert "self._set_scene_mask_row_node(role_row, column, node_id, self.sceneRoiTable, role=role, segment_id=segment_id)" in populate


def test_timelapsed_scene_discovery_runs_segmentation_role_detection() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    discover = source.split("    def _on_discover_scene_timepoints", 1)[1].split("\n    def ", 1)[0]

    assert "self._populate_scene_roi_rows_from_timepoints(discovery.timepoints)" in discover
    assert "for timepoint_index in range(self.sceneRegistrationTable.rowCount):" in discover
    assert "source_node = self._scene_selected_table_node(timepoint_index, 2, self.sceneRegistrationTable)" in discover
    assert "self._apply_scene_detected_roles_for_timepoint(timepoint_index, source_node)" in discover


def test_timelapsed_scene_role_status_updates_when_roi_selector_changes() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    configure = source.split("    def _configure_scene_role_row", 1)[1].split("\n    def ", 1)[0]
    add_roi = source.split("    def _add_scene_roi", 1)[1].split("\n    def ", 1)[0]
    refresh = source.split("    def _update_scene_role_status_for_selector", 1)[1].split("\n    def ", 1)[0]

    assert "def _connect_scene_role_selector_status" in source
    assert "def _update_scene_role_status_for_selector" in source
    assert "self._connect_scene_role_selector_status(selector)" in configure
    assert "self._connect_scene_role_selector_status(selector)" in add_roi
    assert "selector.currentIndexChanged.connect(" in source
    assert "if self.sceneRoiTable.cellWidget(row, column) is selector:" in refresh
    assert "self._update_scene_role_status(row)" in refresh


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
    assert "tibiaright" in source
    assert 'subject_id=metadata.get("subject_id", "MANUAL")' in source
    assert 'session_id=metadata.get("session_id", f"T{idx}")' in source
    assert "def _prefer_manual_aim_candidate" in source
    assert "existing = entry.get(\"image\")" in source
    assert "entry[\"image\"] = self._prefer_manual_aim_candidate(existing, path)" in source
    assert "grouped[str(path.stem)" not in source


def test_timelapsed_batch_structured_fallback_is_success_status() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _manual_sessions_need_correction" in source
    assert "needs_correction = self._manual_sessions_need_correction(manual_sessions)" in source
    assert 'self._set_stage_status("parse", "error" if needs_correction else "done")' in source
    assert '"Parse used fallback"' in source
    assert '"Parse needs correction"' in source


def test_timelapsed_profile_override_preserves_outer_morphology_values() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    override_body = source.split("    def _settings_override", 1)[1].split("\n    def ", 1)[0]

    assert '"periosteal_kernelsize": int(self.maskOuterKernel.value)' in override_body
    assert '"periosteal_open_radius": int(self.maskOuterOpen.value)' in override_body
    assert '"use_laplace_hamming_contour_support"' not in override_body


def test_timelapsed_keeps_contour_generation_out_of_visible_workflow() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    build_ui = source.split("    def _build_ui", 1)[1].split("\n    def _style_primary_run_button", 1)[0]
    override_body = source.split("    def _settings_override", 1)[1].split("\n    def ", 1)[0]

    assert 'discoveryBox.text = "Discovery / Import"' in build_ui
    assert "self.batchLayout.addWidget(discoveryBox)" in build_ui
    assert "discoveryLayout.addRow(_label(\"Copy raw inputs\"" in build_ui
    assert "discoveryLayout.addRow(_label(\"Restructure raw inputs\"" in build_ui
    assert "discoveryLayout.addRow(_label(\"Parse mode\"" in build_ui
    assert "discoveryLayout.addRow(_label(\"Storage mode\"" in build_ui
    assert "maskForm.addRow(_label(\"Copy raw inputs\"" not in build_ui
    assert "maskForm.addRow(_label(\"Storage mode\"" not in build_ui
    assert "settingsLayout.addWidget(maskBox)" not in build_ui
    assert "self.maskGenerationBox.visible = False" in build_ui
    assert '"generate": False' in override_body
    assert '"overwrite": False' in override_body


def test_timelapsed_hides_legacy_mask_generation_controls() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    assert "self.maskGenerationBox.visible = False" in source
    assert "settingsLayout.addWidget(maskBox)" not in source


def test_timelapsed_batch_processing_scope_has_subject_and_site() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "self.processingSubjectCombo = qt.QComboBox()" in source
    assert "self.processingSiteCombo = qt.QComboBox()" in source
    assert 'self.processingSiteCombo.addItem("All sites")' in source
    assert "self.processingSubjectCombo.currentIndexChanged.connect(self._refresh_processing_sites)" in source
    assert 'Processing site' in source
    assert "def _selected_processing_site" in source
    assert "def _refresh_processing_sites" in source
    assert "return scoped, subject, site" in source
    assert "scoped_sessions, scoped_subject, scoped_site = self._sessions_for_processing_scope()" in source
    assert "force_virtual_root=bool(scoped_subject or scoped_site)" in source
    assert '*(["--site", str(scoped_site)] if scoped_site else [])' in source


def test_timelapsed_scene_layout_matches_batch_concepts() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    scene_ui = source.split("    def _build_scene_ui", 1)[1].split("\n    def ", 1)[0]

    assert 'sceneRegistrationBox = qt.QGroupBox("Timepoints")' in scene_ui
    assert 'sceneProfileBox = qt.QGroupBox("Study Profile")' in scene_ui
    assert 'sceneRoiBox = ctk.ctkCollapsibleButton()' in scene_ui
    assert 'sceneRoiBox.text = "Role Mapping"' in scene_ui
    assert "sceneRoiBox.collapsed = True" in scene_ui
    assert "self.sceneRegistrationTable = qt.QTableWidget()" in scene_ui
    assert '"Masks / segments"' in scene_ui
    assert "self.sceneRoiTable = qt.QTableWidget()" in scene_ui
    assert "sceneRoleOrderEdit" not in scene_ui
    assert "sceneApplyRoleOrderButton" not in scene_ui
    assert "Apply segment order" not in scene_ui
    assert "Segment order" not in scene_ui
    assert "self._scene_mask_selector(timepoint_index=timepoint_index)" in source
    assert "self._scene_mask_source_node_id_for_timepoint(timepoint_index)" in source
    assert "self._scene_role_is_complete(row)" not in source
    assert "self.sceneRoiTable.setRowHidden(row, hidden)" not in source
    assert 'self._scene_role_label(role)' in source
    assert '"roi1": "full"' in source
    assert '"roi2": "trab"' in source
    assert '"roi3": "cort"' in source
    assert "if self._scene_role_is_analysis_roi(role):" in source
    assert "return self._normalize_scene_role_name(item.text())" in source
    assert 'sceneActionBox = qt.QGroupBox("Pipeline")' in scene_ui
    assert 'self.sceneComparisonBox = qt.QGroupBox("Current Comparisons")' in scene_ui
    assert "self.sceneComparisonTable = qt.QTableWidget()" in scene_ui
    assert '["Pair", "Mask", "FV/BV", "RV/BV", "AV/BV", "NV/BV"]' in scene_ui
    assert "sceneRegistrationLayout.addWidget(sceneDiscoveryRow)" in scene_ui
    assert "sceneRegistrationLayout.addWidget(sceneWorkspaceRow)" in scene_ui
    assert "layout.addWidget(sceneWorkspaceBox)" not in scene_ui
    assert 'sceneAdvancedBox.text = "Advanced Settings"' not in scene_ui
    assert 'Processing workspace' in scene_ui
    assert "form.addRow(_label(\"Results folder\"" not in scene_ui
    assert "layout.addWidget(self.sceneComparisonBox)" in scene_ui
    assert "sceneRegistrationLayout.addWidget(sceneRoiBox)" in scene_ui
    assert scene_ui.index("sceneRegistrationLayout.addWidget(self.sceneRegistrationTable)") < scene_ui.index(
        "sceneRegistrationLayout.addWidget(sceneRoiBox)"
    )
    assert scene_ui.index("sceneRegistrationLayout.addWidget(sceneRoiBox)") < scene_ui.index(
        "layout.addWidget(sceneRegistrationBox)"
    )
    assert scene_ui.index("layout.addWidget(sceneRegistrationBox)") < scene_ui.index("layout.addWidget(sceneProfileBox)")
    assert scene_ui.index("layout.addWidget(sceneProfileBox)") < scene_ui.index("layout.addWidget(sceneActionBox)")
    comparison_ui = scene_ui.split('self.sceneComparisonBox = qt.QGroupBox("Current Comparisons")', 1)[1].split(
        'layout.addWidget(self.sceneComparisonBox)', 1
    )[0]
    assert "sceneComparisonLayout.addWidget(self.sceneExportCsvButton)" in comparison_ui
    assert "def _set_scene_comparison_rows" in source
    assert "self._set_scene_comparison_rows(rows)" in source
    assert "def _load_scene_results_table(self, plan, *, show=False, prefer_saved=False):" in source
    assert "def _apply_scene_segment_order(self):" not in source
    assert "def _scene_selected_table_node(self, row, column, table):" in source
    assert "role = self._scene_role_at_row(row)" in source


def test_timelapsed_wrapper_never_launches_mask_generation() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    run_masks = source.split("    def _on_run_masks", 1)[1].split("\n    def ", 1)[0]
    assert '"generate-masks"' not in run_masks
    assert "Mask generation moved to Bone Contouring" in run_masks
    run_timelapse = source.split("    def _on_run_timelapse", 1)[1].split("\n    def ", 1)[0]
    assert "self._run_skips_mask_generation = True" in run_timelapse
    run_full = source.split("    def _on_run_full_pipeline", 1)[1].split("\n    def ", 1)[0]
    assert "self._run_skips_mask_generation = True" in run_full
    assert "skip_mask_generation = True" not in run_full
    scene_refresh = source.split("    def _run_scene_analysis_for_missing_pair_mode", 1)[1].split("\n    def ", 1)[0]
    assert "self._run_skips_mask_generation = True" in scene_refresh


def test_timelapsed_batch_preflights_required_masks_before_launch() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    run_full = source.split("    def _on_run_full_pipeline", 1)[1].split("\n    def ", 1)[0]
    assert "missing_inputs = self._missing_batch_required_inputs(" in run_full
    assert "if missing_inputs:" in run_full
    assert "Missing required Timelapsed input" in run_full
    assert run_full.index("missing_inputs = self._missing_batch_required_inputs(") < run_full.index("self._run(run_args)")

    validator = source.split("    def _missing_batch_required_inputs", 1)[1].split("\n    def ", 1)[0]
    assert "registration mask" in validator
    assert "analysis ROI" in validator
    assert "bone segmentation" in validator


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


def test_timelapsed_scene_tables_display_named_default_rois() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    display_labels = source.split("    def _scene_role_display_labels", 1)[1].split("\n    def ", 1)[0]
    display_name = source.split("    def _scene_display_compartment_name", 1)[1].split("\n    def ", 1)[0]

    assert "def _scene_role_display_labels" in source
    assert "def _scene_display_compartment_name" in source
    assert "labels[role] = label" in display_labels
    assert "table_labels = self._scene_role_display_labels()" in display_name
    assert "if normalized in table_labels:" in display_name
    assert "return table_labels[normalized]" in display_name
    assert '"roi1": "full"' in display_name
    assert '"roi2": "trab"' in display_name
    assert '"roi3": "cort"' in display_name
    assert "self._scene_display_compartment_name(metric_row.get(\"compartment\", \"full\"))" in source
    assert "self._scene_display_compartment_name(compartment)" in source


def test_timelapsed_scene_preview_skips_synthetic_roi_union() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _scene_compartment_is_interactive_source" in source
    assert '"roi_union"' in source
    result_rows = source.split("    def _scene_result_rows_from_loaded_remodelling", 1)[1].split("\n    def ", 1)[0]
    assert "ctx = self._parse_remodelling_source_context(source_path)" in result_rows
    assert "source_by_pair = {}" in result_rows
    assert "rank = 0 if self._scene_compartment_is_interactive_source(ctx.get(\"compartment\", \"\")) else 1" in result_rows
    assert "self._compute_pair_union_remodelling_preview(" in result_rows
    assert "preview_inputs = self._get_interactive_preview_inputs(source_path)" in result_rows
    pair_metrics = source.split("    def _refresh_pair_metrics_for_current_selection", 1)[1].split("\n    def ", 1)[0]
    assert "if not self._scene_compartment_is_interactive_source(ctx.get(\"compartment\", \"\")):" in pair_metrics


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


def test_scene_interactive_preview_caches_density_delta_for_live_updates() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    get_inputs = source.split("    def _get_interactive_preview_inputs", 1)[1].split("\n    def ", 1)[0]
    delta_helper = source.split("    def _preview_delta_for_current_settings", 1)[1].split("\n    def ", 1)[0]
    preview_helper = source.split(
        "    def _compute_pair_remodelling_preview_from_cached_delta", 1
    )[1].split("\n    def ", 1)[0]

    assert "delta_zyx = (img_t1 - img_t0).astype(np.float32, copy=False)" in get_inputs
    assert '"delta_zyx": delta_zyx' in get_inputs
    assert "delta_cache = preview_inputs.setdefault(\"delta_cache\", {})" in delta_helper
    assert 'cache_key = ("gaussian", round(sigma, 6))' in delta_helper
    assert "maybe_smooth_density(" in delta_helper
    assert "from timelapsedhrpqct.analysis import compute_pair_remodelling_preview_from_delta" in preview_helper
    assert "delta = self._preview_delta_for_current_settings(preview_inputs)" in preview_helper
    assert "compute_pair_remodelling_preview_from_delta" in preview_helper


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
    assert "sorted(" in result_rows
    assert "source_by_pair.items()" in result_rows
    assert "key=lambda item: self._remodelling_source_sort_key(item[1][1])" in result_rows
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


def test_pair_mode_change_marks_analysis_settings_dirty() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "self.analysisPairModeCombo.currentIndexChanged.connect(self._on_analysis_pair_mode_changed)" in source
    assert "def _on_analysis_pair_mode_changed" in source
    pair_changed = source.split("    def _on_analysis_pair_mode_changed", 1)[1].split("\n    def ", 1)[0]
    assert "self._mark_analysis_settings_dirty()" in pair_changed
    assert "_refresh_scene_results_table_from_loaded_remodelling" not in pair_changed
    assert "_run_scene_analysis_for_missing_pair_mode" not in pair_changed


def test_scene_pair_mode_is_analysis_setting_not_display_filter() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    result_rows = source.split("    def _scene_result_rows_from_loaded_remodelling", 1)[1].split("\n    def ", 1)[0]
    assert "_filter_remodelling_source_paths_for_pair_mode" not in source
    assert "return self._detect_missing_scene_baseline_pairs(rows)" in result_rows
    assert "_compose_scene_baseline_rows_from_adjacent" not in source
    assert "def _scene_result_row_matches_pair_mode" not in source
    assert "def _filter_scene_result_rows_for_pair_mode" not in source
    saved_rows = source.split("    def _scene_result_rows(self, plan)", 1)[1].split("\n    def ", 1)[0]
    assert "return self._detect_missing_scene_baseline_pairs(rows)" in saved_rows


def test_scene_baseline_mode_runs_analysis_when_true_baseline_pairs_are_missing() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _warn_missing_scene_baseline_pairs" in source
    assert "def _run_scene_analysis_for_missing_pair_mode" in source
    warning = source.split("    def _warn_missing_scene_baseline_pairs", 1)[1].split("\n    def ", 1)[0]
    assert 'self._current_analysis_pair_mode() != "baseline"' in warning
    assert "expected_pairs" in warning
    assert "missing_pairs" in warning
    assert "self._last_missing_scene_baseline_pairs = list(missing_pairs)" in warning
    pair_changed = source.split("    def _on_analysis_pair_mode_changed", 1)[1].split("\n    def ", 1)[0]
    assert "self._mark_analysis_settings_dirty()" in pair_changed
    assert "self._run_scene_analysis_for_missing_pair_mode()" not in pair_changed
    assert "self._refresh_scene_results_table_from_loaded_remodelling()" not in pair_changed
    apply_update = source.split("    def _on_apply_interactive_remodelling", 1)[1].split("\n    def ", 1)[0]
    assert "self._refresh_scene_results_table_from_loaded_remodelling()" in apply_update
    assert "if self._run_scene_analysis_for_missing_pair_mode():" in apply_update
    refresh = source.split("    def _run_scene_analysis_for_missing_pair_mode", 1)[1].split("\n    def ", 1)[0]
    assert '"analyse"' in refresh
    assert "self._clear_scene_analysis_outputs_for_refresh(plan)" in refresh
    assert "self._last_scene_plan = plan" in refresh
    assert "self._scene_settings_override()" in refresh
    assert "pair_mode_label" in refresh
    assert "Pair mode = {pair_mode_label}" in refresh


def test_scene_comparison_table_clears_before_repopulate() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    setter = source.split("    def _set_scene_comparison_rows", 1)[1].split("\n    def ", 1)[0]
    assert "self.sceneComparisonTable.clearContents()" in setter
    assert "seen_row_keys" in setter
    current_setter = source.split("    def _update_current_comparison_table", 1)[1].split("\n    def ", 1)[0]
    assert "self.currentComparisonTable.clearContents()" in current_setter
    assert "seen_row_keys" in current_setter


def test_analysis_options_are_expanded_for_custom_batch_or_scene_mode() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    analysis_setup = source.split('analysisSectionBox.text = "Analysis Options"', 1)[1].split(
        'settingsBox = ctk.ctkCollapsibleButton()', 1
    )[0]
    assert "analysisSectionBox.collapsed = True" in analysis_setup
    assert "self.analysisSectionBox = analysisSectionBox" in analysis_setup
    visibility = source.split("    def _update_batch_analysis_options_visibility", 1)[1].split("\n    def ", 1)[0]
    assert "self.analysisSectionBox.visible = scene_mode or custom" in visibility
    assert "self.analysisSectionBox.collapsed = False" in visibility


def test_remodelling_selection_drives_visible_scalar_overlay() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "currentIndexChanged.connect(self._on_remodelling_selection_changed)" in source
    assert "def _activate_remodelling_display_for_current_selection" in source
    assert "slicer.util.setSliceViewerLayers(foreground=node, foregroundOpacity=0.65, fit=False)" in source
    assert "slicer.util.setSliceViewerLayers(label=node, fit=False)" not in source
    assert '"vtkMRMLScalarVolumeNode"' in source
    assert "display.SetVisibility(other_node is node)" in source
    assert "self._style_remodelling_scalar_volume(node, activate_display=False)" in source
    assert "self._center_slices_on_node(node, fit_to_bounds=True)" in source
    select_first = source.split("    def _select_first_scene_remodelling_output", 1)[1].split("\n    def ", 1)[0]
    assert "self.remodellingFullSegCombo.setCurrentIndex(0)" in select_first
    assert "self._activate_remodelling_display_for_current_selection()" in select_first


def test_scene_loadback_prewarms_selected_interactive_preview_cache() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    load_outputs = source.split("    def _load_scene_run_outputs", 1)[1].split("\n    def ", 1)[0]
    prewarm = source.split("    def _prewarm_selected_scene_preview_cache", 1)[1].split("\n    def ", 1)[0]

    assert "def _prewarm_selected_scene_preview_cache" in source
    assert "self._prewarm_selected_scene_preview_cache()" in load_outputs
    assert "source_path = self._selected_remodelling_source_path()" in prewarm
    assert "self._get_interactive_preview_inputs(source_path)" in prewarm
    assert '_ = self._preview_delta_for_current_settings(preview_inputs)' in prewarm
    assert "self._compute_pair_union_remodelling_preview(ctx, source_path=source_path)" in prewarm
    assert "[preview] prewarmed interactive cache" in prewarm


def test_scene_loadback_prewarms_selected_union_from_saved_scene_data() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    prewarm = source.split("    def _prewarm_selected_scene_preview_cache", 1)[1].split("\n    def ", 1)[0]

    assert "if source_node is None:" in prewarm
    assert "prewarming display-union output from saved scene data" in prewarm
    assert "self._compute_pair_union_remodelling_preview(ctx, source_path=source_path)" in prewarm
    assert "No loaded per-ROI remodelling source is available for this pair" not in prewarm


def test_scene_union_update_can_recompute_from_selected_union_source_path() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    get_inputs = source.split("    def _get_interactive_preview_inputs", 1)[1].split("\n    def ", 1)[0]
    union_update = source.split("    def _compute_pair_union_remodelling_preview", 1)[1].split("\n    def ", 1)[0]
    apply_update = source.split("    def _on_apply_interactive_remodelling", 1)[1].split("\n    def ", 1)[0]

    assert 'if compartment == "roi_union":' in get_inputs
    assert 'compartment_for_valid_mask = "full"' in get_inputs
    assert "def _compute_pair_union_remodelling_preview(self, target_ctx, source_path=None)" in source
    assert "source_path = str(source_path or \"\")" in union_update
    assert "if not source_path and source_node is None:" in union_update
    assert "preview_inputs = self._get_interactive_preview_inputs(source_path)" in union_update
    assert "self._compute_pair_union_remodelling_preview(" in apply_update
    assert "source_path=source_path" in apply_update


def test_scene_union_update_refreshes_tables_from_recomputed_roi_metrics() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    apply_update = source.split("    def _on_apply_interactive_remodelling", 1)[1].split("\n    def ", 1)[0]

    union_branch = apply_update.split("if not self._scene_compartment_is_interactive_source", 1)[1].split(
        "            return\n\n        view_state = self._capture_slice_view_state()",
        1,
    )[0]
    assert "metric_rows = self._compute_pair_metric_rows(preview_inputs)" in union_branch
    assert "self._set_pair_metric_rows(metric_rows)" in union_branch
    assert "self._refresh_scene_results_table_from_loaded_remodelling()" in union_branch
    assert "self._set_scene_comparison_rows(scene_rows)" not in union_branch


def test_scene_results_refresh_recomputes_each_pair_from_union_or_roi_sources() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    result_rows = source.split("    def _scene_result_rows_from_loaded_remodelling", 1)[1].split("\n    def ", 1)[0]

    assert "source_by_pair = {}" in result_rows
    assert "rank = 0 if self._scene_compartment_is_interactive_source(ctx.get(\"compartment\", \"\")) else 1" in result_rows
    assert "for _pair_key, (_rank, source_path, ctx) in sorted(" in result_rows
    assert "if self._scene_compartment_is_interactive_source(ctx.get(\"compartment\", \"\")):" in result_rows
    assert "preview_inputs = self._get_interactive_preview_inputs(source_path)" in result_rows
    assert "preview, preview_inputs, _source_node, _compartments = self._compute_pair_union_remodelling_preview(" in result_rows
    assert "source_path=source_path" in result_rows


def test_remodelling_scalar_overlay_uses_discrete_label_rendering() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    style = source.split("    def _style_remodelling_scalar_volume", 1)[1].split("\n    def ", 1)[0]

    assert "SetInterpolate(False)" in style
    assert "SetWindowLevel(5.0, 2.5)" in style
    assert "TimelapsedHRpQCT_RemodellingColors" in source
    assert "NamesInitialisedOn" not in source


def test_scene_interactive_recompute_keeps_saved_input_path_with_scalar_display() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    apply_update = source.split("    def _on_apply_interactive_remodelling", 1)[1].split("\n    def ", 1)[0]

    assert 'source_path = str(full_seg.GetAttribute("TimelapsedHRpQCT.RemodellingSourcePath") or "")' in apply_update
    assert "preview_inputs = self._get_interactive_preview_inputs(source_path)" in apply_update
    assert "new_full, _preview = self._create_remodelling_display_from_array(" in apply_update
    assert "source_path=source_path" in apply_update
    assert "slicer.mrmlScene.RemoveNode(full_seg)" in apply_update


def test_remodelling_loadback_uses_slicer_loader_to_preserve_geometry() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    load_remodelling = source.split("    def _load_remodelling_as_segmentation", 1)[1].split("\n    def ", 1)[0]

    assert "ok, remodelling_node = self._load_volume_node(labelmap_path)" in load_remodelling
    assert "sitk.GetArrayFromImage(remodelling_img)" not in load_remodelling
    assert "self._style_remodelling_scalar_volume(remodelling_node" in load_remodelling
    assert "remodelling_node.SetAttribute(\"TimelapsedHRpQCT.RemodellingSourcePath\"" in load_remodelling


def test_interactive_preview_reuses_display_node_geometry() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    apply_update = source.split("    def _on_apply_interactive_remodelling", 1)[1].split("\n    def ", 1)[0]

    assert "geometry_source_node=full_seg" in apply_update
    assert "def _copy_volume_geometry" in source
    create_display = source.split("    def _create_remodelling_display_from_array", 1)[1].split("\n    def ", 1)[0]
    assert "geometry_source_node=None" in create_display
    assert "self._copy_volume_geometry(geometry_source_node, full_seg)" in create_display


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


def test_scene_results_table_is_not_loaded_as_mrml_table_by_default() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    load_table = source.split("    def _load_scene_results_table(self", 1)[1].split("\n    def ", 1)[0]
    assert "if show:" in load_table
    assert "table_node = self._load_scene_results_table_node(rows, plan)" in load_table
    assert load_table.index("if show:") < load_table.index("table_node = self._load_scene_results_table_node")
    assert "Scene result table node is only created when explicitly shown" in load_table


def test_scene_loadback_keeps_scene_nodes_native_by_default() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _apply_scene_baseline_transforms" in source
    assert "SetAndObserveTransformNodeID(transform_node.GetID())" in source
    assert "from-ses-(?P<moving>.+?)_to-ses-" in source
    assert "loaded_transform_nodes" in source
    assert 'SLICER_TIMELAPSED_APPLY_SCENE_TRANSFORMS' in source
    assert "applied_transforms = self._apply_scene_baseline_transforms(plan, loaded_transform_nodes)" in source
    assert "loaded_result_rows = self._load_scene_results_table(plan, show=True, prefer_saved=True)" in source
    load_outputs = source.split("    def _load_scene_run_outputs", 1)[1].split("\n    def ", 1)[0]
    assert "apply_scene_transforms = str(os.environ.get(\"SLICER_TIMELAPSED_APPLY_SCENE_TRANSFORMS\", \"\")" in load_outputs
    assert "if apply_scene_transforms:" in load_outputs


def test_scene_loadback_only_auto_applies_linear_tfm_transforms() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    key_helper = source.split("    def _scene_baseline_transform_key", 1)[1].split("\n    def ", 1)[0]

    assert 'if Path(path).suffix.lower() != ".tfm":' in key_helper
    assert "return None" in key_helper
    assert r"\.(?:tfm|h5)$" not in key_helper


def test_scene_initial_transform_selector_is_linear_only() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert 'self._scene_node_selector(["vtkMRMLLinearTransformNode"])' in source
    assert 'self._scene_node_selector(["vtkMRMLTransformNode"])' not in source
    candidates = source.split("    def _scene_node_candidates", 1)[1].split("\n    def ", 1)[0]
    assert "not self._scene_transform_is_supported_initial_transform(node)" in candidates
    selected_node = source.split("    def _scene_selected_node_id", 1)[1].split("\n    def ", 1)[0]
    assert "not self._scene_transform_is_supported_initial_transform(node)" in selected_node


def test_scene_seed_transform_registry_uses_only_pairwise_transform_files() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    parser = source.split("    def _scene_transform_pair_and_kind_from_path", 1)[1].split("\n    def ", 1)[0]
    seed_body = source.split("    def _seed_scene_transform_registry", 1)[1].split("\n    def ", 1)[0]

    assert "from-ses-(?P<moving>.+?)_to-ses-(?P<fixed>.+?)" in parser
    assert "baseline|final|pairwise" in parser
    assert "str(match.group(\"kind\")).lower()" in parser
    assert "parsed_transform = self._scene_transform_pair_and_kind_from_path(transform_path)" in seed_body
    assert 'if parsed_kind != "pairwise":' in seed_body
    assert "registration reuse needs adjacent pairwise transforms" in seed_body
    assert "moving_session = parsed_moving" in seed_body
    assert "fixed_session = parsed_fixed" in seed_body
    assert "it will only be reused for that pair" in seed_body


def test_scene_discover_selects_adjacent_pairwise_initial_transforms() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    discover = source.split("    def _on_discover_scene_timepoints", 1)[1].split("\n    def ", 1)[0]
    selector = source.split("    def _select_scene_initial_transforms_for_registration_reuse", 1)[1].split("\n    def ", 1)[0]
    ranking = source.split("    def _scene_transform_rank_for_registration_reuse", 1)[1].split("\n    def ", 1)[0]

    assert "self._select_scene_initial_transforms_for_registration_reuse()" in discover
    assert "pair_mode_for_registration" not in selector
    assert "pair_mode = self._current_analysis_pair_mode()" not in selector
    assert "fixed_session = session_ids[row - 1]" in selector
    assert 'if str(kind).lower() == "pairwise"' in selector
    assert 'ranks = {"pairwise": 0}' in ranking


def test_scene_loadback_does_not_load_nonlinear_h5_transforms() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    load_outputs = source.split("    def _load_scene_run_outputs", 1)[1].split("\n    def ", 1)[0]

    assert 'for pattern in ("*.tfm",)' in load_outputs
    assert '"*.h5"' not in load_outputs


def test_scene_mask_loadback_respects_selected_roi_roles() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    load_masks = source.split("    def _load_scene_run_masks", 1)[1].split("\n    def ", 1)[0]

    assert "selected_roles = set(self._scene_requested_mask_roles())" in load_masks
    assert 'selected_roles.update({"regmask", "seg"})' in load_masks
    assert "if str(role) not in selected_roles:" in load_masks
    assert "continue" in load_masks


def test_scene_cleans_generated_nonlinear_transform_nodes_before_discovery_and_loadback() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    cleanup = source.split("    def _remove_scene_run_nonlinear_transform_nodes", 1)[1].split("\n    def ", 1)[0]
    discover = source.split("    def _on_discover_scene_timepoints", 1)[1].split("\n    def ", 1)[0]
    loadback = source.split("    def _load_scene_run_outputs", 1)[1].split("\n    def ", 1)[0]

    assert 'storage_path.suffix.lower() != ".h5"' in cleanup
    assert '"/TimelapsedScene/derivatives/Timelapsed/scene_runs/"' in cleanup
    assert "scene.RemoveNode(node)" in cleanup
    assert "self._remove_scene_run_nonlinear_transform_nodes()" in discover
    assert "self._remove_scene_run_nonlinear_transform_nodes()" in loadback


def test_scene_completion_keeps_pipeline_stages_done_after_loadback() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    finished = source.split("    def _on_finished", 1)[1].split("\n    def ", 1)[0]

    assert "self._adopt_scene_run_as_current_dataset(scene_plan)" in finished
    assert "self._load_scene_run_outputs(scene_plan)" in finished
    assert 'for s in ("dataset", "parse", "registration", "analysis"):' in finished
    assert 'self._set_stage_status(s, "done")' in finished
    assert 'self._set_scene_stage_message("Current step: complete")' in finished


def test_interactive_update_recomputes_display_union_from_available_rois() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    apply_update = source.split("    def _on_apply_interactive_remodelling", 1)[1].split("\n    def ", 1)[0]
    union_update = source.split("    def _compute_pair_union_remodelling_preview", 1)[1].split("\n    def ", 1)[0]

    assert "def _compute_pair_union_remodelling_preview" in source
    assert "def _scene_remodelling_context_matches_pair" in source
    assert "preview, preview_inputs, _source_node, compartments = self._compute_pair_union_remodelling_preview(" in apply_update
    assert "source_path=source_path" in apply_update
    assert "Updating remodelling ROI union" in apply_update
    assert "remodelling ROI union updated" in apply_update
    assert "valid_union |=" in union_update
    assert "self._pair_metric_compartments(preview_inputs)" in union_update
    assert "self._compute_pair_remodelling_preview_from_cached_delta(" in union_update
    assert "compute_pair_remodelling_preview(" not in union_update
    assert "using saved per-ROI analysis rows instead of interactive recomputation" not in apply_update


def test_scene_export_temporarily_detaches_display_transforms() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _detach_scene_export_transforms" in source
    assert "def _restore_scene_export_transforms" in source
    detach = source.split("    def _detach_scene_export_transforms", 1)[1].split("\n    def ", 1)[0]
    export = source.split("    def _export_scene_node", 1)[1].split("\n    def ", 1)[0]
    assert "node.GetTransformNodeID()" in detach
    assert "node.SetAndObserveTransformNodeID(None)" in detach
    assert "self._detach_scene_export_transforms(" in export
    assert "self._restore_scene_export_transforms(detached_transforms)" in export
    assert export.index("self._detach_scene_export_transforms(") < export.index("slicer.util.saveNode(node_to_save")
    assert export.index("self._restore_scene_export_transforms(detached_transforms)") > export.index("slicer.util.saveNode(node_to_save")


def test_scene_export_copies_stored_tfm_transforms_without_mrml_roundtrip() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    export = source.split("    def _export_scene_node", 1)[1].split("\n    def ", 1)[0]

    assert 'node.IsA("vtkMRMLTransformNode")' in export
    assert "self._scene_transform_storage_path(node)" in export
    assert "shutil.copy2(stored_transform_path, path)" in export
    assert 'Path(stored_transform_path).suffix.lower() == ".tfm"' in export
    assert export.index("shutil.copy2(stored_transform_path, path)") < export.index("slicer.util.saveNode(node_to_save")
    assert "def _scene_transform_storage_path" in source


def test_remodelling_loadback_does_not_create_slice_reference_node() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    load_remodelling = source.split("    def _load_remodelling_as_segmentation", 1)[1].split("\n    def ", 1)[0]

    assert "TimelapsedHRpQCT.SliceReference" not in load_remodelling
    assert "reference_node" not in load_remodelling
    assert "ok, remodelling_node = self._load_volume_node(labelmap_path)" in load_remodelling
    assert "self._center_slices_on_node(remodelling_node, fit_to_bounds=True)" in load_remodelling
    assert "setSliceViewerLayers(background=reference_node" not in load_remodelling


def test_timelapsed_scene_mask_policy_is_per_table_cell() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "self._scene_mask_selector(timepoint_index=timepoint_index)" in source
    assert 'self._add_scene_combo_item(selector, "None", "__none__"' in source
    assert 'selector.addItem("Generate", "__generate__")' not in source
    assert '"__generate__"' not in source
    assert "def _scene_selected_mask_policy" in source
    assert "def _scene_requested_mask_roles" in source
    assert "def _scene_segmentation_requested" in source
    assert "def _scene_analysis_compartments" in source
    assert "_decode_scene_mask_choice(value)" in source
    assert "slicer.mrmlScene.GetNodeByID(node_id) is None" in source
    assert 'value == "__none__"' in source
    assert 'return "none"' in source
    assert 'not any(role in masks_cfg["roles"] for role in ("trab", "cort", "trab_roi", "cort_roi"))' in source
    assert 'masks_cfg["inner"] = inner_cfg' in source
    assert "sceneMaskPolicyCombo" not in source


def test_timelapsed_scene_mask_selector_exposes_segmentation_segments() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _scene_segmentation_segment_choices" in source
    assert "_encode_scene_mask_choice(candidate.node_id, segment_id)" in source
    assert "def _scene_selected_mask_segment_id" in source
    assert "self._scene_selected_mask_segment_id(registration_row" in source
    assert '"registration_roi": (timepoint.reg_mask_node_id, timepoint.reg_mask_segment_id' in source
    assert "segment_id=segment_id" in source


def test_scene_segment_role_matching_accepts_readable_segment_names() -> None:
    assert scene_segment_matches_role("Full mask", "", "full")
    assert scene_segment_matches_role("Trabecular mask", "", "trab")
    assert scene_segment_matches_role("Cortical mask", "", "cort")
    assert scene_segment_matches_role("Full mask", "", "full_roi")
    assert scene_segment_matches_role("Trabecular mask", "", "trab_roi")
    assert scene_segment_matches_role("Cortical mask", "", "cort_roi")
    assert scene_segment_matches_role("Bone segmentation", "", "seg")
    assert scene_segment_matches_role("Bone segmentation", "", "segmentation")
    assert scene_segment_matches_role("Full mask", "", "registration_roi")
    assert scene_segment_matches_role("Anything", "trab", "trab")
    assert not scene_segment_matches_role("Full mask", "", "seg")


def test_timelapsed_scene_auto_role_mapping_uses_full_segment_for_registration_roi() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    auto_role_block = source.split("    def _apply_scene_detected_roles_for_timepoint", 1)[1].split("\n    def ", 1)[0]

    assert '"registration_roi", "segmentation", "roi1", "roi2", "roi3"' in auto_role_block
    assert 'lookup_role = "full" if self._normalize_scene_role_name(role) == "registration_roi" else role' in auto_role_block
    assert "self._scene_segment_id_for_node_role(source_node_id, lookup_role)" in auto_role_block
    assert "if not segment_id and lookup_role != role:" in auto_role_block


def test_timelapsed_reload_callbacks_guard_destroyed_qt_combos() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    for function_name in (
        "_on_scene_profile_changed",
        "_sync_scene_profile_from_batch_profile",
        "_selected_processing_subject",
        "_selected_processing_site",
        "_refresh_processing_subjects",
        "_refresh_processing_sites",
        "_selected_config_profile",
    ):
        block = source.split(f"    def {function_name}", 1)[1].split("\n    def ", 1)[0]
        assert "self._qt_object_alive" in block
        assert "RuntimeError, ValueError" in block or function_name.startswith("_refresh_processing")


def test_timelapsed_scene_mask_selector_sets_item_tooltips() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "def _add_scene_combo_item" in source
    assert "qt.Qt.ToolTipRole" in source
    assert "selector.setItemData(index, tooltip_text, qt.Qt.ToolTipRole)" in source
    assert "Segmentation: {candidate.name}" in source
    assert "Segment: {segment_name}" in source


def test_timelapsed_scene_settings_use_existing_masks_only() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    scene_override = source.split("    def _scene_settings_override", 1)[1].split("\n    def ", 1)[0]
    scene_run = source.split("    def _on_run_scene_pipeline", 1)[1].split("\n    def ", 1)[0]

    assert 'masks_cfg["generate"] = False' in scene_override
    assert 'masks_cfg["overwrite"] = False' in scene_override
    assert 'masks_cfg["roles"] = self._scene_requested_mask_roles()' in scene_override
    assert "Scene mask request:" not in scene_run


def test_timelapsed_scene_mask_policy_choices_sync_by_column() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "self._syncing_scene_mask_policy = False" in source
    assert "def _on_scene_mask_policy_changed" in source
    assert 'if value != "__none__"' in source
    assert "other is selector" in source
    assert "other.setCurrentIndex(index)" in source
    add_timepoint = source.split("    def _add_scene_timepoint", 1)[1].split("\n    def ", 1)[0]
    assert "self._refresh_scene_roi_columns()" in add_timepoint
    assert "role_row = self._scene_role_row_index(role)" in add_timepoint
    assert "self._scene_role_row_index(\"initial_transform\")" in add_timepoint


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


def test_timelapsed_scene_plan_separates_registration_and_analysis_masks(tmp_path: Path) -> None:
    plan = build_timelapsed_scene_plan(
        results_root=tmp_path,
        subject_id="SAMPLE001",
        site="tibia",
        timepoints=[
            TimelapsedSceneTimepoint(
                session_id="ses-1",
                image_node_id="v1",
                reg_mask_node_id="reg1",
                reg_mask_policy="node",
            ),
            TimelapsedSceneTimepoint(
                session_id="ses-2",
                image_node_id="v2",
                reg_mask_node_id="reg2",
                reg_mask_policy="node",
            ),
        ],
        rois=[
            TimelapsedSceneRoiSelection(
                role="full",
                node_ids=("full1", "full2"),
                policies=("node", "node"),
            ),
            TimelapsedSceneRoiSelection(
                role="roi_medial",
                node_ids=("medial1", "medial2"),
                policies=("node", "node"),
            ),
        ],
        run_id="scene-test",
    )

    assert plan.timepoints[0].reg_mask_path.name == "sub-SAMPLE001_ses-1_site-tibia_regmask.nii.gz"
    assert plan.rois[0].paths[0].name == "sub-SAMPLE001_ses-1_site-tibia_mask-full.nii.gz"
    assert plan.rois[1].paths[1].name == "sub-SAMPLE001_ses-2_site-tibia_mask-roi_medial.nii.gz"


def test_scene_discovery_ignores_generated_mask_loadback() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate(
                node_id="image-00",
                name="sub-001_ses-00_site-radius_image",
                node_class="vtkMRMLScalarVolumeNode",
            ),
            TimelapsedSceneNodeCandidate(
                node_id="full-00",
                name="sub-001_ses-00_site-radius_mask-full",
                node_class="vtkMRMLLabelMapVolumeNode",
                attributes={"TimelapsedHRpQCT.GeneratedMask": "1"},
            ),
            TimelapsedSceneNodeCandidate(
                node_id="image-04",
                name="sub-001_ses-04_site-radius_image",
                node_class="vtkMRMLScalarVolumeNode",
            ),
        ]
    )

    assert len(discovery.timepoints) == 2
    assert discovery.mask_count == 0
    assert discovery.matched_mask_count == 0
    assert discovery.timepoints[0].full_mask_node_id == ""
    assert discovery.timepoints[0].full_mask_policy == "none"


def test_scene_discovery_ignores_generated_mask_loadback_from_storage_path() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate(
                node_id="image-00",
                name="sub-001_ses-00_site-radius_image",
                node_class="vtkMRMLScalarVolumeNode",
            ),
            TimelapsedSceneNodeCandidate(
                node_id="full-00",
                name="sub-001_ses-00_site-radius_mask-full",
                node_class="vtkMRMLLabelMapVolumeNode",
                attributes={
                    "StorageFileName": (
                        "/tmp/SlicerBoneImagingToolbox/TimelapsedScene/derivatives/Timelapsed/"
                        "scene_runs/run-1/output/derivatives/Timelapse/sub-001/"
                        "ses-00/xct/stacks/sub-001_ses-00_voi-radius_mask-full.nii.gz"
                    )
                },
            ),
            TimelapsedSceneNodeCandidate(
                node_id="image-04",
                name="sub-001_ses-04_site-radius_image",
                node_class="vtkMRMLScalarVolumeNode",
            ),
        ]
    )

    assert len(discovery.timepoints) == 2
    assert discovery.mask_count == 0
    assert discovery.matched_mask_count == 0


def test_scene_discovery_keeps_bone_contouring_masks() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate(
                node_id="image-00",
                name="sub-001_ses-00_site-radiusleft_image",
                node_class="vtkMRMLScalarVolumeNode",
            ),
            TimelapsedSceneNodeCandidate(
                node_id="full-00",
                name="sub-001_ses-00_site-radiusleft_mask-full",
                node_class="vtkMRMLSegmentationNode",
                attributes={"BoneImaging.MaskRoles": "full,trab,cort,seg"},
            ),
        ]
    )

    assert len(discovery.timepoints) == 1
    assert discovery.mask_count == 1
    assert discovery.matched_mask_count == 4
    assert discovery.timepoints[0].full_mask_node_id == "full-00"
    assert discovery.timepoints[0].trab_mask_node_id == "full-00"
    assert discovery.timepoints[0].cort_mask_node_id == "full-00"
    assert discovery.timepoints[0].seg_mask_node_id == "full-00"
    assert discovery.timepoints[0].full_mask_policy == "node"


def test_timelapsed_scene_plan_respects_none_mask_policy(tmp_path: Path) -> None:
    plan = build_timelapsed_scene_plan(
        results_root=tmp_path,
        subject_id="SAMPLE001",
        site="radius",
        timepoints=[
            TimelapsedSceneTimepoint(
                session_id="ses-1",
                image_node_id="v1",
                full_mask_node_id="full1",
                full_mask_policy="node",
                trab_mask_policy="none",
                cort_mask_policy="none",
                seg_mask_node_id="seg1",
                seg_mask_policy="node",
            ),
            TimelapsedSceneTimepoint(
                session_id="ses-2",
                image_node_id="v2",
                full_mask_node_id="full2",
                full_mask_policy="node",
                trab_mask_policy="none",
                cort_mask_policy="none",
                seg_mask_node_id="seg2",
                seg_mask_policy="node",
            ),
        ],
        run_id="abc",
    )

    assert plan.timepoints[0].full_mask_path is not None
    assert plan.timepoints[0].seg_mask_path is not None
    assert plan.timepoints[0].trab_mask_path is None
    assert plan.timepoints[0].cort_mask_path is None


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


def test_timelapsed_scene_run_warns_on_roi_segmentation_overlap_diagnostics_without_blocking() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    ).read_text(encoding="utf-8")

    run_scene = source.split("    def _on_run_scene_pipeline", 1)[1].split("\n    def ", 1)[0]
    assert "self._validate_scene_analysis_roi_overlap(plan)" in run_scene

    validator = source.split("    def _validate_scene_analysis_roi_overlap", 1)[1].split("\n    def ", 1)[0]
    assert "roi_arr & seg_arr" in validator
    assert "does not overlap" in validator
    assert "selected segmentation" in validator
    assert "Check Role Mapping" in validator
    assert "roi_vox=" in validator
    assert "seg_vox=" in validator
    assert "overlap_vox=" in validator
    assert "slicer.util.warningDisplay" in validator
    assert "raise ValueError" not in validator


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


def test_timelapsed_scene_discovery_reuses_scene_run_output_transforms() -> None:
    transform_storage_path = (
        "/tmp/SlicerBoneImagingToolbox/TimelapsedScene/derivatives/Timelapsed/"
        "scene_runs/run-1/output/derivatives/Registration/sub-SAMPLE001/"
        "ses-04/xct/baseline/sub-SAMPLE001_ses-04_voi-radiusleft_stack-01_"
        "from-ses-04_to-ses-00_baseline.tfm"
    )
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate("img00", "sub-SAMPLE001_ses-00_site-radiusleft_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("img04", "sub-SAMPLE001_ses-04_site-radiusleft_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate(
                "tfm04",
                "Loaded transform",
                "vtkMRMLTransformNode",
                {"StorageFileName": transform_storage_path},
            ),
        ]
    )

    assert [timepoint.session_id for timepoint in discovery.timepoints] == ["00", "04"]
    assert discovery.timepoints[1].transform_node_id == "tfm04"


def test_timelapsed_scene_discovery_ignores_nonlinear_h5_initial_transforms() -> None:
    discovery = discover_timelapsed_scene_timepoints(
        [
            TimelapsedSceneNodeCandidate("img00", "sub-SAMPLE001_ses-00_site-radiusleft_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate("img04", "sub-SAMPLE001_ses-04_site-radiusleft_image", "vtkMRMLScalarVolumeNode"),
            TimelapsedSceneNodeCandidate(
                "h5",
                "Loaded nonlinear transform",
                "vtkMRMLTransformNode",
                {
                    "StorageFileName": (
                        "/tmp/scene/output/derivatives/Registration/sub-SAMPLE001/"
                        "ses-04/xct/baseline/sub-SAMPLE001_ses-04_voi-radiusleft_stack-01_"
                        "from-ses-04_to-ses-00_baseline.h5"
                    )
                },
            ),
        ]
    )

    assert [timepoint.session_id for timepoint in discovery.timepoints] == ["00", "04"]
    assert discovery.timepoints[1].transform_node_id == ""


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


def test_timelapsed_scene_discovery_status_separates_mask_nodes_from_role_matches() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    discover = source.split("    def _on_discover_scene_timepoints", 1)[1].split("\n    def ", 1)[0]

    assert "mask node(s)" in discover
    assert "matched mask role(s)" in discover
    assert '{discovery.matched_mask_count}/' not in discover


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
