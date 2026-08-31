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
    assert 'self.sceneInterruptButton = qt.QPushButton("✕ Cancel")' in source
    assert 'self.sceneExportCsvButton = qt.QPushButton("Export CSV")' in source
    assert "self.sceneInterruptButton.clicked.connect(self._on_cancel_run)" in source
    assert "self.sceneExportCsvButton.clicked.connect(self._on_export_scene_comparison_csv)" in source
    assert "self._style_primary_run_button(self.runTimelapseBtn)" in source
    assert "self._style_primary_run_button(self.sceneRunButton)" in source
    assert "button.setMinimumHeight(34)" in source
    assert "QPushButton { background:#1f6feb; color:white;" in source
    assert "actions.addWidget(self.sceneRunButton)" not in source
    assert "sceneActionLayout.addWidget(self.sceneRunButton)" in source
    assert "sceneSecondaryActionLayout.addWidget(self.sceneExportCsvButton)" in source
    assert "sceneSecondaryActionLayout.addWidget(self.sceneInterruptButton)" in source
    assert "self.sceneInterruptButton.enabled = running" in source
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
    assert "def _scene_selected_mask_policy" in source
    assert "def _scene_mask_generation_requested" in source
    assert "def _scene_segmentation_generation_requested" in source
    assert "def _scene_settings_override" in source
    assert 'masks_cfg["roles"] = self._scene_requested_mask_roles()' in source
    assert 'masks_cfg["generate_segmentation"] = self._scene_segmentation_generation_requested()' in source
    assert 'inner_cfg["contour_method"] = "none"' in source
    assert 'analysis_cfg["compartments"] = self._scene_analysis_compartments()' in source
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
    assert '"overwrite": not bool(getattr(self, "doNotGenerateMasksCheck", None) and self.doNotGenerateMasksCheck.checked)' in source
    assert 'masks_cfg["overwrite"] = scene_generates_masks' in source
    assert "TimelapsedHRpQCT.GeneratedMask" in source
    assert 'qt.QGroupBox("Interactive Preview")' not in source
    assert '"Auto update"' not in source
    assert '"Update remodelling image"' not in source
    assert "analysisSectionBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "settingsBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)" in source
    assert "self.layout.addWidget(analysisSectionBox)" in source
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


def test_timelapsed_batch_series_summary_is_parented_and_collapsed() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert 'analysisSectionBox.text = "Analysis Options"' in source
    assert "analysisSectionBox.collapsed = False" in source
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


def test_timelapsed_mask_settings_are_context_aware() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    build_ui = source.split("    def _build_ui", 1)[1].split("\n    def _style_primary_run_button", 1)[0]
    mask_method_body = source.split("    def _on_mask_method_changed", 1)[1].split("\n    def ", 1)[0]
    contour_method_body = source.split("    def _on_periosteal_contour_method_changed", 1)[1].split("\n    def ", 1)[0]
    override_body = source.split("    def _settings_override", 1)[1].split("\n    def ", 1)[0]

    assert 'discoveryBox.text = "Discovery / Import"' in build_ui
    assert "self.batchLayout.addWidget(discoveryBox)" in build_ui
    assert "discoveryLayout.addRow(_label(\"Copy raw inputs\"" in build_ui
    assert "discoveryLayout.addRow(_label(\"Restructure raw inputs\"" in build_ui
    assert "discoveryLayout.addRow(_label(\"Parse mode\"" in build_ui
    assert "discoveryLayout.addRow(_label(\"Storage mode\"" in build_ui
    assert "maskForm.addRow(_label(\"Copy raw inputs\"" not in build_ui
    assert "maskForm.addRow(_label(\"Storage mode\"" not in build_ui
    assert 'maskForm.addRow(_label("Bone segmentation method"' in build_ui
    assert 'maskForm.addRow(_label("Mask method"' not in build_ui
    assert 'self.maskSegmentationSectionLabel = qt.QLabel("<b>Bone Segmentation</b>")' in build_ui
    assert 'self.maskPeriostealSectionLabel = qt.QLabel("<b>Full / Periosteal Contour</b>")' in build_ui
    assert 'self.maskEndostealSectionLabel = qt.QLabel("<b>Endosteal / Trab-Cort Contour</b>")' in build_ui
    assert build_ui.find('maskForm.addRow(_label("Bone segmentation method"') < build_ui.find("maskForm.addRow(self.maskLowLabel")
    assert "self.maskEndostealContour" in build_ui
    assert "self.maskEndostealThreshold" in build_ui
    assert "self.maskEndostealKernel" in build_ui
    periosteal_row = build_ui.find('"Full/periosteal contour"')
    endosteal_row = build_ui.find('"Endosteal/trab-cort contour"')
    assert build_ui.find("maskForm.addRow(self.maskLaplaceEpsilonLabel") < periosteal_row
    assert periosteal_row < build_ui.find("maskForm.addRow(self.maskContourSupportThresholdLabel")
    assert build_ui.find("maskForm.addRow(self.maskOuterKernelLabel") < endosteal_row
    assert endosteal_row < build_ui.find("maskForm.addRow(self.maskEndostealThresholdLabel")
    assert "label.visible = is_lh" in mask_method_body
    assert 'self.maskSigmaLabel.visible = method == "seg_gauss"' in mask_method_body
    assert "self.maskContourSigmaLabel.visible = method" not in mask_method_body
    assert "self.maskContourSupportThresholdLabel.visible = method" not in mask_method_body
    assert 'self.maskLowLabel.text = "Contour support threshold"' not in mask_method_body
    assert "is_geodesic = method == \"geodesic_fracture\"" in contour_method_body
    assert "endosteal_method = str(self.maskEndostealContour.currentData" in contour_method_body
    assert "contour_support_visible = any_standard_contour" in contour_method_body
    assert 'method_name in {"seg_gauss", "laplace_hamming"}' not in contour_method_body
    assert "label.visible = not is_geodesic" in contour_method_body
    assert "self.maskGeodesicThresholdLabel.visible = is_geodesic" in contour_method_body
    assert "self.maskEndostealThresholdLabel.visible = endosteal_method != \"none\"" in contour_method_body
    assert "self.maskEndostealKernelLabel.visible = endosteal_method != \"none\"" in contour_method_body
    apply_profile_body = source.split("    def _apply_config_dict_to_controls", 1)[1].split("\n    def ", 1)[0]
    assert "self._on_mask_method_changed(self.maskMethod.currentText)" in apply_profile_body
    assert '"gaussian_sigma": float(self.maskSigma.value)' in override_body
    assert '"gaussian_sigma": float(self.maskContourSigma.value)' in override_body
    assert '"periosteal_threshold": contour_support_threshold' in override_body
    assert '"endosteal_threshold": float(self.maskEndostealThreshold.value)' in override_body
    assert '"endosteal_kernelsize": int(self.maskEndostealKernel.value)' in override_body
    assert '"laplace_hamming_threshold": float(self.maskLaplaceThreshold.value)' in override_body
    assert '"laplace_hamming_low_pass_cutoff": float(self.maskLaplaceLowPass.value)' in override_body
    assert '"laplace_hamming_high_pass_cutoff": float(self.maskLaplaceHighPass.value)' in override_body
    assert '"laplace_hamming_epsilon": float(self.maskLaplaceEpsilon.value)' in override_body
    assert '"contour_method": endosteal_contour_method' in override_body
    assert '"cort_threshold": float(getattr(self, "_lh_cort_support_threshold", 450.0))' in override_body


def test_timelapsed_scene_hides_batch_skip_mask_generation_control() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    mode_changed = source.split("    def _on_timelapsed_mode_changed", 1)[1].split("\n    def ", 1)[0]

    assert "self.doNotGenerateMasksLabel" in source
    assert "self.doNotGenerateMasksCheck.visible = not scene_mode" in mode_changed
    assert "self.doNotGenerateMasksLabel.visible = not scene_mode" in mode_changed


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

    assert 'sceneInputBox = qt.QGroupBox("Scene Input")' in scene_ui
    assert 'sceneTimepointBox = qt.QGroupBox("Timepoints")' in scene_ui
    assert 'sceneActionBox = qt.QGroupBox("Pipeline")' in scene_ui
    assert 'self.sceneComparisonBox = qt.QGroupBox("Current Comparisons")' in scene_ui
    assert "self.sceneComparisonTable = qt.QTableWidget()" in scene_ui
    assert '["Pair", "Mask", "FV/BV", "RV/BV", "AV/BV", "NV/BV"]' in scene_ui
    assert 'sceneWorkspaceBox.text = "Processing Workspace"' in scene_ui
    assert "sceneWorkspaceBox.collapsed = True" in scene_ui
    assert 'sceneAdvancedBox.text = "Advanced Settings"' not in scene_ui
    assert 'Processing workspace' in scene_ui
    assert "form.addRow(_label(\"Results folder\"" not in scene_ui
    assert "layout.addWidget(self.sceneComparisonBox)" in scene_ui
    assert "def _set_scene_comparison_rows" in source
    assert "self._set_scene_comparison_rows(rows)" in source
    assert "def _load_scene_results_table(self, plan, *, show=False, prefer_saved=False):" in source


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


def test_analysis_options_are_expanded_by_default() -> None:
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
    assert "analysisSectionBox.collapsed = False" in analysis_setup


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
    assert "self._center_slices_on_node(node, fit_to_bounds=True)" in source
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
    assert "def _scene_selected_mask_policy" in source
    assert "def _scene_requested_mask_roles" in source
    assert "def _scene_segmentation_requested" in source
    assert "def _scene_segmentation_generation_requested" in source
    assert "def _scene_analysis_compartments" in source
    assert "slicer.mrmlScene.GetNodeByID(value) is None" in source
    assert 'value == "__none__"' in source
    assert 'return "none"' in source
    assert 'not any(role in masks_cfg["roles"] for role in ("trab", "cort"))' in source
    assert 'masks_cfg["inner"] = inner_cfg' in source
    assert "sceneMaskPolicyCombo" not in source


def test_timelapsed_scene_mask_generation_flag_follows_table_generate_policy() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "HRpQCTTools"
        / "TimelapsedHRpQCT"
        / "TimelapsedHRpQCT.py"
    )
    source = module_path.read_text(encoding="utf-8")
    scene_override = source.split("    def _scene_settings_override", 1)[1].split("\n    def ", 1)[0]
    scene_run = source.split("    def _on_run_scene_pipeline", 1)[1].split("\n    def ", 1)[0]

    assert "scene_generates_masks = self._scene_mask_generation_requested()" in scene_override
    assert 'masks_cfg["generate"] = scene_generates_masks' in scene_override
    assert 'masks_cfg["overwrite"] = scene_generates_masks' in scene_override
    assert 'masks_cfg["roles"] = self._scene_requested_mask_roles()' in scene_override
    assert "Scene mask request:" in scene_run
    assert "outer_kernel=" in scene_run
    assert "outer_open=" in scene_run


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
    assert 'if value not in {"__generate__", "__none__"}' in source
    assert "other is selector" in source
    assert "other.setCurrentIndex(index)" in source
    add_timepoint = source.split("    def _add_scene_timepoint", 1)[1].split("\n    def ", 1)[0]
    assert "selector.currentIndexChanged.connect(" in add_timepoint
    assert "self._on_scene_mask_policy_changed(selector, column)" in add_timepoint


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
    assert discovery.timepoints[0].full_mask_policy == "generate"


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
                        "scene_runs/run-1/output/derivatives/TimelapsedHRpQCT/sub-001/"
                        "site-radius/ses-00/stacks/sub-001_ses-00_site-radius_mask-full.nii.gz"
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


def test_timelapsed_scene_plan_respects_none_mask_policy(tmp_path: Path) -> None:
    plan = build_timelapsed_scene_plan(
        results_root=tmp_path,
        subject_id="SAMPLE001",
        site="radius",
        timepoints=[
            TimelapsedSceneTimepoint(
                session_id="ses-1",
                image_node_id="v1",
                full_mask_policy="generate",
                trab_mask_policy="none",
                cort_mask_policy="none",
                seg_mask_policy="generate",
            ),
            TimelapsedSceneTimepoint(
                session_id="ses-2",
                image_node_id="v2",
                full_mask_policy="generate",
                trab_mask_policy="none",
                cort_mask_policy="none",
                seg_mask_policy="generate",
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
