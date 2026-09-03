from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_segmentation_method_descriptors_define_xct2_choices() -> None:
    from SlicerBoneImagingToolboxLib.segmentation_methods import (
        BONE_SEGMENTATION_METHODS,
        ENDOSTEAL_CONTOUR_METHODS,
        PERIOSTEAL_CONTOUR_METHODS,
    )

    assert BONE_SEGMENTATION_METHODS["seg_gauss"].label == "Gaussian"
    assert BONE_SEGMENTATION_METHODS["laplace_hamming"].label == "Laplace-Hamming"
    assert BONE_SEGMENTATION_METHODS["adaptive"].label == "Adaptive"

    assert PERIOSTEAL_CONTOUR_METHODS["standard"].label == "Standard"
    assert PERIOSTEAL_CONTOUR_METHODS["geodesic_fracture"].label == "Geodesic Fracture"
    assert ENDOSTEAL_CONTOUR_METHODS["standard"].label == "Standard"


def test_segmentation_module_strips_legacy_scanner_prefixes_from_method_labels() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _clean_method_label(label):" in source
    assert '"XCT2 "' in source
    assert "self.segmentationMethodCombo.addItem(_clean_method_label(descriptor.label), value)" in source
    assert "self.periostealContourCombo.addItem(_clean_method_label(descriptor.label), value)" in source
    assert "self.endostealContourCombo.addItem(_clean_method_label(descriptor.label), value)" in source


def test_contouring_module_has_custom_icon_and_author_credit() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")
    icon_path = ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "Resources" / "Icons" / "SegmentationHRpQCT.png"

    assert icon_path.is_file()
    assert "parent.icon = qt.QIcon(str(Path(__file__).with_name(\"Resources\") / \"Icons\" / \"SegmentationHRpQCT.png\"))" in source
    assert "Author: Matthias Walle" in source


def test_method_descriptors_do_not_include_external_segmentation_workflows() -> None:
    from SlicerBoneImagingToolboxLib.segmentation_methods import (
        BONE_SEGMENTATION_METHODS,
        ENDOSTEAL_CONTOUR_METHODS,
        PERIOSTEAL_CONTOUR_METHODS,
        method_supports_site,
    )

    external_token = "or" + "mir"
    all_method_ids = (
        *BONE_SEGMENTATION_METHODS,
        *PERIOSTEAL_CONTOUR_METHODS,
        *ENDOSTEAL_CONTOUR_METHODS,
    )
    assert not any(external_token in method_id.lower() for method_id in all_method_ids)
    for method in BONE_SEGMENTATION_METHODS.values():
        for site in ("radius", "tibia", "knee"):
            assert method_supports_site(method, site)


def test_expert_parameter_groups_are_driven_by_selected_algorithms() -> None:
    from SlicerBoneImagingToolboxLib.segmentation_methods import selected_parameter_groups

    groups = selected_parameter_groups(
        bone_method="seg_gauss",
        periosteal_method="geodesic_fracture",
        endosteal_method="none",
    )

    assert groups == {
        "Bone segmentation": ("gaussian_sigma", "trab_threshold", "cort_threshold"),
        "Periosteal contour": ("geodesic_bone_threshold", "fill_holes"),
    }

    groups = selected_parameter_groups(
        bone_method="laplace_hamming",
        periosteal_method="standard",
        endosteal_method="standard",
    )

    assert "laplace_hamming_threshold" in groups["Bone segmentation"]
    assert "periosteal_threshold" in groups["Periosteal contour"]
    assert "fill_holes" in groups["Periosteal contour"]
    assert "endosteal_threshold" in groups["Endosteal contour"]


def test_segmentation_module_uses_method_descriptors_for_dynamic_expert_fields() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "from SlicerBoneImagingToolboxLib.segmentation_methods import" in source
    assert "def _refresh_method_dependent_ui(self):" in source
    assert "selected_parameter_groups(" in source


def test_scene_debug_keyword_is_supported_by_generate_bone_masks() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    signature_start = source.index("    def generate_bone_masks(")
    signature_end = source.index("    ):", signature_start)
    signature = source[signature_start:signature_end]

    assert "debug_output_dir=None" in signature
    assert "debug_output_dir=debug_dir" in source


def test_contouring_module_has_single_generate_workflow_without_scene_batch_tabs() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")
    builder = source[
        source.index("    def _build_segmentation_section(self):")
        : source.index("    def _labelmap_selector(self):")
    ]

    assert "self.toolTabs = qt.QTabWidget()" not in builder
    assert 'addTab(generate_tab, "Scene")' not in builder
    assert "self._build_batch_tab()" not in builder
    assert "self.layout.addWidget(contouring_widget)" in builder
    assert 'self.createButton = qt.QPushButton("Generate")' in builder
    assert 'self.exportProfileButton = qt.QPushButton("Export Profile")' in builder
    assert 'form.addRow("Profile", self.contourProfileCombo)' in builder
    assert 'form.addRow("Output prefix", self.outputPrefixEdit)' not in builder
    assert 'form.addRow("Parameters", self.parameterModeCombo)' not in builder
    assert 'form.addRow("Modality preset", self.modalityCombo)' not in builder
    assert 'form.addRow("Site preset", self.siteCombo)' not in builder


def test_contouring_scene_uses_combined_scanner_site_profiles() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "CONTOUR_PROFILE_PRESETS = (" in source
    assert '"XtremeCT I - Radius", "xct1", "radius", "laplace_hamming", "standard", "standard"' in source
    assert '"XtremeCT II - Radius", "xct2", "radius", "seg_gauss", "standard", "standard"' in source
    assert '"XtremeCT II Geodesic - Radius", "xct2", "radius", "seg_gauss", "geodesic_fracture", "standard"' in source
    assert '"XtremeCT II LH - Radius", "xct2", "radius", "laplace_hamming", "standard", "standard"' in source
    assert '"periosteal_method": periosteal_method' in source
    assert '"endosteal_method": endosteal_method' in source
    assert "def _apply_profile_preset(self):" in source
    assert "def _current_contour_profile(self):" in source
    assert 'default_profile_index = self.contourProfileCombo.findText("XtremeCT II - Radius")' in source


def test_contouring_module_keeps_live_profile_summary_callback() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _update_batch_options_summary(self" in source
    assert source.count("._update_batch_options_summary)") >= 1


def test_contouring_method_selectors_live_inside_expert_sections() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")
    builder = source[
        source.index("    def _build_segmentation_section(self):")
        : source.index("    def _labelmap_selector(self):")
    ]

    assert 'form.addRow("Bone segmentation", self.segmentationMethodCombo)' not in builder
    assert 'form.addRow("Periosteal (outer) contour", self.periostealContourCombo)' not in builder
    assert 'form.addRow("Endosteal (inner) contour", self.endostealContourCombo)' not in builder
    assert 'segmentation_form.addRow("Method", self.segmentationMethodCombo)' in builder
    assert 'periosteal_form.addRow("Method", self.periostealContourCombo)' in builder
    assert 'endosteal_form.addRow("Method", self.endostealContourCombo)' in builder
    assert builder.index('segmentation_form.addRow("Method", self.segmentationMethodCombo)') < builder.index(
        'segmentation_form.addRow("Trab threshold", self.trabThresholdSpin)'
    )
    assert builder.index('periosteal_form.addRow("Method", self.periostealContourCombo)') < builder.index(
        'periosteal_form.addRow("Aligned contour support", self.segmentationAlignedSupportCheck)'
    )
    assert builder.index('endosteal_form.addRow("Method", self.endostealContourCombo)') < builder.index(
        'endosteal_form.addRow("Trab close radius", self.trabCloseSpin)'
    )


def test_manual_contouring_method_changes_do_not_reapply_profile_or_toggle_modes() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _mark_contour_methods_custom(self):" in source
    marker_method = source[
        source.index("    def _mark_contour_methods_custom(self):")
        : source.index("    def _apply_preset_values", source.index("    def _mark_contour_methods_custom(self):"))
    ]
    assert "parameterModeCombo" not in marker_method
    assert "self._set_combo_by_data(self.parameterModeCombo" not in source
    assert "self._suppressMethodCustomSwitch" in source
    assert "self.segmentationMethodCombo.currentIndexChanged.connect(self._on_segmentation_method_changed)" in source
    assert "self.periostealContourCombo.currentIndexChanged.connect(self._on_contour_method_changed)" in source
    assert "self.endostealContourCombo.currentIndexChanged.connect(self._on_contour_method_changed)" in source
    segmentation_handler = source[
        source.index("    def _on_segmentation_method_changed(self):")
        : source.index("    def _on_contour_method_changed(self):")
    ]
    assert "self._mark_contour_methods_custom()" in segmentation_handler
    assert "self._apply_preset_values(" not in segmentation_handler
    profile_apply = source[
        source.index("    def _apply_profile_preset(self):")
        : source.index("    def _apply_modality_preset(self):")
    ]
    assert "self._suppressMethodCustomSwitch = True" in profile_apply
    assert "self._suppressMethodCustomSwitch = previous_suppression" in profile_apply


def test_contouring_profile_export_is_visible_above_generate() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")
    builder = source[
        source.index("    def _build_segmentation_section(self):")
        : source.index("    def _labelmap_selector(self):")
    ]
    assert 'form.addRow("Profiles", self.customRecipeRowWidget)' not in builder
    assert "expert_layout.addWidget(self.customRecipeRowWidget)" in builder
    assert 'self.exportProfileButton = qt.QPushButton("Export Profile")' in builder
    assert 'self.workflowDisplayNameEdit = qt.QLineEdit()' in builder
    assert 'profile_form.addRow("Workflow display name", self.workflowDisplayNameEdit)' in builder
    assert builder.index('endosteal_form.addRow("Endosteal threshold", self.endostealThresholdSpin)') < builder.index(
        'profile_form.addRow("Workflow display name", self.workflowDisplayNameEdit)'
    )
    assert builder.index("generate_layout.addWidget(self.expertSettingsButton)") < builder.index(
        'self.createButton = qt.QPushButton("Generate")'
    )
    assert "self.customRecipeRowWidget.visible = not preset_mode" not in source
    assert "self.customRecipeLabel.visible = not preset_mode" not in source


def test_contouring_scene_uses_input_volume_name_without_output_prefix_field() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")
    builder = source[
        source.index("    def _build_segmentation_section(self):")
        : source.index("    def _labelmap_selector(self):")
    ]
    create_method = source[
        source.index("    def _create_segmentation(self):")
        : source.index("    def _create_geodesic_progress_dialog", source.index("    def _create_segmentation(self):"))
    ]

    assert "self.outputPrefixEdit = qt.QLineEdit()" not in builder
    assert 'form.addRow("Output prefix", self.outputPrefixEdit)' not in builder
    assert "output_prefix=None" in create_method
    assert "self.outputPrefixEdit" not in create_method


def test_contouring_module_does_not_expose_duplicate_batch_runner() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _build_batch_tab(self):" not in source
    assert "self.batchRunButton" not in source
    assert "self.batchSummaryTable" not in source
    assert "def _queue_batch_row(self, row):" not in source
    assert "def _process_next_batch_job(self):" not in source
    assert "bone_contour_batch_worker.py" not in source
    assert "def _batch_worker_finished(self, row, process, *signal_args)" not in source
    assert "def _cancel_batch_row(self, row):" not in source
    assert "self.batchKeepLoadedCheck" not in source


def test_contour_batch_can_use_row_site_specific_defaults() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _site_contour_params(self, site):" in source
    assert 'for label, value in [("Auto", "auto"), ("Radius", "radius"), ("Tibia", "tibia"), ("Knee", "knee")]' in source
    assert 'for label, value in [("Preset", "preset"), ("Custom", "custom")]' not in source
    assert "self.modalityCombo.currentIndexChanged.connect(self._on_modality_changed)" in source
    assert "self.segmentationMethodCombo.currentIndexChanged.connect(self._on_segmentation_method_changed)" in source
    assert "def _apply_modality_preset(self):" in source
    assert "def _apply_preset_values(self, *args, update_segmentation_method=False):" in source
    preset_start = source.index("    def _apply_preset_values(self, *args, update_segmentation_method=False):")
    preset_end = source.index("    def _apply_modality_preset(self):", preset_start)
    preset_method = source[preset_start:preset_end]
    assert preset_method.index("self._apply_profile_preset()") < preset_method.index("self._apply_segmentation_preset()")
    assert preset_method.index("self._apply_segmentation_preset()") < preset_method.index("self._apply_site_preset()")
    assert 'target_method = "laplace_hamming" if modality == "xct1" else "seg_gauss"' in source
    assert "self._set_combo_by_data(self.segmentationMethodCombo, target_method)" in source
    assert source.index('form.addRow("Input volume", self.volumeSelector)') < source.index('form.addRow("Profile", self.contourProfileCombo)')
    assert 'form.addRow("Parameters", self.parameterModeCombo)' not in source
    assert 'form.addRow("Modality preset", self.modalityCombo)' not in source
    assert 'form.addRow("Site preset", self.siteCombo)' not in source
    assert "def _refresh_parameter_mode_ui(self):" in source
    assert "widget.visible = bool(preset_mode)" not in source
    assert "self.expertSettingsButton = ctk.ctkCollapsibleButton()" in source
    assert "self.expertSettingsButton.collapsed = True" in source
    assert "self.expertSettingsButton.collapsed = False" in source
    assert "self.openEditorCheck" not in source
    assert "Open Segment Editor" not in source
    assert "open_segment_editor=False" in source
    assert 'self.createButton = qt.QPushButton("Generate")' in source
    assert "background:#1f6feb" in source
    assert source.index("generate_layout.addWidget(self.expertSettingsButton)") < source.index(
        'self.createButton = qt.QPushButton("Generate")'
    )
    assert "form.addRow(self.createButton)" not in source
    assert "generate_layout.addWidget(self.createButton)" in source
    assert source.index("generate_layout.addWidget(self.expertSettingsButton)") < source.index(
        "generate_layout.addWidget(self.createButton)"
    )
    assert 'site_defaults = SITE_PRESETS.get(str(site), SITE_PRESETS["radius"])' in source
    assert 'self.outerKernelSpin.value = 12 if modality == "xct1" else int(outer["periosteal_kernelsize"])' in source
    assert 'site = self._selected_site(volume_node=self.volumeSelector.currentNode(), strict=False)' in source
    assert 'if site not in SITE_PRESETS:\n            site = "radius"' in source
    assert 'outer["periosteal_kernelsize"] = 12' in source
    assert 'outer["periosteal_open_radius"] = 1' in source
    assert "def _lh_threshold(self, site, modality):" in source
    assert 'if str(modality) == "xct1" and str(site) in {"radius", "tibia"}:' in source
    assert "return 15000.0" in source
    assert "lh_threshold = self._lh_threshold(selected_site, modality) if use_site_defaults else float(self.lhThresholdSpin.value)" in source
    assert "def _lh_threshold(self, site, modality):" in source
    assert 'if str(modality) == "xct1" and str(site) in {"radius", "tibia"}:' in source
    assert "return 15000.0" in source
    assert "lh_threshold = self._lh_threshold(selected_site, modality) if use_site_defaults else float(self.lhThresholdSpin.value)" in source
    assert '"use_adaptive_threshold": bool(outer["use_adaptive_threshold"])' in source
    assert '"use_adaptive_threshold": bool(inner["use_adaptive_threshold"])' in source
    assert '"use_adaptive_threshold": False' in source
    assert "self.segmentationAlignedSupportCheck.checked = True" in source
    assert "def _collect_params(self, site=None, use_site_defaults=False):" in source
    assert "if use_site_defaults and site in SITE_PRESETS:" in source
    assert "params.update(self._site_contour_params(site))" in source


def test_contour_custom_parameters_can_be_saved_as_user_profiles() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert 'self.loadRecipeButton = qt.QPushButton("Load Profile")' in source
    assert 'self.exportProfileButton = qt.QPushButton("Export Profile")' in source
    assert 'self.deleteProfileButton = qt.QPushButton("Delete Profile")' in source
    assert 'form.addRow("Profiles", self.customRecipeRowWidget)' not in source
    assert "expert_layout.addWidget(self.customRecipeRowWidget)" in source
    assert 'self.workflowDisplayNameEdit = qt.QLineEdit()' in source
    assert '"display_name": display_name' in source
    assert 'display_name = str(self.workflowDisplayNameEdit.text or "").strip()' in source
    assert "def _default_recipe_dir(self):" in source
    assert 'tool_profile_dir("bone-contouring")' in source
    assert "def _save_custom_recipe(self):" in source
    assert "def _load_custom_recipe(self):" in source
    assert "def _delete_selected_custom_profile(self):" in source
    assert '"user_profile_path"' in source
    assert "path.unlink()" in source
    assert "delete_profile(" in source
    assert "Built-in profiles cannot be deleted" in source
    assert "def _populate_contour_profile_combo(self):" in source
    assert "def _add_user_contour_profiles(self):" in source
    assert 'list_profiles("bone-contouring")' in source
    assert "load_profile_payload(record)" in source
    assert '"user_profile_path": str(record.path)' in source
    assert "if profile.get(\"schema\") == \"bone-contour-recipe-v1\":" in source
    assert "self._apply_recipe(profile)" in source
    assert "return\n        self._apply_profile_preset()" in source
    assert '"schema": "bone-contour-recipe-v1"' in source
    assert '"methods": {' in source
    assert '"parameters": self._collect_params(site=site, use_site_defaults=False)' in source
    assert "def _apply_recipe(self, recipe):" in source
    assert "self._set_combo_by_data(self.parameterModeCombo, \"custom\")" not in source
    assert "def _apply_params_to_widgets(self, params):" in source
    assert "self.customRecipeRowWidget.visible = not preset_mode" not in source
    assert source.index('endosteal_form.addRow("Endosteal threshold", self.endostealThresholdSpin)') < source.index(
        "expert_layout.addWidget(self.customRecipeRowWidget)"
    )


def test_contour_default_profile_is_applied_after_expert_controls_exist() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")
    builder = source[
        source.index("    def _build_segmentation_section(self):")
        : source.index("    def _labelmap_selector(self):")
    ]

    assert 'default_profile_index = self.contourProfileCombo.findText("XtremeCT II - Radius")' in builder
    assert builder.index("self.gaussSigmaSpin = self._double_spin") < builder.index(
        'default_profile_index = self.contourProfileCombo.findText("XtremeCT II - Radius")'
    )
    assert "self._apply_preset_values(update_segmentation_method=False)" in builder


def test_contour_module_does_not_duplicate_setup_update_controls() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "Check toolbox updates" not in source
    assert "updateToolboxButton" not in source
    assert "_check_toolbox_updates" not in source
    assert "run_toolbox_update_dialog" not in source
    assert 'form.addRow("Status", self.pipelineStatusLabel)' in source


def test_contouring_module_leaves_batch_discovery_to_batch_processor() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _build_batch_tab(self):" not in source
    assert "def _discover_batch_images(self):" not in source
    assert "def _batch_item_output_dir(self, item, output_root=None):" not in source
    assert "def _queue_batch_row(self, row):" not in source
    assert "def _process_next_batch_job(self):" not in source
    assert "self.batchSummaryTable" not in source
    assert "self.batchRunButton" not in source
    assert "bone_contour_batch_worker.py" not in source


def test_contouring_scene_writes_aim_masks_with_provenance() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _image_output_stem(path):" in source
    assert 'aim_index = upper.find(".AIM")' in source
    assert "def _write_mask_aim_if_supported" in source
    assert 'metadata["source_file"] = str(source_path)' in source
    assert 'metadata["contour_role"] = str(role)' in source
    assert 'metadata["content_type"] = str(content_type)' in source
    assert "aim_io.write_aim(" in source
    assert 'mask=content_type == "mask"' in source
    assert "Output format must be one of: auto, aim, nifti." in source
    assert "write_aim_output = output_format == \"aim\"" in source
    assert "AIM output requires an AIM input" in source
    assert "def _write_mask_sidecar(self, mask_path" in source
    assert '"schema": "bone-contour-mask-provenance-v1"' in source
    assert '"algorithm_metadata": dict(metadata or {})' in source
    assert 'metadata["provenance_sidecars"] = sidecars' in source
    assert 'metadata["aim_outputs"] = aim_written' in source
    assert 'metadata["output_format"] = "aim"' in source
    assert "output_prefix=output_prefix" in source
    assert "if write_aim_output:" in source
    assert "slicer.util.saveNode(label_node, str(out_path))" in source
    assert "Could not write AIM" in source


def test_obsolete_contour_batch_worker_is_not_packaged_with_shared_toolbox_library() -> None:
    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "SlicerBoneImagingToolboxLib/bone_contour_batch_worker.py" not in source
    assert not (ROOT / "SlicerBoneImagingToolboxLib" / "bone_contour_batch_worker.py").exists()


def test_laplace_hamming_batch_can_recover_aim_source_from_storage_or_path() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "storage_node = volume_node.GetStorageNode()" in source
    assert "file_name = storage_node.GetFileName()" in source
    assert 'if file_name and ".aim" in str(file_name).lower()' in source
    assert "volume_node.SetAttribute(AIM_SOURCE_ATTRIBUTE, str(image_path))" in source


def test_scene_laplace_hamming_prefers_original_aim_source_over_density_reconstruction() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")
    method = source[
        source.index("    def _laplace_hamming_support_image")
        : source.index("    def _sitk_to_labelmap", source.index("    def _laplace_hamming_support_image"))
    ]

    assert method.index("source_path = self._volume_source_aim_path(volume_node)") < method.index(
        "return self._density_image_to_laplace_hamming_native"
    )
    assert '"py_aimio_native_int16"' in method
    assert '"imported_density_to_native_int16"' in method


def test_scene_generation_prefers_source_aim_density_for_processing_image() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")
    method = source[
        source.index("    def _volume_to_sitk")
        : source.index("    def _node_aim_metadata", source.index("    def _volume_to_sitk"))
    ]

    assert "source_path = self._volume_source_aim_path(volume_node)" in method
    assert 'aim_io.read_aim(source_path, scaling="density")' in method
    assert "if image.GetSize() == selected_image.GetSize():" in method
    assert "image.CopyInformation(selected_image)" in method
    assert 'self._lastProcessingImageReader = "py_aimio_density"' in method
    assert 'self._lastProcessingImageReader = "selected_slicer_volume"' in method


def test_scene_generation_forces_local_bone_contouring_import_when_available() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _import_bone_contouring(self):" in source
    assert "for name in list(sys.modules):" in source
    assert 'if name == "bone_contouring" or name.startswith("bone_contouring."):' in source
    assert "del sys.modules[name]" in source
    assert "module_path = Path(getattr(module, \"__file__\", \"\")).resolve()" in source
    assert "self._import_bone_contouring()" in source


def test_scene_generation_sets_stable_segment_role_colors() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "SEGMENT_COLORS = {" in source
    assert '"full": (0.2, 0.8, 0.25)' in source
    assert '"trab": (0.0, 0.75, 1.0)' in source
    assert '"cort": (1.0, 0.55, 0.1)' in source
    assert '"seg": (1.0, 0.95, 0.3)' in source
    assert "segment.SetColor(color[0], color[1], color[2])" in source


def test_contour_generation_writes_fea_material_labelmap_from_scene_batch_path() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert 'generated.metadata["emitted_label_roles"] = ["fea-materials"]' in source
    assert 'outputs["fea-materials"] = self._sitk_to_labelmap(' in source
    assert "binary=False" in source
    assert 'output_dir / f"{stem}_desc-{role}_label.AIM"' in source
    assert 'output_dir / f"{stem}_desc-{role}_label.nii.gz"' in source
    assert "content_type=\"label\"" in source


def test_contour_custom_profiles_use_shared_profile_registry() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert 'self.loadRecipeButton = qt.QPushButton("Load Profile")' in source
    assert 'self.exportProfileButton = qt.QPushButton("Export Profile")' in source
    assert 'from bone_imaging_derivatives import register_profile_asset, tool_profile_dir' in source
    assert 'tool_profile_dir("bone-contouring")' in source
    assert "register_profile_asset(" in source
    assert '"bone-contouring",' in source
    assert '"kind": "contour-recipe"' in source
    assert '"display_name": display_name' in source


def test_contouring_scene_generation_has_site_preset_param_policy() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _use_site_preset_params(self):" in source
    assert "strict=self._use_site_preset_params()" in source
    assert "use_site_defaults=self._use_site_preset_params()" in source


def test_xct2_gaussian_standard_contours_take_full_compartment_generation_path() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    standard_path_start = source.index(
        'if (\n            periosteal_contour_method == "standard"'
    )
    standard_path_end = source.index("        else:", standard_path_start)
    standard_path = source[standard_path_start:standard_path_end]

    assert 'endosteal_contour_method == "standard"' in standard_path
    assert 'segmentation_method != "none"' in standard_path
    assert "generate_masks_from_image(" in standard_path
    assert "generated.metadata[\"emitted_roles\"]" in source
    assert "emitted_roles = metadata.get(\"emitted_roles\", [])" in source


def test_xct2_gaussian_with_no_contour_outputs_uses_global_trab_threshold() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "internal_compartment_only" not in source
    assert "global_threshold_without_compartments = (" in source
    assert 'segmentation_method == "seg_gauss"' in source
    assert "not compartment_split_requested" in source
    assert "seg_xyz = (segmentation_image_xyz >= trab_threshold) & full_xyz" in source
    assert "No cortical mask was provided; Gaussian segmentation used the trabecular threshold" in source


def test_slicer_extension_has_no_external_segmentation_workflow_references() -> None:
    token = "or" + "mir"
    paths = [
        ROOT / "SlicerBoneImagingToolboxLib",
        ROOT / "HRpQCTTools",
        ROOT / "README.md",
        ROOT / "docs",
    ]
    scanned_suffixes = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".cmake"}
    matches = []
    for base in paths:
        candidates = [base] if base.is_file() else base.rglob("*")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in scanned_suffixes:
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            if token in text.lower():
                matches.append(str(candidate.relative_to(ROOT)))

    assert matches == []
