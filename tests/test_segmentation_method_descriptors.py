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


def test_contour_batch_table_exposes_row_jobs_and_uses_scene_settings() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert 'self.batchRunButton = qt.QPushButton("Run All")' in source
    assert "batch_layout.addWidget(self.batchRunButton)" in source
    assert "button_row.addWidget(self.batchRunButton)" not in source
    assert source.index("batch_layout.addWidget(self.batchSummaryTable)") < source.index(
        "batch_layout.addWidget(self.batchRunButton)"
    )
    assert 'setHorizontalHeaderLabels(["Image", "Subject", "Session", "Site", "Action", "Status"])' in source
    assert 'action = "Load" if self._batchRowOutputs.get(row) else "Run"' in source
    assert "self.batchSummaryTable.setItem(row, 5" in source
    assert "self.batchSummaryTable.setCellWidget(row, 4, button)" in source
    assert "def _resize_batch_table_columns(self):" in source
    assert "def _queue_batch_row(self, row):" in source
    assert "def _process_next_batch_job(self):" in source
    assert "qt.QTimer.singleShot(0, self._process_next_batch_job)" in source
    assert '"Queued"' in source
    assert "self._start_batch_worker(row)" in source
    assert "process = qt.QProcess()" in source
    assert "qt.QProcess(self)" not in source
    assert "def _python_slicer_executable(self):" in source
    assert 'executable.with_name("PythonSlicer")' in source
    assert "def _qbytearray_to_text(self, raw):" in source
    assert "readyReadStandardOutput.connect" in source
    assert "readyReadStandardError.connect" in source
    assert 'environment.insert("PYTHONUNBUFFERED", "1")' in source
    assert "ITK_AUTOLOAD_PATH" in source
    assert "bone_contour_batch_worker.py" in source
    assert "def _batch_worker_finished(self, row, process, *signal_args)" in source
    assert "lambda *signal_args, row=row, process=process" in source
    assert 'self._set_batch_action(row, "Cancel")' in source
    assert "def _cancel_batch_row(self, row):" in source
    assert "self._batchProcess.terminate()" in source
    assert "self._batchProcess.kill()" in source
    assert "self._load_batch_row_outputs(row)" in source
    assert "keep_loaded=False" in source
    assert "self.batchKeepLoadedCheck" not in source
    assert "self.batchOptionsSummaryLabel" in source
    assert "Batch settings:" in source
    assert 'self.batchOutputFormatCombo = qt.QComboBox()' in source
    assert '("Auto", "auto"), ("AIM", "aim"), ("NIfTI", "nifti")' in source
    assert "format={self._combo_label(self.batchOutputFormatCombo)" in source
    assert "self.batchContourSettingsCombo" not in source
    assert "parameters={self._combo_label(self.parameterModeCombo)" in source
    assert "segmentation_method=str(self.segmentationMethodCombo.currentData)" in source
    assert "periosteal_contour_method=str(self.periostealContourCombo.currentData)" in source
    assert "endosteal_contour_method=str(self.endostealContourCombo.currentData)" in source
    assert "output_format=self._batch_output_format()" in source
    assert "params=self._collect_params(site=site, use_site_defaults=self._use_site_preset_params())" in source
    assert "def _find_existing_batch_outputs(self, item):" in source
    assert "def _preferred_existing_batch_outputs(self, item):" in source
    assert "prefer_aim = output_format == \"aim\"" in source
    assert "self._batchRowOutputs[row] = outputs" in source
    assert "Finished {format_label}" in source


def test_contour_batch_can_use_row_site_specific_defaults() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _site_contour_params(self, site):" in source
    assert 'for label, value in [("Auto", "auto"), ("Radius", "radius"), ("Tibia", "tibia"), ("Knee", "knee")]' in source
    assert 'for label, value in [("Preset", "preset"), ("Custom", "custom")]' in source
    assert "self.modalityCombo.currentIndexChanged.connect(self._on_modality_changed)" in source
    assert "self.segmentationMethodCombo.currentIndexChanged.connect(self._on_segmentation_method_changed)" in source
    assert "def _apply_modality_preset(self):" in source
    assert "def _apply_preset_values(self, update_segmentation_method=False):" in source
    preset_start = source.index("    def _apply_preset_values(self, update_segmentation_method=False):")
    preset_end = source.index("    def _apply_modality_preset(self):", preset_start)
    preset_method = source[preset_start:preset_end]
    assert "if update_segmentation_method:" in preset_method
    assert preset_method.index("self._apply_modality_preset()") < preset_method.index("self._apply_segmentation_preset()")
    assert preset_method.index("self._apply_segmentation_preset()") < preset_method.index("self._apply_site_preset()")
    assert 'target_method = "laplace_hamming" if modality == "xct1" else "seg_gauss"' in source
    assert "self._set_combo_by_data(self.segmentationMethodCombo, target_method)" in source
    assert source.index('form.addRow("Input volume", self.volumeSelector)') < source.index('form.addRow("Parameters", self.parameterModeCombo)') < source.index('form.addRow("Modality preset", self.modalityCombo)')
    assert "def _refresh_parameter_mode_ui(self):" in source
    assert "widget.visible = bool(preset_mode)" in source
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
    assert "def _use_site_preset_params(self):" in source
    assert 'return str(self._combo_data(self.parameterModeCombo, "preset")) == "preset"' in source


def test_contour_custom_parameters_can_be_saved_as_user_recipes() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert 'self.loadRecipeButton = qt.QPushButton("Load Recipe")' in source
    assert 'self.saveRecipeButton = qt.QPushButton("Save Recipe")' in source
    assert 'form.addRow("Custom recipe", self.customRecipeRowWidget)' in source
    assert "def _default_recipe_dir(self):" in source
    assert "bone-contour-recipes" in source
    assert "def _save_custom_recipe(self):" in source
    assert "def _load_custom_recipe(self):" in source
    assert '"schema": "bone-contour-recipe-v1"' in source
    assert '"methods": {' in source
    assert '"parameters": self._collect_params(site=site, use_site_defaults=False)' in source
    assert "def _apply_recipe(self, recipe):" in source
    assert "self._set_combo_by_data(self.parameterModeCombo, \"custom\")" in source
    assert "def _apply_params_to_widgets(self, params):" in source
    assert "self.customRecipeRowWidget.visible = not preset_mode" in source
    assert source.index('form.addRow("Custom recipe", self.customRecipeRowWidget)') < source.index(
        "generate_layout.addWidget(self.expertSettingsButton)"
    )


def test_contour_module_does_not_duplicate_setup_update_controls() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "Check toolbox updates" not in source
    assert "updateToolboxButton" not in source
    assert "_check_toolbox_updates" not in source
    assert "run_toolbox_update_dialog" not in source
    assert 'form.addRow("Status", self.pipelineStatusLabel)' in source


def test_contour_batch_unparsed_rows_fall_back_to_selected_site_and_input_folder() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert '"subject": "", "session": "", "site": ""' in source
    assert 'parsed.get("subject") or "Unparsed"' in source
    assert 'parsed.get("session") or "Unparsed"' in source
    assert 'parsed.get("site") or "Use selected"' in source
    assert 'output_root_text = str(self.batchInputRootEdit.currentPath or "").strip()' in source
    assert 'subject = item.get("subject") or "unparsed"' in source
    assert 'site = self._batch_item_site(item)' in source
    assert 'session = item.get("session") or _image_output_stem(image_path)' in source


def test_contour_batch_writes_only_aim_masks_for_aim_inputs() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def _image_output_stem(path):" in source
    assert 'aim_index = upper.find(".AIM")' in source
    assert "def _write_mask_aim_if_supported" in source
    assert 'metadata["source_file"] = str(source_path)' in source
    assert 'metadata["mask_role"] = str(role)' in source
    assert "aim_io.write_aim(" in source
    assert "mask=True" in source
    assert "Output format must be one of: auto, aim, nifti." in source
    assert "write_aim_output = output_format == \"aim\"" in source
    assert "AIM output requires an AIM input" in source
    assert 'output_dir / f"{stem}_mask-{role}.AIM"' in source
    assert "def _write_mask_sidecar(self, mask_path" in source
    assert '"schema": "bone-contour-mask-provenance-v1"' in source
    assert '"algorithm_metadata": dict(metadata or {})' in source
    assert 'metadata["provenance_sidecars"] = sidecars' in source
    assert 'metadata["aim_outputs"] = aim_written' in source
    assert 'metadata["output_format"] = "aim"' in source
    assert "output_prefix=_image_output_stem(image_path)" in source
    assert "if write_aim_output:" in source
    assert "slicer.util.saveNode(label_node, str(out_path))" in source
    assert "Could not write AIM" in source


def test_contour_batch_loads_aim_outputs_back_as_segmentations() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "def read_mask_image_file(self, mask_path):" in source
    assert "aim_io.read_aim(mask_path, scaling=\"native\")" in source
    assert 'slicer.mrmlScene.AddNewNodeByClass(\n                "vtkMRMLSegmentationNode"' in source
    assert '"full": "Full mask"' in source
    assert '"seg": "Bone segmentation"' in source
    assert "self.logic._add_sitk_segment(" in source
    assert "Loaded batch segmentation with masks" in source
    assert "def _center_slices_on_node(self, node_to_center):" in source
    assert "node_to_center.GetRASBounds(bounds)" in source
    assert "slice_node.JumpSliceByOffsetting(cx, cy, cz)" in source
    assert "slice_node.JumpSliceByCentering(cx, cy, cz)" in source
    assert "def _fit_slice_node_to_bounds(self, slice_node, widget, bounds, view_name):" in source
    assert "slice_node.SetFieldOfView(float(target_x), float(target_y), z_fov)" in source
    assert "FitSliceToAll" not in source
    assert "slicer.util.setSliceViewerLayers(label=segmentation_node" not in source
    assert "def _find_loaded_batch_source_volume(self, item):" in source
    assert "source_volume = self._find_loaded_batch_source_volume(item)" in source
    assert "def _set_slicer_volume_geometry_from_sitk_image(node, image):" in source
    assert "lps_to_ras = np.diag([-1.0, -1.0, 1.0])" in source
    assert "node.SetIJKToRASMatrix(matrix)" in source
    assert "_set_slicer_volume_geometry_from_sitk_image(reference_node, mask_image)" in source
    assert "reference_node.SetHideFromEditors(True)" in source
    assert "background_node = source_volume if source_volume is not None else reference_node" in source
    assert "slicer.util.setSliceViewerLayers(background=background_node, fit=False)" in source
    assert "if reference_node is not None and source_volume is not None:" in source
    assert "display_node.SetVisibility(True)" in source
    assert "self._center_slices_on_node(background_node or segmentation_node)" in source
    assert "ScancoIOLogic().import_image(" not in source


def test_contour_batch_worker_is_packaged_with_shared_toolbox_library() -> None:
    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "SlicerBoneImagingToolboxLib/bone_contour_batch_worker.py" in source


def test_contour_batch_worker_writes_mask_provenance_sidecars() -> None:
    source = (ROOT / "SlicerBoneImagingToolboxLib" / "bone_contour_batch_worker.py").read_text(encoding="utf-8")

    assert "def _write_sidecar(mask_path:" in source
    assert '"schema": "bone-contour-mask-provenance-v1"' in source
    assert '"parameters": dict(config.get("params") or {})' in source
    assert '"source_metadata": dict(source_metadata or {})' in source
    assert '"provenance_sidecars": sidecars' in source
    assert "print(\"[worker] generating masks\"" in source


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
