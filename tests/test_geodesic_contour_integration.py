from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "HRpQCTSegmentation" / "HRpQCTSegmentation.py"
PIPELINE_MODULE = Path(__file__).resolve().parents[1] / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py"


def test_segmentation_module_exposes_geodesic_method_for_local_testing():
    source = MODULE.read_text()

    assert "geodesic_fracture" in source
    assert "hrpqct-geodesic-contour" in source
    assert "install_or_update_geodesic_contour" in source
    assert "install_or_update_contouring_dependencies" in source
    assert "Install / Update contouring dependencies" in source
    assert "self.installButton.clicked.connect(self._install_contouring_dependencies)" in source
    assert "pip_install(\"edt>=2.4\")" in source
    assert "--no-deps -e" in source
    assert "importlib.invalidate_caches()" in source
    assert "sys.modules.pop(\"hrpqct_geodesic_contour\"" in source
    assert "missing required API arguments" in source
    assert "from hrpqct_geodesic_contour import contour" in source
    assert "np.transpose(arr_zyx, (2, 1, 0))" in source
    assert "progress_callback=progress_callback" in source
    assert "cancel_callback=cancel_callback" in source
    assert "fill_holes=bool(geodesic_params.get(\"fill_holes\", True))" in source
    assert "qt.QProgressDialog" in source
    assert "_geodesic_cancel_requested" in source
    assert "ContourCancelledError" not in source
    assert "installGeodesicButton" not in source
    assert "Install local geodesic contour" not in source


def test_segmentation_module_splits_segmentation_and_contour_choices():
    source = MODULE.read_text()

    assert "form.addRow(\"Bone segmentation\", self.segmentationMethodCombo)" in source
    assert "form.addRow(\"Periosteal (outer) contour\", self.periostealContourCombo)" in source
    assert "form.addRow(\"Endosteal (inner) contour\", self.endostealContourCombo)" in source
    assert "SEGMENTATION_METHODS = {\"seg_gauss\", \"adaptive\", \"laplace_hamming\", \"none\"}" in source
    assert "PERIOSTEAL_CONTOUR_METHODS = {\"standard\", \"geodesic_fracture\", \"none\"}" in source
    assert "ENDOSTEAL_CONTOUR_METHODS = {\"standard\", \"none\"}" in source
    assert "segmentation_method=segmentation_method" in source
    assert "periosteal_contour_method=periosteal_method" in source
    assert "endosteal_contour_method=endosteal_method" in source
    assert "generated.metadata[\"periosteal_contour_method\"] = periosteal_contour_method" in source
    assert "generated.metadata[\"endosteal_contour_method\"] = endosteal_contour_method" in source
    assert "from dataclasses import asdict" in source
    assert "outer_options = asdict(contour_params.outer)" in source
    assert "inner_options = asdict(contour_params.inner)" in source
    assert "options=outer_options" in source
    assert "options=inner_options" in source
    assert "form.addRow(\"Segmentation method\", self.methodCombo)" not in source


def test_segmentation_module_skips_compartments_without_endosteal_split():
    source = MODULE.read_text()

    assert "compartment_split_requested = endosteal_contour_method == \"standard\"" in source
    assert "compartment_split_generated = compartment_split_requested" in source
    assert "generated.metadata[\"compartment_split_generated\"]" in source
    assert "generated.metadata[\"compartment_split_reason\"] = \"endosteal_contour_method_none\"" in source
    assert "generated.trab = numpy_xyz_to_sitk_binary(empty_xyz, image)" in source
    assert "generated.cort = numpy_xyz_to_sitk_binary(empty_xyz, image)" in source
    assert "output_specs = [spec for spec in output_specs if spec[0] in {\"full\", \"seg\"}]" in source
    assert "generated.metadata[\"emitted_roles\"]" in source
    assert "cort_xyz = _ensure_bool(full_xyz)" not in source


def test_segmentation_module_does_not_emit_full_mask_without_outer_contour():
    source = MODULE.read_text()

    assert "periosteal_contour_generated = periosteal_contour_method != \"none\"" in source
    assert "full_xyz = np.ones_like(image_xyz, dtype=bool)" in source
    assert "generated.metadata[\"periosteal_contour_generated\"]" in source
    assert "generated.metadata[\"periosteal_contour_reason\"] = \"periosteal_contour_method_none\"" in source
    assert "output_specs = [spec for spec in output_specs if spec[0] != \"full\"]" in source
    assert "generated.metadata[\"voxel_counts\"][\"full\"] = 0" in source


def test_segmentation_module_defaults_to_segmentation_node_only():
    source = MODULE.read_text()

    assert "create_labelmaps=False" in source
    assert "self.createLabelmapsCheck" not in source
    assert "create_labelmaps=False" in source
    assert "label_text = \"\"" in source
    assert "returnNode=True" not in source
    assert "slicer.util.updateSegmentBinaryLabelmapFromArray(" in source
    assert "_remove_empty_duplicate_segmentation_nodes(segmentation_node)" in source
    assert "node.IsA(\"vtkMRMLSegmentationNode\")" in source
    assert "node.GetSegmentation().GetNumberOfSegments() == 0" in source
    assert "display_node.SetOpacity2DFill(0.85)" in source
    assert "display_node.SetAllSegmentsVisibility2DFill(True)" in source
    assert "display_node.SetAllSegmentsOpacity2DFill(0.85)" in source


def test_gaussian_segmentation_without_compartments_uses_global_trab_threshold():
    source = MODULE.read_text()

    assert "global_threshold_without_compartments = (" in source
    assert "segmentation_method == \"seg_gauss\"" in source
    assert "not compartment_split_requested" in source
    assert "seg_xyz = (segmentation_image_xyz >= trab_threshold) & full_xyz" in source
    assert "generated.metadata[\"segmentation_warning\"]" in source
    assert "No cortical mask was provided; Gaussian segmentation used the trabecular threshold" in source
    assert "segmentation_threshold_applied_global" in source
    assert "warning_text = f\" Warning: {metadata.get('segmentation_warning')}\"" in source


def test_laplace_hamming_segmentation_is_forced_from_support_mask():
    source = MODULE.read_text()

    assert "if segmentation_method == \"laplace_hamming\" and segmentation_image is not None:" in source
    assert "lh_support_xyz = _contour_support_binarization_xyz(" in source
    assert "full_mask_xyz=full_xyz" in source
    assert "seg_xyz = _ensure_bool(lh_support_xyz) & full_xyz" in source
    assert "generated.seg = numpy_xyz_to_sitk_binary(seg_xyz, image)" in source
    assert "Laplace-Hamming produced an empty bone segmentation" in source


def test_laplace_hamming_uses_core_native_scanco_input_convention():
    source = MODULE.read_text()

    assert "AIM_METADATA_ATTRIBUTE = \"HRpQCT.AIMMetadata\"" in source
    assert "AIM_SCALING_ATTRIBUTE = \"HRpQCT.AIMScaling\"" in source
    assert "from timelapsedhrpqct.io.aim import density_to_native_int16" in source
    assert "segmentation_input_unit\": \"scanco_native_int16\"" in source
    assert "segmentation_input_reader\": \"imported_density_to_native_int16\"" in source
    assert "read_aim(source_path, scaling=\"native\")" in source
    assert "segmentation_input_reader\": \"py_aimio_native_int16\"" in source
    assert "scanco_hu_int16" not in source
    assert "py_aimio_hu_int16" not in source
    assert "segmentation_node.CreateDefaultDisplayNodes()" in source
    assert "segmentation_node.SetAttribute(f\"HRpQCT.{key}\", str(generated.metadata[key]))" in source
    assert "Method=laplace_hamming; input={metadata.get('segmentation_input_unit')}" in source


def test_generated_masks_keep_aim_metadata_for_export():
    source = MODULE.read_text()

    assert "def _copy_aim_attributes" in source
    assert "AIM_METADATA_ATTRIBUTE, AIM_SOURCE_ATTRIBUTE, AIM_SCALING_ATTRIBUTE" in source
    assert "self._copy_aim_attributes(reference_node, label_node)" in source
    assert "self._copy_aim_attributes(volume_node, segmentation_node)" in source


def test_segmentation_installer_keeps_slicer_numpy_constraints():
    source = MODULE.read_text()

    assert 'CORE_PIP_CONSTRAINTS = ("numpy>=1.26,<2.0", "scikit-image>=0.24,<0.26", "tifffile<2026")' in source
    assert 'slicer.util.pip_uninstall("pyjpegls")' in source
    assert '" ".join(["timelapsed-hrpqct", *CORE_PIP_CONSTRAINTS])' in source


def test_laplace_hamming_shows_busy_progress_dialog():
    source = MODULE.read_text()

    assert "elif segmentation_method == \"laplace_hamming\":" in source
    assert "Running Laplace-Hamming bone segmentation..." in source
    assert "Laplace-Hamming Segmentation" in source
    assert "def _create_busy_progress_dialog" in source
    assert "dialog.setCancelButton(None)" in source


def test_segmentation_module_uses_tabs_for_tool_groups():
    source = MODULE.read_text()

    assert "self.toolTabs = qt.QTabWidget()" in source
    assert "self.toolTabs.addTab(generate_tab, \"Generate\")" in source
    assert "self.toolTabs.addTab(derive_tab, \"Derive Labels\")" in source
    assert "Create HOM Material Labels" in source
    assert "Generate Missing Mask" in source
    assert "Mask Operations" in source
    assert "Union" in source
    assert "Relabel Nonzero Voxels" in source
    assert "Validate Mask Set" in source
    assert "Count Selected Masks" in source
    assert "create_material_label_volume" in source
    assert "create_missing_mask_volume" in source
    assert "create_boolean_mask_volume" in source


def test_timelapsed_pipeline_exposes_geodesic_periosteal_contour_config():
    source = PIPELINE_MODULE.read_text()

    assert "self.maskPeriostealContour = qt.QComboBox()" in source
    assert "self.maskPeriostealContour.addItem(\"geodesic_fracture\", \"geodesic_fracture\")" in source
    assert "self.studyProfileCombo.currentIndexChanged.connect(self._on_apply_study_profile)" in source
    assert "Periosteal contour" in source
    assert "self.maskGeodesicThreshold" in source
    assert "self.maskGeodesicFillHoles" in source
    assert "_PIPELINE_LOCAL_REPO = _TOOLBOX_ROOT.parent / \"TimelapsedHRpQCT\"" in source
    assert "_PIPELINE_LOCAL_SRC = _PIPELINE_LOCAL_REPO / \"src\"" in source
    assert "def _local_pipeline_usable" in source
    assert "if _local_pipeline_usable(_PIPELINE_LOCAL_REPO, _PIPELINE_LOCAL_SRC)" in source
    assert "os.environ[\"PYTHONPATH\"]" in source
    assert "slicer.util.pip_install(\"hrpqct-geodesic-contour>=0.1.1\")" in source
    assert "timelapsed-hrpqct>={MIN_PIPELINE_VERSION}" in source
    assert "outer_cfg = {" in source
    assert "\"contour_method\": periosteal_contour_method" in source
    assert "\"geodesic_bone_threshold\": float(self.maskGeodesicThreshold.value)" in source
    assert "\"geodesic_fill_holes\": bool(self.maskGeodesicFillHoles.checked)" in source
    assert "if selected_profile == \"ped-fx\":" in source
    assert "mask_method = \"seg_gauss\"" in source
    assert "profile_segmentation_cfg = profile_masks_cfg.get(\"segmentation\") or {}" in source
    assert "profile_segmentation_cfg.get(\"method\")" in source
    assert "periosteal_contour_method = \"geodesic_fracture\"" in source
    assert "masks_override[\"roles\"] = [\"full\"]" in source
    assert "masks_override[\"inner\"] = {\"contour_method\": \"none\"}" in source
    assert "profile_initial_translation = profile_multistack_cfg.get(\"initial_translation_voxels\")" in source
    assert "\"initial_translation_voxels\": initial_translation_voxels" in source
    assert "\"masks\": masks_override" in source
    assert "timelapsed-hrpqct and contour dependency installation finished" in source


def test_timelapsed_pipeline_writes_sparse_override_config_for_profiles():
    source = PIPELINE_MODULE.read_text()

    create_override_start = source.index("    def create_override_config(self, settings_dict, results_root=None):")
    create_override_end = source.index("    def cleanup_temp_files", create_override_start)
    create_override_source = source[create_override_start:create_override_end]

    assert "self.default_config_path()" not in create_override_source
    assert "yaml.safe_dump(settings_dict" in create_override_source
    assert "slicer_run_configs" in create_override_source
