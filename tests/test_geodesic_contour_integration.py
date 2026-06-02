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
    assert "os.environ[\"PYTHONPATH\"]" in source
    assert "slicer.util.pip_install(\"hrpqct-geodesic-contour>=0.1.1\")" in source
    assert "outer_cfg = {" in source
    assert "\"contour_method\": periosteal_contour_method" in source
    assert "\"geodesic_bone_threshold\": float(self.maskGeodesicThreshold.value)" in source
    assert "\"geodesic_fill_holes\": bool(self.maskGeodesicFillHoles.checked)" in source
    assert "if selected_profile == \"ped-fx\":" in source
    assert "mask_method = \"seg_gauss\"" in source
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
