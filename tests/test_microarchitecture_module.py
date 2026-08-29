from __future__ import annotations

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "HRpQCTTools" / "BoneMicroarchitecture" / "BoneMicroarchitecture.py"


def test_microarchitecture_module_is_registered_with_toolbox_manifest_and_cmake() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = (ROOT / "toolbox_modules.json").read_text(encoding="utf-8")

    assert "add_subdirectory(HRpQCTTools/BoneMicroarchitecture)" in cmake
    assert '"path": "HRpQCTTools/BoneMicroarchitecture"' in manifest
    assert '"title": "Bone Microarchitecture"' in manifest
    assert '"section": "HR-pQCT"' in manifest


def test_microarchitecture_module_wraps_toolbox_microarchitecture_api() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "from bone_microarchitecture import compute_microarchitecture" in source
    assert "core_result = compute_microarchitecture(" in source
    assert "slicer.util.saveNode" in source
    assert "return self._volume_to_sitk(" in source
    assert "sitk.sitkUInt8" in source
    assert "return sitk.ReadImage(str(path), pixel_type)" in source
    assert "slicer.util.loadVolume" in source
    assert "vtkMRMLTableNode" in source
    assert "write_measurement_csv" in source
    assert "metadata.version(\"bone-microarchitecture\")" in source
    assert "def core_runtime_status(self):" in source
    assert 'sys.platform == "darwin"' in source
    assert "pyobjc-framework-Metal>=10" in source
    assert "pyopencl>=2024.1" in source
    assert "Microarchitecture core available from local source." in source
    assert "Microarchitecture core source found but not ready" in source
    assert "Microarchitecture core installed but not ready" in source
    assert ("OR" + "MiR-XCT") not in source


def test_microarchitecture_module_accepts_segmentation_nodes_generated_by_segmentation_module() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert '"vtkMRMLSegmentationNode"' in source
    assert "ExportSegmentsToLabelmapNode" in source
    assert "_segment_id_for_role" in source
    assert "selected_segment_id" in source
    assert "_segment_tag_value(segment, \"HRpQCT.Role\")" in source
    assert 'segment.GetTag("HRpQCT.Role")' not in source
    assert "vtk.mutable(\"\")" in source
    assert "_refresh_segment_combo" in source
    assert 'hasattr(reference_node, "CopyOrientation")' in source
    assert '"Trabecular mask"' in source
    assert '"Trabecular compartment mask"' in source
    assert '"Full mask"' in source
    assert '"Bone segmentation"' in source
    assert '"Cortical mask"' in source
    assert '"Cortical compartment mask"' in source


def test_microarchitecture_exports_segmentation_segments_with_shared_geometry() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "reference_node=None" in source
    assert "reference_node=reference_node" in source
    assert "_first_available_reference_node(" in source
    assert "roles_and_segment_ids = [" in source
    assert "if optional_node is trabecular_segmentation_node:" in source
    assert "ExportSegmentsToLabelmapNode(" in source
    assert "reference_node," in source
    assert "EXTENT_REFERENCE_GEOMETRY" in source
    assert "Shared reference geometry keeps segmentation segment exports on the same grid." in source


def test_microarchitecture_module_uses_core_for_bmd_and_thickness_compartments() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "sitk.GetArrayFromImage(trab_seg)" in source
    assert "sitk.GetArrayFromImage(peri_mask)" in source
    assert "sitk.GetArrayFromImage(cort_mask)" in source
    assert "Select a bone segmentation so trabecular and cortical bone measures can be intersected" in source
    assert "common_region_node=None" in source
    assert "common_region = self._volume_to_sitk_uint8(" in source
    assert "from SlicerBoneImagingToolboxLib.masks import clip_mask_to_region" in source
    assert "bone_seg = clip_mask_to_region(bone_seg, common_region)" in source
    assert "peri_mask = clip_mask_to_region(peri_mask, common_region)" in source
    assert "trab_seg = clip_mask_to_region(trab_seg, common_region)" in source
    assert "cort_mask = clip_mask_to_region(cort_mask, common_region)" in source
    assert "grayscale=None if bmd_image is None else sitk.GetArrayFromImage(bmd_image)" in source
    assert "for map_role, array in core_result.maps.items()" in source
    assert "_array_to_sitk_like(array, trab_seg)" in source


def test_microarchitecture_prefers_aimio_calibrated_grayscale_when_source_metadata_exists() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'AIM_SOURCE_ATTRIBUTE = "HRpQCT.AIMSourcePath"' in source
    assert "_calibrated_grayscale_image" in source
    assert "from ScancoIOLib import aim_io" in source
    assert "SCANCO_IO_DIR" in source
    assert 'aim_io.read_image(source_path, scaling="density")' in source
    assert 'metadata["grayscale_reader"] = "aimio-py"' in source
    assert 'metadata["grayscale_units"] = "bmd"' in source


def test_microarchitecture_loads_maps_for_cortical_thickness_and_masked_bmd() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "for map_role, array in core_result.maps.items()" in source
    assert "f\"{prefix}_{map_role.replace('.', '')}_map\"" in source
    assert "_array_to_sitk_like(array, trab_seg)" in source
    assert "Tb.BMD" in source
    assert "Ct.BMD" in source
    assert "Ct.Po.Dm" in source
    assert "TB.BMD" not in source
    assert "CT.BMD" not in source
    assert "_bmd_image(" in source
    assert "attenuation = (image / 1000.0 + 1.0) * float(mu_water)" in source


def test_microarchitecture_widget_exposes_minimal_clean_inputs_and_outputs() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    widget_setup = source[source.index("    def setup(self):", source.index("class BoneMicroarchitectureWidget")) :]

    assert "Grayscale/BMD volume" in source
    assert widget_setup.index("Grayscale/BMD volume") < widget_setup.index("Bone segmentation")
    assert "Bone segmentation" in source
    assert "Full/periosteal mask" in source
    assert "Trabecular compartment mask" in source
    assert "Cortical compartment mask" in source
    assert "Common scan region mask" in source
    assert "self.commonRegionMaskSelector" in widget_setup
    assert "BMD measures use the full compartment regions." in source
    assert "Segment" in source
    assert "Image units" in source
    assert "BMD Calibration" in source
    assert "Thickness Settings" in source
    assert "Bounded EDT" in source
    assert "Exact sphere fitting" in source
    assert 'for label, value in [("Exact sphere fitting", "hildebrand"), ("Bounded EDT", "edt")]' in source
    assert 'thickness_method="hildebrand"' in source
    assert 'thickness_backend="auto"' in source
    assert '("Apple MPS (macOS)", "mps")' in source
    assert '("OpenCL GPU", "opencl")' in source
    assert "default_thickness_backend" in source
    assert "self.thicknessBackendCombo.setCurrentIndex(index)" in source
    assert "Apple MPS" in source
    assert "OpenCL" in source
    assert "requires PyTorch" not in source
    assert "mu_scaling" in source
    assert "rescale_slope" in source
    assert "Output prefix" in source
    assert "Create measurement maps" not in widget_setup
    assert "CSV output path" not in widget_setup
    assert "Export measurements CSV" in source
    assert "qt.QFileDialog.getSaveFileName" in source
    assert "Run microarchitecture" in source
    assert "Install / update microarchitecture core" in source


def test_microarchitecture_run_always_loads_maps_and_shows_table() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    run_method = source[source.index("    def _run_microarchitecture(self):") :]

    assert "create_maps=True" in run_method
    assert "common_region_node=self.commonRegionMaskSelector.currentNode()" in run_method
    assert "common_region_segment_id=self._selected_segment_id(self.commonRegionSegmentCombo)" in run_method
    assert "thickness_method=str(self.thicknessMethodCombo.currentData)" in run_method
    assert "thickness_backend=str(self.thicknessBackendCombo.currentData)" in run_method
    assert "Apple MPS sphere fitting is experimental" in run_method
    assert "csv_output_path=" not in run_method
    assert "self._lastMetrics = dict(metrics)" in run_method
    assert "self._lastMaps = dict(maps)" in run_method
    assert "self.exportCsvButton.enabled = True" in run_method
    assert "self._show_measurement_table(table_node)" in run_method
    assert 'slicer.util.selectModule("Tables")' in source
    assert "tables_widget.setCurrentTableNode(table_node)" in source


def test_microarchitecture_exports_last_measurements_from_button() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "self.exportCsvButton.clicked.connect(self._export_measurements_csv)" in source
    assert "def _export_measurements_csv(self):" in source
    assert "write_measurement_csv(path, self._lastMetrics, self._lastMaps)" in source
    assert 'path = f"{path}.csv"' in source
    assert "self._lastMetrics" in source
    assert "self._lastMaps" in source


def test_microarchitecture_module_records_measurement_provenance_attributes() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'SetAttribute("BoneImaging.Microarchitecture.Engine", "bone_microarchitecture")' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.ThicknessMethod"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.ThicknessBackend"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.CommonRegionNode"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.TrabecularSegmentationID"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.PeriostealMaskID"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.MapRole", map_role)' in source


def test_registered_series_mode_uses_timelapsed_discovery_and_slicer_timelapsed_layout() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'REGISTERED_MICROARCHITECTURE_DIR_NAME = "RegisteredMicroarchitecture"' in source
    assert "TIMELAPSED_LOCAL_SRC" in source
    assert "from timelapsedhrpqct.config.models import DiscoveryConfig" in source
    assert "from timelapsedhrpqct.dataset.discovery import discover_raw_sessions" in source
    assert "discover_registered_series(" in source
    assert "canonicalize_sessions=True" in source
    assert "registered_microarchitecture_root(" in source
    assert 'dataset_root / REGISTERED_MICROARCHITECTURE_DIR_NAME' in source
    assert 'dataset_root / "derivatives" / REGISTERED_MICROARCHITECTURE_DIR_NAME' not in source
    assert "registered_session_output_dir(" in source
    assert '"native_space"' in source
    assert '"microarchitecture"' in source
    assert "sequential_registration_pairs(" in source
    assert "write_registered_series_manifest(" in source
    assert "if not written:" in source
    assert "No registered series measurements were run" in source


def test_registered_series_widget_has_dedicated_tab_and_review_table() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    widget_setup = source[source.index("    def setup(self):", source.index("class BoneMicroarchitectureWidget")) :]

    assert "class BoneMicroarchitecture(ScriptedLoadableModule):" in source
    assert "class BoneMicroarchitectureLogic(ScriptedLoadableModuleLogic):" in source
    assert "class BoneMicroarchitectureWidget(ScriptedLoadableModuleWidget):" in source
    assert "class BoneMicroarchitectureTest(ScriptedLoadableModuleTest):" in source
    assert 'parent.categories = ["Bone Imaging.HR-pQCT"]' in source

    assert "self.modeTabs = qt.QTabWidget()" in widget_setup
    assert 'self.modeTabs.addTab(single_tab, "Single Scan")' in widget_setup
    assert 'self.modeTabs.addTab(series_tab, "Registered Series")' in widget_setup
    assert "RegisteredMicroarchitecture" in widget_setup
    assert "self.seriesDatasetRootEdit" in widget_setup
    assert "self.seriesOutputRootEdit" in widget_setup
    assert "self.seriesSubjectCombo" in widget_setup
    assert "self.seriesSiteCombo" in widget_setup
    assert "All subjects" in widget_setup
    assert "All sites" in widget_setup
    assert "self.seriesSubjectFilterEdit" not in widget_setup
    assert "self.seriesSiteFilterEdit" not in widget_setup
    assert '"Subject filter"' not in widget_setup
    assert '"Site filter"' not in widget_setup
    assert "self.discoverSeriesButton" in widget_setup
    assert "self.runRegisteredSeriesButton" in widget_setup
    assert "self.prepareRegisteredSeriesButton" not in widget_setup
    assert widget_setup.index("self.discoverSeriesButton") < widget_setup.index("form.addRow(\"Subject\"")
    assert widget_setup.index("form.addRow(\"Subject\"") < widget_setup.index("form.addRow(\"Missing masks\"")
    assert 'qt.QPushButton("Run")' in widget_setup
    assert '"Run series measurements"' not in widget_setup
    assert "def _prepare_registered_series(self):" in source
    assert "if not self._lastRegisteredRows:" in source[source.index("    def _run_registered_series(self):") :]
    assert "self.seriesTable.setHorizontalHeaderLabels" in widget_setup
    assert '"Subject", "Site", "Session", "Image", "Bone seg", "Full", "Trab", "Cort", "Status"' in widget_setup
    assert "_discover_registered_series" in source
    assert "_populate_registered_series_filters" in source
    assert "_filtered_registered_rows" in source
    assert "_refresh_registered_series_table" in source
    assert "_prepare_registered_series" in source
    assert "_run_registered_series" in source


def test_registered_series_prepare_can_generate_missing_masks() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    widget_setup = source[source.index("    def _setup_registered_series_tab(self, layout):") :]
    prepare_method = source[source.index("    def _prepare_registered_series(self):") :]

    assert "prepare_registered_series_workspace(" in source
    assert "_complete_registered_series_masks" in source
    assert "_derive_registered_compartment_masks" in source
    assert "_generate_registered_bone_segmentation" in source
    assert "_generate_registered_contours" in source
    assert "from timelapsedhrpqct.processing.masks import resolve_masks" in source
    assert "resolved, provenance = resolve_masks(" in source
    assert 'source.startswith("derived_from_")' in source
    assert 'provenance.get(role, "provided")' in source
    assert "sitk.WriteImage" in source
    assert "self.seriesSegmentationMethodCombo" in widget_setup
    assert "self.seriesPeriostealContourCombo" in widget_setup
    assert "self.seriesEndostealContourCombo" in widget_setup
    assert "Missing masks" in widget_setup
    assert "segmentation_method=str(self.seriesSegmentationMethodCombo.currentData)" in prepare_method
    assert "periosteal_contour_method=str(self.seriesPeriostealContourCombo.currentData)" in prepare_method
    assert "endosteal_contour_method=str(self.seriesEndostealContourCombo.currentData)" in prepare_method


def test_registered_series_run_reports_measured_and_skipped_sessions() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    run_logic = source[source.index("    def run_registered_series_microarchitecture(") :]
    run_widget = source[source.index("    def _run_registered_series(self):") :]

    assert "skipped_rows = []" in run_logic
    assert "skipped_rows.append(" in run_logic
    assert '"skipped_rows": skipped_rows' in run_logic
    assert "measured_count = len(outputs.get(\"session_csvs\", []))" in run_widget
    assert "skipped_count = len(outputs.get(\"skipped_rows\", []))" in run_widget
    assert "skipped_count" in run_widget
    assert "[registered] skipped" in run_widget


def test_registered_series_run_uses_background_qprocess_with_streamed_updates() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    logic_source = source[source.index("class BoneMicroarchitectureLogic") :]
    run_start = source.index("    def _run_registered_series(self):")
    run_end = source.index("    def _on_registered_series_finished", run_start)
    run_widget = source[run_start:run_end]

    assert "qt.QProcess()" in logic_source
    assert 'env.insert("PYTHONUNBUFFERED", "1")' in logic_source
    assert "proc.readyRead.connect(_read_output)" in logic_source
    assert "proc.finished.connect(_finished)" in logic_source
    assert "def run_registered_series_job(" in logic_source
    assert "_write_registered_series_job(" in source
    assert "_on_registered_series_finished" in source
    assert "self.logic.run_registered_series_job(" in run_widget
    assert "self._set_registered_series_running(True)" in run_widget
    assert "self._with_wait_cursor(" not in run_widget
    assert "--registered-series-job" in source
    assert "progress_callback=print_progress" in source
    assert "_registered_progress(" in source
    assert "progress_callback=None" in source
    assert "default_thickness_backend()" in source
    assert "[registered] thickness:" in source
    assert "thickness_backend=resolved_thickness_backend" in source
    assert "[registered] measuring sub-" in source


def test_registered_series_preparation_builds_common_regions_before_measurement() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    prepare_logic = source[source.index("    def prepare_registered_series_workspace(") :]
    run_logic = source[source.index("    def run_registered_series_microarchitecture(") :]

    assert "_build_registered_common_regions(" in prepare_logic
    assert "_register_to_baseline(" in source
    assert "_resample_registered_mask(" in source
    assert "sitk.WriteTransform" in source
    assert "common_space" in source
    assert "common_masks" in source
    assert "native_common" in source
    assert "row[\"native_common_scan_region_path\"]" in source
    assert "measurement_space" in source
    assert "native_image_space_common_region" in source
    assert "self.write_registered_series_manifest(dataset_root, root, rows)" in run_logic


def test_registered_series_common_region_is_scan_fov_not_compartment_intersection() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    common_logic = source[source.index("    def _build_registered_common_regions(") :]
    measurement_logic = source[source.index("    def run_registered_series_microarchitecture(") :]

    assert '"scan_region_common"' in common_logic
    assert '"scan_region_native_common"' in common_logic
    assert "build_common_scan_region(" in common_logic
    assert "CommonRegionSession(" in common_logic
    assert '"native_common_scan_region_path"' in common_logic
    assert "for role in (\"full\", \"trab\", \"cort\"):" not in common_logic
    assert "common_by_role" not in common_logic
    assert "row[\"full_path\"] = common_paths[\"full\"]" not in common_logic
    assert "row[\"trab_path\"] = common_paths[\"trab\"]" not in common_logic
    assert "row[\"cort_path\"] = common_paths[\"cort\"]" not in common_logic
    assert "_clip_registered_mask_to_scan_region(" in measurement_logic
    assert "bone_seg = self._clip_registered_mask_to_scan_region(bone_seg, scan_region)" in measurement_logic
    assert "full_mask = self._clip_registered_mask_to_scan_region(full_mask, scan_region)" in measurement_logic
    assert "trab_mask = self._clip_registered_mask_to_scan_region(trab_mask, scan_region)" in measurement_logic
    assert "cort_mask = self._clip_registered_mask_to_scan_region(cort_mask, scan_region)" in measurement_logic


def test_registered_series_common_regions_use_sequential_composed_transforms() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "PairwiseTransform" in source
    assert "compose_sequential_to_baseline(" in source
    assert "_registered_pairwise_transform_path(" in source
    assert '"pairwise"' in source
    assert '"composed"' in source
    assert "fixed_image=previous_image" in source
    assert "moving_image=image" in source


def test_microarchitecture_batch_delegates_to_package_batch_api() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "from bone_microarchitecture.batch import run_microarchitecture_batch" in source
    assert "run_microarchitecture_batch(" in source


def test_microarchitecture_folder_batch_action_executes_package_api(monkeypatch, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("microarchitecture_batch_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = {}
    monkeypatch.setattr(module, "run_microarchitecture_batch", lambda root, **kwargs: received.update(root=root, **kwargs) or ["done"])

    result = module.BoneMicroarchitectureLogic().run_folder_batch(tmp_path, use_common_region=False, force=True)

    assert result == ["done"]
    assert received == {"root": tmp_path, "use_common_region": False, "force": True, "progress": None}


def test_registered_series_does_not_build_partial_common_regions_for_incomplete_groups() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    prepare_logic = source[source.index("    def prepare_registered_series_workspace(") :]

    assert "_mark_incomplete_registered_groups(" in prepare_logic
    assert "Missing common region" in source
    assert "group has incomplete timepoints" in source
    assert "if any(row.get(\"status\") != \"Ready\" for row in group_rows):" in source
