from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "HRpQCTTools" / "BoneMicroarchitecture" / "BoneMicroarchitecture.py"


def _install_slicer_import_stubs(monkeypatch) -> None:
    qt = types.ModuleType("qt")
    ctk = types.ModuleType("ctk")
    vtk = types.ModuleType("vtk")
    slicer = types.ModuleType("slicer")
    scripted = types.ModuleType("slicer.ScriptedLoadableModule")

    class _Base:
        def __init__(self, *args, **kwargs):
            pass

    scripted.ScriptedLoadableModule = _Base
    scripted.ScriptedLoadableModuleWidget = _Base
    scripted.ScriptedLoadableModuleLogic = _Base
    scripted.ScriptedLoadableModuleTest = _Base
    slicer.ScriptedLoadableModule = scripted
    slicer.util = types.SimpleNamespace()
    slicer.app = types.SimpleNamespace()

    monkeypatch.setitem(sys.modules, "qt", qt)
    monkeypatch.setitem(sys.modules, "ctk", ctk)
    monkeypatch.setitem(sys.modules, "vtk", vtk)
    monkeypatch.setitem(sys.modules, "slicer", slicer)
    monkeypatch.setitem(sys.modules, "slicer.ScriptedLoadableModule", scripted)


def _load_microarchitecture_module(monkeypatch, name: str):
    _install_slicer_import_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_microarchitecture_module_is_registered_with_toolbox_manifest_and_cmake() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = (ROOT / "toolbox_modules.json").read_text(encoding="utf-8")

    assert "add_subdirectory(HRpQCTTools/BoneMicroarchitecture)" in cmake
    assert '"path": "HRpQCTTools/BoneMicroarchitecture"' in manifest
    assert '"title": "Microarchitecture"' in manifest
    assert '"section": "Microstructural Analysis"' in manifest


def test_microarchitecture_module_has_custom_icon_and_author_credit() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    icon_path = ROOT / "HRpQCTTools" / "BoneMicroarchitecture" / "Resources" / "Icons" / "BoneMicroarchitecture.png"

    assert icon_path.is_file()
    assert "parent.icon = qt.QIcon(str(Path(__file__).with_name(\"Resources\") / \"Icons\" / \"BoneMicroarchitecture.png\"))" in source
    assert "Author: Matthias Walle" in source


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
    setup_start = source.index("    def setup(self):", source.index("class BoneMicroarchitectureWidget"))
    setup_end = source.index("    def _run_folder_batch(self):", setup_start)
    widget_setup = source[setup_start:setup_end]

    assert "Grayscale/BMD volume" in source
    assert widget_setup.index("Grayscale/BMD volume") < widget_setup.index("Bone segmentation")
    assert "Bone segmentation" in source
    assert "Full/periosteal mask" in source
    assert "Trabecular compartment mask" in source
    assert "Cortical compartment mask" in source
    assert "Analysis mask" in widget_setup
    assert "Common scan region mask" not in widget_setup
    assert "self.commonRegionMaskSelector" in widget_setup
    assert "BMD measures use the full compartment regions." in source
    assert "Segment" in source
    assert "Image units" not in widget_setup
    assert "Grayscale Calibration" in source
    assert "Calibration source" in source
    assert "Auto / already calibrated" in source
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
    assert "self._write_filtered_measurement_csv(path)" in source
    assert 'path = f"{path}.csv"' in source
    assert "self._lastMetrics" in source
    assert "self._lastMaps" in source


def test_microarchitecture_measurement_table_has_excel_style_filter() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    widget_setup = source[source.index("    def setup(self):", source.index("class BoneMicroarchitectureWidget")) :]

    assert "self.measurementFilterEdit = qt.QLineEdit()" in widget_setup
    assert 'self.measurementFilterEdit.placeholderText = "Filter rows, e.g. Tb.BMD or cortical"' in widget_setup
    assert "self.measurementFilterEdit.textChanged.connect(self._apply_measurement_filter)" in widget_setup
    assert "self.clearMeasurementFilterButton = qt.QPushButton(\"Clear\")" in widget_setup
    assert "self.clearMeasurementFilterButton.clicked.connect(self.measurementFilterEdit.clear)" in widget_setup
    assert 'form.addRow("Filter results", filter_widget)' in widget_setup
    assert "self._lastMeasurementRows = []" in widget_setup
    assert "self._lastMeasurementColumns = []" in widget_setup
    assert "self._lastFilteredTableNode = None" in widget_setup
    assert "def _filtered_measurement_rows(self):" in source
    assert "def _apply_measurement_filter(self" in source
    assert "def _measurement_table_from_rows(self, rows, name):" in source
    assert "def _cache_measurement_rows_from_table(self, table_node):" in source
    assert "def _write_filtered_measurement_csv(self, path):" in source
    assert "self._cache_measurement_rows_from_table(loaded_table)" in source
    assert "self._apply_measurement_filter()" in source


def test_microarchitecture_module_records_measurement_provenance_attributes() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'SetAttribute("BoneImaging.Microarchitecture.Engine", "bone_microarchitecture")' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.ThicknessMethod"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.ThicknessBackend"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.AnalysisMaskNode"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.AnalysisMaskName"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.TrabecularSegmentationID"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.PeriostealMaskID"' in source
    assert 'SetAttribute("BoneImaging.Microarchitecture.MapRole", map_role)' in source


def test_registered_series_mode_uses_timelapsed_discovery_and_slicer_timelapsed_layout() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'REGISTERED_MICROARCHITECTURE_DIR_NAME = "registered"' in source
    assert "TIMELAPSED_LOCAL_SRC" in source
    assert "from timelapsedhrpqct.config.models import DiscoveryConfig" in source
    assert "from timelapsedhrpqct.dataset.discovery import discover_raw_sessions" in source
    assert "discover_registered_series(" in source
    assert "canonicalize_sessions=False" in source
    assert "registered_microarchitecture_root(" in source
    assert 'return self._derivative_family_root(root, "Microarchitecture") / REGISTERED_MICROARCHITECTURE_DIR_NAME' in source
    assert '/ "derivatives" / family' in source
    assert 'if root.name == "derivatives":' in source
    assert 'if root.name == family:' in source
    assert "registered_session_output_dir(" in source
    assert '"native_space"' in source
    assert '"microarchitecture"' in source
    assert "sequential_registration_pairs(" in source
    assert "write_registered_series_manifest(" in source
    assert "if not written:" in source
    assert "No registered series measurements were run" in source


def test_registered_microarchitecture_reuses_shared_timelapsed_registration_derivatives() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "timelapse_pairwise_transform_path" in source
    assert "timelapse_baseline_transform_path" in source
    assert '/ "Registration"' in source
    assert "_read_existing_registered_transform(" in source
    assert "dataset_root=dataset_root" in source
    assert 'source = "reused_registration"' in source


def test_registered_series_is_exposed_as_batch_register_option() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    setup_start = source.index("    def setup(self):", source.index("class BoneMicroarchitectureWidget"))
    setup_end = source.index("    def _run_folder_batch(self):", setup_start)
    widget_setup = source[setup_start:setup_end]

    assert "class BoneMicroarchitecture(ScriptedLoadableModule):" in source
    assert "class BoneMicroarchitectureLogic(ScriptedLoadableModuleLogic):" in source
    assert "class BoneMicroarchitectureWidget(ScriptedLoadableModuleWidget):" in source
    assert "class BoneMicroarchitectureTest(ScriptedLoadableModuleTest):" in source
    assert 'parent.categories = ["Bone Imaging.Microstructural Analysis"]' in source

    assert "self.modeTabs = qt.QTabWidget()" in widget_setup
    assert 'self.modeTabs.addTab(single_tab, "Scene")' in widget_setup
    assert 'self.modeTabs.addTab(batch_tab, "Batch")' in widget_setup
    assert "Registered Series" not in widget_setup
    assert "series_tab" not in widget_setup
    assert "self.folderRegisteredCheck" in widget_setup
    assert 'workflow_form.addRow("Register", self.folderRegisteredCheck)' in widget_setup
    assert "self.folderRegisteredWorkflowCombo" in widget_setup
    assert '"Measure microarchitecture", "measure"' in widget_setup
    assert '"Prepare common region only", "common_region_only"' in widget_setup
    assert "self.folderThicknessMethodCombo" in widget_setup
    assert "self.folderThicknessBackendCombo" in widget_setup
    assert "self.folderBatchLogText" in widget_setup
    assert "self.discoverSeriesButton" not in widget_setup
    assert "self.runRegisteredSeriesButton" not in widget_setup
    assert "self.prepareRegisteredSeriesButton" not in widget_setup
    assert "def _prepare_registered_series(self):" not in source
    assert "def _run_registered_series(self):" not in source
    assert "_write_registered_series_job_for_rows" in source
    assert "self.logic.run_registered_series_job(" in source[source.index("    def _start_next_folder_batch_job(self):") :]


def test_microarchitecture_batch_ui_uses_discovery_table_and_release_labels() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    widget_setup = source[source.index("    def setup(self):", source.index("class BoneMicroarchitectureWidget")) :]

    assert 'self.modeTabs.addTab(single_tab, "Scene")' in widget_setup
    assert "Folder Batch" not in widget_setup
    assert 'qt.QGroupBox("Discovery")' in widget_setup
    assert 'qt.QGroupBox("Workflow")' in widget_setup
    assert "self.folderDiscoverButton" in widget_setup
    assert "self.folderBatchTable" in widget_setup
    assert "self.folderBrowseDatasetButton" in widget_setup
    assert "self._configure_folder_batch_table_for_mode()" in widget_setup
    assert 'headers = ["Action", "Subject", "Site", "Sessions", "Status"]' in source
    assert 'headers = ["Action", "Image", "Subject", "Site", "Session", "Status"]' in source
    assert "self.folderRegisteredCheck" in widget_setup
    assert 'workflow_form.addRow("Register", self.folderRegisteredCheck)' in widget_setup
    assert 'workflow_form.addRow("Registered workflow", self.folderRegisteredWorkflowCombo)' in widget_setup
    assert "self.folderRegisteredWorkflowCombo.enabled = registered" in source
    assert "self.folderRegisteredCheck.toggled.connect(self._update_folder_registered_options)" in widget_setup
    assert "self.folderUseCommonRegionCheck" not in widget_setup
    assert "Use common region" not in widget_setup
    assert "self.folderSkipExistingCheck" in widget_setup
    assert 'workflow_form.addRow("Skip existing", self.folderSkipExistingCheck)' in widget_setup
    assert "force=not bool(self.folderSkipExistingCheck.checked)" in source
    assert "_configure_folder_batch_table_for_mode" in source
    assert 'self.folderRunButton = qt.QPushButton("Run all")' in widget_setup
    assert "_queue_folder_batch_row" in source
    assert "_start_next_folder_batch_job" in source
    assert "_load_folder_batch_outputs" in source
    assert "slicer.util.loadTable(long_csv)" in source
    assert 'loaded_table.SetName(Path(long_csv).stem)' in source
    assert 'self._table_count(self.folderBatchTable, "rowCount")' in source
    assert 'self._table_count(self.folderBatchTable, "columnCount")' in source
    assert "_folderBatchQueue" in source
    assert "Subject filter" not in widget_setup
    assert "Site filter" not in widget_setup
    assert "self.folderSubjectEdit" not in widget_setup
    assert "self.folderSiteEdit" not in widget_setup


def test_microarchitecture_independent_batch_rows_queue_and_load_existing_results() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    queue_method = source[source.index("    def _queue_folder_batch_row(self, row_index):") : source.index("    def _start_next_folder_batch_job(self):")]
    start_method = source[source.index("    def _start_next_folder_batch_job(self):") : source.index("    def _on_folder_batch_job_finished", source.index("    def _start_next_folder_batch_job(self):"))]
    finish_method = source[source.index("    def _on_folder_batch_job_finished", source.index("    def _start_next_folder_batch_job(self):")) : source.index("    def _load_folder_batch_outputs", source.index("    def _on_folder_batch_job_finished"))]
    discovery = source[source.index("    def _discover_folder_batch_groups(self):") : source.index("    def _set_folder_group_status(self, row_index, status):")]

    assert '"mode": "independent"' in queue_method
    assert "self._folderBatchQueue.append(queued)" in queue_method
    assert "Folder batch is already running" not in queue_method
    assert "self.logic.run_folder_batch_job(" in start_method
    assert "self._folderBatchProcess = self.logic.run_folder_batch_job(" in start_method
    assert "self._folderBatchProcess = self.logic.run_registered_series_job(" in start_method
    assert "job.get(\"mode\") == \"independent\"" in start_method
    assert "subject_id=str(job[\"group\"].get(\"subject\", \"\"))" in start_method
    assert "site=str(job[\"group\"].get(\"site\", \"\"))" in start_method
    assert "session_id=str(job[\"group\"].get(\"session\", \"\"))" in start_method
    assert "self._folderBatchGroups[int(row_index)][\"result_path\"]" in finish_method
    assert "self._folder_result_path_for_group(root, group)" in discovery
    assert 'self._set_folder_group_action(row_index, "Load")' in discovery
    assert 'self._set_folder_group_action(row_index, "Run")' in discovery
    assert "self.logic.discover_independent_cases(" in discovery
    assert 'group.get("status") != "Ready"' in queue_method
    assert "Microarchitecture batch row is missing required masks" in queue_method


def test_microarchitecture_logic_merges_independent_discovery_with_core_complete_cases() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    method = source[source.index("    def discover_independent_cases(") : source.index("    def sequential_registration_pairs", source.index("    def discover_independent_cases("))]

    assert "microarchitecture_batch._discover_cases(Path(str(dataset_root)).expanduser())" in method
    assert '"status": "Ready"' in method
    assert "ready_by_key.get(key, row)" in method


def test_microarchitecture_folder_batch_process_finish_slot_handles_qt_signal_variants() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    logic_method = source[source.index("    def run_folder_batch_job(") : source.index("    def run_registered_series_job(")]

    assert "def _finished(*signal_args):" in logic_method
    assert "proc.finished.connect(_finished)" in logic_method
    assert "proc.exitCode()" in logic_method
    assert "on_finished(exit_code, exit_status)" in logic_method


def test_microarchitecture_independent_batch_does_not_fall_back_to_recursive_file_sweep() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    discovery = source[source.index("    def _discover_folder_batch_groups(self):") : source.index("    def _set_folder_group_status(self, row_index, status):")]

    assert "root.rglob" not in discovery
    assert "for path in root.iterdir()" in discovery
    assert "self._is_folder_batch_image_path(path)" in discovery
    assert "_is_folder_batch_image_path" in source


def test_registered_series_consumes_existing_masks_without_generation_controls() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    setup_start = source.index("    def setup(self):", source.index("class BoneMicroarchitectureWidget"))
    setup_end = source.index("    def _run_folder_batch(self):", setup_start)
    widget_setup = source[setup_start:setup_end]
    prepare_start = source.index("    def prepare_registered_series_workspace(")
    prepare_end = source.index("    def _registered_progress(", prepare_start)
    prepare_method = source[prepare_start:prepare_end]
    worker_method = source[source.index("def _run_registered_series_worker(job_path):") :]

    assert "prepare_registered_series_workspace(" in source
    assert '"common_region_only"' in source
    assert "common-region-only workflow complete; measurements skipped" in source
    assert "registered_session_masks_dir(" not in source
    assert "registered_session_mask_path(" not in source
    assert "_write_registered_mask(" not in source
    assert "_complete_registered_series_masks(" not in prepare_method
    assert "_complete_registered_series_masks(" not in source
    assert "_generate_registered_contours(" not in source
    assert "_generate_registered_bone_segmentation(" not in source
    assert "self.seriesSegmentationMethodCombo" not in widget_setup
    assert "self.seriesPeriostealContourCombo" not in widget_setup
    assert "self.seriesEndostealContourCombo" not in widget_setup
    assert "Missing masks" not in widget_setup
    assert "segmentation_method=" not in prepare_method
    assert "periosteal_contour_method=" not in prepare_method
    assert "endosteal_contour_method=" not in prepare_method
    assert 'job["segmentation_method"]' not in worker_method
    assert "generate missing masks" not in widget_setup.lower()


def test_registered_series_run_reports_measured_and_skipped_sessions() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    run_logic = source[source.index("    def run_registered_series_microarchitecture(") :]
    worker_method = source[source.index("def _run_registered_series_worker(job_path):") :]

    assert "skipped_rows = []" in run_logic
    assert "skipped_rows.append(" in run_logic
    assert '"skipped_rows": skipped_rows' in run_logic
    assert "measured_count = len(outputs.get(\"session_csvs\", []))" in worker_method
    assert "skipped_count = len(outputs.get(\"skipped_rows\", []))" in worker_method
    assert "skipped_count" in worker_method
    assert "[registered] skipped" in worker_method


def test_registered_series_run_uses_background_qprocess_with_streamed_updates() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    logic_source = source[source.index("class BoneMicroarchitectureLogic") :]
    run_start = source.index("    def _start_next_folder_batch_job(self):")
    run_end = source.index("    def _on_folder_batch_job_finished", run_start)
    run_widget = source[run_start:run_end]

    assert "qt.QProcess()" in logic_source
    assert 'env.insert("PYTHONUNBUFFERED", "1")' in logic_source
    assert "proc.readyRead.connect(_read_output)" in logic_source
    assert "proc.finished.connect(_finished)" in logic_source
    assert "def run_registered_series_job(" in logic_source
    assert "_write_registered_series_job_for_rows(" in source
    assert "_on_folder_batch_job_finished" in source
    assert "self.logic.run_registered_series_job(" in run_widget
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
    manifest_start = source.index("    def write_registered_series_manifest(")
    manifest_end = source.index("    def prepare_registered_series_workspace(", manifest_start)
    manifest_logic = source[manifest_start:manifest_end]

    assert "_build_registered_common_regions(" in prepare_logic
    assert "_register_to_baseline(" in source
    assert "_resample_registered_mask(" in source
    assert "sitk.WriteTransform" in source
    assert "common_space" in source
    assert "common_masks" not in source
    assert 'workflow="CommonRegion"' in source
    assert '_shared_common_region_root(' in source
    assert '_shared_common_region_common_mask_path(' in source
    assert '_shared_common_region_native_mask_path(' in source
    assert "_registered_common_mask_path(" not in source
    assert "_registered_common_session_mask_path(" not in source
    assert '"registration/adjacent"' not in manifest_logic
    assert '"registration/composed"' not in manifest_logic
    assert '"common_space/common_masks"' not in manifest_logic
    assert 'f"native_space/ses-{row[\'session_id\']}/masks"' not in manifest_logic
    assert 'f"native_space/ses-{row[\'session_id\']}/microarchitecture/maps"' in manifest_logic
    assert 'write_manifest(' in prepare_logic
    assert '"producer": "registered_microarchitecture"' in prepare_logic
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
    assert "_registered_pairwise_transform_path(" not in source
    assert "_registered_transform_path(" not in source
    assert "_shared_timelapsed_pairwise_transform_path(" in source
    assert "_shared_timelapsed_baseline_transform_path(" in source
    assert '"pairwise"' in source
    assert '"baseline"' in source
    assert "fixed_image=previous_image" in source
    assert "moving_image=image" in source


def test_microarchitecture_batch_delegates_to_package_batch_api() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "from bone_microarchitecture.batch import run_microarchitecture_batch" in source
    assert "run_microarchitecture_batch(" in source


def test_microarchitecture_folder_batch_action_executes_package_api(monkeypatch, tmp_path: Path) -> None:
    module = _load_microarchitecture_module(monkeypatch, "microarchitecture_batch_test")
    received = {}
    monkeypatch.setattr(module, "run_microarchitecture_batch", lambda root, **kwargs: received.update(root=root, **kwargs) or ["done"])

    result = module.BoneMicroarchitectureLogic().run_folder_batch(tmp_path, use_common_region=False, force=True)

    assert result == ["done"]
    assert received == {"root": tmp_path, "use_common_region": False, "force": True, "progress": None}


def test_microarchitecture_background_batch_command_carries_folder_options(monkeypatch, tmp_path: Path) -> None:
    module = _load_microarchitecture_module(monkeypatch, "microarchitecture_batch_command_test")

    command = module.BoneMicroarchitectureLogic.folder_batch_command(
        tmp_path, subject_id="S1", site="tibia", use_common_region=False,
        force=True, thickness_method="edt", thickness_backend="opencl",
    )

    assert command == [
        "-m", "bone_microarchitecture.cli", "run-batch", str(tmp_path.resolve()),
        "--subject", "S1", "--site", "tibia",
        "--no-common-region", "--force",
        "--thickness-method", "edt", "--thickness-backend", "opencl",
    ]


def test_microarchitecture_folder_result_path_accepts_session_aliases(monkeypatch, tmp_path: Path) -> None:
    module = _load_microarchitecture_module(monkeypatch, "microarchitecture_result_alias_test")
    widget = object.__new__(module.BoneMicroarchitectureWidget)
    output = (
        tmp_path
        / "derivatives"
        / "Microarchitecture"
        / "sub-STRAMBO_0001"
        / "site-radius_left"
        / "native_space"
        / "ses-Y00"
        / "measurements"
        / "sub-STRAMBO_0001_ses-Y00_site-radius_left_measurements.csv"
    )
    output.parent.mkdir(parents=True)
    output.write_text("Parameter,Mean\nTb.BV/TV,0.1\n", encoding="utf-8")

    resolved = widget._folder_result_path_for_group(
        tmp_path,
        {"subject": "STRAMBO_0001", "site": "radius_left", "session": "00", "mode": "independent"},
    )

    assert resolved == output


def test_microarchitecture_background_job_launches_with_pythonslicer(monkeypatch, tmp_path: Path) -> None:
    module = _load_microarchitecture_module(monkeypatch, "microarchitecture_batch_launch_test")
    started = []

    class Signal:
        def connect(self, _callback): pass

    class Process:
        MergedChannels = 1
        def __init__(self):
            self.readyRead = Signal()
            self.finished = Signal()
        def setProcessChannelMode(self, _mode): pass
        def start(self, executable, arguments): started.append((executable, arguments))

    monkeypatch.setattr(module.qt, "QProcess", Process, raising=False)
    monkeypatch.setattr(module.slicer, "app", type("App", (), {"applicationFilePath": staticmethod(lambda: "/Applications/Slicer.app/Contents/MacOS/Slicer")})(), raising=False)
    module.BoneMicroarchitectureLogic().run_folder_batch_job(tmp_path, force=True, thickness_backend="opencl")

    assert started[0][0].endswith("Contents/bin/PythonSlicer")
    assert started[0][1][:3] == ["-m", "bone_microarchitecture.cli", "run-batch"]
    assert "--force" in started[0][1]
    assert started[0][1][-2:] == ["--thickness-backend", "opencl"]


def test_registered_series_does_not_build_partial_common_regions_for_incomplete_groups() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    prepare_logic = source[source.index("    def prepare_registered_series_workspace(") :]

    assert "_mark_incomplete_registered_groups(" in prepare_logic
    assert "Missing common region" in source
    assert "group has incomplete timepoints" in source
    assert "if any(row.get(\"status\") != \"Ready\" for row in group_rows):" in source
