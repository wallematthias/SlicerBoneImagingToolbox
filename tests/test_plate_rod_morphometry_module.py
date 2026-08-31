from __future__ import annotations

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "HRpQCTTools" / "PlateRodMorphometryHRpQCT" / "PlateRodMorphometryHRpQCT.py"


def test_plate_rod_module_is_registered_with_toolbox_manifest_and_cmake() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = (ROOT / "toolbox_modules.json").read_text(encoding="utf-8")

    assert "add_subdirectory(HRpQCTTools/PlateRodMorphometryHRpQCT)" in cmake
    assert '"path": "HRpQCTTools/PlateRodMorphometryHRpQCT"' in manifest
    assert '"title": "Plate/Rod Morphometry"' in manifest
    assert '"section": "Analysis Methods"' in manifest


def test_plate_rod_module_keeps_algorithm_logic_in_core_package() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "PLATE_ROD_LOCAL_REPO = TOOLBOX_ROOT.parent / \"bone-plate-rod-thinning\"" in source
    assert 'os.environ.get("PLATE_ROD_INSTALL_LOCAL") == "1"' in source
    assert "def _remove_local_core_repo_from_sys_path():" in source
    assert "sys.path.insert(0, str(PLATE_ROD_LOCAL_REPO))" not in source
    assert "from plate_rod_thinning import PlateRodParameters, plate_rod_analysis" in source
    assert "core_result = plate_rod_analysis(" in source
    assert "metadata.version(\"plate-rod-thinning\")" in source
    assert "sk_thinning3D" not in source
    assert "ci_classify_image" not in source
    assert "ci_deskeltonize" not in source


def test_plate_rod_module_includes_network_graph_citation() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "PLATE_ROD_CITATION" in source
    assert "Walle M, Yeritsyan D, Abbasian M, Oftadeh R, Müller R, Nazarian A." in source
    assert "A graph model to describe the network connectivity of trabecular plates and rods." in source
    assert "Front Bioeng Biotechnol. 2024 May 6;12:1384280." in source
    assert "doi: 10.3389/fbioe.2024.1384280" in source
    assert "PMID: 38770275" in source
    assert "PMCID: PMC11103010" in source
    assert "parent.helpText" in source
    assert "parent.acknowledgementText" in source


def test_plate_rod_core_install_uses_pypi_binary_wheel_and_verifies_compiled_backend() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'slicer_pip_install("numpy scipy")' in source
    assert "scikit-image" not in source
    assert (
        '"--upgrade --force-reinstall --prefer-binary "'
    ) in source
    assert (
        '"--only-binary :all: --no-deps plate-rod-thinning>=0.1.6"'
    ) in source
    assert "_remove_local_core_repo_from_sys_path()" in source
    assert "Imported package path:" in source
    assert 'importlib.import_module("plate_rod_thinning._c_backend")' in source
    assert "Use Slicer's bundled Python 3.12 on macOS x86_64 or arm64" in source
    assert 'env["PLATE_ROD_BUILD_EXT"] = "1"' not in source
    assert '"build_ext", "--inplace"' not in source
    assert "Plate/Rod compiled backend build skipped" not in source
    assert "self._lastCoreInstallMessage = message" in source
    assert "subprocess" not in source
    assert "raise RuntimeError(completed.stdout.strip() or \"Could not build plate/rod compiled backend.\")" not in source
    assert "run_toolbox_update_dialog(self.logic.install_or_update_core" not in source
    assert "self.logic.install_or_update_core()" in source
    assert "Installing/updating compiled plate-rod core..." in source


def test_plate_rod_widget_exposes_pipeline_controls_and_outputs() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    widget_setup = source[source.index("    def setup(self):", source.index("class PlateRodMorphometryHRpQCTWidget")) :]

    assert "self.modeTabs = qt.QTabWidget()" in widget_setup
    assert 'self.modeTabs.addTab(scene_tab, "Scene")' in widget_setup
    assert 'self.modeTabs.addTab(batch_tab, "Batch")' in widget_setup
    assert "Folder Batch" not in widget_setup
    assert "def _segmentation_input_row(self, node_selector, segment_selector):" in source
    assert "row_layout = qt.QHBoxLayout(row)" in source
    assert "row_layout.addWidget(node_selector, 2)" in source
    assert "row_layout.addWidget(segment_selector, 1)" in source
    assert "Bone segmentation" in widget_setup
    assert "self.boneSegmentSelector = self._segment_combo()" in widget_setup
    assert "qMRMLSegmentSelectorWidget" not in widget_setup
    assert 'form_layout.addRow("Bone segmentation", self._segmentation_input_row(self.boneSegmentationSelector, self.boneSegmentSelector))' in widget_setup
    assert 'form_layout.addRow("Bone label", self.boneSegmentSelector)' not in widget_setup
    assert "Trabecular compartment mask" in widget_setup
    assert "self.trabecularSegmentSelector = self._segment_combo()" in widget_setup
    assert 'form_layout.addRow("Trabecular compartment mask", self._segmentation_input_row(self.trabecularMaskSelector, self.trabecularSegmentSelector))' in widget_setup
    assert 'form_layout.addRow("Trabecular label", self.trabecularSegmentSelector)' not in widget_setup
    assert "Common scan region mask" in widget_setup
    assert "self.commonRegionMaskSelector" in widget_setup
    assert "self.commonRegionSegmentSelector" in widget_setup
    assert "self.boneSegmentationSelector.currentNodeChanged.connect(self._refresh_bone_segment_selector)" in source
    assert "self.trabecularMaskSelector.currentNodeChanged.connect(self._refresh_trabecular_segment_selector)" in source
    assert "def _refresh_bone_segment_selector(self, node=None):" in source
    assert "def _refresh_trabecular_segment_selector(self, node=None):" in source
    assert "Slenderness" in widget_setup
    assert "Max thinning iterations" in widget_setup
    assert "self.maxIterationsSpinBox = qt.QSpinBox()" in widget_setup
    assert "Minimum plate voxels" in widget_setup
    assert "Minimum rod voxels" in widget_setup
    assert "self.useMetalCheckBox = qt.QCheckBox()" in widget_setup
    assert "self.useMetalCheckBox.checked = True" in widget_setup
    assert 'form_layout.addRow("Acceleration", self.useMetalCheckBox)' in widget_setup
    assert "Output prefix" in widget_setup
    assert "Run plate/rod morphometry" in widget_setup
    assert "self.progressBar = qt.QProgressBar()" in widget_setup
    assert "self.progressBar.setRange(0, 0)" in widget_setup
    assert "self.progressBar.visible = False" in widget_setup
    assert "self.statusLabel = qt.QLabel" in widget_setup
    assert "Install / update compiled plate-rod core" in widget_setup
    assert "def _refresh_core_status(self):" in source
    assert "self._refresh_core_status()" in widget_setup
    assert '"Skeleton topology labels": core_result.topology_labels' in source
    assert "Full-thickness labels" in source
    assert "Component labels" in source
    assert "vtkMRMLTableNode" in source
    assert "TOPOLOGY_LABEL_COLORS" in source
    assert 'set_labelmap_display_colors(node, map_role)' in source
    assert "show_full_thickness_labels_in_3d(nodes.get(\"Full-thickness labels\"))" in source


def test_plate_rod_batch_ui_uses_derivative_discovery_pattern() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    widget_setup = source[source.index("    def setup(self):", source.index("class PlateRodMorphometryHRpQCTWidget")) :]

    assert 'qt.QGroupBox("Discovery")' in widget_setup
    assert "self.folderDiscoverButton" in widget_setup
    assert "self.folderBatchTable" in widget_setup
    assert 'self.folderBatchTable.setHorizontalHeaderLabels(["Subject", "Site", "Sessions"])' in widget_setup
    assert "Subject filter" in widget_setup
    assert "Site filter" in widget_setup
    assert 'self.folderRunButton = qt.QPushButton("Run Batch")' in widget_setup


def test_plate_rod_run_passes_selected_segments_to_logic() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    run_method = source[source.index("    def _run_plate_rod_morphometry(self):") :]

    assert "self._activePlateRodJob = self.logic.prepare_plate_rod_job(" in run_method
    assert "self.logic.run_plate_rod_job(" in run_method
    assert "on_output=self._show_process_output" in run_method
    assert "on_finished=self._on_plate_rod_process_finished" in run_method
    assert "bone_segment_id=self._selected_segment_id(self.boneSegmentSelector)" in run_method
    assert "trabecular_segment_id=self._selected_segment_id(self.trabecularSegmentSelector)" in run_method
    assert "common_region_node=self.commonRegionMaskSelector.currentNode()" in run_method
    assert "common_region_segment_id=self._selected_segment_id(self.commonRegionSegmentSelector)" in run_method
    assert "use_metal=bool(self.useMetalCheckBox.checked)" in run_method
    assert "max_iterations=int(self.maxIterationsSpinBox.value)" in run_method
    assert "self._set_progress(True, \"Reading selected masks...\")" in run_method
    assert "self._set_progress(True, \"Running thinning and morphometry...\")" in run_method
    assert "self.runButton.enabled = False" in run_method


def test_plate_rod_widget_defines_progress_update_helper() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _set_progress(self, running, message):" in source
    assert "self.progressBar.visible = bool(running)" in source
    assert "self.statusLabel.text = str(message)" in source
    assert "slicer.app.processEvents()" in source


def test_plate_rod_module_runs_core_in_qprocess_for_responsive_slicer() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def prepare_plate_rod_job(" in source
    assert "def run_plate_rod_job(self, job, *, on_output=None, on_finished=None):" in source
    assert "proc = qt.QProcess()" in source
    assert "proc.readyReadStandardOutput.connect(_read_stdout)" in source
    assert "proc.readyReadStandardError.connect(_read_stderr)" in source
    assert "proc.finished.connect(_finished)" in source
    assert 'if bool(job.get("use_metal", True)):' in source
    assert 'env.insert("PLATE_ROD_USE_METAL_FULL", "1")' in source
    assert 'else:' in source
    assert 'env.insert("PLATE_ROD_USE_METAL_FULL", "0")' in source
    assert "proc.start(python_exe, [\"-c\", _PLATE_ROD_PROCESS_SCRIPT, str(job[\"job_json_path\"])])" in source
    assert "def _on_plate_rod_process_finished(self, exit_code, exit_status):" in source
    assert "self.logic.load_plate_rod_job_outputs(self._activePlateRodJob)" in source


def test_plate_rod_logic_passes_max_iterations_to_core_parameters() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "max_iterations=200," in source
    assert "max_iterations=int(max_iterations)," in source


def test_plate_rod_status_reports_metal_availability() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'metal_status = "Metal availability unknown"' in source
    assert 'from plate_rod_thinning import metal_backend' in source
    assert 'metal_backend.status()' in source
    assert '"Metal available"' in source
    assert '"Metal unavailable"' in source
    assert 'f"Plate/Rod core available ({version}, compiled backend; {metal_status})' in source


def test_plate_rod_logic_passes_trabecular_mask_and_spacing_for_summary_metrics() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "voxel_spacing_mm=tuple(float(value) for value in bone_image.GetSpacing())," in source
    assert "common_region_node=None" in source
    assert "common_region = self._volume_to_sitk_uint8(" in source
    assert "bone_image = clip_mask_to_region(bone_image, common_region)" in source
    assert "trab_image = clip_mask_to_region(trab_image, common_region)" in source
    assert '"common_region_path": str(common_region_path) if common_region_path else ""' in source
    assert "plate_rod_analysis(trabecular_bone, analysis_mask=trab, parameters=parameters)" in source


def test_plate_rod_module_sets_display_and_provenance_attributes() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'SetAttribute("BoneImaging.PlateRod.Engine", "plate_rod_thinning")' in source
    assert 'SetAttribute("BoneImaging.PlateRod.MapRole", map_role)' in source
    assert 'SetAttribute("BoneImaging.PlateRod.Slenderness"' in source
    assert 'SetAttribute("BoneImaging.PlateRod.CommonRegionNode"' in source
    assert "set_labelmap_display_colors" in source
    assert '"Plate", (0.0, 0.45, 1.0)' in source
    assert '"Rod", (1.0, 0.05, 0.02)' in source
    assert '"Junction", (0.9, 0.1, 0.25)' in source


def test_plate_rod_batch_delegates_to_package_batch_api() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "from plate_rod_thinning.batch import run_plate_rod_batch" in source
    assert "run_plate_rod_batch(" in source


def test_plate_rod_folder_batch_action_executes_package_api(monkeypatch, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("plate_rod_batch_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = {}
    monkeypatch.setattr(module, "run_plate_rod_batch", lambda root, **kwargs: received.update(root=root, **kwargs) or "done")

    result = module.PlateRodMorphometryHRpQCTLogic().run_folder_batch(tmp_path, use_common_region=False, force=True)

    assert result == "done"
    assert received == {"root": tmp_path, "use_common_region": False, "force": True, "progress": None}


def test_plate_rod_background_batch_command_carries_folder_options(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("plate_rod_batch_command_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    command = module.PlateRodMorphometryHRpQCTLogic.folder_batch_command(
        tmp_path, subject_id="S1", site="tibia", use_common_region=False, force=True,
    )

    assert command == [
        "-m", "plate_rod_thinning.cli", "run-batch", str(tmp_path.resolve()),
        "--subject", "S1", "--site", "tibia", "--no-common-region", "--force",
    ]


def test_plate_rod_background_job_launches_with_pythonslicer(monkeypatch, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("plate_rod_batch_launch_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    module.PlateRodMorphometryHRpQCTLogic().run_folder_batch_job(tmp_path, subject_id="S1", site="tibia", force=True)

    assert started[0][0].endswith("Contents/bin/PythonSlicer")
    assert started[0][1][:3] == ["-m", "plate_rod_thinning.cli", "run-batch"]
    assert ["--subject", "S1"] == started[0][1][4:6]
    assert "--force" in started[0][1]


def test_plate_rod_module_builds_3d_surface_preview_for_full_thickness_labels() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def show_full_thickness_labels_in_3d(labelmap_node):" in source
    assert "AddNewNodeByClass(" in source
    assert '"vtkMRMLSegmentationNode"' in source
    assert "ImportLabelmapToSegmentationNode(labelmap_node, segmentation_node)" in source
    assert "segmentation_node.CreateClosedSurfaceRepresentation()" in source
    assert "display_node.SetVisibility(True)" in source
    assert "display_node.SetVisibility3D(True)" in source
    assert "display_node.SetOpacity3D(0.65)" in source
    assert "segment.SetName(name)" in source
    assert "segment.SetColor(color[0], color[1], color[2])" in source
