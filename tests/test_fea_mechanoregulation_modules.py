from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAROSOL_MODULE = ROOT / "HRpQCTTools" / "ParOSolFEA" / "ParOSolFEA.py"
MECHREG_MODULE = ROOT / "HRpQCTTools" / "MechanoregulationHRpQCT" / "MechanoregulationHRpQCT.py"


def test_parosol_module_has_public_metadata_and_root_resolution() -> None:
    source = PAROSOL_MODULE.read_text(encoding="utf-8")
    icon_path = ROOT / "HRpQCTTools" / "ParOSolFEA" / "Resources" / "Icons" / "ParOSolFEA.png"

    assert 'parent.title = "ParOsol-FEA"' in source
    assert 'parent.categories = ["Bone Imaging.FE Analysis"]' in source
    assert icon_path.is_file()
    assert "parent.icon = qt.QIcon(str(Path(__file__).with_name(\"Resources\") / \"Icons\" / \"ParOSolFEA.png\"))" in source
    assert "Author: Matthias Walle" in source
    assert "Private ParOSol" not in source
    assert "extension_root = module_path.parents[2]" in source
    assert "def _active_repositories_root" in source
    bootstrap_body = source.split("def _bootstrap_parosol_source_import_paths", 1)[1].split("\ndef ", 1)[0]
    assert '_active_repositories_root(extension_root) / "parosol-py"' not in bootstrap_body
    assert "SLICER_PAROSOL_SOURCE" in bootstrap_body


def test_parosol_workflow_profiles_use_shared_profile_registry() -> None:
    source = PAROSOL_MODULE.read_text(encoding="utf-8")

    assert "tool_profile_dir(tool)" in source
    assert 'USER_WORKFLOW_ROOT = _shared_profile_tool_root("parosol-fea")' in source
    assert "def _available_user_workflows" in source
    assert 'list_profiles("parosol-fea")' in source
    assert "register_profile_asset(" in source
    assert '"parosol-workflow"' in source
    assert "SlicerParOSolTemplates" not in source


def test_mechanoregulation_module_has_public_metadata_and_root_resolution() -> None:
    source = MECHREG_MODULE.read_text(encoding="utf-8")

    assert 'parent.title = "Mechanoregulation"' in source
    assert 'parent.categories = ["Bone Imaging.Microstructural Analysis"]' in source
    assert "Private Slicer wrapper" not in source
    assert "TOOLBOX_ROOT = Path(__file__).resolve().parents[2]" in source
    assert 'CORE_REQUIREMENT = "bone-mechanoregulation"' in source
    assert "def _active_repositories_root" in source
    assert 'CORE_LOCAL_REPO = _active_repositories_root(TOOLBOX_ROOT) / "BoneMechanoregulation"' in source
    prefer_body = source.split("def _prefer_local_core", 1)[1].split("\ndef ", 1)[0]
    assert "_use_local_core_checkout()" in prefer_body


def test_migrated_modules_do_not_import_private_toolbox_library() -> None:
    combined = "\n".join(
        [
            PAROSOL_MODULE.read_text(encoding="utf-8"),
            MECHREG_MODULE.read_text(encoding="utf-8"),
        ]
    )

    assert "SlicerBoneImagingToolboxPrivateLib" not in combined
    assert "BoneImagingPrivateInstaller" not in combined


def test_parosol_module_contains_derivative_output_helpers() -> None:
    source = PAROSOL_MODULE.read_text(encoding="utf-8")

    assert "def _default_fea_derivative_root" in source
    assert "def _write_fea_derivative_manifest" in source
    assert "DerivativeManifest.create(" in source
    assert "DerivativeRecord(" in source
    assert "write_shared_manifest(" in source
    assert '"backend": "parosol"' in source
    load_results_body = source.split("    def load_results(self):", 1)[1].split("\n    def ", 1)[0]
    assert "self._fea_derivative_context(output_dir)" in load_results_body
    assert "_write_parosol_run_derivative_manifest(output_dir, **context)" in load_results_body


def test_parosol_fea_manifest_writer_merges_existing_records() -> None:
    source = PAROSOL_MODULE.read_text(encoding="utf-8")
    helper_body = source.split("def _write_fea_derivative_manifest", 1)[1].split("\ndef ", 1)[0]

    assert "read_shared_manifest(output_path)" in helper_body
    assert "if not incoming_records:" in helper_body
    assert "merged_by_id" in helper_body
    assert "merged_by_id[record.record_id] = record" in helper_body


def test_parosol_derivative_root_helper_uses_bids_like_tokens() -> None:
    source = PAROSOL_MODULE.read_text(encoding="utf-8")
    helper_body = source.split("def _default_fea_derivative_root", 1)[1].split("\ndef ", 1)[0]

    assert 'f"sub-{subject}"' in helper_body
    assert 'f"site-{site_name}"' in helper_body


def test_parosol_scene_ui_requires_explicit_derivative_dataset_context() -> None:
    source = PAROSOL_MODULE.read_text(encoding="utf-8")

    assert "self.derivativeDatasetRootSelector = ctk.ctkPathLineEdit()" in source
    assert "self.derivativeSubjectEdit = qt.QLineEdit()" in source
    assert "self.derivativeSiteEdit = qt.QLineEdit()" in source
    assert "self.derivativeSessionEdit = qt.QLineEdit()" in source
    assert "def _fea_derivative_context(self, output_dir):" in source


def test_parosol_module_keeps_artifact_batch_helpers_without_exposing_tab() -> None:
    source = PAROSOL_MODULE.read_text(encoding="utf-8")
    setup_start = source.index("    def setup(self):", source.index("class ParOSolFEAWidget"))
    setup_end = source.index("\n    def ", setup_start + len("    def setup(self):"))
    setup_source = source[setup_start:setup_end]

    assert "discover_fea_batch_cases" in source
    assert "build_parosol_case_commands" in source
    assert "batch_profile_support_status" in source
    assert 'self.batchPage, batch_page_layout = self._workflow_tab_page("Batch")' not in source
    assert "self.batchDiscoverButton = qt.QPushButton" not in setup_source
    assert "self.batchRunButton = qt.QPushButton" not in setup_source
    assert "self.batchStopButton = qt.QPushButton" not in setup_source
    assert "self.batchTable = qt.QTableWidget" not in setup_source
    assert "self._feaBatchQueue = []" not in setup_source
    assert "self._feaBatchCurrent = None" not in setup_source
    assert "def _queue_fea_batch_row(self, row):" in source
    assert "def _start_next_fea_batch_job(self):" in source
    assert "def _load_fea_batch_row_outputs(self, row):" in source
    assert "def discover_fea_batch(self):" in source
    assert "def run_fea_batch(self):" in source


def test_mechanoregulation_module_contains_derivative_discovery_helpers() -> None:
    source = MECHREG_MODULE.read_text(encoding="utf-8")

    assert "def discover_mechanoregulation_manifests" in source
    assert "def _mechanoregulation_derivative_roots" in source
    assert '"Mechanoregulation"' in source
    assert '"FEA"' in source
    discover_body = source.split("    def discover_cases(self, path):", 1)[1].split("\n    def ", 1)[0]
    assert "discover_mechanoregulation_manifests(root)" in discover_body


def test_mechanoregulation_ui_uses_scene_and_review_tabs_only() -> None:
    source = MECHREG_MODULE.read_text(encoding="utf-8")
    setup_start = source.index("    def setup(self):", source.index("class MechanoregulationHRpQCTWidget"))
    setup_end = source.index("\n    def ", setup_start + len("    def setup(self):"))
    setup_source = source[setup_start:setup_end]

    assert "self.modeTabs = qt.QTabWidget()" in setup_source
    assert 'self.modeTabs.addTab(scene_tab, "Scene")' in setup_source
    assert 'self.modeTabs.addTab(batch_tab, "Batch")' not in setup_source
    assert 'self.modeTabs.addTab(review_tab, "Review")' in setup_source
    assert 'box.text = "Batch"' not in setup_source
    assert 'box.text = "Scene"' in source
    assert 'box.text = "Review"' in source
    assert 'self.runButton = qt.QPushButton("Run")' in source
    assert 'self.batchDiscoveryGroup = qt.QGroupBox("Discovery")' not in setup_source
    assert 'self.batchWorkflowGroup = qt.QGroupBox("Workflow")' not in setup_source
    assert 'self.sceneDiscoveryGroup = qt.QGroupBox("Discovery")' in source
    assert 'self.sceneWorkflowGroup = qt.QGroupBox("Workflow")' in source
    assert "self.sceneProgressBar = qt.QProgressBar()" in source
    assert "self.sceneCurrentStepLabel = qt.QLabel(\"Current step: idle\")" in source
    scene_body = source.split("    def _build_scene_section", 1)[1].split("\n    def ", 1)[0]
    assert scene_body.index("self.sceneDiscoveryGroup") < scene_body.index("self.sceneRemodellingSelector")
    assert scene_body.index("self.sceneRemodellingSelector") < scene_body.index("self.sceneWorkflowGroup")
    assert scene_body.index("self.sceneWorkflowGroup") < scene_body.index("self.sceneStatusLabel")
    assert scene_body.index("self.sceneStatusLabel") < scene_body.index("self.sceneRunButton")


def test_mechanoregulation_scene_mode_discovers_loaded_nodes_and_runs_case_api() -> None:
    source = MECHREG_MODULE.read_text(encoding="utf-8")

    assert "self.sceneDiscoverButton = qt.QPushButton(\"Discover\")" in source
    assert "self.sceneRunButton = qt.QPushButton(\"Run\")" in source
    assert "self.sceneStopButton = qt.QPushButton(\"Stop\")" in source
    assert "self.sceneRemodellingSelector = slicer.qMRMLNodeComboBox()" in source
    assert "self.sceneSedSelector = slicer.qMRMLNodeComboBox()" in source
    assert "self.sceneAnalysisMaskSelector = slicer.qMRMLNodeComboBox()" in source
    assert "self.sceneProgressBar.visible = True" in source
    assert "self.sceneProgressBar.setRange" in source
    assert "self.sceneCurrentStepLabel.text = self._status_text(message)" in source
    assert "if text.startswith(\"[scene]\")" in source
    assert "\"Remodelling map\"" in source
    assert "\"ParOSol / FEA SED\"" in source
    scene_body = source.split("    def _build_scene_section", 1)[1].split("\n    def ", 1)[0]
    assert "\"Baseline Seg\"" not in scene_body
    assert "\"Trab\"" not in scene_body
    assert "\"Cort\"" not in scene_body
    assert "\"Full\"" not in scene_body
    assert "def discover_scene_cases(self):" in source
    assert "def _scene_volume_nodes" in source
    assert "def _scene_remodelling_candidates" in source
    assert "def _scene_sed_candidates" in source
    assert "def _scene_parosol_output_candidates" in source
    assert "def _stage_scene_case(self, row):" in source
    assert "slicer.util.saveNode" in source
    assert "TimelapseCase(" in source
    assert "run_post_timelapse_case(" in source
    assert "baseline_sed_path" in source
    assert "outputs[\"sed\"].write_bytes" in source
    assert "self._load_scene_mechanoregulation_outputs" in source


def test_mechanoregulation_scene_mode_consumes_loaded_parosol_outputs_without_fea_generation() -> None:
    source = MECHREG_MODULE.read_text(encoding="utf-8")
    scene_body = source.split("    def _build_scene_section", 1)[1].split("\n    def ", 1)[0]
    discover_body = source.split("    def discover_scene_cases(self):", 1)[1].split("\n    def ", 1)[0]
    stage_body = source.split("    def _stage_scene_case(self, row):", 1)[1].split("\n    def ", 1)[0]

    assert 'scene_inputs.addRow("Remodelling map", self.sceneRemodellingSelector)' in scene_body
    assert 'scene_inputs.addRow("ParOSol / FEA SED", self.sceneSedSelector)' in scene_body
    assert 'scene_inputs.addRow("Analysis mask", self.sceneAnalysisMaskSelector)' in scene_body
    assert "self._scene_parosol_output_candidates()" in discover_body
    assert "baseline_sed_path = self._save_scene_node(self._scene_node_from_combo(row, 2)" in stage_body
    assert "analysis_mask_path = self._save_scene_node(" in stage_body
    assert '"generate"' not in stage_body
    assert "Generate baseline SED requires" not in source
    assert "self.sceneRemodellingSelector.setCurrentNode(remodelling_nodes[0])" in source
    assert "self.sceneSedSelector.setCurrentNode(sed_nodes[0])" in source
