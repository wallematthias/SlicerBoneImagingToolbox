from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAROSOL_MODULE = ROOT / "HRpQCTTools" / "ParOSolFEA" / "ParOSolFEA.py"
MECHREG_MODULE = ROOT / "HRpQCTTools" / "MechanoregulationHRpQCT" / "MechanoregulationHRpQCT.py"


def test_parosol_module_has_public_metadata_and_root_resolution() -> None:
    source = PAROSOL_MODULE.read_text(encoding="utf-8")

    assert 'parent.title = "ParOSol FEA"' in source
    assert 'parent.categories = ["Bone Imaging.FEA"]' in source
    assert "Private ParOSol" not in source
    assert "extension_root = module_path.parents[2]" in source
    assert "def _active_repositories_root" in source
    bootstrap_body = source.split("def _bootstrap_parosol_source_import_paths", 1)[1].split("\ndef ", 1)[0]
    assert '_active_repositories_root(extension_root) / "parosol-py"' not in bootstrap_body
    assert "SLICER_PAROSOL_SOURCE" in bootstrap_body


def test_mechanoregulation_module_has_public_metadata_and_root_resolution() -> None:
    source = MECHREG_MODULE.read_text(encoding="utf-8")

    assert 'parent.title = "Bone Mechanoregulation"' in source
    assert 'parent.categories = ["Bone Imaging.Mechanoregulation"]' in source
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


def test_parosol_module_contains_artifact_discovery_batch_tab() -> None:
    source = PAROSOL_MODULE.read_text(encoding="utf-8")

    assert "discover_fea_batch_cases" in source
    assert "build_parosol_case_commands" in source
    assert 'self.batchPage, batch_page_layout = self._workflow_tab_page("Batch")' in source
    assert "self.batchDiscoverButton" in source
    assert "self.batchRunButton" in source
    assert "def discover_fea_batch(self):" in source
    assert "def run_fea_batch(self):" in source
    assert "def _run_next_fea_batch_case(self):" in source


def test_mechanoregulation_module_contains_derivative_discovery_helpers() -> None:
    source = MECHREG_MODULE.read_text(encoding="utf-8")

    assert "def discover_mechanoregulation_manifests" in source
    assert "def _mechanoregulation_derivative_roots" in source
    assert '"Mechanoregulation"' in source
    assert '"FEA"' in source
    discover_body = source.split("    def discover_cases(self, path):", 1)[1].split("\n    def ", 1)[0]
    assert "discover_mechanoregulation_manifests(root)" in discover_body


def test_mechanoregulation_ui_uses_batch_and_review_tabs() -> None:
    source = MECHREG_MODULE.read_text(encoding="utf-8")
    setup_source = source[source.index("    def setup(self):", source.index("class MechanoregulationHRpQCTWidget")) :]

    assert "self.modeTabs = qt.QTabWidget()" in setup_source
    assert 'self.modeTabs.addTab(batch_tab, "Batch")' in setup_source
    assert 'self.modeTabs.addTab(review_tab, "Review")' in setup_source
    assert 'box.text = "Batch"' in source
    assert 'box.text = "Review"' in source
    assert 'self.runButton = qt.QPushButton("Run Batch")' in source
