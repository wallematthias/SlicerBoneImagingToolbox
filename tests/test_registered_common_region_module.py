from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "HRpQCTTools" / "RegisteredCommonRegion" / "RegisteredCommonRegion.py"


def test_registered_common_region_module_is_registered_with_manifest_and_cmake():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = (ROOT / "toolbox_modules.json").read_text(encoding="utf-8")
    link_script = (ROOT / "scripts" / "link_local_toolbox_modules.py").read_text(encoding="utf-8")

    assert "add_subdirectory(HRpQCTTools/RegisteredCommonRegion)" in cmake
    assert '"path": "HRpQCTTools/RegisteredCommonRegion"' in manifest
    assert '"title": "Registered Common Region"' in manifest
    assert '"section": "HR-pQCT"' in manifest
    assert '"HRpQCTTools/RegisteredCommonRegion"' in link_script
    assert '"RegisteredCommonRegion"' in link_script


def test_registered_common_region_module_defines_slicer_classes():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "class RegisteredCommonRegion(ScriptedLoadableModule):" in source
    assert "class RegisteredCommonRegionLogic(ScriptedLoadableModuleLogic):" in source
    assert "class RegisteredCommonRegionWidget(ScriptedLoadableModuleWidget):" in source
    assert "class RegisteredCommonRegionTest(ScriptedLoadableModuleTest):" in source
    assert 'parent.title = "Registered Common Region"' in source
    assert 'parent.categories = ["Bone Imaging.HR-pQCT"]' in source


def test_registered_common_region_widget_has_scene_and_batch_tabs():
    source = MODULE_PATH.read_text(encoding="utf-8")
    setup = source[source.index("    def setup(self):") :]

    assert "self.tabs = qt.QTabWidget()" in setup
    assert 'self.tabs.addTab(scene_tab, "Scene")' in setup
    assert 'self.tabs.addTab(batch_tab, "Batch")' in setup
    assert "Baseline volume" in setup
    assert "Follow-up volume" in setup
    assert "Dataset root" in setup
    assert "Derivatives root" in setup
    assert "Discover" in setup
    assert "Run" in setup


def test_registered_common_region_batch_runs_in_background_qprocess():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def run_batch_job(self, job, *, on_output=None, on_finished=None):" in source
    assert "qt.QProcess()" in source
    assert 'env.insert("PYTHONUNBUFFERED", "1")' in source
    assert "proc.readyReadStandardOutput.connect(_read_stdout)" in source
    assert "proc.readyReadStandardError.connect(_read_stderr)" in source
    assert "proc.finished.connect(_finished)" in source
    assert "--registered-common-region-job" in source
    assert "build_common_scan_region" in source
    assert "write_manifest" in source
