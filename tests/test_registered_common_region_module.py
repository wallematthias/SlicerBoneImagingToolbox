from pathlib import Path
import importlib.util


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


def test_registered_common_region_batch_delegates_to_timelapsed_cli() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert '"-m", "timelapsedhrpqct.cli"' in source
    assert '"common-region", "run"' in source


def test_registered_common_region_builds_one_cli_command_per_selected_group() -> None:
    spec = importlib.util.spec_from_file_location("registered_common_region_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logic = module.RegisteredCommonRegionLogic()

    commands = logic.batch_cli_commands(
        {
            "dataset_root": "/tmp/dataset",
            "rows": [
                {"subject_id": "S1", "site": "tibia"},
                {"subject_id": "S1", "site": "tibia"},
                {"subject_id": "S2", "site": "radius"},
            ],
        }
    )

    assert commands == [
        ["-m", "timelapsedhrpqct.cli", "common-region", "run", str(Path("/tmp/dataset").resolve()), "--subject", "S1", "--site", "tibia"],
        ["-m", "timelapsedhrpqct.cli", "common-region", "run", str(Path("/tmp/dataset").resolve()), "--subject", "S2", "--site", "radius"],
    ]


def test_registered_common_region_reports_custom_root_constraint() -> None:
    spec = importlib.util.spec_from_file_location("registered_common_region_notice_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logic = module.RegisteredCommonRegionLogic()

    notice = logic.batch_output_root_notice(
        {"dataset_root": "/tmp/dataset", "derivatives_root": "/tmp/custom/CommonRegion"}
    )

    assert "custom derivatives root is not supported" in notice


def test_registered_common_region_batch_controller_launches_selected_groups() -> None:
    class FakeLogic:
        def __init__(self):
            self.prepared = None
            self.launched = None

        def discover_batch_series(self, _root):
            return [
                {"subject_id": "S1", "site": "tibia", "session_id": "1"},
                {"subject_id": "S2", "site": "radius", "session_id": "1"},
            ]

        def prepare_batch_job(self, root, derivatives_root, rows):
            self.prepared = (root, derivatives_root, rows)
            return {"rows": rows}

        def run_batch_job(self, job, **kwargs):
            self.launched = (job, kwargs)
            return "process"

    spec = importlib.util.spec_from_file_location("registered_common_region_controller_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logic = FakeLogic()
    controller = module.RegisteredCommonRegionBatchController(logic)
    controller.discover("/tmp/dataset")
    controller.set_selected_groups({("S2", "radius")})

    process = controller.run("/tmp/dataset", "/tmp/dataset/derivatives/CommonRegion")

    assert process == "process"
    assert logic.prepared[2] == [{"subject_id": "S2", "site": "radius", "session_id": "1"}]
    assert logic.launched[0] == {"rows": logic.prepared[2]}


def test_registered_common_region_batch_job_launches_with_pythonslicer(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("registered_common_region_launch_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = []

    class Signal:
        def connect(self, _callback): pass

    class Process:
        def __init__(self):
            self.readyReadStandardOutput = Signal()
            self.readyReadStandardError = Signal()
            self.finished = Signal()
        def setProcessEnvironment(self, _environment): pass
        def start(self, executable, arguments): started.append((executable, arguments))

    class Environment:
        @staticmethod
        def systemEnvironment(): return Environment()
        def insert(self, _key, _value): pass

    monkeypatch.setattr(module.qt, "QProcess", Process, raising=False)
    monkeypatch.setattr(module.qt, "QProcessEnvironment", Environment, raising=False)
    monkeypatch.setattr(module.slicer, "app", type("App", (), {"applicationFilePath": staticmethod(lambda: "/Applications/Slicer.app/Contents/MacOS/Slicer")})(), raising=False)
    module.RegisteredCommonRegionLogic().run_batch_job(
        {"dataset_root": "/tmp/dataset", "rows": [{"subject_id": "S1", "site": "tibia"}]}
    )

    assert started[0][0].endswith("Contents/bin/PythonSlicer")
    assert started[0][1] == ["-m", "timelapsedhrpqct.cli", "common-region", "run", str(Path("/tmp/dataset").resolve()), "--subject", "S1", "--site", "tibia"]
