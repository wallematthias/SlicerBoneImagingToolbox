from pathlib import Path
from types import SimpleNamespace
from importlib import metadata

import pytest

from SlicerBoneImagingToolboxLib import slicer_pip


def test_update_pins_installed_simpleitk_and_cleans_up_constraint(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda name: "2.5.2")
    captured = {}

    def run(args, **kwargs):
        constraint = Path(args[args.index("--constraint") + 1])
        captured["path"] = constraint
        assert constraint.read_text() == "SimpleITK==2.5.2\n"
        assert "--upgrade" in args
        assert "voidspace>=0.1.4" in args
        assert "ITK_AUTOLOAD_PATH" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="Installed")

    monkeypatch.setattr(slicer_pip.subprocess, "run", run)
    assert slicer_pip.slicer_pip_install("--upgrade voidspace>=0.1.4") == "Installed"
    assert not captured["path"].exists()


def test_dependency_reinstall_is_rejected_before_modifying_slicer(monkeypatch):
    monkeypatch.setattr(slicer_pip.subprocess, "run", lambda *a, **k: pytest.fail("pip must not run"))
    with pytest.raises(ValueError, match="--no-deps"):
        slicer_pip.slicer_pip_install("--force-reinstall bone-contouring")


def test_nodeps_compiled_wheel_reinstall_remains_supported(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda name: "2.5.2")
    calls = []
    monkeypatch.setattr(slicer_pip.subprocess, "run", lambda args, **kwargs: (
        calls.append(args) or SimpleNamespace(returncode=0, stdout="Installed")
    ))
    slicer_pip.slicer_pip_install("--force-reinstall --no-deps plate-rod-thinning")
    assert "--force-reinstall" in calls[0]
    assert "--no-deps" in calls[0]


def test_pip_uses_python_slicer_launcher_not_unconfigured_python_real(monkeypatch, tmp_path):
    launcher = tmp_path / "PythonSlicer"
    launcher.touch()
    monkeypatch.setattr(slicer_pip.sys, "executable", str(tmp_path / "python-real"))
    monkeypatch.setattr(metadata, "version", lambda name: "2.5.2")
    calls = []
    monkeypatch.setattr(slicer_pip.subprocess, "run", lambda args, **kwargs: (
        calls.append(args) or SimpleNamespace(returncode=0, stdout="Installed")
    ))
    slicer_pip.slicer_pip_install("--dry-run SimpleITK>=2.3")
    assert calls[0][0] == str(launcher)


def test_missing_simpleitk_fails_without_installing_a_generic_wheel(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    monkeypatch.setattr(slicer_pip.subprocess, "run", lambda *a, **k: pytest.fail("pip must not run"))
    with pytest.raises(RuntimeError, match="Slicer"):
        slicer_pip.slicer_pip_install("bone-contouring")


def test_all_module_installers_use_protected_pip_helper():
    root = Path(__file__).resolve().parents[1]
    for family in ("HRpQCTTools", "CTTools", "IOTools", "Setup"):
        for path in (root / family).rglob("*.py"):
            assert "slicer.util.pip_install(" not in path.read_text(), path


def test_timelapsed_does_not_disable_scene_image_io_or_global_warnings():
    import ast

    path = Path(__file__).resolve().parents[1] / "HRpQCTTools/TimelapsedHRpQCT/TimelapsedHRpQCT.py"
    source = path.read_text()
    assert "SetGlobalWarningDisplay(False)" not in source
    tree = ast.parse(source)
    # CLI subprocesses may have their own environment; the Slicer host must not.
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            if ast.unparse(node.value) == "os.environ":
                assert isinstance(node.slice, ast.Constant)
                assert node.slice.value not in ("ITK_AUTOLOAD_PATH", "SITK_AUTOLOAD_PATH")


@pytest.mark.parametrize("method_name", ["install_wheel_into_slicer", "install_pypi_into_slicer"])
def test_parosol_installs_are_constrained_and_wheel_refresh_is_targeted(monkeypatch, method_name):
    import ast

    path = Path(__file__).resolve().parents[1] / "HRpQCTTools/ParOSolFEA/ParOSolFEA.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {
        "Path": Path, "subprocess": slicer_pip.subprocess,
        "slicer_pip_constraints": slicer_pip.slicer_pip_constraints,
        "_filter_runtime_noise": lambda value: value,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setattr(metadata, "version", lambda name: "2.5.2")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        constraint = Path(args[args.index("--constraint") + 1])
        assert constraint.read_text() == "SimpleITK==2.5.2\n"
        if "--force-reinstall" in args:
            assert "--no-deps" in args
        return SimpleNamespace(returncode=0, stdout="Installed", stderr="")

    monkeypatch.setattr(slicer_pip.subprocess, "run", run)
    logic = SimpleNamespace(
        compatible_wheel_message=lambda value: (True, "Compatible"),
        python_launcher=lambda: "PythonSlicer",
        parosol_environment=lambda: {},
        slicer_local_parosol=lambda: "parosol",
        check_runtime=lambda **kwargs: "Ready",
    )
    positional = ["/tmp/parosol.whl"] if method_name == "install_wheel_into_slicer" else []
    assert namespace[method_name](logic, *positional) == "Ready"
    if positional:
        assert len(calls) == 2
        assert "--force-reinstall" in calls[1]


def test_motion_reinstall_refreshes_only_core_after_resolving_dependencies():
    import ast

    path = Path(__file__).resolve().parents[1] / "HRpQCTTools/MotionScoreHRpQCT/MotionScoreHRpQCT.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_force_reinstall_core_package")
    namespace = {"CORE_PYPI_PACKAGE": "motionscorehrpqct", "MIN_CORE_VERSION": "2.5.11"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    calls = []
    widget = SimpleNamespace(
        _pip_install=lambda *packages, **options: calls.append((packages, options)),
        _core_pip_requirements=lambda: ["motionscorehrpqct>=2.5.11", "numpy>=1.26"],
        _set_license_status=lambda message: None,
        _log=lambda message: None,
        _core_package_ready=lambda: True,
        _update_setup_status=lambda: None,
    )
    assert namespace[method.name](widget)
    assert len(calls) == 2
    assert "--force-reinstall" not in calls[0][1]["extra_args"]
    assert calls[1][0] == ("motionscorehrpqct>=2.5.11",)
    assert "--force-reinstall" in calls[1][1]["extra_args"]
    assert "--no-deps" in calls[1][1]["extra_args"]
