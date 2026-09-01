from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from SlicerBoneImagingToolboxLib.package_status import (
    DEFAULT_RUNTIME_PACKAGES,
    PackageSpec,
    compare_versions,
    install_command,
    install_commands,
    package_status_row,
    resolve_local_editable_repo,
)
from SlicerBoneImagingToolboxLib.slicer_pip import clean_pip_environment


def test_compare_versions_detects_newer_pypi_release() -> None:
    assert compare_versions("2.0.37", "2.0.38") == "outdated"
    assert compare_versions("2.0.38", "2.0.38") == "current"
    assert compare_versions("2.0.39", "2.0.38") == "current"


def test_package_status_row_marks_missing_package_installable() -> None:
    spec = PackageSpec(
        display_name="Timelapsed HR-pQCT",
        package_name="timelapsed-hrpqct",
        import_name="timelapsedhrpqct",
        minimum_version="2.0.37",
    )

    row = package_status_row(
        spec,
        installed_versions={},
        latest_versions={"timelapsed-hrpqct": "2.0.42"},
        validation_errors={},
    )

    assert row.installed_version is None
    assert row.latest_version == "2.0.42"
    assert row.status == "missing"
    assert row.action == "install"


def test_package_status_row_marks_update_available() -> None:
    spec = PackageSpec(
        display_name="MotionScore HR-pQCT",
        package_name="motionscorehrpqct",
        import_name="motionscore",
        minimum_version="2.5.8",
    )

    row = package_status_row(
        spec,
        installed_versions={"motionscorehrpqct": "2.5.8"},
        latest_versions={"motionscorehrpqct": "2.5.9"},
        validation_errors={},
    )

    assert row.installed_version == "2.5.8"
    assert row.latest_version == "2.5.9"
    assert row.status == "update_available"
    assert row.action == "update"


def test_package_status_row_handles_unavailable_pypi_version() -> None:
    spec = PackageSpec(
        display_name="Scanco I/O",
        package_name="aimio-py",
        import_name="aim_io",
        minimum_version="0.1.8",
    )

    row = package_status_row(
        spec,
        installed_versions={"aimio-py": "0.1.8"},
        latest_versions={},
        validation_errors={},
    )

    assert row.installed_version == "0.1.8"
    assert row.latest_version is None
    assert row.status == "current_unknown_latest"
    assert row.action is None


def test_package_status_row_marks_installed_version_below_minimum_as_update_needed() -> None:
    spec = PackageSpec(
        display_name="Timelapsed HR-pQCT",
        package_name="timelapsed-hrpqct",
        import_name="timelapsedhrpqct",
        minimum_version="2.0.37",
    )

    row = package_status_row(
        spec,
        installed_versions={"timelapsed-hrpqct": "2.0.12"},
        latest_versions={},
        validation_errors={},
    )

    assert row.installed_version == "2.0.12"
    assert row.latest_version is None
    assert row.status == "update_available"
    assert row.action == "update"


def test_install_command_uses_upgrade_when_package_is_installed() -> None:
    spec = PackageSpec(
        display_name="Spine Segmentation",
        package_name="spine-segment",
        import_name="spine_segment",
        minimum_version="0.1.0",
        constraints=("numpy>=1.26,<3.0",),
    )

    assert install_command(spec, installed=False) == "--prefer-binary spine-segment>=0.1.0 numpy>=1.26,<3.0"
    assert (
        install_command(spec, installed=True)
        == "--upgrade --prefer-binary spine-segment>=0.1.0 numpy>=1.26,<3.0"
    )


def test_microarchitecture_install_commands_use_local_logic_repo_when_available() -> None:
    specs = {spec.package_name: spec for spec in DEFAULT_RUNTIME_PACKAGES}
    commands = install_commands(specs["bone-microarchitecture"], installed=True)

    assert commands[0] == "--upgrade --prefer-binary numpy>=2.0,<3.0 scipy>=1.18,<2.0"
    assert commands[1].startswith("--upgrade --no-deps -e ")
    assert commands[1].endswith("/bone-microarchitecture")


def test_default_runtime_packages_include_public_tool_cores() -> None:
    package_names = {spec.package_name for spec in DEFAULT_RUNTIME_PACKAGES}

    assert "timelapsed-hrpqct" in package_names
    assert "motionscorehrpqct" in package_names
    assert "aimio-py" in package_names
    assert "bone-contouring" in package_names
    assert "spine-segment" in package_names
    assert "bone-microarchitecture" in package_names
    assert "plate-rod-thinning" in package_names
    assert "parosol-py" in package_names
    assert "bone-mechanoregulation" in package_names
    assert ("or" + "mir-xct") not in package_names


def test_timelapsed_runtime_package_has_user_facing_setup_name() -> None:
    specs = {spec.package_name: spec for spec in DEFAULT_RUNTIME_PACKAGES}

    assert specs["timelapsed-hrpqct"].display_name == "Timelapsed HR-pQCT"
    assert "timelapsed-hrpqct" in specs["timelapsed-hrpqct"].notes


def test_bone_contouring_runtime_package_has_user_facing_setup_name() -> None:
    specs = {spec.package_name: spec for spec in DEFAULT_RUNTIME_PACKAGES}

    assert specs["bone-contouring"].display_name == "Bone Contouring"
    assert specs["bone-contouring"].import_name == "bone_contouring"
    assert "segmentation" in specs["bone-contouring"].notes.lower()


def test_microarchitecture_runtime_package_has_user_facing_setup_name() -> None:
    specs = {spec.package_name: spec for spec in DEFAULT_RUNTIME_PACKAGES}

    assert specs["bone-microarchitecture"].display_name == "Bone Microarchitecture"
    assert specs["bone-microarchitecture"].import_name == "bone_microarchitecture"
    assert "bone-microarchitecture" in specs


def test_plate_rod_runtime_package_requires_compiled_backend_and_binary_reinstall() -> None:
    specs = {spec.package_name: spec for spec in DEFAULT_RUNTIME_PACKAGES}
    spec = specs["plate-rod-thinning"]

    assert spec.display_name == "Plate/Rod Morphometry"
    assert spec.import_name == "plate_rod_thinning"
    assert spec.required_imports == ("plate_rod_thinning._c_backend",)
    assert install_command(spec, installed=True) == (
        "--upgrade --force-reinstall --prefer-binary --only-binary :all: --no-deps "
        "plate-rod-thinning>=0.1.6"
    )


def test_plate_rod_install_commands_do_not_emit_empty_dependency_install() -> None:
    specs = {spec.package_name: spec for spec in DEFAULT_RUNTIME_PACKAGES}
    commands = install_commands(specs["plate-rod-thinning"], installed=False)

    assert len(commands) == 1
    assert commands[0].startswith("--no-deps -e ")
    assert commands[0].endswith("/bone-plate-rod-thinning")
    assert "--prefer-binary" not in commands[0]


def test_fea_and_mechanoregulation_runtime_packages_have_public_setup_names() -> None:
    specs = {spec.package_name: spec for spec in DEFAULT_RUNTIME_PACKAGES}

    assert specs["parosol-py"].display_name == "ParOsol-FEA"
    assert specs["parosol-py"].import_name == "parosol_py"
    assert "ParOSol" in specs["parosol-py"].notes

    assert specs["bone-mechanoregulation"].display_name == "Bone Mechanoregulation"
    assert specs["bone-mechanoregulation"].import_name == "bonemechreg"
    assert "mechanoregulation" in specs["bone-mechanoregulation"].notes.lower()


def test_public_fea_packages_do_not_install_from_adjacent_editable_checkouts() -> None:
    specs = {spec.package_name: spec for spec in DEFAULT_RUNTIME_PACKAGES}

    for package_name in ("parosol-py", "bone-mechanoregulation"):
        command_text = "\n".join(install_commands(specs[package_name], installed=False))
        assert " -e " not in command_text
        assert "/active/" not in command_text


def test_package_status_marks_invalid_runtime_import_as_update_needed() -> None:
    spec = PackageSpec(
        display_name="Plate/Rod Morphometry",
        package_name="plate-rod-thinning",
        import_name="plate_rod_thinning",
        minimum_version="0.1.2",
        required_imports=("plate_rod_thinning._c_backend",),
    )

    row = package_status_row(
        spec,
        installed_versions={"plate-rod-thinning": "0.1.0"},
        latest_versions={"plate-rod-thinning": "0.1.0"},
        validation_errors={"plate-rod-thinning": "Required runtime module is unavailable."},
    )

    assert row.status == "update_available"
    assert row.action == "update"
    assert row.detail == "Required runtime module is unavailable."


def test_clean_pip_environment_removes_stale_compiler_overrides() -> None:
    env = clean_pip_environment(
        {
            "CC": "/Applications/Xcode-26.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang",
            "CXX": "/Applications/Xcode-26.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++",
            "PYTHONPATH": "/tmp/not-for-slicer-pip",
            "PATH": "/usr/bin",
        }
    )

    assert "CC" not in env
    assert "CXX" not in env
    assert "PYTHONPATH" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_default_runtime_packages_include_shared_derivative_contract() -> None:
    package_names = {spec.package_name for spec in DEFAULT_RUNTIME_PACKAGES}

    assert "bone-imaging-derivatives" in package_names


def test_shared_derivative_contract_install_uses_local_worktree_when_available() -> None:
    spec = next(spec for spec in DEFAULT_RUNTIME_PACKAGES if spec.package_name == "bone-imaging-derivatives")

    commands = install_commands(spec, installed=False)

    assert len(commands) == 1
    assert "-e" in commands[0]
    assert "bone-imaging-derivatives" in commands[0]


def test_local_editable_repo_resolves_active_worktree_layout(tmp_path: Path) -> None:
    toolbox_root = tmp_path / "active" / "SlicerBoneImagingToolbox" / ".worktrees" / "derivatives-overhaul"
    local_repo = tmp_path / "active" / "bone-microarchitecture" / ".worktrees" / "derivative-batch"
    local_repo.mkdir(parents=True)
    (local_repo / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    assert resolve_local_editable_repo(toolbox_root, "bone-microarchitecture") == local_repo
