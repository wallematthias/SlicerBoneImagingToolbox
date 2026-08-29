from __future__ import annotations

from dataclasses import dataclass
import json
from importlib import metadata
from pathlib import Path
from urllib import request as urllib_request


PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
HTTP_HEADERS = {"User-Agent": "SlicerBoneImagingToolbox-Setup/1.0"}


@dataclass(frozen=True)
class PackageSpec:
    display_name: str
    package_name: str
    import_name: str
    minimum_version: str
    install_options: tuple[str, ...] = ("--prefer-binary",)
    constraints: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class PackageStatusRow:
    spec: PackageSpec
    installed_version: str | None
    latest_version: str | None
    status: str
    action: str | None
    detail: str = ""


DEFAULT_RUNTIME_PACKAGES = (
    PackageSpec(
        display_name="XCT2 Analysis",
        package_name="timelapsed-hrpqct",
        import_name="timelapsedhrpqct",
        minimum_version="2.0.37",
        constraints=("hrpqct-geodesic-contour>=0.1.1",),
        notes="XCT2/longitudinal HR-pQCT runtime package: timelapsed-hrpqct.",
    ),
    PackageSpec(
        display_name="MotionScore HR-pQCT",
        package_name="motionscorehrpqct",
        import_name="motionscore",
        minimum_version="2.5.8",
        constraints=("numpy>=1.26,<3.0", "scikit-image>=0.24,<0.26", "tifffile<2026"),
        notes="Core motion scoring package. PyTorch extension and model assets are checked separately.",
    ),
    PackageSpec(
        display_name="Scanco I/O",
        package_name="aimio-py",
        import_name="py_aimio",
        minimum_version="0.1.8",
        constraints=("numpy>=1.26,<3.0",),
        notes="Scanco AIM/ISQ/SCV/GOBJ reader-writer stack.",
    ),
    PackageSpec(
        display_name="Spine Segmentation",
        package_name="spine-segment",
        import_name="spine_segment",
        minimum_version="0.1.0",
        notes="CT spine segmentation package for Slicer Python runtime.",
    ),
    PackageSpec(
        display_name="Bone Microarchitecture",
        package_name="bone-microarchitecture",
        import_name="bone_microarchitecture",
        minimum_version="0.1.0",
        constraints=("numpy>=2.0,<3.0", "scipy>=1.18,<2.0"),
        notes="Bone microarchitecture measurements from masks and calibrated grayscale images.",
    ),
)


def _version_parts(version: str) -> tuple:
    parts = []
    for part in str(version or "").replace("-", ".").split("."):
        if not part:
            continue
        digits = ""
        suffix = ""
        for char in part:
            if char.isdigit() and not suffix:
                digits += char
            else:
                suffix += char
        if digits:
            parts.append((0, int(digits), suffix))
        else:
            parts.append((1, part))
    return tuple(parts)


def _parsed_version(version: str):
    try:
        from packaging.version import Version

        return Version(str(version))
    except Exception:
        return _version_parts(version)


def compare_versions(installed_version: str | None, latest_version: str | None) -> str:
    if not installed_version:
        return "missing"
    if not latest_version:
        return "current_unknown_latest"
    try:
        return "outdated" if _parsed_version(installed_version) < _parsed_version(latest_version) else "current"
    except Exception:
        return "current_unknown_latest"


def installed_package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def latest_pypi_version(package_name: str, *, timeout: int = 8) -> str | None:
    url = PYPI_JSON_URL.format(package=package_name)
    req = urllib_request.Request(url, headers=HTTP_HEADERS)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    info = payload.get("info") if isinstance(payload, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    text = str(version or "").strip()
    return text or None


def collect_installed_versions(specs: tuple[PackageSpec, ...] = DEFAULT_RUNTIME_PACKAGES) -> dict[str, str]:
    versions = {}
    for spec in specs:
        version = installed_package_version(spec.package_name)
        if version:
            versions[spec.package_name] = version
    return versions


def collect_latest_versions(
    specs: tuple[PackageSpec, ...] = DEFAULT_RUNTIME_PACKAGES,
    *,
    timeout: int = 8,
) -> dict[str, str]:
    versions = {}
    for spec in specs:
        version = latest_pypi_version(spec.package_name, timeout=timeout)
        if version:
            versions[spec.package_name] = version
    return versions


def package_status_row(
    spec: PackageSpec,
    *,
    installed_versions: dict[str, str],
    latest_versions: dict[str, str],
) -> PackageStatusRow:
    installed = installed_versions.get(spec.package_name)
    latest = latest_versions.get(spec.package_name)
    comparison = compare_versions(installed, latest)
    if installed and compare_versions(installed, spec.minimum_version) == "outdated":
        comparison = "outdated"
    status = "update_available" if comparison == "outdated" else comparison
    action = None
    detail = ""

    if status == "missing":
        action = "install"
        detail = "Not installed in Slicer Python."
    elif status == "update_available":
        action = "update"
        if latest:
            detail = f"Newer PyPI release available: {latest}."
        else:
            detail = f"Installed version is below required minimum {spec.minimum_version}."
    elif status == "current":
        detail = "Installed package is current."
    elif status == "current_unknown_latest":
        detail = "Installed; latest PyPI version could not be checked."

    return PackageStatusRow(
        spec=spec,
        installed_version=installed,
        latest_version=latest,
        status=status,
        action=action,
        detail=detail,
    )


def package_status_rows(
    specs: tuple[PackageSpec, ...] = DEFAULT_RUNTIME_PACKAGES,
    *,
    installed_versions: dict[str, str] | None = None,
    latest_versions: dict[str, str] | None = None,
    timeout: int = 8,
) -> tuple[PackageStatusRow, ...]:
    installed = installed_versions if installed_versions is not None else collect_installed_versions(specs)
    latest = latest_versions if latest_versions is not None else collect_latest_versions(specs, timeout=timeout)
    return tuple(
        package_status_row(spec, installed_versions=installed, latest_versions=latest)
        for spec in specs
    )


def install_command(spec: PackageSpec, *, installed: bool) -> str:
    requirement = f"{spec.package_name}>={spec.minimum_version}"
    args = ["--upgrade"] if installed else []
    args.extend(spec.install_options)
    args.extend([requirement, *spec.constraints])
    return " ".join(args)


def install_commands(spec: PackageSpec, *, installed: bool) -> tuple[str, ...]:
    if spec.package_name == "bone-microarchitecture":
        upgrade = ["--upgrade"] if installed else []
        dependency_command = " ".join([*upgrade, "--prefer-binary", *spec.constraints])
        local_repo = Path(__file__).resolve().parents[1].parent / "bone-microarchitecture"
        if (local_repo / "pyproject.toml").exists():
            package_command = " ".join([*upgrade, "--no-deps", "-e", str(local_repo)])
        else:
            package_command = install_command(spec, installed=installed)
        return (dependency_command, package_command)
    return (install_command(spec, installed=installed),)
