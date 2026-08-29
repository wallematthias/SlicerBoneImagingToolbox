from pathlib import Path
import importlib.util
import sys

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, write_manifest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "HRpQCTTools" / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py"


def test_timelapsed_imports_derivative_discovery_and_planning_helpers():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "from bone_imaging_derivatives import discover_manifests" in source
    assert "from bone_imaging_derivatives import resolve_workflow_plan" in source
    assert "def discover_derivative_prerequisites(" in source


def test_timelapsed_derivative_prerequisites_report_registration_and_common_region():
    source = MODULE_PATH.read_text(encoding="utf-8")
    method = source[source.index("    def discover_derivative_prerequisites(") :]

    assert '"Registration"' in method
    assert '"CommonRegion"' in method
    assert "manifests = discover_manifests(dataset_root)" in method
    assert "available_records.extend(manifest.records)" in method
    assert 'resolve_workflow_plan(\n            "Timelapsed"' in method
    assert '"registration_available": "Registration" in available' in method
    assert '"common_region_available": "CommonRegion" in available' in method
    assert '"planned_steps": list(plan.steps)' in method


def test_timelapsed_prerequisites_use_shared_derivative_planner() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "from bone_imaging_derivatives import resolve_workflow_plan" in source


def test_timelapsed_prerequisites_accept_dataset_or_derivatives_root(tmp_path: Path) -> None:
    module_dir = MODULE_PATH.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("timelapsed_derivatives_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = (
        DerivativeRecord("Registration", "transform_to_reference", "S1", "tibia", "2", None, "reference", tmp_path / "transform.tfm", "generated"),
        DerivativeRecord("CommonRegion", "scan_region_native_common", "S1", "tibia", "1", None, "native", tmp_path / "region-1.nii.gz", "generated"),
        DerivativeRecord("CommonRegion", "scan_region_native_common", "S1", "tibia", "2", None, "native", tmp_path / "region-2.nii.gz", "generated"),
    )
    write_manifest(
        DerivativeManifest.create("Registration", tmp_path, {"name": "test", "version": "1"}, records=records[:1]),
        tmp_path / "derivatives" / "Registration" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=records[1:]),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    logic = module.TimelapsedHRpQCTLogic()

    assert logic.discover_derivative_prerequisites(tmp_path)["registration_available"] is True
    assert logic.discover_derivative_prerequisites(tmp_path / "derivatives")["common_region_available"] is True


def test_timelapsed_ui_mentions_derivative_prerequisites():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "Derivative prerequisites" in source
    assert "Registration/CommonRegion derivatives" in source
