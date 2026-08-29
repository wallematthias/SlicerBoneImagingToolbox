from pathlib import Path


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
    assert "manifests = discover_manifests(derivatives_root)" in method
    assert "available_records.extend(manifest.records)" in method
    assert 'resolve_workflow_plan(\n            "Timelapsed"' in method
    assert '"registration_available": "Registration" in available' in method
    assert '"common_region_available": "CommonRegion" in available' in method
    assert '"planned_steps": list(plan.steps)' in method


def test_timelapsed_prerequisites_use_shared_derivative_planner() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "from bone_imaging_derivatives import resolve_workflow_plan" in source


def test_timelapsed_ui_mentions_derivative_prerequisites():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "Derivative prerequisites" in source
    assert "Registration/CommonRegion derivatives" in source
