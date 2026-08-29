from SlicerBoneImagingToolboxLib.derivatives import DerivativeRecord
from SlicerBoneImagingToolboxLib.workflow_planning import resolve_workflow_plan


def _record(derivative):
    return DerivativeRecord(
        derivative=derivative,
        role="available",
        subject_id="S1",
        site="tibia",
        session_id="1",
        stack_index=1,
        space="native",
        path="path",
        source="generated",
        metadata={},
    )


def test_timelapsed_without_prerequisites_plans_registration_common_region_then_analysis():
    plan = resolve_workflow_plan("Timelapsed", available_records=[], available_inputs={"masks": True})

    assert [step.workflow for step in plan.steps] == ["Registration", "CommonRegion", "Timelapsed"]
    assert not plan.blocked


def test_timelapsed_with_registration_only_plans_common_region_then_analysis():
    plan = resolve_workflow_plan(
        "Timelapsed",
        available_records=[_record("Registration")],
        available_inputs={"masks": True},
    )

    assert [step.workflow for step in plan.steps] == ["CommonRegion", "Timelapsed"]


def test_microarchitecture_with_common_region_available_runs_only_microarchitecture():
    plan = resolve_workflow_plan(
        "Microarchitecture",
        available_records=[_record("CommonRegion")],
        available_inputs={"masks": True},
    )

    assert [step.workflow for step in plan.steps] == ["Microarchitecture"]
    assert plan.steps[0].action == "run"


def test_missing_masks_blocks_mask_consuming_workflow():
    plan = resolve_workflow_plan("Microarchitecture", available_records=[], available_inputs={"masks": False})

    assert plan.blocked
    assert plan.missing_roles == ["masks"]
    assert plan.steps == []
