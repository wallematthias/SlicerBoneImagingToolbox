from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from SlicerBoneImagingToolboxLib.derivatives import DerivativeRecord


WORKFLOW_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "CommonRegion": ("Registration",),
    "Timelapsed": ("Registration", "CommonRegion"),
    "Microarchitecture": (),
    "PlateRodMorphometry": (),
    "FEA": (),
    "Mechanoregulation": ("Registration", "CommonRegion", "FEA"),
    "VoidSpace": (),
}

MASK_CONSUMING_WORKFLOWS = {
    "Microarchitecture",
    "PlateRodMorphometry",
    "Timelapsed",
    "FEA",
    "Mechanoregulation",
    "VoidSpace",
}


@dataclass(frozen=True)
class WorkflowStep:
    workflow: str
    action: str
    reason: str = ""


@dataclass(frozen=True)
class WorkflowPlan:
    requested_workflow: str
    steps: list[WorkflowStep] = field(default_factory=list)
    blocked: bool = False
    missing_roles: list[str] = field(default_factory=list)


def _available_derivatives(records: Sequence[DerivativeRecord]) -> set[str]:
    return {record.derivative for record in records}


def resolve_workflow_plan(
    workflow: str,
    *,
    available_records: Sequence[DerivativeRecord],
    available_inputs: Mapping[str, bool] | None = None,
    generate_missing: bool = True,
) -> WorkflowPlan:
    available_inputs = available_inputs or {}
    if workflow in MASK_CONSUMING_WORKFLOWS and available_inputs.get("masks") is False:
        return WorkflowPlan(requested_workflow=workflow, blocked=True, missing_roles=["masks"])

    available = _available_derivatives(available_records)
    steps: list[WorkflowStep] = []
    for dependency in WORKFLOW_DEPENDENCIES.get(workflow, ()):
        if dependency in available:
            continue
        if not generate_missing:
            return WorkflowPlan(
                requested_workflow=workflow,
                blocked=True,
                missing_roles=[dependency],
            )
        steps.append(
            WorkflowStep(
                workflow=dependency,
                action="generate",
                reason=f"Required by {workflow}",
            )
        )
        available.add(dependency)
    steps.append(WorkflowStep(workflow=workflow, action="run", reason="Requested workflow"))
    return WorkflowPlan(requested_workflow=workflow, steps=steps)
