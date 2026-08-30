from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkflowStageState:
    workflow_dirty: bool = False
    anatomy_dirty: bool = False
    boundary_dirty: bool = False
    loads_dirty: bool = False
    export_dirty: bool = False


class WorkflowStageController:
    def __init__(self, state: WorkflowStageState | None = None):
        self._state = state if isinstance(state, WorkflowStageState) else WorkflowStageState()

    def state(self) -> WorkflowStageState:
        return self._state

    def mark_boundary_preview_stale(self) -> None:
        self._state.boundary_dirty = True
        self._state.loads_dirty = True
        self._state.export_dirty = True

    def mark_load_preview_stale(self) -> None:
        self._state.loads_dirty = True
        self._state.export_dirty = True

    def mark_stage_complete(self, stage: str) -> None:
        token = str(stage).strip().lower()
        if token == "workflow":
            self._state.workflow_dirty = False
            self._state.anatomy_dirty = True
            self._state.boundary_dirty = True
            self._state.loads_dirty = True
            self._state.export_dirty = True
        elif token == "anatomy":
            self._state.workflow_dirty = False
            self._state.anatomy_dirty = False
            self._state.boundary_dirty = True
            self._state.loads_dirty = True
            self._state.export_dirty = True
        elif token == "boundary":
            self._state.boundary_dirty = False
            self._state.loads_dirty = True
            self._state.export_dirty = True
        elif token == "loads":
            self._state.loads_dirty = False
            self._state.export_dirty = True
        elif token == "export":
            self._state.export_dirty = False

    def status_text(self, *, generated: bool) -> str:
        if generated and self._state.boundary_dirty:
            return (
                "Contact regions are stale; Create Regions will refresh "
                "disks and contact-region nodes from the current planes."
            )
        if generated and self._state.loads_dirty:
            return "Load preview is stale; Preview Loads will refresh arrows from the current load table."
        if generated:
            return "Generated contact regions are active; Run ParOSol will use the editable tables."
        if self._state.anatomy_dirty:
            return "Prepare Image to prepare anatomy and resolve workflow planes for editing."
        return ""

    def stage_summary(self, *, generated: bool) -> str:
        contact_status = "not created"
        load_status = "not previewed"
        if generated:
            contact_status = "stale" if self._state.boundary_dirty else "ready"
            load_status = "stale" if self._state.loads_dirty else "ready"
        steps = [
            ("Inputs", "needs review" if self._state.workflow_dirty else "ready"),
            ("Image prep", "needs update" if self._state.anatomy_dirty else "ready"),
            ("Contact regions", contact_status),
            ("Loads", load_status),
            ("Review & run", "stale" if self._state.export_dirty else "ready"),
        ]
        return " | ".join(f"{label}: {status}" for label, status in steps)
