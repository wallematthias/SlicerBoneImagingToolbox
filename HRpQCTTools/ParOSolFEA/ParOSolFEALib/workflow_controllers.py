from __future__ import annotations


class LightweightEditorController:
    def __init__(self, widget):
        self._widget = widget

    def mark_editor_dirty(self) -> None:
        self._widget._mark_boundary_preview_stale()
        if self._has_workflow_replay_state():
            self._widget._workflowReplayResolvedEditorDirty = True

    def mark_loads_dirty(self) -> None:
        if self._has_workflow_replay_state():
            self._widget._workflowReplayResolvedEditorDirty = True
        self._widget._mark_load_preview_stale()

    def _has_workflow_replay_state(self) -> bool:
        return (
            getattr(self._widget, "_workflowReplayResolvedEditor", None) is not None
            or getattr(self._widget, "_workflowReplayContractEditor", None) is not None
        )


class BoundaryPreviewController:
    def __init__(self, widget):
        self._widget = widget

    def preview_disks(self):
        return self._widget._preview_disks_impl()


class PreprocessController:
    def __init__(self, widget):
        self._widget = widget

    def preprocess_inputs(self):
        return self._widget._preprocess_inputs_impl()


class LoadPreviewController:
    def __init__(self, widget):
        self._widget = widget

    def preview_loads(self):
        return self._widget._preview_loads_impl()


class ExecutionController:
    def __init__(self, widget):
        self._widget = widget

    def run_case(self):
        return self._widget._run_case_impl()
