from __future__ import annotations

from SlicerBoneImagingToolboxLib.fea_workflow_labels import (
    workflow_label_overrides_from_scene,
)


def test_scene_label_override_maps_selected_body_and_remaining_process() -> None:
    overrides = workflow_label_overrides_from_scene(
        {"body": 20, "process": 48},
        current_labels=[10, 36],
        selected_registration_label=10,
    )

    assert overrides == {"body": 10, "process": 36}


def test_scene_label_override_leaves_matching_workflow_labels_alone() -> None:
    overrides = workflow_label_overrides_from_scene(
        {"body": 20, "process": 48},
        current_labels=[20, 48],
        selected_registration_label=20,
    )

    assert overrides == {}


def test_scene_label_override_does_not_guess_when_multiple_remaining_labels() -> None:
    overrides = workflow_label_overrides_from_scene(
        {"body": 20, "process": 48},
        current_labels=[10, 36, 37],
        selected_registration_label=10,
    )

    assert overrides == {"body": 10}


def test_scene_label_override_can_map_single_label_workflows() -> None:
    overrides = workflow_label_overrides_from_scene(
        {"bone": 2},
        current_labels=[1],
        selected_registration_label=1,
    )

    assert overrides == {"bone": 1}
