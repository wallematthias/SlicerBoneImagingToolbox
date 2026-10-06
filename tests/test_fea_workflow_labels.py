from __future__ import annotations

import pytest

from SlicerBoneImagingToolboxLib.fea_workflow_labels import (
    validate_material_label_coverage,
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


def test_material_label_coverage_accepts_matching_labels() -> None:
    matched = validate_material_label_coverage(
        present_values=[0, 100, 127],
        configured_labels=[100, 127],
    )

    assert matched == (100, 127)


def test_material_label_coverage_rejects_disjoint_labels() -> None:
    with pytest.raises(ValueError, match=r"contains labels 2, 3, 4.*defines 100, 127"):
        validate_material_label_coverage(
            present_values=[0, 2, 3, 4],
            configured_labels=[100, 127],
        )
