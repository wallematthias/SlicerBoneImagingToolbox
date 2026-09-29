from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


BODY_LABEL_KEYS = ("body", "vertebral_body", "bone")
PROCESS_LABEL_KEYS = ("process", "posterior_elements", "posterior_element")


def workflow_label_overrides_from_scene(
    model_labels: Mapping[str, Any] | None,
    current_labels: Iterable[Any] | None,
    selected_registration_label: Any = None,
) -> dict[str, int]:
    """Return workflow label overrides for the labels present in the Slicer scene.

    Bundled workflows may use canonical labels, while a user-loaded mask can use
    another valid label convention.  The selected registration label is the
    user's explicit anatomy target, so it is safe to use when the workflow's
    target label is not present in the current mask.
    """

    labels = _integer_label_mapping(model_labels)
    scene_labels = _positive_integer_values(current_labels)
    if not labels or not scene_labels:
        return {}

    scene_label_set = set(scene_labels)
    selected = _first_positive_integer(selected_registration_label)
    selected_is_available = selected in scene_label_set if selected is not None else False
    overrides: dict[str, int] = {}

    body_key = _first_existing_key(labels, BODY_LABEL_KEYS)
    if body_key is None and len(labels) == 1:
        body_key = next(iter(labels))
    if body_key is not None and labels[body_key] not in scene_label_set:
        if selected_is_available:
            overrides[body_key] = int(selected)
        elif len(scene_labels) == 1:
            overrides[body_key] = int(scene_labels[0])

    process_key = _first_existing_key(labels, PROCESS_LABEL_KEYS)
    if process_key is not None and labels[process_key] not in scene_label_set:
        assigned = set(overrides.values())
        if selected_is_available:
            assigned.add(int(selected))
        remaining = [value for value in scene_labels if value not in assigned]
        if len(remaining) == 1:
            overrides[process_key] = int(remaining[0])

    return overrides


def _integer_label_mapping(model_labels: Mapping[str, Any] | None) -> dict[str, int]:
    if not isinstance(model_labels, Mapping):
        return {}
    labels: dict[str, int] = {}
    for key, value in model_labels.items():
        integer = _single_positive_integer(value)
        if integer is not None:
            labels[str(key)] = integer
    return labels


def _positive_integer_values(values: Iterable[Any] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    labels = []
    for value in values:
        integer = _single_positive_integer(value)
        if integer is not None:
            labels.append(integer)
    return tuple(sorted(set(labels)))


def _first_positive_integer(value: Any) -> int | None:
    integer = _single_positive_integer(value)
    if integer is not None:
        return integer
    if isinstance(value, (str, bytes)):
        return None
    try:
        iterator = iter(value)
    except TypeError:
        return None
    for item in iterator:
        integer = _single_positive_integer(item)
        if integer is not None:
            return integer
    return None


def _single_positive_integer(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    if integer <= 0:
        return None
    try:
        if abs(float(value) - integer) > 1.0e-6:
            return None
    except (TypeError, ValueError):
        pass
    return integer


def _first_existing_key(labels: Mapping[str, int], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in labels:
            return key
    return None
