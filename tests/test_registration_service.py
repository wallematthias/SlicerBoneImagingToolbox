import sys
import types
from pathlib import Path

import SimpleITK as sitk

from SlicerBoneImagingToolboxLib.registration import (
    RegistrationSession,
    _active_repositories_root,
    register_image_pair,
    register_sequential_series,
)


def _identity():
    return sitk.Transform(3, sitk.sitkIdentity)


def _session(session_id):
    image = sitk.Image([2, 2, 2], sitk.sitkFloat32)
    mask = sitk.Image([2, 2, 2], sitk.sitkUInt8) + 1
    return RegistrationSession(
        subject_id="S1",
        site="tibia",
        session_id=str(session_id),
        stack_index=1,
        image=image,
        registration_mask=mask,
    )


def test_active_repositories_root_handles_regular_checkout():
    assert _active_repositories_root(Path("/repos/SlicerBoneImagingToolbox")) == Path("/repos")


def test_active_repositories_root_handles_worktree_checkout():
    path = Path("/repos/SlicerBoneImagingToolbox/.worktrees/derivatives-overhaul")
    assert _active_repositories_root(path) == Path("/repos")


def test_register_image_pair_wraps_timelapsed_registration(monkeypatch):
    calls = []

    def fake_register_images(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(transform=_identity())

    fake_module = types.SimpleNamespace(
        RegistrationSettings=lambda: "settings",
        register_images=fake_register_images,
    )
    monkeypatch.setitem(sys.modules, "timelapsedhrpqct.processing.registration", fake_module)

    result = register_image_pair(
        fixed_image=sitk.Image([2, 2, 2], sitk.sitkFloat32),
        moving_image=sitk.Image([2, 2, 2], sitk.sitkFloat32),
        fixed_mask=sitk.Image([2, 2, 2], sitk.sitkUInt8) + 1,
        moving_mask=sitk.Image([2, 2, 2], sitk.sitkUInt8) + 1,
    )

    assert result.transform.GetDimension() == 3
    assert calls[0]["settings"] == "settings"


def test_register_sequential_series_creates_adjacent_and_composed_records():
    calls = []

    def fake_register_pair(*, fixed_image, moving_image, fixed_mask, moving_mask):
        calls.append((fixed_image.GetSize(), moving_image.GetSize(), fixed_mask.GetSize(), moving_mask.GetSize()))
        return types.SimpleNamespace(transform=_identity())

    result = register_sequential_series(
        [_session(1), _session(2), _session(3)],
        register_pair=fake_register_pair,
        compose_to_reference=lambda pairwise, reference_session_id, dimension: [
            (reference_session_id, _identity()),
            ("2", _identity()),
            ("3", _identity()),
        ],
    )

    assert len(calls) == 2
    assert [(item.moving_session_id, item.fixed_session_id) for item in result.pairwise] == [
        ("2", "1"),
        ("3", "2"),
    ]
    assert [record.role for record in result.records].count("transform_pairwise") == 2
    assert [record.role for record in result.records].count("transform_composed") == 3
    assert result.reference_session_id == "1"
    assert set(result.composed) == {"1", "2", "3"}
