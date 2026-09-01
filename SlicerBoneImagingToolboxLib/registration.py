from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import SimpleITK as sitk

from SlicerBoneImagingToolboxLib.derivatives import DerivativeRecord


TOOLBOX_ROOT = Path(__file__).resolve().parents[1]


def _active_repositories_root(toolbox_root: Path) -> Path:
    if toolbox_root.parent.name == ".worktrees":
        return toolbox_root.parent.parent.parent
    return toolbox_root.parent


TIMELAPSED_LOCAL_SRC = _active_repositories_root(TOOLBOX_ROOT) / "TimelapsedHRpQCT" / "src"


def ensure_timelapsed_available() -> None:
    if TIMELAPSED_LOCAL_SRC.exists() and str(TIMELAPSED_LOCAL_SRC) not in sys.path:
        sys.path.insert(0, str(TIMELAPSED_LOCAL_SRC))


@dataclass(frozen=True)
class RegistrationSession:
    subject_id: str
    site: str
    session_id: str
    stack_index: int
    image: sitk.Image
    registration_mask: sitk.Image


@dataclass(frozen=True)
class PairwiseRegistration:
    moving_session_id: str
    fixed_session_id: str
    transform: sitk.Transform


@dataclass(frozen=True)
class RegistrationResult:
    reference_session_id: str
    pairwise: list[PairwiseRegistration] = field(default_factory=list)
    composed: dict[str, sitk.Transform] = field(default_factory=dict)
    records: list[DerivativeRecord] = field(default_factory=list)


def register_image_pair(
    *,
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    settings=None,
):
    ensure_timelapsed_available()
    from timelapsedhrpqct.processing.registration import RegistrationSettings, register_images

    return register_images(
        fixed_image=sitk.Cast(fixed_image, sitk.sitkFloat32),
        moving_image=sitk.Cast(moving_image, sitk.sitkFloat32),
        fixed_mask=sitk.Cast(fixed_mask > 0, sitk.sitkUInt8),
        moving_mask=sitk.Cast(moving_mask > 0, sitk.sitkUInt8),
        settings=settings or RegistrationSettings(),
    )


def _compose_with_timelapsed(
    pairwise: list[PairwiseRegistration],
    *,
    reference_session_id: str,
    dimension: int,
) -> list[tuple[str, sitk.Transform]]:
    ensure_timelapsed_available()
    from timelapsedhrpqct.processing.transform_chain import PairwiseTransform, compose_sequential_to_baseline

    chain = [
        PairwiseTransform(session_id=item.moving_session_id, transform=item.transform)
        for item in pairwise
    ]
    composed = compose_sequential_to_baseline(
        pairwise_transforms=chain,
        baseline_session_id=reference_session_id,
        dimension=dimension,
    )
    return [(item.session_id, item.transform) for item in composed]


def register_sequential_series(
    sessions: list[RegistrationSession],
    *,
    register_pair: Callable[..., object] | None = None,
    compose_to_reference: Callable[..., list[tuple[str, sitk.Transform]]] | None = None,
) -> RegistrationResult:
    if not sessions:
        raise ValueError("Registration requires at least one session.")
    reference = sessions[0]
    register_pair = register_pair or register_image_pair
    pairwise: list[PairwiseRegistration] = []
    previous = reference
    for session in sessions[1:]:
        result = register_pair(
            fixed_image=previous.image,
            moving_image=session.image,
            fixed_mask=previous.registration_mask,
            moving_mask=session.registration_mask,
        )
        pairwise.append(
            PairwiseRegistration(
                moving_session_id=session.session_id,
                fixed_session_id=previous.session_id,
                transform=result.transform,
            )
        )
        previous = session

    identity = sitk.Transform(3, sitk.sitkIdentity)
    compose_to_reference = compose_to_reference or _compose_with_timelapsed
    composed_pairs = compose_to_reference(
        pairwise,
        reference_session_id=reference.session_id,
        dimension=reference.image.GetDimension(),
    )
    composed = {session_id: transform for session_id, transform in composed_pairs}
    composed[reference.session_id] = composed.get(reference.session_id, identity)

    records = []
    for item in pairwise:
        records.append(
            DerivativeRecord(
                derivative="Registration",
                role="transform_pairwise",
                subject_id=reference.subject_id,
                site=reference.site,
                session_id=item.moving_session_id,
                stack_index=reference.stack_index,
                space="native",
                path="",
                source="generated",
                metadata={"fixed_session_id": item.fixed_session_id},
            )
        )
    for session in sessions:
        records.append(
            DerivativeRecord(
                derivative="Registration",
                role="transform_composed",
                subject_id=session.subject_id,
                site=session.site,
                session_id=session.session_id,
                stack_index=session.stack_index,
                space="reference",
                path="",
                source="generated",
                metadata={"reference_session_id": reference.session_id},
            )
        )

    return RegistrationResult(
        reference_session_id=reference.session_id,
        pairwise=pairwise,
        composed=composed,
        records=records,
    )
