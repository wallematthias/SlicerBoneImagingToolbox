from __future__ import annotations

from dataclasses import dataclass, field

import SimpleITK as sitk

from SlicerBoneImagingToolboxLib.derivatives import DerivativeRecord
from SlicerBoneImagingToolboxLib.masks import resample_mask, scan_region_mask


@dataclass(frozen=True)
class CommonRegionSession:
    subject_id: str
    site: str
    session_id: str
    stack_index: int
    image: sitk.Image
    transform_to_reference: sitk.Transform


@dataclass(frozen=True)
class CommonRegionResult:
    reference_session_id: str
    common_mask: sitk.Image
    native_masks: dict[str, sitk.Image] = field(default_factory=dict)
    records: list[DerivativeRecord] = field(default_factory=list)


def build_common_scan_region(
    sessions: list[CommonRegionSession],
    *,
    reference_session_id: str | None = None,
) -> CommonRegionResult:
    if not sessions:
        raise ValueError("Common-region construction requires at least one session.")
    reference = sessions[0]
    if reference_session_id is not None:
        reference = next(
            (session for session in sessions if session.session_id == reference_session_id),
            sessions[0],
        )
    common_mask = None
    for session in sessions:
        scan_region = scan_region_mask(session.image)
        scan_region_common = resample_mask(scan_region, reference.image, session.transform_to_reference)
        if common_mask is None:
            common_mask = sitk.Cast(scan_region_common > 0, sitk.sitkUInt8)
        else:
            common_mask = sitk.Cast((common_mask > 0) & (scan_region_common > 0), sitk.sitkUInt8)
    if common_mask is None:
        raise ValueError("Common-region construction did not produce a mask.")

    native_masks = {}
    for session in sessions:
        inverse = session.transform_to_reference.GetInverse()
        native_masks[session.session_id] = resample_mask(common_mask, session.image, inverse)

    records = [
        DerivativeRecord(
            derivative="CommonRegion",
            role="scan_region_common",
            subject_id=reference.subject_id,
            site=reference.site,
            session_id=reference.session_id,
            stack_index=reference.stack_index,
            space="reference",
            path="",
            source="generated",
            metadata={"reference_session_id": reference.session_id},
        )
    ]
    for session in sessions:
        records.append(
            DerivativeRecord(
                derivative="CommonRegion",
                role="scan_region_native_common",
                subject_id=session.subject_id,
                site=session.site,
                session_id=session.session_id,
                stack_index=session.stack_index,
                space="native",
                path="",
                source="generated",
                metadata={"reference_session_id": reference.session_id},
            )
        )

    return CommonRegionResult(
        reference_session_id=reference.session_id,
        common_mask=common_mask,
        native_masks=native_masks,
        records=records,
    )
