from __future__ import annotations

from collections.abc import Sequence

import SimpleITK as sitk


GEOMETRY_ABS_TOLERANCE = 1e-4


def scan_region_mask(image: sitk.Image) -> sitk.Image:
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    return sitk.Cast(mask + 1, sitk.sitkUInt8)


def clip_mask_to_region(mask: sitk.Image, region: sitk.Image | None) -> sitk.Image:
    if region is None:
        return sitk.Cast(mask > 0, sitk.sitkUInt8)
    assert_same_geometry([mask, region], context="mask and common region")
    if region.GetOrigin() != mask.GetOrigin() or region.GetSpacing() != mask.GetSpacing() or region.GetDirection() != mask.GetDirection():
        region = sitk.Image(region)
        region.CopyInformation(mask)
    return sitk.Cast((mask > 0) & (region > 0), sitk.sitkUInt8)


def resample_mask(mask: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    return sitk.Cast(
        sitk.Resample(
            sitk.Cast(mask > 0, sitk.sitkUInt8),
            reference,
            transform,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        > 0,
        sitk.sitkUInt8,
    )


def assert_same_geometry(images: Sequence[sitk.Image], *, context: str = "images") -> None:
    items = list(images)
    if not items:
        return
    sizes = {tuple(image.GetSize()) for image in items}
    if len(sizes) != 1:
        raise ValueError(f"{context} must have matching sizes. Got {sorted(sizes)}.")
    spacings = [tuple(float(value) for value in image.GetSpacing()) for image in items]
    if not _vectors_close(spacings):
        raise ValueError(f"{context} must have matching spacings. Got {spacings}.")
    origins = [tuple(float(value) for value in image.GetOrigin()) for image in items]
    if not _vectors_close(origins):
        raise ValueError(f"{context} must have matching origins. Got {origins}.")
    directions = [tuple(float(value) for value in image.GetDirection()) for image in items]
    if not _vectors_close(directions):
        raise ValueError(f"{context} must have matching directions.")


def _vectors_close(vectors: Sequence[Sequence[float]], *, tolerance: float = GEOMETRY_ABS_TOLERANCE) -> bool:
    if not vectors:
        return True
    reference = tuple(vectors[0])
    for vector in vectors[1:]:
        if len(vector) != len(reference):
            return False
        if any(abs(float(left) - float(right)) > tolerance for left, right in zip(reference, vector)):
            return False
    return True
