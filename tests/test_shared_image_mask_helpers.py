import numpy as np
import pytest
import SimpleITK as sitk

from SlicerBoneImagingToolboxLib.image_io import read_image, read_mask, write_mask
from SlicerBoneImagingToolboxLib.masks import (
    assert_same_geometry,
    clip_mask_to_region,
    resample_mask,
    scan_region_mask,
)


def test_scan_region_mask_matches_image_geometry():
    image = sitk.Image([4, 5, 6], sitk.sitkFloat32)
    image.SetSpacing((0.061, 0.061, 0.061))

    mask = scan_region_mask(image)

    assert mask.GetSize() == image.GetSize()
    assert mask.GetSpacing() == image.GetSpacing()
    assert set(np.unique(sitk.GetArrayFromImage(mask))) == {1}


def test_clip_mask_to_region_keeps_only_shared_support():
    mask = sitk.Image([3, 3, 1], sitk.sitkUInt8) + 1
    region = sitk.Image([3, 3, 1], sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(region)
    arr[:, 1, 1] = 1
    region = sitk.GetImageFromArray(arr)
    region.CopyInformation(mask)

    clipped = clip_mask_to_region(mask, region)

    assert int(sitk.GetArrayFromImage(clipped).sum()) == 1


def test_assert_same_geometry_reports_size_mismatch():
    left = sitk.Image([3, 3, 1], sitk.sitkUInt8)
    right = sitk.Image([4, 3, 1], sitk.sitkUInt8)

    with pytest.raises(ValueError, match="matching sizes"):
        assert_same_geometry([left, right], context="test masks")


def test_assert_same_geometry_tolerates_origin_roundoff():
    left = sitk.Image([3, 3, 1], sitk.sitkUInt8)
    right = sitk.Image([3, 3, 1], sitk.sitkUInt8)
    left.SetOrigin((46.41200128, 38.13000105, 0.0))
    right.SetOrigin((46.41200256, 38.13000107, 0.0))

    assert_same_geometry([left, right], context="test masks")


def test_clip_mask_to_region_tolerates_origin_roundoff():
    mask = sitk.Image([3, 3, 1], sitk.sitkUInt8) + 1
    region = sitk.Image([3, 3, 1], sitk.sitkUInt8) + 1
    mask.SetOrigin((46.41200128, 38.13000105, 0.0))
    region.SetOrigin((46.41200256, 38.13000107, 0.0))

    clipped = clip_mask_to_region(mask, region)

    assert clipped.GetOrigin() == mask.GetOrigin()
    assert int(sitk.GetArrayFromImage(clipped).sum()) == 9


def test_read_write_mask_roundtrip_casts_to_binary(tmp_path):
    image = sitk.GetImageFromArray(np.array([[[0, 2], [3, 0]]], dtype=np.uint8))
    path = tmp_path / "mask.nii.gz"

    write_mask(path, image)
    loaded = read_mask(path)

    assert set(np.unique(sitk.GetArrayFromImage(loaded))) == {0, 1}


def test_read_image_preserves_float_pixels(tmp_path):
    image = sitk.GetImageFromArray(np.array([[[1.5, 2.5]]], dtype=np.float32))
    path = tmp_path / "image.nii.gz"
    sitk.WriteImage(image, str(path))

    loaded = read_image(path)

    assert sitk.GetArrayFromImage(loaded).dtype == np.float32


def test_resample_mask_uses_reference_geometry():
    mask = sitk.Image([3, 3, 1], sitk.sitkUInt8) + 1
    reference = sitk.Image([4, 4, 1], sitk.sitkUInt8)
    transform = sitk.Transform(3, sitk.sitkIdentity)

    resampled = resample_mask(mask, reference, transform)

    assert resampled.GetSize() == reference.GetSize()
    assert resampled.GetPixelID() == sitk.sitkUInt8
