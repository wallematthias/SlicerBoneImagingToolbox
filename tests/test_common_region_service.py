import SimpleITK as sitk

from SlicerBoneImagingToolboxLib.common_region import CommonRegionSession, build_common_scan_region


def _image(size, *, origin=(0.0, 0.0, 0.0)):
    image = sitk.Image(size, sitk.sitkFloat32)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin(origin)
    return image


def test_common_region_builds_scan_fov_masks_only():
    transform = sitk.Transform(3, sitk.sitkIdentity)
    sessions = [
        CommonRegionSession("S1", "tibia", "1", 1, _image([3, 3, 3]), transform),
        CommonRegionSession("S1", "tibia", "2", 1, _image([3, 3, 3]), transform),
    ]

    result = build_common_scan_region(sessions)

    assert result.reference_session_id == "1"
    assert set(result.native_masks) == {"1", "2"}
    assert result.common_mask.GetSize() == sessions[0].image.GetSize()
    assert {record.role for record in result.records} == {
        "scan_region_common",
        "scan_region_native_common",
    }
    assert not {"trab", "cort", "full", "seg"} & {record.role for record in result.records}


def test_common_region_intersects_resampled_scan_support():
    reference = _image([3, 3, 1])
    moving = _image([2, 3, 1], origin=(1.0, 0.0, 0.0))
    transform = sitk.Transform(3, sitk.sitkIdentity)
    sessions = [
        CommonRegionSession("S1", "radius", "1", 1, reference, sitk.Transform(3, sitk.sitkIdentity)),
        CommonRegionSession("S1", "radius", "2", 1, moving, transform),
    ]

    result = build_common_scan_region(sessions)

    assert int(sitk.GetArrayFromImage(result.common_mask).sum()) == 6
    assert result.native_masks["1"].GetSize() == reference.GetSize()
    assert result.native_masks["2"].GetSize() == moving.GetSize()
