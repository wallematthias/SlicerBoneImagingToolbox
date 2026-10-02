from pathlib import Path
import sys

import numpy as np
import pytest


def test_scene_worker_bootstrap_prefers_local_backend(tmp_path):
    from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import scene_worker_args
    args = ["-m", "bone_contouring.unet.scene_worker", "density.npz", "masks.npz", "--device", "cpu"]
    prepared = scene_worker_args(args, tmp_path)
    assert prepared[0] == "-c"
    assert f"sys.path.insert(0, {str(tmp_path)!r})" in prepared[1]
    assert "from bone_contouring.unet.scene_worker import main" in prepared[1]
    assert prepared[2:] == args[2:]
    assert scene_worker_args(args, None) == args
import SimpleITK as sitk


def test_scene_worker_command_uses_snapshot_and_fixed_module(tmp_path):
    from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import build_scene_command
    snapshot = tmp_path/"loaded scan.npz"
    np.savez(snapshot, density=np.zeros((3,4,5)))
    output = tmp_path/"masks.npz"
    command = build_scene_command(sys.executable, snapshot, output, "cuda")
    assert command.program == sys.executable
    assert command.args == ["-m", "bone_contouring.unet.scene_worker", str(snapshot), str(output), "--device", "cuda"]


def test_slicer_model_cache_is_sibling_of_motion_score(tmp_path):
    from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import model_cache_directory
    assert model_cache_directory(tmp_path) == tmp_path/"HRpQCTSegmentation"/"models"


def test_native_aim_nanometer_quantization_preserves_voxel_lattice():
    from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import align_native_aim_mask
    source_spacing = (.06069965288043022, .06069965288043022, .06069940701127052)
    written_spacing = (.06069900095462799,)*3
    position = (1279,666,0)
    reference = sitk.Image((9,7,3), sitk.sitkUInt8)
    reference.SetSpacing(source_spacing)
    reference.SetOrigin(tuple(p*s for p,s in zip(position,source_spacing)))
    mask = sitk.Image(reference)
    mask.SetSpacing(written_spacing)
    mask.SetOrigin(tuple(p*s for p,s in zip(position,written_spacing)))
    source_meta = dict(position=position, offset=(0,0,0), dimensions=(9,7,3),
                       element_size=source_spacing, origin=reference.GetOrigin())
    mask_meta = dict(source_meta, element_size=written_spacing, origin=mask.GetOrigin())
    result = align_native_aim_mask(mask,mask_meta,reference,source_meta)
    assert result.GetSpacing() == reference.GetSpacing()
    assert result.GetOrigin() == reference.GetOrigin()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result),sitk.GetArrayFromImage(mask))
    for key,value in (("position",(1280,666,0)), ("offset",(1,0,0)), ("element_size",(.0608,)*3)):
        with pytest.raises(ValueError):
            align_native_aim_mask(mask,dict(mask_meta,**{key:value}),reference,source_meta)
    reference.SetOrigin((8,9,10))
    with pytest.raises(ValueError):
        align_native_aim_mask(mask,mask_meta,reference,source_meta)


def test_scene_command_rejects_invalid_device_and_stale_outputs(tmp_path):
    from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import build_scene_command
    snapshot = tmp_path/"input.npz"
    snapshot.touch()
    output = tmp_path/"output.npz"
    with pytest.raises(ValueError, match="device"):
        build_scene_command(sys.executable, snapshot, output, "mystery")
    output.touch()
    with pytest.raises(FileExistsError):
        build_scene_command(sys.executable, snapshot, output, "cpu")


@pytest.fixture
def compartment_images():
    trab = np.zeros((3,7,9), dtype=np.uint8)
    cort = np.zeros_like(trab)
    trab[:,2:5,2:7] = 127
    cort[:,1:6,1:8] = 127
    cort[trab>0] = 0
    images = [sitk.GetImageFromArray(a) for a in (cort,trab,np.zeros_like(trab))]
    for image in images:
        image.SetSpacing((.0607,)*3)
        image.SetOrigin((11,-4,3))
    return images


def test_native_compartments_preserved_without_extra_peel(compartment_images):
    from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import validate_deep_learning_compartments
    cort, trab, reference = compartment_images
    masks = validate_deep_learning_compartments(cort,trab,reference)
    for role, source in (("cort",cort),("trab",trab)):
        np.testing.assert_array_equal(sitk.GetArrayFromImage(masks[role]), sitk.GetArrayFromImage(source)>0)
        assert masks[role].GetOrigin() == reference.GetOrigin()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(masks["full"]),
                                 (sitk.GetArrayFromImage(cort)>0)|(sitk.GetArrayFromImage(trab)>0))


@pytest.mark.parametrize("property_name,value", [("Origin", (11.1,-4,3)), ("Spacing", (.1,)*3),
    ("Direction", (-1.,0,0,0,-1.,0,0,0,1.))])
def test_reject_geometry_mismatch(compartment_images, property_name, value):
    from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import validate_deep_learning_compartments
    getattr(compartment_images[0], "Set"+property_name)(value)
    with pytest.raises(ValueError, match="matching"):
        validate_deep_learning_compartments(*compartment_images)


@pytest.mark.parametrize("problem", ["empty","overlap","nonbinary","nan","size"])
def test_reject_invalid_compartments(compartment_images, problem):
    from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import validate_deep_learning_compartments
    cort, trab, reference = compartment_images
    if problem == "size":
        cort = sitk.Image((2,2,2), sitk.sitkUInt8)
    else:
        a = sitk.GetArrayFromImage(cort).astype(np.float32)
        if problem == "empty":
            a[:] = 0
        elif problem == "overlap":
            a[sitk.GetArrayFromImage(trab)>0] = 127
        else:
            a[0,0,0] = 42 if problem == "nonbinary" else np.nan
        cort = sitk.GetImageFromArray(a)
        cort.CopyInformation(reference)
    with pytest.raises(ValueError):
        validate_deep_learning_compartments(cort,trab,reference)


def test_setup_installs_core_and_compatible_numpy_torch():
    from SlicerBoneImagingToolboxLib.package_status import DEFAULT_RUNTIME_PACKAGES, install_command
    spec = next((s for s in DEFAULT_RUNTIME_PACKAGES if s.import_name == "bone_contouring"), None)
    assert spec is not None
    command = install_command(spec, installed=False)
    assert "torch>=2.2" in command
    assert "bone-contouring[unet]>=0.3.3" in command
