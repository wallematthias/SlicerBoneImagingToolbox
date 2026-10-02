"""Unified scene dispatch and independent tissue segmentation contracts."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "HRpQCTTools/SegmentationHRpQCT/SegmentationHRpQCT.py"


def widget_method(name):
    tree = ast.parse(SOURCE.read_text())
    widget = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SegmentationHRpQCTWidget")
    method = next(n for n in widget.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def test_unet_generate_dispatches_without_standard_contouring():
    calls = []
    widget = SimpleNamespace(_uses_unet=lambda: True, _start_unet=lambda: calls.append("unet"),
                             _validate_preset_selection=lambda: None)
    widget_method("_create_segmentation")(widget)
    assert calls == ["unet"]


def test_cleanup_detaches_and_cancels_worker():
    calls = []
    widget = SimpleNamespace(_unet_logic=SimpleNamespace(
        detach_callbacks=lambda: calls.append("detach"), interrupt=lambda: calls.append("cancel")))
    widget_method("cleanup")(widget)
    assert calls == ["detach", "cancel"] and widget._closed


def test_unet_worker_tissue_seg_matches_single_gaussian_of_original_image(tmp_path, monkeypatch):
    """Optional tissue SEG must not use the CNN's normalized/preprocessed input."""
    import SimpleITK as sitk
    from SlicerBoneImagingToolboxLib import unet_contouring_worker as worker

    density = np.random.default_rng(42).uniform(0, 700, (9, 25, 25)).astype(np.float32)
    density[:, 5:10, 4:21] += 500
    trab = np.zeros_like(density, dtype=np.uint8)
    trab[:, :, :13] = 1
    compartments = {"full": np.ones_like(trab), "trab": trab, "cort": 1 - trab}

    class Segmenter:
        def __init__(self, device):
            assert device == "cpu"

        def segment(self, image, progress):
            # Stand in for arbitrary CNN input preprocessing, not the tissue filter.
            image.fill(-1)
            return compartments

    monkeypatch.setitem(sys.modules, "bone_contouring.unet.inference", SimpleNamespace(Segmenter=Segmenter))
    settings = {"method": "gauss", "min_size_voxels": 0, "keep_largest_component": True}
    snapshot, output = tmp_path / "input.npz", tmp_path / "output.npz"
    np.savez(snapshot, density=density, tissue_image=density, spacing=np.array([.0607] * 3),
             segmentation=json.dumps(settings))
    image = sitk.GetImageFromArray(density)
    image.SetSpacing((.0607,) * 3)
    smoothed = sitk.SmoothingRecursiveGaussian(image, 1.2 * .0607)
    once = sitk.GetArrayFromImage(smoothed)
    twice = sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(smoothed, 1.2 * .0607))
    def threshold(values):
        return ((values >= 320) & (trab > 0)) | ((values >= 450) & (compartments["cort"] > 0))
    expected = threshold(once)
    assert expected.any() and not np.array_equal(expected, threshold(twice))
    assert worker.main([str(snapshot), str(output), "--device", "cpu"]) == 0
    with np.load(output) as data:
        np.testing.assert_array_equal(data["seg"] > 0, expected)
        for role, mask in compartments.items():
            np.testing.assert_array_equal(data[role], mask)


@pytest.mark.parametrize("tissue_method", [None, "gauss", "adaptive", "laplace_hamming"])
def test_worker_uses_published_compartments_and_independent_tissue_seg(tmp_path, monkeypatch, tissue_method):
    import bone_contouring
    from SlicerBoneImagingToolboxLib import unet_contouring_worker as worker
    full = np.ones((3, 7, 9), dtype=np.uint8)
    trab = np.zeros_like(full)
    trab[:, 2:5, 2:7] = 1
    expected = {"full": full, "trab": trab, "cort": full - trab}
    devices, calls = [], []
    class Segmenter:
        def __init__(self, device):
            devices.append(device)
        def segment(self, density, progress):
            assert density.shape == full.shape
            return expected
    monkeypatch.setitem(sys.modules, "bone_contouring.unet.inference", SimpleNamespace(Segmenter=Segmenter))
    def generate(image, parameters, **masks):
        import SimpleITK as sitk
        calls.append(parameters.segmentation)
        assert image.GetSpacing() == (.061,) * 3
        assert sitk.GetArrayFromImage(image)[0, 0, 0] == 17  # Native input for LH, BMD otherwise.
        for key, value in masks.items():
            np.testing.assert_array_equal(sitk.GetArrayFromImage(value), expected[key.removesuffix("_mask")])
        return masks["cort_mask"]
    monkeypatch.setattr(bone_contouring, "generate_bone_segmentation", generate)
    config = {"method": tissue_method, "trab_threshold": 321} if tissue_method else None
    snapshot, output = tmp_path / "input.npz", tmp_path / "output.npz"
    np.savez(snapshot, density=full.astype(float), tissue_image=np.full_like(full, 17),
             spacing=np.array([.061] * 3), segmentation=json.dumps(config))
    assert worker.main([str(snapshot), str(output), "--device", "cpu"]) == 0
    assert devices == ["cpu"]
    assert len(calls) == (1 if tissue_method else 0)
    if tissue_method:
        assert calls[0].method == tissue_method and calls[0].trab_threshold == 321
    with np.load(output) as data:
        assert set(data.files) == (set(expected) | {"seg"} if tissue_method else set(expected))
        for key in expected:
            np.testing.assert_array_equal(data[key], expected[key])
    with pytest.raises(FileExistsError):
        worker.main([str(snapshot), str(output)])
