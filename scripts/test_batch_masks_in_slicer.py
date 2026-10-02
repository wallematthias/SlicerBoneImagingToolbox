"""Run with Slicer --no-splash --no-main-window --python-script this-file."""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import slicer
import qt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "IOTools/BatchProcessor")]


def main():
    from BatchProcessor import BatchProcessorLogic, BatchProcessorWidget
    widget = BatchProcessorWidget()
    widget.logic = BatchProcessorLogic()
    widget.toolCombo = qt.QComboBox()
    widget.toolCombo.addItem("Microarchitecture", "microarchitecture")
    widget.profileCombo = qt.QComboBox()
    widget._populate_profile_combo()
    for value, label, registered in (("functional-bone-native", "Native Functional Bone", False),
                                     ("functional-bone", "Registered Functional Bone", True)):
        index = widget.profileCombo.findData(value)
        assert index >= 0, f"Missing UI profile: {value}"
        assert widget.profileCombo.itemText(index) == label
        widget.profileCombo.currentIndex = index
        assert widget._registered_table_mode() == registered
    widget._append_log = print
    widget._center_slices_on_node = lambda node: None
    reference = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "scan")
    slicer.util.updateVolumeFromArray(reference, np.zeros((7, 7, 7), dtype=np.float32))
    reference.SetIJKToRASDirections(-1, 0, 0, 0, -1, 0, 0, 0, 1)
    widget._ensure_loaded_source_volume = lambda *args, **kwargs: reference
    manual = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "scan2_contours")
    manual.CreateDefaultDisplayNodes()
    with tempfile.TemporaryDirectory(prefix="batch-mask-test-") as directory:
        path = Path(directory) / "sub-001_ses-001_voi-radiusleft_desc-trab_mask.nrrd"
        mask = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.uint8))
        mask.SetOrigin((2, 1, 1))  # LPS, distinct from the scan origin.
        sitk.WriteImage(mask, str(path))
        label = widget._load_mask_as_labelmap(path, "trab", reference)
        assert np.allclose(label.GetOrigin(), (-2, -1, 1)), label.GetOrigin()
        slicer.mrmlScene.RemoveNode(label)
        first_row = dict(subject="001", session_value="001", voi_value="radiusleft", image_path="scan1.nrrd")
        second_row = dict(first_row, subject="002", image_path="scan2.nrrd")
        widget._load_bone_contour_outputs_as_segmentation(first_row, [path])
        first = slicer.util.getNode("scan1_contours")
        segment_id = first.GetSegmentation().GetNthSegmentID(0)
        array = slicer.util.arrayFromSegmentBinaryLabelmap(first, segment_id, reference)
        expected = np.zeros((7, 7, 7), dtype=np.uint8)
        expected[1:4, 1:4, 2:5] = 1
        assert np.array_equal(array, expected), "Cropped mask shifted during import"
        widget._load_bone_contour_outputs_as_segmentation(second_row, [path])
        second = next(node for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
                      if node is not manual and node.GetName().startswith("scan2_contours"))
        assert not first.GetDisplayNode().GetVisibility(), "Earlier case still visible"
        assert second.GetDisplayNode().GetVisibility()
        assert manual.GetDisplayNode().GetVisibility(), "Manual overlay was hidden"
        assert manual.GetScene() is not None, "Manual node with the same name was deleted"
        widget._load_bone_contour_outputs_as_segmentation(second_row, [path])
        assert len([node for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
                    if node.GetAttribute("BoneImaging.BatchContourSource") == "scan2.nrrd"]) == 1
        # Registered timepoints are intentionally shown together, but not other subjects.
        grouped = dict(first_row, registered=True)
        widget._load_bone_contour_outputs_as_segmentation(grouped, [path])
        first = slicer.util.getNode("scan1_contours")
        widget._load_bone_contour_outputs_as_segmentation(
            dict(grouped, session_value="002", image_path="scan3.nrrd"), [path])
        assert first.GetDisplayNode().GetVisibility()
        second = next(node for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
                      if node.GetAttribute("BoneImaging.BatchContourSource") == "scan2.nrrd")
        assert not second.GetDisplayNode().GetVisibility()
        # Loading one Functional Bone mode must not delete the other's analysis region.
        analysis = Path(directory) / "functional_bone_analysis_mask.nrrd"
        sitk.WriteImage(mask, str(analysis))
        native_row = dict(first_row, profile="functional-bone-native", registered=False)
        registered_row = dict(first_row, profile="functional-bone", registered=True)
        assert widget._load_functional_bone_analysis_outputs(native_row, [analysis]) == 1
        assert widget._load_functional_bone_analysis_outputs(registered_row, [analysis]) == 1
        regions = [node for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
                   if node.GetAttribute("BoneImaging.MaskRoles") == "functional_bone_analysis"]
        assert len(regions) == 2, "Loading another mode replaced the earlier analysis mask"
        assert {node.GetAttribute("BoneImaging.FunctionalBoneMode") for node in regions} == {"native", "registered"}
        # U-Net batch load must keep LH tissue SEG in the same node as contours.
        import py_aimio
        full = np.zeros((7, 7, 7), dtype=np.uint8)
        full[1:6, 1:6, 1:6] = 1
        trab = np.zeros_like(full)
        trab[2:5, 2:5, 2:5] = 1
        seg = full.copy()
        seg[3] = 0
        masks = dict(full=full, trab=trab, cort=full-trab, seg=seg)
        paths = []
        for role, array in masks.items():
            path = Path(directory) / f"sub-001_ses-001_voi-radiusleft_desc-{role}_mask.AIM"
            py_aimio.write_aim(str(path), (array*127).astype(np.int8),
                              dict(position=(0,0,0), offset=(0,0,0), element_size=(1.,)*3), unit="native")
            paths.append(path)
        widget._load_deep_learning_outputs(first_row, paths)
        result = next(node for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
                      if node.GetAttribute("DeepLearningSegmentationHRpQCT.Model"))
        assert set(result.GetAttribute("BoneImaging.MaskRoles").split(",")) == {"full", "trab", "cort", "seg"}
        segment_id = result.GetSegmentation().GetSegmentIdBySegmentName("Bone segmentation")
        assert np.array_equal(slicer.util.arrayFromSegmentBinaryLabelmap(result, segment_id, reference), seg)
    print("BATCH_MASKS_SLICER_PASS")


try:
    main()
except Exception:
    traceback.print_exc()
    slicer.app.exit(1)
else:
    slicer.app.exit(0)
