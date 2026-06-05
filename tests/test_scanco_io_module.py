from pathlib import Path
import sys

import SimpleITK as sitk


MODULE = Path(__file__).resolve().parents[1] / "ScancoIO" / "ScancoIO.py"
LIB_DIR = MODULE.parent
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from ScancoIOLib import aim_io  # noqa: E402


def test_scanco_io_can_import_aim_as_segmentation_node() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "as_segmentation=False" in source
    assert "slicer.util.loadSegmentation" in source
    assert "slicer.util.loadLabelVolume" not in source
    assert "self._label_image_for_segmentation(image)" in source
    assert '"image_with_aim_metadata_geometry"' in source
    assert "sitk.GetArrayFromImage(image)" in source
    assert "np.uint16" in source
    assert "sitk.Cast(image != 0" not in source
    assert "vtkMRMLSegmentationNode" in source
    assert "Segmentation import requires a reference volume" not in source
    assert 'loaded = slicer.util.loadSegmentation(str(nrrd_path), {"name": name})' in source
    assert "self._attach_matching_reference_volume(volume_node)" not in source
    assert "def _attach_matching_reference_volume" not in source
    assert "def _same_geometry_extent" not in source
    assert "segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_node)" not in source
    assert "display_node.SetOpacity(0.5)" in source
    assert "display_node.SetOpacity2DFill(0.85)" in source
    assert "display_node.SetAllSegmentsVisibility2DFill(True)" in source
    assert "display_node.SetAllSegmentsOpacity2DFill(0.85)" in source
    assert "slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole()" in source
    assert "display_node.SetVisibility2DFill(True)" in source
    assert "display_node.SetVisibility2DOutline(True)" in source
    assert "Segmentation (nonzero mask)" in source
    assert "Labelmap volume (nonzero mask)" not in source
    assert "self.importReferenceSelector" not in source
    assert "returnNode=True" not in source


def test_scanco_io_can_export_labelmaps_and_segmentations_without_binarizing() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert 'volume_node.IsA("vtkMRMLSegmentationNode")' in source
    assert '"vtkMRMLLabelMapVolumeNode"' in source
    assert "ExportAllSegmentsToLabelmapNode" in source
    assert "GetReferenceImageGeometryReferenceRole()" in source
    assert 'self.exportModeCombo.addItem("Label image (preserve labels)", "label")' in source
    assert 'self.exportModeCombo.findData("label")' in source
    assert 'unit="native" if mode == "label" else self.unitCombo.currentData' in source
    assert "arr_zyx = (127 * (arr_zyx > 0)).astype(np.int8)" not in source


def test_scanco_io_updates_volume_name_when_import_path_changes() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "self.importPathEdit.textChanged.connect(self._on_import_path_changed)" in source
    assert "self._lastAutoVolumeName" in source
    assert "def _update_volume_name_from_import_path" in source


def test_aim_metadata_position_is_refreshed_from_image_origin() -> None:
    image = sitk.Image([4, 5, 6], sitk.sitkUInt8)
    image.SetSpacing((0.061, 0.061, 0.061))
    image.SetOrigin(((100 + 0.5) * 0.061, (200 + 0.5) * 0.061, (300 + 0.5) * 0.061))
    metadata = {"position": (0, 0, 0), "offset": (0, 0, 0)}

    aim_io._refresh_position_from_image_geometry(metadata, image)

    assert metadata["position"] == (100, 200, 300)
    assert metadata["offset"] == (0, 0, 0)


def test_segmentation_export_can_restore_aim_geometry_from_metadata() -> None:
    image = sitk.Image([290, 253, 335], sitk.sitkUInt8)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    metadata = {
        "dimensions": (290, 253, 335),
        "element_size": (0.06069965288043022, 0.06069965288043022, 0.060698509216308594),
        "position": (930, 702, 0),
        "offset": (0, 0, 0),
    }

    restored = aim_io.image_with_aim_metadata_geometry(image, metadata)

    assert restored.GetSpacing() == metadata["element_size"]
    assert restored.GetOrigin() == (
        (930 + 0.5) * metadata["element_size"][0],
        (702 + 0.5) * metadata["element_size"][1],
        (0 + 0.5) * metadata["element_size"][2],
    )


def test_mask_write_forces_native_unit_even_with_bmd_metadata() -> None:
    image = sitk.Image([4, 5, 6], sitk.sitkUInt8)
    metadata = {"unit": "bmd"}

    aim_io._prepare_aim_metadata_for_write(metadata, image, mask=True)

    assert metadata["unit"] == "native"
