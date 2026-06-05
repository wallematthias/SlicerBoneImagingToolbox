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
    assert "reference_volume_node=None" in source
    assert "slicer.util.loadSegmentation" in source
    assert "vtkMRMLSegmentationNode" in source
    assert "segmentation_node = volume_node" in source
    assert "segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume_node)" in source
    assert "self._validate_image_matches_reference(label_image, reference_volume_node)" in source
    assert "AIM segmentation dimensions do not match the selected reference volume" in source
    assert "slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole()" in source
    assert "segmentation_node.SetNodeReferenceID(" in source
    assert '"HRpQCT.ReferenceVolume"' in source
    assert "segmentation_node.CreateDefaultDisplayNodes()" in source
    assert "display_node.SetVisibility2DFill(True)" in source
    assert "display_node.SetVisibility2DOutline(True)" in source
    assert "Segmentation (nonzero mask)" in source
    assert "self.importReferenceSelector" in source
    assert "reference_volume_node=self.importReferenceSelector.currentNode() if as_segmentation else None" in source


def test_scanco_io_forces_labelmap_exports_to_binary_mask() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert 'volume_node.IsA("vtkMRMLLabelMapVolumeNode")' in source
    assert "as_mask = True" in source
    assert 'self.exportModeCombo.findData("mask")' in source


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


def test_mask_write_forces_native_unit_even_with_bmd_metadata() -> None:
    image = sitk.Image([4, 5, 6], sitk.sitkUInt8)
    metadata = {"unit": "bmd"}

    aim_io._prepare_aim_metadata_for_write(metadata, image, mask=True)

    assert metadata["unit"] == "native"
