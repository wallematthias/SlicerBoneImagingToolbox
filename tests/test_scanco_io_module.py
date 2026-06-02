from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "ScancoIO" / "ScancoIO.py"


def test_scanco_io_can_import_aim_as_segmentation_node() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "as_segmentation=False" in source
    assert "slicer.util.loadLabelVolume" in source
    assert "vtkMRMLSegmentationNode" in source
    assert "ImportLabelmapToSegmentationNode" in source
    assert "Segmentation (nonzero mask)" in source


def test_scanco_io_updates_volume_name_when_import_path_changes() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "self.importPathEdit.textChanged.connect(self._on_import_path_changed)" in source
    assert "self._lastAutoVolumeName" in source
    assert "def _update_volume_name_from_import_path" in source
