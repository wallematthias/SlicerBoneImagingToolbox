from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "CTTools" / "SpineSegmentationCT"
MODULE = MODULE_DIR / "SpineSegmentationCT.py"


def test_spine_segmentation_ct_module_wraps_spine_segment_dependency() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert 'parent.title = "Spine Segmentation"' in source
    assert 'parent.categories = ["Bone Imaging.CT"]' in source
    assert 'slicer.util.pip_install("spine-segment>=0.1.0")' in source
    assert "qt.QProcess()" in source
    assert '"-m"' in source
    assert '"spine_segment.cli"' in source
    assert '"--localization-only"' in source
    assert '"--level-only"' in source
    assert "vertebral_level" in source
    assert "process_body" in source
    assert "cort_trab" in source
    assert "slicer.util.loadSegmentation" in source
    assert "vtkMRMLMarkupsFiducialNode" in source
    assert "voxel_xyz" in source


def test_spine_segmentation_ct_module_is_registered_as_builtin_tool() -> None:
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    module_cmake = (MODULE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_subdirectory(CTTools/SpineSegmentationCT)" in root_cmake
    assert "set(MODULE_NAME SpineSegmentationCT)" in module_cmake
    assert "SpineSegmentationCT.py" in module_cmake
