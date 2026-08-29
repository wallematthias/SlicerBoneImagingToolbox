from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "CTTools" / "SpineSegmentationCT"
MODULE = MODULE_DIR / "SpineSegmentationCT.py"


def test_spine_segmentation_ct_module_wraps_spine_segment_dependency() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert 'parent.title = "Spine Segmentation"' in source
    assert 'parent.categories = ["Bone Imaging.CT"]' in source
    assert 'slicer.util.pip_install("spine-segment>=0.1.0")' in source
    assert 'CONDA_RUNTIME_ENV = "spine-segment-pytorch"' in source
    assert "Install Slicer Runtime" not in source
    assert "self.installButton.clicked.connect(self._install_core)" not in source
    assert "Install / Update Conda MPS Runtime" in source
    assert "Probe runtime" in source
    assert "RUNTIME_PROBE_SCRIPT" in source
    assert "mps_conv3d_supported" in source
    assert "PYTHONHOME" in source
    assert "PYTHONPATH" in source
    assert "qt.QProcess()" in source
    assert '"-m"' in source
    assert '"spine_segment.cli"' in source
    assert "runtime=self.runtimeCombo.currentData" in source
    assert "conda_python=self._conda_python_path()" in source
    assert '"--localization-only"' in source
    assert '"--level-only"' in source
    assert "vertebral_level" in source
    assert "process_body" in source
    assert "cort_trab" in source
    assert "slicer.util.loadSegmentation" in source
    assert "vtkMRMLMarkupsFiducialNode" in source
    assert "voxel_xyz" in source


def test_spine_segmentation_ct_ui_prioritizes_run_workflow() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert '"Run spine CT segmentation"' in source
    assert '"Runtime setup"' in source
    assert "self.runtimeBox.collapsed = True" in source
    assert '"Full segmentation + centroids"' in source
    assert "Body/process and cort/trab are generated together" in source
    assert "Centroid markers are expected for every completed spine segmentation run" in source
    assert "format_verse_label" in source
    assert 'raw_verse_label = entry.get("label", raw_label)' in source
    assert "label = format_verse_label(raw_verse_label)" in source
    assert "SetNthControlPointDescription" in source


def test_spine_segmentation_ct_module_is_registered_as_builtin_tool() -> None:
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    module_cmake = (MODULE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_subdirectory(CTTools/SpineSegmentationCT)" in root_cmake
    assert "set(MODULE_NAME SpineSegmentationCT)" in module_cmake
    assert "SpineSegmentationCT.py" in module_cmake
