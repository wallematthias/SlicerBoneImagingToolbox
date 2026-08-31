from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "CTTools" / "SpineSegmentationCT"
MODULE = MODULE_DIR / "SpineSegmentationCT.py"


def test_spine_segmentation_ct_module_wraps_spine_segment_dependency() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert 'parent.title = "Spine Segmentation"' in source
    assert 'parent.categories = ["Bone Imaging.Segmentation Methods"]' in source
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

    assert "self.spineModeTabs" in source
    assert '"Scene"' in source
    assert '"Batch"' in source
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


def test_spine_segmentation_ct_exposes_derivative_batch_mode() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "discover_spine_segmentation_batch_cases" in source
    assert "build_spine_segmentation_batch_commands" in source
    assert "self.batchDatasetRootSelector" in source
    assert "self.batchDiscoverButton" in source
    assert "self.batchImageRoleBox" in source
    assert "self.batchRunButton" in source
    assert "run_spine_batch" in source
    assert "derivatives/SpineSegmentationCT" in source
    run_batch = source[
        source.index("    def run_spine_batch(self):") :
        source.index("    def _run_next_spine_batch_case(self):")
    ]
    assert "if self.logic.is_running():" in run_batch
    assert "A spine segmentation process is already running." in run_batch


def test_spine_segmentation_batch_helper_is_packaged() -> None:
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "SlicerBoneImagingToolboxLib/spine_segmentation_batch.py" in root_cmake


def test_spine_segmentation_ct_module_is_registered_as_builtin_tool() -> None:
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    module_cmake = (MODULE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_subdirectory(CTTools/SpineSegmentationCT)" in root_cmake
    assert "set(MODULE_NAME SpineSegmentationCT)" in module_cmake
    assert "SpineSegmentationCT.py" in module_cmake
