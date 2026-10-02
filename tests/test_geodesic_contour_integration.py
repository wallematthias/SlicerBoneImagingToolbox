from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
import bone_contouring

from test_standard_contour_defaults import adapter_namespace


MODULE = Path(__file__).resolve().parents[1] / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
PIPELINE_MODULE = Path(__file__).resolve().parents[1] / "HRpQCTTools" / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py"


def _run_real_adapter(*, outer="standard", inner="none", segmentation="seg_gauss", aligned=False):
    """Run real scientific logic, stopping before the first MRML scene mutation."""
    y, x = np.ogrid[:81, :81]
    radius = (x - 40) ** 2 + (y - 40) ** 2
    plane = np.where(radius < 25 ** 2, 340, 0).astype(np.float32)
    plane[(radius >= 25 ** 2) & (radius < 30 ** 2)] = 900
    image = sitk.GetImageFromArray(np.broadcast_to(plane, (9, 81, 81)).copy())
    image.SetSpacing((.0607,) * 3)
    native = image * 30
    captured = []

    class SceneBoundary(Exception):
        pass

    def stop_before_scene(*args, **kwargs):
        captured.append(kwargs["generated"])
        raise SceneBoundary

    def empty(reference):
        result = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
        result.CopyInformation(reference)
        return result

    logic = SimpleNamespace(
        _import_bone_contouring=lambda: bone_contouring,
        _empty_mask_like=empty,
        _mask_voxel_count=lambda mask: int(np.count_nonzero(sitk.GetArrayViewFromImage(mask))),
        _volume_source_aim_path=lambda node: None,
        _laplace_hamming_support_image=lambda node, density: (
            native, {"segmentation_input_unit": "scanco_native_int16"}),
        _write_scene_debug_artifacts=stop_before_scene,
    )
    namespace = adapter_namespace()
    namespace.update(sitk=sitk, Path=Path)
    with pytest.raises(SceneBoundary):
        namespace["_generate_bone_masks_with_bone_contouring"](
            logic, SimpleNamespace(GetName=lambda: "fixture"), image, site="radius",
            segmentation_method=segmentation, periosteal_contour_method=outer,
            endosteal_contour_method=inner, debug_output_dir="unused",
            params={"modality": "xct2", "segmentation": {
                "gaussian_sigma": 0, "trab_threshold": 320, "cort_threshold": 1000,
                "min_size_voxels": 0, "use_segmentation_aligned_contour_support": aligned,
                "laplace_hamming_low_pass_cutoff": 1, "laplace_hamming_epsilon": 0,
                "laplace_hamming_amplitude": 0, "laplace_hamming_ipl_float_max": 32767,
                "laplace_hamming_int16_max": 32767, "laplace_hamming_min_size_voxels": 0,
            }})
    return captured[0], image


def test_segmentation_module_exposes_geodesic_method_for_local_testing():
    source = MODULE.read_text()

    assert "geodesic_fracture" in source
    assert "hrpqct-geodesic-contour" in source
    assert "install_or_update_geodesic_contour" in source
    assert "install_or_update_contouring_dependencies" in source
    assert "Install / Update contouring dependencies" not in source
    assert "self.installButton.clicked.connect(self._install_contouring_dependencies)" not in source
    assert "pip_install(\"edt>=2.4\")" in source
    assert "--no-deps -e" in source
    assert "importlib.invalidate_caches()" in source
    assert "sys.modules.pop(\"hrpqct_geodesic_contour\"" in source
    assert "missing required API arguments" in source
    assert "from hrpqct_geodesic_contour import contour" in source
    assert "np.transpose(arr_zyx, (2, 1, 0))" in source
    assert "progress_callback=progress_callback" in source
    assert "cancel_callback=cancel_callback" in source
    assert "fill_holes=bool(geodesic_params.get(\"fill_holes\", True))" in source
    assert "qt.QProgressDialog" in source
    assert "_geodesic_cancel_requested" in source
    assert "ContourCancelledError" not in source
    assert "installGeodesicButton" not in source
    assert "Install local geodesic contour" not in source


def test_segmentation_module_splits_segmentation_and_contour_choices():
    source = MODULE.read_text()

    assert "SEGMENTATION_METHODS = set(BONE_SEGMENTATION_METHODS)" in source
    assert "PERIOSTEAL_CONTOUR_METHOD_IDS = set(PERIOSTEAL_CONTOUR_METHODS)" in source
    assert "ENDOSTEAL_CONTOUR_METHOD_IDS = set(ENDOSTEAL_CONTOUR_METHODS)" in source
    assert "segmentation_method=segmentation_method" in source
    assert "periosteal_contour_method=periosteal_method" in source
    assert "endosteal_contour_method=endosteal_method" in source
    generated, _ = _run_real_adapter()
    assert generated.metadata["periosteal_contour_method"] == "standard"
    assert generated.metadata["endosteal_contour_method"] == "none"
    assert "form.addRow(\"Segmentation method\", self.methodCombo)" not in source


def test_segmentation_module_skips_compartments_without_endosteal_split():
    generated, _ = _run_real_adapter()
    assert not generated.metadata["compartment_split_generated"]
    assert generated.metadata["compartment_split_reason"] == "endosteal_contour_method_none"
    assert generated.metadata["voxel_counts"]["full"] > 0
    assert generated.metadata["voxel_counts"]["trab"] == generated.metadata["voxel_counts"]["cort"] == 0


def test_segmentation_module_does_not_emit_full_mask_without_outer_contour():
    generated, _ = _run_real_adapter(outer="none")
    assert not generated.metadata["periosteal_contour_generated"]
    assert generated.metadata["voxel_counts"]["seg"] > 0
    assert all(generated.metadata["voxel_counts"][role] == 0 for role in ("full", "trab", "cort"))


def test_segmentation_module_defaults_to_segmentation_node_only():
    source = MODULE.read_text()

    assert "create_labelmaps=False" in source
    assert "self.createLabelmapsCheck" not in source
    assert "create_labelmaps=False" in source
    assert "label_text = \"\"" in source
    scene_source = source[source.index("def _create_segmentation(self):") : source.index("def _create_geodesic_progress_dialog")]
    assert "returnNode=True" not in scene_source
    assert "slicer.util.updateSegmentBinaryLabelmapFromArray(" in source
    assert "segment.SetName(str(segment_name))" in source
    assert 'segment.SetTag("HRpQCT.Role", str(role))' in source
    assert "self._finalize_segment(segmentation_node, segment_id, segment_name, role)" in source
    assert "_remove_empty_duplicate_segmentation_nodes(segmentation_node)" in source
    assert "node.IsA(\"vtkMRMLSegmentationNode\")" in source
    assert "node.GetSegmentation().GetNumberOfSegments() == 0" in source
    assert "display_node.SetOpacity2DFill(0.85)" in source
    assert "display_node.SetAllSegmentsVisibility2DFill(True)" in source
    assert "display_node.SetAllSegmentsOpacity2DFill(0.85)" in source


def test_gaussian_segmentation_without_compartments_uses_global_trab_threshold():
    generated, image = _run_real_adapter()
    expected = (sitk.GetArrayFromImage(image) >= 320) & (sitk.GetArrayFromImage(generated.full) > 0)
    assert expected.any()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(generated.seg) > 0, expected)
    assert generated.metadata["segmentation_threshold_applied_global"] == 320
    assert "No cortical mask" in generated.metadata["segmentation_warning"]


@pytest.mark.parametrize("aligned", [False, True])
def test_laplace_hamming_passes_native_input_and_preserves_support_setting(aligned):
    generated, _ = _run_real_adapter(segmentation="laplace_hamming", aligned=aligned)
    assert generated.metadata["segmentation_input_unit"] == "scanco_native_int16"
    assert generated.metadata["segmentation_aligned_contour_support"] == aligned
    assert generated.metadata["voxel_counts"]["seg"] > 0
    assert not np.any((sitk.GetArrayFromImage(generated.seg) > 0) &
                      (sitk.GetArrayFromImage(generated.full) == 0))


def test_laplace_hamming_uses_core_native_scanco_input_convention():
    source = MODULE.read_text()

    assert "AIM_METADATA_ATTRIBUTE = \"HRpQCT.AIMMetadata\"" in source
    assert "AIM_SCALING_ATTRIBUTE = \"HRpQCT.AIMScaling\"" in source
    assert "from timelapsedhrpqct.io.aim import density_to_native_int16" in source
    assert "segmentation_input_unit\": \"scanco_native_int16\"" in source
    assert "segmentation_input_reader\": \"imported_density_to_native_int16\"" in source
    assert "read_aim(source_path, scaling=\"native\")" in source
    assert "segmentation_input_reader\": \"py_aimio_native_int16\"" in source
    assert "scanco_hu_int16" not in source
    assert "py_aimio_hu_int16" not in source
    assert "segmentation_node.CreateDefaultDisplayNodes()" in source
    assert "segmentation_node.SetAttribute(f\"HRpQCT.{key}\", str(generated.metadata[key]))" in source
    assert "Method=laplace_hamming; image={processing_reader}; input={metadata.get('segmentation_input_unit')}" in source


def test_generated_masks_keep_aim_metadata_for_export():
    source = MODULE.read_text()

    assert "def _copy_aim_attributes" in source
    assert "AIM_METADATA_ATTRIBUTE, AIM_SOURCE_ATTRIBUTE, AIM_SCALING_ATTRIBUTE" in source
    assert "self._copy_aim_attributes(reference_node, label_node)" in source
    assert "self._copy_aim_attributes(volume_node, segmentation_node)" in source


def test_segmentation_installer_keeps_slicer_numpy_constraints():
    source = MODULE.read_text()

    assert 'BONE_CONTOURING_PIP_CONSTRAINTS = ("numpy>=1.26,<3.0", "SimpleITK>=2.3")' in source
    assert 'slicer.util.pip_uninstall("pyjpegls")' in source
    assert '" ".join(["bone-contouring", *BONE_CONTOURING_PIP_CONSTRAINTS])' in source


def test_laplace_hamming_shows_busy_progress_dialog():
    source = MODULE.read_text()

    assert 'elif segmentation_method == "laplace_hamming":' in source
    assert "Running Laplace-Hamming bone segmentation..." in source
    assert "Laplace-Hamming Segmentation" in source
    assert "def _create_busy_progress_dialog" in source
    assert "dialog.setCancelButton(None)" in source


def test_segmentation_module_uses_single_generate_workflow():
    source = MODULE.read_text()

    builder = source[
        source.index("    def _build_segmentation_section(self):")
        : source.index("    def _labelmap_selector(self):")
    ]
    assert "self.toolTabs = qt.QTabWidget()" not in builder
    assert "self.layout.addWidget(contouring_widget)" in builder
    assert "self.createButton = qt.QPushButton(\"Generate\")" in builder
    assert "self.toolTabs.addTab(derive_tab" not in source


def test_timelapsed_pipeline_exposes_geodesic_periosteal_contour_config():
    source = PIPELINE_MODULE.read_text()

    assert "self.maskPeriostealContour = qt.QComboBox()" in source
    assert "self.maskPeriostealContour.addItem(\"geodesic_fracture\", \"geodesic_fracture\")" in source
    assert "self.sceneProfileCombo.currentIndexChanged.connect(self._on_scene_profile_changed)" in source
    assert "Full/periosteal contour" in source
    assert "self.maskGeodesicThreshold" in source
    assert "self.maskGeodesicFillHoles" in source
    assert "self.maskGeodesicFillHoles.checked = True" in source
    assert "Fill full mask holes" in source
    assert "_TOOLBOX_ROOT = Path(__file__).resolve().parents[2]" in source
    assert "def _resolve_local_pipeline_paths" in source
    assert 'candidate_repo = base / "TimelapsedHRpQCT"' in source
    assert "def _local_pipeline_usable" in source
    assert "if _local_pipeline_usable(_PIPELINE_LOCAL_REPO, _PIPELINE_LOCAL_SRC)" in source
    assert "os.environ[\"PYTHONPATH\"]" in source
    assert "slicer.util.pip_install(\"hrpqct-geodesic-contour>=0.1.1\")" in source
    assert "timelapsed-hrpqct>={MIN_PIPELINE_VERSION}" in source
    assert "outer_cfg = {" in source
    assert "\"contour_method\": periosteal_contour_method" in source
    assert "\"geodesic_bone_threshold\": float(self.maskGeodesicThreshold.value)" in source
    assert "\"fill_holes\": bool(self.maskGeodesicFillHoles.checked)" in source
    assert "\"geodesic_fill_holes\": bool(self.maskGeodesicFillHoles.checked)" in source
    assert "if selected_profile == \"ped-fx\":" in source
    assert "mask_method = \"seg_gauss\"" in source
    assert "profile_masks_cfg = (profile_cfg.get(\"masks\") or {})" in source
    assert "\"segmentation\": segmentation_cfg" in source
    assert "periosteal_contour_method = \"geodesic_fracture\"" in source
    assert "masks_override[\"roles\"] = [\"full\"]" in source
    assert "masks_override[\"inner\"] = {\"contour_method\": \"none\"}" in source
    assert "profile_initial_translation = profile_multistack_cfg.get(\"initial_translation_voxels\")" in source
    assert "\"initial_translation_voxels\": initial_translation_voxels" in source
    assert "\"masks\": masks_override" in source
    assert "Install / Update timelapsed-hrpqct" not in source
    assert "self.installBtn.clicked.connect(self._on_install_pipeline)" not in source


def test_timelapsed_pipeline_writes_sparse_override_config_for_profiles():
    source = PIPELINE_MODULE.read_text()

    create_override_start = source.index("    def create_override_config(self, settings_dict, results_root=None):")
    create_override_end = source.index("    def cleanup_temp_files", create_override_start)
    create_override_source = source[create_override_start:create_override_end]

    assert "self.default_config_path()" not in create_override_source
    assert "yaml.safe_dump(settings_dict" in create_override_source
    assert "slicer_run_configs" in create_override_source
