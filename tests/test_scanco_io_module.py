from pathlib import Path
import ast
import json
import struct
import sys

import numpy as np
import SimpleITK as sitk
import pytest


MODULE = Path(__file__).resolve().parents[1] / "IOTools" / "ScancoIO" / "ScancoIO.py"
LIB_DIR = MODULE.parent
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from ScancoIOLib import aim_io  # noqa: E402


def test_pixel_space_import_is_visibly_labelled_and_retains_warning():
    tree = ast.parse(MODULE.read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_attach_import_metadata")
    namespace = {"json": json, "_json_default": str}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.endswith("_ATTRIBUTE")
            for target in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODULE), "exec"), namespace)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(MODULE), "exec"), namespace)

    class Node:
        def __init__(self):
            self.attributes = {}
            self.name = "scout"

        def SetAttribute(self, key, value):
            self.attributes[key] = value

        def GetName(self):
            return self.name

        def SetName(self, name):
            self.name = name

    node = Node()
    namespace["_attach_import_metadata"](None, node, image_path="scout.SCV",
        scaling="native", load_as="volume", metadata={
            "format": "SCV", "geometry_unit": "pixel", "geometry_warning": "No physical measurements"
        })
    assert node.name == "scout [pixel space]"
    assert node.attributes["BoneImaging.ScancoIO.GeometryUnit"] == "pixel"
    assert node.attributes["BoneImaging.ScancoIO.GeometryWarning"] == "No physical measurements"


def test_scanco_io_can_import_aim_as_segmentation_node() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "as_segmentation=False" in source
    assert 'slicer_pip_install("aimio-py>=0.1.8 numpy>=1.26,<3.0")' in source
    assert "Install / Update Scanco I/O" not in source
    assert "self.installButton.clicked.connect(self._install_core)" not in source
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


def test_scanco_io_registers_aimio_dispatch_file_reader() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert '"read_image"' in source
    assert "class ScancoIOVariantFileReader" in source
    assert "def canLoadFileConfidence(self, filePath)" in source
    assert "def load(self, properties)" in source
    assert "self.parent.loadedNodes = [node.GetID()]" in source
    assert "return 0.95" in source
    for suffix in (".aim", ".isq", ".scv", ".gobj"):
        assert suffix in source


def test_scanco_io_builds_four_explicit_drag_drop_reader_modules() -> None:
    cmake = (LIB_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    expected = {
        "ScancoVolume": ("native", "volume"),
        "ScancoHU": ("hu", "volume"),
        "ScancoDensity": ("density", "volume"),
        "ScancoSegmentation": ("native", "segmentation"),
    }

    for module_name, (scaling, load_as) in expected.items():
        path = LIB_DIR / f"{module_name}.py"
        source = path.read_text(encoding="utf-8")

        assert f"{module_name}.py" in cmake
        assert f"class {module_name}(ScriptedLoadableModule)" in source
        assert "parent.hidden = True" in source
        assert f"class {module_name}FileReader(ScancoIOVariantFileReader)" in source
        assert f'description="{module_name}"' in source
        assert f'file_type="{module_name}"' in source
        assert f'scaling="{scaling}"' in source
        assert f'load_as="{load_as}"' in source


def test_explicit_drag_drop_readers_limit_extensions_by_use() -> None:
    scanco_source = MODULE.read_text(encoding="utf-8")

    assert 'VOLUME_READER_EXTENSIONS = (".aim", ".isq", ".scv", ".rad")' in scanco_source
    assert 'SEGMENTATION_READER_EXTENSIONS = (".aim", ".isq", ".gobj")' in scanco_source


def test_explicit_drag_drop_readers_do_not_accept_scaling_overrides() -> None:
    scanco_source = MODULE.read_text(encoding="utf-8")

    assert 'scaling = self._scaling' in scanco_source
    assert 'load_as = self._load_as' in scanco_source
    assert 'properties.get("scaling",' not in scanco_source
    assert 'properties.get("loadAs",' not in scanco_source


def test_scanco_import_ui_supports_new_formats_without_mu_scaling() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert "Import Scanco image" in source
    assert "AIM/ISQ/SCV/GOBJ/RAD files" in source
    assert '("Density", "density")' in source
    assert '("Native", "native")' in source
    assert '("HU", "hu")' in source
    assert '("Mu", "mu")' not in source
    assert 'self.importAsCombo.addItem("Transform (image geometry)", "transform")' in source


def test_scanco_io_separates_import_and_export_into_tabs() -> None:
    source = MODULE.read_text(encoding="utf-8")
    widget_setup = source[source.index("    def setup(self):", source.index("class ScancoIOWidget")) :]

    assert "self.modeTabs = qt.QTabWidget()" in widget_setup
    assert 'self.modeTabs.addTab(importTab, "Import")' in widget_setup
    assert 'self.modeTabs.addTab(exportTab, "Export")' in widget_setup
    assert "self._build_import_section(importLayout)" in widget_setup
    assert "self._build_export_section(exportLayout)" in widget_setup


def test_aimio_image_dispatcher_supports_all_scanco_extensions() -> None:
    assert aim_io.supported_image_extensions() == (".aim", ".isq", ".scv", ".gobj", ".rad")
    assert aim_io.resolve_image_format("radius.AIM") == "aim"
    assert aim_io.resolve_image_format("radius.ISQ") == "isq"
    assert aim_io.resolve_image_format("scout.SCV") == "scv"
    assert aim_io.resolve_image_format("contour.GOBJ") == "gobj"
    assert aim_io.resolve_image_format("scout.RAD;1") == "rad"


def test_rad_adapter_loads_native_single_plane_with_physical_spacing(tmp_path):
    path = tmp_path / "scout.RAD;1"
    header = bytearray(512)
    header[:16] = b"CTDATA-HEADER_V1"
    struct.pack_into("<i", header, 16, 9)
    struct.pack_into("<3i", header, 44, 3, 2, 1)
    struct.pack_into("<3i", header, 56, 180000, 160000, 0)
    path.write_bytes(header + struct.pack("<6h", -1, 0, 1, 256, 9186, 32767))

    image, metadata = aim_io.read_image(path, scaling="native")

    assert image.GetSize() == (3, 2, 1)
    assert image.GetSpacing() == pytest.approx((0.06, 0.08, 1.0))
    assert image.GetOrigin() == (0.0, 0.0, 0.0)
    assert sitk.GetArrayFromImage(image).tolist() == [[[-1, 0, 1], [256, 9186, 32767]]]
    assert metadata["unit"] == "native"
    assert metadata["format"] == "RAD"
    assert aim_io.suggested_slicer_load_as("scout_mask.RAD") == "volume"
    for scaling in ("hu", "density"):
        with pytest.raises(ValueError, match="RAD.*native"):
            aim_io.read_image(path, scaling=scaling)


def test_only_native_drag_drop_reader_accepts_rad():
    """Evaluate real registrations without importing the GUI-only Slicer modules."""
    tree = ast.parse(MODULE.read_text())
    namespace = {"Path": Path}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.endswith("READER_EXTENSIONS")
            for target in node.targets
        ) or isinstance(node, ast.ClassDef) and node.name == "ScancoIOVariantFileReader":
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODULE), "exec"), namespace)
    for module_name, expected in (("ScancoVolume", 0.95), ("ScancoHU", 0.0),
                                  ("ScancoDensity", 0.0), ("ScancoSegmentation", 0.0)):
        module_tree = ast.parse((MODULE.parent / f"{module_name}.py").read_text())
        reader_class = next(node for node in module_tree.body
                            if isinstance(node, ast.ClassDef) and node.name.endswith("FileReader"))
        exec(compile(ast.Module(body=[reader_class], type_ignores=[]), str(MODULE), "exec"), namespace)
        reader = namespace[f"{module_name}FileReader"](object())
        assert reader.canLoadFileConfidence("scout.RAD;1") == expected


def test_rad_import_ui_selects_native_volume_and_restores_controls_for_ct():
    tree = ast.parse(MODULE.read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_on_import_as_changed")
    namespace = {"_aim_io_module": lambda: aim_io}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(MODULE), "exec"), namespace)

    class Combo:
        def __init__(self, items, selected):
            self.items = items
            self.currentData = selected
            self.enabled = True

        def findData(self, data):
            return self.items.index(data)

        def setCurrentIndex(self, index):
            self.currentData = self.items[index]

    class Widget:
        importPathEdit = type("PathEdit", (), {"text": "scout.RAD;1"})()
        importAsCombo = Combo(["volume", "segmentation", "transform"], "segmentation")
        scalingCombo = Combo(["density", "native", "hu"], "density")

    widget = Widget()
    namespace["_on_import_as_changed"](widget)
    assert widget.importAsCombo.currentData == "volume"
    assert widget.scalingCombo.currentData == "native"
    assert not widget.importAsCombo.enabled
    assert not widget.scalingCombo.enabled
    widget.importPathEdit.text = "scan.ISQ"
    namespace["_on_import_as_changed"](widget)
    assert widget.importAsCombo.enabled
    assert widget.scalingCombo.enabled


def test_drag_drop_auto_load_target_detects_segmentation_inputs() -> None:
    assert aim_io.suggested_slicer_load_as("contour.GOBJ") == "segmentation"
    assert aim_io.suggested_slicer_load_as("SUBJ001_DR_T1_TRAB_MASK.AIM") == "segmentation"
    assert aim_io.suggested_slicer_load_as("subject_radius_SEG.AIM") == "segmentation"
    assert aim_io.suggested_slicer_load_as("subject_REGMASK.AIM") == "segmentation"
    assert aim_io.suggested_slicer_load_as("subject_ROI1.AIM") == "segmentation"
    assert aim_io.suggested_slicer_load_as("radius.AIM") == "volume"
    assert aim_io.suggested_slicer_load_as("scout.SCV") == "volume"


def test_generic_read_image_dispatches_isq_density_scaling(monkeypatch) -> None:
    calls = []

    class FakeAimIO:
        def read_image(self, path, format="auto", **kwargs):
            calls.append((path, format, kwargs))
            return np.zeros((2, 3, 4), dtype=np.int16), {
                "format": "ISQ",
                "dimensions": (4, 3, 2),
                "spacing": (0.1, 0.2, 0.3),
                "origin": (1.0, 2.0, 3.0),
                "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            }

    monkeypatch.setattr(aim_io, "_load_py_aimio", lambda: FakeAimIO())

    image, metadata = aim_io.read_image("radius.ISQ", scaling="density")

    assert calls == [("radius.ISQ", "isq", {"unit": "density"})]
    assert image.GetSize() == (4, 3, 2)
    assert image.GetSpacing() == (0.1, 0.2, 0.3)
    assert image.GetOrigin() == (1.0, 2.0, 3.0)
    assert metadata["format"] == "ISQ"
    assert metadata["unit"] == "density"


def test_generic_read_image_keeps_scout_views_as_single_slice_volumes(monkeypatch) -> None:
    class FakeAimIO:
        def read_image(self, path, format="auto", **kwargs):
            return np.zeros((3, 4), dtype=np.uint8), {
                "format": "SCV",
                "dimensions": (3, 4),
                "spacing": (0.1, 0.2, 1.0),
                "origin": (1.0, 2.0, 0.0),
            }

    monkeypatch.setattr(aim_io, "_load_py_aimio", lambda: FakeAimIO())

    image, metadata = aim_io.read_image("scout.SCV", scaling="hu")

    assert image.GetSize() == (4, 3, 1)
    assert image.GetSpacing() == (0.1, 0.2, 1.0)
    assert metadata["format"] == "SCV"
    assert metadata["unit"] == "native"


def test_scanco_io_can_export_labelmaps_and_segmentations_without_binarizing() -> None:
    source = MODULE.read_text(encoding="utf-8")

    assert 'volume_node.IsA("vtkMRMLSegmentationNode")' in source
    assert '"vtkMRMLLabelMapVolumeNode"' in source
    assert "ExportAllSegmentsToLabelmapNode" in source
    assert "GetReferenceImageGeometryReferenceRole()" in source
    assert 'self.exportModeCombo.addItem("Label image (preserve labels)", "label")' in source
    assert 'self.exportModeCombo.findData("label")' in source
    assert 'unit="native" if mode == "label" else self.unitCombo.currentData' in source
    assert "if is_segmentation and metadata is not None" not in source
    assert "if metadata is not None:\n            image = aim_io.image_with_aim_metadata_geometry(image, metadata)" in source
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


def test_aim_metadata_position_preserves_integer_voxel_origins() -> None:
    image = sitk.Image([4, 5, 6], sitk.sitkUInt8)
    image.SetSpacing((0.082, 0.082, 0.082))
    image.SetOrigin((566 * 0.082, 465 * 0.082, 0.0))
    metadata = {"position": (0, 0, 0), "offset": (0, 0, 0)}

    aim_io._refresh_position_from_image_geometry(metadata, image)

    assert metadata["position"] == (566, 465, 0)
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


def test_mask_write_overwrites_geometry_with_mask_image_geometry() -> None:
    image = sitk.Image([4, 5, 6], sitk.sitkUInt8)
    image.SetSpacing((0.082, 0.083, 0.084))
    image.SetOrigin((46.412, 38.13, 1.23))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0))
    metadata = {
        "dimensions": (1, 1, 1),
        "spacing": (1.0, 1.0, 1.0),
        "element_size": (1.0, 1.0, 1.0),
        "origin": (0.0, 0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        "offset": (0, 0, 0),
    }

    aim_io._prepare_aim_metadata_for_write(metadata, image, mask=True)

    assert metadata["dimensions"] == (4, 5, 6)
    assert metadata["spacing"] == image.GetSpacing()
    assert metadata["element_size"] == image.GetSpacing()
    assert metadata["origin"] == image.GetOrigin()
    assert metadata["direction"] == image.GetDirection()
