from __future__ import annotations

from pathlib import Path
import struct
import xml.etree.ElementTree as ET

from SlicerBoneImagingToolboxLib.volume_rendering_presets import (
    HRPQCT_DENSITY_PRESET_NAME,
    ensure_hrpqct_density_preset,
)


ROOT = Path(__file__).resolve().parents[1]
PRESET_PATH = ROOT / "Setup" / "BoneImagingToolboxSetup" / "Resources" / "HRpQCTVolumeRenderingPresets.mrml"


class PresetLogic:
    def __init__(self, preset=None):
        self.preset = preset
        self.other_preset = object()
        self.presets = [self.other_preset] + ([preset] if preset is not None else [])
        self.added = []
        self.removed = []

    def GetPresetByName(self, name):
        assert name == HRPQCT_DENSITY_PRESET_NAME
        return self.preset

    def AddPreset(self, preset, icon=None, append_to_end=False):
        self.added.append((preset, icon, append_to_end))
        self.preset = object()
        self.presets.append(self.preset)
        return self.preset

    def RemovePreset(self, preset):
        self.removed.append(preset)
        self.presets.remove(preset)
        self.preset = None

    def LoadCustomPresetsScene(self, _path):
        raise AssertionError("Custom preset registration must not replace Slicer's preset scene")


def _pairs(serialized: str) -> list[tuple[float, float]]:
    values = [float(value) for value in serialized.split()]
    count = int(values[0])
    assert len(values) == 1 + count
    return list(zip(values[1::2], values[2::2]))


def _color_points(serialized: str) -> list[tuple[float, float, float, float]]:
    values = [float(value) for value in serialized.split()]
    count = int(values[0])
    assert len(values) == 1 + count
    return [tuple(values[index : index + 4]) for index in range(1, len(values), 4)]


def test_hrpqct_density_preset_keeps_low_density_transparent_and_emphasizes_cortex() -> None:
    preset = ET.parse(PRESET_PATH).getroot().find("VolumeProperty")

    assert preset is not None
    assert preset.attrib["name"] == HRPQCT_DENSITY_PRESET_NAME
    opacity = dict(_pairs(preset.attrib["scalarOpacity"]))
    assert opacity[300.0] == 0.0
    assert 0.0 < opacity[320.0] < opacity[450.0]
    assert opacity[450.0] == 1.0
    assert opacity[1200.0] == 1.0
    assert preset.attrib["effectiveRange"] == "320 1200"


def test_hrpqct_density_preset_avoids_oversized_gpu_lookup_tables() -> None:
    preset = ET.parse(PRESET_PATH).getroot().find("VolumeProperty")

    assert preset is not None
    opacity_x = [point[0] for point in _pairs(preset.attrib["scalarOpacity"])]
    color_x = [point[0] for point in _color_points(preset.attrib["colorTransfer"])]
    assert min(b - a for a, b in zip(opacity_x, opacity_x[1:])) >= 1.0
    assert min(b - a for a, b in zip(color_x, color_x[1:])) >= 1.0


def test_hrpqct_density_preset_replaces_only_its_previous_version() -> None:
    stale_preset = object()
    logic = PresetLogic(stale_preset)
    preset_node = object()

    registered = ensure_hrpqct_density_preset(logic, preset_node)

    assert registered is logic.preset
    assert logic.presets[0] is logic.other_preset
    assert len(logic.presets) == 2
    assert logic.removed == [stale_preset]
    assert logic.added == [(preset_node, None, True)]


def test_hrpqct_density_preset_has_packaged_thumbnail() -> None:
    root = ET.parse(PRESET_PATH).getroot()
    preset = root.find("VolumeProperty")
    storage = root.find("VolumeArchetypeStorage")

    assert preset is not None
    assert storage is not None
    assert preset.attrib["references"] == "IconVolume:vtkMRMLVectorVolumeNodeHRpQCTDensity;"
    assert storage.attrib["fileName"] == "HRpQCT-Density.png"
    icon_path = PRESET_PATH.parent / storage.attrib["fileName"]
    with icon_path.open("rb") as stream:
        header = stream.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", header[16:24]) == (128, 100)


def test_hrpqct_density_preset_resource_is_packaged() -> None:
    cmake = (ROOT / "Setup" / "BoneImagingToolboxSetup" / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "Resources/HRpQCTVolumeRenderingPresets.mrml" in cmake


def test_setup_module_registers_hrpqct_preset_after_initialization() -> None:
    setup_source = (
        ROOT / "Setup" / "BoneImagingToolboxSetup" / "BoneImagingToolboxSetup.py"
    ).read_text(encoding="utf-8")

    assert 'parent.dependencies = ["VolumeRendering"]' in setup_source
    assert 'slicer.app.connect("startupCompleted()", self._register_volume_rendering_presets)' in setup_source
    assert 'volume_rendering_module = getattr(slicer.modules, "volumerendering", None)' in setup_source
    assert "if volume_rendering_module is None:" in setup_source
    assert "qt.QTimer.singleShot(0, self._register_volume_rendering_presets)" not in setup_source
    assert "ensure_hrpqct_density_preset(" in setup_source
