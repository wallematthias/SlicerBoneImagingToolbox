from __future__ import annotations

HRPQCT_DENSITY_PRESET_NAME = "HR-pQCT Density"


def ensure_hrpqct_density_preset(volume_rendering_logic, preset_node):
    """Add the toolbox HR-pQCT preset without replacing Slicer's preset scene."""
    preset = volume_rendering_logic.GetPresetByName(HRPQCT_DENSITY_PRESET_NAME)
    if preset is not None:
        volume_rendering_logic.RemovePreset(preset)

    preset = volume_rendering_logic.AddPreset(preset_node, None, True)
    if preset is None:
        raise RuntimeError(f"Failed to add volume-rendering preset {HRPQCT_DENSITY_PRESET_NAME!r}")
    return preset
