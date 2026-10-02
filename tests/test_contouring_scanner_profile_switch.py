"""Exercise profile selection without importing Slicer's GUI runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (Path(__file__).resolve().parents[1] / "HRpQCTTools"
          / "SegmentationHRpQCT" / "SegmentationHRpQCT.py")


def widget_for(profile, method):
    tree = ast.parse(SOURCE.read_text())
    methods = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "SegmentationHRpQCTWidget").body
    names = {"_apply_preset_values", "_apply_modality_preset"}
    namespace = {}
    exec(compile(ast.Module(body=[node for node in methods
                                 if isinstance(node, ast.FunctionDef) and node.name in names],
                            type_ignores=[]), str(SOURCE), "exec"), namespace)

    class Widget:
        _apply_preset_values = namespace["_apply_preset_values"]
        _apply_modality_preset = namespace["_apply_modality_preset"]

        def __init__(self):
            self.segmentationMethodCombo = SimpleNamespace(currentData=method)
            self.siteCombo = SimpleNamespace(currentData="radius")
            self.events = []

        def _current_contour_profile(self):
            return profile

        def _effective_modality(self):
            return self._scannerPreset

        def _use_site_preset_params(self):
            return False  # Generation uses visible settings, including overrides.

        def _set_combo_by_data(self, combo, data):
            combo.currentData = data

        def _apply_segmentation_preset(self):
            self.events.append(("segmentation_defaults", self.segmentationMethodCombo.currentData))

        def _apply_site_preset(self):
            self.events.append(("site_defaults", self.segmentationMethodCombo.currentData))

        def _apply_recipe(self, recipe):
            self.segmentationMethodCombo.currentData = recipe["methods"]["bone_segmentation"]
            self.events.append(("recipe", recipe))

    return Widget()


@pytest.mark.parametrize("scanner,previous,expected", [
    ("xct1", "seg_gauss", "laplace_hamming"),
    ("xct2", "laplace_hamming", "seg_gauss"),
    ("xct1", "none", "laplace_hamming"),
    ("xct2", "none", "seg_gauss"),
])
def test_builtin_selection_sets_tissue_method_before_applying_defaults(scanner, previous, expected):
    widget = widget_for({"scanner_preset": scanner}, previous)
    widget._apply_preset_values()
    assert widget.segmentationMethodCombo.currentData == expected
    assert widget.events == [("segmentation_defaults", expected), ("site_defaults", expected)]


@pytest.mark.parametrize("method", ["none", "seg_gauss", "laplace_hamming"])
def test_custom_manual_profile_keeps_selected_method(method):
    widget = widget_for({"scanner_preset": "custom"}, method)
    widget._apply_preset_values()
    assert widget.segmentationMethodCombo.currentData == method
    assert widget.siteCombo.currentData == "none"
    assert widget.events == []


@pytest.mark.parametrize("scanner,method", [("xct1", "seg_gauss"), ("xct2", "laplace_hamming")])
def test_saved_recipe_method_wins_over_scanner_default(scanner, method):
    recipe = {"schema": "bone-contour-recipe-v1", "scanner_preset": scanner,
              "methods": {"bone_segmentation": method}}
    widget = widget_for(recipe, "none")
    widget._apply_preset_values()
    assert widget.segmentationMethodCombo.currentData == method
    assert widget.events == [("recipe", recipe)]
