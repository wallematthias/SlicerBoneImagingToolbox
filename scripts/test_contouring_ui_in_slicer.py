"""Run with Slicer --no-splash --no-main-window --python-script this-file."""
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import slicer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HRpQCTTools/SegmentationHRpQCT"),
               str(ROOT / "IOTools/BatchProcessor")]


def main():
    from SegmentationHRpQCT import SegmentationHRpQCTWidget
    from BatchProcessor import BatchProcessorWidget
    widget = SegmentationHRpQCTWidget()
    form = widget._generateForm
    assert form.labelForField(widget.segmentationMethodCombo).text == "Tissue segmentation"
    assert form.labelForField(widget.siteCombo).text == "Site preset"
    assert not hasattr(widget, "modalityCombo")
    assert widget.contourProfileCombo.findData("xct1") >= 0
    assert widget.contourProfileCombo.findData("xct2") >= 0
    assert widget.contourProfileCombo.findData("custom") >= 0
    assert widget.contourProfileCombo.findData("microct") == -1
    assert widget.contourProfileCombo.findData("auto") == -1
    assert form.indexOf(widget.contourProfileCombo) < form.indexOf(widget.contourBackendCombo)
    assert form.indexOf(widget.contourBackendCombo) < form.indexOf(widget.contourSettingsButton)
    assert form.indexOf(widget.contourSettingsButton) < form.indexOf(widget.segmentationMethodCombo)
    assert form.indexOf(widget.segmentationMethodCombo) < form.indexOf(widget.segmentationSettingsButton)
    assert widget.contourBackendCombo.findData("none") >= 0
    assert widget.contourBackendCombo.findData("geodesic") >= 0
    assert widget.siteCombo.currentData == "auto"
    widget._set_combo_by_data(widget.contourProfileCombo, "xct2")

    volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "D0000308")
    slicer.util.updateVolumeFromArray(volume, np.zeros((3, 9, 11), dtype=np.float32))
    volume.SetSpacing(.082, .082, .082)
    volume.SetAttribute("HRpQCT.AIMMetadata", json.dumps(
        {"element_size": [.0607] * 3, "processing_log": {"Site": 20}}))
    widget.volumeSelector.setCurrentNode(volume)
    assert widget._effective_modality() == "xct2"
    assert widget._selected_site(volume_node=volume, strict=True) == "radius"
    assert widget.endostealThresholdSpin.value == 380
    widget._set_combo_by_data(widget.contourProfileCombo, "xct1")
    assert widget._effective_modality() == "xct1" and widget.endostealThresholdSpin.value == 500
    widget._on_input_volume_changed()
    assert widget._effective_modality() == "xct1"  # XCTII header cannot override selection.
    assert widget._recipe_payload("Scanner test")["scanner_preset"] == "xct1"
    widget._set_combo_by_data(widget.contourProfileCombo, "xct2")
    assert widget._effective_modality() == "xct2" and widget.endostealThresholdSpin.value == 380
    widget._set_combo_by_data(widget.segmentationMethodCombo, "laplace_hamming")
    widget._set_combo_by_data(widget.contourBackendCombo, "none")
    assert widget.periostealContourCombo.currentData == widget.endostealContourCombo.currentData == "none"
    assert widget.segmentationMethodCombo.currentData == "laplace_hamming"
    widget._set_combo_by_data(widget.contourBackendCombo, "geodesic")
    assert widget.periostealContourCombo.currentData == "geodesic_fracture"
    assert widget.endostealContourCombo.currentData == "standard"
    widget._set_combo_by_data(widget.contourBackendCombo, "unet")
    widget.contourSettingsButton.collapsed = False
    widget.profileSettingsButton.collapsed = False
    assert not widget.unetDeviceCombo.isHidden() and not widget.customRecipeRowWidget.isHidden()
    assert widget.unetDeviceCombo.parent() == widget.contourSettingsButton
    assert widget._expertSections["Periosteal contour"].isHidden()
    assert widget._expertSections["Endosteal contour"].isHidden()
    widget._set_combo_by_data(widget.segmentationMethodCombo, "seg_gauss")
    widget.segmentationSettingsButton.collapsed = False
    assert widget.gaussSigmaSpin.parent() == widget.segmentationSettingsButton
    widget._set_combo_by_data(widget.segmentationMethodCombo, "laplace_hamming")
    widget._set_combo_by_data(widget.contourBackendCombo, "standard")
    assert widget.segmentationMethodCombo.currentData == "laplace_hamming"
    volume.SetAttribute("HRpQCT.AIMMetadata", json.dumps(
        {"element_size": [.082] * 3, "processing_log": {"Site": 38}}))
    widget._on_input_volume_changed()
    assert widget._effective_modality() == "xct2"  # XCTI header cannot override selection.
    widget._set_combo_by_data(widget.contourProfileCombo, "xct1")
    widget._set_combo_by_data(widget.segmentationMethodCombo, "seg_gauss")
    assert widget.gaussSigmaSpin.value == 1.2
    assert widget.trabThresholdSpin.value == 320 and widget.cortThresholdSpin.value == 450
    assert widget._collect_params()["segmentation"]["gaussian_sigma"] == 1.2
    widget._set_combo_by_data(widget.segmentationMethodCombo, "laplace_hamming")
    assert widget.lhThresholdSpin.value == 15000
    volume.SetAttribute("HRpQCT.AIMMetadata", json.dumps(
        {"element_size": [.0607] * 3, "processing_log": {"Site": 20}}))
    widget._on_input_volume_changed()

    # Legacy recipes load exactly, and changing inputs must not clobber overrides.
    recipe = {"schema": "bone-contour-recipe-v1", "site": "auto", "modality": "xct2",
              "scanner_preset": "auto",
              "methods": {"bone_segmentation": "laplace_hamming", "periosteal_contour": "standard",
                          "endosteal_contour": "standard"},
              "parameters": {"inner": {"endosteal_threshold": 444, "peel": 5},
                             "segmentation": {"gaussian_sigma": .8, "keep_largest_component": True}}}
    widget._apply_recipe(recipe)
    assert widget.contourBackendCombo.currentData == "standard"
    assert widget.endostealThresholdSpin.value == 444 and widget.peelSpin.value == 5
    assert widget.gaussSigmaSpin.value == .8
    assert not hasattr(widget, "keepLargestCheck")
    assert not widget._collect_params()["segmentation"]["keep_largest_component"]
    assert widget._current_contour_profile()["schema"] == "bone-contour-recipe-v1"
    assert widget._recipe_payload("Legacy")["scanner_preset"] == "xct2"
    assert form.indexOf(widget.contourProfileCombo) < form.indexOf(widget.siteCombo)
    assert widget.profileSettingsButton.layout().indexOf(widget.customRecipeRowWidget) >= 0
    custom_data = widget.contourProfileCombo.currentData
    widget._set_combo_by_data(widget.contourProfileCombo, "xct1")
    assert widget.endostealThresholdSpin.value == 500
    widget._set_combo_by_data(widget.contourProfileCombo, custom_data)
    assert widget.endostealThresholdSpin.value == 444 and widget.gaussSigmaSpin.value == .8
    other = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "voi-tibialeft")
    slicer.util.updateVolumeFromArray(other, np.zeros((3, 9, 11), dtype=np.float32))
    other.SetSpacing(.061, .061, .061)
    widget.volumeSelector.setCurrentNode(other)
    assert widget.endostealThresholdSpin.value == 444 and widget.peelSpin.value == 5
    widget.volumeSelector.setCurrentNode(volume)
    assert widget._recipe_payload("Test")["site"] == "auto"
    widget._set_combo_by_data(widget.contourBackendCombo, "unet")
    saved = widget._recipe_payload("CNN scene")
    assert saved["contouring_method"] == "unet"
    widget._apply_recipe(saved)
    assert widget.contourBackendCombo.currentData == "unet"

    # Unknown anatomy prompts a site selection; resolution never selects a scanner.
    volume.SetAttribute("HRpQCT.AIMMetadata", None)
    volume.SetSpacing(.04, .04, .04)
    widget._set_combo_by_data(widget.contourProfileCombo, "xct2")
    assert widget._effective_modality() == "xct2"
    widget._set_combo_by_data(widget.siteCombo, "auto")
    widget._set_combo_by_data(widget.contourBackendCombo, "standard")
    try:
        widget._validate_preset_selection()
    except ValueError as error:
        assert "select" in str(error).lower()
    else:
        raise AssertionError("Unknown automatic site accepted")
    previous_threshold = widget.endostealThresholdSpin.value
    widget._set_combo_by_data(widget.contourProfileCombo, "custom")
    assert widget._effective_modality() == "custom"
    assert widget.endostealThresholdSpin.value == previous_threshold
    assert widget.siteCombo.currentData == "none"
    assert widget._recipe_payload("My Micro-CT")["modality"] == "custom"
    widget._set_combo_by_data(widget.contourBackendCombo, "unet")
    try:
        widget._validate_preset_selection()
    except ValueError as error:
        assert "U-Net" in str(error)
    else:
        raise AssertionError("Custom accepted XCTII-only U-Net")
    widget._set_combo_by_data(widget.contourBackendCombo, "none")
    widget._set_combo_by_data(widget.siteCombo, "none")
    widget._validate_preset_selection()  # Explicit no preset: edited tissue-only settings.
    widget._set_combo_by_data(widget.segmentationMethodCombo, "seg_gauss")
    widget.trabThresholdSpin.value = 610
    widget.cortThresholdSpin.value = 700
    custom_recipe = widget._recipe_payload("My Micro-CT")
    widget.trabThresholdSpin.value = 999
    widget._apply_recipe(custom_recipe)
    assert widget._effective_modality() == "custom"
    assert widget.trabThresholdSpin.value == 610 and widget.cortThresholdSpin.value == 700
    widget._on_input_volume_changed()
    assert widget.trabThresholdSpin.value == 610  # Preserve named custom overrides.
    density = np.zeros((3, 9, 11), dtype=np.float32)
    density[:, 2:7, 2:9] = 1000
    slicer.util.updateVolumeFromArray(volume, density)
    node, _, metadata = widget.logic.generate_bone_masks(
        volume, site="unparsed", segmentation_method="seg_gauss",
        periosteal_contour_method="none", endosteal_contour_method="none",
        params=widget._collect_params(site="unparsed"), create_labelmaps=False)
    assert metadata["emitted_roles"] == ["seg"]
    assert node.GetSegmentation().GetNumberOfSegments() == 1
    assert metadata["voxel_counts"]["seg"] > 0
    assert all(metadata["voxel_counts"][role] == 0 for role in ("full", "trab", "cort"))

    batch = BatchProcessorWidget()
    assert batch.toolCombo.findData("deep_learning_segmentation") == -1
    batch.toolCombo.setCurrentIndex(batch.toolCombo.findData("bone_contouring"))
    batch.profileCombo.setCurrentIndex(batch.profileCombo.findData("unet"))
    assert batch._selected_tool_key() == "deep_learning_segmentation"
    assert not batch.unetDeviceCombo.isHidden()
    assert batch.profileCombo.findData("XtremeCTII-LH") >= 0
    batch.profileCombo.setCurrentIndex(batch.profileCombo.findData("XtremeCTII-LH"))
    assert batch._selected_tool_key() == "bone_contouring"
    assert batch.unetDeviceCombo.isHidden()
    screenshot = os.environ.get("HRPQCT_TEST_UI_SCREENSHOT")
    if screenshot:
        widget.contourProfileCombo.setCurrentIndex(0)
        volume.SetSpacing(.0607, .0607, .0607)
        volume.SetAttribute("HRpQCT.AIMMetadata", json.dumps(
            {"element_size": [.0607] * 3, "processing_log": {"Site": 20}}))
        widget._set_combo_by_data(widget.contourProfileCombo, "xct2")
        widget._set_combo_by_data(widget.siteCombo, "auto")
        widget._set_combo_by_data(widget.contourBackendCombo, "standard")
        widget.contourSettingsButton.collapsed = True
        widget.segmentationSettingsButton.collapsed = True
        widget.profileSettingsButton.collapsed = True
        widget.parent.resize(650, 400)
        widget.parent.show()
        slicer.app.processEvents()
        assert widget.parent.grab().save(screenshot, "PNG")
    widget.cleanup()
    batch.cleanup()
    print("CONTOURING_UI_CHECKS_PASSED", flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    slicer.app.exit(1)
else:
    slicer.app.exit(0)
