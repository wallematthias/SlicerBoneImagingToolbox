"""Regressions for cross-site artifacts and registered scene loading."""
from pathlib import Path
import types

import pytest

from test_batch_processor_module import (
    _import_batch_processor_module,
    test_functional_bone_command_runs_with_core_native_map_generation as _run_functional_case,
)


def test_batch_contouring_offers_scene_exported_standard_recipes(monkeypatch):
    module = _import_batch_processor_module(monkeypatch)
    monkeypatch.setattr(module, "list_profiles", lambda tool: [types.SimpleNamespace(name="Untested recipe", kind="json")], raising=False)
    monkeypatch.setattr(module, "load_profile_payload", lambda record: {"contouring_method": "standard"}, raising=False)
    widget = types.SimpleNamespace(logic=module.BatchProcessorLogic())
    profiles = module.BatchProcessorWidget._profiles_for_tool(widget, "bone_contouring")
    values = [value for _label, value, _registered in profiles]
    assert "Untested recipe" in values
    assert "unet" in values and "XtremeCTI" in values and "XtremeCTII" in values


@pytest.mark.parametrize("profile,directory", [
    ("functional-bone-native", "functional_bone_native_measurements"),
    ("functional-bone", "functional_bone_measurements"),
])
def test_functional_masks_are_discovered_only_for_the_requested_site(monkeypatch, tmp_path, profile, directory):
    module = _import_batch_processor_module(monkeypatch)
    folder = tmp_path / "derivatives/Microarchitecture/sub-001/ses-01/xct" / directory
    folder.mkdir(parents=True)
    masks = {}
    for site in ("radiusleft", "tibialeft"):
        stem = f"sub-001_ses-01_voi-{site}"
        (folder / f"{stem}_measurements.csv").write_text("Parameter,Mean\nCt.Po,0.1\n")
        masks[site] = folder / f"{stem}_desc-functional-bone-analysis_mask.nii.gz"
        masks[site].touch()
    # An old shared mask cannot be assigned safely to either site.
    (folder / "functional_bone_analysis_mask.nii.gz").touch()
    for site in masks:
        row = dict(subject="001", session_value="01", voi_value=site, profile=profile)
        outputs = module.BatchProcessorLogic._functional_bone_output_paths_for_row(tmp_path, row)
        assert masks[site] in outputs
        assert masks["tibialeft" if site == "radiusleft" else "radiusleft"] not in outputs
        assert folder / "functional_bone_analysis_mask.nii.gz" not in outputs


def test_legacy_shared_functional_mask_is_not_loaded_for_a_case(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    folder = tmp_path / "derivatives/Microarchitecture/sub-001/ses-01/xct/functional_bone_measurements"
    folder.mkdir(parents=True)
    (folder / "functional_bone_analysis_mask.nii.gz").touch()
    row = dict(subject="001", session_value="01", voi_value="radiusleft", profile="functional-bone")
    assert module.BatchProcessorLogic._functional_bone_output_paths_for_row(tmp_path, row) == []


@pytest.mark.parametrize("profile,directory,count", [
    ("functional-bone", "functional_bone_measurements", 17),
    ("functional-bone-native", "functional_bone_native_measurements", 26),
])
def test_running_radius_then_tibia_retains_both_functional_masks(monkeypatch, tmp_path, profile, directory, count):
    _run_functional_case(tmp_path, monkeypatch, False, profile, directory, count, "radiusleft")
    folder = tmp_path / "derivatives/Microarchitecture/sub-001/ses-001/xct" / directory
    radius = folder / "sub-001_ses-001_voi-radiusleft_desc-functional-bone-analysis_mask.nii.gz"
    original = radius.read_bytes()
    _run_functional_case(tmp_path, monkeypatch, False, profile, directory, count, "tibialeft")
    assert radius.read_bytes() == original
    assert (folder / "sub-001_ses-001_voi-tibialeft_desc-functional-bone-analysis_mask.nii.gz").is_file()


def test_case_specific_functional_mask_is_routed_as_a_segmentation(monkeypatch):
    module = _import_batch_processor_module(monkeypatch)
    assert module.BatchProcessorWidget._is_functional_bone_analysis_mask(
        Path("sub-001_ses-01_voi-radiusleft_desc-functional-bone-analysis_mask.nii.gz"))
    assert not module.BatchProcessorWidget._is_functional_bone_analysis_mask(
        Path("sub-001_ses-01_voi-radiusleft_measurements.csv"))


def test_loading_legacy_shared_functional_mask_requests_regeneration(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    path = tmp_path / "functional_bone_analysis_mask.nii.gz"
    path.touch()
    messages = []
    widget = object.__new__(module.BatchProcessorWidget)
    widget._append_log = messages.append
    widget._remove_existing_node_named = lambda name: None
    widget._ensure_loaded_source_volume = lambda path: None
    widget._load_mask_as_labelmap = lambda *args: None
    module.slicer.mrmlScene = types.SimpleNamespace(
        AddNewNodeByClass=lambda *args: types.SimpleNamespace(CreateDefaultDisplayNodes=lambda: None),
        RemoveNode=lambda node: None)
    row = dict(subject="001", session_value="01", voi_value="radiusleft", profile="functional-bone")
    assert widget._load_functional_bone_analysis_outputs(row, [path]) == 0
    assert any(path.name in message and "rerun" in message.lower() for message in messages)


def test_loading_voidspace_tables_preserves_other_sites_and_timepoints(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    nodes = {}

    class Table:
        def __init__(self, name, data):
            self.name, self.data = name, data

        def SetName(self, name):
            nodes.pop(self.name, None)
            self.name = name
            nodes[name] = self

    def load_table(path):
        node = Table(Path(path).stem, Path(path).read_text())
        nodes[node.name] = node
        return node

    module.slicer.util.loadTable = load_table
    widget = object.__new__(module.BatchProcessorWidget)
    widget._remove_existing_node_named = lambda name: nodes.pop(name, None)
    widget._show_table_node = lambda node: None
    widget._append_log = lambda message: None
    for site, session, value in (("radiusleft", "01", 11), ("radiusleft", "02", 12), ("tibialeft", "01", 21)):
        folder = tmp_path / site / session
        folder.mkdir(parents=True)
        path = folder / "voidspace_measurements.csv"
        path.write_text(f"VS.V\n{value}\n")
        row = dict(subject="001", session_value=session, voi_value=site, profile="registered")
        widget._load_voidspace_outputs(row, [path])
        widget._load_voidspace_outputs(row, [path])  # Reload is idempotent.
    assert len(nodes) == 3
    assert {node.data for node in nodes.values()} == {"VS.V\n11\n", "VS.V\n12\n", "VS.V\n21\n"}
    assert all("sub-001" in name and "voi-" in name and "ses-" in name for name in nodes)


class _Overlay:
    def __init__(self):
        self.attributes = {}
        self.visible = True

    def GetAttribute(self, name):
        return self.attributes.get(name)

    def SetAttribute(self, name, value):
        self.attributes[name] = value

    def GetDisplayNode(self):
        return self

    def SetVisibility(self, value):
        self.visible = bool(value)


def test_registered_overlay_switch_hides_previous_timepoint_not_same_case_or_manual_nodes(monkeypatch):
    module = _import_batch_processor_module(monkeypatch)
    baseline, followup_common, followup_analysis, manual = [_Overlay() for _ in range(4)]
    nodes = [baseline, followup_common, followup_analysis, manual]
    module.slicer.util.getNodesByClass = lambda kind: nodes
    row = dict(subject="001", voi_value="radiusleft", session_value="01", registered=True)
    show = module.BatchProcessorWidget._show_case_overlay
    show(baseline, row)
    show(followup_common, {**row, "session_value": "02"})
    show(followup_analysis, {**row, "session_value": "02"})
    assert not baseline.visible
    assert followup_common.visible and followup_analysis.visible and manual.visible


@pytest.mark.parametrize("role,visible", [("baseline_seg", False), ("baseline_analysis_mask", False),
                                          ("followup_seg", True), ("followup_analysis_mask", True)])
def test_longitudinal_segments_default_to_followup_without_removing_baseline(monkeypatch, role, visible):
    module = _import_batch_processor_module(monkeypatch)
    segment = types.SimpleNamespace(SetName=lambda name: None, SetColor=lambda *color: None, SetTag=lambda *tag: None)
    visibility = {"segment-id": True}
    segmentation = types.SimpleNamespace(GetNumberOfSegments=lambda: 1, GetNthSegmentID=lambda index: "segment-id",
                                         GetSegment=lambda id: segment)
    display = types.SimpleNamespace(SetSegmentVisibility=lambda id, value: visibility.update({id: bool(value)}))
    node = types.SimpleNamespace(GetSegmentation=lambda: segmentation, GetDisplayNode=lambda: display)
    module.BatchProcessorWidget._name_last_segment(node, "Loaded segment", role)
    assert visibility["segment-id"] is visible
    assert segmentation.GetNumberOfSegments() == 1
