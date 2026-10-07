"""Exercise scientific-method invalidation and native-map scene reporting."""

import json
import types

import numpy as np
import SimpleITK as sitk

from bone_microarchitecture.method import METHOD_ID
from test_batch_processor_module import _import_batch_processor_module
from test_microarchitecture_module import _load_microarchitecture_module


def test_legacy_measurements_do_not_skip_updated_method(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    path = tmp_path / "derivatives/Microarchitecture/sub-001/ses-001/xct/measurements/sub-001_ses-001_voi-radiusleft_measurements.csv"
    path.parent.mkdir(parents=True)
    path.write_text("Parameter,Mean\nCt.Po,0.25\n")
    logic = module.BatchProcessorLogic()
    outputs = logic._discover_existing_outputs(tmp_path, "Microarchitecture")
    assert logic._existing_outputs_for_profile("microarchitecture", False, outputs) == ()
    path.with_suffix(".json").write_text(json.dumps({"measurement_method": METHOD_ID}))
    outputs = logic._discover_existing_outputs(tmp_path, "Microarchitecture")
    assert [item.path for item in logic._existing_outputs_for_profile("microarchitecture", False, outputs)] == [path]


def test_selected_pore_labelmap_is_a_map_not_a_measurement_table(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    artifact = module.BatchArtifact(
        tmp_path / "pore_mask.npy", module.CaseKey("001", "001", "radiusleft", None),
        "material_labelmap", "Microarchitecture", metadata={"map_name": "Ct.Po.Mask"},
    )
    assert module.BatchProcessorLogic._is_map_output_for_tool("microarchitecture", artifact)
    assert not module.BatchProcessorLogic._is_measurement_output_for_tool("microarchitecture", artifact)


def test_scene_reports_native_pores_without_recleaning_common_region(monkeypatch):
    module = _load_microarchitecture_module(monkeypatch, "ipl_microarchitecture_scene")
    cort = np.zeros((9, 41, 41), dtype=np.uint8)
    cort[:, 3:38, 3:38] = 1
    cort[:, 12:29, 12:29] = 0
    bone = cort.copy()
    bone[2:7, 7, 7] = 0
    region = np.zeros_like(bone)
    region[4] = 1
    arrays = {"bone": bone, "peri": np.ones_like(bone), "trab": np.zeros_like(bone), "cort": cort, "region": region}
    nodes = {key: types.SimpleNamespace(GetName=lambda key=key: key, GetID=lambda key=key: key) for key in arrays}
    logic = module.BoneMicroarchitectureLogic()
    monkeypatch.setattr(logic, "_first_available_reference_node", lambda *args: nodes["bone"])
    monkeypatch.setattr(logic, "_volume_to_sitk_uint8", lambda node, *args, **kwargs: sitk.GetImageFromArray(arrays[node.GetName()]))
    attributes = {}
    table = types.SimpleNamespace(SetAttribute=lambda key, value: attributes.update({key: value}))
    monkeypatch.setattr(logic, "_create_measurement_table", lambda *args: table)
    _, _, metrics, maps = logic.compute_trabecular_microarchitecture(
        nodes["trab"], nodes["peri"], bone_segmentation_node=nodes["bone"],
        cortical_mask_node=nodes["cort"], common_region_node=nodes["region"],
        thickness_backend="cpu", create_maps=False,
    )
    assert metrics["Ct.Po.V"] == 1.0
    assert maps["Ct.Po.Mask"].sum() == 1
    assert not maps["Ct.Th"][:4].any()
    assert attributes["BoneImaging.Microarchitecture.measurement_method"] == METHOD_ID
