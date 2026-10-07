from __future__ import annotations

import importlib.util
import csv
import json
import numpy as np
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, read_manifest, write_manifest
from bone_microarchitecture.method import METHOD_METADATA


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "IOTools" / "BatchProcessor" / "BatchProcessor.py"


def _install_slicer_import_stubs(monkeypatch) -> None:
    qt = types.ModuleType("qt")
    ctk = types.ModuleType("ctk")
    slicer = types.ModuleType("slicer")
    vtk = types.ModuleType("vtk")
    scripted = types.ModuleType("slicer.ScriptedLoadableModule")

    class _Base:
        def __init__(self, *args, **kwargs):
            pass

    class _StringArray:
        def SetName(self, *_args, **_kwargs):
            pass

        def InsertNextValue(self, *_args, **_kwargs):
            pass

    scripted.ScriptedLoadableModule = _Base
    scripted.ScriptedLoadableModuleWidget = _Base
    scripted.ScriptedLoadableModuleLogic = _Base
    scripted.ScriptedLoadableModuleTest = _Base
    slicer.ScriptedLoadableModule = scripted
    slicer.util = types.SimpleNamespace()
    slicer.app = types.SimpleNamespace()
    vtk.vtkStringArray = _StringArray

    monkeypatch.setitem(sys.modules, "qt", qt)
    monkeypatch.setitem(sys.modules, "ctk", ctk)
    monkeypatch.setitem(sys.modules, "slicer", slicer)
    monkeypatch.setitem(sys.modules, "vtk", vtk)
    monkeypatch.setitem(sys.modules, "slicer.ScriptedLoadableModule", scripted)


def _import_batch_processor_module(monkeypatch):
    _install_slicer_import_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location("batch_processor_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mark_current_microarchitecture(path):
    Path(path).with_suffix(".json").write_text(json.dumps(METHOD_METADATA), encoding="utf-8")


def _write_complete_unet_case(logic, root, row):
    targets = logic._deep_learning_targets(root, row)
    targets["provenance"].parent.mkdir(parents=True, exist_ok=True)
    masks = {}
    for role in ("full", "trab", "cort"):
        targets[role].write_bytes(b"complete mask fixture")
        targets[f"{role}_sidecar"].write_text(json.dumps({
            "method": "unet", "short_role": role,
            "software": {"name": "bone-contouring", "version": "0.3.0"}}))
        masks[role] = targets[role].name
    targets["provenance"].write_text(json.dumps({
        "masks": masks, "method": "unet", "version": "0.3.0",
        "model": "radius_tibia_final", "device": "mps"}))


def test_export_uses_selected_voidspace_profile_not_stale_group_outputs(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    logic = module.BatchProcessorLogic()
    row = dict(subject="001", session_value="01", voi_value="radiusleft", registered=True,
               profile="registered", output_paths=["unrelated_measurements.csv"])
    current = logic._voidspace_output_dir_for_row(tmp_path, "registered", row)
    current.mkdir(parents=True)
    (current / "voidspace_measurements.csv").write_text("VS.V\n25\n")
    native_row = {**row, "registered": False, "profile": "standard"}
    native = logic._voidspace_output_dir_for_row(tmp_path, "standard", native_row)
    native.mkdir(parents=True)
    (native / "voidspace_measurements.csv").write_text("VS.V\n99\n")
    cases = logic.measurement_export_cases(tmp_path, tool="voidspace", profile="registered", rows=[row])
    from SlicerBoneImagingToolboxLib.measurement_export import export_measurements
    report = export_measurements(tmp_path / "export.csv", dataset_root=tmp_path,
                                 tool="voidspace", profile="registered", cases=cases)
    with report.path.open(encoding="utf-8-sig", newline="") as stream:
        values = list(csv.DictReader(stream))
    assert values[0]["Large.VS.V"] == "25"
    assert "unrelated" not in values[0]["source_files"]


def test_export_pair_metadata_and_missing_rows_are_preserved(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    row = dict(subject="001", session_value="01-02", voi_value="tibia", profile="dynamic")
    cases = module.BatchProcessorLogic().measurement_export_cases(
        tmp_path, tool="voidspace", profile="dynamic", rows=[row])
    assert cases[0]["baseline_session_value"] == "01"
    assert cases[0]["followup_session_value"] == "02"
    assert cases[0]["paths"] == []


def test_export_preserves_stack_identity_from_displayed_rows(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    rows = []
    for stack in (1, 2):
        row = dict(subject="001", session_value="01", voi_value="radiusleft",
                   voi=f"radiusleft stack-{stack:02d}", registered=True)
        rows.append(row)
        folder = tmp_path / "derivatives/Microarchitecture/sub-001/ses-01/xct/registered_measurements"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"sub-001_ses-01_voi-radiusleft_stack-{stack:02d}_measurements.csv").write_text(
            f"Parameter,Mean\nTb.Th,{stack}\n")
        _mark_current_microarchitecture(folder / f"sub-001_ses-01_voi-radiusleft_stack-{stack:02d}_measurements.csv")
    cases = module.BatchProcessorLogic().measurement_export_cases(
        tmp_path, tool="microarchitecture", profile="xtremectii-registered", rows=rows)
    assert [case["stack_index"] for case in cases] == [1, 2]
    from SlicerBoneImagingToolboxLib.measurement_export import export_measurements
    report = export_measurements(tmp_path / "export.csv", dataset_root=tmp_path,
                                 tool="microarchitecture", profile="xtremectii-registered", cases=cases)
    assert report.row_count == 2 and not report.issues


def test_export_mechanoregulation_summary_does_not_require_visualization_files(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    row = dict(subject="001", session_value="01-02", voi_value="radiusleft", profile="standard",
               mechanoregulation_case_id="sub-001_voi-radiusleft_t0-01_t1-02_test")
    base = tmp_path / "derivatives/Mechanoregulation/sub-001/xct/runs" / row["mechanoregulation_case_id"]
    base.mkdir(parents=True)
    path = base / f"{row['mechanoregulation_case_id']}_roi-full_mechanoregulation_summary.csv"
    path.write_text("OR_F\n1.2\n")
    cases = module.BatchProcessorLogic().measurement_export_cases(
        tmp_path, tool="mechanoregulation", profile="standard", rows=[row])
    assert cases[0]["paths"] == [str(path)]


def test_export_button_collects_wide_csv_without_running_analysis(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    logic = module.BatchProcessorLogic()
    row = dict(subject="001", session_value="01", voi_value="radiusleft", profile="standard")
    folder = logic._voidspace_output_dir_for_row(tmp_path, "standard", row)
    folder.mkdir(parents=True)
    (folder / "voidspace_measurements.csv").write_text("VS.V\n25\n")
    output = tmp_path / "chosen.csv"
    monkeypatch.setattr(module.qt, "QFileDialog", types.SimpleNamespace(
        getSaveFileName=lambda *args: str(output)), raising=False)
    messages = []
    monkeypatch.setattr(module.slicer.util, "mainWindow", lambda: None, raising=False)
    monkeypatch.setattr(module.slicer.util, "errorDisplay", messages.append, raising=False)
    monkeypatch.setattr(module.slicer.util, "infoDisplay", messages.append, raising=False)
    monkeypatch.setattr(module.slicer.util, "warningDisplay", messages.append, raising=False)
    widget = module.BatchProcessorWidget()
    widget.logic = logic
    widget._batchRows = [row]
    widget.profileCombo = types.SimpleNamespace(currentData="standard")
    widget.exportCsvButton = types.SimpleNamespace(enabled=True)
    widget._has_active_batch = lambda: False
    widget._selected_tool_key = lambda: "voidspace"
    widget._selected_backend_key = lambda: "local"
    widget._current_local_dataset_root = lambda: str(tmp_path)
    widget._append_log = messages.append
    widget._export_csv()
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["Large.VS.V"] == "25"
    assert any("1" in message and "Export" in message for message in messages)
    assert widget.exportCsvButton.enabled


@pytest.mark.parametrize("exit_code,source_destination", [(0, False), (1, False), (0, True)])
def test_remote_export_uses_fresh_csv_snapshot_and_never_stale_local_results(monkeypatch, tmp_path, exit_code, source_destination):
    module = _import_batch_processor_module(monkeypatch)
    from SlicerBoneImagingToolboxLib.remote_batch import RemoteBatchConfig, SshSlurmBatchBackend

    class Signal:
        def connect(self, callback):
            self.callback = callback

    class Process:
        MergedChannels = 1
        FailedToStart = 0
        def __init__(self):
            self.readyRead, self.finished, self.errorOccurred = Signal(), Signal(), Signal()
        def setProcessChannelMode(self, mode):
            pass
        def start(self, program, args):
            self.argv = [program, *args]
        def readAll(self):
            return b""
        def deleteLater(self):
            pass

    monkeypatch.setattr(module.qt, "QProcess", Process, raising=False)
    widget = module.BatchProcessorWidget()
    widget.logic = module.BatchProcessorLogic()
    widget.exportCsvButton = types.SimpleNamespace(enabled=True)
    widget._batchRows = [dict(subject="001", session_value="01", voi_value="radiusleft", profile="standard")]
    widget._selected_tool_key = lambda: "voidspace"
    messages = []
    widget._append_log = messages.append
    monkeypatch.setattr(module.slicer.util, "errorDisplay", messages.append, raising=False)
    monkeypatch.setattr(module.slicer.util, "infoDisplay", messages.append, raising=False)
    monkeypatch.setattr(module.slicer.util, "warningDisplay", messages.append, raising=False)
    backend = SshSlurmBatchBackend(RemoteBatchConfig(
        name="test", host="test", remote_root="/remote/data", local_root=str(tmp_path),
        python="python", work_dir="/remote/work"))
    widget._remote_backend = lambda **kwargs: backend
    widget._current_remote_dataset_root = lambda: "/remote/data"
    stale = widget.logic._voidspace_output_dir_for_row(tmp_path, "standard", widget._batchRows[0])
    stale.mkdir(parents=True)
    (stale / "voidspace_measurements.csv").write_text("VS.V\n99\n")
    context = dict(destination=str(tmp_path / "export.csv"), dataset_root=str(tmp_path),
                   tool="voidspace", profile="standard", rows=widget._batchRows)
    if source_destination:
        context["destination"] = str(stale / "voidspace_measurements.csv")
    widget._sync_measurements_for_export(context)
    process = widget._csvExportProcess
    target = Path(process.argv[-1])
    snapshot = target.parent.parent
    assert snapshot != tmp_path and "--include=*.csv" in process.argv
    fresh = widget.logic._voidspace_output_dir_for_row(snapshot, "standard", widget._batchRows[0])
    fresh.mkdir(parents=True, exist_ok=True)
    (fresh / "voidspace_measurements.csv").write_text("VS.V\n25\n")
    process.finished.callback(exit_code, 0)
    if source_destination:
        assert any("source" in message.lower() for message in messages)
    elif exit_code == 0:
        with (tmp_path / "export.csv").open(encoding="utf-8-sig") as stream:
            assert list(csv.DictReader(stream))[0]["Large.VS.V"] == "25"
    else:
        assert not (tmp_path / "export.csv").exists()
        assert any("failed" in message.lower() for message in messages)
    assert not snapshot.exists()
    assert widget._csvExportProcess is None and widget.exportCsvButton.enabled
    assert (stale / "voidspace_measurements.csv").read_text() == "VS.V\n99\n"


def test_unet_is_a_bone_contouring_profile_and_routes_to_published_worker(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    logic = module.BatchProcessorLogic()
    assert any(value == "unet" for _label, value, _registered in logic.profiles_for_tool("bone_contouring"))
    image = tmp_path / "sub-01_ses-01_voi-radiusleft_xct.AIM"
    image.write_bytes(b"aim fixture")
    row = dict(subject="01", session_value="01", voi_value="radiusleft", image_path=str(image), device="cpu")
    command = logic.command_for_row(tmp_path, tool="bone_contouring", profile="unet", row=row)
    assert command[:2] == ["-m", "bone_contouring.unet.cli"]
    assert command[command.index("--device") + 1] == "cpu"
    assert Path(command[command.index("--dataset-root") + 1]) == tmp_path.resolve()


def test_unet_batch_resumes_lh_when_compartments_complete(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    raw = tmp_path / "sub-001/ses-1/xct/sub-001_ses-1_voi-radiusleft_xct.AIM"
    raw.parent.mkdir(parents=True)
    raw.touch()
    logic = module.BatchProcessorLogic()
    row = logic.discover_rows(tmp_path, tool="bone_contouring", profile="unet", registered=False)[0][0]
    _write_complete_unet_case(logic, tmp_path, row)
    logic.publish_deep_learning_manifest(tmp_path, row)
    row = logic.discover_rows(tmp_path, tool="bone_contouring", profile="unet", registered=False)[0][0]
    assert row["action"] == "Run"
    assert "LH" in row["status"]
    args = logic.command_for_row(tmp_path, tool="bone_contouring", profile="unet", row=row)
    assert "--dataset-root" in args


def test_batch_unet_profile_selects_existing_pipeline_without_replacing_profile_list(monkeypatch):
    module = _import_batch_processor_module(monkeypatch)
    widget = types.SimpleNamespace(toolCombo=types.SimpleNamespace(currentData="bone_contouring"),
                                   profileCombo=types.SimpleNamespace(currentData="unet"))
    assert module.BatchProcessorWidget._selected_tool_key(widget) == "deep_learning_segmentation"
    widget.profileCombo.currentData = "my-custom-profile"
    assert module.BatchProcessorWidget._selected_tool_key(widget) == "bone_contouring"


def test_batch_custom_standard_profiles_remain_available_but_scene_cnn_recipes_are_not_misrun(monkeypatch):
    module = _import_batch_processor_module(monkeypatch)
    records = [types.SimpleNamespace(name="My LH", kind="json"),
               types.SimpleNamespace(name="My scene CNN", kind="json"),
               types.SimpleNamespace(name="My tissue-only scene", kind="json"),
               types.SimpleNamespace(name="Legacy tissue-only", kind="json"),
               types.SimpleNamespace(name="Broken recipe", kind="json"),
               types.SimpleNamespace(name="Other standard", kind="json")]
    monkeypatch.setattr(module, "list_profiles", lambda tool: records, raising=False)
    def payload(record):
        if record.name == "Broken recipe":
            raise ValueError("Invalid JSON")
        if record.name == "My LH":
            return {"schema": "bone-contour-recipe-v1",
                    "methods": {"periosteal_contour": "standard", "endosteal_contour": "standard",
                                "bone_segmentation": "laplace_hamming"}}
        if record.name == "Legacy tissue-only":
            return {"schema": "bone-contour-recipe-v1",
                    "methods": {"periosteal_contour": "none", "endosteal_contour": "none",
                                "bone_segmentation": "laplace_hamming"}}
        return {"contouring_method": {"My scene CNN": "unet", "My tissue-only scene": "none"}.get(record.name, "standard")}
    monkeypatch.setattr(module, "load_profile_payload", payload, raising=False)
    widget = types.SimpleNamespace(logic=module.BatchProcessorLogic())
    profiles = module.BatchProcessorWidget._profiles_for_tool(widget, "bone_contouring")
    values = [value for _label, value, _registered in profiles]
    assert "My LH" in values and "unet" in values
    assert "My scene CNN" not in values
    assert "My tissue-only scene" not in values and "Broken recipe" not in values
    assert "Legacy tissue-only" not in values
    assert "Other standard" in values


@pytest.mark.parametrize("publication_fails", [True, False])
def test_unet_manifest_failure_is_not_reported_done_or_load(monkeypatch, tmp_path, publication_fails):
    module = _import_batch_processor_module(monkeypatch)
    events = []
    def failed_publication(*args):
        if publication_fails:
            raise OSError("read-only manifest")
    process = types.SimpleNamespace(exitCode=lambda: 0)
    shell = types.SimpleNamespace(
        _append_process_output=lambda proc: None, _batchProcessOutput={},
        _batchProcess=process, _batchCancelled=False,
        profileCombo=types.SimpleNamespace(currentData="published"),
        _job_matches_current_row=lambda *args: True,
        logic=types.SimpleNamespace(publish_deep_learning_manifest=failed_publication),
        _append_log=lambda message: events.append(message),
        _set_row_status=lambda i, status: events.append(status),
        _set_row_action=lambda i, action: events.append(action),
        _refresh_row_output_paths=lambda *args, **kwargs: None,
        _set_group_action_load_if_outputs_are_ready=lambda *args: None,
        _start_next_batch_job=lambda: events.append("next"))
    module.BatchProcessorWidget._batch_process_finished(shell, 0, process,
        {"tool": "deep_learning_segmentation", "local_root": str(tmp_path)})
    assert "Done" not in events and "Load" not in events
    assert ("Publish" if publication_fails else "Missing") in events and events[-1] == "next"


def test_complete_marker_without_manifest_offers_lh_completion_not_load(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    raw = tmp_path / "sub-001/ses-1/xct/sub-001_ses-1_voi-radiusleft_xct.AIM"
    raw.parent.mkdir(parents=True)
    raw.touch()
    logic = module.BatchProcessorLogic()
    row = logic.discover_rows(tmp_path, tool="deep_learning_segmentation", profile="published", registered=False)[0][0]
    _write_complete_unet_case(logic, tmp_path, row)
    assert logic._deep_learning_output_paths(tmp_path, row) == []
    row = logic.discover_rows(tmp_path, tool="deep_learning_segmentation", profile="published", registered=False)[0][0]
    assert row["action"] == "Run"
    logic.publish_deep_learning_manifest(tmp_path, row)
    assert logic.discover_rows(tmp_path, tool="deep_learning_segmentation", profile="published", registered=False)[0][0]["action"] == "Run"


def test_publish_action_retries_manifest_only(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    row = dict(action="Publish")
    events = []
    shell = types.SimpleNamespace(_batchRows=[row],
        _effective_row_action=lambda row: row["action"],
        logic=types.SimpleNamespace(publish_deep_learning_manifest=lambda *args: events.append("publish")),
        _current_local_dataset_root=lambda: tmp_path,
        _refresh_row_output_paths=lambda *args, **kwargs: events.append("refresh") or ["mask.AIM"],
        _set_row_status=lambda i, value: events.append(value),
        _set_row_action=lambda i, value: events.append(value),
        _queue_row=lambda i: pytest.fail("Must not rerun inference"))
    module.BatchProcessorWidget._on_row_action(shell, 0)
    assert events == ["publish", "refresh", "Done", "Load"]


def test_publish_conflicting_outputs_never_become_loadable(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    row = dict(subject="001", session_value="1", voi_value="radiusleft",
               stack_index=None, image_path=str(tmp_path/"sub-001/ses-1/xct/scan.AIM"), action="Publish")
    logic = module.BatchProcessorLogic()
    _write_complete_unet_case(logic, tmp_path, row)
    full = logic._deep_learning_targets(tmp_path, row)["full"].with_suffix(".nrrd")
    full.write_bytes(b"preserved older cortex method")
    record = DerivativeRecord("BoneContours", "periosteal_mask", "001", "radiusleft", "1",
                              None, "native", full, "generated", content_type="mask")
    target = tmp_path/"derivatives/BoneContours/manifest.json"
    write_manifest(DerivativeManifest.create("BoneContours", tmp_path,
        {"name":"test", "version":"1"}, records=(record,)), target)
    events = []
    shell = types.SimpleNamespace(_batchRows=[row], logic=logic,
        _effective_row_action=lambda row: row["action"],
        _current_local_dataset_root=lambda: tmp_path,
        _refresh_row_output_paths=lambda *args, **kwargs: logic._deep_learning_output_paths(tmp_path, row),
        _set_row_status=lambda i, value: events.append(value),
        _set_row_action=lambda i, value: events.append(value),
        _append_log=lambda value: events.append(value))
    module.BatchProcessorWidget._on_row_action(shell, 0)
    assert "Load" not in events and "Done" not in events
    assert full.read_bytes() == b"preserved older cortex method"


def test_unet_blocks_partial_other_format_compartments(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    source = tmp_path/"sub-001/ses-1/xct/sub-001_ses-1_voi-radiusleft_xct.AIM"
    source.parent.mkdir(parents=True)
    source.touch()
    full = tmp_path/"derivatives/BoneContours/sub-001/ses-1/xct/sub-001_ses-1_voi-radiusleft_desc-full_mask.nrrd"
    full.parent.mkdir(parents=True)
    full.write_bytes(b"preserve")
    logic = module.BatchProcessorLogic()
    row = logic.discover_rows(tmp_path, tool="deep_learning_segmentation", profile="published", registered=False)[0][0]
    assert row["action"] == "Missing"
    with pytest.raises(FileExistsError):
        logic.command_for_row(tmp_path, tool="deep_learning_segmentation", profile="published", row=row)


def test_unet_batch_discovers_aim_only_and_uses_device_without_science_options(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    raw = tmp_path/"sub-001"/"ses-1"/"xct"
    raw.mkdir(parents=True)
    image = raw/"sub-001_ses-1_voi-radiusleft_xct.AIM"
    image.touch()
    (raw/"sub-001_ses-1_voi-radiusleft_xct.nii.gz").touch()
    logic = module.BatchProcessorLogic()
    rows, _ = logic.discover_rows(tmp_path, tool="deep_learning_segmentation", profile="published", registered=False)
    assert len(rows) == 1
    assert rows[0]["action"] == "Run"
    assert rows[0]["image_path"] == str(image)
    rows[0]["device"] = "cuda"
    args = logic.command_for_row(tmp_path, tool="deep_learning_segmentation", profile="published", row=rows[0])
    assert args[:3] == ["-m", "bone_contouring.unet.cli", str(image)]
    assert args[-2:] == ["--device", "cuda"]
    assert Path(args[args.index("--output")+1]) == tmp_path/"derivatives"/"BoneContours"/"sub-001"/"ses-1"/"xct"


def test_unet_completed_outputs_have_discoverable_bone_contours_manifest(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    raw = tmp_path/"sub-001"/"ses-1"/"xct"
    raw.mkdir(parents=True)
    image = raw/"sub-001_ses-1_voi-radiusleft_xct.AIM"
    image.touch()
    logic = module.BatchProcessorLogic()
    rows, _ = logic.discover_rows(tmp_path, tool="deep_learning_segmentation", profile="published", registered=False)
    row = rows[0]
    output = logic._deep_learning_output_directory(tmp_path, row)
    output.mkdir(parents=True)
    for role in ("full", "trab", "cort"):
        (output/f"sub-001_ses-1_voi-radiusleft_desc-{role}_mask.AIM").touch()
    assert logic._deep_learning_output_paths(tmp_path, row) == []
    _write_complete_unet_case(logic, tmp_path, row)
    logic.publish_deep_learning_manifest(tmp_path, row)
    manifest = json.loads((tmp_path/"derivatives"/"BoneContours"/"manifest.json").read_text())
    assert len(manifest["records"]) == 3
    assert all(not Path(record["path"]).is_absolute() for record in manifest["records"])
    assert {record["role"] for record in manifest["records"]} == {"periosteal_mask", "trabecular_mask", "cortical_mask"}
    from bone_contouring.batch import discover_bone_contouring_batch
    downstream = discover_bone_contouring_batch(tmp_path)
    assert {artifact.role for artifact in downstream[0].outputs} == {"full", "trab", "cort"}
    assert downstream[0].status == "ready"  # missing tissue SEG and material labels
    rows, _ = logic.discover_rows(tmp_path, tool="deep_learning_segmentation", profile="published", registered=False)
    assert rows[0]["action"] == "Run"  # reuse contours and add missing LH tissue SEG


def test_unet_partial_existing_contours_are_blocked_without_overwrite(monkeypatch, tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    raw = tmp_path/"sub-001"/"ses-1"/"xct"
    raw.mkdir(parents=True)
    (raw/"sub-001_ses-1_voi-radiusleft_xct.AIM").touch()
    output = tmp_path/"derivatives"/"BoneContours"/"sub-001"/"ses-1"/"xct"
    output.mkdir(parents=True)
    existing = output/"sub-001_ses-1_voi-radiusleft_desc-cort_mask.AIM"
    existing.write_bytes(b"keep existing cortex")
    logic = module.BatchProcessorLogic()
    rows,_ = logic.discover_rows(tmp_path, tool="deep_learning_segmentation", profile="published", registered=False)
    assert rows[0]["action"] == "Missing"
    with pytest.raises(FileExistsError):
        logic.command_for_row(tmp_path, tool="deep_learning_segmentation", profile="published", row=rows[0])
    assert existing.read_bytes() == b"keep existing cortex"


def test_unet_remote_ui_points_to_ssh_cli_without_submitting(monkeypatch):
    module = _import_batch_processor_module(monkeypatch)
    shell = types.SimpleNamespace()
    rows,message = module.BatchProcessorWidget._discover_remote_rows(shell,"/local","/remote","deep_learning_segmentation","published",False)
    assert rows == []
    assert "SSH" in message


def test_unet_complete_other_method_contours_are_loadable_and_preserved(monkeypatch,tmp_path):
    module = _import_batch_processor_module(monkeypatch)
    raw = tmp_path/"sub-001"/"ses-1"/"xct"
    raw.mkdir(parents=True)
    (raw/"sub-001_ses-1_voi-radiusleft_xct.AIM").touch()
    output = tmp_path/"derivatives"/"BoneContours"/"sub-001"/"ses-1"/"xct"
    output.mkdir(parents=True)
    records=[]
    for short,role in (("full","periosteal_mask"),("trab","trabecular_mask"),("cort","cortical_mask")):
        path=output/f"sub-001_ses-1_voi-radiusleft_desc-{short}_mask.AIM"
        path.write_bytes(b"other method contour")
        records.append(DerivativeRecord("BoneContours",role,"001","radiusleft","1",None,"native",path,"generated",content_type="mask",software={"name":"bone-contouring","version":"0.2.3"}))
    write_manifest(DerivativeManifest.create("BoneContours",tmp_path,{"name":"test","version":"1"},records=records),tmp_path/"derivatives"/"BoneContours"/"manifest.json")
    logic=module.BatchProcessorLogic()
    rows,_=logic.discover_rows(tmp_path,tool="deep_learning_segmentation",profile="published",registered=False)
    assert rows[0]["action"] == "Load"
    with pytest.raises(FileExistsError):
        logic.command_for_row(tmp_path,tool="deep_learning_segmentation",profile="published",row=rows[0])


@pytest.mark.parametrize("unet", [False, True])
def test_unet_load_row_uses_correct_importer_without_misattribution(monkeypatch,tmp_path,unet):
    module=_import_batch_processor_module(monkeypatch)
    row=dict(subject="001",session_value="1",voi_value="radiusleft",stack_index=None,
             output_paths=[str(tmp_path/"mask.AIM")])
    logic=module.BatchProcessorLogic()
    marker=logic._deep_learning_targets(tmp_path,row)["provenance"]
    if unet:
        marker.parent.mkdir(parents=True)
        marker.write_text("{}")
    loaded=[]
    shell=types.SimpleNamespace(_batchRows=[row],logic=logic,
        _selected_tool_key=lambda:"deep_learning_segmentation",_selected_backend_key=lambda:"local",
        _current_local_dataset_root=lambda:tmp_path,_deduplicated_paths=lambda paths:paths,
        _load_deep_learning_outputs=lambda row,paths:loaded.append("unet"),
        _load_bone_contour_outputs_as_segmentation=lambda row,paths:loaded.append("existing"))
    module.BatchProcessorWidget._load_row_outputs(shell,0)
    assert loaded == (["unet"] if unet else ["existing"])


def test_unet_discovery_preserves_stack_identity(monkeypatch,tmp_path):
    module=_import_batch_processor_module(monkeypatch)
    raw=tmp_path/"sub-001"/"ses-1"/"xct"
    raw.mkdir(parents=True)
    for stack in (1,2):
        (raw/f"sub-001_ses-1_voi-radiusleft_stack-{stack:02d}_xct.AIM").touch()
    rows,_=module.BatchProcessorLogic().discover_rows(tmp_path,tool="deep_learning_segmentation",profile="published",registered=False)
    assert {row["stack_index"] for row in rows} == {1,2}


def test_unet_virtual_stack_is_not_mistaken_for_full_scan(monkeypatch,tmp_path):
    module=_import_batch_processor_module(monkeypatch)
    raw=tmp_path/"sub-001"/"ses-1"/"xct"
    raw.mkdir(parents=True)
    image=raw/"sub-001_ses-1_voi-radiusleft_xct.AIM"
    image.touch()
    image.with_suffix(".AIM.json").write_text(json.dumps({"virtual_stacks":[{"stack_index":1,"slice_start":0,"slice_stop":3}]}))
    logic=module.BatchProcessorLogic()
    rows,_=logic.discover_rows(tmp_path,tool="deep_learning_segmentation",profile="published",registered=False)
    assert rows[0]["action"] == "Missing"
    with pytest.raises(ValueError):
        logic.command_for_row(tmp_path,tool="deep_learning_segmentation",profile="published",row=rows[0])


@pytest.mark.parametrize("segmentation", ["gauss", "laplace_hamming"])
def test_native_unet_handoff_only_adds_missing_seg_and_material(monkeypatch,tmp_path,segmentation):
    """Catches contour recomputation/overwrite and loss of U-Net provenance."""
    py_aimio=pytest.importorskip("py_aimio")
    pytest.importorskip("bone_contouring.unet")
    from bone_contouring.unet.aim import write_masks
    from bone_contouring import batch as contour_batch
    module=_import_batch_processor_module(monkeypatch)
    raw=tmp_path/"sub-001"/"ses-1"/"xct"
    raw.mkdir(parents=True)
    source=raw/"sub-001_ses-1_voi-radiusleft_xct.AIM"
    full=np.zeros((5,30,32),dtype=np.uint8)
    full[:,4:26,4:28]=1
    trab=np.zeros_like(full)
    trab[:,7:23,7:25]=1
    cort=full-trab
    metadata=dict(position=(0,0,0),offset=(0,0,0),element_size=(.061,)*3,
                  processing_log="Mu_Scaling 8192\nHU: mu water 0.24\nDensity: slope 1500\nDensity: intercept -100\n")
    py_aimio.write_aim(str(source),full.astype(np.int16)*8000,metadata,unit="native")
    logic=module.BatchProcessorLogic()
    row=logic.discover_rows(tmp_path,tool="deep_learning_segmentation",profile="published",registered=False)[0][0]
    output=logic._deep_learning_output_directory(tmp_path,row)
    paths=write_masks(output,source.stem,dict(full=full,trab=trab,cort=cort),metadata,source=source,device="cpu")
    original={path:path.read_bytes() for path in paths.values()}
    logic.publish_deep_learning_manifest(tmp_path,row)
    def no_recontouring(*args,**kwargs):
        raise AssertionError("Existing U-Net compartments must be reused")
    monkeypatch.setattr(contour_batch,"generate_masks_from_image",no_recontouring)
    recipe={"profile":"XtremeCTII-LH", "site":"radius"} if segmentation == "laplace_hamming" else {
        "modality":"xct2", "site":"radius", "segmentation":segmentation}
    generated=contour_batch.run_bone_contouring_batch(tmp_path,**recipe)
    assert {record.role for record in generated} == {"bone_segmentation","material_labelmap"}
    assert all(path.read_bytes() == before for path,before in original.items())
    manifest=read_manifest(tmp_path/"derivatives"/"BoneContours"/"manifest.json")
    assert len(manifest.records)==5
    assert sum(record.metadata.get("method") == "unet" for record in manifest.records)==3
    assert contour_batch.discover_bone_contouring_batch(tmp_path)[0].status=="loadable"
    combined=logic.discover_rows(tmp_path,tool="bone_contouring",profile="unet",registered=False)[0][0]
    if segmentation == "laplace_hamming":
        assert combined["action"] == "Load"
        assert len(logic._deep_learning_output_paths(tmp_path,combined)) == 5
    else:
        assert combined["action"] == "Missing"
        assert "SEG conflicts" in combined["status"]
        assert logic._deep_learning_output_paths(tmp_path,combined) == []
        with pytest.raises(ValueError):
            logic.command_for_row(tmp_path,tool="bone_contouring",profile="unet",row=combined)


def test_source_lookup_with_duplicate_names_requires_exact_path(monkeypatch,tmp_path):
    module=_import_batch_processor_module(monkeypatch)
    first,second=tmp_path/"one"/"scan.AIM",tmp_path/"two"/"scan.AIM"
    def node(path):
        return types.SimpleNamespace(GetName=lambda:"scan",GetAttribute=lambda key:str(path),GetStorageNode=lambda:None)
    one,two=node(first),node(second)
    module.slicer.util.getNodesByClass=lambda name:[one,two]
    assert module.BatchProcessorWidget._find_loaded_source_volume(second,require_source_path=True) is two


def test_batch_processor_module_is_registered_in_toolbox() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "toolbox_modules.json").read_text(encoding="utf-8"))

    assert "add_subdirectory(IOTools/BatchProcessor)" in cmake
    assert any(module["path"] == "IOTools/BatchProcessor" for module in manifest["modules"])
    entry = next(module for module in manifest["modules"] if module["path"] == "IOTools/BatchProcessor")
    assert entry["title"] == "Batch Processor"
    assert entry["section"] == "I/O"


def test_batch_processor_module_uses_shared_discovery_and_batch_contract() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "class BatchProcessor(ScriptedLoadableModule):" in source
    assert 'parent.title = "Batch Processor"' in source
    assert 'parent.categories = ["Bone Imaging.I/O"]' in source
    assert "parent.contributors = [\"Matthias Walle\"]" in source
    assert "from bone_imaging_derivatives import (" in source
    assert "discover_raw_xct_images" in source
    assert "discover_derivative_artifacts" in source
    assert "discover_manifests" in source


def test_dataset_naming_helper_reloads_derivative_wrapper_for_slicer_reload_cache() -> None:
    source = (ROOT / "IOTools" / "DatasetNamingHelper" / "DatasetNamingHelper.py").read_text(encoding="utf-8")

    assert "import SlicerBoneImagingToolboxLib.derivatives as _derivatives_api" in source
    assert "importlib.reload(_derivatives_api)" in source
    assert "split_identity_metadata = _derivatives_api.split_identity_metadata" in source


def test_batch_processor_remote_ui_hides_private_paths_but_keeps_resource_overrides() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "SLICER_BONE_BATCH_BACKEND" in source
    assert "_serverBackendEnabled" in source
    assert "previous = self.backendCombo.blockSignals(True)" in source
    assert 'if not hasattr(self, "statusLabel") or not hasattr(self, "table"):' in source
    assert "show_backend_selector = len(self._batchBackends) > 1 or self._serverBackendEnabled" in source
    assert "self.backendLabel.visible = show_backend_selector" in source
    assert "self.backendCombo.visible = show_backend_selector" in source
    assert "self.serverRootEdit" not in source
    assert '"Server directory"' not in source
    assert '"Local processing directory"' not in source
    assert "self.serverTimeEdit" in source
    assert '"Wall time"' in source
    assert "self.serverMemoryEdit" in source
    assert '"Memory"' in source
    assert "self.serverCpusSpin" in source
    assert '"CPUs"' in source
    assert "f\"--cpus-per-task={values['cpus']}\"" in source
    assert "f\"--ntasks={values['cpus']}\" if mpi else f\"--cpus-per-task={values['cpus']}\"" in source
    assert 'mpi=backend_key != "local" and str(job.get("tool") or "") == "fea"' in source
    assert '"OMP_NUM_THREADS": thread_count' in source
    assert '"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": thread_count' in source
    assert "Remote discovery returned invalid JSON" in source


def test_batch_processor_exposes_runtime_backend_refresh_hook() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _refresh_batch_backends(self):" in source
    assert "self.backendCombo.clear()" in source
    assert "self.backendCombo.addItem(backend.label, backend.key)" in source
    assert "self._apply_backend_visibility()" in source


def test_batch_processor_remote_jobs_are_submitted_and_loaded_lazily() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "self._remoteJobs" in source
    assert "self._remotePollTimer" in source
    assert "parse_job_id" in source
    assert "Submitted" in source
    assert "def _poll_remote_jobs(self):" in source
    assert "def _remote_job_terminal_state" in source
    assert "if self._selected_backend_key() == \"server\":" in source
    assert "self._sync_remote_outputs_for_tool(" in source
    assert 'row.pop("output_paths", None)' in source
    assert "self._refresh_row_output_paths(row_index)" in source
    assert "syncing remote outputs" in source


def test_batch_processor_checks_remote_backend_availability_before_server_work() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _remote_backend_available(self, backend):" in source
    assert "backend.availability_argv()" in source
    assert "backend.shell_argv(command)" in source
    assert "timeout=backend.config.connect_timeout_seconds + 2" in source
    assert "Remote server is not reachable" in source
    assert "if not self._remote_backend_available(remote_backend):" in source


def test_batch_processor_matches_running_jobs_by_stable_row_identity() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _row_identity(self, row):" in source
    assert "return self._row_identity(self._batchRows[row_index]) == self._row_identity(job.get(\"row\") or {})" in source
    assert "dict(self._batchRows[row_index]) == dict(job.get(\"row\") or {})" not in source


def test_remote_fea_completion_surfaces_solver_log_tail() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _append_remote_job_log_tail(self, remote_job, backend):" in source
    assert "self._append_remote_job_log_tail(remote_job, backend)" in source
    assert 'if str(remote_job.get("tool") or "") == "fea":' in source


def test_fea_discovery_keeps_rows_with_missing_material_labelmap() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "source = case.first_artifact(preferred_roles)" in source
    assert "if source is None:\n                ok = False" in source
    assert "source=missing FEA input" in source


def test_batch_processor_remote_mechanoregulation_uses_core_case_discovery() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'if tool == "mechanoregulation":\n            return self._discover_remote_mechanoregulation_rows(backend, profile)' in source
    assert "def _remote_mechanoregulation_discovery_command" in source
    assert "from bonemechreg.timelapse import available_case_rois, case_outputs, discover_timelapse_cases" in source
    assert '"mechanoregulation_case_id": str(case.case_id)' in source


def test_batch_processor_remote_timelapse_discovers_existing_outputs() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'if tool == "timelapse":\n            derivative_records.extend(self._discover_remote_timelapse_outputs(backend))' in source
    assert "def _remote_timelapse_outputs_command" in source
    assert "pairwise_remodelling_table" in source
    assert "remodelling_image" in source


def test_batch_processor_remote_fea_completion_publishes_canonical_sed() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert '"row": dict(job.get("row") or {})' in source
    assert 'if str(remote_job.get("tool") or "") == "fea":\n                self._publish_remote_fea_outputs(remote_job, backend)' in source
    assert "def _finish_remote_job(self, row_key, remote_job, backend, status, state_text):" in source
    assert "def _publish_remote_fea_outputs(self, remote_job, backend):" in source
    assert "base / 'maps'" in source
    assert "'_map-sed.nii.gz'" in source


def test_batch_processor_remote_fea_rows_use_remote_artifact_memory() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "fea_outputs = self._remote_fea_outputs_by_key(derivative_records, profile)" in source
    assert "def _remote_fea_outputs_by_key(self, derivative_records, profile):" in source
    assert 'row["action"] = "Load"' in source
    assert 'artifact.role not in {"sed_map", "summary_table"}' in source
    assert "preferred_contours" in source
    assert "prerequisite_status" in source
    assert "def normalized_dataset_status(" in source
    assert 'root.glob("sub-*")' in source
    assert 'xct_dir = session_dir / "xct"' in source
    assert "Dataset Naming Helper" in source
    assert "Execution backend" in source
    assert '("Bone Contouring", "bone_contouring")' in source
    assert '"bone_contouring": "BoneContours"' in source
    assert "Server" in source
    assert "LocalBatchBackend" in (ROOT / "SlicerBoneImagingToolboxLib" / "batch_backends.py").read_text(encoding="utf-8")
    assert "Server backends are configured in private adapters" in source
    assert 'self.skipExistingCheck.checked = True' in source
    assert 'self.table.setHorizontalHeaderLabels(headers)' in source
    assert 'headers = ["Action", "Subject", "Session", "VOI", "Status", "Input"]' in source
    assert "Registered Microarchitecture" not in source
    assert "registered_microarchitecture" not in source
    assert "self.toolCombo.currentIndexChanged.connect(self._on_tool_changed)" in source
    assert "def _profiles_for_tool(self, tool):" in source
    assert "def profile_requests_registration(tool: str, profile: str) -> bool:" in source
    assert "registerCheck" not in source
    assert 'action = str(row.get("action") or "")' in source
    assert 'button = qt.QPushButton(action)' in source
    assert 'button.enabled = action in {"Run", "Load", "Publish"}' in source
    assert "self.table.setSpan(row_index, 0, span, 1)" in source
    assert "def _table_rows_for_tool(" in source
    assert "def command_for_row(" in source
    assert "def _subprocess_args(self, args):" in source
    assert '"timelapsedhrpqct.cli": _local_repo_path("Timelapsed" + "HRpQCT", "src")' in source
    assert "from {module} import main" in source
    assert 'process_args = self._subprocess_args(args)' in source
    assert "process.start(process_program, process_args)" in source
    assert '"bone_contouring": ("bone_contouring.cli", "run-batch")' in source
    assert '"microarchitecture": ("bone_microarchitecture.cli", "run-batch")' in source
    assert '"plate_rod": ("plate_rod_thinning.cli", "run-batch")' in source
    assert '"timelapse": ("timelapsedhrpqct.cli", "run")' in source
    assert "discover_raw_xct_images(root)" in source
    assert 'discover_derivative_artifacts(root, "IPLContours")' in source
    assert 'discover_derivative_artifacts(root, "ImportedContours")' in source
    assert 'discover_derivative_artifacts(root, "BoneContours")' in source


def test_batch_processor_module_reports_unsupported_one_row_commands() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "does not expose a one-row batch processor command yet" in source
    assert "if tool_key not in self._CLI_COMMANDS:" in source


def test_batch_processor_exposes_mask_and_label_algebra_tool() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert '("Mask And Label Algebra", "mask_label_algebra")' in source
    assert '"mask_label_algebra": ("bone_contouring.cli", "mask-label-algebra")' in source
    assert '"mask_label_algebra": "BoneContours"' in source


def test_mask_label_algebra_table_ignores_bone_contours_as_inputs(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    (xct_dir / "sub-001_ses-001_voi-radiusleft_xct.AIM").write_bytes(b"")
    contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
    contour_dir.mkdir(parents=True)
    for role in ("full", "cort", "seg"):
        (contour_dir / f"sub-001_ses-001_voi-radiusleft_desc-{role}_mask.AIM").write_bytes(b"")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="mask_label_algebra",
        profile="standard",
        registered=False,
    )

    assert rows[0]["action"] == "Missing"
    assert "full=" not in rows[0]["input"]
    assert "cort=" not in rows[0]["input"]
    assert "seg=" not in rows[0]["input"]


def test_batch_processor_passes_named_bone_contouring_profiles_and_colors_segments() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert '("XtremeCT I", "XtremeCTI", False)' in source
    assert '("XtremeCT II", "XtremeCTII", False)' in source
    assert '("XtremeCT II - Geodesic", "XtremeCTII-Geodesic", False)' in source
    assert '("XtremeCT II - LH", "XtremeCTII-LH", False)' in source
    assert 'args.extend(["--profile", profile_value])' in source
    assert '_SEGMENT_COLORS = {' in source
    assert '"full": (0.2, 0.8, 0.25)' in source
    assert '"trab": (0.0, 0.75, 1.0)' in source
    assert '"cort": (1.0, 0.55, 0.1)' in source
    assert '"seg": (0.45, 0.45, 0.45)' in source
    assert '"baseline_seg": (0.40, 0.40, 0.40)' in source
    assert '"followup_seg": (0.58, 0.58, 0.58)' in source
    assert "segment.SetColor(color[0], color[1], color[2])" in source


def test_batch_processor_fea_profiles_are_labelmap_shortcuts() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert '("XtremeCT I", "XtremeCTI", False)' in source
    assert '("XtremeCT II", "XtremeCTII", False)' in source
    assert '("Load history 3", "load_history_3", False)' in source
    assert '("Load history 6", "load_history_6", False)' in source
    assert "discover_fea_batch_cases" in source
    assert "build_parosol_case_commands" in source
    assert '"parosol_py.cli": _local_repo_path("parosol-py", "src")' in source


def test_batch_processor_fea_and_mechanoregulation_use_low_blue_high_red_sed_colormap() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "BatchProcessor_FEA_SED_JET" in source
    assert "display_node.SetAndObserveColorNodeID(color_node.GetID())" in source
    assert "vtkMRMLColorTableNodeRainbow" not in source
    assert "vtkMRMLColorTableNodeFileColdToHotRainbow.txt" not in source
    assert "abs(4.0 * t - 3.0)" in source
    assert "abs(4.0 * t - 1.0)" in source
    assert 'if role == "sed":\n                    self._style_fea_volume(node, path, force=True)' in source


def test_batch_mechanoregulation_forces_sed_colormap_for_aligned_display_path(monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)

    class DisplayNode:
        color_node_id = None

        def SetAndObserveColorNodeID(self, value):
            self.color_node_id = value

        def AutoWindowLevelOn(self):
            pass

        def AutoThresholdOff(self):
            pass

    class VolumeNode:
        def __init__(self):
            self.display = DisplayNode()

        def CreateDefaultDisplayNodes(self):
            pass

        def GetDisplayNode(self):
            return self.display

    node = VolumeNode()
    color_node = types.SimpleNamespace(GetID=lambda: "sed-color")
    monkeypatch.setattr(module.BatchProcessorWidget, "_fea_sed_color_node", staticmethod(lambda: color_node))

    module.BatchProcessorWidget._style_fea_volume(
        node,
        Path("surface-events_sed_on_events_grid.nii.gz"),
        force=True,
    )

    assert node.display.color_node_id == "sed-color"


def test_batch_mechanoregulation_display_preserves_index_aligned_sed_with_spacing_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sitk = pytest.importorskip("SimpleITK")
    module = _import_batch_processor_module(monkeypatch)
    sed_path = tmp_path / "sed.nii.gz"
    events_path = tmp_path / "surface-events.nii.gz"
    sed = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    sed.SetSpacing((0.0606999993, 0.0606999993, 0.0606999993))
    sed.SetOrigin((-566.0, -465.0, 0.0))
    sed.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sed[1, 1, 1] = 7.0
    events = sitk.Image(4, 4, 4, sitk.sitkUInt8)
    events.SetSpacing((0.0606996529, 0.0606996529, 0.0606964305))
    events.SetOrigin((46.412, 38.130, 0.0))
    sitk.WriteImage(sed, str(sed_path))
    sitk.WriteImage(events, str(events_path))
    widget = object.__new__(module.BatchProcessorWidget)
    widget._append_log = lambda _message: None

    aligned_path = widget._mechanoregulation_aligned_sed_display_path(
        {"sed_path": str(sed_path)},
        events_path,
    )

    aligned = sitk.ReadImage(str(aligned_path))
    assert np.allclose(aligned.GetOrigin(), events.GetOrigin())
    assert float(sitk.GetArrayFromImage(aligned).sum()) == 7.0


def test_batch_processor_mechanoregulation_loads_events_as_sed_linked_segmentation() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _mechanoregulation_surface_events_path(" in source
    assert "surface-events" in source
    assert "def _mechanoregulation_aligned_sed_display_path(" in source
    assert "mechanoregulation_display" in source
    assert "def _load_mechanoregulation_remodelling_as_segmentation(" in source
    assert "def _load_binary_event_labelmap(" in source
    assert '("resorption", "Resorption", np.isin(remodelling_array, (1,)))' in source
    assert '("formation", "Formation", np.isin(remodelling_array, (3, 4)))' in source
    assert '"quiescent"' not in source[source.index("def _load_mechanoregulation_remodelling_as_segmentation("):source.index("def _load_timelapse_outputs(")]
    assert 'segmentation_node.SetAttribute("BoneImaging.Mechanoregulation.RemodellingSource", str(path))' in source
    assert 'segmentation_node.SetAttribute("BoneImaging.Mechanoregulation.SEDNodeID", sed_node.GetID())' in source
    assert "slicer.util.setSliceViewerLayers(background=sed_node, fit=False)" in source


def test_batch_processor_fea_rows_use_fea_input_material_sources(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    raw = xct_dir / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    fea_input = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-fea-input_label.AIM"
    raw.write_bytes(b"")
    fea_input.write_bytes(b"")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="fea",
        profile="XtremeCTII",
        registered=False,
    )

    assert len(rows) == 1
    assert rows[0]["action"] == "Run"
    assert rows[0]["status"] == "Ready"
    assert rows[0]["image_path"] == str(fea_input)
    assert rows[0]["input"] == f"source={fea_input.name}"
    assert raw.name not in rows[0]["input"]

    command = module.BatchProcessorLogic().command_for_row(
        tmp_path,
        tool="fea",
        profile="XtremeCTII",
        row=rows[0],
        force=False,
    )

    assert command[:3] == ["-m", "parosol_py.cli", str(fea_input)]
    assert command[3:5] == ["--profile", "XtremeCTII"]
    assert "--session" not in command
    assert "--site" in command


def test_batch_processor_remote_fea_rows_choose_material_labelmap_over_raw_image(monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    logic = module.BatchProcessorLogic()
    key = module.CaseKey("001", "001", "radiusleft")
    raw = module.BatchArtifact(
        Path("/home/mwalle/data/sub-001/ses-001/xct/sub-001_ses-001_voi-radiusleft_xct.AIM"),
        key,
        "image",
        None,
        "remote",
        {},
    )
    material = module.BatchArtifact(
        Path("/home/mwalle/data/derivatives/BoneContours/sub-001/ses-001/xct/sub-001_ses-001_voi-radiusleft_desc-fea-input_label.AIM"),
        key,
        "material_labelmap",
        "BoneContours",
        "remote",
        {},
    )

    rows, _message = logic._fea_rows_from_cases(
        Path("/home/mwalle/data"),
        "XtremeCTI",
        logic._fea_cases_from_batch_artifacts([raw, material]),
    )

    assert len(rows) == 1
    assert rows[0]["image_path"] == str(material.path)
    assert rows[0]["input"] == f"source={material.path.name}"
    assert raw.path.name not in rows[0]["input"]


def test_batch_processor_fea_publishes_canonical_sed_and_summary(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    fea_input = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-fea-input_label.AIM"
    fea_input.write_bytes(b"")
    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="fea",
        profile="XtremeCTII",
        registered=False,
    )
    row = rows[0]
    run_dir = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "xct"
        / "runs"
        / "sub-001_ses-001_site-radiusleft_XtremeCTII"
    )
    sed = run_dir / "fields" / "sed.nii.gz"
    sed.parent.mkdir(parents=True)
    sed.write_bytes(b"sed")
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "mechanics": {
                    "generalized_stiffness": {"value": 1234.5, "units": "N/mm"},
                    "stiffness": {"z": 1234.5},
                },
                "failure": {
                    "failure_generalized_load": {"value": -678.9, "units": "N"},
                    "failure_load": {"z": -678.9},
                },
            }
        ),
        encoding="utf-8",
    )

    published = module.BatchProcessorLogic.publish_fea_batch_outputs(tmp_path, row, "XtremeCTII")

    map_path = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "maps"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTII_map-sed.nii.gz"
    )
    table_path = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "measurements"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTII_fea.csv"
    )
    assert published == [map_path, table_path]
    assert map_path.read_bytes() == b"sed"
    assert "Stiffness" in table_path.read_text(encoding="utf-8")
    assert "1234.5" in table_path.read_text(encoding="utf-8")
    assert "-678.9" in table_path.read_text(encoding="utf-8")

    manifest = read_manifest(tmp_path / "derivatives" / "FEA" / "manifest.json")
    roles = {(record.role, record.path.name) for record in manifest.records}
    assert ("sed_map", map_path.name) in roles
    assert ("summary_table", table_path.name) in roles


def test_batch_processor_fea_rows_load_from_canonical_derivatives(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    fea_input = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-fea-input_label.AIM"
    fea_input.write_bytes(b"")
    map_path = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "maps"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTII_map-sed.nii.gz"
    )
    table_path = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "measurements"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTII_fea.csv"
    )
    map_path.parent.mkdir(parents=True)
    table_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"sed")
    table_path.write_text("Sample,Profile,Stiffness,Failure load\nsub-001 ses-001 voi-radiusleft,XtremeCTII,1,2\n")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="fea",
        profile="XtremeCTII",
        registered=False,
    )

    assert rows[0]["action"] == "Load"
    assert rows[0]["status"] == "Done"
    assert rows[0]["output_paths"] == [str(map_path), str(table_path)]


def test_batch_processor_fea_outputs_are_scoped_by_profile(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    fea_input = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-fea-input_label.AIM"
    fea_input.write_bytes(b"")
    xtreme_table = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "measurements"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTII_fea.csv"
    )
    xtreme_table.parent.mkdir(parents=True)
    xtreme_table.write_text("Sample,Profile,Stiffness,Failure load\nsub-001 ses-001 voi-radiusleft,XtremeCTII,1,2\n")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="fea",
        profile="load_history_3",
        registered=False,
    )

    assert rows[0]["action"] == "Run"
    assert "output_paths" not in rows[0]


def test_batch_processor_fea_summary_includes_load_history_scale_factors(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    fea_input = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-fea-input_label.AIM"
    fea_input.write_bytes(b"")
    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="fea",
        profile="load_history_3",
        registered=False,
    )
    row = rows[0]
    run_dir = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "xct"
        / "runs"
        / "sub-001_ses-001_site-radiusleft_load_history_3"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "postprocess": {
                    "load_history": {
                        "details": {
                            "scaling_factors": [0.1, 0.2, 0.3],
                            "input_load_amplitudes": [10.0, 20.0, 30.0],
                        },
                        "results": {"estimated_loads": [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    published = module.BatchProcessorLogic.publish_fea_batch_outputs(tmp_path, row, "load_history_3")
    table_path = published[-1]
    text = table_path.read_text(encoding="utf-8")

    assert "Scale factors" in text
    assert "Input load amplitudes" in text
    assert "0.1;0.2;0.3" in text
    assert "10.0;20.0;30.0" in text


def test_batch_processor_regular_fea_summary_hides_load_history_columns(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    case = types.SimpleNamespace(
        subject_id="001",
        session_id="001",
        site="radiusleft",
        first_artifact=lambda roles: types.SimpleNamespace(path=tmp_path / "fea_input.AIM"),
    )
    table_path = tmp_path / "summary.csv"

    module.BatchProcessorLogic._write_fea_summary_csv(
        table_path,
        case,
        "XtremeCTII",
        {
            "mechanics": {"generalized_stiffness": {"value": 12.0}},
            "failure": {"failure_generalized_load": {"value": -3.0}},
            "postprocess": {
                "load_history": {
                    "details": {"scaling_factors": [1, 2, 3]},
                    "results": {"estimated_loads": [{"value": 4}]},
                }
            },
        },
    )

    header = table_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "Sample,Profile,Stiffness (N/mm),Failure load (N)"


def test_batch_processor_mechanoregulation_discovers_remodelling_rows_with_matching_sed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _import_batch_processor_module(monkeypatch)
    remodelling = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "xct"
        / "analysis"
        / "visualize"
        / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-5_remodelling.nii.gz"
    )
    baseline = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "transformed"
        / "sub-001_ses-001_voi-radiusleft_image-fused.nii.gz"
    )
    sed = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "maps"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTII_map-sed.nii.gz"
    )
    for path in (remodelling, baseline, sed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    rows, message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="mechanoregulation",
        profile="XtremeCTII",
        registered=False,
    )

    assert message == "Discovered 1 mechanoregulation row(s)."
    assert rows[0]["action"] == "Run"
    assert rows[0]["status"] == "Ready"
    assert rows[0]["subject"] == "001"
    assert rows[0]["session"] == "001-002"
    assert rows[0]["voi"] == "radiusleft"
    assert rows[0]["image_path"] == str(remodelling)
    assert rows[0]["input"] == f"remodelling={remodelling.name}\nsed={sed.name}\nfull=whole remodelling grid"

    command = module.BatchProcessorLogic().command_for_row(
        tmp_path,
        tool="mechanoregulation",
        profile="XtremeCTII",
        row=rows[0],
        force=True,
    )

    assert command == [
        "-m",
        "bonemechreg.cli",
        "run",
        str(tmp_path.resolve()),
        "--profile",
        "XtremeCTII",
        "--case-id",
        rows[0]["mechanoregulation_case_id"],
        "--verbose",
        "--reanalyze",
    ]


def test_batch_processor_mechanoregulation_reports_missing_profile_sed(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    remodelling = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "xct"
        / "analysis"
        / "visualize"
        / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-5_remodelling.nii.gz"
    )
    baseline = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "transformed"
        / "sub-001_ses-001_voi-radiusleft_image-fused.nii.gz"
    )
    for path in (remodelling, baseline):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="mechanoregulation",
        profile="XtremeCTII",
        registered=False,
    )

    assert rows[0]["action"] == "Missing"
    assert rows[0]["status"] == "Missing SED"
    assert rows[0]["input"] == (
        f"remodelling={remodelling.name}\nsed=missing XtremeCTII baseline SED\nfull=whole remodelling grid"
    )


def test_batch_processor_mechanoregulation_completed_rows_are_loadable(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    remodelling = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "xct"
        / "analysis"
        / "visualize"
        / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-5_remodelling.nii.gz"
    )
    baseline = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "transformed"
        / "sub-001_ses-001_voi-radiusleft_image-fused.nii.gz"
    )
    sed = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "maps"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTII_map-sed.nii.gz"
    )
    for path in (remodelling, baseline, sed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="mechanoregulation",
        profile="XtremeCTII",
        registered=False,
    )
    case_id = rows[0]["mechanoregulation_case_id"]
    out_dir = tmp_path / "derivatives" / "Mechanoregulation" / "sub-001" / "xct" / "runs" / case_id
    csv_path = out_dir / f"{case_id}_roi-full_mechanoregulation_summary.csv"
    curves = out_dir / f"{case_id}_roi-full_conditional_curves.png"
    schulte = out_dir / f"{case_id}_roi-full_schulte_binned_curves.png"
    summary = out_dir / f"{case_id}_roi-full_mechanoregulation_summary.json"
    surface_events = out_dir / f"{case_id}_roi-full_surface-events.nii.gz"
    for path in (csv_path, curves, schulte, summary, surface_events):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"out")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="mechanoregulation",
        profile="XtremeCTII",
        registered=False,
    )

    assert rows[0]["action"] == "Load"
    assert rows[0]["status"] == "Done"
    assert rows[0]["image_path"] == str(remodelling)
    assert rows[0]["sed_path"] == str(sed)
    assert rows[0]["output_paths"] == [str(csv_path), str(curves), str(schulte), str(summary), str(surface_events)]


def test_batch_processor_mechanoregulation_partial_outputs_remain_runnable(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    remodelling = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "xct"
        / "analysis"
        / "visualize"
        / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-5_remodelling.nii.gz"
    )
    baseline = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "transformed"
        / "sub-001_ses-001_voi-radiusleft_image-fused.nii.gz"
    )
    sed = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "maps"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTII_map-sed.nii.gz"
    )
    for path in (remodelling, baseline, sed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="mechanoregulation",
        profile="XtremeCTII",
        registered=False,
    )
    case_id = rows[0]["mechanoregulation_case_id"]
    out_dir = tmp_path / "derivatives" / "Mechanoregulation" / "sub-001" / "xct" / "runs" / case_id
    csv_path = out_dir / f"{case_id}_roi-full_mechanoregulation_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(b"partial")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="mechanoregulation",
        profile="XtremeCTII",
        registered=False,
    )

    assert rows[0]["action"] == "Run"
    assert rows[0]["status"] == "Ready"
    assert "output_paths" not in rows[0]


def test_batch_processor_mechanoregulation_outputs_allow_optional_surface_events(
    tmp_path: Path, monkeypatch
) -> None:
    module = _import_batch_processor_module(monkeypatch)
    case = object()
    outputs = {
        key: tmp_path / f"{key}.out"
        for key in ("csv", "curves", "schulte_curves", "summary")
    }
    for path in outputs.values():
        path.write_bytes(b"out")

    found = module.BatchProcessorLogic._mechanoregulation_output_paths_for_case(
        case,
        lambda _case, *, roi: outputs,
        lambda _case: {"full": tmp_path / "mask.nii.gz"},
    )

    assert found == list(outputs.values())


def test_batch_processor_compacts_mechanoregulation_summary_for_table_view(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    summary = tmp_path / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_roi-full_mechanoregulation_summary.csv"
    summary.write_text(
        "\n".join(
            [
                "roi,CCR,CCR_low_threshold,CCR_high_threshold,OR_R,OR_R_CI_low,OR_R_CI_high,OR_F,OR_F_CI_low,OR_F_CI_high,extra",
                "full,0.48,121.5,188.0,1.27,0.90,1.80,2.47,1.80,3.20,ignored",
            ]
        ),
        encoding="utf-8",
    )
    widget = object.__new__(module.BatchProcessorWidget)

    compact = widget._write_mechanoregulation_summary_table_csv([summary])

    rows = list(csv.DictReader(compact.open(newline="", encoding="utf-8")))
    assert rows == [
        {"ROI": "full", "Metric": "CCR", "Unit": "fraction", "Low conf": "", "Median": "0.48", "High conf": ""},
        {"ROI": "full", "Metric": "Lazy min", "Unit": "% normalized SED", "Low conf": "", "Median": "121.5", "High conf": ""},
        {"ROI": "full", "Metric": "Lazy max", "Unit": "% normalized SED", "Low conf": "", "Median": "188", "High conf": ""},
        {"ROI": "full", "Metric": "ORR", "Unit": "% per 1% SED decrease", "Low conf": "0.9", "Median": "1.27", "High conf": "1.8"},
        {"ROI": "full", "Metric": "ORF", "Unit": "% per 1% SED increase", "Low conf": "1.8", "Median": "2.47", "High conf": "3.2"},
    ]


def test_microarchitecture_registered_profile_groups_timepoints_with_spanning_action(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    records = []
    registration_records = []
    common_records = []
    for session in ("001", "002", "003"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        image = xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM"
        image.write_bytes(b"")
        common_dir = tmp_path / "derivatives" / "CommonRegion" / "sub-001" / f"ses-{session}" / "xct"
        common_dir.mkdir(parents=True)
        common_mask = common_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz"
        common_mask.write_bytes(b"")
        common_records.append(
            DerivativeRecord(
                "CommonRegion",
                "scan_region_native_common",
                "001",
                "radiusleft",
                session,
                None,
                "native",
                common_mask,
                "generated",
                content_type="mask",
            )
        )
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        if session != "001":
            transform_dir = tmp_path / "derivatives" / "Registration" / "sub-001" / f"ses-{session}" / "xct" / "pairwise"
            transform_dir.mkdir(parents=True)
            transform = transform_dir / f"sub-001_ses-{session}_voi-radiusleft_from-ses-{session}_to-ses-001_pairwise.tfm"
            transform.write_text("# transform\n", encoding="utf-8")
            registration_records.append(
                DerivativeRecord(
                    "Registration",
                    "transform_pairwise",
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "fixed",
                    transform,
                    "generated",
                )
            )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create(
            "Registration",
            tmp_path,
            {"name": "test", "version": "1"},
            records=registration_records,
        ),
        tmp_path / "derivatives" / "Registration" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=common_records),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    rows, message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="microarchitecture",
        profile="xtremectii-registered",
        registered=False,
    )

    assert message == "Discovered 3 row(s)."
    assert rows == [
        {
            "action": "Run",
            "action_row_span": 3,
            "group_id": "001|radiusleft|",
            "subject": "001",
            "session": "001",
            "session_value": "001",
            "voi": "radiusleft",
            "status": "Ready",
            "registered": True,
            "image_path": str(tmp_path / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.AIM"),
            "input": (
                "sub-001_ses-001_voi-radiusleft_xct.AIM\n"
                "seg=sub-001_ses-001_voi-radiusleft_desc-seg_mask.AIM\n"
                "full=sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM\n"
                "trab=sub-001_ses-001_voi-radiusleft_desc-trab_mask.AIM\n"
                "cort=sub-001_ses-001_voi-radiusleft_desc-cort_mask.AIM\n"
                "common=sub-001_ses-001_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz"
            ),
            "voi_value": "radiusleft",
        },
        {
            "action": "",
            "action_row_span": 0,
            "group_id": "001|radiusleft|",
            "subject": "001",
            "session": "002",
            "session_value": "002",
            "voi": "radiusleft",
            "status": "Ready",
            "registered": True,
            "image_path": str(tmp_path / "sub-001" / "ses-002" / "xct" / "sub-001_ses-002_voi-radiusleft_xct.AIM"),
            "input": (
                "sub-001_ses-002_voi-radiusleft_xct.AIM\n"
                "seg=sub-001_ses-002_voi-radiusleft_desc-seg_mask.AIM\n"
                "full=sub-001_ses-002_voi-radiusleft_desc-full_mask.AIM\n"
                "trab=sub-001_ses-002_voi-radiusleft_desc-trab_mask.AIM\n"
                "cort=sub-001_ses-002_voi-radiusleft_desc-cort_mask.AIM\n"
                "common=sub-001_ses-002_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz\n"
                "registration=sub-001_ses-002_voi-radiusleft_from-ses-002_to-ses-001_pairwise.tfm"
            ),
            "voi_value": "radiusleft",
        },
        {
            "action": "",
            "action_row_span": 0,
            "group_id": "001|radiusleft|",
            "subject": "001",
            "session": "003",
            "session_value": "003",
            "voi": "radiusleft",
            "status": "Ready",
            "registered": True,
            "image_path": str(tmp_path / "sub-001" / "ses-003" / "xct" / "sub-001_ses-003_voi-radiusleft_xct.AIM"),
            "input": (
                "sub-001_ses-003_voi-radiusleft_xct.AIM\n"
                "seg=sub-001_ses-003_voi-radiusleft_desc-seg_mask.AIM\n"
                "full=sub-001_ses-003_voi-radiusleft_desc-full_mask.AIM\n"
                "trab=sub-001_ses-003_voi-radiusleft_desc-trab_mask.AIM\n"
                "cort=sub-001_ses-003_voi-radiusleft_desc-cort_mask.AIM\n"
                "common=sub-001_ses-003_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz\n"
                "registration=sub-001_ses-003_voi-radiusleft_from-ses-003_to-ses-001_pairwise.tfm"
            ),
            "voi_value": "radiusleft",
        }
    ]


def test_plate_rod_registered_profile_groups_timepoints_like_microarchitecture(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    common_records = []
    for session in ("001", "002", "003"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        common = (
            tmp_path
            / "derivatives"
            / "CommonRegion"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / f"sub-001_ses-{session}_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz"
        )
        common.parent.mkdir(parents=True)
        common.write_bytes(b"")
        common_records.append(
            DerivativeRecord(
                "CommonRegion",
                "scan_region_native_common",
                "001",
                "radiusleft",
                session,
                None,
                "native",
                common,
                "generated",
                content_type="mask",
            )
        )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=common_records),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="plate_rod",
        profile="standard-registered",
        registered=False,
    )
    command = module.BatchProcessorLogic().command_for_row(
        tmp_path,
        tool="plate_rod",
        profile="standard-registered",
        row=rows[0],
    )

    assert [row["action"] for row in rows] == ["Run", "", ""]
    assert rows[0]["action_row_span"] == 3
    assert all(row["registered"] for row in rows)
    assert "seg=sub-001_ses-001_voi-radiusleft_desc-seg_mask.AIM" in rows[0]["input"]
    assert "trab=sub-001_ses-001_voi-radiusleft_desc-trab_mask.AIM" in rows[0]["input"]
    assert "sub-001_ses-001_voi-radiusleft_xct.AIM" not in rows[0]["input"]
    assert "full=" not in rows[0]["input"]
    assert "cort=" not in rows[0]["input"]
    assert "--session" not in command
    assert "--require-common-region" in command


def test_plate_rod_native_rows_pass_session_filter_and_no_common_region(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    row = {
        "subject": "001",
        "session": "001",
        "session_value": "001",
        "voi": "radiusleft",
        "voi_value": "radiusleft",
        "registered": False,
    }

    command = module.BatchProcessorLogic().command_for_row(
        tmp_path,
        tool="plate_rod",
        profile="standard",
        row=row,
    )

    assert command == [
        "-m",
        "plate_rod_thinning.cli",
        "run-batch",
        str(tmp_path.resolve()),
        "--subject",
        "001",
        "--session",
        "001",
        "--site",
        "radiusleft",
        "--no-common-region",
    ]


def test_plate_rod_registered_and_native_outputs_are_separate_load_states(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    output_records = []
    common_records = []
    for session in ("001", "002", "003"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (("bone_segmentation", "seg"), ("trabecular_mask", "trab")):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        common = tmp_path / "derivatives" / "CommonRegion" / "sub-001" / f"ses-{session}" / "xct" / f"sub-001_ses-{session}_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz"
        common.parent.mkdir(parents=True)
        common.write_bytes(b"")
        common_records.append(
            DerivativeRecord("CommonRegion", "scan_region_native_common", "001", "radiusleft", session, None, "native", common, "generated", content_type="mask")
        )
        native_table = tmp_path / "derivatives" / "PlateRodMorphometry" / "sub-001" / f"ses-{session}" / "xct" / "measurements" / f"sub-001_ses-{session}_voi-radiusleft_desc-plate-rod-measurements.csv"
        native_table.parent.mkdir(parents=True)
        native_table.write_text("subject_id,site,session_id\n001,radiusleft,001\n", encoding="utf-8")
        map_path = tmp_path / "derivatives" / "PlateRodMorphometry" / "sub-001" / f"ses-{session}" / "xct" / "maps" / f"sub-001_ses-{session}_voi-radiusleft_desc-plate-rod-label.npy"
        map_path.parent.mkdir(parents=True)
        map_path.write_bytes(b"")
        output_records.extend(
            [
                DerivativeRecord("PlateRodMorphometry", "plate_rod_measurements_table", "001", "radiusleft", session, None, "table", native_table, "generated", content_type="table", metadata={"use_common_region": False}),
                DerivativeRecord("PlateRodMorphometry", "plate_rod_label_map", "001", "radiusleft", session, None, "native", map_path, "generated", content_type="image", metadata={"use_common_region": False}),
            ]
        )
    write_manifest(DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records), tmp_path / "derivatives" / "BoneContours" / "manifest.json")
    write_manifest(DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=common_records), tmp_path / "derivatives" / "CommonRegion" / "manifest.json")
    write_manifest(DerivativeManifest.create("PlateRodMorphometry", tmp_path, {"name": "test", "version": "1"}, records=output_records), tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json")

    native_rows, _ = module.BatchProcessorLogic().discover_rows(tmp_path, tool="plate_rod", profile="standard", registered=False)
    registered_rows, _ = module.BatchProcessorLogic().discover_rows(tmp_path, tool="plate_rod", profile="standard-registered", registered=False)

    assert [row["action"] for row in native_rows] == ["Load", "Load", "Load"]
    assert [row["action"] for row in registered_rows] == ["Run", "", ""]
    assert all("output_paths" not in row for row in registered_rows)


def test_registered_microarchitecture_requires_native_common_region(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    records = []
    registration_records = []
    for session in ("001", "002", "003"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        if session != "001":
            transform = (
                tmp_path
                / "derivatives"
                / "Registration"
                / "sub-001"
                / f"ses-{session}"
                / "xct"
                / "pairwise"
                / f"sub-001_ses-{session}_voi-radiusleft_from-ses-{session}_to-ses-001_pairwise.tfm"
            )
            transform.parent.mkdir(parents=True)
            transform.write_text("# transform\n", encoding="utf-8")
            registration_records.append(
                DerivativeRecord(
                    "Registration",
                    "transform_pairwise",
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "fixed",
                    transform,
                    "generated",
                )
            )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("Registration", tmp_path, {"name": "test", "version": "1"}, records=registration_records),
        tmp_path / "derivatives" / "Registration" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="microarchitecture",
        profile="xtremectii-registered",
        registered=False,
    )

    assert rows[0]["action"] == "Missing"
    assert {row["status"] for row in rows} == {"Missing common region"}


def test_bone_contouring_ignores_registered_table_mode(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    for session in ("001", "002"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")

    rows, message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="bone_contouring",
        profile="standard",
        registered=True,
    )

    assert message == "Discovered 2 row(s)."
    assert [row["action"] for row in rows] == ["Run", "Run"]
    assert all("action_row_span" not in row for row in rows)


def test_partial_bone_contouring_rows_remain_runnable(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    (xct_dir / "sub-001_ses-001_voi-radiusleft_xct.AIM").write_bytes(b"")
    contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
    contour_dir.mkdir(parents=True)
    output = contour_dir / "sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM"
    output.write_bytes(b"")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="bone_contouring",
        profile="standard",
        registered=False,
    )

    assert rows[0]["action"] == "Run"
    assert rows[0]["output_paths"] == [str(output)]


def test_complete_bone_contouring_rows_are_loadable(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    (xct_dir / "sub-001_ses-001_voi-radiusleft_xct.AIM").write_bytes(b"")
    contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
    contour_dir.mkdir(parents=True)
    outputs = []
    for role, suffix in (
        ("seg", "mask"),
        ("full", "mask"),
        ("trab", "mask"),
        ("cort", "mask"),
        ("fea-input", "label"),
    ):
        output = contour_dir / f"sub-001_ses-001_voi-radiusleft_desc-{role}_{suffix}.AIM"
        output.write_bytes(b"")
        outputs.append(str(output))

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="bone_contouring",
        profile="standard",
        registered=False,
    )

    assert rows[0]["action"] == "Load"
    assert set(rows[0]["output_paths"]) == set(outputs)


def test_bone_contouring_completion_reuses_ipl_contours(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    (xct_dir / "sub-001_ses-001_voi-radiusleft_xct.AIM").write_bytes(b"")
    outputs = []
    for family, role, suffix in (
        ("IPLContours", "trab", "mask"),
        ("IPLContours", "cort", "mask"),
        ("BoneContours", "seg", "mask"),
        ("BoneContours", "full", "mask"),
        ("BoneContours", "fea-input", "label"),
    ):
        directory = tmp_path / "derivatives" / family / "sub-001" / "ses-001" / "xct"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"sub-001_ses-001_voi-radiusleft_desc-{role}_{suffix}.AIM"
        output.write_bytes(b"")
        outputs.append(str(output))

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="bone_contouring",
        profile="standard",
        registered=False,
    )

    assert rows[0]["action"] == "Load"
    assert set(rows[0]["output_paths"]) == set(outputs)


def test_batch_load_can_rediscover_outputs_written_after_analyze(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    (xct_dir / "sub-001_ses-001_voi-radiusleft_xct.AIM").write_bytes(b"")

    logic = module.BatchProcessorLogic()
    rows, _message = logic.discover_rows(
        tmp_path,
        tool="bone_contouring",
        profile="XtremeCTI",
        registered=False,
    )
    assert rows[0]["action"] == "Run"
    assert "output_paths" not in rows[0]

    contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
    contour_dir.mkdir(parents=True)
    output = contour_dir / "sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM"
    output.write_bytes(b"")

    assert logic.rediscover_row_output_paths(tmp_path, "bone_contouring", rows[0]) == [str(output)]


def test_microarchitecture_existing_outputs_are_profile_mode_specific(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    native = module.BatchArtifact(
        tmp_path / "derivatives" / "Microarchitecture" / "sub-001" / "ses-001" / "xct" / "measurements" / "native.csv",
        module.CaseKey("001", "001", "radiusleft", None),
        "measurements_table",
        "Microarchitecture",
        metadata={**METHOD_METADATA, "use_common_region": False},
    )
    registered = module.BatchArtifact(
        tmp_path
        / "derivatives"
        / "Microarchitecture"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "registered"
        / "measurements"
        / "registered.csv",
        module.CaseKey("001", "001", "radiusleft", None),
        "measurements_table",
        "Microarchitecture",
        metadata={**METHOD_METADATA, "use_common_region": True},
    )

    logic = module.BatchProcessorLogic()

    assert logic._existing_outputs_for_profile("microarchitecture", False, (native, registered)) == (native,)
    assert logic._existing_outputs_for_profile("microarchitecture", True, (native, registered)) == (registered,)
    assert logic._existing_outputs_for_profile("timelapse", True, (native, registered)) == (native, registered)


def test_native_microarchitecture_outputs_do_not_make_registered_profile_loadable(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    common_records = []
    registration_records = []
    output_records = []
    for session in ("001", "002", "003"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        image = xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM"
        image.write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        common = (
            tmp_path
            / "derivatives"
            / "CommonRegion"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / f"sub-001_ses-{session}_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz"
        )
        common.parent.mkdir(parents=True)
        common.write_bytes(b"")
        common_records.append(
            DerivativeRecord(
                "CommonRegion",
                "scan_region_native_common",
                "001",
                "radiusleft",
                session,
                None,
                "native",
                common,
                "generated",
                content_type="mask",
            )
        )
        native_measurement = (
            tmp_path
            / "derivatives"
            / "Microarchitecture"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / "measurements"
            / f"sub-001_ses-{session}_voi-radiusleft_measurements.csv"
        )
        native_measurement.parent.mkdir(parents=True)
        native_measurement.write_text("Parameter,Mean\nTb.N,1.0\n", encoding="utf-8")
        _mark_current_microarchitecture(native_measurement)
        output_records.append(
            DerivativeRecord(
                "Microarchitecture",
                "measurements_table",
                "001",
                "radiusleft",
                session,
                None,
                "table",
                native_measurement,
                "generated",
                content_type="table",
                metadata={"use_common_region": False},
            )
        )
        native_map = (
            tmp_path
            / "derivatives"
            / "Microarchitecture"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / "maps"
            / f"sub-001_ses-{session}_voi-radiusleft_map-tb-th.nii.gz"
        )
        native_map.parent.mkdir(parents=True, exist_ok=True)
        native_map.write_bytes(b"")
        output_records.append(
            DerivativeRecord(
                "Microarchitecture",
                "trabecular_thickness_map",
                "001",
                "radiusleft",
                session,
                None,
                "native",
                native_map,
                "generated",
                content_type="image",
                metadata={"use_common_region": False, "map_name": "Tb.Th"},
            )
        )
        if session != "001":
            transform = (
                tmp_path
                / "derivatives"
                / "Registration"
                / "sub-001"
                / f"ses-{session}"
                / "xct"
                / "pairwise"
                / f"sub-001_ses-{session}_voi-radiusleft_from-ses-{session}_to-ses-001_pairwise.tfm"
            )
            transform.parent.mkdir(parents=True)
            transform.write_text("# transform\n", encoding="utf-8")
            registration_records.append(
                DerivativeRecord(
                    "Registration",
                    "transform_pairwise",
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "fixed",
                    transform,
                    "generated",
                )
            )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=common_records),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("Registration", tmp_path, {"name": "test", "version": "1"}, records=registration_records),
        tmp_path / "derivatives" / "Registration" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("Microarchitecture", tmp_path, {"name": "test", "version": "1"}, records=output_records),
        tmp_path / "derivatives" / "Microarchitecture" / "manifest.json",
    )

    logic = module.BatchProcessorLogic()
    native_rows, _ = logic.discover_rows(tmp_path, tool="microarchitecture", profile="xtremectii", registered=False)
    registered_rows, _ = logic.discover_rows(
        tmp_path, tool="microarchitecture", profile="xtremectii-registered", registered=False
    )

    assert [row["action"] for row in native_rows] == ["Load", "Load", "Load"]
    assert [row["action"] for row in registered_rows] == ["Run", "", ""]
    assert all("output_paths" not in row for row in registered_rows)


def test_registered_microarchitecture_outputs_do_not_make_native_profile_loadable(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    output_records = []
    for session in ("001", "002", "003"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        image = xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM"
        image.write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        registered_measurement = (
            tmp_path
            / "derivatives"
            / "Microarchitecture"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / "registered_measurements"
            / f"sub-001_ses-{session}_voi-radiusleft_measurements.csv"
        )
        registered_measurement.parent.mkdir(parents=True, exist_ok=True)
        registered_measurement.write_text("Parameter,Mean\nTb.N,1.0\n", encoding="utf-8")
        native_map = (
            tmp_path
            / "derivatives"
            / "Microarchitecture"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / "maps"
            / f"sub-001_ses-{session}_voi-radiusleft_map-tb-th.nii.gz"
        )
        native_map.parent.mkdir(parents=True, exist_ok=True)
        native_map.write_bytes(b"")
        output_records.extend(
            [
                DerivativeRecord(
                    "Microarchitecture",
                    "measurements_table",
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "table",
                    registered_measurement,
                    "generated",
                    content_type="table",
                    metadata={"use_common_region": True},
                ),
                DerivativeRecord(
                    "Microarchitecture",
                    "trabecular_thickness_map",
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    native_map,
                    "generated",
                    content_type="image",
                    metadata={"use_common_region": False, "map_name": "Tb.Th"},
                ),
            ]
        )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("Microarchitecture", tmp_path, {"name": "test", "version": "1"}, records=output_records),
        tmp_path / "derivatives" / "Microarchitecture" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="microarchitecture",
        profile="xtremectii",
        registered=False,
    )

    assert [row["action"] for row in rows] == ["Run", "Run", "Run"]
    assert all("output_paths" not in row for row in rows)


def test_microarchitecture_rows_show_found_and_missing_masks(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    image = xct_dir / "sub-001_ses-001_voi-radiusleft_stack-02_xct.AIM"
    image.write_bytes(b"")
    contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
    contour_dir.mkdir(parents=True)
    records = []
    for role, filename_role in (("bone_segmentation", "seg"), ("periosteal_mask", "full")):
        mask = contour_dir / f"sub-001_ses-001_voi-radiusleft_stack-02_desc-{filename_role}_mask.AIM"
        mask.write_bytes(b"")
        records.append(
            DerivativeRecord(
                "BoneContours",
                role,
                "001",
                "radiusleft",
                "001",
                2,
                "native",
                mask,
                "generated",
                content_type="mask",
            )
        )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="microarchitecture",
        profile="standard",
        registered=False,
    )

    assert rows == [
        {
            "action": "Missing",
            "subject": "001",
            "session": "001",
            "voi": "radiusleft stack-02",
            "session_value": "001",
            "voi_value": "radiusleft",
            "registered": False,
            "status": "Missing trab, cort",
            "image_path": str(image),
            "input": (
                "sub-001_ses-001_voi-radiusleft_stack-02_xct.AIM\n"
                "seg=sub-001_ses-001_voi-radiusleft_stack-02_desc-seg_mask.AIM\n"
                "full=sub-001_ses-001_voi-radiusleft_stack-02_desc-full_mask.AIM"
            ),
        }
    ]


def test_stack_one_registration_does_not_match_unstacked_series_row(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    registration_records = []
    for session in ("001", "002"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
    transform = (
        tmp_path
        / "derivatives"
        / "Registration"
        / "sub-001"
        / "ses-002"
        / "xct"
        / "pairwise"
        / "sub-001_ses-002_voi-radiusleft_stack-01_from-ses-002_to-ses-001_pairwise.tfm"
    )
    transform.parent.mkdir(parents=True)
    transform.write_text("# transform\n", encoding="utf-8")
    registration_records.append(
        DerivativeRecord(
            "Registration",
            "transform_pairwise",
            "001",
            "radiusleft",
            "002",
            1,
            "fixed",
            transform,
            "generated",
        )
    )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("Registration", tmp_path, {"name": "test", "version": "1"}, records=registration_records),
        tmp_path / "derivatives" / "Registration" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="timelapse",
        profile="standard",
        registered=True,
    )

    assert "registration=" not in rows[1]["input"]


def test_multistack_timelapse_profiles_group_all_stacks_into_one_series_action(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    for session in ("001", "002"):
        for stack in (1, 2):
            xct_dir = tmp_path / "sub-BMLT006" / f"ses-{session}" / "xct"
            xct_dir.mkdir(parents=True, exist_ok=True)
            image = xct_dir / f"sub-BMLT006_ses-{session}_voi-knee_stack-{stack:02d}_xct.AIM"
            image.write_bytes(b"")
            contour_dir = tmp_path / "derivatives" / "ImportedContours" / "sub-BMLT006" / f"ses-{session}" / "xct"
            contour_dir.mkdir(parents=True, exist_ok=True)
            for role, filename_role in (
                ("bone_segmentation", "seg"),
                ("periosteal_mask", "full"),
                ("trabecular_mask", "trab"),
                ("cortical_mask", "cort"),
            ):
                mask = contour_dir / f"sub-BMLT006_ses-{session}_voi-knee_stack-{stack:02d}_desc-{filename_role}_mask.AIM"
                mask.write_bytes(b"")
                contour_records.append(
                    DerivativeRecord(
                        "ImportedContours",
                        role,
                        "BMLT006",
                        "knee",
                        session,
                        stack,
                        "native",
                        mask,
                        "generated",
                        content_type="mask",
                    )
                )
    write_manifest(
        DerivativeManifest.create("ImportedContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "ImportedContours" / "manifest.json",
    )

    standard_rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="timelapse",
        profile="standard",
        registered=True,
    )
    multistack_rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="timelapse",
        profile="multistack",
        registered=True,
    )
    pedfx_rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="timelapse",
        profile="ped-fx",
        registered=True,
    )

    assert [row["action"] for row in standard_rows].count("Run") == 2
    assert [row["action"] for row in multistack_rows].count("Run") == 1
    assert multistack_rows[0]["action_row_span"] == 4
    assert [row["action"] for row in pedfx_rows].count("Run") == 1
    assert pedfx_rows[0]["action_row_span"] == 4
    command = module.BatchProcessorLogic().command_for_row(
        tmp_path,
        tool="timelapse",
        profile="multistack",
        row=multistack_rows[0],
    )
    assert "--session" not in command
    assert "--site" in command
    assert command[command.index("--site") + 1] == "knee"
    assert command[command.index("--profile") + 1] == "multistack"


def test_unregistered_rows_do_not_list_registration_inputs(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    registration_records = []
    for session in ("001", "002"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
    transform = tmp_path / "derivatives" / "Registration" / "sub-001" / "ses-002" / "xct" / "pairwise" / "sub-001_ses-002_voi-radiusleft_from-ses-002_to-ses-001_pairwise.tfm"
    transform.parent.mkdir(parents=True)
    transform.write_text("# transform\n", encoding="utf-8")
    registration_records.append(
        DerivativeRecord(
            "Registration",
            "transform_pairwise",
            "001",
            "radiusleft",
            "002",
            None,
            "fixed",
            transform,
            "generated",
        )
    )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("Registration", tmp_path, {"name": "test", "version": "1"}, records=registration_records),
        tmp_path / "derivatives" / "Registration" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="microarchitecture",
        profile="standard",
        registered=False,
    )

    assert all("registration=" not in row["input"] for row in rows)


def test_registration_inputs_prefer_imported_registration(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    for session in ("001", "002"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
    generated = tmp_path / "derivatives" / "Registration" / "sub-001" / "ses-002" / "xct" / "pairwise" / "generated_pairwise.tfm"
    imported = (
        tmp_path
        / "derivatives"
        / "ImportedRegistration"
        / "sub-001"
        / "ses-002"
        / "xct"
        / "pairwise"
        / "imported_pairwise.tfm"
    )
    registration_records = []
    imported_records = []
    for family, path, records in (
        ("Registration", generated, registration_records),
        ("ImportedRegistration", imported, imported_records),
    ):
        path.parent.mkdir(parents=True)
        path.write_text("# transform\n", encoding="utf-8")
        records.append(
            DerivativeRecord(
                family,
                "transform_pairwise",
                "001",
                "radiusleft",
                "002",
                None,
                "fixed",
                path,
                "generated" if family == "Registration" else "provided",
            )
        )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("Registration", tmp_path, {"name": "test", "version": "1"}, records=registration_records),
        tmp_path / "derivatives" / "Registration" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("ImportedRegistration", tmp_path, {"name": "test", "version": "1"}, records=imported_records),
        tmp_path / "derivatives" / "ImportedRegistration" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="timelapse",
        profile="standard",
        registered=True,
    )

    moving_row = next(row for row in rows if row["session"] == "002")
    assert "registration=imported_pairwise.tfm" in moving_row["input"]
    assert "generated_pairwise.tfm" not in moving_row["input"]


def test_timelapse_discovered_profiles_group_timepoints(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    for session in ("001", "002", "003"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )

    rows, message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="timelapse",
        profile="xct1-standard",
        registered=False,
    )

    assert message == "Discovered 3 row(s)."
    assert [row["action"] for row in rows] == ["Run", "", ""]
    assert rows[0]["action_row_span"] == 3
    assert rows[1]["action_row_span"] == 0
    assert rows[2]["action_row_span"] == 0
    assert {row["group_id"] for row in rows} == {"001|radiusleft|"}


def test_profile_change_rediscovers_current_dataset() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "registerCheck" not in source
    assert "self.profileCombo.currentIndexChanged.connect(self._on_profile_changed)" in source
    assert "def _on_profile_changed(self" in source
    assert "self._analyze_dataset()" in source[source.index("    def _on_profile_changed(") :]
    assert "def _on_tool_changed(self" in source
    assert "self._populate_profile_combo()" in source[source.index("    def _on_tool_changed(") :]
    assert "self._analyze_dataset()" in source[source.index("    def _on_tool_changed(") :]


def test_batch_profiles_are_tool_specific_and_encode_registration() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "TOOL_PROFILES = {" in source
    assert '"microarchitecture": (' in source
    assert '("XtremeCT II", "xtremectii", False)' in source
    assert '("XtremeCT II - registered", "xtremectii-registered", True)' in source
    assert '("Registered Functional Bone", "functional-bone", True)' in source
    assert '("Native Functional Bone", "functional-bone-native", False)' in source
    assert '"timelapse": (' in source
    assert '("Standard", "standard", True)' in source
    assert '("ETH-UofC", "eth-uofc", True)' in source
    assert '("Shriners", "shriners", True)' in source
    assert "from timelapsedhrpqct.config.profiles import list_config_profiles" in source
    assert "for value in list_config_profiles():" in source
    assert '"xct1-standard": "XtremeCT I Standard"' in source
    assert '"ped-fx": "Pediatric Fracture"' in source
    assert '"ucsf": "UCSF"' in source
    assert '"shriners": "Shriners"' in source
    assert "eth-uofc-compatibility" not in source
    assert "shriners-compatibility" not in source
    assert '"bone_contouring": (' in source
    assert '"voidspace": (' in source
    assert '("Voidspace", "standard", False)' in source
    assert '("Registered voidspace", "registered", True)' in source
    assert '("Dynamic voidspace", "dynamic", True)' in source
    assert '("Voidspace", "voidspace")' in source
    assert '"voidspace": "Voidspace"' in source
    assert '"voidspace.cli"' in source
    assert "from voidspace import analyze_maps, run_case" in source
    assert 'if tool == "voidspace":' in source
    assert "def profile_groups_timepoints(tool: str, profile: str) -> bool:" in source
    assert "def _row_indices_for_group_action(self, row_index):" in source
    assert "for child_index in self._row_indices_for_group_action(row_index):" in source
    assert "self.profileCombo.clear()" in source
    assert 'for label, value, _registered in self._profiles_for_tool(tool):' in source
    assert "return self.logic.profile_requests_registration(" in source
    assert 'args.append("--no-common-region")' in source
    assert 'args.append("--require-common-region")' in source
    assert 'if tool == "timelapse" and profile_value:' in source


@pytest.mark.parametrize("voidspace_profile", ["native", "registered"])
def test_functional_bone_profile_uses_common_region_minus_voidspace(tmp_path: Path, monkeypatch, voidspace_profile) -> None:
    module = _import_batch_processor_module(monkeypatch)
    records = []
    common_records = []
    for session in ("001", "002"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        image = xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM"
        image.write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        common_mask = (
            tmp_path
            / "derivatives"
            / "CommonRegion"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / f"sub-001_ses-{session}_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz"
        )
        common_mask.parent.mkdir(parents=True)
        common_mask.write_bytes(b"")
        common_records.append(
            DerivativeRecord(
                "CommonRegion",
                "scan_region_native_common",
                "001",
                "radiusleft",
                session,
                None,
                "native",
                common_mask,
                "generated",
                content_type="mask",
            )
        )
        voidspace_dir = (
            tmp_path
            / "derivatives"
            / "Voidspace"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / voidspace_profile
            / "voi-radiusleft"
        )
        voidspace_dir.mkdir(parents=True)
        (voidspace_dir / "voidspace_large_mask.nii.gz").write_bytes(b"")
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=common_records),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    logic = module.BatchProcessorLogic()
    rows, _message = logic.discover_rows(
        tmp_path,
        tool="microarchitecture",
        profile="functional-bone",
        registered=False,
    )

    assert [row["action"] for row in rows] == ["Run", ""]
    assert rows[0]["profile"] == "functional-bone"
    assert rows[0]["voidspace_mask_path"].endswith("voidspace_large_mask.nii.gz")
    assert "voidspace=voidspace_large_mask.nii.gz" in rows[0]["input"]
    assert all(row["registered"] for row in rows)
    command = logic.command_for_row(tmp_path, tool="microarchitecture", profile="functional-bone", row=rows[0])
    assert command[0] == "-c"
    assert "functional_common = common & ~voidspace" in command[1]
    assert command[command.index("--segmentation") + 1].endswith("_desc-seg_mask.AIM")
    assert command[command.index("--full-mask") + 1].endswith("_desc-full_mask.AIM")
    assert command[command.index("--trab-mask") + 1].endswith("_desc-trab_mask.AIM")
    assert command[command.index("--cort-mask") + 1].endswith("_desc-cort_mask.AIM")
    assert command[command.index("--common-region") + 1].endswith("native-common_mask.nii.gz")
    assert command[command.index("--voidspace-mask") + 1].endswith("voidspace_large_mask.nii.gz")
    assert command[command.index("--dataset-root") + 1] == str(tmp_path)
    assert command[command.index("--output-dir") + 1].endswith("/functional_bone_measurements")
    assert "_desc-functional-bone-analysis_mask{analysis_extension}" in command[1]
    assert "_mask_on_reference_grid" in command[1]
    row_without_cort = dict(rows[0])
    row_without_cort.pop("cort_mask_path")
    with pytest.raises(ValueError, match="cortical mask"):
        logic.command_for_row(tmp_path, tool="microarchitecture", profile="functional-bone", row=row_without_cort)


def test_functional_bone_reports_missing_native_or_registered_voidspace(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    records = []
    common_records = []
    for session in ("001", "002"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        common_mask = tmp_path / "derivatives" / "CommonRegion" / "sub-001" / f"ses-{session}" / "xct" / "common.nii.gz"
        common_mask.parent.mkdir(parents=True)
        common_mask.write_bytes(b"")
        common_records.append(
            DerivativeRecord(
                "CommonRegion",
                "scan_region_native_common",
                "001",
                "radiusleft",
                session,
                None,
                "native",
                common_mask,
                "generated",
                content_type="mask",
            )
        )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create(
            "CommonRegion",
            tmp_path,
            {"name": "test", "version": "1"},
            records=common_records,
        ),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="microarchitecture",
        profile="functional-bone",
        registered=False,
    )

    assert rows[0]["action"] == "Missing"
    assert rows[0]["status"] == "Missing voidspace: run native or registered Voidspace"


def test_functional_bone_existing_outputs_include_analysis_mask(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    records = []
    common_records = []
    for session in ("001", "002"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (
            ("bone_segmentation", "seg"),
            ("periosteal_mask", "full"),
            ("trabecular_mask", "trab"),
            ("cortical_mask", "cort"),
        ):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        common_mask = tmp_path / "derivatives" / "CommonRegion" / "sub-001" / f"ses-{session}" / "xct" / "common.nii.gz"
        common_mask.parent.mkdir(parents=True)
        common_mask.write_bytes(b"")
        common_records.append(
            DerivativeRecord(
                "CommonRegion",
                "scan_region_native_common",
                "001",
                "radiusleft",
                session,
                None,
                "native",
                common_mask,
                "generated",
                content_type="mask",
            )
        )
        voidspace_dir = tmp_path / "derivatives" / "Voidspace" / "sub-001" / f"ses-{session}" / "xct" / "registered" / "voi-radiusleft"
        voidspace_dir.mkdir(parents=True)
        (voidspace_dir / "voidspace_large_mask.nii.gz").write_bytes(b"")
        functional_dir = (
            tmp_path
            / "derivatives"
            / "Microarchitecture"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / "functional_bone_measurements"
        )
        functional_dir.mkdir(parents=True)
        (functional_dir / f"sub-001_ses-{session}_voi-radiusleft_measurements.csv").write_text("Parameter,Mean\nTb.N,1.0\n")
        _mark_current_microarchitecture(functional_dir / f"sub-001_ses-{session}_voi-radiusleft_measurements.csv")
        (functional_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-functional-bone-analysis_mask.nii.gz").write_bytes(b"")
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=common_records),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="microarchitecture",
        profile="functional-bone",
        registered=False,
    )

    assert rows[0]["action"] == "Load"
    assert any(path.endswith("_desc-functional-bone-analysis_mask.nii.gz") for path in rows[0]["output_paths"])


@pytest.mark.parametrize("profile, directory, expected_count", [
    ("functional-bone", "functional_bone_measurements", 17),
    ("functional-bone-native", "functional_bone_native_measurements", 26),
])
@pytest.mark.parametrize("cropped_voidspace", [False, True])
@pytest.mark.parametrize("site", ["radiusleft", "tibialeft"])
def test_functional_bone_command_runs_with_core_native_map_generation(tmp_path: Path, monkeypatch, cropped_voidspace, profile, directory, expected_count, site) -> None:
    pytest.importorskip("scipy")
    sitk = pytest.importorskip("SimpleITK")
    module = _import_batch_processor_module(monkeypatch)
    image = np.full((3, 3, 3), 100.0, dtype=np.float32)
    full = np.ones((3, 3, 3), dtype=np.uint8)
    trab = np.ones((3, 3, 3), dtype=np.uint8)
    cort = np.zeros((3, 3, 3), dtype=np.uint8)
    cort[:, :, 0] = 1
    seg = np.ones((3, 3, 3), dtype=np.uint8)
    common = np.ones((3, 3, 3), dtype=np.uint8)
    common[2] = 0
    voidspace = np.zeros((3, 3, 3), dtype=np.uint8)
    voidspace[1, 1, 1] = 1
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True, exist_ok=True)
    image_path = xct_dir / f"sub-001_ses-001_voi-{site}_xct.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(image), str(image_path))
    contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
    contour_dir.mkdir(parents=True, exist_ok=True)
    contour_paths = {}
    for filename_role, array in {"seg": seg, "full": full, "trab": trab, "cort": cort}.items():
        path = contour_dir / f"sub-001_ses-001_voi-{site}_desc-{filename_role}_mask.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
        contour_paths[filename_role] = path
    common_path = tmp_path / "common.nii.gz"
    voidspace_path = tmp_path / "voidspace_large_mask.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(common), str(common_path))
    void_image = sitk.GetImageFromArray(voidspace)
    if cropped_voidspace:
        void_image = sitk.RegionOfInterest(void_image, [1, 1, 1], [1, 1, 1])
    sitk.WriteImage(void_image, str(voidspace_path))
    maps_dir = tmp_path / "derivatives" / "Microarchitecture" / "sub-001" / "ses-001" / "xct" / "maps"
    row = {
        "subject": "001",
        "session": "001",
        "session_value": "001",
        "voi": site,
        "voi_value": site,
        "image_path": str(image_path),
        "seg_path": str(contour_paths["seg"]),
        "full_mask_path": str(contour_paths["full"]),
        "trab_mask_path": str(contour_paths["trab"]),
        "cort_mask_path": str(contour_paths["cort"]),
        "voidspace_mask_path": str(voidspace_path),
    }
    if profile == "functional-bone":
        row["common_region_path"] = str(common_path)

    command = module.BatchProcessorLogic().command_for_row(
        tmp_path,
        tool="microarchitecture",
        profile=profile,
        row=row,
    )
    assert ("--common-region" in command) == (profile == "functional-bone")
    subprocess.run(
        [sys.executable, *command, "--thickness-method", "edt", "--thickness-backend", "cpu"],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [
                    str(ROOT.parent / "bone-microarchitecture" / "src"),
                    str(ROOT.parent / "bone-imaging-derivatives" / "src"),
                ]
            ),
        },
    )

    output = (
        tmp_path
        / "derivatives"
        / "Microarchitecture"
        / "sub-001"
        / "ses-001"
        / "xct"
        / directory
        / f"sub-001_ses-001_voi-{site}_measurements.csv"
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = {row["Parameter"]: row for row in csv.DictReader(handle)}
    assert float(rows["Tt.BMD"]["Mean"]) == 100.0
    analysis_path = output.parent / f"sub-001_ses-001_voi-{site}_desc-functional-bone-analysis_mask.nii.gz"
    analysis_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(analysis_path)))
    assert np.count_nonzero(analysis_mask) == expected_count
    assert not (maps_dir / f"sub-001_ses-001_voi-{site}_map-functional-bone.npy").exists()
    assert any(maps_dir.glob(f"sub-001_ses-001_voi-{site}_map-*.nii.gz"))


@pytest.mark.parametrize("voidspace_profile", ["native", "registered"])
@pytest.mark.parametrize("registered_request", [False, True])
def test_native_functional_bone_discovery_needs_native_voidspace_not_common_region(tmp_path, monkeypatch, voidspace_profile, registered_request):
    module = _import_batch_processor_module(monkeypatch)
    records = []
    for session in ("001", "002"):
        image = tmp_path / "sub-001" / f"ses-{session}" / "xct" / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"")
        for role, short in (("bone_segmentation", "seg"), ("periosteal_mask", "full"),
                            ("trabecular_mask", "trab"), ("cortical_mask", "cort")):
            path = tmp_path / "derivatives/BoneContours/sub-001" / f"ses-{session}" / "xct" / f"sub-001_ses-{session}_voi-radiusleft_desc-{short}_mask.AIM"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            records.append(DerivativeRecord("BoneContours", role, "001", "radiusleft", session,
                                            None, "native", path, "generated", content_type="mask"))
        void = tmp_path / "derivatives/Voidspace/sub-001" / f"ses-{session}" / "xct" / voidspace_profile / "voi-radiusleft/voidspace_large_mask.nii.gz"
        void.parent.mkdir(parents=True)
        void.write_bytes(b"")
    write_manifest(DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=records),
                   tmp_path / "derivatives/BoneContours/manifest.json")
    logic = module.BatchProcessorLogic()
    rows, _ = logic.discover_rows(tmp_path, tool="microarchitecture", profile="functional-bone-native", registered=registered_request)
    assert len(rows) == 2
    assert all(not row["registered"] for row in rows)
    assert all(row["profile"] == "functional-bone-native" for row in rows)
    if voidspace_profile == "native":
        assert [row["action"] for row in rows] == ["Run", "Run"]
        command = logic.command_for_row(tmp_path, tool="microarchitecture", profile="functional-bone-native", row=rows[0])
        assert "--common-region" not in command
        assert "/native/" in command[command.index("--voidspace-mask") + 1]
        assert command[command.index("--output-dir") + 1].endswith("/functional_bone_native_measurements")
        for row in rows:
            folder = tmp_path / "derivatives/Microarchitecture/sub-001" / f"ses-{row['session_value']}" / "xct/functional_bone_native_measurements"
            folder.mkdir(parents=True)
            (folder / f"sub-001_ses-{row['session_value']}_voi-radiusleft_measurements.csv").write_text("Parameter,Mean\nTt.BMD,100\n")
            _mark_current_microarchitecture(folder / f"sub-001_ses-{row['session_value']}_voi-radiusleft_measurements.csv")
            (folder / f"sub-001_ses-{row['session_value']}_voi-radiusleft_desc-functional-bone-analysis_mask.nii.gz").write_bytes(b"")
        loaded_rows, _ = logic.discover_rows(tmp_path, tool="microarchitecture", profile="functional-bone-native", registered=registered_request)
        assert [row["action"] for row in loaded_rows] == ["Load", "Load"]
        assert all(any(path.endswith("_desc-functional-bone-analysis_mask.nii.gz") for path in row["output_paths"]) for row in loaded_rows)
    else:
        assert [row["action"] for row in rows] == ["Missing", "Missing"]
        assert all("native Voidspace" in row["status"] for row in rows)


@pytest.mark.parametrize("profile, registered, directory", [
    ("xtremectii", False, "measurements"),
    ("xtremectii-registered", True, "registered_measurements"),
    ("functional-bone", True, "functional_bone_measurements"),
    ("functional-bone-native", False, "functional_bone_native_measurements"),
])
def test_functional_bone_measurement_discovery_isolates_profile_outputs(tmp_path, monkeypatch, profile, registered, directory):
    module = _import_batch_processor_module(monkeypatch)
    base = tmp_path / "derivatives/Microarchitecture/sub-001/ses-001/xct"
    for folder in ("measurements", "registered_measurements", "functional_bone_measurements", "functional_bone_native_measurements"):
        path = base / folder / "sub-001_ses-001_voi-radiusleft_measurements.csv"
        path.parent.mkdir(parents=True)
        path.write_text("Parameter,Mean\nTt.BMD,100\n")
        _mark_current_microarchitecture(path)
    logic = module.BatchProcessorLogic()
    outputs = logic._discover_existing_outputs(tmp_path, "Microarchitecture")
    selected = logic._existing_outputs_for_profile("microarchitecture", registered, outputs, profile)
    assert [item.path.parent.name for item in selected] == [directory]
    if profile.startswith("functional-bone"):
        mask = base / directory / "sub-001_ses-001_voi-radiusleft_desc-functional-bone-analysis_mask.nii.gz"
        mask.write_bytes(b"")
        row = dict(subject="001", session="001", voi="radiusleft", profile=profile, registered=registered)
        assert logic.rediscover_row_output_paths(tmp_path, "microarchitecture", row) == [str(selected[0].path), str(mask)]


def test_functional_bone_load_routes_analysis_region_as_segmentation() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    load_outputs = source.split("    def _load_row_outputs", 1)[1].split("\n    def _load_voidspace_outputs", 1)[0]
    assert "functional_analysis_paths = [path for path in output_paths if self._is_functional_bone_analysis_mask(path)]" in load_outputs
    assert "output_paths = [path for path in output_paths if not self._is_functional_bone_analysis_mask(path)]" in load_outputs
    assert "self._load_functional_bone_analysis_overlays(row_index)" in load_outputs
    assert "self._load_functional_bone_analysis_outputs(row, functional_analysis_paths)" in load_outputs
    assert "def _load_functional_bone_analysis_overlays(self, row_index):" in source
    assert 'name.startswith("functional_bone_analysis_mask")' in source
    assert '"functional_bone_analysis": (0.30, 0.72, 0.55)' in source


def test_voidspace_batch_command_uses_explicit_segmentation_and_optional_mask(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    logic = module.BatchProcessorLogic()
    row = {
        "subject": "001",
        "session": "001",
        "session_value": "001",
        "voi": "radiusleft",
        "voi_value": "radiusleft",
        "seg_path": str(tmp_path / "seg.AIM"),
        "full_mask_path": str(tmp_path / "full.AIM"),
        "common_region_path": str(tmp_path / "mask.nii.gz"),
    }

    native = logic.command_for_row(tmp_path, tool="voidspace", profile="standard", row=row, force=True)
    registered = logic.command_for_row(tmp_path, tool="voidspace", profile="registered", row=row)

    assert native == [
        "-m",
        "voidspace.cli",
        "run-case",
        "--segmentation",
        str(tmp_path / "seg.AIM"),
        "--output-dir",
        str(tmp_path / "derivatives" / "Voidspace" / "sub-001" / "ses-001" / "xct" / "native" / "voi-radiusleft"),
        "--mask",
        str(tmp_path / "full.AIM"),
        "--force",
    ]
    assert registered[0] == "-c"
    assert "from voidspace import intersect_masks" in registered[1]
    assert registered[registered.index("--segmentation") + 1] == str(tmp_path / "seg.AIM")
    assert registered[registered.index("--full-mask") + 1] == str(tmp_path / "full.AIM")
    assert registered[registered.index("--common-region") + 1] == str(tmp_path / "mask.nii.gz")
    assert registered[registered.index("--native-output-dir") + 1] == str(
        tmp_path / "derivatives" / "Voidspace" / "sub-001" / "ses-001" / "xct" / "native" / "voi-radiusleft"
    )
    assert "/registered/" in registered[registered.index("--output-dir") + 1]


@pytest.mark.parametrize("profile", ["registered", "dynamic"])
def test_voidspace_batch_clips_maps_after_morphology(tmp_path: Path, monkeypatch, profile) -> None:
    sitk = pytest.importorskip("SimpleITK")
    module = _import_batch_processor_module(monkeypatch)
    yy, xx = np.indices((65, 65))
    radius = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
    full = np.broadcast_to(radius < 25, (31, 65, 65)).copy()
    bone = full & np.broadcast_to(radius >= 15, full.shape)
    common = np.zeros(full.shape, dtype=bool)
    common[5:26] = True
    for name, array in (("seg", bone), ("full", full), ("common", common)):
        image = sitk.GetImageFromArray(array.astype(np.uint8))
        image.SetSpacing((.2, .2, .2))
        sitk.WriteImage(image, str(tmp_path / f"{name}.nii.gz"))
    row = dict(subject="001", session_value="001", voi_value="radiusleft",
               seg_path=str(tmp_path / "seg.nii.gz"), full_mask_path=str(tmp_path / "full.nii.gz"),
               common_region_path=str(tmp_path / "common.nii.gz"))
    if profile == "dynamic":
        for timepoint in ("baseline", "followup"):
            row[f"{timepoint}_seg_path"] = row["seg_path"]
            row[f"{timepoint}_full_mask_path"] = row["full_mask_path"]
            row[f"{timepoint}_output_dir"] = str(tmp_path / timepoint)
    logic = module.BatchProcessorLogic()
    command = logic.command_for_row(tmp_path, tool="voidspace", profile=profile, row=row, force=True)
    subprocess.run([sys.executable, *command], check=True, env={**os.environ, "PYTHONPATH": str(ROOT.parent / "voidspace/src")})
    directory = Path(row["baseline_output_dir"]) if profile == "dynamic" else logic._voidspace_output_dir_for_row(tmp_path, profile, row)
    all_void = sitk.GetArrayFromImage(sitk.ReadImage(str(directory / "voidspace_all_mask.nii.gz")))
    large_void = sitk.GetArrayFromImage(sitk.ReadImage(str(directory / "voidspace_large_mask.nii.gz")))
    assert all_void[5:26, 32, 32].all(), "Common-region clipping changed morphology near its boundary"
    assert not all_void[:5].any() and not all_void[26:].any()
    assert not np.any((large_void > 0) & (all_void == 0)), "All voidspace must contain large voidspace"


def test_dynamic_voidspace_command_runs_registered_voidspace_before_compare(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    logic = module.BatchProcessorLogic()
    row = {
        "subject": "001",
        "session": "001-002",
        "session_value": "001-002",
        "voi": "radiusleft",
        "voi_value": "radiusleft",
        "baseline_seg_path": str(tmp_path / "ses-001" / "seg.nii.gz"),
        "baseline_full_mask_path": str(tmp_path / "ses-001" / "full.nii.gz"),
        "baseline_output_dir": str(tmp_path / "dynamic" / "baseline"),
        "followup_seg_path": str(tmp_path / "ses-002" / "seg.nii.gz"),
        "followup_full_mask_path": str(tmp_path / "ses-002" / "full.nii.gz"),
        "followup_output_dir": str(tmp_path / "dynamic" / "followup"),
        "common_region_path": str(tmp_path / "common.nii.gz"),
    }

    command = logic.command_for_row(tmp_path, tool="voidspace", profile="dynamic", row=row, force=True)

    assert command[0] == "-c"
    assert "from voidspace import compare, run_case" in command[1]
    assert "from voidspace import intersect_masks" in command[1]
    assert 'mask_path=Path(args.baseline_output_dir) / "voidspace_analysis_mask.nii.gz"' in command[1]
    assert command[command.index("--baseline-seg") + 1] == str(tmp_path / "ses-001" / "seg.nii.gz")
    assert command[command.index("--baseline-full-mask") + 1] == str(tmp_path / "ses-001" / "full.nii.gz")
    assert command[command.index("--followup-seg") + 1] == str(tmp_path / "ses-002" / "seg.nii.gz")
    assert command[command.index("--followup-full-mask") + 1] == str(tmp_path / "ses-002" / "full.nii.gz")
    assert command[command.index("--common-region") + 1] == str(tmp_path / "common.nii.gz")
    assert command[command.index("--baseline-output-dir") + 1] == str(tmp_path / "dynamic" / "baseline")
    assert command[command.index("--followup-output-dir") + 1] == str(tmp_path / "dynamic" / "followup")
    assert command[-1] == "--force"


def test_masked_voidspace_profiles_group_timepoints_with_one_visible_run_button(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    common_records = []
    for session in ("001", "002"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (("bone_segmentation", "seg"), ("periosteal_mask", "full")):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        common = (
            tmp_path
            / "derivatives"
            / "CommonRegion"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / f"sub-001_ses-{session}_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz"
        )
        common.parent.mkdir(parents=True)
        common.write_bytes(b"")
        common_records.append(
            DerivativeRecord(
                "CommonRegion",
                "scan_region_native_common",
                "001",
                "radiusleft",
                session,
                None,
                "native",
                common,
                "generated",
                content_type="mask",
            )
        )
        transformed_dir = (
            tmp_path
            / "derivatives"
            / "Timelapse"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / "transformed"
        )
        transformed_dir.mkdir(parents=True)
        (transformed_dir / f"sub-001_ses-{session}_voi-radiusleft_image-fused.nii.gz").write_bytes(b"")
        (transformed_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-seg_mask-fused.nii.gz").write_bytes(b"")
        (transformed_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-full_mask-fused.nii.gz").write_bytes(b"")
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=common_records),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    logic = module.BatchProcessorLogic()
    rows, _message = logic.discover_rows(
        tmp_path,
        tool="voidspace",
        profile="registered",
        registered=True,
    )

    assert [row["action"] for row in rows] == ["Run", ""]
    assert rows[0]["action_row_span"] == 2
    assert rows[1]["action_row_span"] == 0
    assert all(row["registered"] for row in rows)
    assert "mask=sub-001_ses-001_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz" in rows[0]["input"]
    assert "registration=" not in rows[0]["input"]
    command = logic.command_for_row(tmp_path, tool="voidspace", profile="registered", row=rows[1])
    assert command[0] == "-c"
    assert command[command.index("--common-region") + 1] == str(
        tmp_path
        / "derivatives"
        / "CommonRegion"
        / "sub-001"
        / "ses-002"
        / "xct"
        / "sub-001_ses-002_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz"
    )

    assert command[command.index("--segmentation") + 1].endswith("BoneContours/sub-001/ses-002/xct/sub-001_ses-002_voi-radiusleft_desc-seg_mask.AIM")
    assert command[command.index("--full-mask") + 1].endswith("BoneContours/sub-001/ses-002/xct/sub-001_ses-002_voi-radiusleft_desc-full_mask.AIM")
    assert command[command.index("--native-output-dir") + 1].endswith("Voidspace/sub-001/ses-002/xct/native/voi-radiusleft")
    assert "/registered/" in command[command.index("--output-dir") + 1]


def test_dynamic_voidspace_discovers_adjacent_registered_pairs(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    contour_records = []
    common_records = []
    pair_common = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "xct"
        / "analysis"
        / "common_regions"
        / "sub-001_voi-radiusleft_desc-full_common-alltimepoints.nii.gz"
    )
    pair_common.parent.mkdir(parents=True)
    pair_common.write_bytes(b"")
    pairwise_csv = pair_common.parent.parent / "sub-001_voi-radiusleft_pairwise_remodelling.csv"
    pairwise_csv.write_text(
        "subject_id,compartment,t0,t1,common_region_path,site\n"
        f"001,full,001,002,{pair_common},radiusleft\n"
        f"001,full,002,003,{pair_common},radiusleft\n",
        encoding="utf-8",
    )
    for session in ("001", "002", "003"):
        xct_dir = tmp_path / "sub-001" / f"ses-{session}" / "xct"
        xct_dir.mkdir(parents=True)
        (xct_dir / f"sub-001_ses-{session}_voi-radiusleft_xct.AIM").write_bytes(b"")
        contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / f"ses-{session}" / "xct"
        contour_dir.mkdir(parents=True)
        for role, filename_role in (("bone_segmentation", "seg"), ("periosteal_mask", "full")):
            mask = contour_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-{filename_role}_mask.AIM"
            mask.write_bytes(b"")
            contour_records.append(
                DerivativeRecord(
                    "BoneContours",
                    role,
                    "001",
                    "radiusleft",
                    session,
                    None,
                    "native",
                    mask,
                    "generated",
                    content_type="mask",
                )
            )
        native_common = (
            tmp_path
            / "derivatives"
            / "CommonRegion"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / f"sub-001_ses-{session}_voi-radiusleft_desc-scan-region-native-common_mask.nii.gz"
        )
        native_common.parent.mkdir(parents=True)
        native_common.write_bytes(b"")
        common_records.append(
            DerivativeRecord(
                "CommonRegion",
                "scan_region_native_common",
                "001",
                "radiusleft",
                session,
                None,
                "native",
                native_common,
                "generated",
                content_type="mask",
            )
        )
        transformed_dir = (
            tmp_path
            / "derivatives"
            / "Timelapse"
            / "sub-001"
            / f"ses-{session}"
            / "xct"
            / "transformed"
        )
        transformed_dir.mkdir(parents=True)
        (transformed_dir / f"sub-001_ses-{session}_voi-radiusleft_image-fused.nii.gz").write_bytes(b"")
        (transformed_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-seg_mask-fused.nii.gz").write_bytes(b"")
        (transformed_dir / f"sub-001_ses-{session}_voi-radiusleft_desc-full_mask-fused.nii.gz").write_bytes(b"")
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=contour_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=common_records),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="voidspace",
        profile="dynamic",
        registered=True,
    )

    assert [row["session"] for row in rows] == ["001-002", "002-003"]
    assert [row["action"] for row in rows] == ["Run", ""]
    assert rows[0]["baseline_seg_path"].endswith("ses-001/xct/transformed/sub-001_ses-001_voi-radiusleft_desc-seg_mask-fused.nii.gz")
    assert rows[0]["followup_seg_path"].endswith("ses-002/xct/transformed/sub-001_ses-002_voi-radiusleft_desc-seg_mask-fused.nii.gz")
    assert rows[1]["baseline_seg_path"].endswith("ses-002/xct/transformed/sub-001_ses-002_voi-radiusleft_desc-seg_mask-fused.nii.gz")
    assert rows[1]["followup_seg_path"].endswith("ses-003/xct/transformed/sub-001_ses-003_voi-radiusleft_desc-seg_mask-fused.nii.gz")
    assert rows[0]["baseline_output_dir"].endswith("Voidspace/sub-001/ses-001-002/xct/dynamic/voi-radiusleft/baseline")
    assert rows[0]["followup_output_dir"].endswith("Voidspace/sub-001/ses-001-002/xct/dynamic/voi-radiusleft/followup")
    assert rows[1]["baseline_output_dir"].endswith("Voidspace/sub-001/ses-002-003/xct/dynamic/voi-radiusleft/baseline")
    assert rows[1]["followup_output_dir"].endswith("Voidspace/sub-001/ses-002-003/xct/dynamic/voi-radiusleft/followup")
    assert "/registered/" not in rows[0]["baseline_output_dir"]
    assert "/registered/" not in rows[0]["followup_output_dir"]
    assert rows[0]["common_region_path"] == str(pair_common)
    assert rows[1]["common_region_path"] == str(pair_common)


def test_voidspace_batch_output_paths_are_deterministic_without_manifest(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    row = {
        "subject": "001",
        "session_value": "001",
        "voi_value": "radiusleft",
        "profile": "registered",
        "common_region_path": str(tmp_path / "mask.nii.gz"),
    }
    output_dir = (
        tmp_path
        / "derivatives"
        / "Voidspace"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "registered"
        / "voi-radiusleft"
    )
    output_dir.mkdir(parents=True)
    for name in ("voidspace_measurements.csv", "voidspace_large_mask.AIM", "voidspace_all_mask.AIM"):
        (output_dir / name).write_bytes(b"")

    paths = module.BatchProcessorLogic().rediscover_row_output_paths(tmp_path, "voidspace", row)

    assert paths == [
        str(output_dir / "voidspace_measurements.csv"),
        str(output_dir / "voidspace_all_mask.AIM"),
        str(output_dir / "voidspace_large_mask.AIM"),
    ]


def test_dynamic_voidspace_output_paths_are_deterministic_without_manifest(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    row = {
        "subject": "001",
        "session_value": "001-002",
        "voi_value": "radiusleft",
        "profile": "dynamic",
    }
    output_dir = (
        tmp_path
        / "derivatives"
        / "Voidspace"
        / "sub-001"
        / "ses-001-002"
        / "xct"
        / "dynamic"
        / "voi-radiusleft"
    )
    output_dir.mkdir(parents=True)
    for name in (
        "voidspace_change_measurements.csv",
        "voidspace_stable_mask.nii.gz",
        "voidspace_expanded_mask.nii.gz",
        "voidspace_contracted_mask.nii.gz",
    ):
        (output_dir / name).write_bytes(b"")

    paths = module.BatchProcessorLogic().rediscover_row_output_paths(tmp_path, "voidspace", row)

    assert paths == [
        str(output_dir / "voidspace_change_measurements.csv"),
        str(output_dir / "voidspace_stable_mask.nii.gz"),
        str(output_dir / "voidspace_expanded_mask.nii.gz"),
        str(output_dir / "voidspace_contracted_mask.nii.gz"),
    ]


def test_voidspace_discovery_marks_existing_outputs_loadable_without_manifest(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    (xct_dir / "sub-001_ses-001_voi-radiusleft_xct.AIM").write_bytes(b"")
    contour_dir = tmp_path / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
    contour_dir.mkdir(parents=True)
    seg = contour_dir / "sub-001_ses-001_voi-radiusleft_desc-seg_mask.AIM"
    seg.write_bytes(b"")
    write_manifest(
        DerivativeManifest.create(
            "BoneContours",
            tmp_path,
            {"name": "test", "version": "1"},
            records=[
                DerivativeRecord(
                    "BoneContours",
                    "bone_segmentation",
                    "001",
                    "radiusleft",
                    "001",
                    None,
                    "native",
                    seg,
                    "generated",
                    content_type="mask",
                )
            ],
        ),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )
    output_dir = (
        tmp_path
        / "derivatives"
        / "Voidspace"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "native"
        / "voi-radiusleft"
    )
    output_dir.mkdir(parents=True)
    for name in ("voidspace_measurements.csv", "voidspace_large_mask.AIM", "voidspace_all_mask.AIM"):
        (output_dir / name).write_bytes(b"")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="voidspace",
        profile="standard",
        registered=False,
    )

    assert rows[0]["action"] == "Load"
    assert rows[0]["status"] == "Done"
    assert rows[0]["output_paths"] == [
        str(output_dir / "voidspace_measurements.csv"),
        str(output_dir / "voidspace_all_mask.AIM"),
        str(output_dir / "voidspace_large_mask.AIM"),
    ]


def test_batch_processor_excludes_motion_scoring_from_shared_batch_selector() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert '"motion_scoring": (' not in source
    assert '("Motion Scoring", "motion_scoring")' not in source
    assert '"motion_scoring": "MotionScoring"' not in source


def test_bone_contouring_batch_profile_hint_explains_site_resolution() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "self.profileHintLabel = qt.QLabel()" in source
    assert "For Bone Contouring, radius/tibia/knee settings are selected automatically from each row's VOI." in source
    assert 'workflow_form.addRow("", self.profileHintLabel)' in source
    assert "workflow_layout.addWidget(self.profileHintLabel)" not in source
    assert "def _is_selected_profile_shipped(self):" in source
    assert 'self._selected_tool_key() == "bone_contouring" and self._is_selected_profile_shipped()' in source
    assert "self._update_profile_hint()" in source[source.index("    def _on_tool_changed(") :]
    assert "self._update_profile_hint()" in source[source.index("    def _on_profile_changed(") :]
    assert "self._update_profile_hint()" in source[source.index("    def _populate_profile_combo(") :]
    assert "self.profileCombo.toolTip = hint if show_hint else \"\"" in source


def test_batch_table_clears_old_spans_and_expands_input_on_double_click() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "self.table.clearSpans()" in source
    assert "self.table.cellDoubleClicked.connect(self._on_table_cell_double_clicked)" in source
    assert "def _on_table_cell_double_clicked(self, row, column):" in source
    assert "headers.index(\"Input\")" in source
    assert "self.table.resizeRowToContents(row)" in source
    assert "self.table.resizeColumnToContents(column)" in source
    assert "slicer.util.infoDisplay(item.text())" not in source
    assert "self.table.horizontalHeader().setStretchLastSection(True)" in source


def test_batch_row_buttons_start_process_queue_and_log_output() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "self._batchRows = []" in source
    assert "self._batchQueue = []" in source
    assert "self.batchLog = qt.QTextEdit()" in source
    assert "self.runAllButton.clicked.connect(self._queue_all_rows)" in source
    assert "button.clicked.connect(lambda _checked=False, index=row_index: self._on_row_action(index))" in source
    assert "def _start_next_batch_job(self):" in source
    assert "process = qt.QProcess()" in source
    assert "process.readyRead.connect(lambda process=process: self._append_process_output(process))" in source
    assert "process.finished.connect(" in source
    assert "self._append_log(" in source
    assert 'self._set_row_action(row_index, "Queued")' in source
    assert 'self._set_row_action(row_index, "Running")' in source
    assert 'self._set_row_action(row_index, "Load")' in source
    assert "def _set_row_action(self, row_index, action):" in source
    assert "def _clean_process_output(text: str) -> str:" in source
    assert "_SUPPRESSED_PROCESS_OUTPUT_MARKERS = (" in source
    assert "Error ImageIO factory did not return an ImageIOBase: MRMLIDImageIO" in source


def test_queued_batch_jobs_snapshot_tool_profile_and_row() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _batch_job_for_row(self, row_index):" in source
    assert '"tool": self._selected_tool_key()' in source
    assert '"profile": str(self.profileCombo.currentData or "")' in source
    assert 'row = dict(self._batchRows[row_index])' in source
    assert '"row": row' in source
    assert "job = self._batch_job_for_row(child_index)" in source
    assert "self._batchQueue.append(job)" in source
    assert "job = self._batchQueue.pop(0)" in source
    assert 'tool=str(job.get("tool") or "")' in source
    assert 'profile=str(job.get("profile") or "")' in source
    assert 'row=dict(job.get("row") or {})' in source
    assert 'if self._has_active_batch():' in source
    assert 'self._append_log("[batch] Tool/profile change will not affect already queued jobs.")' in source
    finish_handler = source[
        source.index("    def _batch_process_finished(") : source.index("    def _refresh_row_output_paths(", source.index("    def _batch_process_finished("))
    ]
    assert 'if self._selected_tool_key() == "fea"' not in finish_handler


def test_queued_jobs_from_different_tools_do_not_collide_by_visible_row_index(monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    widget = object.__new__(module.BatchProcessorWidget)
    row = {"subject": "001", "session": "001", "voi": "radius", "stack_index": "1"}
    queued_job = {
        "row_index": 2,
        "tool": "bone_contouring",
        "profile": "XtremeCTI",
        "backend": "local",
        "local_root": "/dataset",
        "remote_root": "",
        "row": row,
    }
    widget._batchQueue = [queued_job]
    widget._batchRows = [
        {"action": "Missing"},
        {"action": "Missing"},
        {**row, "action": "Run"},
    ]
    widget.toolCombo = types.SimpleNamespace(currentData="microarchitecture")
    widget.profileCombo = types.SimpleNamespace(currentData="standard")
    widget.skipExistingCheck = types.SimpleNamespace(checked=True)
    widget.cancelBatchButton = types.SimpleNamespace(enabled=False)
    widget._selected_backend_key = lambda: "local"
    widget._current_local_dataset_root = lambda: "/dataset"
    widget._current_remote_dataset_root = lambda: ""
    widget._effective_row_action = lambda candidate: candidate.get("action")
    widget._set_row_status = lambda index, status: widget._batchRows[index].update(status=status)
    widget._set_row_action = lambda index, action: widget._batchRows[index].update(action=action)
    widget._append_log = lambda _message: None
    widget._start_next_batch_job = lambda: None

    widget._queue_all_rows()

    assert len(widget._batchQueue) == 2
    assert widget._batchQueue[1]["row_index"] == 2
    assert widget._batchQueue[1]["tool"] == "microarchitecture"


def test_grouped_voidspace_finish_updates_visible_group_action() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    finish_handler = source[
        source.index("    def _batch_process_finished(") : source.index("    def _mark_remote_job_submitted(", source.index("    def _batch_process_finished("))
    ]

    assert "self._set_group_action_load_if_outputs_are_ready(row_index, tool_key)" in finish_handler
    assert "def _set_group_action_load_if_outputs_are_ready(self, row_index, tool_key):" in source
    assert 'if str(tool_key or "") != "voidspace":' in source
    assert "for index in self._row_indices_for_group_action(visible_index):" in source
    assert 'self._set_row_action(visible_index, "Load")' in source


def test_active_batches_allow_selector_changes_but_jobs_keep_snapshots() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _job_description(self, job):" in source
    assert 'job.get("tool")' in source
    assert 'job.get("profile")' in source
    assert "Tool/profile change will not affect already queued jobs." in source
    assert "self._update_table_headers()" in source
    assert 'tool=str(job.get("tool") or "")' in source
    assert 'profile=str(job.get("profile") or "")' in source


def test_batch_processor_populates_backend_combo_from_registry() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "available_batch_backends()" in source
    assert 'self.backendCombo.addItem(backend.label, backend.key)' in source
    assert 'server_selected = self._selected_backend_key() != "local"' in source
    assert 'len(self._batchBackends) > 1 or self._serverBackendEnabled' in source
    assert 'if backend_key != "local" and remote_backend is not None:' in source


def test_batch_processor_enables_server_when_private_backend_registered() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'self._serverBackendEnabled = "server" in self._batchBackends or os.environ.get(SLICER_BONE_BATCH_BACKEND, "")' in source
    assert "self._batchBackends = self._available_batch_backends()" in source


def test_timelapse_outputs_are_discovered_as_series_outputs(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    logic = module.BatchProcessorLogic()
    analysis_dir = tmp_path / "derivatives" / "Timelapse" / "sub-001" / "xct" / "analysis"
    visualize_dir = analysis_dir / "visualize"
    visualize_dir.mkdir(parents=True)
    table = analysis_dir / "sub-001_voi-radiusleft_pairwise_remodelling.csv"
    map_1 = visualize_dir / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-5_remodelling.nii.gz"
    map_2 = visualize_dir / "sub-001_voi-radiusleft_desc-roi_union_t0-002_t1-003_thr-225p0_cluster-5_remodelling.nii.gz"
    for path in (table, map_1, map_2):
        path.write_text("", encoding="utf-8")

    row = {"subject": "001", "session": "001", "session_value": "001", "voi": "radiusleft", "voi_value": "radiusleft"}

    assert logic.rediscover_row_output_paths(tmp_path, "timelapse", row) == [
        str(table),
        str(map_1),
        str(map_2),
    ]


def test_timelapse_group_loads_when_pairwise_outputs_exist(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    logic = module.BatchProcessorLogic()
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    image_1 = xct_dir / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    image_1.write_bytes(b"")
    xct_dir_2 = tmp_path / "sub-001" / "ses-002" / "xct"
    xct_dir_2.mkdir(parents=True)
    image_2 = xct_dir_2 / "sub-001_ses-002_voi-radiusleft_xct.AIM"
    image_2.write_bytes(b"")
    analysis_dir = tmp_path / "derivatives" / "Timelapse" / "sub-001" / "xct" / "analysis"
    visualize_dir = analysis_dir / "visualize"
    visualize_dir.mkdir(parents=True)
    table = analysis_dir / "sub-001_voi-radiusleft_pairwise_remodelling.csv"
    table.write_text("", encoding="utf-8")
    remodelling = visualize_dir / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-5_remodelling.nii.gz"
    remodelling.write_text("", encoding="utf-8")

    image_records = module.discover_raw_xct_images(tmp_path)
    existing = logic._discover_existing_outputs(tmp_path, "Timelapse")
    rows = logic._table_rows_for_tool(
        tmp_path,
        image_records,
        contour_artifacts=(),
        registration_records=(),
        common_region_records=(),
        existing_outputs=existing,
        tool="timelapse",
        profile="standard",
        registered=True,
    )

    assert rows[0]["action"] == "Load"
    assert str(table) in rows[0]["output_paths"]
    assert str(remodelling) in rows[0]["output_paths"]


def test_timelapse_batch_loader_uses_timelapsed_remodelling_style() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'if self._selected_tool_key() == "timelapse":' in source
    assert "def _load_timelapse_outputs(self, row, output_paths):" in source
    assert "def _load_timelapse_summary_table(self, csv_path: Path, row):" in source
    assert "def _table_node_from_csv(csv_path: Path, name: str):" in source
    assert "def _show_table_node(self, table_node):" in source
    assert "slicer.mrmlScene.AddNewNodeByClass(\"vtkMRMLTableNode\", name)" in source
    assert "vtk.vtkStringArray()" in source
    assert "self._show_table_node(table_node)" in source
    assert "GetLayoutWithTable" in source
    assert "SetActiveTableID(table_node.GetID())" in source
    assert "PropagateTableSelection()" in source
    assert "slicer.util.loadTable(str(summary_path)" not in source
    assert "def _write_timelapse_summary_csv(csv_path: Path, row) -> Path:" in source
    assert 'headers = ["Sample", "Pair", "Profile", "ROI", "FV/BV", "RV/BV", "AV/BV", "NV/BV"]' in source
    assert "def _style_timelapse_remodelling_volume(node, path):" in source
    assert "def _remodelling_color_node():" in source
    assert "Timelapse_RemodellingColors" in source
    assert '1: ("resorption", 1.00, 0.05, 0.70, 1.0)' in source
    assert '2: ("quiescent", 0.62, 0.62, 0.62, 0.32)' in source
    assert '3: ("formation", 1.00, 0.48, 0.00, 1.0)' in source
    assert "display_node.SetWindowLevel(5.0, 2.5)" in source
    assert "*.AIM" in source
    assert "NamesInitialisedOn" not in source


def test_timelapse_summary_table_keeps_only_core_columns(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    csv_path = tmp_path / "sub-001_voi-radiusleft_pairwise_remodelling.csv"
    csv_path.write_text(
        "subject_id,site,compartment,t0,t1,threshold,cluster_min_size,common_region_path,"
        "formation_frac_bv0,resorption_frac_bv0,profile\n"
        "001,radiusleft,full,001,002,225,5,/tmp/common.nii.gz,0.01,0.03,xct1-standard\n",
        encoding="utf-8",
    )

    summary_path = module.BatchProcessorWidget._write_timelapse_summary_csv(
        csv_path,
        {"subject": "001", "voi_value": "radiusleft"},
    )
    text = Path(summary_path).read_text(encoding="utf-8")
    try:
        assert text.splitlines()[0] == "Sample,Pair,Profile,ROI,FV/BV,RV/BV,AV/BV,NV/BV"
        assert "sub-001 voi-radiusleft,001-002,xct1-standard,full,0.01,0.03,0.04,-0.02" in text
        assert "common_region_path" not in text
        assert "threshold" not in text
        assert "cluster_min_size" not in text
        assert "SlicerBoneImagingToolbox/BatchProcessor/tables" in str(summary_path)
    finally:
        Path(summary_path).unlink(missing_ok=True)


def test_timelapse_batch_load_groups_tables_and_maps_in_scene_folder() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    load_timelapse = source.split("    def _load_timelapse_outputs", 1)[1].split("\n    def ", 1)[0]

    assert "folder_name = self._timelapse_folder_name(row)" in load_timelapse
    assert "self._load_timelapse_summary_table(path, row)" in load_timelapse
    assert "self._put_node_in_subject_hierarchy_folder(node, folder_name)" in load_timelapse
    assert 'return f"sub-{subject}_voi-{voi}_timelapse-remodelling"' in source


def test_microarchitecture_batch_load_groups_tables_and_maps_in_scene_folder() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    load_outputs = source.split("    def _load_row_outputs", 1)[1].split("\n    def _load_timelapse_outputs", 1)[0]
    refresh_outputs = source.split("    def _refresh_row_output_paths", 1)[1].split("\n    def _load_row_outputs", 1)[0]

    assert 'self._selected_tool_key() == "microarchitecture"' in load_outputs
    assert 'tool_key in {"microarchitecture", "plate_rod", "voidspace"}' in refresh_outputs
    assert 'int(row.get("action_row_span") or 0) > 1' in refresh_outputs
    assert "for offset in range(int(row.get(\"action_row_span\") or 0)):" in refresh_outputs
    assert "if is_table and node is not None:" in load_outputs
    assert "self._show_table_node(node)" in load_outputs
    assert "elif self._selected_tool_key() == \"microarchitecture\" and node is not None:" in load_outputs
    assert "folder_name = self._microarchitecture_map_folder_name(path, row)" in load_outputs
    assert "self._style_microarchitecture_volume(node, path)" in load_outputs
    assert "self._put_node_in_subject_hierarchy_folder(node, folder_name)" in load_outputs
    assert 'self._selected_tool_key() in {"microarchitecture", "plate_rod"} and bool(row.get("registered"))' in load_outputs
    assert "self._load_registered_common_region_overlays(row_index)" in load_outputs


def test_plate_rod_batch_loads_npy_maps_and_groups_outputs_in_scene_folder() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    load_outputs = source.split("    def _load_row_outputs", 1)[1].split("\n    def _load_timelapse_outputs", 1)[0]

    assert "def _load_plate_rod_npy_map(path: Path):" in source
    assert "def _plate_rod_color_node():" in source
    assert '"PlateRodMorphometry_Colors"' in source
    assert '1: ("plate", 0.10, 0.45, 1.00, 1.0)' in source
    assert '2: ("rod", 1.00, 0.10, 0.08, 1.0)' in source
    assert "vtkMRMLColorTableNodeFileGenericAnatomyColors" not in source
    assert 'self._selected_tool_key() == "plate_rod" and path.suffix.lower() == ".npy"' in load_outputs
    assert "self._style_plate_rod_volume(node, path)" in load_outputs
    assert "self._put_node_in_subject_hierarchy_folder(node, self._plate_rod_output_folder_name(path, row))" in load_outputs
    assert 'suffix = "_xct_registered_plate-rod" if "/registered_measurements/" in path_text else "_xct_plate-rod"' in source
    assert "def _load_registered_common_region_overlays(self, row_index):" in source
    assert "def _load_common_region_outputs_as_segmentation(self, row, output_paths):" in source
    assert '"common_region": (0.72, 0.42, 1.0)' in source


def test_voidspace_batch_loads_maps_through_aim_safe_mask_loader() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    load_outputs = source.split("    def _load_row_outputs", 1)[1].split("\n    def _load_fea_outputs", 1)[0]

    assert 'if self._selected_tool_key() == "voidspace":' in load_outputs
    assert 'if self._selected_tool_key() == "voidspace" and int(row.get("action_row_span") or 0) > 1:' in load_outputs
    assert 'if str(row.get("profile") or "").strip() == "registered":' in load_outputs
    assert "dataset_root = self._current_local_dataset_root()" in load_outputs
    assert "for child_index in self._row_indices_for_group_action(row_index):" in load_outputs
    assert "self.logic.rediscover_row_output_paths(" in load_outputs
    assert "self._refresh_row_output_paths(child_index)" not in load_outputs
    assert 'child_row["output_paths"] = [str(path) for path in child_paths]' in load_outputs
    assert "self._load_voidspace_outputs(child_row, child_paths)" in load_outputs
    assert "output_paths = [Path(path) for path in self._refresh_row_output_paths(row_index)]" in load_outputs
    assert "self._load_voidspace_outputs(row, output_paths)" in load_outputs
    assert "def _load_voidspace_outputs(self, row, output_paths):" in source
    assert 'seg_path_text = str(row.get("seg_path") or "").strip()' in source
    assert 'segmentation_inputs = [(seg_path, "seg")] if seg_path is not None and seg_path.exists() else []' in source
    assert '("baseline_seg_path", "baseline_seg")' in source
    assert '("followup_seg_path", "followup_seg")' in source
    assert '("baseline_output_dir", "baseline_analysis_mask")' in source
    assert '("followup_output_dir", "followup_analysis_mask")' in source
    assert "if not segmentation_inputs and not analysis_inputs and not voidspace_inputs:" in source
    assert "for path, role in [*segmentation_inputs, *analysis_inputs, *voidspace_inputs]:" in source
    assert "self._load_mask_as_labelmap(path, role, reference_node)" in source
    assert '"seg": "Bone segmentation"' in source
    assert '"baseline_seg": "Baseline bone segmentation"' in source
    assert '"followup_seg": "Follow-up bone segmentation"' in source
    assert '"baseline_analysis_mask": "Baseline analysis mask"' in source
    assert '"followup_analysis_mask": "Follow-up analysis mask"' in source
    assert '"voidspace_quiescent": "Quiescent voidspace"' in source
    assert '"voidspace_expanded": "Expanded voidspace"' in source
    assert '"voidspace_contracted": "Contracted voidspace"' in source
    assert '"seg": (0.45, 0.45, 0.45)' in source
    assert '"voidspace_quiescent": (0.78, 0.78, 0.78)' in source
    assert '"voidspace_expanded": (0.0, 0.68, 0.78)' in source
    assert '"voidspace_contracted": (0.92, 0.78, 0.18)' in source
    assert '"voidspace_analysis": (0.45, 0.70, 0.65)' in source
    assert '"baseline_analysis_mask": (0.50, 0.72, 0.68)' in source
    assert '"followup_analysis_mask": (0.35, 0.62, 0.74)' in source
    assert '"voidspace_stable_mask" in name' in source
    assert '"voidspace_analysis_mask" in name' in source
    assert "before_count = segmentation_node.GetSegmentation().GetNumberOfSegments()" in source
    assert "after_count = segmentation_node.GetSegmentation().GetNumberOfSegments()" in source
    assert "if after_count <= before_count:" in source
    assert "Skipping empty voidspace mask" in source
    assert "voidspace_large_mask" in source
    assert "voidspace_all_mask" in source
    assert 'if profile != "registered":' in source


def test_batch_processor_finds_common_region_paths_for_registered_load(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    image_dir = tmp_path / "sub-001" / "ses-002" / "xct"
    image_dir.mkdir(parents=True)
    image = image_dir / "sub-001_ses-002_voi-radiusleft_xct.AIM"
    image.write_bytes(b"")
    common = (
        tmp_path
        / "derivatives"
        / "CommonRegion"
        / "sub-001"
        / "ses-002"
        / "xct"
        / "masks"
        / "sub-001_ses-002_voi-radiusleft_mask-scan-region_native_common.nii.gz"
    )
    common.parent.mkdir(parents=True)
    common.write_bytes(b"")
    write_manifest(
        DerivativeManifest.create(
            "CommonRegion",
            tmp_path,
            {"name": "test", "version": "1"},
            records=[
                DerivativeRecord(
                    "CommonRegion",
                    "scan_region_native_common",
                    "001",
                    "radiusleft",
                    "002",
                    None,
                    "native",
                    common,
                    "generated",
                    content_type="mask",
                )
            ],
        ),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    paths = module.BatchProcessorLogic().common_region_paths_for_row(
        tmp_path,
        {
            "subject": "001",
            "session": "002",
            "voi_value": "radiusleft",
            "stack_index": None,
            "registered": True,
        },
    )

    assert paths == [str(common)]


def test_plate_rod_batch_requests_metal_backend_on_macos() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    process_environment = source.split("    def _process_environment", 1)[1].split("\n    def _subprocess_args", 1)[0]

    assert 'self._selected_tool_key() == "plate_rod" and sys.platform == "darwin"' in process_environment
    assert 'environment.insert("PLATE_ROD_USE_METAL", "1")' in process_environment
    assert 'environment.insert("PLATE_ROD_USE_METAL_FULL", "1")' in process_environment


def test_batch_processor_exposes_cancel_button_for_running_jobs() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'self.cancelBatchButton = qt.QPushButton("Cancel")' in source
    assert "self.cancelBatchButton.clicked.connect(self._cancel_batch)" in source
    assert "def _cancel_batch(self):" in source
    assert "process.terminate()" in source
    assert "process.kill()" in source


def test_skip_existing_off_makes_loadable_rows_runnable() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "self.skipExistingCheck.toggled.connect(self._on_skip_existing_toggled)" in source
    assert "def _effective_row_action(self, row):" in source
    assert 'if action == "Load" and not bool(self.skipExistingCheck.checked):' in source
    assert 'return "Run"' in source
    assert "action = self._effective_row_action(row)" in source
    assert 'if self._effective_row_action(row) != "Run":' in source
    assert '"force": not bool(self.skipExistingCheck.checked)' in source
    assert 'force=bool(job.get("force"))' in source


def test_bone_contour_outputs_load_as_segmentation_nodes() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _load_bone_contour_outputs_as_segmentation(self, row, output_paths):" in source
    assert "self._current_tool()" not in source
    assert 'self._selected_tool_key() == "bone_contouring"' in source
    assert "vtkMRMLSegmentationNode" in source
    assert "ImportLabelmapToSegmentationNode" in source
    assert "SetReferenceImageGeometryParameterFromVolumeNode" in source
    assert "slicer.util.loadLabelVolume" in source
    assert 'row.get("image_path")' in source
    assert '"image_path": str(record.path)' in source
    assert "def _ensure_loaded_source_volume(self, image_path, *, require_source_path=False):" in source
    assert 'ScancoIOLogic().import_image(image_path, scaling="density"' in source


def test_bone_contour_loader_skips_material_label_outputs(monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    widget = module.BatchProcessorWidget

    label_path = Path("sub-SAMPLE341_ses-001_voi-tibia_desc-fea-input_label.AIM")
    mask_path = Path("sub-SAMPLE341_ses-001_voi-tibia_desc-full_mask.AIM")
    fea_input_path = Path("sub-SAMPLE341_ses-001_voi-tibia_desc-hom-ls-model_label.AIM")

    assert not widget._is_bone_contour_segmentation_output(label_path)
    assert not widget._is_bone_contour_segmentation_output(fea_input_path)
    assert widget._is_bone_contour_segmentation_output(mask_path)
    assert widget._mask_role_from_path(label_path) == "fea-input"

    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "No BoneContours mask/label outputs were discovered for this row." in source
    assert "if self._is_label_output(path):" in source
    assert "def _is_fea_material_role(role: str) -> bool:" in source


def test_mask_label_algebra_loads_imported_contours_not_generated_label(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    imported_dir = tmp_path / "derivatives" / "ImportedContours" / "sub-SAMPLE341" / "ses-001" / "xct"
    generated_dir = tmp_path / "derivatives" / "BoneContours" / "sub-SAMPLE341" / "ses-001" / "xct"
    imported_dir.mkdir(parents=True)
    generated_dir.mkdir(parents=True)
    imported_full = imported_dir / "sub-SAMPLE341_ses-001_voi-tibia_desc-full_mask.AIM"
    imported_trab = imported_dir / "sub-SAMPLE341_ses-001_voi-tibia_desc-trab_mask.AIM"
    generated_label = generated_dir / "sub-SAMPLE341_ses-001_voi-tibia_desc-fea-input_label.AIM"
    for path in (imported_full, imported_trab, generated_label):
        path.write_bytes(b"")

    imported_records = [
        DerivativeRecord(
            "ImportedContours",
            role,
            "SAMPLE341",
            "tibia",
            "001",
            None,
            "native",
            path,
            "provided",
            content_type="mask",
        )
        for role, path in (
            ("periosteal_mask", imported_full),
            ("trabecular_mask", imported_trab),
        )
    ]
    generated_records = [
        DerivativeRecord(
            "BoneContours",
            "material_labelmap",
            "SAMPLE341",
            "tibia",
            "001",
            None,
            "native",
            generated_label,
            "derived",
            metadata={"short_role": "fea-input", "workflow": "mask_label_algebra"},
            content_type="label",
        )
    ]
    write_manifest(
        DerivativeManifest.create("ImportedContours", tmp_path, {"name": "test", "version": "1"}, records=imported_records),
        tmp_path / "derivatives" / "ImportedContours" / "manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("BoneContours", tmp_path, {"name": "test", "version": "1"}, records=generated_records),
        tmp_path / "derivatives" / "BoneContours" / "manifest.json",
    )

    paths = module.BatchProcessorLogic().rediscover_row_output_paths(
        tmp_path,
        "mask_label_algebra",
        {"subject": "SAMPLE341", "session_value": "001", "voi_value": "tibia"},
    )

    assert paths == sorted([str(imported_full), str(imported_trab)])
    assert str(generated_label) not in paths


def test_batch_processor_module_does_not_expose_legacy_layout_terms() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "RegisteredMicroarchitecture" not in source
    assert "native_space" not in source
    assert "TimelapsedHRpQCT" not in source


def test_individual_modules_do_not_expose_duplicate_batch_tabs() -> None:
    repo = Path(__file__).resolve().parents[1]
    module_paths = [
        repo / "HRpQCTTools" / "BoneMicroarchitecture" / "BoneMicroarchitecture.py",
        repo / "HRpQCTTools" / "PlateRodMorphometryHRpQCT" / "PlateRodMorphometryHRpQCT.py",
        repo / "HRpQCTTools" / "MechanoregulationHRpQCT" / "MechanoregulationHRpQCT.py",
        repo / "HRpQCTTools" / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py",
        repo / "HRpQCTTools" / "ParOSolFEA" / "ParOSolFEA.py",
        repo / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py",
        repo / "CTTools" / "SpineSegmentationCT" / "SpineSegmentationCT.py",
    ]

    for path in module_paths:
        source = path.read_text(encoding="utf-8")
        assert '.addTab(batch_tab, "Batch")' not in source, path
        assert '.addTab(batchPage, "Batch")' not in source, path
        assert '.addTab(self.batchPage, "Batch")' not in source, path
        assert '.addTab(self.batchModePage, "Batch")' not in source, path
        if path.name not in {"TimelapsedHRpQCT.py"}:
            assert "batch_tab = qt.QWidget()" not in source, path
            assert "self.batchPage = qt.QWidget()" not in source, path
            assert "self.batchPage = qt.QScrollArea()" not in source, path
            assert "self._build_batch_tab()" not in source, path
