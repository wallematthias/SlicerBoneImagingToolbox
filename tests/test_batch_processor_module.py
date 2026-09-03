from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path
import sys
import types

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, read_manifest, write_manifest


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
    assert "preferred_contours" in source
    assert "prerequisite_status" in source
    assert "def normalized_dataset_status(" in source
    assert 'root.glob("sub-*")' in source
    assert 'xct_dir = session_dir / "xct"' in source
    assert "Dataset Naming Helper" in source
    assert "Execution backend" in source
    assert '("Bone Contouring", "bone_contouring")' in source
    assert '"bone_contouring": "BoneContours"' in source
    assert "Local" in source
    assert "Server" in source
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
    assert 'button.enabled = action in {"Run", "Load"}' in source
    assert "self.table.setSpan(row_index, 0, span, 1)" in source
    assert "def _table_rows_for_tool(" in source
    assert "def command_for_row(" in source
    assert "def _subprocess_args(self, args):" in source
    assert '"timelapsedhrpqct.cli": TOOLBOX_ROOT.parent / ("Timelapsed" + "HRpQCT") / "src"' in source
    assert "from {module} import main" in source
    assert 'process_args = self._subprocess_args(args)' in source
    assert "process.start(self._python_slicer_executable(), process_args)" in source
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
    assert '"seg": (1.0, 0.95, 0.3)' in source
    assert "segment.SetColor(color[0], color[1], color[2])" in source


def test_batch_processor_fea_profiles_are_labelmap_shortcuts() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert '("XtremeCT I", "XtremeCTI", False)' in source
    assert '("XtremeCT II", "XtremeCTII", False)' in source
    assert '("Load history 3", "load_history_3", False)' in source
    assert '("Load history 6", "load_history_6", False)' in source
    assert "discover_fea_batch_cases" in source
    assert "build_parosol_case_commands" in source
    assert '"parosol_py.cli": TOOLBOX_ROOT.parent / "parosol-py" / "src"' in source


def test_batch_processor_fea_and_mechanoregulation_use_low_blue_high_red_sed_colormap() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "BatchProcessor_FEA_SED_JET" in source
    assert "display_node.SetAndObserveColorNodeID(color_node.GetID())" in source
    assert "vtkMRMLColorTableNodeRainbow" not in source
    assert "vtkMRMLColorTableNodeFileColdToHotRainbow.txt" not in source
    assert "abs(4.0 * t - 3.0)" in source
    assert "abs(4.0 * t - 1.0)" in source
    assert 'if role == "sed":\n                    self._style_fea_volume(node, path)' in source


def test_batch_processor_mechanoregulation_loads_events_as_sed_linked_segmentation() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "def _load_mechanoregulation_remodelling_as_segmentation(" in source
    assert "def _load_binary_event_labelmap(" in source
    assert '("resorption", "Resorption", np.isin(remodelling_array, (1,)))' in source
    assert '("formation", "Formation", np.isin(remodelling_array, (3, 4)))' in source
    assert '"quiescent"' not in source[source.index("def _load_mechanoregulation_remodelling_as_segmentation("):source.index("def _load_timelapse_outputs(")]
    assert 'segmentation_node.SetAttribute("BoneImaging.Mechanoregulation.RemodellingSource", str(path))' in source
    assert 'segmentation_node.SetAttribute("BoneImaging.Mechanoregulation.SEDNodeID", sed_node.GetID())' in source
    assert "slicer.util.setSliceViewerLayers(background=sed_node, fit=False)" in source


def test_batch_processor_fea_rows_use_hom_ls_material_sources(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    raw = xct_dir / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    hom_ls = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-HOM_LS_map.AIM"
    raw.write_bytes(b"")
    hom_ls.write_bytes(b"")

    rows, _message = module.BatchProcessorLogic().discover_rows(
        tmp_path,
        tool="fea",
        profile="XtremeCTII",
        registered=False,
    )

    assert len(rows) == 1
    assert rows[0]["action"] == "Run"
    assert rows[0]["status"] == "Ready"
    assert rows[0]["image_path"] == str(hom_ls)
    assert rows[0]["input"] == f"source={hom_ls.name}"
    assert raw.name not in rows[0]["input"]

    command = module.BatchProcessorLogic().command_for_row(
        tmp_path,
        tool="fea",
        profile="XtremeCTII",
        row=rows[0],
        force=False,
    )

    assert command[:3] == ["-m", "parosol_py.cli", str(hom_ls)]
    assert command[3:5] == ["--profile", "XtremeCTII"]
    assert "--session" not in command
    assert "--site" in command


def test_batch_processor_fea_publishes_canonical_sed_and_summary(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    xct_dir = tmp_path / "sub-001" / "ses-001" / "xct"
    xct_dir.mkdir(parents=True)
    hom_ls = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-HOM_LS_map.AIM"
    hom_ls.write_bytes(b"")
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
    hom_ls = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-HOM_LS_map.AIM"
    hom_ls.write_bytes(b"")
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
    hom_ls = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-HOM_LS_map.AIM"
    hom_ls.write_bytes(b"")
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
    hom_ls = xct_dir / "sub-001_ses-001_voi-radiusleft_desc-HOM_LS_map.AIM"
    hom_ls.write_bytes(b"")
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
        first_artifact=lambda roles: types.SimpleNamespace(path=tmp_path / "hom_ls.AIM"),
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
    for path in (csv_path, curves, schulte, summary):
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
    assert rows[0]["output_paths"] == [str(csv_path), str(curves), str(schulte), str(summary)]


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


def test_batch_processor_includes_saved_bone_contouring_profiles() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "list_profiles" in source
    assert 'list_profiles("bone-contouring")' in source
    assert "profiles.append((record.name, record.name, False))" in source


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


def test_loadable_bone_contouring_rows_include_output_paths(tmp_path: Path, monkeypatch) -> None:
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

    assert rows[0]["action"] == "Load"
    assert rows[0]["output_paths"] == [str(output)]


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
        metadata={"use_common_region": False},
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
        metadata={"use_common_region": True},
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
    assert "self.profileCombo.clear()" in source
    assert 'for label, value, _registered in self._profiles_for_tool(tool):' in source
    assert "return self.logic.profile_requests_registration(" in source
    assert 'args.append("--no-common-region")' in source
    assert 'args.append("--require-common-region")' in source
    assert 'if tool == "timelapse" and profile_value:' in source


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
    assert '"row": dict(self._batchRows[row_index])' in source
    assert "self._batchQueue.append(self._batch_job_for_row(row_index))" in source
    assert "job = self._batchQueue.pop(0)" in source
    assert 'tool=str(job.get("tool") or "")' in source
    assert 'profile=str(job.get("profile") or "")' in source
    assert 'row=dict(job.get("row") or {})' in source
    assert 'if self._has_active_batch():' in source
    assert 'self._append_log("[batch] Tool/profile change will apply after the active queue finishes.")' in source
    finish_handler = source[
        source.index("    def _batch_process_finished(") : source.index("    def _refresh_row_output_paths(", source.index("    def _batch_process_finished("))
    ]
    assert 'if self._selected_tool_key() == "fea"' not in finish_handler


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
    assert 'tool_key in {"microarchitecture", "plate_rod"}' in refresh_outputs
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
    assert "def _ensure_loaded_source_volume(self, image_path):" in source
    assert 'ScancoIOLogic().import_image(image_path, scaling="density"' in source


def test_bone_contour_loader_accepts_material_label_outputs(monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    widget = module.BatchProcessorWidget

    label_path = Path("sub-SAMPLE341_ses-001_voi-tibia_desc-fea-materials_label.AIM")
    mask_path = Path("sub-SAMPLE341_ses-001_voi-tibia_desc-full_mask.AIM")

    assert widget._is_bone_contour_segmentation_output(label_path)
    assert widget._is_bone_contour_segmentation_output(mask_path)
    assert widget._mask_role_from_path(label_path) == "fea-materials"
    assert "fea-materials" in module._SEGMENT_COLORS

    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "No BoneContours mask/label outputs were discovered for this row." in source
    assert "if self._is_label_output(path):" in source


def test_mask_label_algebra_loads_imported_contours_not_generated_label(tmp_path: Path, monkeypatch) -> None:
    module = _import_batch_processor_module(monkeypatch)
    imported_dir = tmp_path / "derivatives" / "ImportedContours" / "sub-SAMPLE341" / "ses-001" / "xct"
    generated_dir = tmp_path / "derivatives" / "BoneContours" / "sub-SAMPLE341" / "ses-001" / "xct"
    imported_dir.mkdir(parents=True)
    generated_dir.mkdir(parents=True)
    imported_full = imported_dir / "sub-SAMPLE341_ses-001_voi-tibia_desc-full_mask.AIM"
    imported_trab = imported_dir / "sub-SAMPLE341_ses-001_voi-tibia_desc-trab_mask.AIM"
    generated_label = generated_dir / "sub-SAMPLE341_ses-001_voi-tibia_desc-fea-materials_label.AIM"
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
            metadata={"short_role": "fea-materials", "workflow": "mask_label_algebra"},
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
