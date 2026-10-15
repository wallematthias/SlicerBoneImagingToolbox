from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _export(tmp_path, tool, cases, profile="standard"):
    module = importlib.import_module("SlicerBoneImagingToolboxLib.measurement_export")
    destination = tmp_path / "cohort.csv"
    report = module.export_measurements(
        destination, dataset_root=tmp_path, tool=tool, profile=profile, cases=cases,
    )
    with destination.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return report, rows


def _csv(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _case(paths, **overrides):
    return dict(subject="001", session_value="01", voi_value="radiusleft",
                stack_index=None, paths=paths, **overrides)


def _rename_manifest(root, scans, *, recorded_root=None):
    """Create real normalized inputs and their naming-helper provenance."""
    recorded_root = str(recorded_root or root).rstrip("/\\")
    renames = []
    for relative, original in scans:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        renames.append({"original_path": original, "renamed_path": f"{recorded_root}/{relative}"})
    manifest = root / "dataset_rename_manifest.json"
    manifest.write_text(json.dumps({"version": 1, "dataset_root": recorded_root, "renames": renames}))
    return manifest


@pytest.mark.parametrize("tool,name,text", [
    ("microarchitecture", "case_measurements.csv", "Parameter,Mean\nTb.Th,0.12\n"),
    ("voidspace", "voidspace_measurements.csv", "VS.V\n25\n"),
    ("plate_rod", "case_desc-plate-rod-measurements.csv",
     "subject_id,site,session_id,pBV/TV\n001,radiusleft,01,0.2\n"),
    ("fea", "case_fea.csv", "Profile,Stiffness (N/mm)\nstandard,123\n"),
])
def test_export_original_filename_is_shared_across_single_scan_workflows(tmp_path, tool, name, text):
    _rename_manifest(tmp_path, [
        ("sub-001/ses-01/xct/sub-001_ses-01_voi-radiusleft_xct.AIM", "/scanner/SCAN, original.AIM;1"),
        ("sub-001/ses-01/xct/sub-001_ses-01_voi-tibialeft_xct.AIM", "/scanner/WRONG_SITE.AIM"),
        ("derivatives/IPLContours/sub-001/ses-01/xct/sub-001_ses-01_voi-radiusleft_desc-cort_mask.AIM",
         "/scanner/CORT_MASK.AIM"),
    ])
    report, rows = _export(tmp_path, tool, [_case([_csv(tmp_path, name, text)])])
    assert rows[0]["original_filename"] == "SCAN, original.AIM;1"
    assert rows[0]["baseline_original_filename"] == rows[0]["followup_original_filename"] == ""
    assert report.row_count == 1 and not report.issues and not report.warnings


@pytest.mark.parametrize("recorded_root", ["/old/server/dataset", r"D:\SOBER\dataset"])
def test_original_filenames_survive_moved_dataset_and_keep_stacks_separate(tmp_path, recorded_root):
    manifest = _rename_manifest(tmp_path, [
        ("sub-001/ses-01/xct/sub-001_ses-01_voi-radiusleft_stack-01_xct.AIM", r"D:\raw\STACK1.AIM"),
        ("sub-001/ses-01/xct/sub-001_ses-01_voi-radiusleft_stack-02_xct.AIM", r"D:\raw\STACK2.AIM"),
    ], recorded_root=recorded_root)
    # Windows manifests use backslashes even when later opened on another OS.
    if recorded_root.startswith("D:"):
        payload = json.loads(manifest.read_text())
        for item in payload["renames"]:
            item["renamed_path"] = item["renamed_path"].replace("/", "\\")
        manifest.write_text(json.dumps(payload))
    path = _csv(tmp_path, "case_measurements.csv", "Parameter,Mean\nTb.Th,0.12\n")
    case = _case([path]); case["stack_index"] = 2
    report, rows = _export(tmp_path, "microarchitecture", [case])
    assert rows[0]["original_filename"] == "STACK2.AIM"
    assert not report.warnings


def test_timelapse_resolves_original_filenames_for_each_exported_pair(tmp_path):
    _rename_manifest(tmp_path, [
        (f"sub-001/ses-{session}/xct/sub-001_ses-{session}_voi-radiusleft_xct.AIM", f"/raw/T{index}.AIM")
        for index, session in enumerate(("01", "02", "03"), 1)
    ])
    path = _csv(tmp_path, "case_pairwise_remodelling.csv",
                "subject_id,compartment,t0,t1,threshold,cluster_min_size,formation_vox\n"
                "001,trab,01,02,225,1,10\n001,trab,02,03,225,1,11\n")
    report, rows = _export(tmp_path, "timelapse", [_case([path])])
    assert [(r["baseline_original_filename"], r["followup_original_filename"]) for r in rows] == [
        ("T1.AIM", "T2.AIM"), ("T2.AIM", "T3.AIM")]
    assert all(r["original_filename"] == "" for r in rows)
    assert not report.warnings


@pytest.mark.parametrize("tool,name,text", [
    ("voidspace", "voidspace_change_measurements.csv", "expanded.VS.V\n5\n"),
    ("mechanoregulation", "case_roi-full_mechanoregulation_summary.csv", "OR_F\n1.2\n"),
])
def test_pair_export_keeps_valid_filename_when_other_timepoint_is_ambiguous(tmp_path, tool, name, text):
    manifest = _rename_manifest(tmp_path, [
        ("sub-001/ses-01/xct/sub-001_ses-01_voi-radiusleft_xct.AIM", "/raw/BASELINE.AIM"),
        ("sub-001/ses-02/xct/sub-001_ses-02_voi-radiusleft_xct.AIM", "/raw/FOLLOWUP.AIM"),
    ])
    payload = json.loads(manifest.read_text())
    payload["renames"].append({**payload["renames"][1], "original_path": "/other/AMBIGUOUS.AIM"})
    manifest.write_text(json.dumps(payload))
    path = _csv(tmp_path, name, text)
    if tool == "mechanoregulation":
        Path(path).with_suffix(".json").write_text(json.dumps({"profile": "dynamic"}))
    case = _case([path], baseline_session_value="01", followup_session_value="02")
    report, rows = _export(tmp_path, tool, [case], "dynamic")
    assert rows[0]["baseline_original_filename"] == "BASELINE.AIM"
    assert rows[0]["followup_original_filename"] == ""
    assert report.row_count == 1 and not report.issues
    assert any("ambiguous" in warning.lower() and "02" in warning for warning in report.warnings)


@pytest.mark.parametrize("manifest_text", [None, "{broken", '{"renames": {}}', '{"renames": [null]}', '{"renames": []}'])
def test_missing_or_invalid_provenance_warns_without_omitting_measurements(tmp_path, manifest_text):
    if manifest_text is not None:
        (tmp_path / "dataset_rename_manifest.json").write_text(manifest_text)
    path = _csv(tmp_path, "case_measurements.csv", "Parameter,Mean\nTb.Th,0.12\n")
    report, rows = _export(tmp_path, "microarchitecture", [_case([path])])
    assert rows[0]["original_filename"] == "" and rows[0]["Tb.Th.Mean"] == "0.12"
    assert report.row_count == 1 and not report.issues and report.warnings


def test_relative_rename_targets_and_duplicate_entries_do_not_create_false_ambiguity(tmp_path):
    relative = "sub-001/ses-01/xct/sub-001_ses-01_voi-radiusleft_xct.AIM"
    manifest = _rename_manifest(tmp_path, [(relative, "/raw/ORIGINAL.AIM")])
    payload = json.loads(manifest.read_text())
    payload["renames"][0]["renamed_path"] = relative
    payload["renames"] *= 2
    payload["renames"].append({"original_path": "/raw/SIDECAR.json", "renamed_path": relative + ".json"})
    manifest.write_text(json.dumps(payload))
    path = _csv(tmp_path, "case_measurements.csv", "Parameter,Mean\nTb.Th,0.12\n")
    report, rows = _export(tmp_path, "microarchitecture", [_case([path])])
    assert rows[0]["original_filename"] == "ORIGINAL.AIM" and not report.warnings


def test_matching_subject_but_wrong_session_site_or_stack_never_supplies_original_name(tmp_path):
    _rename_manifest(tmp_path, [
        ("sub-001/ses-02/xct/sub-001_ses-02_voi-radiusleft_xct.AIM", "/raw/WRONG_SESSION.AIM"),
        ("sub-001/ses-01/xct/sub-001_ses-01_voi-tibialeft_xct.AIM", "/raw/WRONG_SITE.AIM"),
        ("sub-001/ses-01/xct/sub-001_ses-01_voi-radiusleft_stack-01_xct.AIM", "/raw/WRONG_STACK.AIM"),
    ])
    path = _csv(tmp_path, "case_measurements.csv", "Parameter,Mean\nTb.Th,0.12\n")
    report, rows = _export(tmp_path, "microarchitecture", [_case([path])])
    assert rows[0]["original_filename"] == "" and rows[0]["Tb.Th.Mean"] == "0.12"
    assert not report.issues and any("no matching rename entry" in w for w in report.warnings)


def test_multiple_raw_images_for_same_case_do_not_choose_the_only_known_original(tmp_path):
    relative = "sub-001/ses-01/xct/sub-001_ses-01_voi-radiusleft_xct.AIM"
    _rename_manifest(tmp_path, [(relative, "/raw/ORIGINAL.AIM")])
    (tmp_path / relative.replace(".AIM", ".nii.gz")).touch()
    path = _csv(tmp_path, "case_measurements.csv", "Parameter,Mean\nTb.Th,0.12\n")
    report, rows = _export(tmp_path, "microarchitecture", [_case([path])])
    assert rows[0]["original_filename"] == "" and report.row_count == 1
    assert any("ambiguous" in warning for warning in report.warnings)


def test_microarchitecture_exports_primary_values_without_distribution_columns(tmp_path):
    path = _csv(tmp_path, "case_measurements.csv",
                "Parameter,Mean,Median,SD,Units\nTb.Th,0.12,0.11,0.03,mm\nCt.Th,0.8,,nan,mm\n")
    second = _csv(tmp_path, "other_measurements.csv", "Parameter,Mean,Units\nTb.Th,0,mm\n")
    case = _case([second])
    case["subject"] = "002"
    report, rows = _export(tmp_path, "microarchitecture", [_case([path]), case])
    assert report.row_count == 2 and not report.issues
    assert rows[0]["subject_id"] == "001"
    assert rows[0]["Tb.Th.Mean"] == "0.12"
    assert "Tb.Th.SD" not in rows[0]
    assert "Tb.Th.Units" not in rows[0]
    assert "Ct.Th.Median" not in rows[0]
    assert rows[1]["Tb.Th.Mean"] == "0" and rows[1]["Ct.Th.Mean"] == ""
    assert json.loads(rows[0]["source_files"]) == ["case_measurements.csv"]


def test_voidspace_prefixes_large_metrics_without_inventing_all_metrics(tmp_path):
    path = _csv(tmp_path, "voidspace_measurements.csv", "VS.TV,VS.V,Tt.V,VS.N\n11.66,25,215,2\n")
    report, rows = _export(tmp_path, "voidspace", [_case([path])], "registered")
    assert rows[0]["Large.VS.TV"] == "11.66"
    assert rows[0]["Large.VS.V"] == "25"
    assert "All.VS.V" not in rows[0]
    assert rows[0]["profile"] == "registered" and report.row_count == 1


def test_dynamic_voidspace_keeps_pair_and_change_columns(tmp_path):
    path = _csv(tmp_path, "voidspace_change_measurements.csv", "expanded.VS.V,contracted.VS.V,net.VS.V\n5,3,2\n")
    case = _case([path])
    case.update(session_value="01-02", baseline_session_value="01", followup_session_value="02")
    _, rows = _export(tmp_path, "voidspace", [case], "dynamic")
    assert rows[0]["baseline_session_id"] == "01"
    assert rows[0]["followup_session_id"] == "02"
    assert rows[0]["session_id"] == ""
    assert rows[0]["expanded.VS.V"] == "5"


def test_timelapse_flattens_compartments_and_thresholds_without_averaging(tmp_path):
    path = _csv(tmp_path, "sub-001_voi-radiusleft_pairwise_remodelling.csv",
                "subject_id,compartment,t0,t1,threshold,cluster_min_size,formation_frac_bv0,resorption_frac_bv0\n"
                "001,trab,01,02,225,1,0.10,0.04\n001,cort,01,02,225,1,0.03,0.02\n"
                "001,trab,01,02,300,1,0.07,0.04\n001,trab,02,03,225,1,0.11,0.05\n")
    report, rows = _export(tmp_path, "timelapse", [_case([path, path]), _case([path])])
    assert report.row_count == 2 and report.source_count == 1
    assert rows[0]["baseline_session_id"] == "01" and rows[0]["followup_session_id"] == "02"
    assert rows[0]["trab.threshold-225.cluster-1.formation_volume_fraction"] == "0.10"
    assert rows[0]["cort.threshold-225.cluster-1.formation_volume_fraction"] == "0.03"
    assert rows[0]["trab.threshold-300.cluster-1.formation_volume_fraction"] == "0.07"
    assert rows[1]["cort.threshold-225.cluster-1.formation_volume_fraction"] == ""
    assert rows[0]["trab.threshold-225.cluster-1.net_change_volume_fraction"] == "0.06"
    assert rows[0]["trab.threshold-225.cluster-1.active_volume_fraction"] == "0.14"


def test_mechanoregulation_combines_roi_summaries_but_not_curves(tmp_path):
    full = _csv(tmp_path, "case_roi-full_mechanoregulation_summary.csv", "OR_F,n_surface_voxels\n1.2,100\n")
    trab = _csv(tmp_path, "case_roi-trab_mechanoregulation_summary.csv", "OR_F,n_surface_voxels\n1.4,80\n")
    for path in (full, trab):
        Path(path).with_suffix(".json").write_text(json.dumps({"profile": "standard"}))
    curve = _csv(tmp_path, "case_conditional_curves.csv", "x,y\n0,2\n1,3\n")
    case = _case([full, trab, curve])
    case.update(session_value="01-02", baseline_session_value="01", followup_session_value="02")
    report, rows = _export(tmp_path, "mechanoregulation", [case])
    assert report.source_count == 2 and report.row_count == 1
    assert rows[0]["full.OR_F"] == "1.2" and rows[0]["trab.OR_F"] == "1.4"
    assert "x" not in rows[0]


def test_plate_rod_preserves_wide_summary_and_ignores_element_tables(tmp_path):
    path = _csv(tmp_path, "case_desc-plate-rod-measurements.csv",
                "subject_id,site,session_id,pBV/TV,rBV/TV\n001,radiusleft,01,0.2,0.1\n")
    elements = _csv(tmp_path, "case_elements.csv", "id,volume\n1,2\n2,3\n")
    report, rows = _export(tmp_path, "plate_rod", [_case([path, elements])])
    assert rows[0]["pBV/TV"] == "0.2" and rows[0]["rBV/TV"] == "0.1"
    assert "id" not in rows[0] and report.source_count == 1


def test_fea_flattens_load_components_and_preserves_units(tmp_path):
    path = _csv(tmp_path, "case_fea.csv",
                'Sample,Profile,Stiffness (N/mm),Failure load (N),Estimated loads\n'
                'sub-001 ses-01 voi-radiusleft,load_history,123,456,"1; 2; -3"\n')
    _, rows = _export(tmp_path, "fea", [_case([path])], "load_history")
    assert rows[0]["Stiffness (N/mm)"] == "123"
    assert rows[0]["Estimated loads.1"] == "1" and rows[0]["Estimated loads.3"] == "-3"
    assert "Sample" not in rows[0]


def test_conflicting_case_is_reported_not_silently_overwritten(tmp_path):
    a = _csv(tmp_path, "a_measurements.csv", "Parameter,Mean\nTb.Th,0.1\n")
    b = _csv(tmp_path, "b_measurements.csv", "Parameter,Mean\nTb.Th,0.2\n")
    report, rows = _export(tmp_path, "microarchitecture", [_case([a, b])])
    assert not rows and report.row_count == 0
    assert any("conflict" in issue.lower() and "Tb.Th.Mean" in issue for issue in report.issues)


def test_missing_and_malformed_csvs_are_reported_and_valid_cases_still_export(tmp_path):
    path = _csv(tmp_path, "good_measurements.csv", "Parameter,Mean\nTb.Th,0.1\n")
    malformed = _csv(tmp_path, "bad_measurements.csv", "Parameter,Mean\nTb.Th,0.2,unexpected\n")
    second = _case([malformed]); second["subject"] = "002"
    third = _case([str(tmp_path / "missing_measurements.csv")]); third["subject"] = "003"
    report, rows = _export(tmp_path, "microarchitecture", [_case([path]), second, third])
    assert len(rows) == 1 and len(report.issues) == 2


def test_export_never_overwrites_inputs_or_existing_outputs_without_permission(tmp_path):
    module = importlib.import_module("SlicerBoneImagingToolboxLib.measurement_export")
    path = _csv(tmp_path, "case_measurements.csv", "Parameter,Mean\nTb.Th,0.1\n")
    args = dict(dataset_root=tmp_path, tool="microarchitecture", profile="standard", cases=[_case([path])])
    with pytest.raises(ValueError, match="source"):
        module.export_measurements(path, force=True, **args)
    destination = tmp_path / "cohort.csv"
    destination.write_text("keep existing")
    with pytest.raises(FileExistsError):
        module.export_measurements(destination, **args)
    assert destination.read_text() == "keep existing"


@pytest.mark.parametrize("tool,name,text", [
    ("fea", "case_fea.csv", "Profile,Stiffness (N/mm)\nother,123\n"),
    ("timelapse", "case_pairwise_remodelling.csv",
     "subject_id,compartment,t0,t1,threshold,cluster_min_size,formation_vox,profile\n001,trab,01,02,225,1,5,other\n"),
])
def test_known_source_profile_mismatch_is_reported_not_relabelled(tmp_path, tool, name, text):
    report, rows = _export(tmp_path, tool, [_case([_csv(tmp_path, name, text)])])
    assert not rows and len(report.issues) == 1
    assert "profile" in report.issues[0].lower()


def test_missing_subject_is_not_exported_as_literal_none(tmp_path):
    path = _csv(tmp_path, "case_measurements.csv", "Parameter,Mean\nTb.Th,0.1\n")
    case = _case([path]); case["subject"] = None
    report, rows = _export(tmp_path, "microarchitecture", [case])
    assert not rows and len(report.issues) == 1


def test_mechanoregulation_profiles_and_threshold_variants_are_not_collapsed(tmp_path):
    cases = []
    for threshold in (225, 300):
        name = f"sub-001_voi-radiusleft_t0-01_t1-02_thr-{threshold}_cluster-12_roi-full_mechanoregulation_summary.csv"
        path = Path(_csv(tmp_path, name, f"OR_F\n{threshold}\n"))
        path.with_suffix(".json").write_text(json.dumps({"profile": "XtremeCTII"}))
        case = _case([str(path)])
        case.update(baseline_session_value="01", followup_session_value="02")
        cases.append(case)
    report, rows = _export(tmp_path, "mechanoregulation", cases, "XtremeCTII")
    assert report.row_count == 1 and not report.issues
    assert rows[0]["full.threshold-225.cluster-12.OR_F"] == "225"
    assert rows[0]["full.threshold-300.cluster-12.OR_F"] == "300"
    module = importlib.import_module("SlicerBoneImagingToolboxLib.measurement_export")
    report = module.export_measurements(tmp_path / "other.csv", dataset_root=tmp_path,
                                        tool="mechanoregulation", profile="load_history_3", cases=cases)
    assert report.row_count == 0 and all("profile" in issue for issue in report.issues)


@pytest.mark.parametrize("profile", ["xtremectii", "functional-bone", "functional-bone-native"])
def test_complete_microarchitecture_summary_has_28_columns(tmp_path, profile):
    parameters = [
        "Tt.BMD", "Tb.BMD", "Tb.BV/TV", "Tb.Th", "Tb.Sp", "Tb.N", "Tb.1/N.SD",
        "Tb.BV", "Tb.TV", "Ct.BMD", "Ct.Th", "Ct.Po", "Ct.Po.V", "Ct.Po.Dm", "Ct.BV", "Ct.TV",
    ]
    path = _csv(tmp_path, "case_measurements.csv", "Parameter,Mean,Median,SD,Units\n" +
                "".join(f"{name},0.12,0.11,0.03,mm\n" for name in parameters))
    original = Path(path).read_bytes()
    report, rows = _export(tmp_path, "microarchitecture", [_case([path])], profile)
    assert not report.issues and len(rows[0]) == 28
    assert rows[0]["Tb.1/N.SD.Mean"] == "0.12"  # Primary heterogeneity parameter, not an extra statistic.
    assert Path(path).read_bytes() == original


def test_timelapse_summary_keeps_four_outcomes_and_shared_context_once(tmp_path):
    path = _csv(tmp_path, "case_pairwise_remodelling.csv",
                "subject_id,compartment,t0,t1,threshold,cluster_min_size,formation_frac_bv0,resorption_frac_bv0,"
                "fraction_denominator,followup_days,scan_date_t0,scan_date_t1,formation_vox,profile,ring_artifact_suppression_enabled\n" +
                "".join(f"001,{part},01,02,225,12,0.10,0.04,baseline_bone,365,2025-01-01,2026-01-01,10,standard,True\n"
                        for part in ("full", "cort", "trab")))
    original = Path(path).read_bytes()
    report, rows = _export(tmp_path, "timelapse", [_case([path])])
    assert not report.issues and len(rows[0]) == 28
    assert rows[0]["fraction_denominator"] == "baseline_bone"
    assert rows[0]["followup_days"] == "365"
    assert rows[0]["scan_date_t0"] == "2025-01-01"
    assert rows[0]["full.threshold-225.cluster-12.active_volume_fraction"] == "0.14"
    assert not any("formation_vox" in column or "ring_artifact" in column for column in rows[0])
    assert Path(path).read_bytes() == original


@pytest.mark.parametrize("formation,resorption,net,active", [
    ("0.1", "0.2", "-0.1", "0.3"), ("0", "0", "0", "0"),
    ("nan", "0.2", "", ""), ("0.1", "", "", ""),
])
def test_timelapse_derived_fractions_preserve_sign_zero_and_missing(tmp_path, formation, resorption, net, active):
    path = _csv(tmp_path, "case_pairwise_remodelling.csv",
                "compartment,t0,t1,threshold,cluster_min_size,formation_frac_bv0,resorption_frac_bv0,fraction_denominator\n"
                f"trab,01,02,225,12,{formation},{resorption},bone_union\n")
    _, rows = _export(tmp_path, "timelapse", [_case([path])])
    assert rows[0]["trab.threshold-225.cluster-12.net_change_volume_fraction"] == net
    assert rows[0]["trab.threshold-225.cluster-12.active_volume_fraction"] == active
    assert rows[0]["fraction_denominator"] == "bone_union"  # Never relabel other profile denominators as BV0.


def test_mechanoregulation_omits_aliases_and_sample_counts_but_keeps_cis(tmp_path):
    path = _csv(tmp_path, "case_roi-full_mechanoregulation_summary.csv",
                "CCR,CCR_low_threshold,CCR_high_threshold,binned_lazy_zone_low,binned_lazy_zone_high,"
                "logistic_lazy_zone_low,logistic_lazy_zone_high,OR_F,OR_R,OR_F_CI_low,OR_F_CI_high,"
                "OR_R_CI_low,OR_R_CI_high,OR_F_ratio,n_surface_voxels,formation_odds_increase_percent_per_sed_percent\n"
                "0.6,0.2,0.8,0.2,0.8,0.3,0.7,20,30,10,40,15,45,1.2,100,20\n")
    Path(path).with_suffix(".json").write_text(json.dumps({"profile": "standard"}))
    report, rows = _export(tmp_path, "mechanoregulation", [_case([path])])
    assert not report.issues and len(rows[0]) == 23
    assert rows[0]["full.OR_F"] == "20" and rows[0]["full.OR_F_CI_low"] == "10"
    assert rows[0]["full.logistic_lazy_zone_low"] == "0.3"
    assert "full.CCR_low_threshold" not in rows[0]
    assert "full.OR_F_ratio" not in rows[0] and "full.n_surface_voxels" not in rows[0]


def test_plate_rod_keeps_morphometry_not_settings_counts_or_ratio_alias(tmp_path):
    path = _csv(tmp_path, "case_desc-plate-rod-measurements.csv",
                "subject_id,site,session_id,pBV/TV,rBV/TV,pBV/rBV,PR_volume_ratio,pTb.Th_mm,"
                "plate_count,bone_voxels,classifier,max_iterations\n"
                "001,radiusleft,01,0.2,0.1,2,2,0.15,8,100,saha_topology,1000\n")
    _, rows = _export(tmp_path, "plate_rod", [_case([path])])
    assert rows[0]["pBV/rBV"] == "2" and rows[0]["pTb.Th_mm"] == "0.15"
    assert rows[0]["plate_count"] == "8"  # Biological element count, not implementation voxel count.
    assert "PR_volume_ratio" not in rows[0] and "bone_voxels" not in rows[0]
    assert "classifier" not in rows[0] and "max_iterations" not in rows[0]


def test_dynamic_voidspace_exports_quiescent_once(tmp_path):
    path = _csv(tmp_path, "voidspace_change_measurements.csv",
                "stable.VS.V,quiescent.VS.V,expanded.VS.V,contracted.VS.V,net.VS.V\n8,8,5,3,2\n")
    _, rows = _export(tmp_path, "voidspace", [_case([path])], "dynamic")
    assert len(rows[0]) == 16 and rows[0]["quiescent.VS.V"] == "8"
    assert "stable.VS.V" not in rows[0]


def test_legacy_dynamic_voidspace_stable_alias_is_still_exported(tmp_path):
    path = _csv(tmp_path, "voidspace_change_measurements.csv", "stable.VS.V,net.VS.V\n8,2\n")
    _, rows = _export(tmp_path, "voidspace", [_case([path])], "dynamic")
    assert rows[0]["quiescent.VS.V"] == "8" and "stable.VS.V" not in rows[0]


def test_conflicting_voidspace_aliases_are_reported_not_discarded(tmp_path):
    path = _csv(tmp_path, "voidspace_change_measurements.csv",
                "stable.VS.V,quiescent.VS.V\n8,9\n")
    report, rows = _export(tmp_path, "voidspace", [_case([path])], "dynamic")
    assert not rows and len(report.issues) == 1
    assert "conflicting measurement: quiescent.VS.V" in report.issues[0]


def test_timelapse_conflicting_denominators_are_not_silently_combined(tmp_path):
    path = _csv(tmp_path, "case_pairwise_remodelling.csv",
                "compartment,t0,t1,threshold,cluster_min_size,formation_frac_bv0,resorption_frac_bv0,fraction_denominator\n"
                "trab,01,02,225,12,0.1,0.2,baseline_bone\n"
                "cort,01,02,225,12,0.1,0.2,bone_union\n")
    report, rows = _export(tmp_path, "timelapse", [_case([path])])
    assert not rows and len(report.issues) == 1
    assert "conflicting measurement: fraction_denominator" in report.issues[0]


def test_its_preserves_element_volumes_distinct_from_label_volumes(tmp_path):
    fields = [
        "BV/TV", "pBV/TV", "rBV/TV", "pBV/BV", "rBV/BV", "pBV/rBV",
        "TV_mm3", "BV_mm3", "pBV_mm3", "rBV_mm3", "plate_count", "rod_count", "junction_count",
        "pTb.N", "rTb.N", "PR.N", "pTb.Th_mm", "rTb.Th_mm", "pTb.S_mm2", "rTb.l_mm",
        "P-P Junc.D", "P-R Junc.D", "R-R Junc.D", "aBV/TV",
        "mean_plate_axial_alignment", "mean_rod_axial_alignment",
        "measured_BV_mm3", "measured_pBV_mm3", "measured_rBV_mm3",
    ]
    path = _csv(tmp_path, "case_desc-plate-rod-measurements.csv",
                "subject_id,site,session_id," + ",".join(fields) + "\n001,radiusleft,01," +
                ",".join(str(index) for index in range(len(fields))) + "\n")
    report, rows = _export(tmp_path, "plate_rod", [_case([path])])
    assert not report.issues and len(rows[0]) == 41
    assert rows[0]["BV_mm3"] == "7" and rows[0]["measured_BV_mm3"] == "26"
    assert rows[0]["BV/TV"] == "0"
