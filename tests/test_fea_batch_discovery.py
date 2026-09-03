from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from SlicerBoneImagingToolboxLib.derivatives import (
    DerivativeManifest,
    DerivativeRecord,
    write_manifest,
)
from SlicerBoneImagingToolboxLib.fea_batch import (
    build_parosol_case_commands,
    case_readiness,
    discover_fea_batch_cases,
    discovered_role_options,
    parosol_command_derivative_context,
    role_options_for_workflow,
    workflow_role_requirements,
)


def test_discover_fea_batch_cases_groups_raw_and_derivative_artifacts(tmp_path: Path) -> None:
    """A raw-only scan fallback must not hide calibrated derivative inputs for the same session."""
    raw = tmp_path / "sub-001" / "ses-1" / "site-tibia" / "sub-001_ses-1_site-tibia_image.nii.gz"
    raw.parent.mkdir(parents=True)
    raw.write_text("raw", encoding="utf-8")
    mask = tmp_path / "sub-001" / "ses-1" / "site-tibia" / "sub-001_ses-1_site-tibia_mask-full.nii.gz"
    mask.write_text("mask", encoding="utf-8")

    calibrated = (
        tmp_path
        / "derivatives"
        / "Calibration"
        / "sub-001"
        / "site-tibia"
        / "ses-1"
        / "sub-001_ses-1_site-tibia_desc-bmd_image.nii.gz"
    )
    calibrated.parent.mkdir(parents=True)
    calibrated.write_text("bmd", encoding="utf-8")
    write_manifest(
        tmp_path / "derivatives" / "Calibration" / "manifest.json",
        DerivativeManifest(
            workflow="Calibration",
            version="1",
            dataset_root=str(tmp_path),
            records=[
                DerivativeRecord(
                    derivative="Calibration",
                    role="calibrated_image",
                    subject_id="001",
                    site="tibia",
                    session_id="1",
                    stack_index=None,
                    space="native",
                    path=str(calibrated),
                    source=str(raw),
                )
            ],
        ),
    )

    cases = discover_fea_batch_cases(tmp_path)

    assert len(cases) == 1
    case = cases[0]
    assert (case.subject_id, case.site, case.session_id) == ("001", "tibia", "1")
    assert case.artifact_options("image") == [str(calibrated), str(raw)]
    assert case.artifact_options("mask") == [str(mask)]


def test_discover_fea_batch_cases_supports_normalized_voi_layout_and_bone_contours(tmp_path: Path) -> None:
    raw = tmp_path / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    material = (
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "sub-001_ses-001_voi-radiusleft_desc-material-labelmap_map.nii.gz"
    )
    trab = (
        tmp_path
        / "derivatives"
        / "IPLContours"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "sub-001_ses-001_voi-radiusleft_desc-trab_mask.AIM"
    )
    full = (
        tmp_path
        / "derivatives"
        / "BoneContours"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM"
    )
    for path in (raw, material, trab, full):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    cases = discover_fea_batch_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].key == ("001", "radiusleft", "001")
    assert cases[0].artifact_options("image")[:2] == [str(material), str(raw)]
    assert cases[0].artifact_options("mask") == [str(full), str(trab)]


def test_manifest_classification_uses_record_role_before_derivative_family(tmp_path: Path) -> None:
    """Segmentation derivative manifests must keep cort/full masks distinct from bone segmentation."""
    mask_cort = tmp_path / "derivatives" / "Segmentation" / "sub-001_ses-1_site-tibia_mask-cort.nii.gz"
    seg = tmp_path / "derivatives" / "Segmentation" / "sub-001_ses-1_site-tibia_mask-seg.nii.gz"
    mask_cort.parent.mkdir(parents=True)
    mask_cort.write_text("cort", encoding="utf-8")
    seg.write_text("seg", encoding="utf-8")
    write_manifest(
        tmp_path / "derivatives" / "Segmentation" / "manifest.json",
        DerivativeManifest(
            workflow="Segmentation",
            version="1",
            dataset_root=str(tmp_path),
            records=[
                DerivativeRecord("Segmentation", "mask_cort", "001", "tibia", "1", None, "native", str(mask_cort), ""),
                DerivativeRecord("Segmentation", "segmentation", "001", "tibia", "1", None, "native", str(seg), ""),
            ],
        ),
    )

    cases = discover_fea_batch_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].artifact_options("mask") == [str(mask_cort)]
    assert str(seg) in cases[0].artifact_options("image")


def test_measurement_derivative_maps_are_not_fea_model_images(tmp_path: Path) -> None:
    """Microarchitecture scalar maps are measurements, not calibrated structural inputs for FEA."""
    tb_bmd = tmp_path / "derivatives" / "Microarchitecture" / "sub-001_ses-1_site-tibia_map-tb-bmd.nii.gz"
    tb_bmd.parent.mkdir(parents=True)
    tb_bmd.write_text("map", encoding="utf-8")
    write_manifest(
        tmp_path / "derivatives" / "Microarchitecture" / "manifest.json",
        DerivativeManifest(
            workflow="Microarchitecture",
            version="1",
            dataset_root=str(tmp_path),
            records=[
                DerivativeRecord("Microarchitecture", "tb_bmd_map", "001", "tibia", "1", None, "native", str(tb_bmd), ""),
            ],
        ),
    )

    cases = discover_fea_batch_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].artifact_options("image") == []


def test_workflow_role_requirements_keep_profile_inputs_general() -> None:
    """Workflow requirements must describe roles, not a hard-coded raw image assumption."""
    assert workflow_role_requirements("spine-compression")["image"].preferred_roles[0] == "calibrated_image"
    assert workflow_role_requirements("XtremeCTII")["image"].preferred_roles[0] == "material_labelmap"
    assert workflow_role_requirements("load_history_3")["image"].preferred_roles[0] == "material_labelmap"
    assert workflow_role_requirements("load_history_6")["image"].preferred_roles[0] == "material_labelmap"


def test_batch_discovery_support_is_limited_to_labelmap_profiles_for_now() -> None:
    """The Slicer batch UI should not guess inputs for profiles whose batch contract is not defined yet."""
    from SlicerBoneImagingToolboxLib.fea_batch import batch_profile_support_status

    assert batch_profile_support_status("XtremeCTI") == (True, "")
    assert batch_profile_support_status("XtremeCTII") == (True, "")
    assert batch_profile_support_status("load_history_3") == (True, "")
    assert batch_profile_support_status("load_history_6") == (True, "")
    assert batch_profile_support_status("spine-compression") == (
        False,
        "Batch discovery for this profile is not implemented yet.",
    )


def test_build_parosol_case_commands_use_selected_artifact_roles(tmp_path: Path) -> None:
    """Changing command generation to ignore the selected artifact role would run FEA on the wrong image."""
    image = tmp_path / "sub-001_ses-1_site-radius_material-labelmap.nii.gz"
    mask = tmp_path / "sub-001_ses-1_site-radius_mask-full.nii.gz"
    image.write_text("image", encoding="utf-8")
    mask.write_text("mask", encoding="utf-8")
    cases = discover_fea_batch_cases(tmp_path)

    commands = build_parosol_case_commands(
        tmp_path,
        cases,
        workflow="XtremeCTII",
        selected_roles={"image": "material_labelmap", "mask": "mask_full"},
        dry_run=True,
    )

    assert commands == [
        [
            str(image),
            "--profile",
            "XtremeCTII",
            "--mask",
            str(mask),
            "--dataset-root",
            str(tmp_path),
            "--subject",
            "001",
            "--site",
            "radius",
            "--name",
            "sub-001_ses-1_site-radius_XtremeCTII",
            "--dry-run",
        ]
    ]


def test_legacy_common_region_files_are_masks_not_candidate_images(tmp_path: Path) -> None:
    """A common-region mask without 'mask' in its filename must not be selected as the FEA image."""
    common = (
        tmp_path
        / "TimelapsedHRpQCT"
        / "sub-001"
        / "site-tibia"
        / "analysis"
        / "common_regions"
        / "sub-001_site-tibia_comp-cort_common-alltimepoints.nii.gz"
    )
    common.parent.mkdir(parents=True)
    common.write_text("common", encoding="utf-8")

    cases = discover_fea_batch_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].artifact_options("image") == []
    assert cases[0].artifact_options("mask") == [str(common)]
    assert build_parosol_case_commands(tmp_path, cases, workflow="XtremeCTII") == []


def test_parosol_command_derivative_context_recovers_run_metadata(tmp_path: Path) -> None:
    """Batch manifest writing needs the output folder and session represented by shortcut args."""
    command = [
        "/data/sub-001_ses-2_site-radius_image.nii.gz",
        "--profile",
        "XtremeCTII",
        "--dataset-root",
        str(tmp_path),
        "--subject",
        "001",
        "--site",
        "radius",
        "--name",
        "sub-001_ses-2_site-radius_XtremeCTII",
    ]

    context = parosol_command_derivative_context(command)

    assert context["dataset_root"] == str(tmp_path)
    assert context["subject_id"] == "001"
    assert context["site"] == "radius"
    assert context["session_id"] == "2"
    assert context["output_dir"] == str(
        tmp_path
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "xct"
        / "runs"
        / "sub-001_ses-2_site-radius_XtremeCTII"
    )


def test_remodelling_maps_are_not_candidate_fea_images(tmp_path: Path) -> None:
    """Timelapsed map outputs must not be auto-selected as the structural FEA image."""
    remodelling = (
        tmp_path
        / "TimelapsedHRpQCT"
        / "sub-001"
        / "site-tibia"
        / "analysis"
        / "visualize"
        / "sub-001_site-tibia_comp-full_t0-T1_t1-T2_remodelling.nii.gz"
    )
    remodelling.parent.mkdir(parents=True)
    remodelling.write_text("map", encoding="utf-8")

    cases = discover_fea_batch_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].artifact_options("image") == []
    assert build_parosol_case_commands(tmp_path, cases, workflow="XtremeCTII") == []


def test_xtremect_batch_requires_existing_material_labelmap(tmp_path: Path) -> None:
    """XtremeCT batch consumes model labels created upstream instead of generating them itself."""
    base = tmp_path / "sub-001" / "ses-1" / "site-radius"
    base.mkdir(parents=True)
    seg = base / "sub-001_ses-1_site-radius_mask-seg.nii.gz"
    full = base / "sub-001_ses-1_site-radius_mask-full.nii.gz"
    cort = base / "sub-001_ses-1_site-radius_mask-cort.nii.gz"
    seg_array = np.zeros((2, 3, 4), dtype=np.uint8)
    full_array = np.zeros_like(seg_array)
    cort_array = np.zeros_like(seg_array)
    seg_array[:, :, :] = 1
    full_array[:, :, :] = 1
    cort_array[:, :1, :] = 1
    for path, array in ((seg, seg_array), (full, full_array), (cort, cort_array)):
        sitk.WriteImage(sitk.GetImageFromArray(array), str(path))

    cases = discover_fea_batch_cases(tmp_path)
    commands = build_parosol_case_commands(tmp_path, cases, workflow="XtremeCTII", dry_run=True)

    assert commands == []
    assert case_readiness(cases[0], "XtremeCTII") == (False, ("image",))


def test_discovered_role_options_do_not_write_generated_masks(tmp_path: Path) -> None:
    """Readiness checks must not advertise generated material images or touch the dataset."""
    base = tmp_path / "sub-001" / "ses-1" / "site-radius"
    base.mkdir(parents=True)
    for name in (
        "sub-001_ses-1_site-radius_mask-seg.nii.gz",
        "sub-001_ses-1_site-radius_mask-full.nii.gz",
        "sub-001_ses-1_site-radius_mask-cort.nii.gz",
    ):
        sitk.WriteImage(sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8)), str(base / name))

    cases = discover_fea_batch_cases(tmp_path)

    assert "generated_material_labelmap" not in discovered_role_options(cases, "image")
    assert not any(base.glob("*derived_mask-trab.nii.gz"))


def test_xtremect_material_labelmap_discovery_accepts_model_labels(tmp_path: Path) -> None:
    """Existing material/model labelmaps are sufficient for supported XCT batch profiles."""
    base = tmp_path / "sub-001" / "ses-1" / "site-radius"
    base.mkdir(parents=True)
    model = base / "sub-001_ses-1_site-radius_model-labelmap.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8)), str(model))

    cases = discover_fea_batch_cases(tmp_path)
    commands = build_parosol_case_commands(tmp_path, cases, workflow="XtremeCTII", dry_run=True)

    assert cases[0].artifact_options("image") == [str(model)]
    assert commands[0][0] == str(model)
    assert case_readiness(cases[0], "XtremeCTII") == (True, ())


def test_xtremect_readiness_rejects_segmentation_without_material_labelmap(tmp_path: Path) -> None:
    """Segmentation and masks are not enough for ParOSol batch after contour-derived labels moved upstream."""
    base = tmp_path / "sub-001" / "ses-1" / "site-radius"
    base.mkdir(parents=True)
    for name in (
        "sub-001_ses-1_site-radius_mask-seg.nii.gz",
        "sub-001_ses-1_site-radius_mask-full.nii.gz",
        "sub-001_ses-1_site-radius_mask-cort.nii.gz",
    ):
        sitk.WriteImage(sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8)), str(base / name))

    case = discover_fea_batch_cases(tmp_path)[0]

    assert case_readiness(case, "XtremeCTII") == (False, ("image",))


def test_role_options_for_workflow_hide_unsupported_discovered_artifacts(tmp_path: Path) -> None:
    """The batch UI must show workflow-valid roles, not every discovered file class."""
    base = tmp_path / "sub-001" / "ses-1" / "site-radius"
    base.mkdir(parents=True)
    for name in (
        "sub-001_ses-1_site-radius_mask-seg.nii.gz",
        "sub-001_ses-1_site-radius_mask-full.nii.gz",
        "sub-001_ses-1_site-radius_mask-cort.nii.gz",
        "sub-001_ses-1_site-radius_image.nii.gz",
    ):
        sitk.WriteImage(sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8)), str(base / name))

    cases = discover_fea_batch_cases(tmp_path)

    assert role_options_for_workflow(cases, "XtremeCTII", "image") == []
    assert role_options_for_workflow(cases, "spine-compression", "image") == ["image"]
