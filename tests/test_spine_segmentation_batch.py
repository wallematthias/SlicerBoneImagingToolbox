from pathlib import Path

from SlicerBoneImagingToolboxLib.derivatives import DerivativeManifest, DerivativeRecord, write_manifest
from SlicerBoneImagingToolboxLib.spine_segmentation_batch import (
    build_spine_segmentation_batch_commands,
    discover_spine_segmentation_batch_cases,
    write_spine_segmentation_manifest,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_spine_batch_discovers_ct_images_from_manifest_and_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    manifest_image = root / "derivatives" / "Calibration" / "sub-S1" / "ses-T1" / "ct_calibrated.nii.gz"
    loose_image = root / "sub-S2" / "ses-T1" / "ct.nii.gz"
    _touch(manifest_image)
    _touch(loose_image)
    write_manifest(
        root / "derivatives" / "Calibration" / "manifest.json",
        DerivativeManifest(
            workflow="Calibration",
            version="1",
            dataset_root=str(root),
            records=[
                DerivativeRecord(
                    derivative="Calibration",
                    role="calibrated_image",
                    subject_id="S1",
                    site="spine",
                    session_id="T1",
                    stack_index=None,
                    space="native",
                    path=str(manifest_image),
                    source="test",
                )
            ],
        ),
    )

    cases = discover_spine_segmentation_batch_cases(root)

    assert [(case.subject_id, case.site, case.session_id) for case in cases] == [
        ("S1", "spine", "T1"),
        ("S2", "spine", "T1"),
    ]
    assert cases[0].first_image("calibrated_image").path == str(manifest_image)
    assert cases[1].first_image("image").path == str(loose_image)


def test_spine_batch_command_paths_reuse_inputs_and_write_derivatives(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-S1" / "ses-T1" / "ct.nii.gz"
    _touch(image)
    cases = discover_spine_segmentation_batch_cases(root)

    commands = build_spine_segmentation_batch_commands(
        root,
        cases,
        image_role="image",
        mode="full",
        device="auto",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command.case.subject_id == "S1"
    assert command.input_path == image
    assert command.output_dir == root / "derivatives" / "SpineSegmentationCT" / "sub-S1" / "site-spine" / "ses-T1"
    assert command.cli_args == [
        str(image),
        "--output",
        str(command.output_dir),
        "--device",
        "auto",
        "--overwrite",
    ]


def test_spine_batch_level_mode_uses_level_only_flag(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-S1" / "ses-T1" / "ct.nii.gz"
    _touch(image)
    cases = discover_spine_segmentation_batch_cases(root)

    commands = build_spine_segmentation_batch_commands(root, cases, mode="level")

    assert commands[0].cli_args[-1] == "--level-only"


def test_spine_batch_manifest_records_existing_outputs(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-S1" / "ses-T1" / "ct.nii.gz"
    _touch(image)
    command = build_spine_segmentation_batch_commands(
        root,
        discover_spine_segmentation_batch_cases(root),
    )[0]
    for suffix in ("vertebral-level", "process-body", "cort-trab"):
        _touch(command.output_dir / f"ct_{suffix}.nii.gz")
    _touch(command.output_dir / "ct_centroids.json")

    manifest_path = write_spine_segmentation_manifest(root, [command], module_version="test")

    payload = manifest_path.read_text(encoding="utf-8")
    assert "SpineSegmentationCT" in payload
    assert "vertebral_level_segmentation" in payload
    assert "process_body_segmentation" in payload
    assert "cort_trab_segmentation" in payload
    assert "vertebral_centroids" in payload


def test_spine_batch_resolves_relative_manifest_paths_against_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "derivatives" / "Calibration" / "sub-S1" / "ses-T1" / "ct_calibrated.nii.gz"
    _touch(image)
    write_manifest(
        root / "derivatives" / "Calibration" / "manifest.json",
        DerivativeManifest(
            workflow="Calibration",
            version="1",
            dataset_root=str(root),
            records=[
                DerivativeRecord(
                    derivative="Calibration",
                    role="calibrated_image",
                    subject_id="S1",
                    site="spine",
                    session_id="T1",
                    stack_index=None,
                    space="native",
                    path="derivatives/Calibration/sub-S1/ses-T1/ct_calibrated.nii.gz",
                    source="test",
                )
            ],
        ),
    )

    cases = discover_spine_segmentation_batch_cases(root)

    assert cases[0].first_image("calibrated_image").path == str(image)
