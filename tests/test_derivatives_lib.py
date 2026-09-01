from pathlib import Path

from SlicerBoneImagingToolboxLib.derivatives import (
    DerivativeManifest,
    DerivativeRecord,
    discover_manifests,
    find_records,
    read_manifest,
    write_manifest,
)
from bone_imaging_derivatives import DerivativeManifest as SharedDerivativeManifest
from bone_imaging_derivatives import DerivativeRecord as SharedDerivativeRecord
from bone_imaging_derivatives import write_manifest as write_shared_manifest


def test_manifest_round_trip_preserves_records(tmp_path: Path) -> None:
    manifest = DerivativeManifest(
        workflow="CommonRegion",
        version="1",
        dataset_root=str(tmp_path),
        records=[
            DerivativeRecord(
                derivative="CommonRegion",
                role="scan_region_native_common",
                subject_id="SAMPLE001",
                site="tibia",
                session_id="1",
                stack_index=1,
                space="native",
                path="sub-SAMPLE001/site-tibia/native_space/ses-1/masks/mask.nii.gz",
                source="generated",
                metadata={"reference_session": "1"},
            )
        ],
    )
    path = tmp_path / "derivatives" / "CommonRegion" / "manifest.json"

    write_manifest(path, manifest)
    loaded = read_manifest(path)

    assert loaded.workflow == "CommonRegion"
    assert loaded.records[0].role == "scan_region_native_common"
    assert loaded.records[0].metadata["reference_session"] == "1"


def test_find_records_filters_by_subject_site_role(tmp_path: Path) -> None:
    manifest = DerivativeManifest(
        workflow="Registration",
        version="1",
        dataset_root=str(tmp_path),
        records=[
            DerivativeRecord(
                "Registration",
                "transform_composed",
                "S1",
                "tibia",
                "1",
                1,
                "reference",
                "a.tfm",
                "generated",
                {},
            ),
            DerivativeRecord(
                "Registration",
                "transform_pairwise",
                "S1",
                "tibia",
                "2",
                1,
                "native",
                "b.tfm",
                "generated",
                {},
            ),
            DerivativeRecord(
                "Registration",
                "transform_composed",
                "S2",
                "radius",
                "1",
                1,
                "reference",
                "c.tfm",
                "generated",
                {},
            ),
        ],
    )

    matches = find_records(
        manifest,
        derivative="Registration",
        role="transform_composed",
        subject_id="S1",
        site="tibia",
    )

    assert [record.path for record in matches] == ["a.tfm"]


def test_discover_manifests_finds_registered_derivative_manifests(tmp_path: Path) -> None:
    common_manifest = DerivativeManifest(
        workflow="CommonRegion",
        version="1",
        dataset_root=str(tmp_path),
        records=[],
    )
    registration_manifest = DerivativeManifest(
        workflow="Registration",
        version="1",
        dataset_root=str(tmp_path),
        records=[],
    )
    write_manifest(tmp_path / "derivatives" / "CommonRegion" / "manifest.json", common_manifest)
    write_manifest(tmp_path / "derivatives" / "Registration" / "manifest.json", registration_manifest)

    discovered = discover_manifests(tmp_path / "derivatives")

    assert [manifest.workflow for manifest in discovered] == ["CommonRegion", "Registration"]


def test_derivative_shim_delegates_contract_io_to_shared_package() -> None:
    source = (Path(__file__).resolve().parents[1] / "SlicerBoneImagingToolboxLib" / "derivatives.py").read_text(encoding="utf-8")

    assert "from bone_imaging_derivatives" in source


def test_discovery_keeps_shared_contract_manifests_when_legacy_manifests_exist(tmp_path: Path) -> None:
    write_manifest(
        tmp_path / "derivatives" / "Legacy" / "manifest.json",
        DerivativeManifest(workflow="Legacy", version="1", dataset_root=str(tmp_path)),
    )
    write_shared_manifest(
        SharedDerivativeManifest.create("Registration", tmp_path, {"name": "test", "version": "1"}),
        tmp_path / "derivatives" / "Registration" / "manifest.json",
    )

    discovered = discover_manifests(tmp_path / "derivatives")

    assert {manifest.workflow for manifest in discovered} == {"Legacy", "Registration"}


def test_shared_contract_conversion_preserves_null_stack_index(tmp_path: Path) -> None:
    record = SharedDerivativeRecord(
        "Registration", "transform_to_reference", "S1", "tibia", "2", None,
        "reference", tmp_path / "transform.tfm", "generated",
    )
    write_shared_manifest(
        SharedDerivativeManifest.create("Registration", tmp_path, {"name": "test", "version": "1"}, records=(record,)),
        tmp_path / "derivatives" / "Registration" / "manifest.json",
    )

    discovered = discover_manifests(tmp_path)

    assert discovered[0].records[0].stack_index is None
