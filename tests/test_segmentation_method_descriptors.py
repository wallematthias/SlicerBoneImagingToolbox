from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_segmentation_method_descriptors_define_xct2_choices() -> None:
    from SlicerBoneImagingToolboxLib.segmentation_methods import (
        BONE_SEGMENTATION_METHODS,
        ENDOSTEAL_CONTOUR_METHODS,
        PERIOSTEAL_CONTOUR_METHODS,
    )

    assert BONE_SEGMENTATION_METHODS["seg_gauss"].label == "XCT2 Gaussian"
    assert BONE_SEGMENTATION_METHODS["laplace_hamming"].label == "XCT2 Laplace-Hamming"
    assert BONE_SEGMENTATION_METHODS["adaptive"].label == "XCT2 Adaptive"

    assert PERIOSTEAL_CONTOUR_METHODS["standard"].label == "Standard"
    assert PERIOSTEAL_CONTOUR_METHODS["geodesic_fracture"].label == "Geodesic Fracture"
    assert ENDOSTEAL_CONTOUR_METHODS["standard"].label == "Standard"


def test_method_descriptors_do_not_include_external_segmentation_workflows() -> None:
    from SlicerBoneImagingToolboxLib.segmentation_methods import (
        BONE_SEGMENTATION_METHODS,
        ENDOSTEAL_CONTOUR_METHODS,
        PERIOSTEAL_CONTOUR_METHODS,
        method_supports_site,
    )

    external_token = "or" + "mir"
    all_method_ids = (
        *BONE_SEGMENTATION_METHODS,
        *PERIOSTEAL_CONTOUR_METHODS,
        *ENDOSTEAL_CONTOUR_METHODS,
    )
    assert not any(external_token in method_id.lower() for method_id in all_method_ids)
    for method in BONE_SEGMENTATION_METHODS.values():
        for site in ("radius", "tibia", "knee"):
            assert method_supports_site(method, site)


def test_expert_parameter_groups_are_driven_by_selected_algorithms() -> None:
    from SlicerBoneImagingToolboxLib.segmentation_methods import selected_parameter_groups

    groups = selected_parameter_groups(
        bone_method="seg_gauss",
        periosteal_method="geodesic_fracture",
        endosteal_method="none",
    )

    assert groups == {
        "Bone segmentation": ("gaussian_sigma", "trab_threshold", "cort_threshold"),
        "Periosteal contour": ("geodesic_bone_threshold", "geodesic_fill_holes"),
    }

    groups = selected_parameter_groups(
        bone_method="laplace_hamming",
        periosteal_method="standard",
        endosteal_method="standard",
    )

    assert "laplace_hamming_threshold" in groups["Bone segmentation"]
    assert "periosteal_threshold" in groups["Periosteal contour"]
    assert "endosteal_threshold" in groups["Endosteal contour"]


def test_segmentation_module_uses_method_descriptors_for_dynamic_expert_fields() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "from SlicerBoneImagingToolboxLib.segmentation_methods import" in source
    assert "def _refresh_method_dependent_ui(self):" in source
    assert "selected_parameter_groups(" in source


def test_xct2_gaussian_standard_contours_take_full_compartment_generation_path() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    standard_path_start = source.index(
        'if (\n            periosteal_contour_method == "standard"'
    )
    standard_path_end = source.index("        else:", standard_path_start)
    standard_path = source[standard_path_start:standard_path_end]

    assert 'endosteal_contour_method == "standard"' in standard_path
    assert 'segmentation_method != "none"' in standard_path
    assert "generate_masks_from_image(" in standard_path
    assert "generated.metadata[\"emitted_roles\"]" in source
    assert "emitted_roles = metadata.get(\"emitted_roles\", [])" in source


def test_xct2_gaussian_with_no_contour_outputs_uses_global_trab_threshold() -> None:
    source = (
        ROOT / "HRpQCTTools" / "SegmentationHRpQCT" / "SegmentationHRpQCT.py"
    ).read_text(encoding="utf-8")

    assert "internal_compartment_only" not in source
    assert "global_threshold_without_compartments = (" in source
    assert 'segmentation_method == "seg_gauss"' in source
    assert "not compartment_split_requested" in source
    assert "seg_xyz = (segmentation_image_xyz >= trab_threshold) & full_xyz" in source
    assert "No cortical mask was provided; Gaussian segmentation used the trabecular threshold" in source


def test_slicer_extension_has_no_external_segmentation_workflow_references() -> None:
    token = "or" + "mir"
    paths = [
        ROOT / "SlicerBoneImagingToolboxLib",
        ROOT / "HRpQCTTools",
        ROOT / "README.md",
        ROOT / "docs",
    ]
    scanned_suffixes = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".cmake"}
    matches = []
    for base in paths:
        candidates = [base] if base.is_file() else base.rglob("*")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in scanned_suffixes:
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            if token in text.lower():
                matches.append(str(candidate.relative_to(ROOT)))

    assert matches == []
