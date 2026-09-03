from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1] / "HRpQCTTools" / "DeriveLabelsHRpQCT"
sys.path.insert(0, str(MODULE_DIR))

import DeriveLabelsHRpQCT  # noqa: E402


def test_material_label_arrays_combine_segmentation_and_compartments():
    seg = np.zeros((4, 4, 4), dtype=np.uint8)
    seg[1:3, 1:3, 1:3] = 1
    trab = np.zeros_like(seg)
    trab[1, 1:3, 1:3] = 1
    cort = np.zeros_like(seg)
    cort[2, 1:3, 1:3] = 1

    material, counts = DeriveLabelsHRpQCT.material_labels_from_arrays(
        seg,
        trab,
        cort,
        trab_label=126,
        cort_label=127,
        cort_source="cort_mask",
    )

    assert counts == {"trab": 4, "cort": 4, "cort_source": "cort_mask"}
    assert set(np.unique(material)) == {0, 126, 127}
    assert np.count_nonzero(material == 126) == 4
    assert np.count_nonzero(material == 127) == 4


def test_material_label_arrays_support_derived_cortical_region():
    seg = np.zeros((4, 4, 4), dtype=np.uint8)
    seg[1:3, 1:3, 1:3] = 1
    trab = np.zeros_like(seg)
    trab[1, 1:3, 1:3] = 1
    full = np.zeros_like(seg)
    full[1:3, 1:3, 1:3] = 1
    cort = full.astype(bool) & ~trab.astype(bool)

    material, counts = DeriveLabelsHRpQCT.material_labels_from_arrays(
        seg,
        trab,
        cort,
        cort_source="derived_from_full_minus_trab",
    )

    assert counts == {"trab": 4, "cort": 4, "cort_source": "derived_from_full_minus_trab"}
    assert np.count_nonzero(material == 100) == 4
    assert np.count_nonzero(material == 127) == 4


def test_derive_compartment_arrays_support_any_two_inputs():
    trab = np.zeros((3, 3, 3), dtype=bool)
    trab[0] = True
    cort = np.zeros_like(trab)
    cort[2] = True
    full = trab | cort

    derived_full = DeriveLabelsHRpQCT.derive_compartment_mask_arrays(trab=trab, cort=cort)
    derived_trab = DeriveLabelsHRpQCT.derive_compartment_mask_arrays(full=full, cort=cort)
    derived_cort = DeriveLabelsHRpQCT.derive_compartment_mask_arrays(full=full, trab=trab)

    assert derived_full["derived_role"] == "full"
    assert np.array_equal(derived_full["full"], full)
    assert derived_trab["derived_role"] == "trab"
    assert np.array_equal(derived_trab["trab"], trab)
    assert derived_cort["derived_role"] == "cort"
    assert np.array_equal(derived_cort["cort"], cort)


def test_material_labels_can_use_derived_trabecular_region():
    seg = np.ones((3, 3, 3), dtype=bool)
    full = np.zeros_like(seg)
    full[0:2] = True
    cort = np.zeros_like(seg)
    cort[0] = True
    masks = DeriveLabelsHRpQCT.derive_compartment_mask_arrays(full=full, cort=cort)

    material, counts = DeriveLabelsHRpQCT.material_labels_from_arrays(
        seg,
        masks["trab"],
        masks["cort"],
        trab_label=126,
        cort_label=127,
        cort_source="cort_mask",
    )

    assert counts["trab"] == 9
    assert counts["cort"] == 9
    assert set(np.unique(material)) == {0, 126, 127}


def test_compartment_validation_detects_overlap_and_missing_voxels():
    full = np.ones((2, 2, 2), dtype=bool)
    trab = np.zeros_like(full)
    cort = np.zeros_like(full)
    trab[0, 0, 0] = True
    cort[0, 0, 0] = True

    counts = DeriveLabelsHRpQCT.validate_compartment_mask_arrays(full=full, trab=trab, cort=cort)

    assert counts["valid"] is False
    assert counts["overlap"] == 1
    assert counts["full_not_compartment"] == 7


def test_boolean_mask_operations_and_relabel_nonzero():
    mask_a = np.array([0, 1, 1, 0], dtype=np.uint8)
    mask_b = np.array([1, 1, 0, 0], dtype=np.uint8)

    assert DeriveLabelsHRpQCT.binary_mask_operation_arrays(mask_a, mask_b, "union").tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert DeriveLabelsHRpQCT.binary_mask_operation_arrays(mask_a, mask_b, "intersection").tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert DeriveLabelsHRpQCT.binary_mask_operation_arrays(mask_a, mask_b, "difference").tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert DeriveLabelsHRpQCT.relabel_nonzero_array(mask_a, 127).tolist() == [0, 127, 127, 0]
