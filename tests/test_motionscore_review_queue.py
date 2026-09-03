from SlicerBoneImagingToolboxLib.motionscore_review import next_review_scan_id


def test_next_review_scan_advances_when_scope_keeps_previous_scan_visible() -> None:
    assert next_review_scan_id("scan-1", ["scan-1", "scan-2", "scan-3"]) == "scan-2"


def test_next_review_scan_uses_first_remaining_pending_scan() -> None:
    assert next_review_scan_id("scan-1", ["scan-2", "scan-3"]) == "scan-2"


def test_next_review_scan_wraps_in_all_scans_scope_instead_of_reloading_last() -> None:
    assert next_review_scan_id("scan-3", ["scan-1", "scan-2", "scan-3"]) == "scan-1"


def test_next_review_scan_handles_empty_scope() -> None:
    assert next_review_scan_id("scan-1", []) == ""
