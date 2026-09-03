"""Review-queue helpers for the MotionScore Slicer module."""

from __future__ import annotations

from collections.abc import Iterable


def next_review_scan_id(previous_scan_id: str | None, scan_ids: Iterable[str]) -> str:
    """Return the scan that should be selected after saving a manual grade.

    The review scope may keep the graded scan visible, for example in the
    ``All scans`` scope. Advancing from the explicit previous scan avoids
    rebuilding the combo box onto the same scan and triggering an automatic
    reload of the just-reviewed image.
    """

    ordered_scan_ids = [str(scan_id).strip() for scan_id in scan_ids if str(scan_id).strip()]
    if not ordered_scan_ids:
        return ""

    previous = str(previous_scan_id or "").strip()
    if previous in ordered_scan_ids:
        index = ordered_scan_ids.index(previous)
        if index + 1 < len(ordered_scan_ids):
            return ordered_scan_ids[index + 1]
        if len(ordered_scan_ids) > 1:
            return ordered_scan_ids[0]
        return previous

    return ordered_scan_ids[0]
