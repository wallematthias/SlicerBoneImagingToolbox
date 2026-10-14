"""Lightweight completion checks shared by discovery and batch subprocesses.

Keep this module standard-library-only: its source is embedded in remote jobs,
where the Slicer toolbox itself is not installed. Legacy completed cases need
no new marker file. Masks must be nonempty files and the last-written CSV must
contain a complete numeric measurement row; mask contents are not decoded here.
"""

import csv
import math
from pathlib import Path


MASK_SUFFIXES = (".AIM", ".aim", ".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")
INCOMPLETE_MARKER = ".voidspace_incomplete"


def begin_voidspace_outputs(directory: Path) -> Path:
    """Invalidate old results before writing; leave the marker on interruption."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / INCOMPLETE_MARKER
    marker.touch()
    return marker


def existing_voidspace_map(directory: Path, stem: str) -> Path | None:
    """Find a nonempty output file, including masks with no foreground voxels."""
    for suffix in MASK_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def voidspace_outputs_complete(
    directory: Path, *, dynamic: bool = False, require_analysis_mask: bool = False,
) -> bool:
    """Reject interrupted output sets, including header-only/truncated CSVs."""
    if (directory / INCOMPLETE_MARKER).exists():
        return False
    if dynamic:
        stems = ("voidspace_expanded_mask", "voidspace_contracted_mask")
        if not (existing_voidspace_map(directory, "voidspace_quiescent_mask")
                or existing_voidspace_map(directory, "voidspace_stable_mask")):
            return False
        csv_name = "voidspace_change_measurements.csv"
        columns = ("expanded.VS.V", "contracted.VS.V", "net.VS.V")
    else:
        stems = ("voidspace_all_mask", "voidspace_large_mask")
        csv_name = "voidspace_measurements.csv"
        columns = ("VS.TV", "VS.V", "Tt.V", "VS.N", "VS.Ar")
    if require_analysis_mask:
        stems += ("voidspace_analysis_mask",)
    if any(existing_voidspace_map(directory, stem) is None for stem in stems):
        return False
    try:
        with (directory / csv_name).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if dynamic:
                stable_column = next((name for name in ("quiescent.VS.V", "stable.VS.V")
                                      if name in (reader.fieldnames or ())), None)
                if stable_column is None:
                    return False
                columns += (stable_column,)
            rows = list(reader)
            return bool(rows) and all(
                math.isfinite(float(row[column])) for row in rows for column in columns
            )
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError):
        return False
