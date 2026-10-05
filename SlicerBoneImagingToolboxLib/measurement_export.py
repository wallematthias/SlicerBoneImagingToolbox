"""Lossless, workflow-aware reshaping of case summaries into cohort CSVs.

This module does not compute scientific measurements or discover arbitrary CSVs.
The Batch Processor supplies the selected profile's known case outputs.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile


MEASUREMENT_TOOLS = frozenset({
    "microarchitecture", "voidspace", "timelapse", "plate_rod", "fea", "mechanoregulation",
})
IDENTITY_COLUMNS = (
    "subject_id", "session_id", "baseline_session_id", "followup_session_id",
    "site", "stack_index", "tool", "profile", "source_files",
)
_MISSING = {"", "nan", "none", "null", "na", "n/a", "inf", "+inf", "-inf"}
_LOAD_COMPONENTS = {"Scale factors", "Input load amplitudes", "Estimated loads"}


@dataclass(frozen=True)
class MeasurementExportReport:
    path: Path
    row_count: int
    source_count: int
    issues: tuple[str, ...]


def measurement_csv_paths(tool, paths):
    """Select canonical summary tables, excluding curves and element tables."""
    patterns = {
        "microarchitecture": r".*_measurements\.csv",
        "voidspace": r"voidspace_(?:change_)?measurements\.csv",
        "timelapse": r".*_pairwise_remodelling\.csv",
        "plate_rod": r".*_desc-plate-rod-measurements\.csv",
        "fea": r".*_fea\.csv",
        "mechanoregulation": r".*_roi-.+_mechanoregulation_summary\.csv",
    }
    pattern = patterns.get(tool)
    if pattern is None:
        return []
    return sorted({Path(path).resolve() for path in paths
                   if re.fullmatch(pattern, Path(path).name, re.IGNORECASE)})


def _read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        headers = reader.fieldnames
        if not headers or any(not name or not name.strip() for name in headers):
            raise ValueError("missing CSV header")
        if len(headers) != len(set(headers)):
            raise ValueError("duplicate CSV columns")
        rows = list(reader)
        if not rows:
            raise ValueError("empty measurement table")
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("CSV row length does not match its header")
        return rows


def _value(value):
    text = str(value or "").strip()
    return "" if text.lower() in _MISSING else text


def _identity(case, tool, profile):
    result = {
        "subject_id": str(case.get("subject") or "").removeprefix("sub-"),
        "session_id": str(case.get("session_value", case.get("session")) or ""),
        "baseline_session_id": str(case.get("baseline_session_value") or ""),
        "followup_session_id": str(case.get("followup_session_value") or ""),
        "site": str(case.get("voi_value", case.get("voi")) or ""),
        "stack_index": str(case.get("stack_index") if case.get("stack_index") is not None else ""),
        "tool": tool,
        "profile": profile,
    }
    if result["baseline_session_id"] or result["followup_session_id"]:
        result["session_id"] = ""
    if not result["subject_id"] or not result["site"]:
        raise ValueError("case needs subject and site identifiers")
    return result


def _put(target, key, value):
    value = _value(value)
    if key in IDENTITY_COLUMNS:
        raise ValueError(f"measurement column collides with identifier: {key}")
    if key in target and target[key] != value:
        raise ValueError(f"conflicting measurement: {key}")
    target[key] = value


def _flatten(path, tool, identity):
    rows = _read_rows(path)
    if tool == "mechanoregulation":
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("profile") != identity["profile"]:
            raise ValueError("Mechanoregulation summary metadata profile differs from selected profile")
    groups = {}
    for row in rows:
        context = dict(identity)
        source_profile = row.get("Profile" if tool == "fea" else "profile")
        if tool in {"fea", "timelapse"} and source_profile and source_profile != identity["profile"]:
            raise ValueError(f"CSV profile {source_profile!r} differs from selected profile {identity['profile']!r}")
        prefix = ""
        ignored = set()
        if tool == "timelapse":
            if not all(row.get(key) for key in ("compartment", "t0", "t1", "threshold", "cluster_min_size")):
                raise ValueError("remodelling table lacks pair/compartment/threshold identifiers")
            context.update(session_id="", baseline_session_id=row["t0"], followup_session_id=row["t1"])
            prefix = f"{row['compartment']}.threshold-{row['threshold']}.cluster-{row['cluster_min_size']}."
            ignored = {"subject_id", "compartment", "t0", "t1", "threshold", "cluster_min_size"}
        elif tool == "mechanoregulation":
            roi = re.search(r"_roi-(.+)_mechanoregulation_summary\.csv$", path.name, re.IGNORECASE)
            prefix = f"{roi.group(1)}."
            classification = re.search(r"_desc-([^_]+)_", path.name)
            if classification:
                prefix += f"classification-{classification[1]}."
            variant = re.search(r"_thr-([^_]+)_(?:cluster|cl)-([^_]+)_", path.name)
            if variant:
                prefix += f"threshold-{variant[1].replace('p', '.')}.cluster-{variant[2]}."
        elif tool == "voidspace" and path.name.lower() == "voidspace_measurements.csv":
            # Current core summaries measure the large map, not the all-void map.
            prefix = "Large."
        if row.get("subject_id") and row["subject_id"].removeprefix("sub-") != context["subject_id"]:
            raise ValueError("CSV subject identifier differs from the discovered case")
        if tool == "plate_rod":
            if row.get("session_id") != context["session_id"] or row.get("site") != context["site"]:
                raise ValueError("CSV session/site identifiers differ from the discovered case")
            ignored = {"subject_id", "session_id", "site"}
        key = tuple(context.values())
        data = groups.setdefault(key, {"identity": context, "measurements": {}})["measurements"]
        if tool == "microarchitecture":
            parameter = row.get("Parameter", "").strip()
            if not parameter or "Mean" not in row:
                raise ValueError("microarchitecture table needs Parameter and Mean columns")
            for name, value in row.items():
                if name != "Parameter":
                    _put(data, f"{parameter}.{name}", value)
            continue
        for name, value in row.items():
            if name in ignored or (tool == "fea" and name in {"Sample", "Profile"}):
                continue
            if tool == "fea" and name in _LOAD_COMPONENTS:
                # The existing FEA summary encodes components as semicolon lists.
                if _value(value):
                    for index, component in enumerate(value.split(";"), 1):
                        _put(data, f"{name}.{index}", component)
                continue
            _put(data, prefix + name, value)
    return groups


def _source_name(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def export_measurements(destination, *, dataset_root, tool, profile, cases, force=False, protected_sources=()):
    """Export wide summaries and report omitted missing, malformed or conflicting cases.

    Existing destination files require explicit overwrite permission. Source CSVs
    are never overwritten. No averaging, unit conversion or new measurements are
    performed. Missing/non-finite values are emitted as empty cells.
    """
    if tool not in MEASUREMENT_TOOLS:
        raise ValueError("This workflow does not produce measurement summaries.")
    destination = Path(destination).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    cases = list(cases)
    all_sources = {path for case in cases for path in measurement_csv_paths(tool, case.get("paths", []))}
    all_sources.update(Path(path).expanduser().resolve() for path in protected_sources)
    if destination in all_sources:
        raise ValueError("The export destination must not be a source measurement CSV.")
    if destination.exists() and not force:
        raise FileExistsError(f"Export already exists: {destination}")
    merged, invalid, seen, issues = {}, set(), set(), []
    for case in cases:
        paths = measurement_csv_paths(tool, case.get("paths", []))
        label = f"sub-{case.get('subject', '')} ses-{case.get('session_value', case.get('session', ''))} voi-{case.get('voi_value', case.get('voi', ''))}"
        case_keys = set()
        identity = None
        try:
            identity = _identity(case, tool, profile)
            if not paths:
                raise ValueError("no measurement CSV available")
            for path in paths:
                # Series tables are rediscovered by every session row. Process once.
                token = (path, identity["subject_id"], identity["site"], identity["stack_index"])
                if token in seen:
                    continue
                seen.add(token)
                groups = _flatten(path, tool, identity)
                for key, group in groups.items():
                    case_keys.add(key)
                    target = merged.setdefault(key, {**group["identity"], "measurements": {}, "sources": set()})
                    for name, value in group["measurements"].items():
                        _put(target["measurements"], name, value)
                    target["sources"].add(path)
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            invalid.update(case_keys)
            # Also omit an earlier contribution for this case if a later CSV failed.
            for key, row in merged.items():
                if identity is not None and all(row[name] == identity[name] for name in identity):
                    invalid.add(key)
            issues.append(f"{label}: {exc}")
    rows = [row for key, row in sorted(merged.items()) if key not in invalid]
    columns = sorted({name for row in rows for name in row["measurements"]})
    used_sources = {path for row in rows for path in row["sources"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8-sig",
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=[*IDENTITY_COLUMNS, *columns])
            writer.writeheader()
            for row in rows:
                output = {name: row[name] for name in IDENTITY_COLUMNS if name != "source_files"}
                output["source_files"] = json.dumps([_source_name(path, root) for path in sorted(row["sources"])])
                writer.writerow({**output, **row["measurements"]})
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return MeasurementExportReport(destination, len(rows), len(used_sources), tuple(issues))
