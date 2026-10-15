"""Compact, workflow-aware reshaping of case summaries into cohort CSVs.

The Batch Processor supplies the selected profile's known case outputs. Detailed
source results remain unchanged; net/active remodelling fractions are derived
from their existing formation/resorption fractions without rerunning analysis.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


MEASUREMENT_TOOLS = frozenset({
    "microarchitecture", "voidspace", "timelapse", "plate_rod", "fea", "mechanoregulation",
})
IDENTITY_COLUMNS = (
    "subject_id", "session_id", "baseline_session_id", "followup_session_id",
    "site", "stack_index", "original_filename", "baseline_original_filename",
    "followup_original_filename", "tool", "profile", "source_files",
)
_MISSING = {"", "nan", "none", "null", "na", "n/a", "inf", "+inf", "-inf"}
_LOAD_COMPONENTS = {"Scale factors", "Input load amplitudes", "Estimated loads"}
_MICROARCH_PARAMETERS = frozenset({
    "Tt.BMD", "Tb.BMD", "Tb.BV/TV", "Tb.Th", "Tb.Sp", "Tb.N", "Tb.1/N.SD",
    "Tb.BV", "Tb.TV", "Ct.BMD", "Ct.Th", "Ct.Po", "Ct.Po.V", "Ct.Po.Dm", "Ct.BV", "Ct.TV",
})
_ITS_MEASUREMENTS = frozenset({
    "BV/TV", "pBV/TV", "rBV/TV", "pBV/BV", "rBV/BV", "pBV/rBV",
    "TV_mm3", "BV_mm3", "pBV_mm3", "rBV_mm3",
    "plate_count", "rod_count", "junction_count", "pTb.N", "rTb.N", "PR.N",
    "pTb.Th_mm", "rTb.Th_mm", "pTb.S_mm2", "rTb.l_mm",
    "P-P Junc.D", "P-R Junc.D", "R-R Junc.D", "aBV/TV",
    "mean_plate_axial_alignment", "mean_rod_axial_alignment",
    # Element-derived volumes differ from the full propagated-label volumes.
    "measured_BV_mm3", "measured_pBV_mm3", "measured_rBV_mm3",
})
_MECHANOREG_MEASUREMENTS = frozenset({
    "CCR", "binned_lazy_zone_low", "binned_lazy_zone_high",
    "logistic_lazy_zone_low", "logistic_lazy_zone_high", "OR_F", "OR_R",
    "OR_F_CI_low", "OR_F_CI_high", "OR_R_CI_low", "OR_R_CI_high",
})
_TIMELAPSE_CONTEXT = ("fraction_denominator", "followup_days", "scan_date_t0", "scan_date_t1")


def _remodelling_fractions(data, prefix, row):
    formation = _value(row.get("formation_frac_bv0"))
    resorption = _value(row.get("resorption_frac_bv0"))
    net = active = ""
    if formation and resorption:
        try:
            fv, rv = Decimal(formation), Decimal(resorption)
        except InvalidOperation as exc:
            raise ValueError("invalid formation/resorption fraction") from exc
        if not fv.is_finite() or not rv.is_finite():
            raise ValueError("non-finite formation/resorption fraction")
        net, active = format(fv - rv, "f"), format(fv + rv, "f")
    for name, value in (("formation_volume_fraction", formation),
                        ("resorption_volume_fraction", resorption),
                        ("net_change_volume_fraction", net), ("active_volume_fraction", active)):
        _put(data, prefix + name, value)
    for name in _TIMELAPSE_CONTEXT:
        if name in row:
            _put(data, name, row[name])


@dataclass(frozen=True)
class MeasurementExportReport:
    path: Path
    row_count: int
    source_count: int
    issues: tuple[str, ...]
    warnings: tuple[str, ...] = ()


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
            if parameter in _MICROARCH_PARAMETERS:
                _put(data, f"{parameter}.Mean", row["Mean"])
            continue
        if tool == "timelapse":
            _remodelling_fractions(data, prefix, row)
            continue
        for name, value in row.items():
            if name in ignored or (tool == "fea" and name in {"Sample", "Profile"}):
                continue
            if tool == "plate_rod" and name not in _ITS_MEASUREMENTS:
                continue
            if tool == "mechanoregulation" and name not in _MECHANOREG_MEASUREMENTS:
                continue
            if tool == "voidspace" and name == "stable.VS.V":
                # Old summaries called the same quiescent volume "stable".
                # _put also detects disagreement when both aliases are present.
                name = "quiescent.VS.V"
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


def _portable_path(value):
    """Read manifest paths written on either Windows or POSIX systems."""
    return PurePosixPath(value.replace("\\", "/"))


def _rename_target_relative(value, recorded_root, root):
    target = _portable_path(value)
    for base in (recorded_root, str(root)):
        if base:
            try:
                relative = target.relative_to(_portable_path(base))
                if ".." not in relative.parts:
                    return relative
            except ValueError:
                pass
    if not target.is_absolute() and not re.match(r"^[A-Za-z]:", str(target)) and ".." not in target.parts:
        return target
    return None


def _original_filename_index(root, warnings):
    """Join naming provenance to shared raw-scan discovery, never masks or sidecars."""
    manifest = root / "dataset_rename_manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or not isinstance(payload.get("renames"), list):
            raise ValueError("expected a renames list")
        recorded_root = payload.get("dataset_root") or ""
        if not isinstance(recorded_root, str):
            raise ValueError("invalid dataset_root")
    except (OSError, UnicodeError, ValueError) as exc:
        warnings.append(f"Original filenames unavailable: cannot read {manifest.name}: {exc}")
        return None
    targets = {}
    for item in payload["renames"]:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(name), str) or not item[name].strip()
            for name in ("original_path", "renamed_path")
        ):
            warnings.append(f"Original filenames: ignoring invalid entry in {manifest.name}")
            continue
        relative = _rename_target_relative(item["renamed_path"], recorded_root, root)
        if relative is not None:
            targets.setdefault(relative, set()).add(_portable_path(item["original_path"]))
    try:
        from bone_imaging_derivatives import discover_raw_xct_images

        index = {}
        for scan in discover_raw_xct_images(root):
            key = scan.key
            identity = (key.subject_id, key.session_id, key.voi.lower(),
                        str(key.stack_index) if key.stack_index is not None else "")
            relative = PurePosixPath(scan.path.relative_to(root).as_posix())
            index.setdefault(identity, {})[relative] = targets.get(relative, set())
        return index
    except (ImportError, OSError, ValueError) as exc:
        warnings.append(f"Original filenames unavailable: raw-scan discovery failed: {exc}")
        return None


def _original_filenames(row, index, warnings):
    filenames = dict(original_filename="", baseline_original_filename="", followup_original_filename="")
    if index is None:
        return filenames
    sessions = (("baseline_original_filename", row["baseline_session_id"]),
                ("followup_original_filename", row["followup_session_id"])) if (
        row["baseline_session_id"] or row["followup_session_id"]
    ) else (("original_filename", row["session_id"]),)
    for column, session in sessions:
        if not session:
            continue
        key = (row["subject_id"], session, row["site"].lower(), row["stack_index"])
        scans = index.get(key, {})
        originals = set().union(*scans.values())
        if len(scans) == 1 and len(originals) == 1:
            filenames[column] = next(iter(originals)).name
        else:
            reason = "ambiguous mapping" if len(scans) > 1 or len(originals) > 1 else "no matching rename entry"
            warnings.append(f"Original filename {reason}: sub-{key[0]} ses-{session} voi-{key[2]} stack-{key[3] or 'none'}")
    return filenames


def export_measurements(destination, *, dataset_root, tool, profile, cases, force=False, protected_sources=(),
                        provenance_root=None):
    """Export wide summaries and report omitted missing, malformed or conflicting cases.

    Existing destination files require explicit overwrite permission. Source CSVs
    are never overwritten. No averaging or unit conversion is performed; net and
    active remodelling fractions are formation minus/plus resorption. Other
    values are copied. Missing/non-finite values are emitted as empty cells. Original scan
    names come from the dataset-root rename manifest; unresolved provenance warns
    without omitting measurements. ``provenance_root`` retains local naming
    provenance when summaries are exported from a temporary download snapshot.
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
    warnings = []
    if rows:
        provenance = Path(provenance_root).expanduser().resolve() if provenance_root is not None else root
        originals = _original_filename_index(provenance, warnings)
        for row in rows:
            row.update(_original_filenames(row, originals, warnings))
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
    return MeasurementExportReport(destination, len(rows), len(used_sources), tuple(issues),
                                   tuple(dict.fromkeys(warnings)))
