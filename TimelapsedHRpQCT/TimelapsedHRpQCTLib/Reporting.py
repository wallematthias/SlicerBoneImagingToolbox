from __future__ import annotations

from datetime import date, datetime
from math import isfinite


PROFILE_DISPLAY_ORDER = [
    "eth-uofc",
    "multistack",
    "ped-fx",
    "standard",
    "single-stack",
    "low-memory",
    "xct1-standard",
]

COHORT_DEFAULT_EXPORT_FIELDS = (
    "subject_id",
    "site",
    "compartment",
    "profile",
    "t0",
    "t1",
    "threshold",
    "cluster_min_size",
    "formation_volume_fraction",
    "resorption_volume_fraction",
    "net_change_volume_fraction",
    "active_volume_fraction",
    "BVTV_t0",
    "BVTV_t1",
    "followup_days",
)

COHORT_EXTRA_EXPORT_FIELD_SPECS = (
    ("scan_period_years", "Scan interval in years, reported for context and not used for normalization."),
    ("scan_date_t0", "Baseline scan date, when available in the saved pairwise output."),
    ("scan_date_t1", "Follow-up scan date, when available in the saved pairwise output."),
    ("pair_key", "Session comparison key used to identify the baseline-to-follow-up pair."),
    ("fraction_denominator_vox", "Bone-volume denominator voxel count used for remodelling fractions."),
    ("TV_valid_vox", "Valid total-volume voxel count in the common analysis region."),
    ("BV0_vox", "Baseline bone voxel count."),
    ("BV1_vox", "Follow-up bone voxel count."),
    ("real_overlap_vox", "Voxel count in the real overlapping scan region."),
    ("real_overlap_frac_of_union", "Real overlap as a fraction of the scan-region union."),
    ("formation_vox", "Formation event voxel count after thresholding and cluster filtering."),
    ("resorption_vox", "Resorption event voxel count after thresholding and cluster filtering."),
    ("formation_n_clusters", "Number of retained formation clusters."),
    ("resorption_n_clusters", "Number of retained resorption clusters."),
    ("formation_largest_cluster_vox", "Largest retained formation cluster size in voxels."),
    ("resorption_largest_cluster_vox", "Largest retained resorption cluster size in voxels."),
    ("mean_inside_valid_t0", "Mean baseline density inside the valid common region."),
    ("mean_inside_valid_t1", "Mean follow-up density inside the valid common region."),
    ("delta_mean_valid", "Follow-up minus baseline mean density inside the valid common region."),
    ("corr_valid", "Baseline-follow-up density correlation inside the valid common region."),
    ("rmse_valid", "Baseline-follow-up density RMSE inside the valid common region."),
)


def _as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if isfinite(out) else float("nan")


def _first_numeric(row, names):
    for name in names:
        if name in row and str(row.get(name, "")).strip():
            value = _as_float(row.get(name))
            if isfinite(value):
                return value
    return float("nan")


def _parse_date(value):
    text = str(value or "").strip()
    if not text:
        return None
    for parser in (date.fromisoformat, datetime.fromisoformat):
        try:
            parsed = parser(text)
            return parsed.date() if isinstance(parsed, datetime) else parsed
        except ValueError:
            pass
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def scan_period_days_from_row(row):
    days = _first_numeric(
        row,
        (
            "scan_period_days",
            "scan_interval_days",
            "interval_days",
            "days_between_scans",
            "time_between_scans_days",
            "followup_days",
        ),
    )
    if isfinite(days):
        return days

    years = _first_numeric(
        row,
        (
            "scan_period_years",
            "scan_interval_years",
            "interval_years",
            "years_between_scans",
            "time_between_scans_years",
            "followup_years",
        ),
    )
    if isfinite(years):
        return years * 365.25

    t0_date = None
    t1_date = None
    for t0_name, t1_name in (
        ("t0_date", "t1_date"),
        ("scan_date_t0", "scan_date_t1"),
        ("scan0_date", "scan1_date"),
        ("baseline_scan_date", "followup_scan_date"),
    ):
        t0_date = _parse_date(row.get(t0_name))
        t1_date = _parse_date(row.get(t1_name))
        if t0_date and t1_date:
            break
    if t0_date and t1_date:
        return float((t1_date - t0_date).days)
    return float("nan")


def enrich_cohort_export_row(row):
    enriched = dict(row)
    fv_bv = _as_float(row.get("formation_frac_bv0", row.get("FV_BV")))
    rv_bv = _as_float(row.get("resorption_frac_bv0", row.get("RV_BV")))
    nv_bv = fv_bv - rv_bv if isfinite(fv_bv) and isfinite(rv_bv) else float("nan")
    av_bv = fv_bv + rv_bv if isfinite(fv_bv) and isfinite(rv_bv) else float("nan")

    enriched["FV_BV"] = fv_bv
    enriched["RV_BV"] = rv_bv
    enriched["NV_BV"] = nv_bv
    enriched["AV_BV"] = av_bv
    enriched["formation_volume_fraction"] = fv_bv
    enriched["resorption_volume_fraction"] = rv_bv
    enriched["net_change_volume_fraction"] = nv_bv
    enriched["active_volume_fraction"] = av_bv

    days = scan_period_days_from_row(row)
    enriched["scan_period_days"] = days if isfinite(days) else ""
    enriched["scan_period_years"] = days / 365.25 if isfinite(days) else ""
    if not str(enriched.get("followup_days", "")).strip():
        enriched["followup_days"] = enriched["scan_period_days"]
    return enriched


def project_rows_to_fields(rows, fields):
    selected = list(fields)
    return [{field: row.get(field, "") for field in selected} for row in rows]


def default_export_filename(prefix, *, now=None, suffix=".csv"):
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    clean_prefix = str(prefix or "exported_results").strip().replace(" ", "_")
    clean_suffix = str(suffix or ".csv")
    if not clean_suffix.startswith("."):
        clean_suffix = f".{clean_suffix}"
    return f"{clean_prefix}_{timestamp}{clean_suffix}"
